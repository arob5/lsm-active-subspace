r"""Running SIPNET at a site, and the forward map from calibration parameters to observed features.

:class:`SIPNETSimulator` runs SIPNET with a fixed driver and fixed parameter values, returning NEE
as a rate. :func:`feature_map` wraps a simulator, a map from calibration parameters to SIPNET
parameters, and an observation operator into the forward map $G(\theta) = A\, x(\theta)$, a ProbPipe
``Function`` that runs SIPNET in parallel threads when called on a batch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pydantic
import scipy.sparse as sp
import xarray as xr
from probpipe import Function
from pysipnet import (
    ClimateDrivers,
    ModelFlags,
    SIPNETModel,
    SIPNETRunError,
    SIPNETRunner,
    niwot_reference_parameters,
)

__all__ = ["SIMULATION_FAILURES", "SIPNETSimulator", "feature_map"]

logger = logging.getLogger(__name__)

#: Errors that mean SIPNET could not produce output for a parameter setting: pySIPNET refused the
#: values (``pydantic.ValidationError``), or SIPNET itself stopped (``SIPNETRunError``).
SIMULATION_FAILURES = (pydantic.ValidationError, SIPNETRunError)

#: Micromoles of CO₂ per gram of carbon.
_MICROMOLES_CO2_PER_GRAM_CARBON = 1e6 / 12.011


class SIPNETSimulator:
    """SIPNET at one site: a driver, fixed parameter values, and model options.

    A parameter that is neither passed to :meth:`nee_rate` nor among the fixed values takes
    pySIPNET's reference value for Niwot Ridge.

    Parameters
    ----------
    driver_path : Path
        The meteorological driver, a ``.clim`` file with time labels in UTC.
    fixed_parameters : Mapping of str to float
        Values of SIPNET parameters, keyed by pySIPNET's names.
    flags : ModelFlags
        SIPNET's model options, ``ModelFlags.standard()`` by default.
    """

    def __init__(
        self,
        driver_path: Path,
        fixed_parameters: Mapping[str, float],
        flags: ModelFlags | None = None,
    ) -> None:
        self.driver_path = Path(driver_path)
        self.fixed_parameters = dict(fixed_parameters)
        self._model = SIPNETModel(
            SIPNETRunner(flags=ModelFlags.standard() if flags is None else flags),
            base_params=niwot_reference_parameters(),
            base_climate=ClimateDrivers.from_path(self.driver_path, time_zone="UTC"),
        )

    def run_nee(self, parameters: Mapping[str, float]) -> xr.DataArray:
        """SIPNET's NEE per output step, as pySIPNET returns it (g C m⁻² per step).

        Parameters
        ----------
        parameters : Mapping of str to float
            SIPNET parameters, keyed by pySIPNET's names, overriding the fixed values.
        """
        return self._model(**(self.fixed_parameters | dict(parameters))).outputs["nee"]

    def nee_rate(self, parameters: Mapping[str, float]) -> np.ndarray:
        r"""SIPNET's NEE as a rate in µmol CO₂ m⁻² s⁻¹, one value per output step.

        A step's total $N_s$ (g C m⁻²) over its length $\Delta_s$ (s) becomes the rate
        $10^6 N_s / (12.011\, \Delta_s)$.

        Parameters
        ----------
        parameters : Mapping of str to float
            SIPNET parameters, keyed by pySIPNET's names, overriding the fixed values.
        """
        totals = self.run_nee(parameters)
        seconds = totals.coords["timestep_length"].values / np.timedelta64(1, "s")
        return totals.values / seconds * _MICROMOLES_CO2_PER_GRAM_CARBON


def feature_map(
    simulator: SIPNETSimulator,
    operator: sp.csr_matrix,
    to_sipnet: Callable[[Mapping[str, Any]], Mapping[str, float]],
    *,
    max_workers: int = 8,
    label: str = "nee_features",
) -> Function:
    r"""The forward map $G(\theta) = A\, x(\theta)$ as a ProbPipe ``Function``.

    $x(\theta)$ is SIPNET's NEE rate at the SIPNET parameters ``to_sipnet(theta)``, one value per
    output step, and $A$ is the observation operator. Calling the function on a batch of parameter
    records runs SIPNET once per record, on up to *max_workers* threads (each run is a separate
    SIPNET process, so threads run in parallel).

    If SIPNET cannot run at a parameter setting (see :data:`SIMULATION_FAILURES`), the result for
    that setting is a row of NaN and a warning is logged, so one failure does not stop a batch.

    Parameters
    ----------
    simulator : SIPNETSimulator
        SIPNET at the site.
    operator : scipy.sparse.csr_matrix
        $A$, shape $(n, S)$ for $S$ output steps.
    to_sipnet : callable
        Maps a parameter record ``theta`` to SIPNET parameters, keyed by pySIPNET's names.
    max_workers : int
        The number of SIPNET runs at a time.
    label : str
        The function's label.

    Returns
    -------
    Function
        ``theta -> array of shape (n,)``.
    """
    evaluator = _FeatureEvaluator(simulator, operator, to_sipnet)
    return Function(evaluator.features, label=label, dispatch="thread", max_workers=max_workers)


# STOPGAP(PG-14): a class with a method, not a closure, because ProbPipe's provenance recurses
# without end on a closure that captures a sparse matrix.
class _FeatureEvaluator:
    r"""Evaluates $A\, x(\theta)$ at one parameter record; ``feature_map`` wraps its method."""

    def __init__(
        self,
        simulator: SIPNETSimulator,
        operator: sp.csr_matrix,
        to_sipnet: Callable[[Mapping[str, Any]], Mapping[str, float]],
    ) -> None:
        self.simulator = simulator
        self.operator = operator
        self.to_sipnet = to_sipnet

    def features(self, theta: Mapping[str, Any]) -> Any:
        """The predicted features at *theta*, or a row of NaN if SIPNET produced no output."""
        sipnet_parameters = self.to_sipnet(theta)
        try:
            rate = self.simulator.nee_rate(sipnet_parameters)
        except SIMULATION_FAILURES as failure:
            _log_failure(sipnet_parameters, failure)
            return jnp.full(self.operator.shape[0], jnp.nan)
        return jnp.asarray(self.operator @ rate)


def _log_failure(sipnet_parameters: Mapping[str, float], failure: Exception) -> None:
    """Log a SIPNET run that produced no output, with its parameters and the reason."""
    reason = " ".join(str(failure).split())
    logger.warning(
        "SIPNET produced no output at %s (%s: %s)",
        dict(sipnet_parameters),
        type(failure).__name__,
        reason,
    )
