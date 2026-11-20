r"""The record an ensemble method passes to its callbacks at each step, and the run history.

An EKI result keeps its run's history; :func:`eki_history` reads it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from probpipe import Distribution

from lsmas._stopgaps.likelihood import noise_scale_variable

__all__ = ["StepCallback", "StepRecord", "eki_history", "history_dataset"]

#: Where an EKI result keeps its run's history, among its annotations.
_HISTORY_ANNOTATION = "arviz/eki_history"


@dataclass(frozen=True)
class StepRecord:
    r"""One step of a tempered ensemble method: the evaluated particles, their predictions and the
    step's increment.

    The method moves an ensemble of $J$ particles from the prior ($\beta = 0$) to the posterior
    ($\beta = 1$) in increments $\Delta\beta$. Each step evaluates the forward map $G$ at every
    particle and then conditions on the data with noise covariance $\Gamma / \Delta\beta$.
    When noise scales are inferred, each particle carries its own scales and $\Gamma$ is built from
    the ensemble's common scales, so it changes from step to step.

    Attributes
    ----------
    step : int
        The step's index, from 0.
    beta : float
        The tempering level $\beta$ when the particles were evaluated.
    increment : float
        The increment $\Delta\beta$ the step then took. The last step's increment reaches
        $\beta = 1$, so the final particles, which make up the result, appear in no record.
    parameter_names : tuple of str
        The names of the parameters, in the order of the last axis of :attr:`theta`.
    theta : numpy.ndarray
        The evaluated particles in constrained coordinates, shape $(J, d)$.
    predictions : numpy.ndarray
        $G$ at each particle, shape $(J, n)$. A particle at which the forward map failed has been
        moved, with its predictions, to the centre of the valid particles.
    noise_variance : numpy.ndarray
        The diagonal of $\Gamma$, the noise covariance at $\beta = 1$ that this step's update
        used, shape $(n,)$.
    noise_scales : dict of str to numpy.ndarray
        Each inferred scale's value at each particle, shape $(J,)$, by class, such as ``"day"``;
        empty when every scale is fixed.
    n_failed : int
        How many particles the forward map failed at.
    """

    step: int
    beta: float
    increment: float
    parameter_names: tuple[str, ...]
    theta: np.ndarray
    predictions: np.ndarray
    noise_variance: np.ndarray
    noise_scales: dict[str, np.ndarray]
    n_failed: int


#: A callable an ensemble method calls with every :class:`StepRecord`, in order.
StepCallback = Callable[[StepRecord], None]


def history_dataset(records: Sequence[StepRecord]) -> xr.Dataset:
    """A run's step records as one dataset, with dimensions ``step``, ``member``, ``parameter``
    and ``observation``, and one variable per inferred noise scale, such as ``s2_day``.

    Parameters
    ----------
    records : sequence of StepRecord
        The records of one run, in order.
    """
    names = list(records[0].parameter_names)
    return xr.Dataset(
        {
            "beta": ("step", [record.beta for record in records]),
            "increment": ("step", [record.increment for record in records]),
            "theta": (
                ("step", "member", "parameter"),
                np.stack([record.theta for record in records]),
            ),
            "predictions": (
                ("step", "member", "observation"),
                np.stack([record.predictions for record in records]),
            ),
            "noise_variance": (
                ("step", "observation"),
                np.stack([record.noise_variance for record in records]),
            ),
            "n_failed": ("step", [record.n_failed for record in records]),
            **{
                noise_scale_variable(noise_class): (
                    ("step", "member"),
                    np.stack([record.noise_scales[noise_class] for record in records]),
                )
                for noise_class in records[0].noise_scales
            },
        },
        coords={"step": [record.step for record in records], "parameter": names},
    )


def eki_history(posterior: Distribution) -> xr.DataTree:
    r"""The history of the EKI run that produced *posterior*, one entry per step.

    Its variables are those of :func:`history_dataset`: each step's level ``beta`` and
    ``increment``, its particles ``theta`` (shape ``(step, member, parameter)``), the forward
    map's ``predictions`` at them (``(step, member, observation)``), the ``noise_variance`` the
    step used (``(step, observation)``), ``n_failed``, and each inferred noise scale. The final
    particles, at $\beta = 1$, are the result itself and appear in no step.

    Parameters
    ----------
    posterior : Distribution
        The result of ``condition(method="enskit_eki")``.

    Raises
    ------
    ValueError
        If *posterior* is not the result of an EKI run.
    """
    try:
        return posterior.annotations[_HISTORY_ANNOTATION]
    except (AttributeError, KeyError, TypeError):
        raise ValueError(f"{posterior.label!r} is not the result of an EKI run") from None
