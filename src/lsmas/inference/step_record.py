r"""What an ensemble method reports after each evaluation of the forward map."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = ["StepCallback", "StepRecord", "history_dataset"]


@dataclass(frozen=True)
class StepRecord:
    r"""One step of a tempered ensemble method: the ensemble it evaluated and what it then did.

    The method moves an ensemble of $J$ particles from the prior ($\beta = 0$) to the posterior
    ($\beta = 1$) in increments $\Delta\beta$. Each step evaluates the forward map $G$ at every
    particle and then conditions on the data with noise covariance $\Gamma / \Delta\beta$.

    Attributes
    ----------
    step : int
        The step's index, from 0.
    beta : float
        The tempering level $\beta$ when the particles were evaluated.
    increment : float
        The increment $\Delta\beta$ the step then took; 0 for the final evaluation, after which no
        update is made.
    parameter_names : tuple of str
        The names of the parameters, in the order of the last axis of :attr:`theta`.
    theta : numpy.ndarray
        The evaluated particles in constrained coordinates, shape $(J, d)$.
    predictions : numpy.ndarray
        $G$ at each particle, shape $(J, n)$; a row of NaN where the forward map failed, which
        the method replaces before updating.
    noise_variance : numpy.ndarray
        The diagonal of $\Gamma$, the noise covariance at $\beta = 1$, shape $(n,)$.
    """

    step: int
    beta: float
    increment: float
    parameter_names: tuple[str, ...]
    theta: np.ndarray
    predictions: np.ndarray
    noise_variance: np.ndarray

    @property
    def n_failed(self) -> int:
        """How many particles' predictions were not finite."""
        return int(np.sum(~np.all(np.isfinite(self.predictions), axis=-1)))


#: A callable an ensemble method calls with every :class:`StepRecord`, in order.
StepCallback = Callable[[StepRecord], None]


def history_dataset(records: Sequence[StepRecord]) -> xr.Dataset:
    """A run's step records as one dataset, with dimensions ``step``, ``member``, ``parameter``
    and ``observation``.

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
            "noise_variance": ("observation", records[0].noise_variance),
        },
        coords={"step": [record.step for record in records], "parameter": names},
    )
