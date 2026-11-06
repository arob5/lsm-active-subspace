r"""Estimators of the information matrix in whitened coordinates, each returning a Subspace.

The information matrix averaged over a law $\mu$ of whitened coordinates is

$$
H(\mu) = \mathbb E_\mu\!\left[DG(\xi)^\top \Gamma^{-1} DG(\xi)\right].
$$

Its leading eigenvectors span the directions the data inform most. The estimators differ in what
they need and in which average they estimate:

- :class:`LadderLinearization`, a callback of an ensemble run, accumulates
  $\sum_t \Delta\beta_t\, \widehat{DG}_t^\top \Gamma^{-1} \widehat{DG}_t$ over the tempering
  ladder, at no cost beyond the run's;
- :func:`exact_subspace` averages exact Jacobians at given points, for a forward map JAX can
  differentiate;
- :func:`finite_difference_subspace` averages central-difference Jacobians, at $2d$ forward runs
  per point;
- :func:`covariance_subspace` uses $\Sigma^{-1} - I$, with $\Sigma$ a posterior sample's whitened
  covariance.

A statistical linearization $\widehat{DG}$ of an ensemble is its least-squares slope. For a
Gaussian ensemble it equals the average Jacobian $\mathbb E_\mu[DG]$ (Stein's lemma), so
$\widehat{DG}^\top \Gamma^{-1} \widehat{DG}$ estimates
$\mathbb E_\mu[DG]^\top \Gamma^{-1} \mathbb E_\mu[DG]$, which is smaller than $H(\mu)$ by
$\mathbb E_\mu[(DG - \mathbb E_\mu DG)^\top \Gamma^{-1} (DG - \mathbb E_\mu DG)]$: its eigenvalues
are biased low, while its leading directions are usually close to those of $H(\mu)$. For a linear
forward map and a Gaussian prior, $\Sigma^{-1} - I$ is exactly $DG^\top \Gamma^{-1} DG$.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from enskit import maps
from enskit.distribution import Ensemble

from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.functions import declares_traceable
from lsmas.inference.step_record import StepRecord
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools.subspace import DEFAULT_DISCARDED_INFORMATION, Subspace

__all__ = [
    "LadderLinearization",
    "LinearizedStep",
    "covariance_subspace",
    "exact_subspace",
    "finite_difference_subspace",
    "statistical_linearization",
]

logger = logging.getLogger(__name__)

#: How each step of a ladder is weighted: by its increment, equally, or only the last step.
Weighting = Literal["increment", "uniform", "last"]


@dataclass(eq=False)
class LadderLinearization:
    r"""A callback that accumulates statistical linearizations over an ensemble method's ladder.

    At each step $t$ the method reports its ensemble $\xi_{tj}$, $j = 1, \dots, J$ (whitened here),
    the predictions $G(\xi_{tj})$ and its noise covariance $\Gamma_t$. The callback fits the
    least-squares slope $\widehat{DG}_t$ of the predictions on the particles (EnsKit's statistical
    linearization) and accumulates

    $$
    \hat H = \sum_t w_t\, \widehat{DG}_t^\top\, \Gamma_t^{-1}\, \widehat{DG}_t,
    $$

    with weights set by *weighting*: ``"increment"`` (the default), $w_t = \Delta\beta_t$, a
    Riemann sum over the ladder in which each level counts by how much of the likelihood its step
    conditioned on; ``"uniform"``, $w_t = 1/T$ for $T$ steps; or ``"last"``, only the last step.

    Pass it to an ensemble method, then read the result::

        ladder = LadderLinearization(problem.coordinates)
        problem.condition(method="enskit_eki", method_options={"callbacks": [ladder]})
        subspace = ladder.subspace()

    When the ensemble has no more particles than parameters, the slope is fitted on the span of
    the particles and set to zero off it.

    Attributes
    ----------
    coordinates : Coordinates
        The coordinates that whiten the particles.
    weighting : {"increment", "uniform", "last"}
        How the steps are weighted.
    steps : list of LinearizedStep
        Every step seen so far.
    """

    coordinates: Coordinates
    weighting: Weighting = "increment"
    steps: list[LinearizedStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.weighting not in ("increment", "uniform", "last"):
            raise ValueError(f"unknown weighting {self.weighting!r}")

    def __call__(self, record: StepRecord) -> None:
        xi = self.coordinates.whitened_from_parameters(record.theta)
        slope = statistical_linearization(xi, record.predictions)
        self.steps.append(
            LinearizedStep(
                beta=record.beta,
                increment=record.increment,
                information=slope.T @ (slope / record.noise_variance[:, None]),
            )
        )

    @property
    def information(self) -> np.ndarray:
        r"""$\hat H$, shape $(d, d)$."""
        if not self.steps:
            raise ValueError("no step has been seen yet")
        if self.weighting == "last":
            return self.steps[-1].information
        weights = np.array(
            [step.increment if self.weighting == "increment" else 1.0 for step in self.steps]
        )
        if self.weighting == "uniform":
            weights /= len(self.steps)
        return np.einsum("t,tij->ij", weights, np.stack([s.information for s in self.steps]))

    def subspace(
        self,
        rank: int | None = None,
        discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
    ) -> Subspace:
        r"""The subspace of the leading eigenvectors of $\hat H$.

        Parameters
        ----------
        rank, discarded_information
            As for :meth:`Subspace.from_information`.
        """
        return Subspace.from_information(
            self.information,
            parameter_names=self.coordinates.names,
            estimator=f"ladder_linearization_{self.weighting}",
            rank=rank,
            discarded_information=discarded_information,
        )


@dataclass(frozen=True)
class LinearizedStep:
    r"""One step's contribution to :class:`LadderLinearization`.

    Attributes
    ----------
    beta, increment : float
        The step's level $\beta_t$ and increment $\Delta\beta_t$.
    information : numpy.ndarray
        $\widehat{DG}_t^\top \Gamma_t^{-1} \widehat{DG}_t$, shape $(d, d)$.
    """

    beta: float
    increment: float
    information: np.ndarray


def statistical_linearization(xi: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    r"""The least-squares slope $\widehat{DG}$ of *predictions* on *xi*, shape $(n, d)$.

    $\widehat{DG} = C_{G\xi}\, C_{\xi\xi}^{-1}$ for the ensemble's covariances, computed by EnsKit
    (``maps.statistical_linearization``, minimum-norm when $C_{\xi\xi}$ is singular).

    Parameters
    ----------
    xi : numpy.ndarray
        The particles, shape $(J, d)$.
    predictions : numpy.ndarray
        $G$ at each particle, shape $(J, n)$.
    """
    ensemble = Ensemble(xi=jnp.asarray(xi), predictions=jnp.asarray(predictions))
    fit = maps.statistical_linearization(ensemble, inputs="xi", output="predictions", min_norm=True)
    return np.asarray(fit.map.op.to_dense())


def exact_subspace(
    problem: InverseProblem,
    xi: np.ndarray,
    *,
    scale_values: dict[str, float] | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
    chunk_size: int = 25,
) -> Subspace:
    r"""The subspace of $\hat H = \frac1N \sum_{m} DG(\xi_m)^\top \Gamma^{-1} DG(\xi_m)$, with
    exact Jacobians.

    The points $\xi_m$ are draws of the law to average over, such as prior or posterior draws in
    whitened coordinates. The forward map must be one JAX can differentiate (``traceable=True``),
    as the toy problem's is.

    Parameters
    ----------
    problem : InverseProblem
        The problem whose forward map and noise covariance to use.
    xi : numpy.ndarray
        The points, shape $(N, d)$.
    scale_values : dict of str to float, optional
        The inferred noise scales' values in $\Gamma$; not needed when they are fixed.
    rank, discarded_information
        As for :meth:`Subspace.from_information`.
    chunk_size : int
        How many Jacobians to compute at once; larger is faster and uses more memory.
    """
    if not declares_traceable(problem.forward):
        raise ValueError(
            f"{problem.name}'s forward map cannot be differentiated by JAX; "
            "use finite_difference_subspace"
        )
    forward = problem.forward.using(raw=True)
    coordinates = problem.coordinates
    jacobian = jax.jit(jax.vmap(jax.jacfwd(lambda x: forward(coordinates.parameter_record(x)))))
    precision = 1.0 / problem.noise_variance(scale_values)
    xi = np.asarray(xi, dtype=float)
    total = np.zeros((xi.shape[1], xi.shape[1]))
    for start in range(0, len(xi), chunk_size):
        jacobians = np.asarray(jacobian(jnp.asarray(xi[start : start + chunk_size])))
        total += np.einsum("mni,n,mnj->ij", jacobians, precision, jacobians)
    return Subspace.from_information(
        total / len(xi),
        parameter_names=coordinates.names,
        estimator="exact",
        rank=rank,
        discarded_information=discarded_information,
    )


def finite_difference_subspace(
    problem: InverseProblem,
    xi: np.ndarray,
    *,
    step: float = 0.2,
    scale_values: dict[str, float] | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The subspace of $\hat H$ with Jacobians by central differences in whitened coordinates.

    At each point, column $i$ of $DG(\xi_m)$ is $\left(G(\xi_m + h e_i) - G(\xi_m - h e_i)\right)
    / 2h$, from $2d$ forward runs; the runs of every point are evaluated as one batch. A point at
    which any run fails is left out, with a warning.

    SIPNET writes its output to three decimals, which puts a floor under the differences' noise:
    in these units it is about $0.002 / h$, so $h$ should be at least $0.1$.

    Parameters
    ----------
    problem : InverseProblem
        The problem whose forward map and noise covariance to use.
    xi : numpy.ndarray
        The points, shape $(N, d)$.
    step : float
        $h$, in prior standard deviations.
    scale_values, rank, discarded_information
        As for :func:`exact_subspace`.
    """
    coordinates = problem.coordinates
    precision = 1.0 / problem.noise_variance(scale_values)
    xi = np.asarray(xi, dtype=float)
    dimension = xi.shape[1]
    shifts = step * np.concatenate([np.eye(dimension), -np.eye(dimension)])
    perturbed = (xi[:, None, :] + shifts[None, :, :]).reshape(-1, dimension)
    theta = coordinates.parameters_from_whitened(perturbed)
    predictions = np.asarray(problem.forward(coordinates.bijection.record_batch(theta)).raw)
    predictions = predictions.reshape(len(xi), 2 * dimension, -1)
    informations = []
    for point_predictions in predictions:
        if not np.all(np.isfinite(point_predictions)):
            logger.warning("the forward map failed near a point; it is left out")
            continue
        jacobian = (point_predictions[:dimension] - point_predictions[dimension:]).T / (2 * step)
        informations.append(jacobian.T @ (precision[:, None] * jacobian))
    if not informations:
        raise ValueError("the forward map failed near every point")
    return Subspace.from_information(
        np.mean(informations, axis=0),
        parameter_names=coordinates.names,
        estimator="finite_differences",
        rank=rank,
        discarded_information=discarded_information,
    )


def covariance_subspace(
    problem: InverseProblem,
    theta: np.ndarray,
    *,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The subspace of $\Sigma^{-1} - I$, with $\Sigma$ the whitened covariance of a posterior
    sample.

    The prior's whitened covariance is $I$, so $\Sigma^{-1} - I$ is the precision the data add.
    For a linear forward map and a Gaussian prior it equals $DG^\top \Gamma^{-1} DG$ exactly; it
    needs no forward runs beyond those that produced the sample.

    Parameters
    ----------
    problem : InverseProblem
        The problem whose coordinates whiten the sample.
    theta : numpy.ndarray
        Posterior draws of the calibration parameters, shape $(N, d)$ with $N > d$.
    rank, discarded_information
        As for :meth:`Subspace.from_information`.
    """
    xi = problem.coordinates.whitened_from_parameters(np.asarray(theta, dtype=float))
    if xi.shape[0] <= xi.shape[1]:
        raise ValueError(f"need more draws than parameters, got {xi.shape[0]} for {xi.shape[1]}")
    precision = np.linalg.inv(np.cov(xi, rowvar=False))
    return Subspace.from_information(
        precision - np.eye(xi.shape[1]),
        parameter_names=problem.coordinates.names,
        estimator="covariance",
        rank=rank,
        discarded_information=discarded_information,
    )
