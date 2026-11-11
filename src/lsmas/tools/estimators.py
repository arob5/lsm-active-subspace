r"""Estimators of the information matrix in whitened coordinates, each returning a Subspace.

The information matrix averaged over a law $\mu$ of whitened coordinates is

$$
H(\mu) = \mathbb E_\mu\!\left[DG(z)^\top \Gamma^{-1} DG(z)\right].
$$

Its leading eigenvectors span the directions the data inform most. The estimators differ in what
they need and in which average they estimate:

- :class:`LadderLinearization`, a callback of an ensemble run, accumulates
  $\sum_t \Delta\beta_t\, \widehat{DG}_t^\top \Gamma^{-1} \widehat{DG}_t$ over the tempering
  ladder, with no forward runs beyond the ensemble run's own;
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from enskit import maps
from enskit.distribution import Ensemble

from lsmas._stopgaps.functions import declares_traceable
from lsmas.coordinates import Coordinates
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

    At each step $t$ the method reports its ensemble $z_{tj}$, $j = 1, \dots, J$ (whitened here),
    the predictions $G(z_{tj})$ and its noise covariance $\Gamma_t$. The callback fits the
    least-squares slope $\widehat{DG}_t$ of the predictions on the particles (EnsKit's statistical
    linearization) and accumulates

    $$
    \hat H = \sum_t w_t\, \widehat{DG}_t^\top\, \Gamma_t^{-1}\, \widehat{DG}_t,
    $$

    with weights set by *weighting*: ``"increment"`` (the default), $w_t = \Delta\beta_t$, a
    Riemann sum over the tempering levels in which each step is weighted by its increment;
    ``"uniform"``, $w_t = 1/T$ for $T$ steps; or ``"last"``, only the last step.

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
        z = np.asarray(self.coordinates._whitened(record.theta))
        slope = statistical_linearization(z, record.predictions)
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


def statistical_linearization(z: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    r"""The least-squares slope $\widehat{DG}$ of *predictions* on *z*, shape $(n, d)$.

    $\widehat{DG} = C_{Gz}\, C_{zz}^{-1}$ for the ensemble's covariances, computed by EnsKit
    (``maps.statistical_linearization``, minimum-norm when $C_{zz}$ is singular).

    Parameters
    ----------
    z : numpy.ndarray
        The particles, shape $(J, d)$.
    predictions : numpy.ndarray
        $G$ at each particle, shape $(J, n)$.
    """
    ensemble = Ensemble(z=jnp.asarray(z), predictions=jnp.asarray(predictions))
    fit = maps.statistical_linearization(ensemble, inputs="z", output="predictions", min_norm=True)
    return np.asarray(fit.map.op.to_dense())


def exact_subspace(
    problem: InverseProblem,
    z: np.ndarray,
    *,
    noise_scales: Mapping[str, float | np.ndarray] | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
    chunk_size: int = 25,
) -> Subspace:
    r"""The subspace of $\hat H = \frac1N \sum_{m} DG(z_m)^\top \Gamma^{-1} DG(z_m)$, with
    exact Jacobians.

    The points $z_m$ are draws of the law to average over, such as prior or posterior draws in
    whitened coordinates. The forward map must be one JAX can differentiate (``traceable=True``),
    as the toy problem's is.

    Parameters
    ----------
    problem : InverseProblem
        The problem whose forward map and noise covariance to use.
    z : numpy.ndarray
        The points, shape $(N, d)$.
    noise_scales : Mapping of str to float or numpy.ndarray, optional
        Each inferred noise scale's value in $\Gamma$, by class, such as ``{"day": 1.0, "night":
        1.0}``: one number for every point, or one value per point, shape $(N,)$, such as posterior
        draws of the scales. Not needed when the scales are fixed.
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

    def features_at_whitened(x: jax.Array) -> jax.Array:
        parameters = coordinates._from_whitened(x)
        return forward({name: parameters[j] for j, name in enumerate(coordinates.names)})

    jacobian = jax.jit(jax.vmap(jax.jacfwd(features_at_whitened)))
    z = np.asarray(z, dtype=float)
    precisions = _point_precisions(problem, noise_scales, len(z))
    total = np.zeros((z.shape[1], z.shape[1]))
    for start in range(0, len(z), chunk_size):
        jacobians = np.asarray(jacobian(jnp.asarray(z[start : start + chunk_size])))
        chunk_precisions = precisions[start : start + chunk_size]
        total += np.einsum("mni,mn,mnj->ij", jacobians, chunk_precisions, jacobians)
    return Subspace.from_information(
        total / len(z),
        parameter_names=coordinates.names,
        estimator="exact",
        rank=rank,
        discarded_information=discarded_information,
    )


def finite_difference_subspace(
    problem: InverseProblem,
    z: np.ndarray,
    *,
    step: float = 0.2,
    noise_scales: Mapping[str, float | np.ndarray] | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The subspace of $\hat H$ with Jacobians by central differences in whitened coordinates.

    At each point, column $i$ of $DG(z_m)$ is $\left(G(\z_m + h e_i) - G(\z_m - h e_i)\right)
    / 2h$, from $2d$ forward runs; the runs of every point are evaluated as one batch. A point at
    which any run fails is left out, with a warning.

    SIPNET writes its output to three decimals, which puts a floor under the differences' noise:
    in these units it is about $0.002 / h$, so $h$ should be at least $0.1$.

    Parameters
    ----------
    problem : InverseProblem
        The problem whose forward map and noise covariance to use.
    z : numpy.ndarray
        The points, shape $(N, d)$.
    step : float
        $h$, in prior standard deviations.
    noise_scales, rank, discarded_information
        As for :func:`exact_subspace`.
    """
    coordinates = problem.coordinates
    z = np.asarray(z, dtype=float)
    precisions = _point_precisions(problem, noise_scales, len(z))
    dimension = z.shape[1]
    shifts = step * np.concatenate([np.eye(dimension), -np.eye(dimension)])
    perturbed = (z[:, None, :] + shifts[None, :, :]).reshape(-1, dimension)
    perturbed_theta = coordinates._records(coordinates._from_whitened(perturbed))
    predictions = np.asarray(problem.forward(perturbed_theta).raw)
    predictions = predictions.reshape(len(z), 2 * dimension, -1)
    per_point_information = []
    for point_predictions, precision in zip(predictions, precisions, strict=True):
        if not np.all(np.isfinite(point_predictions)):
            logger.warning("the forward map failed near a point; it is left out")
            continue
        jacobian = (point_predictions[:dimension] - point_predictions[dimension:]).T / (2 * step)
        per_point_information.append(jacobian.T @ (precision[:, None] * jacobian))
    if not per_point_information:
        raise ValueError("the forward map failed near every point")
    return Subspace.from_information(
        np.mean(per_point_information, axis=0),
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
    z = np.asarray(problem.coordinates._whitened(np.asarray(theta, dtype=float)))
    if z.shape[0] <= z.shape[1]:
        raise ValueError(f"need more draws than parameters, got {z.shape[0]} for {z.shape[1]}")
    precision = np.linalg.inv(np.cov(z, rowvar=False))
    return Subspace.from_information(
        precision - np.eye(z.shape[1]),
        parameter_names=problem.coordinates.names,
        estimator="covariance",
        rank=rank,
        discarded_information=discarded_information,
    )


def _point_precisions(
    problem: InverseProblem,
    noise_scales: Mapping[str, float | np.ndarray] | None,
    n_points: int,
) -> np.ndarray:
    r"""The diagonal of $\Gamma^{-1}$ at each point, shape $(N, n)$, from each inferred scale's
    value: one number, or one value per point."""
    inferred = problem.likelihood.inferred_scale_classes
    if not inferred:
        return np.broadcast_to(
            1.0 / problem.noise_variance(), (n_points, len(problem.noise_variance()))
        )
    if missing := sorted(set(inferred) - set(noise_scales or {})):
        raise ValueError(f"the noise scales are inferred; give values for {missing}")
    per_point = {
        noise_class: np.broadcast_to(
            np.asarray(noise_scales[noise_class], dtype=float), (n_points,)
        )
        for noise_class in inferred
    }
    return np.stack(
        [
            1.0 / problem.noise_variance({c: values[m] for c, values in per_point.items()})
            for m in range(n_points)
        ]
    )
