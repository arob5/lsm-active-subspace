r"""Estimators of the active subspace: the leading eigenvectors of an information matrix.

The information matrix averaged over a distribution $\mu$ of the parameters is

$$
H(\mu) = \mathbb E_{\theta \sim \mu}\!\left[DG(z)^\top \Gamma^{-1} DG(z)\right],
\qquad z = z(\theta),
$$

where $z$ are the whitened coordinates of $\theta$ (:mod:`lsmas.coordinates`), $DG(z)$ is the
Jacobian of the forward map with respect to them, and $\Gamma$ is the noise covariance widened by
the learning rate (``InverseProblem.noise_variance``). Its leading eigenvectors span the directions
the data inform most. The subspace is always in whitened coordinates; the inputs are in the
problem's own parameters.

**The estimators**, which differ in what they need and in what they cost:

- :func:`exact_subspace` averages exact Jacobians (:func:`jacobians`), for a forward map JAX can
  differentiate;
- :func:`finite_difference_subspace` averages central-difference Jacobians, at $2d$ forward runs
  per draw;
- :func:`linearization_subspace` fits one statistical linearization to the draws, at one forward
  run per draw;
- :func:`eki_linearization_subspace` fits one statistical linearization to each step of an EKI
  run and averages them over the steps, with no forward runs beyond the run's own;
- :func:`covariance_subspace` uses the whitened covariance of a posterior's draws, with no forward
  runs.

Each builds its information matrix with :func:`information_matrix`.

**What they read from the problem.** At most three things: the forward map ``problem.forward``,
the coordinates ``problem.coordinates`` (which depend on the prior), and the noise covariance
$\Gamma$. None of them uses the observed data. Each function's description says which it reads.

**The distribution $\mu$.** :func:`exact_subspace`, :func:`finite_difference_subspace`,
:func:`linearization_subspace` and :func:`covariance_subspace` take the problem and $\mu$, as one
of:

- a distribution held as draws (an ``EmpiricalDistribution``), such as a posterior from MCMC or
  EKI, or a reference posterior: every draw is used, with its weight;
- a batch of draws (a ``RecordBatch``), such as ``sample(problem.prior, 500)``: every draw is
  used, with equal weights;
- any other distribution, such as ``problem.prior``: ``num_samples`` draws are taken from it, from
  the current ``workflow_run``'s random stream.

Giving ``num_samples`` for the first two resamples them. The draws may be of the whole record of
the problem's parameters, with the calibration parameters under ``theta``, or of the record
``theta`` alone.

**Noise scales.** When the problem infers its noise scales, $\Gamma$ depends on them. If the
draws of $\mu$ include the scales (the record ``s2``), as a posterior's do, each draw's own scales
are used for that draw. Otherwise, give their values as ``noise_scales``, such as
``{"day": 1.0, "night": 1.0}``; this also overrides the draws' scales.

**Statistical linearization.** A statistical linearization $\widehat{DG}$ of an ensemble is its
least-squares slope. For a Gaussian ensemble it equals the average Jacobian $\mathbb E_\mu[DG]$
(Stein's lemma), so $\widehat{DG}^\top \Gamma^{-1} \widehat{DG}$ estimates
$\mathbb E_\mu[DG]^\top \Gamma^{-1} \mathbb E_\mu[DG]$, which is smaller than $H(\mu)$ by
$\mathbb E_\mu[(DG - \mathbb E_\mu DG)^\top \Gamma^{-1} (DG - \mathbb E_\mu DG)]$: its eigenvalues
are biased low, while its leading directions are usually close to those of $H(\mu)$. For a linear
forward map and a Gaussian prior, $\Sigma^{-1} - I$ is exactly $DG^\top \Gamma^{-1} DG$.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from enskit import maps
from enskit.distribution import Ensemble
from probpipe import Distribution, EmpiricalDistribution, NumericRecordBatch, RecordBatch, sample

from lsmas._stopgaps.functions import declares_traceable
from lsmas._stopgaps.likelihood import NOISE_SCALES
from lsmas.coordinates import inverse
from lsmas.inference.step_record import eki_history
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools.subspace import DEFAULT_DISCARDED_INFORMATION, Subspace

__all__ = [
    "Weighting",
    "covariance_subspace",
    "eki_linearization_subspace",
    "exact_subspace",
    "finite_difference_subspace",
    "information_matrix",
    "jacobians",
    "linearization_subspace",
    "statistical_linearization",
]

logger = logging.getLogger(__name__)

#: How :func:`eki_linearization_subspace` weights the steps of an EKI run: by their increments,
#: equally, or only the last step.
Weighting = Literal["increment", "uniform", "last"]


def exact_subspace(
    problem: InverseProblem,
    distribution: Distribution | RecordBatch,
    *,
    num_samples: int | None = None,
    noise_scales: Mapping[str, float] | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
    batch_size: int = 25,
) -> Subspace:
    r"""The active subspace of the information matrix averaged over a distribution, with exact
    Jacobians.

    For a distribution $\mu$ of the parameters, with draws $\theta_m$ and weights $w_m$, $m = 1,
    \dots, N$ (see the module's description), this computes

    $$
    \hat H(\mu) = \sum_{m=1}^{N} w_m\, DG(z_m)^\top\, \Gamma_m^{-1}\, DG(z_m),
    \qquad z_m = z(\theta_m),
    $$

    an estimate of $H(\mu) = \mathbb E_\mu[DG(z)^\top \Gamma^{-1} DG(z)]$, and returns the subspace
    of its leading eigenvectors (:meth:`Subspace.from_information`). The Jacobians are
    :func:`jacobians`, so the forward map must be one JAX can differentiate. It runs no forward
    model evaluations beyond the differentiation itself.

    Reads from *problem*: ``forward``, ``coordinates`` and the noise covariance $\Gamma$
    (``noise_variance``). The observed data are not used.

    Parameters
    ----------
    problem : InverseProblem
        The problem.
    distribution : Distribution or RecordBatch
        $\mu$: a distribution of the problem's parameters, such as ``problem.prior`` or a
        posterior, or a batch of draws of them, such as ``sample(problem.prior, 500)``. See the
        module's description for how it is used.
    num_samples : int, optional
        Draw this many samples of *distribution*. Required for a distribution that is not held as
        draws, such as the prior; otherwise every held draw is used, with its weight.
    noise_scales : Mapping of str to float, optional
        The value of each inferred noise scale, by class, such as ``{"day": 1.0, "night": 1.0}``,
        used at every draw. By default each draw's own scales are used, when its draws include
        them; not needed when the scales are fixed.
    rank, discarded_information
        As for :meth:`Subspace.from_information`.
    batch_size : int
        How many Jacobians JAX computes at once; larger is faster and uses more memory.

    Returns
    -------
    Subspace
        With ``estimator="exact"``.

    Examples
    --------
    >>> exact_subspace(toy, toy.prior, num_samples=500)          # doctest: +SKIP
    >>> exact_subspace(toy, reference.posterior)                  # doctest: +SKIP
    """
    draws, weights = _draws_to_average(distribution, num_samples)
    z = _whitened_parameters(problem, draws)
    noise_variance = _noise_variance_at_draws(problem, draws, noise_scales)
    jacobians_at_draws = _whitened_jacobians(problem, z, batch_size)
    return Subspace.from_information(
        information_matrix(jacobians_at_draws, noise_variance, weights),
        parameter_names=problem.coordinates.names,
        estimator="exact",
        rank=rank,
        discarded_information=discarded_information,
    )


def finite_difference_subspace(
    problem: InverseProblem,
    distribution: Distribution | RecordBatch,
    *,
    num_samples: int | None = None,
    step: float = 0.2,
    noise_scales: Mapping[str, float] | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The active subspace of the information matrix averaged over a distribution, with
    Jacobians by central differences.

    As :func:`exact_subspace`, but column $j$ of each Jacobian is the central difference

    $$
    DG(z_m)\, e_j \approx \frac{G(z_m + h e_j) - G(z_m - h e_j)}{2h},
    $$

    with $e_j$ the $j$-th unit vector in whitened coordinates and $h$ the *step*, in prior
    standard deviations. It needs no gradients, at a cost of $2d$ forward runs per draw, all
    evaluated as one batch. A draw at which any of its runs fails is left out, with a warning,
    and the weights of the others are renormalized.

    SIPNET writes its output to three decimals, which puts a floor under the differences' noise:
    in these units it is about $0.002 / h$, so $h$ should be at least $0.1$.

    Reads from *problem*: ``forward``, ``coordinates`` and the noise covariance $\Gamma$. The
    observed data are not used.

    Parameters
    ----------
    problem, distribution, num_samples, noise_scales, rank, discarded_information
        As for :func:`exact_subspace`.
    step : float
        $h$.

    Returns
    -------
    Subspace
        With ``estimator="finite_differences"``.

    Raises
    ------
    ValueError
        If the forward map fails near every draw.
    """
    draws, weights = _draws_to_average(distribution, num_samples)
    z = _whitened_parameters(problem, draws)
    noise_variance = _noise_variance_at_draws(problem, draws, noise_scales)
    dimension = problem.coordinates.dimension
    shifts = step * np.concatenate([np.eye(dimension), -np.eye(dimension)])
    shifted_z = (z[:, None, :] + shifts[None, :, :]).reshape(-1, dimension)
    predictions = _predictions_at_whitened(problem, shifted_z).reshape(len(z), 2 * dimension, -1)
    central_differences = (predictions[:, :dimension] - predictions[:, dimension:]) / (2 * step)
    jacobians_at_draws = np.swapaxes(central_differences, 1, 2)
    succeeded = np.all(np.isfinite(predictions), axis=(1, 2))
    if not succeeded.any():
        raise ValueError("the forward map failed near every draw")
    if not succeeded.all():
        logger.warning(
            "the forward map failed near %d draws; they are left out", (~succeeded).sum()
        )
    return Subspace.from_information(
        information_matrix(
            jacobians_at_draws[succeeded],
            noise_variance[succeeded],
            weights[succeeded] / weights[succeeded].sum(),
        ),
        parameter_names=problem.coordinates.names,
        estimator="finite_differences",
        rank=rank,
        discarded_information=discarded_information,
    )


def linearization_subspace(
    problem: InverseProblem,
    distribution: Distribution | RecordBatch,
    *,
    num_samples: int | None = None,
    noise_scales: Mapping[str, float] | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The active subspace of the information from a statistical linearization over draws of a
    distribution, without gradients.

    It runs the forward model at the draws $\theta_m$ of $\mu$ (see the module's description),
    fits the least-squares slope $\widehat{DG}$ of the predictions $G(\theta_m)$ on the whitened
    draws $z_m$ (:func:`statistical_linearization`), and returns the subspace of the leading
    eigenvectors of

    $$
    \hat H = \widehat{DG}^\top\, \bar\Gamma^{-1}\, \widehat{DG},
    $$

    with $\bar\Gamma^{-1}$ the average of the draws' noise precisions. For draws from a Gaussian,
    $\widehat{DG}$ estimates the average Jacobian $\mathbb E_\mu[DG]$, so $\hat H$ estimates
    $\mathbb E_\mu[DG]^\top \Gamma^{-1} \mathbb E_\mu[DG]$, which is smaller than $H(\mu)$: its
    eigenvalues are biased low, while its leading directions are usually close to those of
    $H(\mu)$. Each draw costs one forward run, all evaluated as one batch; draws at which the
    forward map fails are left out, with a warning. The predictions are paired with the draws they
    came from, which is why this function, rather than its caller, runs the forward model.

    Reads from *problem*: ``forward``, ``coordinates`` and the noise covariance $\Gamma$. The
    observed data are not used.

    Parameters
    ----------
    problem, distribution, num_samples, noise_scales, rank, discarded_information
        As for :func:`exact_subspace`. The draws must have equal weights; for a distribution
        whose held draws do not, give *num_samples* to resample it.

    Returns
    -------
    Subspace
        With ``estimator="statistical_linearization"``.

    Raises
    ------
    ValueError
        If the held draws have unequal weights, or fewer than two draws remain.
    """
    draws, weights = _draws_to_average(distribution, num_samples)
    if not np.allclose(weights, weights[0]):
        raise ValueError(
            "a statistical linearization needs equally weighted draws; give num_samples to "
            "resample the distribution"
        )
    z = _whitened_parameters(problem, draws)
    noise_variance = _noise_variance_at_draws(problem, draws, noise_scales)
    predictions = np.asarray(problem.forward(_parameter_draws(draws)).raw).reshape(len(z), -1)
    succeeded = np.all(np.isfinite(predictions), axis=1)
    if not succeeded.all():
        logger.warning("the forward map failed at %d draws; they are left out", (~succeeded).sum())
    if succeeded.sum() < 2:
        raise ValueError("a statistical linearization needs at least two draws")
    slope = statistical_linearization(z[succeeded], predictions[succeeded])
    return Subspace.from_information(
        information_matrix(slope, _average_noise_variance(noise_variance[succeeded])),
        parameter_names=problem.coordinates.names,
        estimator="statistical_linearization",
        rank=rank,
        discarded_information=discarded_information,
    )


def eki_linearization_subspace(
    problem: InverseProblem,
    eki_posterior: EmpiricalDistribution,
    *,
    weighting: Weighting = "increment",
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The active subspace of the statistical linearizations along an EKI run, averaged over its
    steps.

    At step $t$ of the run, the particles $\theta_{t1}, \dots, \theta_{tJ}$ approximate the
    tempered distribution $\pi_{\beta_t}$, and EKI has run the forward model at each of them. This
    function reads those particles and predictions from the run's history
    (:func:`~lsmas.inference.eki_history`), fits the statistical linearization $\widehat{DG}_t$ of
    each step's predictions on its whitened particles (:func:`statistical_linearization`), and
    returns the subspace of the leading eigenvectors of

    $$
    \hat H = \sum_{t=0}^{T-1} w_t\, \widehat{DG}_t^\top\, \Gamma_t^{-1}\, \widehat{DG}_t,
    $$

    with $\Gamma_t$ the noise covariance step $t$ used (at the ensemble's common noise scales,
    when they are inferred) and weights set by *weighting*:

    - ``"increment"`` (the default): $w_t = \Delta\beta_t$, the step's increment. The increments
      sum to 1, so $\hat H$ is a Riemann sum over the tempering levels of the linearized
      information, $\int_0^1 \mathbb E_{\pi_\beta}[DG]^\top \Gamma^{-1}
      \mathbb E_{\pi_\beta}[DG]\, d\beta$;
    - ``"uniform"``: $w_t = 1/T$ for $T$ steps;
    - ``"last"``: only the last step, the ensemble closest to the posterior.

    It runs no forward model evaluations, and it does not change the EKI run, which it only reads.

    Reads from *problem*: ``coordinates``.

    Parameters
    ----------
    problem : InverseProblem
        The problem the EKI run conditioned.
    eki_posterior : EmpiricalDistribution
        The result of ``problem.condition(method="enskit_eki")``.
    weighting : {"increment", "uniform", "last"}
        How the steps are weighted.
    rank, discarded_information
        As for :meth:`Subspace.from_information`.

    Returns
    -------
    Subspace
        With ``estimator="eki_linearization_<weighting>"``.

    Raises
    ------
    ValueError
        If *eki_posterior* is not the result of an EKI run, or *weighting* is unknown.

    Examples
    --------
    >>> eki_posterior = toy.condition(method="enskit_eki")                  # doctest: +SKIP
    >>> eki_linearization_subspace(toy, eki_posterior)                       # doctest: +SKIP
    """
    history = eki_history(eki_posterior)
    coordinates = problem.coordinates
    particles = NumericRecordBatch.from_vector(
        coordinates.spec, history["theta"].values, level_names=["step", "member"]
    )
    z_at_steps = np.asarray(coordinates.whiten(particles).to_vector())
    predictions_at_steps = history["predictions"].values
    slopes = np.stack(
        [
            statistical_linearization(z, predictions)
            for z, predictions in zip(z_at_steps, predictions_at_steps, strict=True)
        ]
    )
    weights = _step_weights(history["increment"].values, weighting)
    return Subspace.from_information(
        information_matrix(slopes, history["noise_variance"].values, weights),
        parameter_names=coordinates.names,
        estimator=f"eki_linearization_{weighting}",
        rank=rank,
        discarded_information=discarded_information,
    )


def covariance_subspace(
    problem: InverseProblem,
    distribution: Distribution | RecordBatch,
    *,
    num_samples: int | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The active subspace of the precision a posterior adds to the prior, from its draws.

    For a posterior $\mu$ with whitened covariance $\Sigma$, estimated from its (weighted) draws,
    this returns the subspace of the leading eigenvectors of $\Sigma^{-1} - I$. The prior's
    whitened covariance is $I$, so $\Sigma^{-1} - I$ is the precision the data add; for a linear
    forward map and a Gaussian prior it equals $DG^\top \Gamma^{-1} DG$ exactly. It runs no
    forward model evaluations: it needs only draws of the posterior, more of them than there are
    parameters.

    Reads from *problem*: ``coordinates``.

    Parameters
    ----------
    problem, distribution, num_samples, rank, discarded_information
        As for :func:`exact_subspace`; *distribution* is a posterior, or draws of one.

    Returns
    -------
    Subspace
        With ``estimator="covariance"``.

    Raises
    ------
    ValueError
        If there are no more draws than parameters.
    """
    draws, weights = _draws_to_average(distribution, num_samples)
    z = _whitened_parameters(problem, draws)
    dimension = problem.coordinates.dimension
    if len(z) <= dimension:
        raise ValueError(f"need more draws than parameters, got {len(z)} for {dimension}")
    covariance = np.cov(z, rowvar=False, aweights=weights)
    return Subspace.from_information(
        np.linalg.inv(covariance) - np.eye(dimension),
        parameter_names=problem.coordinates.names,
        estimator="covariance",
        rank=rank,
        discarded_information=discarded_information,
    )


def jacobians(problem: InverseProblem, theta: Any, *, batch_size: int = 25) -> np.ndarray:
    r"""The Jacobian $DG(z)$ of the forward map in whitened coordinates, at given parameters.

    For the forward map $G$ and the whitened coordinates $z$ of the parameters (see
    :mod:`lsmas.coordinates`), $G(z) = G(\theta(z))$ and

    $$
    DG(z)_{ij} = \frac{\partial G_i}{\partial z_j}(z), \qquad i = 1, \dots, n,\quad
    j = 1, \dots, d,
    $$

    evaluated at $z = z(\theta)$ for each given $\theta$. Column $j$ says how the predictions
    change as $z_j$ changes. JAX computes it exactly, so the forward map must be one JAX can
    differentiate (``traceable=True``), as the toy problem's is; SIPNET's is not.

    Reads from *problem*: ``forward`` and ``coordinates``.

    Parameters
    ----------
    problem : InverseProblem
        The problem whose forward map to differentiate.
    theta : Record, RecordBatch or Mapping
        Values of the parameter record ``theta``: one, such as
        ``quantile(problem.prior, 0.5)["theta"]``, or a batch, such as
        ``sample(problem.prior, 100)["theta"]``.
    batch_size : int
        How many Jacobians JAX computes at once; larger is faster and uses more memory.

    Returns
    -------
    numpy.ndarray
        Shape $(n, d)$ for one setting, or the batch's shape followed by $(n, d)$.

    Raises
    ------
    ValueError
        If JAX cannot differentiate the forward map.
    """
    z = np.asarray(problem.coordinates.whiten(theta).to_vector())
    jacobian_rows = _whitened_jacobians(
        problem, z.reshape(-1, problem.coordinates.dimension), batch_size
    )
    return jacobian_rows.reshape(*z.shape[:-1], *jacobian_rows.shape[1:])


def statistical_linearization(z: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    r"""The least-squares slope $\widehat{DG}$ of *predictions* on *z*, shape $(n, d)$.

    Fitting $G(z_j) \approx c + \widehat{DG}\, z_j$ by least squares over an ensemble gives

    $$
    \widehat{DG} = C_{Gz}\, C_{zz}^{-1},
    $$

    with $C_{Gz}$ and $C_{zz}$ the ensemble's covariances, computed by EnsKit
    (``maps.statistical_linearization``, minimum-norm when $C_{zz}$ is singular, which happens
    when the ensemble has no more members than parameters). This is the low-level regression;
    :func:`linearization_subspace` draws the ensemble, runs the forward model and whitens the
    draws itself. To estimate the active subspace, *z* must be whitened coordinates, since the
    information matrix is defined in them, and row $j$ of *predictions* must be the forward map at
    row $j$ of *z*.

    Parameters
    ----------
    z : numpy.ndarray
        The ensemble in whitened coordinates, shape $(J, d)$.
    predictions : numpy.ndarray
        $G$ at each member of the ensemble, shape $(J, n)$.
    """
    ensemble = Ensemble(z=jnp.asarray(z), predictions=jnp.asarray(predictions))
    linear_fit = maps.statistical_linearization(
        ensemble, inputs="z", output="predictions", min_norm=True
    )
    return np.asarray(linear_fit.map.op.to_dense())


def information_matrix(
    jacobian: np.ndarray, noise_variance: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    r"""The information matrix of one Jacobian, or the weighted sum over several.

    For Jacobians $J_m$ ($n \times d$), noise covariances $\Gamma_m = \operatorname{diag}(\gamma_m)$
    and weights $w_m$,

    $$
    \sum_{m} w_m\, J_m^\top\, \Gamma_m^{-1}\, J_m,
    $$

    a $d \times d$ matrix. For one Jacobian, this is $J^\top \Gamma^{-1} J$, the curvature of the
    data misfit $\tfrac12 (y - G)^\top \Gamma^{-1} (y - G)$ when $G$ is replaced by its
    linearization.

    Parameters
    ----------
    jacobian : array_like
        One Jacobian, shape $(n, d)$, or several, shape $(N, n, d)$: exact (:func:`jacobians`),
        by finite differences, or a statistical linearization (:func:`statistical_linearization`).
    noise_variance : array_like
        The diagonal of $\Gamma$: shape $(n,)$, shared by every Jacobian, or $(N, n)$, one per
        Jacobian, such as ``problem.noise_variance()``.
    weights : array_like, optional
        $w_m$, shape $(N,)$; by default $1/N$ each, an average.

    Returns
    -------
    numpy.ndarray
        Shape $(d, d)$.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    jacobian_batch = jacobian[None] if jacobian.ndim == 2 else jacobian
    variance_batch = np.broadcast_to(
        np.asarray(noise_variance, dtype=float), jacobian_batch.shape[:2]
    )
    if weights is None:
        weights = np.full(len(jacobian_batch), 1.0 / len(jacobian_batch))
    return np.einsum(
        "m,mni,mn,mnj->ij",
        np.asarray(weights),
        jacobian_batch,
        1.0 / variance_batch,
        jacobian_batch,
    )


def _draws_to_average(
    distribution: Distribution | RecordBatch, num_samples: int | None
) -> tuple[RecordBatch, np.ndarray]:
    """The draws of the distribution an estimator averages over, and their weights, which sum to
    1 (see the module's description)."""
    if isinstance(distribution, RecordBatch):
        distribution = EmpiricalDistribution(distribution)
    if num_samples is not None:
        return sample(distribution, num_samples), np.full(num_samples, 1.0 / num_samples)
    if isinstance(distribution, EmpiricalDistribution):
        weights = np.asarray(distribution.weights, dtype=float).reshape(-1)
        return distribution.atoms, weights / weights.sum()
    raise ValueError(
        f"{distribution.label!r} is not held as draws; give num_samples to draw from it"
    )


def _parameter_draws(draws: RecordBatch) -> RecordBatch:
    """The draws of the parameter record ``theta``: a field of draws of a whole prior or posterior,
    or the draws themselves when they are draws of ``theta`` alone."""
    try:
        return draws["theta"]
    except KeyError:
        return draws


def _whitened_parameters(problem: InverseProblem, draws: RecordBatch) -> np.ndarray:
    """The whitened coordinates of each draw's parameters, shape $(N, d)$."""
    z = problem.coordinates.whiten(_parameter_draws(draws)).to_vector()
    return np.asarray(z).reshape(-1, problem.coordinates.dimension)


def _noise_variance_at_draws(
    problem: InverseProblem, draws: RecordBatch, noise_scales: Mapping[str, float] | None
) -> np.ndarray:
    r"""The diagonal of $\Gamma$ at each draw, shape $(N, n)$: at the given *noise_scales*, or at
    each draw's own scales when they are inferred and the draws hold them."""
    n_draws = int(np.prod(draws.batch_shape))
    inferred = problem.likelihood.inferred_scale_classes
    if not inferred:
        return np.broadcast_to(problem.noise_variance(), (n_draws, len(problem.noise_variance())))
    if noise_scales is None:
        noise_scales = _drawn_noise_scales(problem, draws)
    if missing := sorted(set(inferred) - set(noise_scales or {})):
        raise ValueError(
            f"the noise scales are inferred; give values for {missing} as noise_scales, or a "
            "distribution whose draws include them"
        )
    scales_per_draw = {
        noise_class: np.broadcast_to(np.asarray(noise_scales[noise_class], dtype=float), (n_draws,))
        for noise_class in inferred
    }
    return np.stack(
        [
            problem.noise_variance(
                {noise_class: scales[draw] for noise_class, scales in scales_per_draw.items()}
            )
            for draw in range(n_draws)
        ]
    )


def _drawn_noise_scales(
    problem: InverseProblem, draws: RecordBatch
) -> dict[str, np.ndarray] | None:
    """Each inferred noise scale's value at each draw, when the draws hold the record ``s2``."""
    try:
        draws[NOISE_SCALES]
    except KeyError:
        return None
    return {
        noise_class: scale_draws.reshape(-1)
        for noise_class, scale_draws in problem.likelihood.inferred_scale_draws(draws).items()
    }


def _average_noise_variance(noise_variance: np.ndarray) -> np.ndarray:
    r"""The noise variance whose precision is the draws' average precision,
    $\bar\Gamma = (\frac1N \sum_m \Gamma_m^{-1})^{-1}$, shape $(n,)$."""
    return 1.0 / np.mean(1.0 / noise_variance, axis=0)


def _step_weights(increments: np.ndarray, weighting: Weighting) -> np.ndarray:
    """The weight of each step of an EKI run, as :func:`eki_linearization_subspace` describes."""
    if weighting == "increment":
        return np.asarray(increments, dtype=float)
    if weighting == "uniform":
        return np.full(len(increments), 1.0 / len(increments))
    if weighting == "last":
        return np.eye(len(increments))[-1]
    raise ValueError(f"unknown weighting {weighting!r}")


def _predictions_at_whitened(problem: InverseProblem, z: np.ndarray) -> np.ndarray:
    """The forward map at each row of whitened coordinates *z*, shape $(N, n)$."""
    to_parameters = inverse(problem.coordinates.whiten)
    z_records = NumericRecordBatch.from_vector(
        to_parameters.input_spec["z"], jnp.asarray(z), level_names="point"
    )
    return np.asarray(problem.forward(to_parameters(z_records)).raw)


def _whitened_jacobians(problem: InverseProblem, z: np.ndarray, batch_size: int) -> np.ndarray:
    """$DG(z)$ at each row of whitened coordinates *z*, shape $(N, n, d)$, computed *batch_size*
    rows at a time."""
    jacobian_function = _whitened_jacobian_function(problem)
    return np.concatenate(
        [
            np.asarray(jacobian_function(jnp.asarray(z[start : start + batch_size])))
            for start in range(0, len(z), batch_size)
        ]
    )


@functools.lru_cache(maxsize=8)
def _whitened_jacobian_function(problem: InverseProblem) -> Callable[[jax.Array], jax.Array]:
    r"""$z \mapsto DG(z)$ for a batch of points, compiled once per problem."""
    if not declares_traceable(problem.forward):
        raise ValueError(
            f"{problem.name}'s forward map cannot be differentiated by JAX; "
            "use finite_difference_subspace or linearization_subspace"
        )
    forward = problem.forward.using(raw=True)
    to_parameters = inverse(problem.coordinates.whiten).using(raw=True)
    names = problem.coordinates.names

    def features_at_whitened(z: jax.Array) -> jax.Array:
        return forward(to_parameters(dict(zip(names, z, strict=True))))

    return jax.jit(jax.vmap(jax.jacfwd(features_at_whitened)))
