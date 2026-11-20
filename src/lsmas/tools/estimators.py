r"""Estimators of the active subspace: the leading eigenvectors of an information matrix.

For a model ``likelihood & prior`` with forward map $G$ and noise variance
$s^2_{k(i)}\sigma^2_i$, conditioned with learning rates $\eta$, the information matrix averaged
over a distribution $\mu$ of the parameters is

$$
H_\eta(\mu) = \mathbb E_{\mu}\!\left[DG(z)^\top\, \Gamma^{-1}\, DG(z)\right],
\qquad \Gamma = \operatorname{diag}\!\left(s^2_{k(i)}\, \sigma^2_i / \eta_i\right),
$$

where $z$ are the whitened coordinates of the calibration parameters $\theta$
(:mod:`lsmas.coordinates`) and $DG(z)$ is the Jacobian of the forward map with respect to them. Its
leading eigenvectors span the directions the data inform most. The subspace is always in whitened
coordinates; the distribution $\mu$ is of the model's own parameters.

**The estimators**, which differ in what they need and in what they cost:

- :func:`gradient_based_subspace` averages exact Jacobians, for a forward map JAX can
  differentiate;
- :func:`finite_difference_subspace` averages central-difference Jacobians, at $2d$ forward runs
  per sample;
- :func:`linearization_subspace` fits one statistical linearization to samples of $\mu$, at one
  forward run per sample;
- :func:`eki_linearization_subspace` fits one statistical linearization to each step of an EKI
  run and averages them over the steps, with no forward runs beyond the run's own;
- :func:`covariance_subspace` uses the whitened covariance of a posterior's samples, with no
  forward runs.

**What they read from the model.** The law of the observations at a value *parameters* of the
parameters, ``condition_on(model, parameters)``: its mean is $G(\theta)$ and its variance is
$s^2_{k(i)}\sigma^2_i$, at the value's own noise scales when the model infers them. They read the
coordinates with ``Coordinates.from_model(model)``. The observed data are not used.

**The distribution $\mu$**, *parameter_distribution*, is a law of the model's parameters: a
prior or a posterior, or ``EmpiricalDistribution(samples)`` for samples in hand. Over a law held
as samples, every sample is used with its weight; from any other law, *num_samples* samples are
drawn from the current ``workflow_run``'s stream.

**Statistical linearization.** The statistical linearization of $G$ under $\mu$ is its
least-squares slope on $z$, $\widehat{DG} = \operatorname{Cov}(G, z)\operatorname{Cov}(z)^{-1}$
(:func:`statistical_linearization`). For a Gaussian $\mu$ it equals the average Jacobian
$\mathbb E_\mu[DG]$ (Stein's lemma), so $\widehat{DG}^\top \Gamma^{-1} \widehat{DG}$ estimates
$\mathbb E_\mu[DG]^\top \Gamma^{-1} \mathbb E_\mu[DG]$, which is smaller than $H(\mu)$ by
$\mathbb E_\mu[(DG - \mathbb E_\mu DG)^\top \Gamma^{-1} (DG - \mathbb E_\mu DG)]$: its eigenvalues
are biased low, while its leading directions are usually close to those of $H(\mu)$. For a linear
forward map and a Gaussian prior, $\Sigma^{-1} - I$ is exactly $DG^\top \Gamma^{-1} DG$.
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from probpipe import (
    Distribution,
    EmpiricalDistribution,
    NumericRecordBatch,
    condition_on,
    mean,
    variance,
)

from lsmas._stopgaps.jacobian import jacobian
from lsmas._stopgaps.operations import cov, expectation, pushforward
from lsmas.coordinates import Coordinates, inverse
from lsmas.inference.step_record import eki_history
from lsmas.tools.subspace import DEFAULT_DISCARDED_INFORMATION, Subspace

__all__ = [
    "Weighting",
    "covariance_subspace",
    "eki_linearization_subspace",
    "finite_difference_subspace",
    "gradient_based_subspace",
    "information_matrix",
    "linearization_subspace",
    "statistical_linearization",
]

#: How :func:`eki_linearization_subspace` weights the steps of an EKI run: by their increments,
#: equally, or only the last step.
Weighting = Literal["increment", "uniform", "last"]


def gradient_based_subspace(
    model: Distribution,
    parameter_distribution: Distribution,
    *,
    learning_rate: Any = 1.0,
    num_samples: int | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The active subspace of the information matrix averaged over a distribution of the
    parameters, with exact Jacobians.

    For a distribution $\mu$ of the parameters, this computes

    $$
    H_\eta(\mu) = \mathbb E_\mu\!\left[DG(z)^\top\, \Gamma^{-1}\, DG(z)\right],
    \qquad \Gamma = \operatorname{diag}\!\left(s^2_{k(i)}\, \sigma^2_i / \eta_i\right),
    $$

    by ProbPipe's ``expectation`` over $\mu$, and returns the subspace of its leading eigenvectors
    (:meth:`Subspace.from_information`). At each value of the parameters, $DG(z)$ is the Jacobian
    of the mean of the law of the observations, $G(\theta(z))$, computed exactly by JAX, so the
    forward map must be one JAX can differentiate. Each value's own noise scales set its $\Gamma$.
    It runs no forward model evaluations beyond the differentiation itself.

    Parameters
    ----------
    model : Distribution
        The model ``likelihood & prior``, such as ``problem.model``.
    parameter_distribution : Distribution
        $\mu$: a law of the model's parameters, such as the prior, a posterior, or
        ``EmpiricalDistribution(samples)``.
    learning_rate : float or array_like
        $\eta$, one rate or one per observation, such as ``problem.learning_rate["nee"]``. The
        default 1 gives the information of the ordinary posterior.
    num_samples : int, optional
        How many samples to draw from a $\mu$ not held as samples.
    rank, discarded_information
        As for :meth:`Subspace.from_information`.

    Returns
    -------
    Subspace
        With ``estimator="gradient"``.

    Examples
    --------
    >>> eta = toy.learning_rate["nee"]                                        # doctest: +SKIP
    >>> gradient_based_subspace(                                              # doctest: +SKIP
    ...     toy.model, toy.prior, learning_rate=eta, num_samples=500
    ... )
    """
    coordinates = Coordinates.from_model(model)
    to_theta = inverse(coordinates.whiten)

    def information_at(parameters: Any) -> Any:
        def mean_observations(z: Any) -> Any:
            return mean(condition_on(model, parameters.replace({"theta": to_theta(z)})))

        jacobian_at_z = jacobian(mean_observations, coordinates.whiten(parameters["theta"]))
        noise_variance = variance(condition_on(model, parameters)) / learning_rate
        return information_matrix(jacobian_at_z, noise_variance)

    information = expectation(parameter_distribution, information_at, num_samples=num_samples)
    return Subspace.from_information(
        np.asarray(information),
        parameter_names=coordinates.names,
        estimator="gradient",
        rank=rank,
        discarded_information=discarded_information,
    )


def finite_difference_subspace(
    model: Distribution,
    parameter_distribution: Distribution,
    *,
    learning_rate: Any = 1.0,
    num_samples: int | None = None,
    step: float = 0.2,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The active subspace of the information matrix averaged over a distribution of the
    parameters, with Jacobians by central differences.

    As :func:`gradient_based_subspace`, but column $j$ of each Jacobian is the central difference

    $$
    DG(z)\, e_j \approx \frac{G(z + h e_j) - G(z - h e_j)}{2h},
    $$

    with $e_j$ the $j$-th unit vector in whitened coordinates and $h$ the *step*, in prior
    standard deviations. It needs no gradients, at a cost of $2d$ forward runs per sample.

    SIPNET writes its output to three decimals, which puts a floor under the differences' noise:
    in these units it is about $0.002 / h$, so $h$ should be at least $0.1$.

    Parameters
    ----------
    model, parameter_distribution, learning_rate, num_samples, rank, discarded_information
        As for :func:`gradient_based_subspace`.
    step : float
        $h$.

    Returns
    -------
    Subspace
        With ``estimator="finite_differences"``.

    Raises
    ------
    ValueError
        If the forward map fails near any sample.
    """
    coordinates = Coordinates.from_model(model)
    to_theta = inverse(coordinates.whiten)

    def information_at(parameters: Any) -> Any:
        def mean_observations(z: Any) -> Any:
            return mean(condition_on(model, parameters.replace({"theta": to_theta(z)})))

        jacobian_at_z = jacobian(
            mean_observations,
            coordinates.whiten(parameters["theta"]),
            method="central_differences",
            step=step,
        )
        noise_variance = variance(condition_on(model, parameters)) / learning_rate
        return information_matrix(jacobian_at_z, noise_variance)

    information = expectation(parameter_distribution, information_at, num_samples=num_samples)
    if not np.all(np.isfinite(information)):
        raise ValueError("the forward map failed near at least one sample")
    return Subspace.from_information(
        np.asarray(information),
        parameter_names=coordinates.names,
        estimator="finite_differences",
        rank=rank,
        discarded_information=discarded_information,
    )


def linearization_subspace(
    model: Distribution,
    parameter_distribution: Distribution,
    *,
    learning_rate: Any = 1.0,
    num_samples: int | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The active subspace of the information from a statistical linearization over a
    distribution of the parameters, without gradients.

    It takes the joint law, under $\mu$, of the whitened coordinates $z$, the forward map's
    predictions $G(\theta)$ and the noise precision $\Gamma^{-1}$, fits the slope
    $\widehat{DG} = \operatorname{Cov}(G, z)\operatorname{Cov}(z)^{-1}$
    (:func:`statistical_linearization`), and returns the subspace of the leading eigenvectors of

    $$
    \hat H = \widehat{DG}^\top\, \bar\Gamma^{-1}\, \widehat{DG},
    \qquad \bar\Gamma^{-1} = \mathbb E_\mu\!\left[\Gamma^{-1}\right].
    $$

    For a Gaussian $\mu$, $\widehat{DG}$ estimates the average Jacobian $\mathbb E_\mu[DG]$, so
    $\hat H$ estimates $\mathbb E_\mu[DG]^\top \Gamma^{-1} \mathbb E_\mu[DG]$, which is smaller
    than $H_\eta(\mu)$: its eigenvalues are biased low, while its leading directions are usually
    close to those of $H_\eta(\mu)$. Each sample costs one forward run, and the samples are
    weighted by their weights.

    Parameters
    ----------
    model, parameter_distribution, learning_rate, num_samples, rank, discarded_information
        As for :func:`gradient_based_subspace`.

    Returns
    -------
    Subspace
        With ``estimator="statistical_linearization"``.

    Raises
    ------
    ValueError
        If the forward map fails at any sample.
    """
    coordinates = Coordinates.from_model(model)

    def linearization_inputs(parameters: Any) -> dict[str, Any]:
        observations = condition_on(model, parameters)
        return {
            "z": coordinates.whiten(parameters["theta"]),
            "predictions": mean(observations),
            "noise_precision": learning_rate / variance(observations),
        }

    joint = pushforward(parameter_distribution, linearization_inputs, num_samples=num_samples)
    slope = statistical_linearization(joint, inputs="z", outputs="predictions")
    information = information_matrix(slope, 1 / mean(joint["noise_precision"]))
    if not np.all(np.isfinite(information)):
        raise ValueError("the forward map failed at at least one sample")
    return Subspace.from_information(
        np.asarray(information),
        parameter_names=coordinates.names,
        estimator="statistical_linearization",
        rank=rank,
        discarded_information=discarded_information,
    )


def eki_linearization_subspace(
    model: Distribution,
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

    with $\Gamma_t$ the noise covariance step $t$ used, which includes the learning rate (at the
    ensemble's common noise scales, when they are inferred), and weights set by *weighting*:

    - ``"increment"`` (the default): $w_t = \Delta\beta_t$, the step's increment. The increments
      sum to 1, so $\hat H$ is a Riemann sum over the tempering levels of the linearized
      information, $\int_0^1 \mathbb E_{\pi_\beta}[DG]^\top \Gamma^{-1}
      \mathbb E_{\pi_\beta}[DG]\, d\beta$;
    - ``"uniform"``: $w_t = 1/T$ for $T$ steps;
    - ``"last"``: only the last step, the ensemble closest to the posterior.

    It runs no forward model evaluations, and it does not change the EKI run, which it only reads.

    Parameters
    ----------
    model : Distribution
        The model the EKI run conditioned, such as ``problem.model``; it gives the coordinates.
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
    >>> eki_linearization_subspace(toy.model, eki_posterior)                 # doctest: +SKIP
    """
    history = eki_history(eki_posterior)
    coordinates = Coordinates.from_model(model)
    particles = NumericRecordBatch.from_vector(
        coordinates.spec, history["theta"].values, level_names=["step", "member"]
    )
    z_at_steps = np.asarray(coordinates.whiten(particles).to_vector())
    ensembles = [
        EmpiricalDistribution(
            NumericRecordBatch({"z": z, "predictions": predictions}, level_names="member")
        )
        for z, predictions in zip(z_at_steps, history["predictions"].values, strict=True)
    ]
    weights = _step_weights(history["increment"].values, weighting)
    information = sum(
        weight * information_matrix(statistical_linearization(ensemble), noise_variance)
        for weight, ensemble, noise_variance in zip(
            weights, ensembles, history["noise_variance"].values, strict=True
        )
    )
    return Subspace.from_information(
        np.asarray(information),
        parameter_names=coordinates.names,
        estimator=f"eki_linearization_{weighting}",
        rank=rank,
        discarded_information=discarded_information,
    )


def covariance_subspace(
    model: Distribution,
    parameter_distribution: Distribution,
    *,
    num_samples: int | None = None,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
) -> Subspace:
    r"""The active subspace of the precision a posterior adds to the prior, from its samples.

    For a posterior $\mu$ with whitened covariance $\Sigma$, estimated from its (weighted) samples,
    this returns the subspace of the leading eigenvectors of $\Sigma^{-1} - I$. The prior's
    whitened covariance is $I$, so $\Sigma^{-1} - I$ is the precision the data add; for a linear
    forward map and a Gaussian prior it equals $DG^\top \Gamma^{-1} DG$ exactly. It runs no
    forward model evaluations, and needs no learning rate: the one that shaped the posterior is
    already in its samples. It needs more samples than there are parameters.

    Parameters
    ----------
    model : Distribution
        The model, such as ``problem.model``; it gives the coordinates.
    parameter_distribution : Distribution
        $\mu$: a posterior, or ``EmpiricalDistribution(samples)`` of one.
    num_samples, rank, discarded_information
        As for :func:`gradient_based_subspace`.

    Returns
    -------
    Subspace
        With ``estimator="covariance"``.

    Raises
    ------
    ValueError
        If there are no more samples than parameters.
    """
    coordinates = Coordinates.from_model(model)
    z = pushforward(
        parameter_distribution,
        lambda parameters: coordinates.whiten(parameters["theta"]),
        num_samples=num_samples,
    )
    if z.num_atoms <= coordinates.dimension:
        raise ValueError(
            f"need more samples than parameters, got {z.num_atoms} for {coordinates.dimension}"
        )
    return Subspace.from_information(
        np.linalg.inv(cov(z)) - np.eye(coordinates.dimension),
        parameter_names=coordinates.names,
        estimator="covariance",
        rank=rank,
        discarded_information=discarded_information,
    )


def statistical_linearization(
    joint: Distribution, *, inputs: str = "z", outputs: str = "predictions"
) -> np.ndarray:
    r"""The least-squares slope of one part of a joint law on another.

    For the outputs $G$ and the inputs $z$ of the joint law,

    $$
    \widehat{DG} = \operatorname{Cov}(G, z)\, \operatorname{Cov}(z)^{+},
    $$

    the slope of the best affine fit $G \approx c + \widehat{DG}\, z$, with $^{+}$ the
    pseudo-inverse, which gives the minimum-norm slope when $\operatorname{Cov}(z)$ is singular
    (as it is for an ensemble with no more members than parameters). Taking the joint law keeps
    each $G$ paired with the $z$ it came from, and a law held as weighted samples is fitted with
    its weights. To estimate the active subspace, $z$ must be whitened coordinates.

    Parameters
    ----------
    joint : Distribution
        A law with the outputs *inputs* and *outputs*, such as the pushforward of a distribution of
        the parameters to the whitened coordinates and the predictions.
    inputs, outputs : str
        The joint's outputs $z$ and $G$.

    Returns
    -------
    numpy.ndarray
        Shape $(n, d)$, for $G$ of size $n$ and $z$ of size $d$.
    """
    return np.asarray(cov(joint[outputs], joint[inputs]) @ jnp.linalg.pinv(cov(joint[inputs])))


def information_matrix(jacobian: Any, noise_variance: Any) -> jnp.ndarray:
    r"""The information matrix $J^\top \Gamma^{-1} J$ of one Jacobian.

    For a Jacobian $J$ ($n \times d$) and a noise covariance $\Gamma = \operatorname{diag}(\gamma)$,
    a $d \times d$ matrix: the curvature of the data misfit
    $\tfrac12 (y - G)^\top \Gamma^{-1} (y - G)$ when $G$ is replaced by its linearization.

    Parameters
    ----------
    jacobian : array_like
        $J$, shape $(n, d)$: exact, by finite differences, or a statistical linearization.
    noise_variance : array_like
        $\gamma$, the diagonal of $\Gamma$, shape $(n,)$.
    """
    jacobian = jnp.asarray(jacobian)
    return jacobian.T @ (jacobian / jnp.asarray(noise_variance)[:, None])


def _step_weights(increments: np.ndarray, weighting: Weighting) -> np.ndarray:
    """The weight of each step of an EKI run, as :func:`eki_linearization_subspace` describes."""
    if weighting == "increment":
        return np.asarray(increments, dtype=float)
    if weighting == "uniform":
        return np.full(len(increments), 1.0 / len(increments))
    if weighting == "last":
        return np.eye(len(increments))[-1]
    raise ValueError(f"unknown weighting {weighting!r}")
