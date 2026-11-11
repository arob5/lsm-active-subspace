r"""Scores of an approximate posterior and an estimated subspace against a reference.

The posterior metrics compare draws of an approximation $q$ with the reference $p$ coordinate by
coordinate, in whitened coordinates $\xi_j$ (and $\log s^2_k$ for inferred noise scales), each
divided by the reference's standard deviation $\sigma_j$ of that coordinate:

- the **marginal 1-Wasserstein distance** $W_1(q_j, p_j) / \sigma_j$, which is sensitive to both
  location and spread; its mean and maximum over coordinates are the headline scores;
- the **mean error** $\lvert \mathbb E_q \xi_j - \mathbb E_p \xi_j \rvert / \sigma_j$;
- the **sd error** $\lvert \log(\mathrm{sd}_q\, \xi_j / \sigma_j) \rvert$;
- the **Gaussian 2-Wasserstein distance** between Gaussian fits of $q$ and $p$, which is sensitive
  to correlations;
- the **coverage** of $p$'s draws by $q$'s central 90% interval, a diagnostic only: it cannot
  exceed 1, so it does not penalize an approximation that is too wide.

The subspace metrics compare a subspace with basis $W$ of rank $r$ with the reference information
matrix $H_{\mathrm{ref}}$:

- the **captured information** $\operatorname{tr}(W^\top H_{\mathrm{ref}} W) / \sum_{i \le r}
  \lambda_i(H_{\mathrm{ref}})$, 1 when $W$ spans the reference's top $r$ eigenvectors;
- the **largest principal angle** between the span of $W$ and that of the reference's top $r$
  eigenvectors, in degrees.

Two independent halves of a reference score about 0.04 in mean marginal $W_1$, which is the floor
below which differences are Monte Carlo noise.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats

from lsmas._stopgaps.likelihood import noise_scale_variable
from lsmas.algorithms.result import AlgorithmResult
from lsmas.experiments.reference import Reference
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools.subspace import Subspace

__all__ = [
    "captured_information",
    "gaussian_wasserstein",
    "interval_coverage",
    "largest_principal_angle",
    "marginal_wasserstein",
    "mean_errors",
    "posterior_scores",
    "score",
    "score_names",
    "sd_errors",
    "subspace_scores",
]


def score_names(problem: InverseProblem) -> list[str]:
    """The names of the scores :func:`score` gives for *problem*, in order."""
    posterior = [
        f"{metric}_{summary}"
        for metric, summaries in (
            ("marginal_w1", ("mean", "max")),
            ("mean_error", ("mean", "max")),
            ("sd_error", ("mean", "max")),
        )
        for summary in summaries
    ] + ["gaussian_w2", "coverage90_mean", "coverage90_min"]
    scales = [
        f"marginal_w1_{noise_scale_variable(noise_class)}"
        for noise_class in problem.likelihood.inferred_scale_classes
    ]
    subspace = ["subspace_rank", "captured_information", "largest_principal_angle_degrees"]
    return posterior + scales + subspace + ["forward_runs"]


def score(
    result: AlgorithmResult, reference: Reference, problem: InverseProblem
) -> dict[str, float]:
    """Every score of an algorithm's result against a reference, as one flat row.

    Parameters
    ----------
    result : AlgorithmResult
        The algorithm's result.
    reference : Reference
        The reference posterior of the same problem and noise-scale setting.
    problem : InverseProblem
        The problem, whose coordinates whiten the draws.
    """
    theta = problem.coordinates.array_from_draws(result.posterior.atoms).reshape(
        -1, problem.coordinates.dimension
    )
    noise_scales = {
        noise_class: draws.reshape(-1)
        for noise_class, draws in problem.likelihood.inferred_scale_draws(
            result.posterior.atoms
        ).items()
    }
    return (
        posterior_scores(theta, noise_scales, reference, problem)
        | subspace_scores(result.subspace, reference)
        | {"forward_runs": result.total_forward_runs}
    )


def posterior_scores(
    theta: np.ndarray,
    noise_scales: dict[str, np.ndarray],
    reference: Reference,
    problem: InverseProblem,
) -> dict[str, float]:
    r"""The posterior metrics of draws against a reference, summarized over coordinates.

    Parameters
    ----------
    theta : numpy.ndarray
        Draws of the calibration parameters, shape $(N, d)$.
    noise_scales : dict of str to numpy.ndarray
        Draws of each inferred noise scale, shape $(N,)$, by class; empty when the scales are fixed.
    reference : Reference
        The reference.
    problem : InverseProblem
        The problem, whose coordinates whiten the draws.

    Returns
    -------
    dict of str to float
        ``marginal_w1_mean``, ``marginal_w1_max``, ``mean_error_mean``, ``mean_error_max``,
        ``sd_error_mean``, ``sd_error_max``, ``gaussian_w2``, ``coverage90_mean`` and
        ``coverage90_min`` over the whitened parameters, and ``marginal_w1_s2_<class>`` for each
        inferred noise scale, on the logarithm of the scale.
    """
    approximation = problem.coordinates.whitened_from_parameters(theta)
    target = problem.coordinates.whitened_from_parameters(reference.parameter_draws())
    distances = marginal_wasserstein(approximation, target)
    location, spread = mean_errors(approximation, target), sd_errors(approximation, target)
    coverage = interval_coverage(approximation, target)
    scores = {
        "marginal_w1_mean": distances.mean(),
        "marginal_w1_max": distances.max(),
        "mean_error_mean": location.mean(),
        "mean_error_max": location.max(),
        "sd_error_mean": spread.mean(),
        "sd_error_max": spread.max(),
        "gaussian_w2": gaussian_wasserstein(approximation, target),
        "coverage90_mean": coverage.mean(),
        "coverage90_min": coverage.min(),
    }
    reference_scales = reference.noise_scale_draws()
    for noise_class, draws in noise_scales.items():
        log_draws = np.log(np.asarray(draws, dtype=float))[:, None]
        log_reference = np.log(reference_scales[noise_class])[:, None]
        name = f"marginal_w1_{noise_scale_variable(noise_class)}"
        scores[name] = marginal_wasserstein(log_draws, log_reference)[0]
    return {name: float(value) for name, value in scores.items()}


def subspace_scores(subspace: Subspace, reference: Reference) -> dict[str, float]:
    """The subspace metrics of a subspace against a reference's information matrix.

    Returns
    -------
    dict of str to float
        ``subspace_rank``, ``captured_information`` and ``largest_principal_angle_degrees``.
    """
    return {
        "subspace_rank": float(subspace.rank),
        "captured_information": captured_information(subspace.basis, reference.information),
        "largest_principal_angle_degrees": largest_principal_angle(
            subspace.basis, reference.information
        ),
    }


def marginal_wasserstein(approximation: np.ndarray, target: np.ndarray) -> np.ndarray:
    r"""$W_1(q_j, p_j) / \sigma_j$ for each coordinate $j$, shape $(d,)$.

    Parameters
    ----------
    approximation, target : numpy.ndarray
        Draws of $q$ and of $p$, shapes $(N_q, d)$ and $(N_p, d)$.
    """
    return np.array(
        [
            stats.wasserstein_distance(approximation[:, j], target[:, j])
            for j in range(target.shape[1])
        ]
    ) / target.std(axis=0)


def mean_errors(approximation: np.ndarray, target: np.ndarray) -> np.ndarray:
    r"""$\lvert \mathbb E_q \xi_j - \mathbb E_p \xi_j \rvert / \sigma_j$ for each coordinate."""
    return np.abs(approximation.mean(axis=0) - target.mean(axis=0)) / target.std(axis=0)


def sd_errors(approximation: np.ndarray, target: np.ndarray) -> np.ndarray:
    r"""$\lvert \log(\mathrm{sd}_q\, \xi_j / \sigma_j) \rvert$ for each coordinate."""
    return np.abs(np.log(approximation.std(axis=0) / target.std(axis=0)))


def interval_coverage(
    approximation: np.ndarray, target: np.ndarray, probability: float = 0.9
) -> np.ndarray:
    """The share of the target's draws inside the approximation's central interval, per
    coordinate; *probability* is ideal."""
    tail = (1 - probability) / 2
    low, high = np.quantile(approximation, [tail, 1 - tail], axis=0)
    return np.mean((target >= low) & (target <= high), axis=0)


def gaussian_wasserstein(approximation: np.ndarray, target: np.ndarray) -> float:
    r"""The 2-Wasserstein distance between the Gaussian fits of two sets of draws,

    $$
    W_2^2 = \lVert m_q - m_p \rVert^2
    + \operatorname{tr}\!\left(\Sigma_q + \Sigma_p
    - 2\left(\Sigma_p^{1/2} \Sigma_q \Sigma_p^{1/2}\right)^{1/2}\right).
    $$
    """
    mean_gap = approximation.mean(axis=0) - target.mean(axis=0)
    approximation_covariance = np.atleast_2d(np.cov(approximation, rowvar=False))
    target_covariance = np.atleast_2d(np.cov(target, rowvar=False))
    target_root = linalg.sqrtm(target_covariance)
    cross = linalg.sqrtm(target_root @ approximation_covariance @ target_root).real
    squared = mean_gap @ mean_gap + np.trace(
        approximation_covariance + target_covariance - 2 * cross
    )
    return float(np.sqrt(max(squared, 0.0)))


def captured_information(basis: np.ndarray, information: np.ndarray) -> float:
    r"""The captured information $\operatorname{tr}(W^\top H W) / \sum_{i \le r} \lambda_i(H)$ of a
    basis $W$ of rank $r$."""
    rank = basis.shape[1]
    best = np.sort(np.linalg.eigvalsh(information))[::-1][:rank].sum()
    return float(np.trace(basis.T @ information @ basis) / best)


def largest_principal_angle(basis: np.ndarray, information: np.ndarray) -> float:
    """The largest principal angle, in degrees, between the span of *basis* and that of the top
    eigenvectors of *information*, as many as *basis* has columns."""
    rank = basis.shape[1]
    top = np.linalg.eigh(information)[1][:, ::-1][:, :rank]
    cosines = np.linalg.svd(basis.T @ top, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cosines.min(), -1.0, 1.0))))
