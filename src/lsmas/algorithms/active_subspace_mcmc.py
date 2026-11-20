r"""Active-subspace MCMC: explore with EKI, estimate the subspace, sample the reduced posterior.

The algorithm, for a problem with $d$ calibration parameters:

1. **Explore.** Tempered EKI with $J$ particles conditions the problem (inferring the noise scales
   when the problem does).
2. **Estimate.** From the run's history, $\hat H = \sum_t \Delta\beta_t\, \widehat{DG}_t^\top
   \Gamma_t^{-1} \widehat{DG}_t$ (:func:`~lsmas.tools.eki_linearization_subspace`); the subspace
   is spanned by its leading $r$ eigenvectors, with $r$ the smallest rank whose discarded
   eigenvalues sum to at most $\varepsilon$ unless $r$ is given.
3. **Reduce.** The plug-in reduced problem in the active variables $a = W^\top z$
   (:func:`~lsmas.tools.reduced_problem`).
4. **Sample.** Random-walk Metropolis (ProbPipe's ``blackjax_rwmh``) on the reduced problem, its
   chains started at the projection $W^\top \bar z$ of EKI's final ensemble mean (and at the
   ensemble's median noise scales), with ProbPipe's adaptive warmup.
5. **Lift.** Each draw of $a$ is combined with the inactive part of an independent prior draw,
   $z = W a + W_\perp b$ (:meth:`~lsmas.tools.ReducedProblem.lift`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from probpipe import EmpiricalDistribution, workflow_run

from lsmas.algorithms.result import AlgorithmResult
from lsmas.inference import eki_history
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools import ReducedProblem, Weighting, eki_linearization_subspace, reduced_problem
from lsmas.tools.subspace import DEFAULT_DISCARDED_INFORMATION

__all__ = ["DEFAULT_SAMPLER_OPTIONS", "active_subspace_mcmc"]

#: The random-walk Metropolis run's defaults: 4 chains of 500 warmup steps and 1000 draws.
DEFAULT_SAMPLER_OPTIONS = {"num_chains": 4, "num_warmup": 500, "num_results": 1000}


def active_subspace_mcmc(
    problem: InverseProblem,
    *,
    n_particles: int = 100,
    rank: int | None = None,
    discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
    weighting: Weighting = "increment",
    sampler_options: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> AlgorithmResult:
    r"""Calibrate *problem* by active-subspace MCMC (see the module's description).

    Parameters
    ----------
    problem : InverseProblem
        The problem to calibrate.
    n_particles : int
        $J$, EKI's ensemble size.
    rank : int, optional
        $r$. By default the smallest rank whose discarded eigenvalues sum to at most
        *discarded_information*.
    discarded_information : float
        $\varepsilon$, the bound on the discarded eigenvalues when the rank is not given.
    weighting : {"increment", "uniform", "last"}
        How the EKI steps are weighted in $\hat H$
        (see :func:`~lsmas.tools.eki_linearization_subspace`).
    sampler_options : Mapping, optional
        Options of ``blackjax_rwmh`` that replace :data:`DEFAULT_SAMPLER_OPTIONS` and the starting
        point, such as ``{"num_results": 2000}``.
    seed : int
        The root seed. EKI, the sampler and the lift each get an independent seed spawned from it,
        so runs with different root seeds share no random stream.

    Returns
    -------
    AlgorithmResult
        The lifted posterior, the subspace, the forward runs of each stage, and the EKI run and
        reduced posterior they came from.

    Examples
    --------
    >>> result = active_subspace_mcmc(load_inverse_problem("toy"), seed=1)        # doctest: +SKIP
    >>> result.subspace.rank, result.forward_runs                  # doctest: +SKIP
    (7, {'exploration': 600, 'sampling': 6004})
    """
    eki_seed, rwmh_seed, lift_seed = _independent_stage_seeds(seed)
    with workflow_run(seed=eki_seed):
        eki_posterior = problem.condition(
            method="enskit_eki", method_options={"n_particles": n_particles}
        )
    subspace = eki_linearization_subspace(
        problem,
        eki_posterior,
        weighting=weighting,
        rank=rank,
        discarded_information=discarded_information,
    )
    reduced = reduced_problem(problem, subspace)
    rwmh_options = (
        DEFAULT_SAMPLER_OPTIONS
        | {"init": _rwmh_start_at_eki_mean(reduced, eki_posterior)}
        | dict(sampler_options or {})
    )
    with workflow_run(seed=rwmh_seed):
        reduced_posterior = reduced.condition(method="blackjax_rwmh", method_options=rwmh_options)
    with workflow_run(seed=lift_seed):
        posterior = reduced.lift(reduced_posterior)
    return AlgorithmResult(
        posterior=posterior,
        subspace=subspace,
        forward_runs={
            "exploration": _count_eki_forward_runs(problem, eki_posterior),
            "sampling": _count_rwmh_forward_runs(rwmh_options),
        },
        exploration_posterior=eki_posterior,
        reduced_posterior=reduced_posterior,
    )


def _independent_stage_seeds(seed: int) -> tuple[int, int, int]:
    """Independent seeds for EKI, random-walk Metropolis and the lift, spawned from one root."""
    eki_seed, rwmh_seed, lift_seed = (
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(3)
    )
    return eki_seed, rwmh_seed, lift_seed


def _rwmh_start_at_eki_mean(
    reduced: ReducedProblem, eki_posterior: EmpiricalDistribution
) -> np.ndarray:
    r"""The chains' initial state: $W^\top \bar z$ for the mean $\bar z$ of EKI's final
    ensemble, then each inferred noise scale's median over the ensemble, in the order of the
    reduced prior's event."""
    coordinates = reduced.full_problem.coordinates
    eki_z = np.asarray(coordinates.whiten(eki_posterior.atoms["theta"]).to_vector())
    eki_mean_z = eki_z.reshape(-1, coordinates.dimension).mean(axis=0)
    scale_medians = [
        np.median(draws)
        for draws in reduced.likelihood.inferred_scale_draws(eki_posterior.atoms).values()
    ]
    return np.concatenate([reduced.subspace.active_variables(eki_mean_z), scale_medians])


def _count_eki_forward_runs(problem: InverseProblem, eki_posterior: EmpiricalDistribution) -> int:
    """Forward runs of EKI: one ensemble per step, and a final one when scales are inferred."""
    n_steps, n_particles, _ = eki_history(eki_posterior)["theta"].shape
    final_evaluation = 1 if problem.likelihood.inferred_scale_classes else 0
    return n_particles * (n_steps + final_evaluation)


def _count_rwmh_forward_runs(rwmh_options: Mapping[str, Any]) -> int:
    """Forward runs of random-walk Metropolis: one per step of each chain, and one at its start."""
    steps_per_chain = int(rwmh_options["num_warmup"]) + int(rwmh_options["num_results"]) + 1
    return int(rwmh_options["num_chains"]) * steps_per_chain
