r"""Active-subspace MCMC: explore with EKI, estimate the subspace, sample the reduced posterior.

The algorithm, for a problem with $d$ calibration parameters:

1. **Explore.** Tempered EKI with $J$ particles conditions the problem (inferring the noise scales
   when the problem does), and a :class:`~lsmas.tools.LadderLinearization` callback accumulates
   $\hat H = \sum_t \Delta\beta_t\, \widehat{DG}_t^\top \Gamma_t^{-1} \widehat{DG}_t$.
2. **Estimate.** The subspace is spanned by the leading $r$ eigenvectors of $\hat H$, with $r$ the
   smallest rank whose discarded eigenvalues sum to at most $\varepsilon$ unless $r$ is given.
3. **Reduce.** The plug-in reduced problem in the active variables $a = W^\top \xi$
   (:func:`~lsmas.tools.reduced_problem`).
4. **Sample.** Random-walk Metropolis (ProbPipe's ``blackjax_rwmh``) on the reduced problem, its
   chains started at the projection $W^\top \bar\xi$ of EKI's final ensemble mean (and at the
   ensemble's median noise scales), with ProbPipe's adaptive warmup.
5. **Lift.** Each draw of $a$ is combined with the inactive part of an independent prior draw,
   $\xi = W a + W_\perp b$ (:meth:`~lsmas.tools.ReducedProblem.lift`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from probpipe import Distribution, workflow_run

from lsmas.algorithms.result import AlgorithmResult
from lsmas.inference import StepRecord
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools import LadderLinearization, ReducedProblem, Subspace, reduced_problem
from lsmas.tools.estimators import Weighting
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
        How the ladder's steps are weighted in $\hat H$
        (see :class:`~lsmas.tools.LadderLinearization`).
    sampler_options : Mapping, optional
        Options of ``blackjax_rwmh`` that replace :data:`DEFAULT_SAMPLER_OPTIONS` and the starting
        point, such as ``{"num_results": 2000}``.
    seed : int
        Seeds EKI, the sampler and the lift, each with its own stream.

    Returns
    -------
    AlgorithmResult
        The lifted posterior, the subspace, the forward runs of each stage, and the EKI run and
        reduced posterior they came from.

    Examples
    --------
    >>> result = active_subspace_mcmc(load("toy"), seed=1)        # doctest: +SKIP
    >>> result.subspace.rank, result.forward_runs                  # doctest: +SKIP
    (7, {'exploration': 600, 'sampling': 6004})
    """
    exploration, records, ladder = _explore(problem, n_particles, weighting, seed)
    subspace = ladder.subspace(rank, discarded_information)
    reduced = reduced_problem(problem, subspace)
    options = (
        DEFAULT_SAMPLER_OPTIONS
        | {"init": _starting_point(reduced, problem, exploration)}
        | dict(sampler_options or {})
    )
    with workflow_run(seed=seed + 1):
        reduced_posterior = reduced.condition(method="blackjax_rwmh", method_options=options)
    return AlgorithmResult(
        posterior=reduced.lift(reduced_posterior, key=jax.random.key(seed + 2)),
        subspace=subspace,
        forward_runs={
            "exploration": _exploration_runs(problem, records),
            "sampling": _sampling_runs(options),
        },
        exploration=exploration,
        reduced_posterior=reduced_posterior,
    )


def _explore(
    problem: InverseProblem, n_particles: int, weighting: Weighting, seed: int
) -> tuple[Distribution, list[StepRecord], LadderLinearization]:
    """EKI on *problem*, with the ladder accumulator and a record of every step."""
    ladder = LadderLinearization(problem.coordinates, weighting=weighting)
    records: list[StepRecord] = []
    with workflow_run(seed=seed):
        exploration = problem.condition(
            method="enskit_eki",
            method_options={"n_particles": n_particles, "callbacks": [ladder, records.append]},
        )
    return exploration, records, ladder


def _starting_point(
    reduced: ReducedProblem, problem: InverseProblem, exploration: Distribution
) -> np.ndarray:
    r"""The chains' start: $W^\top \bar\xi$ for EKI's final ensemble, then each inferred noise
    scale's median over the ensemble, in the order of the reduced prior's event."""
    theta = problem.coordinates.array_from_draws(exploration.atoms).reshape(
        -1, problem.coordinates.dimension
    )
    xi = problem.coordinates.whitened_from_parameters(theta)
    subspace: Subspace = reduced.subspace
    active = subspace.active_variables(xi.mean(axis=0))
    scales = [
        np.median(np.asarray(exploration.atoms[slot].raw))
        for slot in problem.likelihood.random_scale_slots
    ]
    return np.concatenate([active, scales])


def _exploration_runs(problem: InverseProblem, records: list[StepRecord]) -> int:
    """Forward runs of EKI: one ensemble per step, and a final one when scales are inferred."""
    n_particles = records[0].theta.shape[0]
    final_evaluation = 1 if problem.likelihood.random_scale_slots else 0
    return n_particles * (len(records) + final_evaluation)


def _sampling_runs(options: Mapping[str, Any]) -> int:
    """Forward runs of random-walk Metropolis: one per step of each chain, and one at its start."""
    return int(options["num_chains"]) * (
        int(options["num_warmup"]) + int(options["num_results"]) + 1
    )
