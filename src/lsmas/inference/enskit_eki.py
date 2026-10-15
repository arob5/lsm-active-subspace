r"""Tempered ensemble Kalman inversion (EKI) from EnsKit, as a ProbPipe inference method.

Registered under the name ``"enskit_eki"``; it runs only when a call names it::

    with workflow_run(seed=1):
        posterior = problem.condition(method="enskit_eki", method_options={"n_particles": 100})
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from enskit import kalman, linalg
from enskit.algorithms import eki
from enskit.distribution import Ensemble
from probpipe import Distribution
from probpipe.inference import Feasibility, InferenceMethod

from lsmas._stopgaps.coordinates import ParameterBijection, block_values
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas._stopgaps.probpipe_private import (
    UnnormalizedConditional,
    make_posterior,
    model_factors,
    run_seed,
)
from lsmas._stopgaps.tempering import TemperedLikelihood
from lsmas.inference.step_record import StepCallback, StepRecord, history_dataset

__all__ = ["EnsKitEKI"]

#: The prior component that holds the calibration parameters.
_PARAMETER_BLOCK = "theta"


class EnsKitEKI(InferenceMethod):
    r"""Tempered ensemble Kalman inversion with perturbed observations.

    The target is the (generalized) posterior of parameters $\theta$ under a Gaussian likelihood,

    $$
    \pi(\theta \mid y) \propto \pi(\theta)\, \mathcal N\!\left(y \mid G(\theta), \Gamma\right),
    \qquad \Gamma = \operatorname{diag}\!\left(\frac{s^2_{k(i)} v_i}{\eta_i}\right),
    $$

    where the noise scales $s^2_k$ are fixed and $\eta_i$ are the learning rates (all 1 for the
    ordinary posterior). Particles $z_j$, $j = 1, \dots, J$, live in unconstrained coordinates
    $z = b^{-1}(\theta)$ and start as prior draws. The method climbs a ladder
    $0 = \beta_0 < \beta_1 < \dots < \beta_T = 1$: going from $\beta_t$ to
    $\beta_{t+1} = \beta_t + \Delta\beta_t$ conditions on $y$ with noise
    $\Gamma / \Delta\beta_t$, so the steps' likelihoods multiply to
    $\mathcal N(y \mid G, \Gamma)$. Each step moves every particle by the ensemble Kalman update

    $$
    z_j \leftarrow z_j + C_{zG} \left(C_{GG} + \Gamma / \Delta\beta_t\right)^{-1}
    \left(y + \varepsilon_j - G(z_j)\right),
    \qquad \varepsilon_j \sim \mathcal N(0, \Gamma / \Delta\beta_t),
    $$

    with $C_{zG}$ and $C_{GG}$ the ensemble covariances. $\Delta\beta_t$ is the largest increment
    that keeps the effective sample size of the weights $\exp(-\Delta\beta_t \Phi_j)$, with
    $\Phi_j = \tfrac12 \lVert \Gamma^{-1/2}(y - G(z_j)) \rVert^2$, above a fraction of $J$. For a
    linear $G$ and a Gaussian prior on $z$ the ensemble's law tends to the posterior as
    $J \to \infty$; otherwise the result is an approximation.

    A particle at which the forward map fails is moved to the centre of the others before the
    update. ``method_options``:

    - ``n_particles`` (int, default 100): $J$.
    - ``ess_fraction`` (float, default 0.5): the fraction of $J$ the effective sample size keeps.
    - ``callbacks`` (sequence of callables, default none): each is called with a
      :class:`~lsmas.inference.StepRecord` after every evaluation of the ensemble.
    - ``max_steps`` (int, default 1000): the most steps to take; at least
      1000, because an increment can be as small as 0.001.

    The result is the empirical law of the final particles. Its annotations hold the run's history
    as the group ``arviz/eki_history`` (see :func:`~lsmas.inference.step_record.history_dataset`).

    Applies to a model ``likelihood * prior`` whose likelihood is a Gaussian simulator likelihood
    with fixed noise scales, possibly tempered by a learning rate, and whose prior is over the
    parameter record ``theta`` alone.
    """

    _method_options = ("n_particles", "ess_fraction", "callbacks", "max_steps")

    @property
    def name(self) -> str:
        return "enskit_eki"

    def supported_types(self) -> tuple[type, ...]:
        return (UnnormalizedConditional,)

    def check(self, target: Any, /, **options: Any) -> Feasibility:
        reason = _infeasibility_reason(target)
        return Feasibility(reason is None, reason or "")

    def execute(self, target: Any, /, **options: Any) -> Distribution:
        self._check_options(options)
        problem = _GaussianInverseProblem.from_target(target)
        key_prior, key_state = jax.random.split(run_seed(self.name))
        particles = problem.prior_draws(key_prior, options.get("n_particles", 100))
        initial = eki.EKIState(Ensemble(u=jnp.asarray(particles)), key=key_state)
        final, records = _run(problem, initial, options)
        theta = problem.bijection.to_constrained(np.asarray(final.ensemble["u"]))
        return make_posterior(
            [jnp.asarray(theta)],
            parents=(target,),
            method=self.name,
            event_spec=target.event_spec,
            annotations=xr.DataTree.from_dict({"eki_history": history_dataset(records)}),
            n_particles=theta.shape[0],
            n_steps=len(records),
        )


@dataclass(frozen=True, eq=False)
class _GaussianInverseProblem:
    """What EKI reads from a conditioning target: forward map, data, noise and prior."""

    likelihood: GaussianSimulatorLikelihood
    learning_rate: np.ndarray
    observations: np.ndarray
    prior: Any
    bijection: ParameterBijection

    @classmethod
    def from_target(cls, target: Any) -> _GaussianInverseProblem:
        factors = model_factors(target)
        likelihood, learning_rate = _untempered(factors.likelihood)
        return cls(
            likelihood=likelihood,
            learning_rate=np.broadcast_to(learning_rate, likelihood.reference_variance.shape),
            observations=np.asarray(factors.observed, dtype=float),
            prior=factors.prior,
            bijection=ParameterBijection.from_prior(factors.prior, _PARAMETER_BLOCK),
        )

    @property
    def noise_variance(self) -> np.ndarray:
        r"""The diagonal of $\Gamma$: each observation's noise variance over its learning rate."""
        return self.likelihood.noise_variances() / self.learning_rate

    def prior_draws(self, key: Any, n_particles: int) -> np.ndarray:
        """*n_particles* prior draws in unconstrained coordinates, shape $(J, d)$."""
        draws = self.prior._sample(key, (int(n_particles),))
        theta = block_values(draws, _PARAMETER_BLOCK, self.bijection.names)
        return self.bijection.to_unconstrained(theta)

    def forward_in_unconstrained_coordinates(self, z: Any) -> np.ndarray:
        """$G$ at each row of unconstrained coordinates *z*, shape $(J, n)$; EnsKit's simulator."""
        theta = self.bijection.to_constrained(np.asarray(z))
        return np.asarray(self.likelihood.forward(self.bijection.record_batch(theta)).raw())


def _run(
    problem: _GaussianInverseProblem, initial: Any, options: dict[str, Any]
) -> tuple[Any, list[StepRecord]]:
    """Climb the tempering ladder from *initial*; return the final state and every step's record."""
    callbacks: Sequence[StepCallback] = options.get("callbacks", ())
    noise = linalg.PSDDiagonal(jnp.asarray(problem.noise_variance))
    steps = eki.iterate(
        initial,
        problem.forward_in_unconstrained_coordinates,
        jnp.asarray(problem.observations),
        noise,
        update_rule=kalman.Matheron(),
        schedule=eki.AdaptiveESSSchedule(ess_fraction=float(options.get("ess_fraction", 0.5))),
        on_failure="repair",
        max_steps=int(options.get("max_steps", 1000)),
    )
    state, records = initial, []
    for state, history_entry, evaluation in steps:  # noqa: B007  (the last state is the result)
        record = _step_record(problem, history_entry, evaluation)
        for callback in callbacks:
            callback(record)
        records.append(record)
    return state, records


def _step_record(
    problem: _GaussianInverseProblem, history_entry: Any, evaluation: Any
) -> StepRecord:
    """The :class:`StepRecord` of one evaluation of the ensemble."""
    unconstrained = np.asarray(evaluation.ensemble["u"])
    return StepRecord(
        step=int(history_entry.step),
        beta=float(history_entry.beta),
        increment=float(history_entry.increment),
        parameter_names=problem.bijection.names,
        theta=problem.bijection.to_constrained(unconstrained),
        predictions=np.asarray(evaluation.ensemble[eki.PREDICTION]),
        noise_variance=problem.noise_variance,
    )


def _untempered(likelihood: Any) -> tuple[Any, np.ndarray]:
    """The likelihood beneath any tempering, and the learning rate (1 when untempered)."""
    if isinstance(likelihood, TemperedLikelihood):
        return likelihood.likelihood, likelihood.learning_rate
    return likelihood, np.asarray(1.0)


def _infeasibility_reason(target: Any) -> str | None:
    """Why EKI cannot condition *target*, or None if it can."""
    factors = model_factors(target)
    if factors is None:
        return "the model must be a product likelihood * prior, conditioned on its observations"
    likelihood, _ = _untempered(factors.likelihood)
    if not isinstance(likelihood, GaussianSimulatorLikelihood):
        return "the likelihood must be a Gaussian simulator likelihood"
    if likelihood.random_scale_slots:
        return (
            "the noise scales must be fixed; inferring them "
            f"({', '.join(likelihood.random_scale_slots)}) is not supported yet"
        )
    if set(factors.prior.event_spec.components) != {_PARAMETER_BLOCK}:
        return f"the prior must be over the parameter record {_PARAMETER_BLOCK!r} alone"
    return None
