r"""Tempered ensemble Kalman inversion (EKI) from EnsKit, as a ProbPipe inference method.

Registered under the name ``"enskit_eki"``; it runs only when a call names it::

    with workflow_run(seed=1):
        posterior = problem.condition(method="enskit_eki", method_options={"n_particles": 100})
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from enskit import kalman, linalg
from enskit.algorithms import eki
from enskit.distribution import Ensemble
from probpipe import Distribution, InverseGamma, NumericRecordBatch
from probpipe.inference import Feasibility, InferenceMethod, InferenceTarget, make_posterior

from lsmas._stopgaps.coordinates import ParameterBijection, block_values
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas._stopgaps.sampling import draw_with_key
from lsmas._stopgaps.tempering import TemperedLikelihood
from lsmas.inference.noise_scales import NoiseScaleConditionals
from lsmas.inference.step_record import StepCallback, StepRecord, history_dataset
from lsmas.priors import factors_over

__all__ = ["EnsKitEKI"]

#: The prior's output that holds the calibration parameters.
_PARAMETER_BLOCK = "theta"


class EnsKitEKI(InferenceMethod):
    r"""Tempered ensemble Kalman inversion with perturbed observations, and Gibbs draws of any
    inferred noise scales.

    The target is the (generalized) posterior of the parameters $\theta$, and of any inferred class
    noise scales $s^2_k$, under a Gaussian likelihood:

    $$
    \pi(\theta, s^2 \mid y) \propto \pi(\theta) \prod_{k \text{ inferred}} \mathrm{IG}(s^2_k;
    \alpha_k, b_k)\; \mathcal N\!\left(y \mid G(\theta), \Gamma(s^2)\right),
    \qquad \Gamma(s^2) = \operatorname{diag}\!\left(\frac{s^2_{k(i)} v_i}{\eta_i}\right),
    $$

    with $\eta_i$ the learning rates (all 1 for the ordinary posterior). Particles $z_j$,
    $j = 1, \dots, J$, live in unconstrained coordinates $z = b^{-1}(\theta)$; they and their scales
    $s^2_j$ start as prior draws. The method moves through tempering levels
    $0 = \beta_0 < \beta_1 < \dots < \beta_T = 1$, the level $\beta$ targeting the tempered
    posterior with likelihood $\mathcal N(y \mid G, \Gamma / \beta)$. Each step evaluates $G$ at
    every particle (EnsKit's ``eki.evaluate``, the step's one batch of forward runs) and then:

    1. **Increment.** $\Delta\beta_t$ is the largest increment whose importance weights from
       $\beta_t$ to $\beta_t + \Delta\beta_t$, with the scales integrated out, keep an effective
       sample size of at least a fraction of $J$
       (:meth:`~lsmas.inference.noise_scales.NoiseScaleConditionals.largest_increment`). With
       every scale fixed the weights are $\exp(-\Delta\beta_t \Phi_j)$,
       $\Phi_j = \tfrac12 \lVert \Gamma^{-1/2}(y - G(z_j)) \rVert^2$.
    2. **Kalman update** (EnsKit's ``eki.assimilate``): every particle moves by

       $$
       z_j \leftarrow z_j + C_{zG} \left(C_{GG} + \bar\Gamma / \Delta\beta_t\right)^{-1}
       \left(y + \varepsilon_j - G(z_j)\right),
       \qquad \varepsilon_j \sim \mathcal N(0, \bar\Gamma / \Delta\beta_t),
       $$

       with $C_{zG}$ and $C_{GG}$ the ensemble covariances and $\bar\Gamma = \Gamma(\bar s^2)$ at
       the ensemble's common scales $\bar s^2_k = (\frac1J \sum_j s^{-2}_{jk})^{-1}$.
    3. **Gibbs draw of the scales**, after the moved particles are evaluated: each particle's
       inferred scales are drawn from their exact conditional law at the new level
       (:class:`~lsmas.inference.noise_scales.NoiseScaleConditionals`).

    For a linear $G$, a Gaussian prior on $z$ and fixed scales, the ensemble's law tends to the
    posterior as $J \to \infty$; otherwise the result is an approximation. A particle at which the
    forward map fails is moved to the centre of the valid particles before the update.

    ``method_options``:

    - ``n_particles`` (int, default 100): $J$.
    - ``ess_fraction`` (float, default 0.5): the fraction of $J$ the effective sample size keeps.
    - ``callbacks`` (sequence of callables, default none): each is called with a
      :class:`~lsmas.inference.StepRecord` after every step's increment is chosen.
    - ``max_steps`` (int, default 1000): the most steps to take before giving up; each increment is
      at least 0.001, so 1000 steps always reach $\beta = 1$.

    The result is the empirical law of the final particles, with their scales when inferred. Its
    annotations hold the run's history as the group ``arviz/eki_history`` (see
    :func:`~lsmas.inference.step_record.history_dataset`).

    Applies to a model ``likelihood & prior`` whose likelihood is a Gaussian simulator likelihood,
    possibly tempered by a learning rate, and whose prior is over the parameter record ``theta``
    and the inferred scales, each scale with an inverse-gamma prior.
    """

    _method_options = ("n_particles", "ess_fraction", "callbacks", "max_steps")

    @property
    def name(self) -> str:
        return "enskit_eki"

    @property
    def description(self) -> str:
        return "Tempered ensemble Kalman inversion (EnsKit) for Gaussian simulator likelihoods."

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    def check(self, target: InferenceTarget, /, **options: Any) -> Feasibility:
        reason = _infeasibility_reason(target)
        return Feasibility(reason is None, reason or "")

    def execute(self, target: InferenceTarget, /, *, key: Any, **options: Any) -> Distribution:
        self._check_options(options)
        problem = _GaussianInverseProblem.from_target(target)
        key_prior, key_kalman, key_scales = jax.random.split(key, 3)
        initial_particles = problem.draw_initial_particles(
            key_prior, options.get("n_particles", 100)
        )
        final_particles, step_records = _run_tempered_eki(
            problem,
            initial_particles,
            key_kalman=key_kalman,
            key_scales=key_scales,
            options=options,
        )
        return make_posterior(
            target,
            [_particles_as_target_vectors(target, problem, final_particles)],
            method=self.name,
            annotations=xr.DataTree.from_dict({"eki_history": history_dataset(step_records)}),
            n_particles=final_particles.z.shape[0],
            n_steps=len(step_records),
        )


@dataclass(frozen=True)
class _ParticlesWithScales:
    """An ensemble: unconstrained parameters $z$, shape $(J, d)$, and class scales, $(J, K)$."""

    z: np.ndarray
    scales: np.ndarray


@dataclass(frozen=True, eq=False)
class _GaussianInverseProblem:
    """What EKI reads from a conditioning target: forward map, data, noise model and prior."""

    likelihood: GaussianSimulatorLikelihood
    observations: np.ndarray
    prior: Any
    bijection: ParameterBijection
    scale_conditionals: NoiseScaleConditionals

    @classmethod
    def from_target(cls, target: InferenceTarget) -> _GaussianInverseProblem:
        factors = target.factors()
        likelihood, learning_rate = _likelihood_and_learning_rate(factors.likelihood)
        observations = np.asarray(factors.observed, dtype=float)
        return cls(
            likelihood=likelihood,
            observations=observations,
            prior=factors.prior,
            bijection=ParameterBijection.from_prior(factors.prior, _PARAMETER_BLOCK),
            scale_conditionals=NoiseScaleConditionals.from_problem(
                likelihood,
                _inverse_gamma_scale_priors(factors.prior, likelihood.random_scale_slots),
                observations,
                learning_rate,
            ),
        )

    def draw_initial_particles(self, key: Any, n_particles: int) -> _ParticlesWithScales:
        """*n_particles* prior draws: parameters in unconstrained coordinates, and class scales."""
        draws = draw_with_key(self.prior, key, (n_particles,))
        theta = block_values(draws, _PARAMETER_BLOCK, self.bijection.names)
        conditionals = self.scale_conditionals
        scales = np.stack(
            [
                np.full(n_particles, fixed) if slot is None else np.asarray(draws[slot], float)
                for slot, fixed in zip(conditionals.slots, conditionals.fixed_scales, strict=True)
            ],
            axis=-1,
        )
        return _ParticlesWithScales(z=self.bijection.to_unconstrained(theta), scales=scales)

    def forward_in_unconstrained_coordinates(self, z: Any) -> np.ndarray:
        """$G$ at each row of unconstrained coordinates *z*, shape $(J, n)$; EnsKit's simulator."""
        theta = self.bijection.to_constrained(np.asarray(z))
        return np.asarray(self.likelihood.forward(self.bijection.record_batch(theta)).raw)


def _run_tempered_eki(
    problem: _GaussianInverseProblem,
    initial_particles: _ParticlesWithScales,
    *,
    key_kalman: Any,
    key_scales: Any,
    options: Mapping[str, Any],
) -> tuple[_ParticlesWithScales, list[StepRecord]]:
    r"""Run tempered EKI from *initial_particles*.

    Returns the final particles and every step's record.

    Raises
    ------
    RuntimeError
        If the ladder does not reach $\beta = 1$ within ``max_steps`` steps.
    """
    callbacks: Sequence[StepCallback] = options.get("callbacks", ())
    ess_fraction = float(options.get("ess_fraction", 0.5))
    max_steps = int(options.get("max_steps", 1000))
    conditionals = problem.scale_conditionals
    state = eki.EKIState(Ensemble(u=jnp.asarray(initial_particles.z)), key=key_kalman)
    scales = initial_particles.scales
    evaluation = _evaluate_forward_map_at_particles(problem, state, scales)
    step_records: list[StepRecord] = []
    while float(state.beta) < 1.0:
        if len(step_records) == max_steps:
            raise RuntimeError(
                f"EKI reached only beta = {float(state.beta):.4g} in {max_steps} steps"
            )
        beta = float(state.beta)
        predictions = np.asarray(evaluation.ensemble[eki.PREDICTION])
        increment = conditionals.largest_increment(
            conditionals.quadratic_forms(predictions), beta, ess_fraction
        )
        noise_variance = conditionals.noise_variance(conditionals.common_scales(scales))
        record = _build_step_record(problem, evaluation, beta, increment, scales, noise_variance)
        for callback in callbacks:
            callback(record)
        step_records.append(record)
        state = eki.assimilate(
            state,
            evaluation,
            increment,
            jnp.asarray(problem.observations),
            linalg.PSDDiagonal(jnp.asarray(noise_variance)),
            update_rule=kalman.Matheron(),
        )
        if float(state.beta) < 1.0 or conditionals.is_inferred.any():
            evaluation = _evaluate_forward_map_at_particles(problem, state, scales)
            scales = _gibbs_draw_noise_scales(problem, evaluation, state, scales, key_scales)
    return _ParticlesWithScales(
        z=_final_particles_as_evaluated(state, evaluation), scales=scales
    ), step_records


def _final_particles_as_evaluated(state: Any, evaluation: Any) -> np.ndarray:
    r"""The final particles, as repaired by their evaluation if they were evaluated at $\beta = 1$.

    With inferred scales the final ensemble is evaluated, so that its scales can be drawn, and any
    particle at which the forward map failed was moved to the valid particles' centre; returning
    the evaluated particles means each particle's scales were drawn from its own predictions.
    """
    if int(evaluation.step) == int(state.step):
        return np.asarray(evaluation.ensemble["u"])
    return np.asarray(state.ensemble["u"])


def _evaluate_forward_map_at_particles(
    problem: _GaussianInverseProblem, state: Any, scales: np.ndarray
) -> Any:
    """EnsKit's evaluation of *state*: the forward map at every particle, failures repaired."""
    conditionals = problem.scale_conditionals
    noise = conditionals.noise_variance(conditionals.common_scales(scales))
    return eki.evaluate(
        state,
        problem.forward_in_unconstrained_coordinates,
        jnp.asarray(problem.observations),
        linalg.PSDDiagonal(jnp.asarray(noise)),
        on_failure="repair",
    )


def _gibbs_draw_noise_scales(
    problem: _GaussianInverseProblem,
    evaluation: Any,
    state: Any,
    scales: np.ndarray,
    key: Any,
) -> np.ndarray:
    """The particles' scales drawn at the state's level, or *scales* if none is inferred."""
    conditionals = problem.scale_conditionals
    if not conditionals.is_inferred.any():
        return scales
    predictions = np.asarray(evaluation.ensemble[eki.PREDICTION])
    return conditionals.draw_scales(
        conditionals.quadratic_forms(predictions),
        float(state.beta),
        jax.random.fold_in(key, int(state.step)),
    )


def _build_step_record(
    problem: _GaussianInverseProblem,
    evaluation: Any,
    beta: float,
    increment: float,
    scales: np.ndarray,
    noise_variance: np.ndarray,
) -> StepRecord:
    """The :class:`StepRecord` of one step."""
    unconstrained = np.asarray(evaluation.ensemble["u"])
    conditionals = problem.scale_conditionals
    return StepRecord(
        step=int(evaluation.step),
        beta=beta,
        increment=float(increment),
        parameter_names=problem.bijection.names,
        theta=problem.bijection.to_constrained(unconstrained),
        predictions=np.asarray(evaluation.ensemble[eki.PREDICTION]),
        noise_variance=noise_variance,
        noise_scales={
            slot: scales[:, k] for k, slot in enumerate(conditionals.slots) if slot is not None
        },
        n_failed=unconstrained.shape[0] - int(evaluation.n_valid),
    )


def _likelihood_and_learning_rate(likelihood: Any) -> tuple[Any, np.ndarray]:
    """The likelihood beneath any tempering, and the learning rate (1 when untempered)."""
    if isinstance(likelihood, TemperedLikelihood):
        return likelihood.likelihood, likelihood.learning_rate
    return likelihood, np.asarray(1.0)


def _inverse_gamma_scale_priors(prior: Any, slots: Sequence[str]) -> dict[str, InverseGamma]:
    """The inverse-gamma priors of the inferred scales *slots*, by slot; others are left out."""
    return {
        slot: factor
        for slot, factor in factors_over(prior, slots).items()
        if isinstance(factor, InverseGamma)
    }


def _particles_as_target_vectors(
    target: InferenceTarget, problem: _GaussianInverseProblem, particles: _ParticlesWithScales
) -> jnp.ndarray:
    """The final particles, with their inferred scales, as vectors of the target's event."""
    theta = problem.bijection.to_constrained(particles.z)
    fields: dict[str, Any] = {
        _PARAMETER_BLOCK: {
            name: jnp.asarray(theta[:, index]) for index, name in enumerate(problem.bijection.names)
        }
    }
    for k, slot in enumerate(problem.scale_conditionals.slots):
        if slot is not None:
            fields[slot] = jnp.asarray(particles.scales[:, k])
    batch = NumericRecordBatch(fields, "draw", element=target.event_spec.term_spec)
    return batch.to_vector()


def _infeasibility_reason(target: InferenceTarget) -> str | None:
    """Why EKI cannot condition *target*, or None if it can."""
    factors = target.factors()
    if factors is None:
        return "the model must be a merge likelihood & prior, conditioned on its observations"
    likelihood, _ = _likelihood_and_learning_rate(factors.likelihood)
    if not isinstance(likelihood, GaussianSimulatorLikelihood):
        return "the likelihood must be a Gaussian simulator likelihood"
    slots = set(likelihood.random_scale_slots)
    if set(factors.prior.event_spec.outputs) != {_PARAMETER_BLOCK} | slots:
        return (
            f"the prior must be over the parameter record {_PARAMETER_BLOCK!r} and the inferred "
            f"noise scales {sorted(slots)}, and nothing else"
        )
    if missing := sorted(slots - _inverse_gamma_scale_priors(factors.prior, sorted(slots)).keys()):
        return f"each inferred noise scale needs an inverse-gamma prior; these have none: {missing}"
    return None
