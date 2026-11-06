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
    $s^2_j$ start as prior draws. The method climbs a ladder $0 = \beta_0 < \beta_1 < \dots <
    \beta_T = 1$, the level $\beta$ targeting the tempered posterior with likelihood
    $\mathcal N(y \mid G, \Gamma / \beta)$. Each step evaluates $G$ at every particle (EnsKit's
    ``eki.evaluate``, the step's one batch of forward runs) and then:

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
    - ``max_steps`` (int, default 1000): the most steps to take before giving up.

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
        start = problem.prior_draws(key_prior, options.get("n_particles", 100))
        final, records = _climb_ladder(
            problem, start, key_kalman=key_kalman, key_scales=key_scales, options=options
        )
        return make_posterior(
            target,
            [_vectors_of_target(target, problem, final)],
            method=self.name,
            annotations=xr.DataTree.from_dict({"eki_history": history_dataset(records)}),
            n_particles=final.z.shape[0],
            n_steps=len(records),
        )


@dataclass(frozen=True)
class _Particles:
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
        likelihood, learning_rate = _untempered(factors.likelihood)
        observations = np.asarray(factors.observed, dtype=float)
        return cls(
            likelihood=likelihood,
            observations=observations,
            prior=factors.prior,
            bijection=ParameterBijection.from_prior(factors.prior, _PARAMETER_BLOCK),
            scale_conditionals=NoiseScaleConditionals.from_problem(
                likelihood, _scale_priors(factors.prior), observations, learning_rate
            ),
        )

    def prior_draws(self, key: Any, n_particles: int) -> _Particles:
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
        return _Particles(z=self.bijection.to_unconstrained(theta), scales=scales)

    def forward_in_unconstrained_coordinates(self, z: Any) -> np.ndarray:
        """$G$ at each row of unconstrained coordinates *z*, shape $(J, n)$; EnsKit's simulator."""
        theta = self.bijection.to_constrained(np.asarray(z))
        return np.asarray(self.likelihood.forward(self.bijection.record_batch(theta)).raw)


def _climb_ladder(
    problem: _GaussianInverseProblem,
    start: _Particles,
    *,
    key_kalman: Any,
    key_scales: Any,
    options: Mapping[str, Any],
) -> tuple[_Particles, list[StepRecord]]:
    """Climb the tempering ladder from *start*; return the final particles and every step's record.

    Raises
    ------
    RuntimeError
        If the ladder does not reach $\\beta = 1$ within ``max_steps`` steps.
    """
    callbacks: Sequence[StepCallback] = options.get("callbacks", ())
    ess_fraction = float(options.get("ess_fraction", 0.5))
    max_steps = int(options.get("max_steps", 1000))
    conditionals = problem.scale_conditionals
    state = eki.EKIState(Ensemble(u=jnp.asarray(start.z)), key=key_kalman)
    scales = start.scales
    evaluation = _evaluate(problem, state, scales)
    records: list[StepRecord] = []
    while float(state.beta) < 1.0:
        if len(records) == max_steps:
            raise RuntimeError(
                f"EKI reached only beta = {float(state.beta):.4g} in {max_steps} steps"
            )
        beta = float(state.beta)
        predictions = np.asarray(evaluation.ensemble[eki.PREDICTION])
        increment = conditionals.largest_increment(
            conditionals.quadratic_forms(predictions), beta, ess_fraction
        )
        noise_variance = conditionals.noise_variance(conditionals.common_scales(scales))
        record = _step_record(problem, evaluation, beta, increment, scales, noise_variance)
        for callback in callbacks:
            callback(record)
        records.append(record)
        state = eki.assimilate(
            state,
            evaluation,
            increment,
            jnp.asarray(problem.observations),
            linalg.PSDDiagonal(jnp.asarray(noise_variance)),
            update_rule=kalman.Matheron(),
        )
        if float(state.beta) < 1.0 or conditionals.inferred.any():
            evaluation = _evaluate(problem, state, scales)
            scales = _draw_scales(problem, evaluation, state, scales, key_scales)
    return _Particles(z=np.asarray(state.ensemble["u"]), scales=scales), records


def _evaluate(problem: _GaussianInverseProblem, state: Any, scales: np.ndarray) -> Any:
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


def _draw_scales(
    problem: _GaussianInverseProblem,
    evaluation: Any,
    state: Any,
    scales: np.ndarray,
    key: Any,
) -> np.ndarray:
    """The particles' scales drawn at the state's level, or *scales* if none is inferred."""
    conditionals = problem.scale_conditionals
    if not conditionals.inferred.any():
        return scales
    predictions = np.asarray(evaluation.ensemble[eki.PREDICTION])
    return conditionals.draw(
        conditionals.quadratic_forms(predictions),
        float(state.beta),
        jax.random.fold_in(key, int(state.step)),
    )


def _step_record(
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


def _untempered(likelihood: Any) -> tuple[Any, np.ndarray]:
    """The likelihood beneath any tempering, and the learning rate (1 when untempered)."""
    if isinstance(likelihood, TemperedLikelihood):
        return likelihood.likelihood, likelihood.learning_rate
    return likelihood, np.asarray(1.0)


def _scale_priors(prior: Any) -> dict[str, InverseGamma]:
    """The inverse-gamma factors of *prior*, by the name of the scale each one is over."""
    factors = getattr(prior, "factors", ())
    return {
        next(iter(factor.event_spec.outputs)): factor
        for factor in factors
        if isinstance(factor, InverseGamma)
    }


def _vectors_of_target(
    target: InferenceTarget, problem: _GaussianInverseProblem, particles: _Particles
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
    likelihood, _ = _untempered(factors.likelihood)
    if not isinstance(likelihood, GaussianSimulatorLikelihood):
        return "the likelihood must be a Gaussian simulator likelihood"
    slots = set(likelihood.random_scale_slots)
    if set(factors.prior.event_spec.outputs) != {_PARAMETER_BLOCK} | slots:
        return (
            f"the prior must be over the parameter record {_PARAMETER_BLOCK!r} and the inferred "
            f"noise scales {sorted(slots)}, and nothing else"
        )
    if missing := sorted(slots - _scale_priors(factors.prior).keys()):
        return f"each inferred noise scale needs an inverse-gamma prior; these have none: {missing}"
    return None
