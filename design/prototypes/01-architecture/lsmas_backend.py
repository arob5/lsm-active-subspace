"""Prototype: a structured Gaussian-simulator likelihood and an EnsKit EKI method for ProbPipe.

Plumbing only. The EKI-with-scales update here is a placeholder; the algorithm
itself is specified in design doc 3.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from enskit import kalman, linalg
from enskit.algorithms import eki
from enskit.distribution import Ensemble
from probpipe import Distribution, NumericArraySpec, NumericRecordBatch, OutputSpec, sample, workflow_run
from probpipe.distributions import ConditionalDistribution
from probpipe.inference import Feasibility, InferenceMethod
from probpipe.inference import inference_method_registry

# STOPGAP(PG-04, PG-05) -- private imports
from probpipe.operations._condition import _UnnormalizedConditional
from probpipe.inference._approximate_distribution import make_posterior
from probpipe.inference._inference_utils import model_factors, run_seed, unconstrained_coordinates, flat_unflatten


class GaussianSimulatorLikelihood(ConditionalDistribution):
    """A proper data model: y | theta, scales ~ N(G(theta), diag(s2[class] * v)).

    Sampling and density describe the same law. No learning rate here.
    STOPGAP(PG-02).
    """

    def __init__(self, forward, *, component: str, theta_spec, reference_variance, classes,
                 scale_slots: Mapping[str, str], label="likelihood"):
        n = len(reference_variance)
        given = {"theta": theta_spec, **{slot: NumericArraySpec(()) for slot in scale_slots.values()}}
        super().__init__(given, OutputSpec(**{component: NumericArraySpec((n,))}), label=label)
        self.forward = forward  # a probpipe Function theta -> (n,)
        self.component = component
        self.reference_variance = np.asarray(reference_variance, float)
        self.classes = np.asarray(classes)
        self.scale_slots = dict(scale_slots)  # class -> given slot name

    def variances(self, given) -> np.ndarray:
        s2 = np.empty_like(self.reference_variance)
        for cls, slot in self.scale_slots.items():
            s2[self.classes == cls] = float(given[slot])
        return s2 * self.reference_variance

    def pointwise_log_prob(self, given, value) -> np.ndarray:
        """log N(y_i | G_i, var_i) for each feature i."""
        r = np.asarray(value) - np.asarray(self._conditional_mean(given))
        var = self.variances(given)
        return -0.5 * r**2 / var - 0.5 * np.log(2 * np.pi * var)

    def _condition_on(self, given, /, **options):
        lik = self

        class _Law(Distribution):
            def __init__(self):
                super().__init__(lik.event_spec, label=f"{lik.label}_at_given")

            def _sample(self, key, sample_shape=()):
                return lik._conditional_sample(given, key, sample_shape)

            def _log_prob(self, value):
                return lik._conditional_log_prob(given, value)

            def _unnormalized_log_prob(self, value):  # SupportsLogProb needs both names
                return self._log_prob(value)

        return _Law()

    def _conditional_mean(self, given):
        return self.forward(given["theta"]).raw()

    def _conditional_sample(self, given, key, sample_shape=()):
        g = np.asarray(self._conditional_mean(given))
        z = np.asarray(jax.random.normal(key, (*sample_shape, g.shape[-1])))
        return jnp.asarray(g + np.sqrt(self.variances(given)) * z)

    def _conditional_log_prob(self, given, value):
        return jnp.asarray(np.sum(self.pointwise_log_prob(given, value)))

    def _conditional_unnormalized_log_prob(self, given, value):  # SupportsConditionalLogProb needs both names
        return self._conditional_log_prob(given, value)


class TemperedLikelihood(ConditionalDistribution):
    """The generalized-Bayes factor prod_i N(y_i | G_i, var_i)^{eta_i} -- a loss term, not a law.

    Claims only an unnormalized conditional log-density, so nothing can sample it.
    Private to tempered_condition_on. STOPGAP(PG-03, PG-12).
    """

    def __init__(self, base: GaussianSimulatorLikelihood, learning_rate):
        super().__init__(base.given_spec, base.event_spec, label=f"{base.label}^eta")
        self.base = base
        self.learning_rate = np.broadcast_to(np.asarray(learning_rate, float), base.reference_variance.shape)

    def _condition_on(self, given, /, **options):
        lik = self

        class _Factor(Distribution):
            def __init__(self):
                super().__init__(lik.event_spec, label=f"{lik.label}_at_given")

            def _unnormalized_log_prob(self, value):
                return lik._conditional_unnormalized_log_prob(given, value)

        return _Factor()

    def _conditional_unnormalized_log_prob(self, given, value):
        return jnp.asarray(np.sum(self.learning_rate * self.base.pointwise_log_prob(given, value)))


def tempered_condition_on(model, data, *, learning_rate, **using):
    """condition_on(model, data) under the generalized-Bayes update with learning rate eta.

    Ideal ProbPipe: condition_on.using(learning_rate={...}, **using)(model, data).
    STOPGAP(PG-03).
    """
    import functools, operator
    from probpipe import condition_on
    observed = set(data)
    liks = [f for f in model.factors if set(f.event_spec.components) & observed]
    rest = [f for f in model.factors if not set(f.event_spec.components) & observed]
    (lik,) = liks
    tempered = TemperedLikelihood(lik, learning_rate)
    prior = functools.reduce(operator.mul, rest)
    return condition_on.using(**using)(tempered * prior, data)


class EnsKitEKI(InferenceMethod):
    """Tempered EKI from EnsKit, with a Gibbs refresh of inverse-gamma class scales (placeholder)."""

    _method_options = ("n_particles", "ess_fraction", "callbacks", "max_steps")

    @property
    def name(self) -> str:
        return "enskit_eki"

    def supported_types(self):
        return (_UnnormalizedConditional,)

    def check(self, target, /, **options) -> Feasibility:
        f = model_factors(target)
        lik = None if f is None else f.likelihood
        lik = lik.base if isinstance(lik, TemperedLikelihood) else lik
        if not isinstance(lik, GaussianSimulatorLikelihood):
            return Feasibility(False, "needs a GaussianSimulatorLikelihood (optionally tempered) times a prior")
        return Feasibility(True)

    def execute(self, target, /, **options):
        self._check_options(options)
        f = model_factors(target)
        if isinstance(f.likelihood, TemperedLikelihood):
            lik, eta_vec = f.likelihood.base, f.likelihood.learning_rate
        else:
            lik, eta_vec = f.likelihood, np.ones_like(f.likelihood.reference_variance)
        J = int(options.get("n_particles", 50))
        callbacks: Sequence[Callable] = options.get("callbacks", ())
        key = run_seed(self.name)
        k_prior, k_state, k_gibbs = jax.random.split(key, 3)

        # initial ensemble: prior draws; theta moved to unconstrained coordinates z
        with workflow_run(seed=int(jax.random.randint(k_prior, (), 0, 2**31 - 1))):
            draws = sample(f.prior, sample_shape=(J,))
        # STOPGAP(PG-05): locate each field's slot in
        # the prior's flat vector by unflattening an index probe (every leaf here is a scalar).
        coords = unconstrained_coordinates(f.prior)
        probe = flat_unflatten(f.prior)(jnp.arange(coords.size, dtype=jnp.float32))
        nested = probe.to_nested_dict()
        theta_idx = {n: int(v) for n, v in nested["theta"].items()}
        scale_idx = {slot: int(nested[slot]) for slot in lik.scale_slots.values()}
        theta_names = list(theta_idx)
        tcols = np.array([theta_idx[n] for n in theta_names])
        flat0 = np.zeros((J, coords.size))
        for n in theta_names:
            flat0[:, theta_idx[n]] = np.asarray(draws["theta"].raw()[n])
        for slot, i in scale_idx.items():
            flat0[:, i] = np.asarray(draws[slot].raw())
        z_full = np.asarray(jax.vmap(coords.inverse)(jnp.asarray(flat0)))
        z = z_full[:, tcols]
        s2 = {slot: flat0[:, i].copy() for slot, i in scale_idx.items()}

        def to_flat(u, s2):
            zf = np.array(z_full); zf[:, tcols] = np.asarray(u)
            flat = np.array(jax.vmap(coords.forward)(jnp.asarray(zf)))
            for slot, i in scale_idx.items():
                flat[:, i] = s2[slot]
            return flat

        def forward(u):  # EnsKit simulator contract: (J, d) -> (J, n); failures are NaN rows
            flat = to_flat(u, s2)
            batch = NumericRecordBatch({n: flat[:, theta_idx[n]] for n in theta_names}, "member")
            return np.asarray(lik.forward(batch).raw())

        y = np.asarray(lik_obs := f.observed)
        a, b = 2.0, 1.678  # PLACEHOLDER (design 03): read the IG hyperparameters from the prior factors
        state = eki.EKIState(Ensemble(u=jnp.asarray(z)), key=k_state)
        schedule = eki.AdaptiveESSSchedule(ess_fraction=float(options.get("ess_fraction", 0.5)))
        for step in range(int(options.get("max_steps", 50))):
            # common noise: harmonic-mean scale per class, tempered by eta
            common = {slot: 1.0 / np.mean(1.0 / v) for slot, v in s2.items()}
            R = linalg.PSDDiagonal(jnp.asarray(lik.variances(common) / eta_vec))
            ev = eki.evaluate(state, forward, y, R, on_failure="repair")
            inc = schedule.next_increment(ev)
            for callback in callbacks:
                callback(step=step, beta=float(state.beta), increment=inc, evaluation=ev, noise=R)
            if inc is None:
                break
            state = eki.assimilate(state, ev, inc, y, R, update_rule=kalman.Matheron())
            # Gibbs refresh of each particle's scales at the new level (pre-update predictions)
            G = np.asarray(ev.ensemble[eki.PREDICTION])
            beta = float(state.beta)
            for cls, slot in lik.scale_slots.items():
                m = lik.classes == cls
                q = np.sum((y[m] - G[:, m]) ** 2 / lik.reference_variance[m], axis=1)
                eta = eta_vec[m][0]
                k_gibbs, k = jax.random.split(k_gibbs)
                shape = a + 0.5 * beta * eta * m.sum()
                rate = b + 0.5 * beta * eta * q
                s2[slot] = rate / np.asarray(jax.random.gamma(k, shape, (J,)))
            if beta >= 1.0:
                break

        flat = to_flat(state.ensemble["u"], s2)
        return make_posterior(
            [jnp.asarray(flat)], parents=(target,), method=self.name,
            event_spec=target.event_spec, 
            n_particles=J,
        )


inference_method_registry.register(EnsKitEKI())
