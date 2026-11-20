"""Design 10: the prototype likelihood, whose noise scales ``s2`` are an optional input."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import Distribution, InputSpec, NumericArraySpec, NumericRecordSpec, OutputSpec
from probpipe.distributions import ConditionalDistribution


class ScaledNoiseLikelihood(ConditionalDistribution):
    """y_i | theta, s2 ~ N(G_i(theta), s2_k(i) sigma2_i); s2 is optional, with default scales."""

    def __init__(self, forward, *, output_name, reference_variance, classes, default_scales, label):
        self.forward = forward
        self._raw_forward = forward.using(raw=True)
        self.reference_variance = jnp.asarray(reference_variance, dtype=float)
        self.default_scales = dict(default_scales)
        self._class_index = np.array(
            [list(self.default_scales).index(c) for c in np.asarray(classes)]
        )
        scale_record = NumericRecordSpec(**{c: NumericArraySpec(()) for c in self.default_scales})
        given = InputSpec(theta=forward.input_spec["theta"], s2=scale_record).with_optional("s2")
        event = OutputSpec(**{output_name: NumericArraySpec((len(self._class_index),))})
        super().__init__(given, event, label=label)

    def noise_variance(self, given: Mapping[str, Any]) -> jax.Array:
        scales = given["s2"] if "s2" in given else self.default_scales  # noqa: SIM401 (records have no get)
        class_scales = jnp.stack([jnp.asarray(scales[c]) for c in self.default_scales], axis=-1)
        return class_scales[..., self._class_index] * self.reference_variance

    def pointwise_log_prob(self, given, value):
        residuals = jnp.asarray(value) - jnp.asarray(self._raw_forward(given["theta"]))
        variances = self.noise_variance(given)
        return -0.5 * (residuals**2 / variances + jnp.log(2.0 * jnp.pi * variances))

    def _conditional_sample(self, given, key, sample_shape=()):
        mean = jnp.asarray(self._raw_forward(given["theta"]))
        noise = jax.random.normal(key, (*sample_shape, *mean.shape))
        return mean + jnp.sqrt(self.noise_variance(given)) * noise

    def _condition_on(self, given, /, **options):
        return _LawAtGiven(self, given)

    def _conditional_log_prob(self, given, value):
        return jnp.sum(self.pointwise_log_prob(given, value))

    def _conditional_unnormalized_log_prob(self, given, value):
        return self._conditional_log_prob(given, value)


class _LawAtGiven(Distribution):
    def __init__(self, likelihood, given):
        super().__init__(likelihood.event_spec, label=f"{likelihood.label} at given values")
        self._likelihood, self._given = likelihood, given

    def _log_prob(self, value):
        return self._likelihood._conditional_log_prob(self._given, value)

    def _unnormalized_log_prob(self, value):
        return self._log_prob(value)

    def _sample(self, key, sample_shape=()):
        return self._likelihood._conditional_sample(self._given, key, sample_shape)

    def _variance(self):
        return self._likelihood.noise_variance(self._given)

    def _mean(self):
        return jnp.asarray(self._likelihood._raw_forward(self._given["theta"]))
