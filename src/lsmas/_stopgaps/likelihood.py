r"""A Gaussian likelihood around a simulator, with one noise scale per class of observations.

ProbPipe can express this likelihood as a kernel that returns a ``Normal``, but such a kernel hides
its parts: ensemble Kalman methods need the forward map, the reference variances and which scale
multiplies which observations, separately. ProbPipe would ideally provide a family built from a
forward ``Function`` and a scaled noise covariance, claiming the conditional mean and variance.
:class:`GaussianSimulatorLikelihood` is that family's shape; once ProbPipe has it, the class can be
replaced by the ProbPipe family and the code that reads it keeps working.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import Distribution, InputSpec, NumericArraySpec, NumericRecordSpec, OutputSpec
from probpipe.distributions import ConditionalDistribution

__all__ = ["NOISE_SCALES", "GaussianSimulatorLikelihood", "noise_scale_variable"]

#: The likelihood's optional input that holds the noise scales $s^2_k$, one field per class, such
#: as ``s2/day``.
NOISE_SCALES = "s2"


# STOPGAP(PG-02): ProbPipe has no simulator-plus-Gaussian-noise family.
# STOPGAP(PG-11): both density method names are defined, so ProbPipe recognizes the density.
class GaussianSimulatorLikelihood(ConditionalDistribution):
    r"""Observations of a simulator with independent Gaussian noise, scaled per class.

    Each observation $i = 1, \dots, n$ belongs to a class $k(i)$ (such as day or night) and has a
    reference variance $\sigma^2_i > 0$. Given the parameters $\theta$ and one scale $s^2_k > 0$ per
    class,

    $$
    y_i \mid \theta, s^2 \sim \mathcal N\!\left(G_i(\theta),\; s^2_{k(i)}\, \sigma^2_i\right)
    \quad \text{independently},
    $$

    where $G$ is the forward map. The scales are the optional input ``s2``, a record with one field
    per class, such as ``s2/day``. In a model ``likelihood & prior`` whose prior has the output
    ``s2``, they are parameters with that prior. In one whose prior does not, they are the
    *default_scales*.

    The likelihood's law at given values claims its mean $G(\theta)$ and its variance
    $s^2_{k(i)}\, \sigma^2_i$, so ``variance(condition_on(model, parameters))`` is the noise
    variance at the value *parameters* of the parameters; it runs no forward map.

    A model ``likelihood & prior`` can be sampled directly: ``sample(model, n)`` draws the
    parameters and then the observations, evaluating $G$ once per draw.

    Parameters
    ----------
    forward : Function
        The forward map $G$: a ProbPipe ``Function`` of the parameter record ``theta``, which it
        declares as its one input, returning an array of shape $(n,)$. Calling it on a batch of
        records evaluates it once per record. If JAX cannot trace it, it must declare that
        (``traceable=False``) and an output spec with a concrete shape and dtype, so that ProbPipe
        can call it from inside traced code.
    output_name : str
        The name of the output that holds the observations, such as ``"nee"``.
    reference_variance : array_like
        $\sigma^2_i$ for each observation, shape $(n,)$.
    classes : array_like of str
        $k(i)$ for each observation, shape $(n,)$.
    default_scales : Mapping of str to float
        The scale of each class when the model does not give ``s2``, such as
        ``{"day": 1.0, "night": 1.0}``. Its keys are the classes, in order.
    label : str
        The kernel's label.

    Raises
    ------
    ValueError
        If *forward* does not declare ``theta`` as its one input, or the classes of the
        observations are not the classes of *default_scales*.
    """

    def __init__(
        self,
        forward: Any,
        *,
        output_name: str,
        reference_variance: Any,
        classes: Any,
        default_scales: Mapping[str, float],
        label: str = "likelihood",
    ) -> None:
        self.forward = forward
        self._untracked_forward = forward.using(raw=True)
        self.output_name = output_name
        self.reference_variance = np.asarray(reference_variance, dtype=float)
        self.classes = np.asarray(classes)
        self.default_scales = {k: float(scale) for k, scale in default_scales.items()}
        _check_classes_have_scales(self.classes, self.default_scales)
        self._class_index = np.array([self.noise_classes.index(k) for k in self.classes])
        scale_record = NumericRecordSpec(
            **{noise_class: NumericArraySpec(()) for noise_class in self.noise_classes}
        )
        given_spec = InputSpec(
            theta=_declared_parameter_record(forward), **{NOISE_SCALES: scale_record}
        ).with_optional(NOISE_SCALES)
        n_observations = self.reference_variance.shape[0]
        event_spec = OutputSpec(**{output_name: NumericArraySpec((n_observations,))})
        super().__init__(given_spec, event_spec, label=label)

    @property
    def noise_classes(self) -> tuple[str, ...]:
        """The classes of observation, in order: the fields of ``s2``."""
        return tuple(self.default_scales)

    def pointwise_log_prob(self, given: Mapping[str, Any], value: Any) -> jax.Array:
        r"""Each observation's log-density, shape $(n,)$.

        $$
        \log \mathcal N\!\left(y_i \mid G_i(\theta),\; s^2_{k(i)}\, \sigma^2_i\right),
        \qquad i = 1, \dots, n.
        $$
        """
        residuals = jnp.asarray(value) - self._conditional_mean(given)
        variances = self._noise_variance(given)
        return -0.5 * (residuals**2 / variances + jnp.log(2.0 * jnp.pi * variances))

    def _noise_variance(self, given: Mapping[str, Any]) -> jax.Array:
        r"""$s^2_{k(i)}\, \sigma^2_i$, at the scales in *given* or else the default scales."""
        # A ProbPipe record has no get.
        scales = given[NOISE_SCALES] if NOISE_SCALES in given else self.default_scales  # noqa: SIM401
        class_scales = jnp.stack(
            [jnp.asarray(scales[noise_class]) for noise_class in self.noise_classes], axis=-1
        )
        return class_scales[..., self._class_index] * self.reference_variance

    def _condition_on(self, given: Mapping[str, Any], /, **options: Any) -> Distribution:
        return _GaussianSimulatorLaw(self, given)

    def _conditional_mean(self, given: Mapping[str, Any]) -> jax.Array:
        return jnp.asarray(self._untracked_forward(given["theta"]))

    def _conditional_sample(
        self, given: Mapping[str, Any], key: Any, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        mean = self._conditional_mean(given)
        noise = jax.random.normal(key, (*sample_shape, *mean.shape))
        return mean + jnp.sqrt(self._noise_variance(given)) * noise

    def _conditional_log_prob(self, given: Mapping[str, Any], value: Any) -> jax.Array:
        return jnp.sum(self.pointwise_log_prob(given, value))

    def _conditional_unnormalized_log_prob(self, given: Mapping[str, Any], value: Any) -> Any:
        return self._conditional_log_prob(given, value)


class _GaussianSimulatorLaw(Distribution):
    """The likelihood's law of the observations at given values."""

    def __init__(self, likelihood: GaussianSimulatorLikelihood, given: Mapping[str, Any]) -> None:
        super().__init__(likelihood.event_spec, label=f"{likelihood.label} at given values")
        self._likelihood = likelihood
        self._given = given

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return self._likelihood._conditional_sample(self._given, key, sample_shape)

    def _log_prob(self, value: Any) -> Any:
        return self._likelihood._conditional_log_prob(self._given, value)

    def _unnormalized_log_prob(self, value: Any) -> Any:
        return self._log_prob(value)

    def _mean(self) -> jax.Array:
        return self._likelihood._conditional_mean(self._given)

    def _variance(self) -> jax.Array:
        return self._likelihood._noise_variance(self._given)


def noise_scale_variable(noise_class: str) -> str:
    """The flat name of a class's noise scale in tables and files, such as ``s2_day``."""
    return f"{NOISE_SCALES}_{noise_class}"


def _declared_parameter_record(forward: Any) -> Any:
    """The declaration of the record ``theta`` that *forward* takes as its one input."""
    declared = forward.input_spec
    if declared is None or list(declared) != ["theta"]:
        raise ValueError(
            f"the forward map must declare one input, theta; {forward.label!r} declares "
            f"{None if declared is None else list(declared)}"
        )
    return declared["theta"]


def _check_classes_have_scales(classes: np.ndarray, default_scales: Mapping[str, float]) -> None:
    """Raise ``ValueError`` unless the classes of the observations are those of the scales."""
    classes_without_scale = sorted(set(classes.tolist()) - set(default_scales))
    scales_without_observations = sorted(set(default_scales) - set(classes.tolist()))
    if classes_without_scale or scales_without_observations:
        raise ValueError(
            f"every class needs one scale; classes without a scale: {classes_without_scale}, "
            f"scales without observations: {scales_without_observations}"
        )
