r"""A Gaussian likelihood around a simulator, with one noise scale per class of observations.

ProbPipe can express this likelihood as a kernel that returns a ``Normal``, but such a kernel hides
its parts: ensemble Kalman methods need the forward map, the noise variances and which scale
multiplies which observations, separately. ProbPipe would ideally provide a family built from a
forward ``Function`` and a scaled noise covariance, claiming the conditional mean and covariance.
:class:`GaussianSimulatorLikelihood` is that family's shape; once ProbPipe has it, the class can be
replaced by the ProbPipe family and the methods that read it keep working.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import Distribution, NumericArraySpec, NumericRecordSpec, OutputSpec
from probpipe.distributions import ConditionalDistribution

__all__ = ["INFERRED", "NOISE_SCALES", "GaussianSimulatorLikelihood", "noise_scale_variable"]

#: The given record that holds the inferred noise scales $s^2_k$, one field per class, such as
#: ``s2/day``.
NOISE_SCALES = "s2"

#: The scale of a class whose noise scale is a field of :data:`NOISE_SCALES`, not a fixed number.
INFERRED = "inferred"


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

    where $G$ is the forward map. A class's scale is either a fixed number or inferred. The
    inferred scales are the fields of the given record ``s2``, one per class, such as ``s2/day``,
    so that a prior can be placed on them.

    A model ``likelihood & prior`` can be sampled directly: ``sample(model, n)`` draws the
    parameters and then the observations, evaluating $G$ once per draw.

    Parameters
    ----------
    forward : Function
        The forward map $G$: a ProbPipe ``Function`` of the parameter record ``theta``, which it
        declares as its one input, returning an array of shape $(n,)$. The likelihood's given
        ``theta`` is that declared record. Calling it on a batch of records evaluates it once per
        record. If
        JAX cannot trace it, it must declare that (``traceable=False``) and an output spec with a
        concrete shape and dtype, so that ProbPipe can call it from inside traced code.
    output_name : str
        The name of the output that holds the observations, such as ``"nee"``.
    reference_variance : array_like
        $\sigma^2_i$ for each observation, shape $(n,)$.
    classes : array_like of str
        $k(i)$ for each observation, shape $(n,)$.
    scales : Mapping of str to float or str
        For each class, its fixed scale, or ``"inferred"`` (:data:`INFERRED`).
    label : str
        The kernel's label.

    Raises
    ------
    ValueError
        If *forward* does not declare ``theta`` as its one input, or a class of observation has no
        scale, or a scale is neither a number nor ``"inferred"``.
    """

    def __init__(
        self,
        forward: Any,
        *,
        output_name: str,
        reference_variance: Any,
        classes: Any,
        scales: Mapping[str, str | float],
        label: str = "likelihood",
    ) -> None:
        self.forward = forward
        self._untracked_forward = forward.using(raw=True)
        self.output_name = output_name
        self.reference_variance = np.asarray(reference_variance, dtype=float)
        self.classes = np.asarray(classes)
        self.scales = dict(scales)
        _check_classes_have_scales(self.classes, self.scales)
        self._class_index = np.array([list(self.scales).index(cls) for cls in self.classes])
        n_observations = self.reference_variance.shape[0]
        given_spec = {"theta": _declared_parameter_record(forward)}
        if self.inferred_scale_classes:
            given_spec[NOISE_SCALES] = NumericRecordSpec(
                **{noise_class: NumericArraySpec(()) for noise_class in self.inferred_scale_classes}
            )
        event_spec = OutputSpec(**{output_name: NumericArraySpec((n_observations,))})
        super().__init__(given_spec, event_spec, label=label)

    @property
    def inferred_scale_classes(self) -> tuple[str, ...]:
        """The classes whose noise scale is inferred: the fields of the given record ``s2``."""
        return tuple(noise_class for noise_class, scale in self.scales.items() if scale == INFERRED)

    def noise_variances(self, noise_scales: Mapping[str, Any] | None = None) -> jax.Array:
        r"""The noise variances $s^2_{k(i)}\, \sigma^2_i$.

        Parameters
        ----------
        noise_scales : Mapping of str to array_like, optional
            A value for each inferred scale, by class: a value of the record ``s2``, such as
            ``{"day": 2.0, "night": 1.0}``. Values of shape $(m,)$ give variances of shape
            $(m, n)$, one row per value. Not needed when every scale is fixed.

        Returns
        -------
        jax.Array
            Shape $(n,)$, or $(m, n)$ for batched scale values.
        """
        noise_scales = {} if noise_scales is None else noise_scales
        per_class = jnp.broadcast_arrays(
            *(
                jnp.asarray(noise_scales[noise_class] if scale == INFERRED else scale)
                for noise_class, scale in self.scales.items()
            )
        )
        class_scales = jnp.stack(per_class, axis=-1)
        return class_scales[..., self._class_index] * self.reference_variance

    def inferred_scale_draws(self, draws: Any) -> dict[str, np.ndarray]:
        """The values of each inferred noise scale in *draws*, by class.

        Parameters
        ----------
        draws : Record, RecordBatch or Mapping
            Draws that hold the record ``s2``, such as a posterior's atoms, tracked or raw.

        Returns
        -------
        dict of str to numpy.ndarray
            Each with the leading shape of *draws*; empty when every scale is fixed.
        """
        raw = draws.raw if hasattr(draws, "raw") else draws
        return {
            noise_class: np.asarray(raw[NOISE_SCALES][noise_class], dtype=float)
            for noise_class in self.inferred_scale_classes
        }

    def pointwise_log_prob(self, given: Mapping[str, Any], value: Any) -> jax.Array:
        r"""Each observation's log-density, shape $(n,)$.

        $$
        \log \mathcal N\!\left(y_i \mid G_i(\theta),\; s^2_{k(i)}\, \sigma^2_i\right),
        \qquad i = 1, \dots, n.
        $$
        """
        residuals = jnp.asarray(value) - self._conditional_mean(given)
        variances = self.noise_variances(given.get(NOISE_SCALES))
        return -0.5 * (residuals**2 / variances + jnp.log(2.0 * jnp.pi * variances))

    def _condition_on(self, given: Mapping[str, Any], /, **options: Any) -> Distribution:
        return _GaussianSimulatorLaw(self, given)

    def _conditional_mean(self, given: Mapping[str, Any]) -> jax.Array:
        return jnp.asarray(self._untracked_forward(given["theta"]))

    def _conditional_sample(
        self, given: Mapping[str, Any], key: Any, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        mean = self._conditional_mean(given)
        noise = jax.random.normal(key, (*sample_shape, *mean.shape))
        return mean + jnp.sqrt(self.noise_variances(given.get(NOISE_SCALES))) * noise

    def _conditional_log_prob(self, given: Mapping[str, Any], value: Any) -> jax.Array:
        return jnp.sum(self.pointwise_log_prob(given, value))

    def _conditional_unnormalized_log_prob(self, given: Mapping[str, Any], value: Any) -> Any:
        return self._conditional_log_prob(given, value)


class _GaussianSimulatorLaw(Distribution):
    """The likelihood's law of the observations at fixed given values."""

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


def _declared_parameter_record(forward: Any) -> Any:
    """The declaration of the record ``theta`` that *forward* takes as its one input."""
    declared = forward.input_spec
    if declared is None or list(declared) != ["theta"]:
        raise ValueError(
            f"the forward map must declare one input, theta; {forward.label!r} declares "
            f"{None if declared is None else list(declared)}"
        )
    return declared["theta"]


def noise_scale_variable(noise_class: str) -> str:
    """The flat name of a class's noise scale in tables and files, such as ``s2_day``."""
    return f"{NOISE_SCALES}_{noise_class}"


def _check_classes_have_scales(classes: np.ndarray, scales: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless every class of observation has exactly one scale, and each
    scale is a number or ``"inferred"``."""
    missing = sorted(set(classes.tolist()) - set(scales))
    unused = sorted(set(scales) - set(classes.tolist()))
    if missing or unused:
        raise ValueError(
            f"every class needs one scale; classes without a scale: {missing}, "
            f"scales without observations: {unused}"
        )
    if unknown := sorted(
        noise_class
        for noise_class, scale in scales.items()
        if isinstance(scale, str) and scale != INFERRED
    ):
        raise ValueError(f"a scale is a number or {INFERRED!r}; the scales of {unknown} are not")
