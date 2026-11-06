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
from probpipe import Distribution, NumericArraySpec, OutputSpec
from probpipe.distributions import ConditionalDistribution

__all__ = ["GaussianSimulatorLikelihood"]


# STOPGAP(PG-02): ProbPipe has no simulator-plus-Gaussian-noise family.
# STOPGAP(PG-11): both density method names are defined, so ProbPipe recognizes the density.
class GaussianSimulatorLikelihood(ConditionalDistribution):
    r"""Observations of a simulator with independent Gaussian noise, scaled per class.

    Each observation $i = 1, \dots, n$ belongs to a class $k(i)$ (such as day or night) and has a
    reference variance $v_i > 0$. Given the parameters $\theta$ and one scale $s^2_k > 0$ per class,

    $$
    y_i \mid \theta, s^2 \sim \mathcal N\!\left(G_i(\theta),\; s^2_{k(i)} v_i\right)
    \quad \text{independently},
    $$

    where $G$ is the forward map. A class's scale is either a given slot of the kernel, so that a
    prior can be placed on it, or a fixed number.

    Parameters
    ----------
    forward : Function
        The forward map $G$: a ProbPipe ``Function`` of the parameter record ``theta`` returning an
        array of shape $(n,)$. Calling it on a batch of records evaluates it once per record.
    component : str
        The name of the observations, such as ``"nee"``.
    theta_spec : NumericRecordSpec
        The declaration of the parameter record.
    reference_variance : array_like
        $v$, shape $(n,)$.
    classes : array_like of str
        $k(i)$ for each observation, shape $(n,)$.
    scales : Mapping of str to str or float
        For each class, the name of its scale's given slot, or its fixed value.
    label : str
        The kernel's label.
    """

    def __init__(
        self,
        forward: Any,
        *,
        component: str,
        theta_spec: Any,
        reference_variance: Any,
        classes: Any,
        scales: Mapping[str, str | float],
        label: str = "likelihood",
    ) -> None:
        self.forward = forward
        self.component = component
        self.reference_variance = np.asarray(reference_variance, dtype=float)
        self.classes = np.asarray(classes)
        self.scales = dict(scales)
        _check_classes_have_scales(self.classes, self.scales)
        self._class_index = np.array([list(self.scales).index(cls) for cls in self.classes])
        n_observations = self.reference_variance.shape[0]
        given_spec = {"theta": theta_spec} | {
            slot: NumericArraySpec(()) for slot in self.random_scale_slots
        }
        event_spec = OutputSpec(**{component: NumericArraySpec((n_observations,))})
        super().__init__(given_spec, event_spec, label=label)

    @property
    def random_scale_slots(self) -> tuple[str, ...]:
        """The given slots that hold scales, in class order."""
        return tuple(scale for scale in self.scales.values() if isinstance(scale, str))

    def noise_variances(self, scale_values: Mapping[str, Any] | None = None) -> np.ndarray:
        r"""The noise variances $s^2_{k(i)} v_i$.

        Parameters
        ----------
        scale_values : Mapping of str to array_like, optional
            A value for each scale slot. Values of shape $(m,)$ give variances of shape $(m, n)$,
            one row per value. Not needed when every scale is fixed.

        Returns
        -------
        numpy.ndarray
            Shape $(n,)$, or $(m, n)$ for batched scale values.
        """
        scale_values = {} if scale_values is None else scale_values
        per_class = np.broadcast_arrays(
            *(
                np.asarray(scale_values[scale] if isinstance(scale, str) else scale, dtype=float)
                for scale in self.scales.values()
            )
        )
        class_scales = np.stack(per_class, axis=-1)
        return class_scales[..., self._class_index] * self.reference_variance

    def pointwise_log_prob(self, given: Mapping[str, Any], value: Any) -> np.ndarray:
        r"""Each observation's log-density, shape $(n,)$.

        $$
        \log \mathcal N\!\left(y_i \mid G_i(\theta),\; s^2_{k(i)} v_i\right),
        \qquad i = 1, \dots, n.
        $$
        """
        residuals = np.asarray(value) - np.asarray(self._conditional_mean(given))
        variances = self.noise_variances(given)
        return -0.5 * (residuals**2 / variances + np.log(2.0 * np.pi * variances))

    def _condition_on(self, given: Mapping[str, Any], /, **options: Any) -> Distribution:
        return _GaussianSimulatorLaw(self, given)

    def _conditional_mean(self, given: Mapping[str, Any]) -> Any:
        return self.forward(given["theta"]).raw()

    def _conditional_sample(
        self, given: Mapping[str, Any], key: Any, sample_shape: tuple[int, ...] = ()
    ) -> Any:
        mean = np.asarray(self._conditional_mean(given))
        noise = np.asarray(jax.random.normal(key, (*sample_shape, mean.shape[-1])))
        return jnp.asarray(mean + np.sqrt(self.noise_variances(given)) * noise)

    def _conditional_log_prob(self, given: Mapping[str, Any], value: Any) -> Any:
        return jnp.asarray(np.sum(self.pointwise_log_prob(given, value)))

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


def _check_classes_have_scales(classes: np.ndarray, scales: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless every class of observation has exactly one scale."""
    missing = sorted(set(classes.tolist()) - set(scales))
    unused = sorted(set(scales) - set(classes.tolist()))
    if missing or unused:
        raise ValueError(
            f"every class needs one scale; classes without a scale: {missing}, "
            f"scales without observations: {unused}"
        )
