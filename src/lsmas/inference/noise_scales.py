r"""The noise scales' conditional law along a tempering ladder, for EKI with inferred noise scales.

At tempering level $\beta$ the target is joint in the parameters and the class noise scales,

$$
\pi_\beta(\theta, s^2) \propto \pi(\theta) \prod_k \mathrm{IG}\!\left(s^2_k;\, \alpha_k, b_k\right)
\prod_i \mathcal N\!\left(y_i;\, G_i(\theta),\, s^2_{k(i)} v_i\right)^{\beta \eta_i}.
$$

Given $\theta$, each inferred scale is conjugate:

$$
s^2_k \mid \theta \sim \mathrm{IG}\!\left(\alpha_k + \tfrac{\beta}{2} n_k,\;
b_k + \tfrac{\beta}{2} q_k(\theta)\right),
\qquad
n_k = \sum_{i:\, k(i) = k} \eta_i,
\qquad
q_k(\theta) = \sum_{i:\, k(i) = k} \eta_i \frac{\left(y_i - G_i(\theta)\right)^2}{v_i}.
$$

:class:`NoiseScaleConditionals` evaluates these laws and the importance weights between two levels
with the scales integrated out, which set the tempering increments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from probpipe import InverseGamma

__all__ = ["MIN_INCREMENT", "NoiseScaleConditionals"]

#: The smallest tempering increment a step may take, EnsKit's default.
MIN_INCREMENT = 1e-3


@dataclass(frozen=True, eq=False)
class NoiseScaleConditionals:
    r"""The conditional laws of the inferred noise scales given the parameters, and the importance
    weights and tempering increments computed from them.

    Classes are indexed by $k$ in the order of :attr:`classes`. A class whose scale is fixed has no
    law; it enters the weights through its fixed scale.

    Attributes
    ----------
    classes : tuple of str
        The classes of observation, such as ``("day", "night")``.
    slots : tuple of str or None
        For each class, the name of its inferred scale, such as ``"s2_day"``, or ``None`` if the
        class's scale is fixed.
    fixed_scales : numpy.ndarray
        For each class, its fixed scale; NaN where the scale is inferred. Shape $(K,)$.
    concentrations, prior_scales : numpy.ndarray
        $\alpha_k$ and $b_k$ of each inferred scale's inverse-gamma prior; NaN where fixed.
    membership : numpy.ndarray
        ``membership[k, i]`` is whether observation $i$ is in class $k$. Shape $(K, n)$.
    observations, reference_variance, learning_rate : numpy.ndarray
        $y$, $v_{k(i)}$ and $\eta$, each of shape $(n,)$.
    """

    classes: tuple[str, ...]
    slots: tuple[str | None, ...]
    fixed_scales: np.ndarray
    concentrations: np.ndarray
    prior_scales: np.ndarray
    membership: np.ndarray
    observations: np.ndarray
    reference_variance: np.ndarray
    learning_rate: np.ndarray

    @classmethod
    def from_problem(
        cls,
        likelihood: Any,
        scale_priors: Mapping[str, InverseGamma],
        observations: Any,
        learning_rate: Any,
    ) -> NoiseScaleConditionals:
        r"""The conditionals of a Gaussian simulator likelihood whose inferred scales have
        inverse-gamma priors.

        Parameters
        ----------
        likelihood : GaussianSimulatorLikelihood
            Its classes, reference variances and scales.
        scale_priors : Mapping of str to InverseGamma
            The prior of each inferred scale, by slot name.
        observations, learning_rate : array_like
            $y$ and $\eta$, shape $(n,)$.
        """
        classes = tuple(likelihood.scales)
        slots = tuple(s if isinstance(s, str) else None for s in likelihood.scales.values())
        return cls(
            classes=classes,
            slots=slots,
            fixed_scales=np.array(
                [np.nan if isinstance(s, str) else float(s) for s in likelihood.scales.values()]
            ),
            concentrations=np.array(
                [np.nan if s is None else float(scale_priors[s].concentration) for s in slots]
            ),
            prior_scales=np.array(
                [np.nan if s is None else float(scale_priors[s].scale) for s in slots]
            ),
            membership=np.stack([np.asarray(likelihood.classes) == k for k in classes]),
            observations=np.asarray(observations, dtype=float),
            reference_variance=np.asarray(likelihood.reference_variance, dtype=float),
            learning_rate=np.broadcast_to(
                np.asarray(learning_rate, dtype=float), likelihood.reference_variance.shape
            ),
        )

    @property
    def is_inferred(self) -> np.ndarray:
        """Whether each class's scale is inferred, shape $(K,)$."""
        return np.array([slot is not None for slot in self.slots])

    @property
    def effective_counts(self) -> np.ndarray:
        r"""$n_k = \sum_{i:\, k(i) = k} \eta_i$, shape $(K,)$."""
        return self.membership @ self.learning_rate

    def quadratic_forms(self, predictions: Any) -> np.ndarray:
        r"""$q_k$ at each particle's predictions $G(\theta_j)$, shape $(J, K)$."""
        residuals = self.observations - np.asarray(predictions, dtype=float)
        weighted = self.learning_rate * residuals**2 / self.reference_variance
        return weighted @ self.membership.T

    def draw_scales(self, quadratic_forms: np.ndarray, beta: float, key: Any) -> np.ndarray:
        r"""One draw of each particle's scales from their conditional law at level $\beta$.

        Returns shape $(J, K)$: the drawn $s^2_{jk}$ for inferred classes, the fixed scale
        otherwise.
        """
        shape, rate = self._conditional_shape_and_rate(quadratic_forms, beta)
        gamma_shape = np.broadcast_to(np.where(self.is_inferred, shape, 1.0), rate.shape)
        gamma = np.asarray(jax.random.gamma(key, gamma_shape))
        return np.where(self.is_inferred, rate / gamma, self.fixed_scales)

    def log_importance_weights(
        self, quadratic_forms: np.ndarray, beta: float, increment: float
    ) -> np.ndarray:
        r"""The log importance weights from level $\beta$ to $\beta' = \beta + \Delta\beta$.

        They are the ratio $\pi_{\beta'}(\theta_j) / \pi_\beta(\theta_j)$ of the targets'
        $\theta$-marginals, up to a constant: with $A_k(\beta) = \alpha_k + \beta n_k / 2$ and
        $B_{jk}(\beta) = b_k + \beta q_{jk} / 2$,

        $$
        \log w_j = \sum_{k \text{ inferred}} \left[A_k(\beta) \log B_{jk}(\beta)
        - A_k(\beta') \log B_{jk}(\beta')\right]
        - \frac{\Delta\beta}{2} \sum_{k \text{ fixed}} \frac{q_{jk}}{s^2_k}.
        $$

        Shape $(J,)$.
        """
        before_shape, before_rate = self._conditional_shape_and_rate(quadratic_forms, beta)
        after_shape, after_rate = self._conditional_shape_and_rate(
            quadratic_forms, beta + increment
        )
        inferred_terms = before_shape * np.log(before_rate) - after_shape * np.log(after_rate)
        fixed_terms = -increment / 2 * quadratic_forms / self.fixed_scales
        return np.sum(np.where(self.is_inferred, inferred_terms, fixed_terms), axis=1)

    def largest_increment(
        self,
        quadratic_forms: np.ndarray,
        beta: float,
        ess_fraction: float,
        min_increment: float = MIN_INCREMENT,
    ) -> float:
        r"""The largest $\Delta\beta \le 1 - \beta$ whose weights keep an effective sample size of
        at least ``ess_fraction`` $\times J$, found by bisection, and at least *min_increment*.

        The effective sample size of weights $w$ is $(\sum_j w_j)^2 / \sum_j w_j^2$; it falls from
        $J$ as $\Delta\beta$ grows. The floor *min_increment* (EnsKit's default) bounds the number
        of steps; on a step where the floor applies, the effective sample size falls below the
        target.
        """
        n_particles = quadratic_forms.shape[0]

        def keeps_enough(increment: float) -> bool:
            return _effective_sample_size(
                self.log_importance_weights(quadratic_forms, beta, increment)
            ) >= (ess_fraction * n_particles)

        if keeps_enough(1.0 - beta):
            return 1.0 - beta
        low, high = 0.0, 1.0 - beta
        for _ in range(60):
            middle = (low + high) / 2
            low, high = (middle, high) if keeps_enough(middle) else (low, middle)
        return min(max(low, min_increment), 1.0 - beta)

    def noise_variance(self, class_scales: np.ndarray) -> np.ndarray:
        r"""The diagonal of $\Gamma = \operatorname{diag}(s^2_{k(i)} v_i / \eta_i)$, shape $(n,)$.

        Parameters
        ----------
        class_scales : array_like
            One scale per class, shape $(K,)$.
        """
        per_observation = np.asarray(class_scales, dtype=float) @ self.membership
        return per_observation * self.reference_variance / self.learning_rate

    @staticmethod
    def common_scales(scales: np.ndarray) -> np.ndarray:
        r"""One scale per class for the whole ensemble: the mean precision's inverse,
        $\bar s^2_k = \left(\tfrac1J \sum_j s^{-2}_{jk}\right)^{-1}$, shape $(K,)$.

        The precisions $1 / s^2$ are averaged because the Kalman gain uses $\Gamma^{-1}$.
        """
        return 1.0 / np.mean(1.0 / np.asarray(scales, dtype=float), axis=0)

    def _conditional_shape_and_rate(
        self, quadratic_forms: np.ndarray, beta: float
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""$A_k(\beta)$, shape $(K,)$, and $B_{jk}(\beta)$, shape $(J, K)$."""
        shape = self.concentrations + beta * self.effective_counts / 2
        rate = self.prior_scales + beta * quadratic_forms / 2
        return shape, rate


def _effective_sample_size(log_importance_weights: np.ndarray) -> float:
    r"""$(\sum_j w_j)^2 / \sum_j w_j^2$ for weights given by their logarithms."""
    weights = np.exp(log_importance_weights - np.max(log_importance_weights))
    return float(weights.sum() ** 2 / (weights**2).sum())
