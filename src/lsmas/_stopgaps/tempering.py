r"""Conditioning with a learning rate: the generalized-Bayes update.

A learning rate $\eta \in (0, 1]$ tempers the likelihood when conditioning,

$$
\pi_\eta(\theta \mid y) \propto \pi(\theta)\, p(y \mid \theta)^{\eta},
$$

which treats the data as worth fewer independent observations than they number. This is a property
of the update, not of the data model: $p(y \mid \theta)^{\eta}$ is not the density of a law of $y$,
so the model itself stays an ordinary generative model.

ProbPipe's ``condition_on`` does not accept a learning rate yet. Ideally one would write
``condition_on.using(learning_rate={"nee": eta}, method=...)(model, data)``.
:func:`tempered_condition_on` takes the same arguments, so each call can be replaced by that one
directly, after which this module can be deleted.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from probpipe import Distribution, condition_on
from probpipe.distributions import ConditionalDistribution

__all__ = ["TemperedLikelihood", "tempered_condition_on"]


# STOPGAP(PG-03): replace calls with condition_on.using(learning_rate=..., ...) once available.
def tempered_condition_on(
    model: Distribution,
    data: Mapping[str, Any],
    *,
    learning_rate: Mapping[str, Any],
    **using: Any,
) -> Distribution:
    r"""Condition *model* on *data*, tempering each observed component's likelihood.

    For a model $p(y \mid \theta)\, \pi(\theta)$ and learning rates $\eta_i$ for the elements
    $y_i$ of an observed component, the target is

    $$
    \pi_\eta(\theta \mid y) \propto \pi(\theta) \prod_i p(y_i \mid \theta)^{\eta_i}.
    $$

    A scalar rate tempers the whole component's density; an array of rates, one per element,
    needs a likelihood with independent elements (see :class:`TemperedLikelihood`). An observed
    component without a rate is not tempered.

    Parameters
    ----------
    model : Distribution
        A product ``likelihood * prior`` whose likelihood factors produce the observed components.
    data : Mapping of str to array_like
        The observed value of each component.
    learning_rate : Mapping of str to float or array_like
        The rate for each observed component to temper.
    **using
        Controls of ``condition_on``, such as ``method`` and ``method_options``.

    Returns
    -------
    Distribution
        Whatever ``condition_on.using(**using)`` returns for the tempered target.

    Raises
    ------
    ValueError
        If a learning rate names a component that is not observed.
    """
    unknown = sorted(set(learning_rate) - set(data))
    if unknown:
        raise ValueError(f"learning rates given for components that are not observed: {unknown}")
    factors = [
        _tempered_factor(factor, learning_rate) if _produces_any(factor, learning_rate) else factor
        for factor in model.factors
    ]
    tempered_model = functools.reduce(operator.mul, factors)
    return condition_on.using(**using)(tempered_model, data)


# STOPGAP(PG-03, PG-12): exists only to form a conditioning target; never returned to callers.
class TemperedLikelihood(ConditionalDistribution):
    r"""A likelihood raised to a learning rate, as a factor of a conditioning target.

    For a likelihood $p(y \mid \theta)$ and rates $\eta$, its log-density is

    $$
    \ell_\eta(\theta; y) =
    \begin{cases}
      \eta \log p(y \mid \theta) & \text{for a scalar } \eta, \\[2pt]
      \sum_i \eta_i \log p(y_i \mid \theta) & \text{for one rate per element,}
    \end{cases}
    $$

    the second requiring a likelihood whose elements are independent and which reports
    ``pointwise_log_prob``. $\ell_\eta$ is not the log-density of a law of $y$: it can be used to
    condition on observed data, and it cannot be sampled. It therefore claims an unnormalized
    density and nothing else.

    Parameters
    ----------
    likelihood : ConditionalDistribution
        The likelihood to temper. It must have a normalized conditional density.
    learning_rate : float or array_like
        $\eta$, a scalar or one rate per element of the observations.
    """

    def __init__(self, likelihood: ConditionalDistribution, learning_rate: Any) -> None:
        self.likelihood = likelihood
        self.learning_rate = np.asarray(learning_rate, dtype=float)
        _check_can_temper(likelihood, self.learning_rate)
        super().__init__(
            likelihood.given_spec, likelihood.event_spec, label=f"{likelihood.label} tempered"
        )

    def _condition_on(self, given: Mapping[str, Any], /, **options: Any) -> Distribution:
        return _TemperedFactor(self, given)

    def _conditional_unnormalized_log_prob(self, given: Mapping[str, Any], value: Any) -> Any:
        if self.learning_rate.ndim == 0:
            return self.learning_rate * self.likelihood._conditional_log_prob(given, value)
        pointwise = self.likelihood.pointwise_log_prob(given, value)
        return jnp.asarray(np.sum(self.learning_rate * pointwise))


class _TemperedFactor(Distribution):
    """A tempered likelihood at fixed given values: an unnormalized density and nothing else."""

    def __init__(self, tempered: TemperedLikelihood, given: Mapping[str, Any]) -> None:
        super().__init__(tempered.event_spec, label=f"{tempered.label} at given values")
        self._tempered = tempered
        self._given = given

    def _unnormalized_log_prob(self, value: Any) -> Any:
        return self._tempered._conditional_unnormalized_log_prob(self._given, value)


def _produces_any(factor: Any, components: Mapping[str, Any]) -> bool:
    """Whether *factor* produces at least one of *components*."""
    return bool(set(factor.event_spec.components) & set(components))


def _tempered_factor(factor: Any, learning_rate: Mapping[str, Any]) -> TemperedLikelihood:
    """*factor* tempered by the rate of the one component it produces."""
    (component,) = factor.event_spec.components
    return TemperedLikelihood(factor, learning_rate[component])


def _check_can_temper(likelihood: Any, learning_rate: np.ndarray) -> None:
    """Raise unless *likelihood* has what tempering it with *learning_rate* needs."""
    if not hasattr(likelihood, "_conditional_log_prob"):
        raise TypeError(
            f"{likelihood.label!r} has no normalized conditional density, so tempering it "
            f"is not well defined"
        )
    if np.any(learning_rate <= 0) or np.any(learning_rate > 1):
        raise ValueError("learning rates must lie in (0, 1]")
    if learning_rate.ndim > 0 and not hasattr(likelihood, "pointwise_log_prob"):
        raise TypeError(
            f"one learning rate per element needs a likelihood with independent elements that "
            f"reports pointwise_log_prob; {likelihood.label!r} does not"
        )
