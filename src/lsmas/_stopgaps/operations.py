r"""ProbPipe's ``expectation``, ``pushforward`` and ``cov`` as they will behave once three issues
are fixed.

- **ProbPipe #716.** Over a law held as samples, ``expectation`` hands the integrand each value
  with its fields in alphabetical order, and a ``Function`` called on it refuses it. Here every
  value reaches the integrand with its fields in their declared order.
- **ProbPipe #717 and #585.** ``pushforward`` uses at most 256 held samples, resampling silently
  beyond that, and ignores ``num_samples``. Here it uses every held sample, and ``num_samples``
  sets the number of samples drawn from any other law.
- **ProbPipe #718.** ``cov`` gives the covariance of one law only. Here ``cov(y, x)`` is the
  cross-covariance of two coupled marginals of one joint law, such as ``joint["g"]`` and
  ``joint["x"]``.

The functions take ``num_samples`` as a keyword rather than through ``.using``. When the pin moves
past the fixes, callers import ProbPipe's operations instead and this module goes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from probpipe import EmpiricalDistribution, NumericRecordSpec, Record, function, mean
from probpipe import cov as probpipe_cov
from probpipe import expectation as probpipe_expectation
from probpipe import pushforward as probpipe_pushforward

__all__ = ["cov", "expectation", "pushforward"]


# STOPGAP(PG-21): ProbPipe #716, held samples reach the integrand with their fields reordered.
def expectation(
    distribution: Any, f: Callable[[Any], Any], *, num_samples: int | None = None
) -> Any:
    r"""$\mathbb E[f(X)]$ for $X \sim$ *distribution*, as ProbPipe's ``expectation``.

    Over a law held as samples (an ``EmpiricalDistribution``), every sample is used with its
    weight. Over any other law, *num_samples* samples are drawn from the current ``workflow_run``'s
    stream, or ProbPipe's default number when it is not given.

    Parameters
    ----------
    distribution : Distribution
        The law to integrate against.
    f : callable
        Maps one value of *distribution* to an array.
    num_samples : int, optional
        How many samples to draw from a law not held as samples.
    """
    operation = (
        probpipe_expectation
        if num_samples is None
        else probpipe_expectation.using(num_samples=num_samples)
    )
    return operation(distribution, _in_declared_order(f, distribution))


# STOPGAP(PG-10): ProbPipe #717 and #585; pushforward resamples held samples, ignores num_samples.
def pushforward(
    distribution: Any, f: Callable[[Any], Any], *, num_samples: int | None = None
) -> Any:
    r"""The law of $f(X)$ for $X \sim$ *distribution*, as ProbPipe's ``pushforward``.

    Over a law held as samples (an ``EmpiricalDistribution``), $f$ is evaluated at every sample,
    which keeps its weight, as ``expectation`` uses every sample. Over any other law,
    *num_samples* samples are drawn from the current ``workflow_run``'s stream.

    Parameters
    ----------
    distribution : Distribution
        The law to push forward.
    f : callable
        Maps one value of *distribution* to a value of the result.
    num_samples : int, optional
        How many samples to draw from a law not held as samples.
    """
    if isinstance(distribution, EmpiricalDistribution):
        num_samples = distribution.num_atoms
    operation = (
        probpipe_pushforward
        if num_samples is None
        else probpipe_pushforward.using(max_evaluations=num_samples)
    )
    return operation(distribution, _in_declared_order(f, distribution))


# STOPGAP(PG-23): ProbPipe #718, no cross-covariance of two coupled marginals.
def cov(y: Any, x: Any = None) -> Any:
    r"""The covariance of *y*, or the cross-covariance $\operatorname{Cov}(y, x)$ of two coupled
    marginals of one joint law.

    $$
    \operatorname{Cov}(y, x) = \mathbb E\!\left[y\, x^\top\right]
    - \mathbb E[y]\, \mathbb E[x]^\top,
    $$

    a matrix with one row per coordinate of *y* and one column per coordinate of *x*. The two
    marginals are sampled together from their joint, as ProbPipe samples coupled marginals in any
    call; over a joint held as samples every sample is used, with its weight.

    Parameters
    ----------
    y : Distribution
        A law, such as ``joint["g"]``.
    x : Distribution, optional
        A coupled marginal of the same joint, such as ``joint["x"]``. Without it, the result is
        ProbPipe's ``cov(y)``.
    """
    if x is None:
        return probpipe_cov(y)
    outer = function(lambda y, x: jnp.outer(_flat(y), _flat(x)), traceable=True)
    joint = y.parent
    if isinstance(joint, EmpiricalDistribution):
        outer = outer.using(max_evaluations=joint.num_atoms)
    expected_outer = jnp.asarray(mean(outer(y, x)).raw)
    return expected_outer - jnp.outer(_flat(mean(y)), _flat(mean(x)))


def _flat(value: Any) -> Any:
    """A value as one vector: a record's ``to_vector`` layout, or an array raveled."""
    if hasattr(value, "to_vector"):
        return jnp.asarray(value.to_vector())
    return jnp.ravel(jnp.asarray(getattr(value, "raw", value)))


def _in_declared_order(f: Callable[[Any], Any], distribution: Any) -> Callable[[Any], Any]:
    """*f*, called on each value with its fields in the order *distribution* declares them."""
    spec = distribution.event_spec.term_spec
    if not isinstance(spec, NumericRecordSpec):
        return f

    def ordered_f(value: Any) -> Any:
        return f(Record(_ordered_fields(value, spec), label="value"))

    return ordered_f


def _ordered_fields(value: Any, spec: Any) -> Any:
    """The nested mapping of *value*'s fields in the order of the record spec *spec*."""
    if not isinstance(spec, NumericRecordSpec):
        return value
    return {name: _ordered_fields(value[name], spec.at_path(name)) for name in spec.fields}
