r"""Exact quantiles for a scalar law transformed by a monotone bijector.

ProbPipe's ``quantile`` is exact for a law that states its quantiles, through the ``_quantile``
method of the ``SupportsQuantile`` capability, and for a product of such laws. A
``BijectorTransformedDistribution`` states none, so ``quantile`` falls back to Monte Carlo for it,
and for any product that contains it, such as a prior with a logit-normal factor. Yet a strictly
monotone map carries quantiles over exactly. :class:`MonotoneTransformedDistribution` adds
``_quantile`` to a transformed law, so that ``quantile`` takes its exact route. Once ProbPipe
states the quantiles of a law transformed by a monotone bijector, the class can be replaced by
``BijectorTransformedDistribution`` and this module deleted.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from probpipe import BijectorTransformedDistribution, quantile

__all__ = ["MonotoneTransformedDistribution"]


# STOPGAP(PG-16): use BijectorTransformedDistribution once ProbPipe states the quantiles of a law
# transformed by a monotone bijector.
class MonotoneTransformedDistribution(BijectorTransformedDistribution):
    r"""The law of $Y = g(X)$ for a scalar $X$ and a strictly monotone bijector $g$, with exact
    quantiles.

    For $X$ with quantile function $q_X$,

    $$
    q_Y(p) = \begin{cases} g\!\left(q_X(p)\right) & g \text{ increasing}, \\
    g\!\left(q_X(1 - p)\right) & g \text{ decreasing}, \end{cases}
    $$

    since $\{X \le x\}$ and $\{Y \le g(x)\}$ are the same event when $g$ increases, and
    $\{X \ge x\}$ and $\{Y \le g(x)\}$ are when it decreases. It is constructed like
    ``BijectorTransformedDistribution`` and behaves like it in every other way.

    Parameters
    ----------
    output_name : str
        The name of the law's output.
    base : Distribution
        The law of $X$, a scalar law with closed-form quantiles.
    bijector : Function
        $g$, strictly monotone on the real line, as the bijectors ProbPipe assigns to supports are.
    """

    def _quantile(self, q: Any) -> Any:
        """$q_Y$ at the levels *q*, through the base law's exact quantiles."""
        levels = jnp.asarray(q)
        at_minus_one, at_one = self._map(jnp.asarray([-1.0, 1.0]))
        base_levels = levels if at_one > at_minus_one else 1.0 - levels
        return self._map(quantile.using(exact_only=True)(self.base, base_levels).raw)

    def _map(self, x: Any) -> Any:
        """$g(x)$ as a raw array."""
        return jnp.asarray(self.bijector(x).raw)
