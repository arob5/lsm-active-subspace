r"""The exact median of a scalar law, including one transformed by a bijector.

ProbPipe's ``quantile`` is exact for a family that states its quantiles, such as ``Normal`` or
``LogNormal``, but a ``BijectorTransformedDistribution`` states none, so ``quantile`` falls back to
Monte Carlo: approximate, and different from run to run. Yet a quantile passes exactly through a
strictly monotone map. ProbPipe would ideally state the quantiles of a scalar law transformed by a
monotone bijector. :func:`median` uses that fact for the median, so its calls can be replaced by
``quantile(law, 0.5)`` under ``exact_only`` once ProbPipe does, after which this module can be
deleted.
"""

from __future__ import annotations

from probpipe import BijectorTransformedDistribution, Distribution, quantile

__all__ = ["median"]


# STOPGAP(PG-16): replace calls with quantile(law, 0.5) under exact_only once ProbPipe states the
# quantiles of a law transformed by a monotone bijector.
def median(law: Distribution) -> float:
    r"""The median of a scalar law, computed exactly.

    For a law with closed-form quantiles, the median is its quantile at $\tfrac12$. For
    $Y = g(X)$, with $g$ a strictly monotone bijector of the real line onto an interval,
    $\operatorname{median}(Y) = g(\operatorname{median}(X))$: the events $\{X \le m\}$ and
    $\{Y \le g(m)\}$ coincide when $g$ increases, and $\{X \le m\}$ and $\{Y \ge g(m)\}$ when it
    decreases, and either has probability $\tfrac12$ for a continuous law.

    Parameters
    ----------
    law : Distribution
        A ProbPipe law of a continuous scalar, either with closed-form quantiles or a
        ``BijectorTransformedDistribution`` of such a law.

    Raises
    ------
    ResolutionError
        If neither *law* nor, for a transformed law, its base has closed-form quantiles.
    """
    if isinstance(law, BijectorTransformedDistribution):
        return float(law.bijector(median(law.base)).raw)
    return float(quantile.using(exact_only=True)(law, 0.5).raw)
