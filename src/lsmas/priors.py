r"""Prior families specified by a central interval rather than by their parameters.

Expert knowledge usually comes as "between 140 and 450, most likely". These constructors return the
ProbPipe law whose central interval of the given probability is exactly that range.
"""

from __future__ import annotations

import numpy as np
from probpipe import (
    BijectorTransformedDistribution,
    Distribution,
    LogNormal,
    Normal,
    bijector_for,
    unit_interval,
)
from scipy import stats

__all__ = ["log_normal_from_interval", "logit_normal_from_interval"]


def log_normal_from_interval(
    component: str, lower: float, upper: float, *, probability: float = 0.95
) -> Distribution:
    r"""The log-normal law whose central interval of probability $p$ is $[\ell, u]$.

    $\log X \sim \mathcal N(\mu, \sigma^2)$ with

    $$
    \mu = \tfrac12 (\log \ell + \log u), \qquad
    \sigma = \frac{\log u - \log \ell}{2\, \Phi^{-1}\!\left(\tfrac{1 + p}{2}\right)},
    $$

    so the median is the geometric mean $\sqrt{\ell u}$.

    Parameters
    ----------
    component : str
        The name of the law's draws.
    lower, upper : float
        $0 < \ell < u$.
    probability : float
        $p \in (0, 1)$.
    """
    location, scale = _normal_from_interval(np.log(lower), np.log(upper), probability)
    return LogNormal(component, location, scale)


def logit_normal_from_interval(
    component: str, lower: float, upper: float, *, probability: float = 0.95
) -> Distribution:
    r"""The logit-normal law on $(0, 1)$ whose central interval of probability $p$ is $[\ell, u]$.

    $\operatorname{logit} X \sim \mathcal N(\mu, \sigma^2)$ with

    $$
    \mu = \tfrac12 \left(\operatorname{logit} \ell + \operatorname{logit} u\right), \qquad
    \sigma = \frac{\operatorname{logit} u - \operatorname{logit} \ell}
                  {2\, \Phi^{-1}\!\left(\tfrac{1 + p}{2}\right)},
    $$

    where $\operatorname{logit} x = \log\{x / (1 - x)\}$.

    Parameters
    ----------
    component : str
        The name of the law's draws.
    lower, upper : float
        $0 < \ell < u < 1$.
    probability : float
        $p \in (0, 1)$.
    """
    location, scale = _normal_from_interval(
        stats.logistic.ppf(lower), stats.logistic.ppf(upper), probability
    )
    base = Normal(f"logit_{component}", location, scale)
    return BijectorTransformedDistribution(component, base, bijector_for(unit_interval))


def _normal_from_interval(lower: float, upper: float, probability: float) -> tuple[float, float]:
    """The mean and standard deviation of the normal law with central interval [lower, upper]."""
    if not lower < upper:
        raise ValueError(f"the interval must have lower < upper, got [{lower}, {upper}]")
    half_width_in_sds = stats.norm.ppf(0.5 + probability / 2)
    return float((lower + upper) / 2), float((upper - lower) / (2 * half_width_in_sds))
