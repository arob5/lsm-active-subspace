r"""Priors: families specified by a central interval, and independent priors built from a table.

Expert knowledge usually comes as "between 140 and 450, most likely". The interval constructors
return the ProbPipe law whose central interval of the given probability is exactly that range.
:func:`laws_from_table` builds one law per row of a prior table, and :func:`independent_prior`
combines such laws into the prior of a block of calibration parameters.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from probpipe import (
    BijectorTransformedDistribution,
    Distribution,
    LogNormal,
    Normal,
    Uniform,
    bijector_for,
    unit_interval,
)
from probpipe.distributions import FactoredDistribution
from scipy import stats

from lsmas._stopgaps.quantiles import median

__all__ = [
    "independent_prior",
    "laws_from_table",
    "log_normal_from_interval",
    "logit_normal_from_interval",
    "prior_medians",
]

#: The numbers each family of a prior table is stated by.
_FAMILY_NUMBERS = {
    "log_normal": ("lower", "upper"),
    "logit_normal": ("lower", "upper"),
    "normal": ("mean", "sd"),
    "uniform": ("lower", "upper"),
}


def laws_from_table(table: pd.DataFrame) -> dict[str, Distribution]:
    r"""One law per parameter of a prior table, each named by its parameter.

    Each row states a family by its own numbers: ``log_normal`` and ``logit_normal`` by their
    central 95% interval [``lower``, ``upper``] (see :func:`log_normal_from_interval` and
    :func:`logit_normal_from_interval`), ``normal`` by ``mean`` and standard deviation ``sd``, and
    ``uniform`` by its bounds [``lower``, ``upper``].

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by parameter name, with a ``family`` column and the columns of the numbers above;
        a number the row's family does not use is NaN.

    Raises
    ------
    ValueError
        If a row names an unknown family, lacks a number its family uses, gives one it does not,
        or gives numbers that state no law: an interval or bounds out of order, or a standard
        deviation that is not positive.
    """
    return {str(name): _law(str(name), row) for name, row in table.iterrows()}


def prior_medians(table: pd.DataFrame) -> dict[str, float]:
    """Each parameter's prior median, exactly, by name.

    Parameters
    ----------
    table : pandas.DataFrame
        A prior table, as :func:`laws_from_table` reads it.
    """
    return {name: median(law) for name, law in laws_from_table(table).items()}


def independent_prior(laws: Mapping[str, Distribution], group: str = "theta") -> Distribution:
    r"""The product of independent laws, with their outputs grouped under *group*.

    For laws $\pi_j$ of parameters $\theta_j$, the prior is $\pi(\theta) = \prod_j \pi_j(\theta_j)$,
    and its draws read $\theta_j$ at ``group/name``.

    Parameters
    ----------
    laws : Mapping of str to Distribution
        Each law, by the name of its output.
    group : str
        The output that holds the parameters as a record.

    Raises
    ------
    ValueError
        If *laws* is empty.
    """
    if not laws:
        raise ValueError("an independent prior needs at least one law")
    product = FactoredDistribution(list(laws.values()))
    return product.rename({name: f"{group}/{name}" for name in laws})


def log_normal_from_interval(
    output_name: str, lower: float, upper: float, *, probability: float = 0.95
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
    output_name : str
        The name of the law's output, by which its draws are read.
    lower, upper : float
        $0 < \ell < u$.
    probability : float
        $p \in (0, 1)$.
    """
    location, scale = _normal_from_interval(np.log(lower), np.log(upper), probability)
    return LogNormal(output_name, location, scale)


def logit_normal_from_interval(
    output_name: str, lower: float, upper: float, *, probability: float = 0.95
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
    output_name : str
        The name of the law's output, by which its draws are read.
    lower, upper : float
        $0 < \ell < u < 1$.
    probability : float
        $p \in (0, 1)$.
    """
    location, scale = _normal_from_interval(
        stats.logistic.ppf(lower), stats.logistic.ppf(upper), probability
    )
    base = Normal(f"logit_{output_name}", location, scale)
    return BijectorTransformedDistribution(output_name, base, bijector_for(unit_interval))


def _law(name: str, row: pd.Series) -> Distribution:
    """The law that one row of a prior table states."""
    family = row["family"]
    if family not in _FAMILY_NUMBERS:
        raise ValueError(f"{name}: unknown family {family!r}; known: {sorted(_FAMILY_NUMBERS)}")
    first, second = _numbers(name, row, _FAMILY_NUMBERS[family])
    if family == "log_normal":
        return log_normal_from_interval(name, first, second)
    if family == "logit_normal":
        return logit_normal_from_interval(name, first, second)
    if family == "normal":
        if not second > 0:
            raise ValueError(f"{name}: the standard deviation must be positive, got {second}")
        return Normal(name, first, second)
    if not first < second:
        raise ValueError(f"{name}: the bounds must have lower < upper, got [{first}, {second}]")
    return Uniform(name, first, second)


def _numbers(name: str, row: pd.Series, used: tuple[str, ...]) -> list[float]:
    """The numbers *used* from *row*, after checking that it gives exactly those."""
    stated = {column for columns in _FAMILY_NUMBERS.values() for column in columns}
    if missing := [column for column in used if pd.isna(row[column])]:
        raise ValueError(f"{name}: family {row['family']!r} needs {missing}")
    if extra := sorted(column for column in stated - set(used) if pd.notna(row[column])):
        raise ValueError(f"{name}: family {row['family']!r} does not use {extra}")
    return [float(row[column]) for column in used]


def _normal_from_interval(lower: float, upper: float, probability: float) -> tuple[float, float]:
    """The mean and standard deviation of the normal law with central interval [lower, upper]."""
    if not lower < upper:
        raise ValueError(f"the interval must have lower < upper, got [{lower}, {upper}]")
    half_width_in_sds = stats.norm.ppf(0.5 + probability / 2)
    return float((lower + upper) / 2), float((upper - lower) / (2 * half_width_in_sds))
