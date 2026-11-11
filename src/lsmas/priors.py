r"""Priors: families specified by a central interval, and independent priors built from a table.

Prior knowledge of a parameter is often an interval, such as 140 to 450. The interval constructors
return the ProbPipe law whose central interval of the given probability is exactly that range.
:func:`laws_from_table` builds one law per row of a prior table, and :func:`independent_prior`
combines such laws into the prior of a block of calibration parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from probpipe import (
    Distribution,
    LogNormal,
    Normal,
    Uniform,
    bijector_for,
    unit_interval,
)
from probpipe.distributions import FactoredDistribution
from scipy import stats

from lsmas._stopgaps.quantiles import MonotoneTransformedDistribution

__all__ = [
    "factors_over",
    "independent_prior",
    "laws_from_table",
    "log_normal_from_interval",
    "logit_normal_from_interval",
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
    return {str(name): _law_from_row(str(name), row) for name, row in table.iterrows()}


def independent_prior(laws: Mapping[str, Distribution], group: str = "theta") -> Distribution:
    r"""The product of independent laws, with their outputs grouped under *group*.

    For laws $\pi_{0,j}$ of parameters $\theta_j$, the prior is
    $\pi_0(\theta) = \prod_j \pi_{0,j}(\theta_j)$, and its draws read $\theta_j$ at ``group/name``.

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


def factors_over(prior: Distribution, outputs: Iterable[str]) -> dict[str, Distribution]:
    """The factors of a product prior that are laws of exactly one of *outputs*, by that output.

    A prior built with ``&``, such as ``theta_prior & scale_prior``, is a product of factors; this
    finds, for instance, the prior of the noise-scale record ``s2``. An output with no factor of
    its own is left out.

    Parameters
    ----------
    prior : Distribution
        A product of factors, such as an inverse problem's prior.
    outputs : iterable of str
        The outputs to look for, such as ``["s2"]``.
    """
    outputs_sought = set(outputs)
    factors_by_output = {}
    for factor in getattr(prior, "factors", ()):
        factor_outputs = set(factor.event_spec.outputs)
        if len(factor_outputs) == 1 and factor_outputs <= outputs_sought:
            factors_by_output[factor_outputs.pop()] = factor
    return factors_by_output


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
    location, scale = _normal_location_and_scale_from_interval(
        np.log(lower), np.log(upper), probability
    )
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
    location, scale = _normal_location_and_scale_from_interval(
        stats.logistic.ppf(lower), stats.logistic.ppf(upper), probability
    )
    base = Normal(f"logit_{output_name}", location, scale)
    return MonotoneTransformedDistribution(output_name, base, bijector_for(unit_interval))


def _law_from_row(name: str, row: pd.Series) -> Distribution:
    """The law that one row of a prior table states."""
    family = row["family"]
    if family not in _FAMILY_NUMBERS:
        raise ValueError(f"{name}: unknown family {family!r}; known: {sorted(_FAMILY_NUMBERS)}")
    numbers = _family_numbers_from_row(name, row, _FAMILY_NUMBERS[family])
    if family == "log_normal":
        return log_normal_from_interval(name, numbers["lower"], numbers["upper"])
    if family == "logit_normal":
        return logit_normal_from_interval(name, numbers["lower"], numbers["upper"])
    if family == "normal":
        if not numbers["sd"] > 0:
            raise ValueError(
                f"{name}: the standard deviation must be positive, got {numbers['sd']}"
            )
        return Normal(name, numbers["mean"], numbers["sd"])
    if not numbers["lower"] < numbers["upper"]:
        raise ValueError(
            f"{name}: the bounds must have lower < upper, "
            f"got [{numbers['lower']}, {numbers['upper']}]"
        )
    return Uniform(name, numbers["lower"], numbers["upper"])


def _family_numbers_from_row(
    name: str, row: pd.Series, used_columns: tuple[str, ...]
) -> dict[str, float]:
    """The numbers in *used_columns* of *row*, by column, after checking that it gives exactly
    those."""
    every_number_column = {column for columns in _FAMILY_NUMBERS.values() for column in columns}
    if missing := [column for column in used_columns if pd.isna(row[column])]:
        raise ValueError(f"{name}: family {row['family']!r} needs {missing}")
    if extra := sorted(
        column for column in every_number_column - set(used_columns) if pd.notna(row[column])
    ):
        raise ValueError(f"{name}: family {row['family']!r} does not use {extra}")
    return {column: float(row[column]) for column in used_columns}


def _normal_location_and_scale_from_interval(
    lower: float, upper: float, probability: float
) -> tuple[float, float]:
    """The mean and standard deviation of the normal law with central interval [lower, upper]."""
    if not lower < upper:
        raise ValueError(f"the interval must have lower < upper, got [{lower}, {upper}]")
    half_width_in_sds = stats.norm.ppf(0.5 + probability / 2)
    return float((lower + upper) / 2), float((upper - lower) / (2 * half_width_in_sds))
