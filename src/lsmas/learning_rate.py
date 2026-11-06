r"""The learning rate that sets how much the NEE observations count.

Errors in neighbouring half-monthly NEE features are strongly correlated, so treating the features
as independent would overstate what the data say. The likelihood is instead tempered by a learning
rate chosen so that each class of observations counts as a fixed number of independent observations
per year of data.
"""

from __future__ import annotations

import numpy as np

__all__ = ["effective_sample_size_learning_rate"]


def effective_sample_size_learning_rate(
    classes: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    *,
    observations_per_class_year: float,
) -> np.ndarray:
    r"""Each feature's learning rate, from a fixed effective sample size per class-year.

    For class $k$ with $n_k$ features spread over $M_k$ distinct calendar months, the data span
    $M_k / 12$ class-years. With $c$ effective observations per class-year, the class's effective
    sample size is $n_{\mathrm{eff},k} = c\, M_k / 12$ and its learning rate is

    $$
    \eta_k = \min\!\left(1,\; \frac{n_{\mathrm{eff},k}}{n_k}\right)
           = \min\!\left(1,\; \frac{c\, M_k}{12\, n_k}\right).
    $$

    Counting months rather than features makes the information proportional to the time the record
    spans, unaffected by how many features each month happens to produce.

    Parameters
    ----------
    classes : array_like of str
        Each feature's class, shape $(n,)$.
    years, months : array_like of int
        The calendar year and month of each feature, shape $(n,)$.
    observations_per_class_year : float
        $c$.

    Returns
    -------
    numpy.ndarray
        $\eta_{k(i)}$ for each feature $i$, shape $(n,)$.
    """
    classes = np.asarray(classes)
    calendar_months = np.asarray(years) * 12 + np.asarray(months)
    rates = np.empty(classes.shape, dtype=float)
    for cls in np.unique(classes):
        in_class = classes == cls
        n_months = np.unique(calendar_months[in_class]).size
        effective_size = observations_per_class_year * n_months / 12
        rates[in_class] = min(1.0, effective_size / in_class.sum())
    return rates
