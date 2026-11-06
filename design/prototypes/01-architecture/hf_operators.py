"""Candidate NEE observation operators, each a matrix over the model's three-hourly steps.

Every operator here is a set of **features**; feature :math:`f` is the mean
of the three-hourly NEE rate over a set of steps :math:`S_f`,

.. math::

    y_f = \\frac{1}{|S_f|} \\sum_{s \\in S_f} x^{\\mathrm{obs}}_s, \\qquad
    \\mathcal H_f(\\mathcal M) = \\frac{1}{|S_f|} \\sum_{s \\in S_f} x_s(\\theta),

so the operator is the matrix :math:`A` with :math:`A_{fs} = 1/|S_f|` for
:math:`s \\in S_f`, applied alike to the observed and the modeled series:
the model is sampled at exactly the steps the data are (a matched mask).

A step is **usable** when all three of its hours are present and at least
``MINIMUM_MEASURED_HOURS`` of them were measured (flag 0). Steps are
classified by the sun (``data``): **day** when the sun is above the horizon
for the whole step, **night** when below it for the whole step, and
**transition** otherwise. Calendar groupings use local standard time
(UTC-5) at the step's start.

Operators (:func:`build`):

``twelve_hour``
    The experiment's current operator, for reference: 00-12 and 12-24 UTC
    windows of each day, gap-filled values allowed, kept when half the hours
    were measured.
``daily_day_night``
    Each day's mean over its usable day steps, and each night's over its
    usable night steps (the night of the evening it starts).
``halfmonth_day_night``, ``monthly_day_night``
    The same means over each half month, or month, of each year.
``monthly_diurnal``
    For each month of each year, the mean of each of the eight three-hour
    bins of the local day: the month's mean diurnal cycle.
``climatological_diurnal``
    The same pooled over the years: the mean diurnal cycle of each calendar
    month.

A feature needs at least ``minimum_steps`` usable steps.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["MINIMUM_MEASURED_HOURS", "OPERATOR_NAMES", "Operator", "build", "step_classes"]

#: The fewest measured hours of three that make a step usable.
MINIMUM_MEASURED_HOURS = 2

#: The operators :func:`build` knows.
OPERATOR_NAMES = (
    "twelve_hour",
    "daily_day_night",
    "halfmonth_day_night",
    "monthly_day_night",
    "monthly_diurnal",
    "climatological_diurnal",
)

#: The default fewest usable steps per feature, by operator.
_MINIMUM_STEPS = {
    "twelve_hour": 4,
    "daily_day_night": 1,
    "halfmonth_day_night": 4,
    "monthly_day_night": 8,
    "monthly_diurnal": 5,
    "climatological_diurnal": 10,
}


@dataclass(frozen=True)
class Operator:
    """One operator: its matrix ``A`` ``(n, S)``, and a table describing its features."""

    name: str
    matrix: sp.csr_matrix
    features: pd.DataFrame

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    def apply(self, series) -> np.ndarray:
        """``A x`` for a series ``(S,)`` or a batch ``(J, S)``, NaN steps outside every feature allowed."""
        series = np.asarray(series)
        filled = np.nan_to_num(series)
        return (self.matrix @ filled.T).T


def step_classes(steps: pd.DataFrame) -> pd.Series:
    """``day``, ``night`` or ``transition`` per step, by the sun."""
    fraction = steps["daylight_fraction"]
    return pd.Series(
        np.where(fraction >= 1.0, "day", np.where(fraction <= 0.0, "night", "transition")),
        index=steps.index,
    )


def build(name: str, steps: pd.DataFrame, *, period=(2012, 2020), minimum_steps=None) -> Operator:
    """The operator *name* over the step table *steps* (``data.step_table``), within *period*."""
    minimum_steps = _MINIMUM_STEPS[name] if minimum_steps is None else minimum_steps
    table = _annotated(steps, period)
    if name == "twelve_hour":
        return _twelve_hour(table, period)
    usable = table[table["usable"]]
    if name.endswith("day_night"):
        usable = usable[usable["class"] != "transition"]
        resolution = name.removesuffix("_day_night")
        keys = {
            "daily": ["class", "night_date"],
            "halfmonth": ["class", "year", "month", "half"],
            "monthly": ["class", "year", "month"],
        }[resolution]
    elif name == "monthly_diurnal":
        keys = ["year", "month", "bin"]
    elif name == "climatological_diurnal":
        keys = ["month", "bin"]
    else:
        raise KeyError(f"unknown operator {name!r}; one of {OPERATOR_NAMES}")
    return _from_groups(name, usable, keys, minimum_steps, len(steps))


# ── helpers ──


def _annotated(steps: pd.DataFrame, period) -> pd.DataFrame:
    """*steps* with their position, class, calendar keys and usability."""
    first, last = period
    local_start = steps.index - pd.Timedelta(hours=8)  # step start (-3 h) in UTC-5
    table = steps.assign(
        position=np.arange(len(steps)),
        **{"class": step_classes(steps).to_numpy()},
        year=local_start.year,
        month=local_start.month,
        half=np.where(local_start.day <= 15, 1, 2),
        bin=local_start.hour,
        local_date=local_start.normalize(),
    )
    # A night belongs to the date of the evening it starts.
    table["night_date"] = table["local_date"] - pd.to_timedelta(
        (table["bin"] < 12).astype(int), unit="D"
    )
    table["night_date"] = table["night_date"].where(table["class"] == "night", table["local_date"])
    table["usable"] = (
        table["observed"].notna()
        & (table["measured_hours"] >= MINIMUM_MEASURED_HOURS)
        & (table["year"] >= first)
        & (table["year"] <= last)
    )
    return table


def _from_groups(name, usable, keys, minimum_steps, n_steps) -> Operator:
    """One feature per group of *usable* by *keys* with at least *minimum_steps* steps."""
    groups = usable.groupby(keys, sort=True)
    rows, columns, values, records = [], [], [], []
    for key, group in groups:
        if len(group) < minimum_steps:
            continue
        f = len(records)
        rows.extend([f] * len(group))
        columns.extend(group["position"].tolist())
        values.extend([1.0 / len(group)] * len(group))
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        record.update(
            n_steps=len(group),
            time=group.index[len(group) // 2],
            first_step=group.index[0],
            last_step=group.index[-1],
            day_of_year=float(np.mean(group.index.dayofyear)),
            daylight_fraction=float(group["daylight_fraction"].mean()),
        )
        records.append(record)
    features = pd.DataFrame(records)
    if "class" not in features:
        features["class"] = np.where(
            features["daylight_fraction"] >= 0.999,
            "day",
            np.where(features["daylight_fraction"] <= 0.001, "night", "transition"),
        )
    matrix = sp.csr_matrix((values, (rows, columns)), shape=(len(records), n_steps))
    return Operator(name, matrix, features)


def _twelve_hour(table, period) -> Operator:
    """The current 00-12 / 12-24 UTC windows, gap-filled values, half the hours measured."""
    utc_start = table.index - pd.Timedelta(hours=3)
    window = pd.DataFrame(
        {
            "class": np.where(utc_start.hour < 12, "night", "day"),
            "utc_date": utc_start.normalize(),
            "position": table["position"].to_numpy(),
            "observed": table["observed"].to_numpy(),
            "measured_hours": table["measured_hours"].to_numpy(),
            "daylight_fraction": table["daylight_fraction"].to_numpy(),
            "utc_year": utc_start.year,
        },
        index=table.index,
    )
    first, last = period
    window = window[(window["utc_year"] >= first) & (window["utc_year"] <= last)]
    rows, columns, values, records = [], [], [], []
    for (klass, date), group in window.groupby(["class", "utc_date"], sort=True):
        if len(group) != 4 or group["observed"].isna().any():
            continue
        if group["measured_hours"].sum() < 6:
            continue
        f = len(records)
        rows.extend([f] * 4)
        columns.extend(group["position"].tolist())
        values.extend([0.25] * 4)
        records.append(
            {
                "class": klass,
                "utc_date": date,
                "n_steps": 4,
                "time": group.index[-1],
                "day_of_year": float(group.index[-1].dayofyear),
                "year": group.index[-1].year,
                "month": group.index[-1].month,
                "daylight_fraction": float(group["daylight_fraction"].mean()),
            }
        )
    matrix = sp.csr_matrix((values, (rows, columns)), shape=(len(records), len(table)))
    return Operator("twelve_hour", matrix, pd.DataFrame(records))
