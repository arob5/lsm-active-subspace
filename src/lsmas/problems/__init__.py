"""Inverse problems, ready to condition.

>>> problem = load("harvard_forest_dev")          # doctest: +SKIP
>>> print(problem.describe())                      # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lsmas.problems.harvard_forest_dev import harvard_forest_dev
from lsmas.problems.inverse_problem import InverseProblem

__all__ = ["PROBLEMS", "InverseProblem", "harvard_forest_dev", "load"]

#: Every available problem, by name.
PROBLEMS: dict[str, Callable[..., InverseProblem]] = {"harvard_forest_dev": harvard_forest_dev}


def load(name: str, **options: Any) -> InverseProblem:
    """The problem called *name*, built with *options*.

    Parameters
    ----------
    name : str
        One of :data:`PROBLEMS`.
    **options
        Passed to the problem's constructor, such as ``scales="inferred"``.
    """
    if name not in PROBLEMS:
        raise KeyError(f"no problem called {name!r}; available: {sorted(PROBLEMS)}")
    return PROBLEMS[name](**options)
