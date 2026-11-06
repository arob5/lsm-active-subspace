"""Inverse problems, ready to condition.

>>> problem = load("harvard_forest_small")        # doctest: +SKIP
>>> print(problem.describe())                      # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lsmas.problems.harvard_forest import (
    SMALL_PROBLEM_PARAMETERS,
    harvard_forest,
    harvard_forest_small,
)
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.problems.toy import toy

__all__ = [
    "PROBLEMS",
    "SMALL_PROBLEM_PARAMETERS",
    "InverseProblem",
    "harvard_forest",
    "harvard_forest_small",
    "load",
    "toy",
]

#: Every available problem, by name.
PROBLEMS: dict[str, Callable[..., InverseProblem]] = {
    "harvard_forest": harvard_forest,
    "harvard_forest_small": harvard_forest_small,
    "toy": toy,
}


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
