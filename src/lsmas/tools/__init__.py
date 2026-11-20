"""Building blocks for parameter-reduction algorithms: subspaces, their estimators, and reduced
problems."""

from lsmas.tools.estimators import (
    Weighting,
    covariance_subspace,
    eki_linearization_subspace,
    exact_subspace,
    finite_difference_subspace,
    information_matrix,
    jacobians,
    linearization_subspace,
    statistical_linearization,
)
from lsmas.tools.reduced import ReducedProblem, reduced_problem
from lsmas.tools.subspace import Subspace, principal_angles, suggest_rank

__all__ = [
    "ReducedProblem",
    "Subspace",
    "Weighting",
    "covariance_subspace",
    "eki_linearization_subspace",
    "exact_subspace",
    "finite_difference_subspace",
    "information_matrix",
    "jacobians",
    "linearization_subspace",
    "principal_angles",
    "reduced_problem",
    "statistical_linearization",
    "suggest_rank",
]
