import numpy as np
import pytest

from lsmas.tools import Subspace, principal_angles, suggest_rank

NAMES = ("a", "b", "c", "d")


def information_with(eigenvalues, seed=0):
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(4, 4)))
    return rotation @ np.diag(eigenvalues) @ rotation.T, rotation


def test_rank_keeps_the_discarded_information_below_the_bound():
    assert suggest_rank([5.0, 2.0, 0.3, 0.1]) == 2
    assert suggest_rank([5.0, 2.0, 0.3, 0.1], discarded_information=0.1) == 3
    assert suggest_rank([5.0, 2.0, 0.3, 0.1], discarded_information=10.0) == 1


def test_subspace_recovers_eigenvalues_and_spans_the_leading_eigenvectors():
    information, rotation = information_with([0.1, 5.0, 0.3, 2.0])
    subspace = Subspace.from_information(information, parameter_names=NAMES, estimator="exact")
    np.testing.assert_allclose(subspace.eigenvalues, [5.0, 2.0, 0.3, 0.1])
    assert subspace.rank == 2
    leading = rotation[:, [1, 3]]
    np.testing.assert_allclose(subspace.basis @ subspace.basis.T, leading @ leading.T, atol=1e-12)


def test_active_and_inactive_variables_recompose_the_whitened_coordinates():
    information, _ = information_with([5.0, 2.0, 0.3, 0.1])
    subspace = Subspace.from_information(information, parameter_names=NAMES, estimator="exact")
    z = np.random.default_rng(1).normal(size=(7, 4))
    recomposed = subspace.whitened_from_active_and_inactive(
        subspace.active_variables(z), subspace.inactive_variables(z)
    )
    np.testing.assert_allclose(recomposed, z, atol=1e-12)


def test_full_rank_activity_scores_are_the_diagonal():
    information, _ = information_with([5.0, 2.0, 0.3, 0.1])
    subspace = Subspace.from_information(
        information, parameter_names=NAMES, estimator="exact", rank=4
    )
    scores = subspace.activity_scores()
    np.testing.assert_allclose(scores[list(NAMES)], np.diag(information))
    assert list(scores) == sorted(scores, reverse=True)


def test_loadings_name_parameters_and_directions():
    information, _ = information_with([5.0, 2.0, 0.3, 0.1])
    loadings = Subspace.from_information(
        information, parameter_names=NAMES, estimator="exact"
    ).loadings()
    assert list(loadings.index) == list(NAMES)
    assert list(loadings.columns) == ["direction_1", "direction_2"]


def test_each_eigenvector_points_toward_its_largest_entry():
    information, _ = information_with([5.0, 2.0, 0.3, 0.1])
    eigenvectors = Subspace.from_information(
        information, parameter_names=NAMES, estimator="exact"
    ).eigenvectors
    largest = np.abs(eigenvectors).argmax(axis=0)
    assert np.all(eigenvectors[largest, range(4)] > 0)


def test_rank_must_lie_between_one_and_the_dimension():
    information, _ = information_with([5.0, 2.0, 0.3, 0.1])
    subspace = Subspace.from_information(information, parameter_names=NAMES, estimator="exact")
    with pytest.raises(ValueError, match="between 1 and 4"):
        subspace.with_rank(0)
    assert subspace.with_rank(3).basis.shape == (4, 3)


def test_principal_angles_between_spans():
    identity = np.eye(4)
    np.testing.assert_allclose(principal_angles(identity[:, :2], identity[:, [1, 0]]), [0.0, 0.0])
    np.testing.assert_allclose(principal_angles(identity[:, :2], identity[:, 2:]), [90.0, 90.0])
    tilted = np.array([[np.cos(0.3)], [np.sin(0.3)], [0.0], [0.0]])
    np.testing.assert_allclose(principal_angles(identity[:, :2], tilted), [0.0], atol=1e-6)
    np.testing.assert_allclose(principal_angles(identity[:, :1], tilted), [np.degrees(0.3)])


def test_subspaces_compare_only_over_the_same_parameters():
    information, _ = information_with([5.0, 2.0, 0.3, 0.1])
    first = Subspace.from_information(information, parameter_names=NAMES, estimator="exact")
    np.testing.assert_allclose(first.principal_angles(first.with_rank(1)), [0.0], atol=1e-6)
    other = Subspace.from_information(information, parameter_names=tuple("wxyz"), estimator="x")
    with pytest.raises(ValueError, match="different parameters"):
        first.principal_angles(other)


def test_repr_names_the_estimator_rank_and_leading_eigenvalues():
    information, _ = information_with([5.0, 2.0, 0.3, 0.1])
    subspace = Subspace.from_information(information, parameter_names=NAMES, estimator="exact")
    assert repr(subspace) == "Subspace(estimator='exact', rank=2 of 4, leading eigenvalues [5, 2])"
