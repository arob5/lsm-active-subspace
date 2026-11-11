import numpy as np
import pytest
from probpipe import NumericRecordBatch, workflow_run

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    exact_linear_information,
    exact_linear_posterior,
    linear_problem,
)
from lsmas.problems import load_inverse_problem
from lsmas.tools import (
    LadderLinearization,
    covariance_subspace,
    exact_subspace,
    finite_difference_subspace,
    statistical_linearization,
)

SCALES = {"day": 1.5, "night": 0.7}
LEARNING_RATE = np.where(CLASSES == "day", 0.5, 0.8)

#: Exact up to floating point: each estimate is exact for a linear forward map.
EXACT = 1e-6


@pytest.fixture
def problem(observations):
    return linear_problem(observations, SCALES, LEARNING_RATE)


@pytest.fixture
def exact_information(problem):
    return exact_linear_information(SCALES, LEARNING_RATE, problem.coordinates.whitening_factor)


def test_statistical_linearization_of_a_linear_map_is_its_matrix():
    xi = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(statistical_linearization(xi, xi @ FORWARD_MATRIX.T), FORWARD_MATRIX)


def test_exact_subspace_matches_the_linear_information(problem, exact_information):
    xi = np.random.default_rng(1).normal(size=(30, 2))
    subspace = exact_subspace(problem, xi)
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == "exact"


def test_finite_differences_match_the_linear_information(problem, exact_information):
    xi = np.random.default_rng(2).normal(size=(3, 2))
    subspace = finite_difference_subspace(problem, xi)
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)


@pytest.mark.parametrize("weighting", ["increment", "uniform", "last"])
def test_ladder_linearization_of_a_linear_model_is_its_information(
    problem, exact_information, weighting
):
    ladder = LadderLinearization(problem.coordinates, weighting=weighting)
    with workflow_run(seed=0):
        problem.condition(
            method="enskit_eki", method_options={"n_particles": 50, "callbacks": [ladder]}
        )
    assert len(ladder.steps) > 1
    assert sum(step.increment for step in ladder.steps) == pytest.approx(1.0)
    np.testing.assert_allclose(ladder.information, exact_information, rtol=EXACT)
    assert ladder.subspace().estimator == f"ladder_linearization_{weighting}"


def test_covariance_subspace_recovers_the_linear_information(
    problem, exact_information, observations
):
    mean, covariance = exact_linear_posterior(observations, SCALES, LEARNING_RATE)
    theta = np.random.default_rng(3).multivariate_normal(mean, covariance, size=200_000)
    subspace = covariance_subspace(problem, theta)
    np.testing.assert_allclose(subspace.information, exact_information, rtol=0.05)


def test_an_unknown_weighting_is_refused(problem):
    with pytest.raises(ValueError, match="unknown weighting"):
        LadderLinearization(problem.coordinates, weighting="geometric")


@pytest.mark.smoke
def test_ladder_directions_capture_the_exact_ladder_information_on_the_toy():
    toy = load_inverse_problem("toy")
    ladder = LadderLinearization(toy.coordinates)
    records = []
    with workflow_run(seed=1):
        toy.condition(
            method="enskit_eki",
            method_options={"n_particles": 100, "callbacks": [ladder, records.append]},
        )

    def whitened(theta):
        records = NumericRecordBatch.from_vector(toy.coordinates.spec, theta, level_names="member")
        return np.asarray(toy.coordinates.whiten(records).to_vector())

    exact = sum(
        record.increment * exact_subspace(toy, whitened(record.theta)).information
        for record in records
    )
    basis = ladder.subspace(rank=3).basis
    best = np.sort(np.linalg.eigvalsh(exact))[::-1][:3].sum()
    assert np.trace(basis.T @ exact @ basis) / best > 0.95


def test_exact_jacobians_need_a_differentiable_forward_map():
    with pytest.raises(ValueError, match="cannot be differentiated"):
        exact_subspace(load_inverse_problem("harvard_forest_small"), np.zeros((1, 10)))


def test_inferred_scales_need_values_for_the_noise_covariance():
    toy = load_inverse_problem("toy", scales="inferred")
    with pytest.raises(ValueError, match="give values for \\['day', 'night'\\]"):
        toy.noise_variance()
    assert toy.noise_variance({"day": 1.0, "night": 1.0}).shape == (400,)
