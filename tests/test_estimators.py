import numpy as np
import pytest
from probpipe import EmpiricalDistribution, NumericRecordBatch, quantile, workflow_run

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
    jacobians,
    linearization_subspace,
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


def draws_of(theta, scales=None):
    """Draws of the linear model's parameters (a, b), and of its noise scales if given."""
    theta = np.asarray(theta, dtype=float)
    fields = {"theta": {"a": theta[:, 0], "b": theta[:, 1]}}
    if scales is not None:
        fields["s2"] = {name: np.full(len(theta), value) for name, value in scales.items()}
    return NumericRecordBatch(fields, "draw")


def some_draws(count, seed):
    return draws_of(np.random.default_rng(seed).normal(size=(count, 2)))


def test_statistical_linearization_of_a_linear_map_is_its_matrix():
    z = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(statistical_linearization(z, z @ FORWARD_MATRIX.T), FORWARD_MATRIX)


def test_jacobians_of_a_linear_map_in_whitened_coordinates(problem):
    # theta is unconstrained here, so G(z) = F (m + L z) and DG(z) = F L at every point.
    expected = FORWARD_MATRIX @ problem.coordinates.whitening_factor
    batch = jacobians(problem, some_draws(5, seed=2))
    assert batch.shape == (5, 6, 2)
    np.testing.assert_allclose(batch, np.broadcast_to(expected, batch.shape), rtol=EXACT)
    one_setting = jacobians(problem, {"a": 0.3, "b": -1.0})
    np.testing.assert_allclose(one_setting, expected, rtol=EXACT)


def test_exact_subspace_matches_the_linear_information(problem, exact_information):
    subspace = exact_subspace(problem, some_draws(30, seed=1))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == "exact"


def test_a_distribution_without_draws_is_sampled(problem, exact_information):
    with workflow_run(seed=4):
        subspace = exact_subspace(problem, problem.prior, num_samples=20)
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    with pytest.raises(ValueError, match="not held as draws; give num_samples"):
        exact_subspace(problem, problem.prior)


def test_held_draws_are_averaged_with_their_weights(problem):
    # Two draws whose noise scales differ: the weights set how much each one's Gamma counts.
    linear_inferred = linear_problem(
        problem.data["y"], {"day": "inferred", "night": "inferred"}, LEARNING_RATE
    )
    draws = NumericRecordBatch(
        {
            "theta": {"a": np.zeros(2), "b": np.zeros(2)},
            "s2": {"day": np.array([1.0, 4.0]), "night": np.array([1.0, 4.0])},
        },
        "draw",
    )
    weighted = exact_subspace(
        linear_inferred, EmpiricalDistribution(draws, weights=np.array([3.0, 1.0]))
    )
    factor = linear_inferred.coordinates.whitening_factor
    expected = 0.75 * exact_linear_information(
        {"day": 1.0, "night": 1.0}, LEARNING_RATE, factor
    ) + 0.25 * exact_linear_information({"day": 4.0, "night": 4.0}, LEARNING_RATE, factor)
    np.testing.assert_allclose(weighted.information, expected, rtol=EXACT)


def test_noise_scales_come_from_the_draws_or_are_given(problem, exact_information):
    inferred = linear_problem(
        problem.data["y"], {"day": "inferred", "night": "inferred"}, LEARNING_RATE
    )
    theta = np.random.default_rng(5).normal(size=(10, 2))
    from_draws = exact_subspace(inferred, draws_of(theta, scales=SCALES))
    given = exact_subspace(inferred, draws_of(theta), noise_scales=SCALES)
    np.testing.assert_allclose(from_draws.information, exact_information, rtol=EXACT)
    np.testing.assert_allclose(given.information, exact_information, rtol=EXACT)
    with pytest.raises(ValueError, match="give values for \\['day', 'night'\\] as noise_scales"):
        exact_subspace(inferred, draws_of(theta))


def test_finite_differences_match_the_linear_information(problem, exact_information):
    subspace = finite_difference_subspace(problem, some_draws(3, seed=2))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)


def test_a_linearization_of_a_linear_model_is_its_information(problem, exact_information):
    subspace = linearization_subspace(problem, some_draws(20, seed=6))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == "statistical_linearization"
    unequal = EmpiricalDistribution(some_draws(4, seed=7), weights=np.array([1.0, 1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="equally weighted draws; give num_samples"):
        linearization_subspace(problem, unequal)
    with workflow_run(seed=8):
        resampled = linearization_subspace(problem, unequal, num_samples=50)
    np.testing.assert_allclose(resampled.information, exact_information, rtol=EXACT)


@pytest.mark.parametrize("weighting", ["increment", "uniform", "last"])
def test_ladder_linearization_of_a_linear_model_is_its_information(
    problem, exact_information, weighting
):
    ladder = LadderLinearization(problem, weighting=weighting)
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
    subspace = covariance_subspace(problem, EmpiricalDistribution(draws_of(theta)))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=0.05)
    with pytest.raises(ValueError, match="more draws than parameters"):
        covariance_subspace(problem, some_draws(2, seed=9))


def test_an_unknown_weighting_is_refused(problem):
    with pytest.raises(ValueError, match="unknown weighting"):
        LadderLinearization(problem, weighting="geometric")


@pytest.mark.smoke
def test_ladder_directions_capture_the_exact_ladder_information_on_the_toy():
    toy = load_inverse_problem("toy")
    ladder = LadderLinearization(toy)
    records = []
    with workflow_run(seed=1):
        toy.condition(
            method="enskit_eki",
            method_options={"n_particles": 100, "callbacks": [ladder, records.append]},
        )

    def particles(record):
        return NumericRecordBatch.from_vector(
            toy.coordinates.spec, record.theta, level_names="member"
        )

    exact = sum(
        record.increment * exact_subspace(toy, particles(record)).information for record in records
    )
    basis = ladder.subspace(rank=3).basis
    best = np.sort(np.linalg.eigvalsh(exact))[::-1][:3].sum()
    assert np.trace(basis.T @ exact @ basis) / best > 0.95


def test_exact_jacobians_need_a_differentiable_forward_map():
    small = load_inverse_problem("harvard_forest_small")
    with pytest.raises(ValueError, match="cannot be differentiated"):
        jacobians(small, quantile(small.prior, 0.5)["theta"])


def test_inferred_scales_need_values_for_the_noise_covariance():
    toy = load_inverse_problem("toy", scales="inferred")
    with pytest.raises(ValueError, match="give values for \\['day', 'night'\\]"):
        toy.noise_variance()
    assert toy.noise_variance({"day": 1.0, "night": 1.0}).shape == (400,)
