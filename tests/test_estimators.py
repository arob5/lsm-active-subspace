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
from lsmas.inference import eki_history
from lsmas.problems import load_inverse_problem
from lsmas.tools import (
    covariance_subspace,
    eki_linearization_subspace,
    exact_subspace,
    finite_difference_subspace,
    information_matrix,
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


def linear_model_draws(theta, scales=None):
    """Draws of the linear model's parameters (a, b), and of its noise scales if given."""
    theta = np.asarray(theta, dtype=float)
    fields = {"theta": {"a": theta[:, 0], "b": theta[:, 1]}}
    if scales is not None:
        fields["s2"] = {name: np.full(len(theta), value) for name, value in scales.items()}
    return NumericRecordBatch(fields, "draw")


def random_linear_model_draws(n_draws, seed):
    return linear_model_draws(np.random.default_rng(seed).normal(size=(n_draws, 2)))


def test_statistical_linearization_of_a_linear_map_is_its_matrix():
    z = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(statistical_linearization(z, z @ FORWARD_MATRIX.T), FORWARD_MATRIX)


def test_jacobians_of_a_linear_map_in_whitened_coordinates(problem):
    # theta is unconstrained here, so G(z) = F (m + L z) and DG(z) = F L at every point.
    expected = FORWARD_MATRIX @ problem.coordinates.whitening_factor
    batch = jacobians(problem, random_linear_model_draws(5, seed=2)["theta"])
    assert batch.shape == (5, 6, 2)
    np.testing.assert_allclose(batch, np.broadcast_to(expected, batch.shape), rtol=EXACT)
    one_setting = jacobians(problem, {"a": 0.3, "b": -1.0})
    np.testing.assert_allclose(one_setting, expected, rtol=EXACT)


def test_exact_subspace_matches_the_linear_information(problem, exact_information):
    subspace = exact_subspace(problem, random_linear_model_draws(30, seed=1))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == "exact"


def test_a_distribution_without_draws_is_sampled(problem, exact_information):
    with workflow_run(seed=4):
        subspace = exact_subspace(problem, problem.prior, num_samples=20)
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    with pytest.raises(ValueError, match="not held as draws; give num_samples"):
        exact_subspace(problem, problem.prior)


def test_draws_held_by_a_distribution_are_averaged_with_their_weights(problem):
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
    weighted_subspace = exact_subspace(
        linear_inferred, EmpiricalDistribution(draws, weights=np.array([3.0, 1.0]))
    )
    factor = linear_inferred.coordinates.whitening_factor
    expected = 0.75 * exact_linear_information(
        {"day": 1.0, "night": 1.0}, LEARNING_RATE, factor
    ) + 0.25 * exact_linear_information({"day": 4.0, "night": 4.0}, LEARNING_RATE, factor)
    np.testing.assert_allclose(weighted_subspace.information, expected, rtol=EXACT)


def test_noise_scales_come_from_the_draws_or_are_given(problem, exact_information):
    inferred = linear_problem(
        problem.data["y"], {"day": "inferred", "night": "inferred"}, LEARNING_RATE
    )
    theta = np.random.default_rng(5).normal(size=(10, 2))
    subspace_with_drawn_scales = exact_subspace(inferred, linear_model_draws(theta, scales=SCALES))
    subspace_with_given_scales = exact_subspace(
        inferred, linear_model_draws(theta), noise_scales=SCALES
    )
    np.testing.assert_allclose(
        subspace_with_drawn_scales.information, exact_information, rtol=EXACT
    )
    np.testing.assert_allclose(
        subspace_with_given_scales.information, exact_information, rtol=EXACT
    )
    with pytest.raises(ValueError, match="give values for \\['day', 'night'\\] as noise_scales"):
        exact_subspace(inferred, linear_model_draws(theta))


def test_finite_differences_match_the_linear_information(problem, exact_information):
    subspace = finite_difference_subspace(problem, random_linear_model_draws(3, seed=2))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)


def test_a_linearization_of_a_linear_model_is_its_information(problem, exact_information):
    subspace = linearization_subspace(problem, random_linear_model_draws(20, seed=6))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == "statistical_linearization"
    unequally_weighted = EmpiricalDistribution(
        random_linear_model_draws(4, seed=7), weights=np.array([1.0, 1.0, 1.0, 2.0])
    )
    with pytest.raises(ValueError, match="equally weighted draws; give num_samples"):
        linearization_subspace(problem, unequally_weighted)
    with workflow_run(seed=8):
        resampled_subspace = linearization_subspace(problem, unequally_weighted, num_samples=50)
    np.testing.assert_allclose(resampled_subspace.information, exact_information, rtol=EXACT)


@pytest.fixture(scope="module")
def linear_eki_posterior():
    observations = np.array([1.2, -2.0, 1.9, -0.4, -1.3, -2.2])
    with workflow_run(seed=0):
        return linear_problem(observations, SCALES, LEARNING_RATE).condition(
            method="enskit_eki", method_options={"n_particles": 50}
        )


@pytest.mark.parametrize("weighting", ["increment", "uniform", "last"])
def test_eki_linearization_of_a_linear_model_is_its_information(
    problem, exact_information, linear_eki_posterior, weighting
):
    subspace = eki_linearization_subspace(problem, linear_eki_posterior, weighting=weighting)
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == f"eki_linearization_{weighting}"


def test_information_matrix_of_one_jacobian_and_of_a_weighted_batch():
    jacobian = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    noise_variance = np.array([1.0, 2.0, 4.0])
    one_jacobian_information = information_matrix(jacobian, noise_variance)
    np.testing.assert_allclose(
        one_jacobian_information, jacobian.T @ np.diag(1 / noise_variance) @ jacobian
    )
    batch = information_matrix(
        np.stack([jacobian, 2 * jacobian]), np.stack([noise_variance, noise_variance]), [0.25, 0.75]
    )
    np.testing.assert_allclose(batch, (0.25 + 0.75 * 4) * one_jacobian_information)
    np.testing.assert_allclose(
        information_matrix(np.stack([jacobian, jacobian]), noise_variance), one_jacobian_information
    )


def test_covariance_subspace_recovers_the_linear_information(
    problem, exact_information, observations
):
    mean, covariance = exact_linear_posterior(observations, SCALES, LEARNING_RATE)
    theta = np.random.default_rng(3).multivariate_normal(mean, covariance, size=200_000)
    subspace = covariance_subspace(problem, EmpiricalDistribution(linear_model_draws(theta)))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=0.05)
    with pytest.raises(ValueError, match="more draws than parameters"):
        covariance_subspace(problem, random_linear_model_draws(2, seed=9))


def test_an_unknown_weighting_is_refused(problem, linear_eki_posterior):
    with pytest.raises(ValueError, match="unknown weighting"):
        eki_linearization_subspace(problem, linear_eki_posterior, weighting="geometric")
    with pytest.raises(ValueError, match="not the result of an EKI run"):
        eki_linearization_subspace(
            problem, EmpiricalDistribution(random_linear_model_draws(5, seed=1))
        )


@pytest.mark.smoke
def test_eki_directions_capture_the_exact_information_along_the_run_on_the_toy():
    toy = load_inverse_problem("toy")
    with workflow_run(seed=1):
        eki_posterior = toy.condition(method="enskit_eki", method_options={"n_particles": 100})
    history = eki_history(eki_posterior)
    particles_at_steps = NumericRecordBatch.from_vector(
        toy.coordinates.spec, history["theta"].values, level_names=["step", "member"]
    )
    exact_information = sum(
        increment * exact_subspace(toy, particles_at_steps[step]).information
        for step, increment in enumerate(history["increment"].values)
    )
    basis = eki_linearization_subspace(toy, eki_posterior, rank=3).basis
    largest_possible = np.sort(np.linalg.eigvalsh(exact_information))[::-1][:3].sum()
    assert np.trace(basis.T @ exact_information @ basis) / largest_possible > 0.95


def test_exact_jacobians_need_a_differentiable_forward_map():
    small = load_inverse_problem("harvard_forest_small")
    with pytest.raises(ValueError, match="cannot be differentiated"):
        jacobians(small, quantile(small.prior, 0.5)["theta"])


def test_inferred_scales_need_values_for_the_noise_covariance():
    toy = load_inverse_problem("toy", scales="inferred")
    with pytest.raises(ValueError, match="give values for \\['day', 'night'\\]"):
        toy.noise_variance()
    assert toy.noise_variance({"day": 1.0, "night": 1.0}).shape == (400,)
