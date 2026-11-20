import jax.numpy as jnp
import numpy as np
import pytest
from probpipe import (
    EmpiricalDistribution,
    Function,
    NumericArraySpec,
    NumericRecordBatch,
    condition_on,
    mean,
    sample,
    workflow_run,
)

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    REFERENCE_VARIANCE,
    THETA_SPEC,
    exact_linear_information,
    exact_linear_posterior,
    linear_problem,
    linear_theta_prior,
)
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas.coordinates import inverse
from lsmas.inference import eki_history
from lsmas.problems import load_inverse_problem
from lsmas.tools import (
    covariance_subspace,
    eki_linearization_subspace,
    finite_difference_subspace,
    gradient_based_subspace,
    information_matrix,
    jacobian,
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


def linear_model_samples(theta, scales=None, weights=None):
    """Samples of the linear model's parameters (a, b), and of its noise scales if given."""
    theta = np.asarray(theta, dtype=float)
    fields = {"theta": {"a": theta[:, 0], "b": theta[:, 1]}}
    if scales is not None:
        fields["s2"] = {name: np.full(len(theta), value) for name, value in scales.items()}
    return EmpiricalDistribution(NumericRecordBatch(fields, "sample"), weights=weights)


def random_linear_model_samples(n_samples, seed):
    return linear_model_samples(np.random.default_rng(seed).normal(size=(n_samples, 2)))


def test_statistical_linearization_of_a_linear_map_is_its_matrix():
    z = np.random.default_rng(0).normal(size=(20, 2))
    joint = EmpiricalDistribution(
        NumericRecordBatch({"z": z, "predictions": z @ FORWARD_MATRIX.T}, "sample")
    )
    np.testing.assert_allclose(statistical_linearization(joint), FORWARD_MATRIX, rtol=EXACT)


def test_the_jacobian_of_the_mean_observations_in_whitened_coordinates(problem):
    # theta is unconstrained here, so G(z) = F (m + L z) and DG(z) = F L at every point.
    coordinates = problem.coordinates
    to_theta = inverse(coordinates.whiten)

    def mean_observations(z):
        return mean(condition_on(problem.model, {"theta": to_theta(z)}))

    z = coordinates.whiten({"a": 0.3, "b": -1.0})
    expected = FORWARD_MATRIX @ coordinates.whitening_factor
    np.testing.assert_allclose(jacobian(mean_observations, z), expected, rtol=EXACT)
    by_differences = jacobian(mean_observations, z, method="central_differences", step=0.5)
    np.testing.assert_allclose(by_differences, expected, rtol=EXACT)


def test_gradient_based_subspace_matches_the_linear_information(problem, exact_information):
    subspace = gradient_based_subspace(
        problem.model, random_linear_model_samples(30, seed=1), learning_rate=LEARNING_RATE
    )
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == "gradient"


def test_a_distribution_not_held_as_samples_is_sampled(problem, exact_information):
    with workflow_run(seed=4):
        subspace = gradient_based_subspace(
            problem.model, problem.prior, learning_rate=LEARNING_RATE, num_samples=20
        )
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)


def test_the_default_learning_rate_gives_the_ordinary_posteriors_information(problem):
    subspace = gradient_based_subspace(problem.model, random_linear_model_samples(5, seed=3))
    expected = exact_linear_information(SCALES, 1.0, problem.coordinates.whitening_factor)
    np.testing.assert_allclose(subspace.information, expected, rtol=EXACT)


def test_samples_are_averaged_with_their_weights_and_their_own_noise_scales(problem):
    inferred = linear_problem(problem.data["y"], "inferred", LEARNING_RATE)
    parameters = NumericRecordBatch(
        {
            "theta": {"a": np.zeros(2), "b": np.zeros(2)},
            "s2": {"day": np.array([1.0, 4.0]), "night": np.array([1.0, 4.0])},
        },
        "sample",
    )
    samples = EmpiricalDistribution(parameters, weights=np.array([3.0, 1.0]))
    subspace = gradient_based_subspace(inferred.model, samples, learning_rate=LEARNING_RATE)
    factor = inferred.coordinates.whitening_factor
    expected = 0.75 * exact_linear_information(
        {"day": 1.0, "night": 1.0}, LEARNING_RATE, factor
    ) + 0.25 * exact_linear_information({"day": 4.0, "night": 4.0}, LEARNING_RATE, factor)
    np.testing.assert_allclose(subspace.information, expected, rtol=EXACT)


def test_finite_differences_match_the_linear_information(problem, exact_information):
    subspace = finite_difference_subspace(
        problem.model, random_linear_model_samples(3, seed=2), learning_rate=LEARNING_RATE
    )
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)


def test_a_linearization_of_a_linear_model_is_its_information(problem, exact_information):
    subspace = linearization_subspace(
        problem.model, random_linear_model_samples(20, seed=6), learning_rate=LEARNING_RATE
    )
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == "statistical_linearization"
    unequally_weighted = linear_model_samples(
        np.random.default_rng(7).normal(size=(4, 2)), weights=np.array([1.0, 1.0, 1.0, 2.0])
    )
    weighted_subspace = linearization_subspace(
        problem.model, unequally_weighted, learning_rate=LEARNING_RATE
    )
    np.testing.assert_allclose(weighted_subspace.information, exact_information, rtol=EXACT)


def failing_linear_forward(theta):
    """The linear forward map, failing (NaN) wherever a > 1.5."""
    a, b = float(np.asarray(theta["a"])), float(np.asarray(theta["b"]))
    return np.full(6, np.nan) if a > 1.5 else FORWARD_MATRIX @ np.array([a, b])


def test_estimators_without_gradients_raise_when_the_forward_map_fails():
    forward = Function(
        failing_linear_forward,
        label="failing_linear",
        input_spec={"theta": THETA_SPEC},
        output_spec=NumericArraySpec((6,), jnp.float64),
        traceable=False,
    )
    likelihood = GaussianSimulatorLikelihood(
        forward,
        output_name="y",
        reference_variance=REFERENCE_VARIANCE,
        classes=CLASSES,
        default_scales=SCALES,
    )
    model = likelihood & linear_theta_prior()
    samples = linear_model_samples([[0.0, 0.0], [3.0, 0.0], [-1.0, 1.0]])
    with pytest.raises(ValueError, match="forward map failed near at least one sample"):
        finite_difference_subspace(model, samples)
    with pytest.raises(ValueError, match="forward map failed at at least one sample"):
        linearization_subspace(model, samples)


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
    subspace = eki_linearization_subspace(problem.model, linear_eki_posterior, weighting=weighting)
    np.testing.assert_allclose(subspace.information, exact_information, rtol=EXACT)
    assert subspace.estimator == f"eki_linearization_{weighting}"


def test_information_matrix_of_one_jacobian():
    jacobian_matrix = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    noise_variance = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        information_matrix(jacobian_matrix, noise_variance),
        jacobian_matrix.T @ np.diag(1 / noise_variance) @ jacobian_matrix,
    )


def test_covariance_subspace_recovers_the_linear_information(
    problem, exact_information, observations
):
    posterior_mean, covariance = exact_linear_posterior(observations, SCALES, LEARNING_RATE)
    theta = np.random.default_rng(3).multivariate_normal(posterior_mean, covariance, size=200_000)
    subspace = covariance_subspace(problem.model, linear_model_samples(theta))
    np.testing.assert_allclose(subspace.information, exact_information, rtol=0.05)
    with pytest.raises(ValueError, match="more samples than parameters"):
        covariance_subspace(problem.model, random_linear_model_samples(2, seed=9))


def test_an_unknown_weighting_is_refused(problem, linear_eki_posterior):
    with pytest.raises(ValueError, match="unknown weighting"):
        eki_linearization_subspace(problem.model, linear_eki_posterior, weighting="geometric")
    with pytest.raises(ValueError, match="not the result of an EKI run"):
        eki_linearization_subspace(problem.model, random_linear_model_samples(5, seed=1))


@pytest.mark.smoke
def test_eki_directions_capture_the_exact_information_along_the_run_on_the_toy():
    toy = load_inverse_problem("toy")
    learning_rate = toy.learning_rate["nee"]
    with workflow_run(seed=1):
        eki_posterior = toy.condition(method="enskit_eki", method_options={"n_particles": 100})
    history = eki_history(eki_posterior)
    names = toy.coordinates.names

    def particles(step):
        theta = history["theta"].values[step]
        return EmpiricalDistribution(
            NumericRecordBatch(
                {"theta": {name: theta[:, j] for j, name in enumerate(names)}}, "member"
            )
        )

    exact_information = sum(
        increment
        * gradient_based_subspace(
            toy.model, particles(step), learning_rate=learning_rate
        ).information
        for step, increment in enumerate(history["increment"].values)
    )
    basis = eki_linearization_subspace(toy.model, eki_posterior, rank=3).basis
    largest_possible = np.sort(np.linalg.eigvalsh(exact_information))[::-1][:3].sum()
    assert np.trace(basis.T @ exact_information @ basis) / largest_possible > 0.95


def test_exact_jacobians_need_a_differentiable_forward_map():
    small = load_inverse_problem("harvard_forest_small")
    with workflow_run(seed=0):
        samples = EmpiricalDistribution(sample(small.prior, 2))
    with pytest.raises(Exception, match="gradients cannot pass through"):
        gradient_based_subspace(small.model, samples)
