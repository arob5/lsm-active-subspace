import jax
import numpy as np
import pytest
from probpipe import workflow_run

from conftest import CLASSES, exact_linear_information, linear_problem, posterior_moments
from lsmas.problems import load_inverse_problem
from lsmas.tools import Subspace, reduced_problem

SCALES = {"day": 1.5, "night": 0.7}
LEARNING_RATE = np.where(CLASSES == "day", 0.5, 0.8)


@pytest.fixture
def problem(observations):
    return linear_problem(observations, SCALES, LEARNING_RATE)


def linear_subspace(problem, rank):
    information = exact_linear_information(
        SCALES, LEARNING_RATE, problem.coordinates.whitening_factor
    )
    return Subspace.from_information(
        information, parameter_names=problem.coordinates.names, estimator="exact", rank=rank
    )


def exact_plug_in_posterior(problem, subspace):
    r"""Mean and covariance of a | y for the linear model: precision I + Wᵀ H W, where the
    likelihood at ξ = W a is Gaussian in a."""
    coordinates = problem.coordinates
    forward_matrix = np.asarray(
        jax.jacfwd(lambda x: problem.forward.using(raw=True)(coordinates.parameter_record(x)))(
            np.zeros(2)
        )
    )
    offset = np.asarray(problem.forward.using(raw=True)(coordinates.parameter_record(np.zeros(2))))
    reduced_matrix = forward_matrix @ subspace.basis
    precision_weights = 1.0 / problem.noise_variance()
    precision = np.eye(subspace.rank) + reduced_matrix.T @ (
        precision_weights[:, None] * reduced_matrix
    )
    covariance = np.linalg.inv(precision)
    mean = covariance @ reduced_matrix.T @ (precision_weights * (problem.data["y"] - offset))
    return mean, covariance


@pytest.mark.parametrize("rank", [1, 2])
def test_reduced_posterior_is_the_plug_in_posterior(problem, rank):
    subspace = linear_subspace(problem, rank)
    reduced = reduced_problem(problem, subspace)
    assert reduced.coordinates.names == tuple(f"active_{i + 1}" for i in range(rank))
    with workflow_run(seed=0):
        posterior = reduced.condition(method="blackjax_nuts")
    mean, covariance = posterior_moments(posterior)
    exact_mean, exact_covariance = exact_plug_in_posterior(problem, subspace)
    sd = np.sqrt(np.diag(exact_covariance))
    np.testing.assert_allclose(mean, exact_mean, atol=0.1 * sd.max())
    np.testing.assert_allclose(
        np.atleast_2d(covariance), exact_covariance, rtol=0.15, atol=0.05 * sd.max() ** 2
    )


def test_lifted_draws_keep_the_inactive_variables_at_their_prior(problem):
    subspace = linear_subspace(problem, 1)
    reduced = reduced_problem(problem, subspace)
    with workflow_run(seed=1):
        posterior = reduced.condition(method="blackjax_nuts")
    lifted = reduced.lift(posterior, key=jax.random.key(0))
    theta = problem.coordinates.array_from_draws(lifted.atoms)
    xi = problem.coordinates.whitened_from_parameters(theta)
    inactive = subspace.inactive_variables(xi)
    assert inactive.shape == (theta.shape[0], 1)
    assert np.std(inactive) == pytest.approx(1.0, abs=0.1)
    active_lifted = subspace.active_variables(xi)[:, 0]
    active_drawn = reduced.coordinates.array_from_draws(posterior.atoms).ravel()
    np.testing.assert_allclose(np.sort(active_lifted), np.sort(active_drawn), atol=1e-8)


def test_subspace_of_another_problem_is_refused(problem):
    other = Subspace.from_information(
        np.eye(2), parameter_names=("x", "y"), estimator="exact", rank=1
    )
    with pytest.raises(ValueError, match="not the problem's"):
        reduced_problem(problem, other)


@pytest.mark.smoke
def test_reduced_sipnet_forward_at_zero_is_the_full_forward_at_the_prior_centre():
    small = load_inverse_problem("harvard_forest_small")
    subspace = Subspace.from_information(
        np.diag(np.arange(10.0, 0.0, -1.0)),
        parameter_names=small.coordinates.names,
        estimator="exact",
        rank=3,
    )
    reduced = reduced_problem(small, subspace)
    at_zero = np.asarray(reduced.forward({"active_1": 0.0, "active_2": 0.0, "active_3": 0.0}).raw)
    full = small.predicted_observations(small.coordinates.parameters_from_whitened(np.zeros(10)))
    np.testing.assert_allclose(at_zero, full, rtol=1e-12)
