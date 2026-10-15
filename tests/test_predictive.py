import numpy as np
import pytest
from probpipe import InverseGamma, workflow_run

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    REFERENCE_VARIANCE,
    linear_likelihood,
    linear_theta_prior,
)
from lsmas._stopgaps.coordinates import Coordinates
from lsmas.predictive import prior_predictive
from lsmas.problems import InverseProblem, load


def linear_problem():
    likelihood = linear_likelihood({"day": "s2_day", "night": 0.5})
    prior = linear_theta_prior() * InverseGamma("s2_day", 3.0, 2.0)
    return InverseProblem(
        name="linear",
        description="A linear-Gaussian test problem.",
        likelihood=likelihood,
        prior=prior,
        data={"y": np.zeros(6)},
        learning_rate={"y": np.ones(6)},
        coordinates=Coordinates.from_prior(prior, "theta", n_moment_draws=1000),
    )


def test_draws_hold_the_parameters_and_the_observations():
    with workflow_run(seed=0):
        draws = prior_predictive(linear_problem(), 50)
    assert set(draws.raw()) == {"theta", "s2_day", "y"}
    assert np.asarray(draws["y"].raw()).shape == (50, 6)


def test_observations_given_parameters_follow_the_likelihood():
    with workflow_run(seed=1):
        draws = prior_predictive(linear_problem(), 4000)
    theta = np.column_stack([np.asarray(draws[f"theta/{name}"].raw()) for name in "ab"])
    scale = np.asarray(draws["s2_day"].raw())
    residuals = np.asarray(draws["y"].raw()) - theta @ FORWARD_MATRIX.T
    variances = np.where(CLASSES == "day", scale[:, None], 0.5) * REFERENCE_VARIANCE
    standardized = residuals / np.sqrt(variances)
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=0.08)
    np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=0.05)


def test_draws_are_reproducible_under_a_seed():
    problem = linear_problem()
    with workflow_run(seed=2):
        first = np.asarray(prior_predictive(problem, 5)["y"].raw())
    with workflow_run(seed=2):
        second = np.asarray(prior_predictive(problem, 5)["y"].raw())
    np.testing.assert_array_equal(first, second)


@pytest.mark.smoke
def test_harvard_forest_prior_predictive_runs_sipnet():
    with workflow_run(seed=3):
        draws = prior_predictive(load("harvard_forest_dev", scales="inferred"), 8)
    nee = np.asarray(draws["nee"].raw())
    assert nee.shape == (8, 400)
    assert np.all(np.isfinite(nee))
