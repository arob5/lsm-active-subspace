import itertools

import numpy as np
import pytest
from probpipe import InverseGamma, condition_on, cov, mean, workflow_run

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    PRIOR_MEAN,
    PRIOR_SD,
    REFERENCE_VARIANCE,
    linear_likelihood,
    linear_theta_prior,
)
from lsmas._stopgaps.tempering import tempered_condition_on
from lsmas.inference import EnsKitEKI

FIXED_SCALES = {"day": 1.5, "night": 0.7}
LEARNING_RATE = np.where(CLASSES == "day", 0.5, 0.8)


def linear_model():
    return linear_likelihood(FIXED_SCALES) * linear_theta_prior()


def exact_tempered_posterior(observations):
    r"""Mean and covariance of π(θ) ∏ N(y_i | (Hθ)_i, σ_i²)^{η_i}, with σ² = s² v."""
    variances = np.where(CLASSES == "day", FIXED_SCALES["day"], FIXED_SCALES["night"])
    precision_weights = LEARNING_RATE / (variances * REFERENCE_VARIANCE)
    prior_precision = np.diag(1 / PRIOR_SD**2)
    precision = prior_precision + FORWARD_MATRIX.T @ (precision_weights[:, None] * FORWARD_MATRIX)
    covariance = np.linalg.inv(precision)
    posterior_mean = covariance @ (
        prior_precision @ PRIOR_MEAN + FORWARD_MATRIX.T @ (precision_weights * observations)
    )
    return posterior_mean, covariance


def posterior_moments(posterior):
    posterior_mean = np.array([float(v) for v in mean(posterior["theta"]).raw().values()])
    return posterior_mean, np.asarray(cov(posterior["theta"]).raw())


def test_eki_recovers_the_linear_gaussian_tempered_posterior(observations):
    with workflow_run(seed=0):
        posterior = tempered_condition_on(
            linear_model(),
            {"y": observations},
            learning_rate={"y": LEARNING_RATE},
            method="enskit_eki",
            method_options={"n_particles": 4000},
        )
    exact_mean, exact_covariance = exact_tempered_posterior(observations)
    estimated_mean, estimated_covariance = posterior_moments(posterior)
    standard_errors = np.sqrt(np.diag(exact_covariance) / 4000)
    assert np.all(np.abs(estimated_mean - exact_mean) < 5 * standard_errors)
    np.testing.assert_allclose(estimated_covariance, exact_covariance, rtol=0.1, atol=0.01)


def test_callbacks_see_every_step_up_the_ladder(observations):
    records = []
    with workflow_run(seed=1):
        posterior = condition_on.using(
            method="enskit_eki", method_options={"n_particles": 50, "callbacks": [records.append]}
        )(linear_model(), {"y": observations})
    betas = [record.beta for record in records]
    assert betas[0] == 0.0
    assert all(later > earlier for earlier, later in itertools.pairwise(betas))
    assert betas[-1] + records[-1].increment == pytest.approx(1.0)
    assert records[0].parameter_names == ("a", "b")
    assert records[0].theta.shape == (50, 2)
    assert records[0].predictions.shape == (50, 6)
    history = posterior.annotations["arviz/eki_history"]
    np.testing.assert_allclose(history["beta"].values, betas)


def test_runs_are_reproducible_under_a_seed(observations):
    def posterior_mean():
        with workflow_run(seed=2):
            posterior = condition_on.using(method="enskit_eki", method_options={"n_particles": 30})(
                linear_model(), {"y": observations}
            )
        return posterior_moments(posterior)[0]

    np.testing.assert_array_equal(posterior_mean(), posterior_mean())


def test_inferred_noise_scales_are_refused(observations):
    model = linear_likelihood({"day": "s2_day", "night": 1.0}) * (
        linear_theta_prior() * InverseGamma("s2_day", 2.0, 1.0)
    )
    target = condition_on.using(method="unnormalized")(model, {"y": observations})
    feasibility = EnsKitEKI().check(target)
    assert not feasibility.feasible
    assert "noise scales must be fixed" in feasibility.description


def test_method_runs_only_when_named(observations):
    report = condition_on.check(linear_model(), {"y": observations})
    assert report.method != "enskit_eki"


@pytest.mark.smoke
def test_eki_runs_on_the_harvard_forest_development_problem():
    from lsmas.problems import load

    problem = load("harvard_forest_dev")
    records = []
    with workflow_run(seed=3):
        posterior = problem.condition(
            method="enskit_eki", method_options={"n_particles": 16, "callbacks": [records.append]}
        )
    # make_posterior lays out every posterior as chains of draws; EKI's ensemble is one chain.
    theta = problem.coordinates.array_from_draws(posterior.atoms)
    assert theta.shape == (1, 16, 4)
    assert records[0].predictions.shape == (16, 400)
    assert sum(record.n_failed for record in records) == 0
