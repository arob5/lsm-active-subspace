import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from probpipe import (
    Function,
    InverseGamma,
    LogNormal,
    NumericArraySpec,
    condition_on,
    workflow_run,
)

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    REFERENCE_VARIANCE,
    THETA_SPEC,
    exact_linear_posterior,
    linear_likelihood,
    linear_theta_prior,
    posterior_moments,
)
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas._stopgaps.tempering import tempered_condition_on
from lsmas.inference import EnsKitEKI
from lsmas.priors import independent_prior

FIXED_SCALES = {"day": 1.5, "night": 0.7}
LEARNING_RATE = np.where(CLASSES == "day", 0.5, 0.8)


def linear_model():
    return linear_likelihood(FIXED_SCALES) & linear_theta_prior()


def test_eki_recovers_the_linear_gaussian_tempered_posterior(observations):
    with workflow_run(seed=0):
        posterior = tempered_condition_on(
            linear_model(),
            {"y": observations},
            learning_rate={"y": LEARNING_RATE},
            method="enskit_eki",
            method_options={"n_particles": 4000},
        )
    exact_mean, exact_covariance = exact_linear_posterior(observations, FIXED_SCALES, LEARNING_RATE)
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
    assert all(record.increment > 0 for record in records)
    assert betas[-1] + records[-1].increment == pytest.approx(1.0)
    assert records[0].parameter_names == ("a", "b")
    assert records[0].theta.shape == (50, 2)
    assert records[0].predictions.shape == (50, 6)
    assert isinstance(records[0].noise_variance, np.ndarray)
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


def mixed_scale_model():
    """The linear model with the day scale inferred, IG(3, 2), and the night scale fixed."""
    scale_prior = independent_prior({"day": InverseGamma("day", 3.0, 2.0)}, group="s2")
    return linear_likelihood({"day": "inferred", "night": 0.7}) & (
        linear_theta_prior() & scale_prior
    )


def test_inferred_noise_scales_match_nuts(observations):
    def tempered(method, **using):
        with workflow_run(seed=5):
            return tempered_condition_on(
                mixed_scale_model(),
                {"y": observations},
                learning_rate={"y": LEARNING_RATE},
                method=method,
                **using,
            )

    eki_posterior = tempered("enskit_eki", method_options={"n_particles": 4000})
    nuts_posterior = tempered("blackjax_nuts")
    eki_mean, eki_covariance = posterior_moments(eki_posterior)
    nuts_mean, nuts_covariance = posterior_moments(nuts_posterior)
    nuts_sd = np.sqrt(np.diag(nuts_covariance))
    np.testing.assert_allclose(eki_mean, nuts_mean, atol=0.1 * nuts_sd.max())
    np.testing.assert_allclose(np.sqrt(np.diag(eki_covariance)), nuts_sd, rtol=0.15)
    eki_scale = np.asarray(eki_posterior.atoms["s2/day"].raw).ravel()
    nuts_scale = np.asarray(nuts_posterior.atoms["s2/day"].raw).ravel()
    np.testing.assert_allclose(
        np.quantile(eki_scale, [0.1, 0.5, 0.9]), np.quantile(nuts_scale, [0.1, 0.5, 0.9]), rtol=0.15
    )


def test_records_carry_the_inferred_scales(observations):
    records = []
    with workflow_run(seed=6):
        condition_on.using(
            method="enskit_eki", method_options={"n_particles": 40, "callbacks": [records.append]}
        )(mixed_scale_model(), {"y": observations})
    assert set(records[0].noise_scales) == {"day"}
    assert records[0].noise_scales["day"].shape == (40,)
    night = CLASSES == "night"
    assert all(
        np.allclose(record.noise_variance[night] / REFERENCE_VARIANCE[night], 0.7)
        for record in records
    )


def test_a_scale_without_an_inverse_gamma_prior_is_refused(observations):
    scale_prior = independent_prior({"day": LogNormal("day", 0.0, 1.0)}, group="s2")
    model = linear_likelihood({"day": "inferred", "night": 1.0}) & (
        linear_theta_prior() & scale_prior
    )
    target = condition_on.using(method="unnormalized")(model, {"y": observations})
    feasibility = EnsKitEKI().check(target)
    assert not feasibility.feasible
    assert "inverse-gamma prior" in feasibility.description


def test_method_runs_only_when_named(observations):
    report = condition_on.check(linear_model(), {"y": observations})
    assert report.method != "enskit_eki"


@pytest.mark.smoke
def test_eki_runs_on_the_small_harvard_forest_problem():
    from lsmas.problems import load_inverse_problem

    problem = load_inverse_problem("harvard_forest_small")
    records = []
    with workflow_run(seed=3):
        posterior = problem.condition(
            method="enskit_eki", method_options={"n_particles": 16, "callbacks": [records.append]}
        )
    # make_posterior lays out every posterior as chains of draws; EKI's ensemble is one chain.
    theta = problem.coordinates.array_from_draws(posterior.atoms)
    assert theta.shape == (1, 16, 10)
    assert records[0].predictions.shape == (16, 400)
    assert sum(record.n_failed for record in records) == 0


def failing_linear_forward(theta):
    """The linear forward map, failing (NaN) wherever a > 1.5."""
    a, b = float(np.asarray(theta["a"])), float(np.asarray(theta["b"]))
    values = FORWARD_MATRIX @ np.array([a, b])
    return np.full(6, np.nan) if a > 1.5 else values


def test_failed_runs_are_counted_and_repaired(observations):
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
        scales=FIXED_SCALES,
    )
    records = []
    with workflow_run(seed=7):
        posterior = condition_on.using(
            method="enskit_eki", method_options={"n_particles": 50, "callbacks": [records.append]}
        )(likelihood & linear_theta_prior(), {"y": observations})
    assert records[0].n_failed > 0
    assert np.all(np.isfinite(records[0].predictions))
    assert np.all(np.isfinite(posterior_moments(posterior)[0]))
