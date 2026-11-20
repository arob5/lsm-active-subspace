import numpy as np
import pytest
from probpipe import (
    Function,
    SupportsLogProb,
    condition_on,
    log_prob,
    mean,
    sample,
    variance,
    workflow_run,
)
from scipy import stats

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    REFERENCE_VARIANCE,
    linear_forward,
    linear_likelihood,
    linear_scale_prior,
    linear_theta_prior,
)
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood

THETA = {"a": 0.8, "b": -0.6}
SCALES = {"day": 1.5, "night": 0.7}


def expected_variances(day, night):
    return np.where(CLASSES == "day", day, night) * REFERENCE_VARIANCE


def test_the_noise_scales_are_an_optional_input_with_default_scales():
    likelihood = linear_likelihood({"day": 1.0, "night": 2.0})
    assert list(likelihood.given_spec) == ["theta", "s2"]
    assert list(likelihood.given_spec.optional) == ["s2"]
    assert list(likelihood.given_spec["s2"].fields) == ["day", "night"]
    at_defaults = variance(condition_on(likelihood, {"theta": THETA}))
    np.testing.assert_allclose(at_defaults, expected_variances(1.0, 2.0))
    at_scales = variance(condition_on(likelihood, {"theta": THETA, "s2": SCALES}))
    np.testing.assert_allclose(at_scales, expected_variances(SCALES["day"], SCALES["night"]))


def test_the_mean_of_the_law_at_given_values_is_the_forward_map():
    law = condition_on(linear_likelihood(), {"theta": THETA})
    np.testing.assert_allclose(mean(law), FORWARD_MATRIX @ np.array([THETA["a"], THETA["b"]]))


def test_the_noise_scales_are_parameters_exactly_when_the_prior_has_them():
    fixed = linear_likelihood(SCALES) & linear_theta_prior()
    inferred = linear_likelihood() & (linear_theta_prior() & linear_scale_prior())
    with workflow_run(seed=0):
        assert set(sample(fixed).raw) == {"y", "theta"}
        parameters = sample(inferred)
    assert set(parameters.raw) == {"y", "theta", "s2"}
    scales = {noise_class: float(parameters[f"s2/{noise_class}"]) for noise_class in SCALES}
    np.testing.assert_allclose(
        variance(condition_on(inferred, {"theta": parameters["theta"], "s2": parameters["s2"]})),
        expected_variances(scales["day"], scales["night"]),
    )


def test_density_is_the_gaussian_density(observations):
    likelihood = linear_likelihood()
    law = condition_on(likelihood, {"theta": THETA, "s2": SCALES})
    assert isinstance(law, SupportsLogProb)
    mean = FORWARD_MATRIX @ np.array([THETA["a"], THETA["b"]])
    sd = np.sqrt(expected_variances(SCALES["day"], SCALES["night"]))
    expected = stats.norm(mean, sd).logpdf(observations).sum()
    assert float(log_prob(law, observations).raw) == pytest.approx(expected, rel=1e-12)


def test_draws_follow_the_same_law():
    likelihood = linear_likelihood()
    with workflow_run(seed=3):
        law = condition_on(likelihood, {"theta": THETA, "s2": SCALES})
        draws = np.asarray(sample(law, 20_000).raw)
    mean = FORWARD_MATRIX @ np.array([THETA["a"], THETA["b"]])
    variances = expected_variances(SCALES["day"], SCALES["night"])
    standard_errors = np.sqrt(variances / draws.shape[0])
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 4 * standard_errors)
    np.testing.assert_allclose(draws.var(axis=0), variances, rtol=0.05)


def test_a_model_with_this_likelihood_samples_parameters_and_observations_jointly():
    model = linear_likelihood() & (linear_theta_prior() & linear_scale_prior())
    with workflow_run(seed=1):
        draws = sample(model, 4000)
    assert set(draws.raw) == {"y", "theta", "s2"}
    theta = np.column_stack([np.asarray(draws[f"theta/{name}"].raw) for name in "ab"])
    day, night = (np.asarray(draws[f"s2/{noise_class}"].raw)[:, None] for noise_class in SCALES)
    residuals = np.asarray(draws["y"].raw) - theta @ FORWARD_MATRIX.T
    variances = np.where(CLASSES == "day", day, night) * REFERENCE_VARIANCE
    standardized = residuals / np.sqrt(variances)
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=0.08)
    np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=0.05)


def test_every_class_needs_exactly_one_scale():
    with pytest.raises(ValueError, match="classes without a scale"):
        linear_likelihood({"day": 1.0})


def test_the_parameter_record_is_the_forward_maps_declared_input():
    likelihood = linear_likelihood({"day": 1.0, "night": 1.0})
    assert likelihood.given_spec["theta"] == linear_forward.input_spec["theta"]
    undeclared = Function(lambda theta: theta["a"], label="undeclared", traceable=True)
    with pytest.raises(ValueError, match="must declare one input, theta"):
        GaussianSimulatorLikelihood(
            undeclared,
            output_name="y",
            reference_variance=REFERENCE_VARIANCE,
            classes=CLASSES,
            default_scales={"day": 1.0, "night": 1.0},
        )
