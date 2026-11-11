import numpy as np
import pytest
from probpipe import (
    Function,
    InverseGamma,
    SupportsLogProb,
    condition_on,
    log_prob,
    sample,
    workflow_run,
)
from scipy import stats

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    REFERENCE_VARIANCE,
    linear_forward,
    linear_likelihood,
    linear_theta_prior,
)
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas.priors import independent_prior

THETA = {"a": 0.8, "b": -0.6}
SCALES = {"day": 1.5, "night": 0.7}
INFERRED = {"day": "inferred", "night": "inferred"}


def expected_variances(day, night):
    return np.where(CLASSES == "day", day, night) * REFERENCE_VARIANCE


def test_inferred_scales_are_the_fields_of_the_given_record_s2():
    inferred = linear_likelihood({"day": "inferred", "night": 2.0})
    fixed = linear_likelihood({"day": 1.0, "night": 2.0})
    assert set(inferred.given_spec) == {"theta", "s2"}
    assert list(inferred.given_spec["s2"].fields) == ["day"]
    assert inferred.inferred_scale_classes == ("day",)
    assert set(fixed.given_spec) == {"theta"}
    np.testing.assert_allclose(fixed.noise_variances(), expected_variances(1.0, 2.0))


def test_a_scale_is_a_number_or_inferred():
    with pytest.raises(ValueError, match="the scales of \\['day'\\] are not"):
        linear_likelihood({"day": "s2_day", "night": 1.0})


def test_noise_variances_broadcast_over_batches_of_scales():
    likelihood = linear_likelihood({"day": "inferred", "night": 0.5})
    variances = likelihood.noise_variances({"day": np.array([1.0, 2.0, 3.0])})
    assert variances.shape == (3, 6)
    np.testing.assert_allclose(variances[1], expected_variances(2.0, 0.5))


def test_density_is_the_gaussian_density(observations):
    likelihood = linear_likelihood(INFERRED)
    law = condition_on(likelihood, {"theta": THETA, "s2": SCALES})
    assert isinstance(law, SupportsLogProb)
    mean = FORWARD_MATRIX @ np.array([THETA["a"], THETA["b"]])
    sd = np.sqrt(expected_variances(SCALES["day"], SCALES["night"]))
    expected = stats.norm(mean, sd).logpdf(observations).sum()
    assert float(log_prob(law, observations).raw) == pytest.approx(expected, rel=1e-12)


def test_draws_follow_the_same_law():
    likelihood = linear_likelihood(INFERRED)
    with workflow_run(seed=3):
        law = condition_on(likelihood, {"theta": THETA, "s2": SCALES})
        draws = np.asarray(sample(law, 20_000).raw)
    mean = FORWARD_MATRIX @ np.array([THETA["a"], THETA["b"]])
    variances = expected_variances(SCALES["day"], SCALES["night"])
    standard_errors = np.sqrt(variances / draws.shape[0])
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 4 * standard_errors)
    np.testing.assert_allclose(draws.var(axis=0), variances, rtol=0.05)


def test_a_model_with_this_likelihood_samples_parameters_and_observations_jointly():
    likelihood = linear_likelihood({"day": "inferred", "night": 0.5})
    scale_prior = independent_prior({"day": InverseGamma("day", 3.0, 2.0)}, group="s2")
    model = likelihood & (linear_theta_prior() & scale_prior)
    with workflow_run(seed=1):
        draws = sample(model, 4000)
    assert set(draws.raw) == {"y", "theta", "s2"}
    theta = np.column_stack([np.asarray(draws[f"theta/{name}"].raw) for name in "ab"])
    scale = likelihood.inferred_scale_draws(draws)["day"]
    residuals = np.asarray(draws["y"].raw) - theta @ FORWARD_MATRIX.T
    variances = np.where(CLASSES == "day", scale[:, None], 0.5) * REFERENCE_VARIANCE
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
            scales={"day": 1.0, "night": 1.0},
        )
