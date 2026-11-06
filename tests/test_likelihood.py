import numpy as np
import pytest
from probpipe import SupportsLogProb, condition_on, log_prob, sample, workflow_run
from scipy import stats

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    REFERENCE_VARIANCE,
    linear_likelihood,
)

THETA = {"a": 0.8, "b": -0.6}
SCALES = {"s2_day": 1.5, "s2_night": 0.7}


def expected_variances(day, night):
    return np.where(CLASSES == "day", day, night) * REFERENCE_VARIANCE


def test_scales_are_given_slots_or_fixed():
    random = linear_likelihood({"day": "s2_day", "night": "s2_night"})
    fixed = linear_likelihood({"day": 1.0, "night": 2.0})
    assert set(random.given_spec) == {"theta", "s2_day", "s2_night"}
    assert set(fixed.given_spec) == {"theta"}
    np.testing.assert_allclose(fixed.noise_variances(), expected_variances(1.0, 2.0))


def test_noise_variances_broadcast_over_batches_of_scales():
    likelihood = linear_likelihood({"day": "s2_day", "night": 0.5})
    variances = likelihood.noise_variances({"s2_day": np.array([1.0, 2.0, 3.0])})
    assert variances.shape == (3, 6)
    np.testing.assert_allclose(variances[1], expected_variances(2.0, 0.5))


def test_density_is_the_gaussian_density(observations):
    likelihood = linear_likelihood({"day": "s2_day", "night": "s2_night"})
    law = condition_on(likelihood, {"theta": THETA, **SCALES})
    assert isinstance(law, SupportsLogProb)
    mean = FORWARD_MATRIX @ np.array([THETA["a"], THETA["b"]])
    sd = np.sqrt(expected_variances(SCALES["s2_day"], SCALES["s2_night"]))
    expected = stats.norm(mean, sd).logpdf(observations).sum()
    assert float(log_prob(law, observations).raw()) == pytest.approx(expected, rel=1e-12)


def test_draws_follow_the_same_law():
    likelihood = linear_likelihood({"day": "s2_day", "night": "s2_night"})
    with workflow_run(seed=3):
        law = condition_on(likelihood, {"theta": THETA, **SCALES})
        draws = np.asarray(sample(law, 20_000).raw())
    mean = FORWARD_MATRIX @ np.array([THETA["a"], THETA["b"]])
    variances = expected_variances(SCALES["s2_day"], SCALES["s2_night"])
    standard_errors = np.sqrt(variances / draws.shape[0])
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 4 * standard_errors)
    np.testing.assert_allclose(draws.var(axis=0), variances, rtol=0.05)


def test_every_class_needs_exactly_one_scale():
    with pytest.raises(ValueError, match="classes without a scale"):
        linear_likelihood({"day": 1.0})
