import numpy as np
import pytest
from probpipe import InverseGamma, condition_on, log_prob, unnormalized_log_prob
from probpipe.distributions import (
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
)
from scipy import stats

from conftest import (
    CLASSES,
    FORWARD_MATRIX,
    REFERENCE_VARIANCE,
    linear_likelihood,
    linear_theta_prior,
)
from lsmas._stopgaps.tempering import TemperedLikelihood, tempered_condition_on

LEARNING_RATE = np.where(CLASSES == "day", 0.2, 0.4)


@pytest.fixture
def model():
    likelihood = linear_likelihood({"day": "s2_day", "night": "s2_night"})
    prior = (
        linear_theta_prior()
        * InverseGamma("s2_day", 2.0, 1.678)
        * InverseGamma("s2_night", 2.0, 1.678)
    )
    return likelihood * prior, prior


def point(a, b, day, night):
    return {"theta": {"a": a, "b": b}, "s2_day": day, "s2_night": night}


def expected_log_density(prior, value, observations, learning_rate):
    """log π(value) + Σ_i η_i log N(y_i | G_i, s²_k(i) v_i), computed independently."""
    mean = FORWARD_MATRIX @ np.array([value["theta"]["a"], value["theta"]["b"]])
    variances = np.where(CLASSES == "day", value["s2_day"], value["s2_night"]) * REFERENCE_VARIANCE
    pointwise = stats.norm(mean, np.sqrt(variances)).logpdf(observations)
    return float(np.sum(learning_rate * pointwise) + log_prob(prior, value).raw())


@pytest.mark.parametrize("learning_rate", [LEARNING_RATE, 0.3, 1.0])
def test_target_density_is_prior_plus_tempered_log_likelihood(model, observations, learning_rate):
    joint, prior = model
    target = tempered_condition_on(
        joint, {"y": observations}, learning_rate={"y": learning_rate}, method="unnormalized"
    )
    first, second = point(0.8, -0.6, 1.5, 0.7), point(-0.2, 0.4, 0.6, 2.0)
    difference = float(unnormalized_log_prob(target, first).raw()) - float(
        unnormalized_log_prob(target, second).raw()
    )
    expected = expected_log_density(prior, first, observations, learning_rate) - (
        expected_log_density(prior, second, observations, learning_rate)
    )
    assert difference == pytest.approx(expected, rel=1e-10)


def test_rate_one_gives_the_ordinary_posterior_target(model, observations):
    joint, _ = model
    data = {"y": observations}
    tempered = tempered_condition_on(joint, data, learning_rate={"y": 1.0}, method="unnormalized")
    plain = condition_on.using(method="unnormalized")(joint, data)
    first, second = point(0.8, -0.6, 1.5, 0.7), point(-0.2, 0.4, 0.6, 2.0)

    def difference(target):
        return float(unnormalized_log_prob(target, first).raw()) - float(
            unnormalized_log_prob(target, second).raw()
        )

    assert difference(tempered) == pytest.approx(difference(plain), rel=1e-10)


def test_rates_must_name_observed_components_and_lie_in_unit_interval(model, observations):
    joint, _ = model
    with pytest.raises(ValueError, match="not observed"):
        tempered_condition_on(joint, {"y": observations}, learning_rate={"z": 0.5})
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        TemperedLikelihood(linear_likelihood({"day": 1.0, "night": 1.0}), 1.5)


def test_tempered_factor_claims_no_sampling():
    tempered = TemperedLikelihood(linear_likelihood({"day": 1.0, "night": 1.0}), 0.5)
    assert not isinstance(tempered, SupportsConditionalSampling)
    assert isinstance(tempered, SupportsConditionalUnnormalizedLogProb)
