import numpy as np
import pytest
from probpipe import log_prob, unnormalized_log_prob
from scipy import stats

from lsmas.problems import load

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def fixed():
    return load("harvard_forest_dev")


@pytest.fixture(scope="module")
def inferred():
    return load("harvard_forest_dev", scales="inferred")


THETA = {
    "photosynthetic_capacity": 250.0,
    "respiration_share": 0.09,
    "optimum_photosynthesis_temperature": 22.0,
    "soil_respiration_q10": 2.0,
}


def test_noise_scales_are_parameters_only_when_inferred(fixed, inferred):
    assert list(fixed.prior.event_spec.components) == ["theta"]
    assert list(inferred.prior.event_spec.components) == ["theta", "s2_day", "s2_night"]
    assert fixed.coordinates.names == inferred.coordinates.names


def test_describe_names_the_parameters_and_data(fixed):
    summary = fixed.describe()
    assert "photosynthetic_capacity" in summary
    assert "nee (400 values)" in summary


def test_conditioning_target_is_the_generalized_posterior(fixed):
    target = fixed.condition(method="unnormalized")
    other = THETA | {"soil_respiration_q10": 1.6}
    difference = float(unnormalized_log_prob(target, {"theta": THETA}).raw()) - float(
        unnormalized_log_prob(target, {"theta": other}).raw()
    )

    def expected(theta):
        predicted = np.asarray(fixed.forward(theta).raw())
        variances = fixed.likelihood.noise_variances()
        pointwise = stats.norm(predicted, np.sqrt(variances)).logpdf(fixed.data["nee"])
        prior = float(log_prob(fixed.prior, {"theta": theta}).raw())
        return np.sum(fixed.learning_rate["nee"] * pointwise) + prior

    assert difference == pytest.approx(expected(THETA) - expected(other), rel=1e-8)


def test_unknown_problem_is_refused():
    with pytest.raises(KeyError, match="no problem called"):
        load("mars")
