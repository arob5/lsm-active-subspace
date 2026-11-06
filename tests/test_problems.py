import numpy as np
import pytest
from probpipe import log_prob, sample, unnormalized_log_prob, workflow_run
from scipy import stats

from lsmas.data import load_harvard_forest
from lsmas.priors import prior_medians
from lsmas.problems import SMALL_PROBLEM_PARAMETERS, load

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def small():
    return load("harvard_forest_small")


@pytest.fixture(scope="module")
def small_inferred():
    return load("harvard_forest_small", scales="inferred")


@pytest.fixture(scope="module")
def full():
    return load("harvard_forest")


THETA = {
    "photosynthetic_capacity": 250.0,
    "half_saturation_light": 11.0,
    "light_extinction_coefficient": 0.5,
    "leaf_on_growth": 95.0,
    "vapor_pressure_deficit_slope": 0.05,
    "vapor_pressure_deficit_exponent": 1.7,
    "soil_water_holding_capacity": 47.0,
    "water_use_efficiency": 12.0,
    "respiration_share": 0.09,
    "soil_respiration_flux_at_10c": 420.0,
}


def test_full_problem_calibrates_every_parameter_of_the_prior_table(full):
    assert full.coordinates.names == tuple(load_harvard_forest().prior_table.index)
    assert full.coordinates.dimension == 46


def test_small_problem_calibrates_its_ten(small):
    assert small.coordinates.names == SMALL_PROBLEM_PARAMETERS == tuple(THETA)


def test_small_problem_is_the_full_one_with_the_rest_at_their_medians(small, full):
    medians = prior_medians(load_harvard_forest().prior_table)
    small_features = np.asarray(small.forward(THETA).raw)
    full_features = np.asarray(full.forward(medians | THETA).raw)
    np.testing.assert_array_equal(small_features, full_features)


def test_noise_scales_are_parameters_only_when_inferred(small, small_inferred):
    assert list(small.prior.event_spec.outputs) == ["theta"]
    assert list(small_inferred.prior.event_spec.outputs) == ["theta", "s2_day", "s2_night"]
    assert small.coordinates.names == small_inferred.coordinates.names


def test_describe_names_the_parameters_and_data(small):
    summary = small.describe()
    assert "10 of the 46 calibration parameters" in summary
    assert "photosynthetic_capacity" in summary
    assert "nee (400 values)" in summary


def test_conditioning_target_is_the_generalized_posterior(small):
    target = small.condition(method="unnormalized")
    other = THETA | {"soil_respiration_flux_at_10c": 300.0}
    difference = float(unnormalized_log_prob(target, {"theta": THETA}).raw) - float(
        unnormalized_log_prob(target, {"theta": other}).raw
    )

    def expected(theta):
        predicted = np.asarray(small.forward(theta).raw)
        variances = small.likelihood.noise_variances()
        pointwise = stats.norm(predicted, np.sqrt(variances)).logpdf(small.data["nee"])
        prior = float(log_prob(small.prior, {"theta": theta}).raw)
        return np.sum(small.learning_rate["nee"] * pointwise) + prior

    assert difference == pytest.approx(expected(THETA) - expected(other), rel=1e-8)


def test_model_draws_run_sipnet(small_inferred):
    with workflow_run(seed=3):
        draws = sample(small_inferred.model, 8)
    assert set(draws.raw) == {"nee", "theta", "s2_day", "s2_night"}
    nee = np.asarray(draws["nee"].raw)
    assert nee.shape == (8, 400)
    assert np.all(np.isfinite(nee))


def test_full_prior_predictive_runs_without_failures(full):
    with workflow_run(seed=5):
        nee = np.asarray(sample(full.model, 16)["nee"].raw)
    assert nee.shape == (16, 400)
    assert np.all(np.isfinite(nee))


def test_model_draws_are_reproducible_under_a_seed(small):
    def nee_draws():
        with workflow_run(seed=4):
            return np.asarray(sample(small.model, 4)["nee"].raw)

    np.testing.assert_array_equal(nee_draws(), nee_draws())


def test_conditioning_requires_a_method(small):
    with pytest.raises(TypeError, match="method"):
        small.condition(method_options={"n_particles": 4})


def test_unknown_problem_is_refused():
    with pytest.raises(KeyError, match="no problem called"):
        load("mars")
