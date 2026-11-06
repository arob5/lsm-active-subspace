import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsmas.data import load_harvard_forest
from lsmas.parameters import calibration_to_sipnet
from lsmas.priors import prior_medians
from lsmas.problems import load
from lsmas.problems.toy import TOY_PARAMETERS, ToyCanopyModel, _growing_degree_days
from lsmas.sipnet import SIPNETSimulator


@pytest.fixture(scope="module")
def data():
    return load_harvard_forest()


@pytest.fixture(scope="module")
def model(data):
    return ToyCanopyModel.from_data(data)


@pytest.fixture(scope="module")
def theta(data):
    medians = prior_medians(data.prior_table)
    return {name: jnp.asarray(medians[name]) for name in TOY_PARAMETERS}


def test_toy_problem_calibrates_its_fourteen_parameters():
    problem = load("toy")
    assert problem.coordinates.names == TOY_PARAMETERS
    assert "nee (400 values)" in problem.describe()


def test_temperature_effect_is_one_at_the_optimum_and_zero_one_range_away(model, theta):
    optimum = theta["optimum_photosynthesis_temperature"]
    half_width = theta["photosynthesis_temperature_range"]
    at = {"optimum": optimum, "edge": optimum + half_width, "beyond": optimum - 2 * half_width}
    probe = dataclasses.replace(model, air_temperature=jnp.stack(list(at.values())))
    np.testing.assert_allclose(probe.temperature_effect(theta), [1.0, 0.0, 0.0], atol=1e-12)


def test_light_effect_of_a_thin_canopy_is_the_top_layers(model, theta):
    leaf_area_index = jnp.full(model.radiation.shape, 1e-9)
    top_layer = 1 - 2.0 ** (-model.radiation / theta["half_saturation_light"])
    expected = jnp.where(model.radiation > 0, top_layer, 0.0)
    np.testing.assert_allclose(model.light_effect(theta, leaf_area_index), expected, atol=1e-8)


def test_growing_degree_days_restart_each_year():
    year = np.array([2012, 2012, 2012, 2013, 2013])
    temperature = np.array([8.0, -4.0, 16.0, 8.0, 8.0])
    np.testing.assert_allclose(
        _growing_degree_days(year, temperature, np.full(5, 0.125)), [1.0, 1.0, 3.0, 1.0, 2.0]
    )


def test_forward_map_is_differentiable_through_probpipe(theta):
    forward = load("toy").forward.using(raw=True)
    jacobian = jax.jacfwd(forward)(theta)["soil_respiration_q10"]
    step = 1e-6
    shifted = theta | {"soil_respiration_q10": theta["soil_respiration_q10"] + step}
    difference = (np.asarray(forward(shifted)) - np.asarray(forward(theta))) / step
    np.testing.assert_allclose(jacobian, difference, rtol=1e-4, atol=1e-6)


@pytest.mark.smoke
def test_toy_resembles_sipnet_at_the_prior_medians(data, model, theta):
    simulator = SIPNETSimulator(
        data.driver_path,
        calibration_to_sipnet(prior_medians(data.prior_table), data.fixed_sipnet_parameters),
    )
    sipnet_features = data.operator @ simulator.nee_rate({})
    toy_features = np.asarray(model.features(theta))
    assert np.corrcoef(sipnet_features, toy_features)[0, 1] > 0.95
