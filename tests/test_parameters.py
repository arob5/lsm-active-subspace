from collections.abc import Mapping

import pytest

from lsmas.data import load_harvard_forest
from lsmas.parameters import (
    SIPNET_PARAMETERS,
    allocation_to_sipnet,
    calibration_to_sipnet,
    initial_pools_to_sipnet,
    photosynthesis_temperatures_to_sipnet,
    photosynthesis_to_sipnet,
    respiration_rates_to_sipnet,
    soil_respiration_to_sipnet,
)


@pytest.fixture(scope="module")
def data():
    return load_harvard_forest()


@pytest.fixture(scope="module")
def fixed(data):
    return data.fixed_sipnet_parameters


@pytest.fixture(scope="module")
def theta(fixed):
    """Every calibration parameter that :func:`calibration_to_sipnet` reads, at 0.5."""
    recorder = _ReadRecorder()
    calibration_to_sipnet(recorder, fixed)
    return dict.fromkeys(recorder.read, 0.5)


class _ReadRecorder(Mapping):
    """Answers 0.5 for any name and remembers which names were read."""

    def __init__(self):
        self.read = set()

    def __getitem__(self, name):
        self.read.add(name)
        return 0.5

    def __iter__(self):
        return iter(self.read)

    def __len__(self):
        return len(self.read)


def test_photosynthesis_map_inverts_the_identifiable_combinations():
    a_max, daily_fraction, respiration_fraction, carbon_fraction = 120.0, 0.86, 0.09, 0.47
    capacity = a_max * (daily_fraction + respiration_fraction) / carbon_fraction
    share = respiration_fraction / (daily_fraction + respiration_fraction)
    sipnet = photosynthesis_to_sipnet(
        capacity,
        share,
        daily_mean_photosynthesis_fraction=daily_fraction,
        leaf_carbon_fraction=carbon_fraction,
    )
    assert sipnet["max_photosynthesis_rate"] == pytest.approx(a_max, rel=1e-12)
    assert sipnet["foliar_respiration_fraction"] == pytest.approx(respiration_fraction, rel=1e-12)


def test_temperature_map_puts_the_minimum_one_range_below_the_optimum():
    sipnet = photosynthesis_temperatures_to_sipnet(22.0, 23.5)
    assert sipnet == {
        "optimum_photosynthesis_temperature": 22.0,
        "min_photosynthesis_temperature": -1.5,
    }


def test_respiration_map_reproduces_each_rate_at_10c():
    rates = {"wood": (0.0155, 1.8), "fine_root": (0.6, 2.7), "coarse_root": (0.02, 3.2)}
    sipnet = respiration_rates_to_sipnet(
        **{f"{pool}_rate_at_10c": rate for pool, (rate, _) in rates.items()},
        **{f"{pool}_q10": q10 for pool, (_, q10) in rates.items()},
    )
    for pool, (rate, q10) in rates.items():
        base = sipnet[f"base_{pool}_respiration_rate"]
        assert base * q10 ** (10 / 10) == pytest.approx(rate, rel=1e-12)
        assert sipnet[f"{pool}_respiration_q10"] == q10


def test_soil_respiration_map_reproduces_the_flux_at_10c():
    soil_carbon, flux, q10 = 10_000.0, 424.0, 2.1
    sipnet = soil_respiration_to_sipnet(flux, q10, soil_carbon=soil_carbon)
    flux_at_10c = soil_carbon * sipnet["base_soil_respiration_rate"] * q10 ** (10 / 10)
    assert flux_at_10c == pytest.approx(flux, rel=1e-12)
    assert sipnet["soil_respiration_q10"] == q10


def test_allocation_map_takes_successive_shares():
    sipnet = allocation_to_sipnet(0.2, 0.5, 0.25)
    assert sipnet["leaf_allocation"] == 0.2
    assert sipnet["wood_allocation"] == pytest.approx(0.4, rel=1e-12)
    assert sipnet["fine_root_allocation"] == pytest.approx(0.1, rel=1e-12)


@pytest.mark.parametrize("shares", [(1e-9, 1e-9, 1e-9), (1 - 1e-9, 1 - 1e-9, 1 - 1e-9)])
def test_allocation_map_is_valid_at_extreme_shares(shares):
    fractions = allocation_to_sipnet(*shares).values()
    assert all(fraction >= 0 for fraction in fractions)
    assert sum(fractions) <= 1


def test_initial_pools_map_reproduces_each_pool():
    wood, coarse_roots, fine_roots = 8600.0, 1700.0, 300.0
    sipnet = initial_pools_to_sipnet(wood, coarse_roots, fine_roots)
    total = sipnet["total_wood_carbon"]
    assert total == pytest.approx(10_600.0, rel=1e-12)
    assert total * sipnet["coarse_root_fraction"] == pytest.approx(coarse_roots, rel=1e-12)
    assert total * sipnet["fine_root_fraction"] == pytest.approx(fine_roots, rel=1e-12)
    remaining = 1 - sipnet["coarse_root_fraction"] - sipnet["fine_root_fraction"]
    assert total * remaining == pytest.approx(wood, rel=1e-12)


def test_the_prior_table_names_exactly_the_calibration_parameters(data, theta):
    assert set(data.prior_table.index) == theta.keys()


def test_calibration_and_fixed_values_set_every_sipnet_parameter(theta, fixed):
    assert len(theta) + len(fixed) == len(SIPNET_PARAMETERS) == 58
    assert calibration_to_sipnet(theta, fixed).keys() == SIPNET_PARAMETERS


def test_every_calibration_parameter_reaches_sipnet(theta, fixed):
    at_theta = calibration_to_sipnet(theta, fixed)
    for name in theta:
        assert calibration_to_sipnet(theta | {name: 0.6}, fixed) != at_theta, name


def test_a_parameter_both_calibrated_and_fixed_is_refused(theta, fixed):
    with pytest.raises(ValueError, match="both calibrated and fixed: \\['leaf_off_day'\\]"):
        calibration_to_sipnet(theta, fixed | {"leaf_off_day": 285.0})


def test_a_parameter_neither_calibrated_nor_fixed_is_refused(theta, fixed):
    without = {name: value for name, value in fixed.items() if name != "leaf_on_day"}
    with pytest.raises(ValueError, match="neither calibrated nor fixed: \\['leaf_on_day'\\]"):
        calibration_to_sipnet(theta, without)


def test_a_fixed_value_that_is_not_a_sipnet_parameter_is_refused(theta, fixed):
    with pytest.raises(ValueError, match="not SIPNET parameters: \\['leaf_of_day'\\]"):
        calibration_to_sipnet(theta, fixed | {"leaf_of_day": 285.0})
