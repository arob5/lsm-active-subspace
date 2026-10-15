import pytest

from lsmas.parameters import photosynthesis_to_sipnet, soil_respiration_to_sipnet


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


def test_soil_respiration_map_reproduces_the_flux_at_10c():
    soil_carbon, flux, q10 = 10_000.0, 424.0, 2.1
    sipnet = soil_respiration_to_sipnet(flux, q10, soil_carbon=soil_carbon)
    flux_at_10c = soil_carbon * sipnet["base_soil_respiration_rate"] * q10 ** (10 / 10)
    assert flux_at_10c == pytest.approx(flux, rel=1e-12)
    assert sipnet["soil_respiration_q10"] == q10
