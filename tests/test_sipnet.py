import logging

import numpy as np
import pandas as pd
import pytest
from probpipe import NumericRecordBatch, quantile

from lsmas.data import load_harvard_forest
from lsmas.parameters import calibration_to_sipnet, photosynthesis_to_sipnet
from lsmas.priors import independent_prior, laws_from_table
from lsmas.sipnet import SIPNETSimulator, feature_map

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def data():
    return load_harvard_forest()


@pytest.fixture(scope="module")
def simulator(data):
    """SIPNET with every calibration parameter at its prior median."""
    at_medians = calibration_to_sipnet(
        quantile(independent_prior(laws_from_table(data.prior_table)), 0.5)["theta"],
        data.fixed_sipnet_parameters,
    )
    return SIPNETSimulator(data.driver_path, at_medians)


def test_output_steps_are_the_observation_operators_columns(data, simulator):
    totals = simulator.run_nee({})
    assert pd.DatetimeIndex(totals["time"].values).equals(data.step_ends)


def test_a_run_holds_every_output_on_the_same_steps(data, simulator):
    outputs = simulator.run({})
    assert {"net_ecosystem_exchange", "gross_primary_production", "leaf_carbon"} <= set(
        outputs.data_vars
    )
    assert pd.DatetimeIndex(outputs["time"].values).equals(data.step_ends)
    nee = simulator.run_nee({})
    assert outputs["net_ecosystem_exchange"].equals(nee)
    assert "bounds" not in nee["time"].attrs  # it has no time_bounds to point to


def test_a_parameter_left_unset_is_refused(data):
    partial = SIPNETSimulator(data.driver_path, data.fixed_sipnet_parameters)
    with pytest.raises(ValueError, match=r"not set: .*'leaf_off_day'"):
        partial.run_nee({})


def test_rate_is_the_step_total_per_second_in_micromoles(simulator):
    totals = simulator.run_nee({}).values
    expected = totals / (3 * 3600) * 1e6 / 12.011
    np.testing.assert_allclose(simulator.nee_rate({}), expected, rtol=1e-12)


def test_leaf_carbon_fraction_is_redundant_given_capacity_and_share(data, simulator):
    fixed = data.fixed_sipnet_parameters

    def nee_with_carbon_fraction(carbon_fraction):
        parameters = photosynthesis_to_sipnet(
            250.0,
            0.09,
            daily_mean_photosynthesis_fraction=fixed["daily_mean_photosynthesis_fraction"],
            leaf_carbon_fraction=carbon_fraction,
        )
        return simulator.nee_rate(parameters | {"leaf_carbon_fraction": carbon_fraction})

    np.testing.assert_array_equal(nee_with_carbon_fraction(0.466), nee_with_carbon_fraction(0.40))


def _to_sipnet(theta):
    return {"optimum_photosynthesis_temperature": float(theta["optimum_temperature"])}


def test_feature_map_runs_a_batch_in_threads_as_it_would_in_sequence(data, simulator):
    forward = feature_map(
        simulator, data.operator, _to_sipnet, parameter_names=["optimum_temperature"], max_workers=4
    )
    batch = NumericRecordBatch({"optimum_temperature": np.array([18.0, 22.0, 26.0])}, "member")
    threaded = np.asarray(forward(batch).raw)
    sequential = np.asarray(forward.using(executor="sequential")(batch).raw)
    assert threaded.shape == (3, 400)
    np.testing.assert_array_equal(threaded, sequential)
    assert not np.allclose(threaded[0], threaded[2])


def test_a_failed_run_gives_a_row_of_nan_and_a_warning(data, simulator, caplog):
    def impossible_allocation(theta):
        return {
            "leaf_allocation": float(theta["leaf"]),
            "wood_allocation": 0.5,
            "fine_root_allocation": 0.1,
        }

    forward = feature_map(simulator, data.operator, impossible_allocation, parameter_names=["leaf"])
    batch = NumericRecordBatch({"leaf": np.array([0.1, 0.9])}, "member")
    with caplog.at_level(logging.WARNING, logger="lsmas.sipnet"):
        features = np.asarray(forward(batch).raw)
    assert np.all(np.isfinite(features[0]))
    assert np.all(np.isnan(features[1]))
    assert "SIPNET produced no output" in caplog.text
    assert "ValidationError" in caplog.text and "allocation" in caplog.text


def test_the_forward_map_declares_its_parameter_record(data, simulator):
    forward = feature_map(
        simulator, data.operator, _to_sipnet, parameter_names=["optimum_temperature"]
    )
    assert list(forward.input_spec["theta"].fields) == ["optimum_temperature"]
