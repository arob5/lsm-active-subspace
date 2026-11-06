import numpy as np
import pandas as pd
import pytest
from probpipe import Normal, sample, workflow_run

from lsmas._stopgaps.quantiles import median
from lsmas.data import load_harvard_forest
from lsmas.priors import (
    independent_prior,
    laws_from_table,
    log_normal_from_interval,
    logit_normal_from_interval,
)

TABLE = pd.DataFrame(
    {
        "family": ["log_normal", "logit_normal", "normal", "uniform"],
        "lower": [140.0, 0.04, np.nan, 0.0],
        "upper": [450.0, 0.20, np.nan, 2.0],
        "mean": [np.nan, np.nan, 22.0, np.nan],
        "sd": [np.nan, np.nan, 2.5, np.nan],
    },
    index=pd.Index(["capacity", "share", "temperature", "fraction"], name="parameter"),
)


@pytest.mark.parametrize(
    ("constructor", "lower", "upper"),
    [(log_normal_from_interval, 140.0, 450.0), (logit_normal_from_interval, 0.04, 0.20)],
)
def test_central_interval_holds_the_stated_probability(constructor, lower, upper):
    law = constructor("x", lower, upper, probability=0.9)
    with workflow_run(seed=0):
        draws = np.asarray(sample(law, 200_000).raw)
    np.testing.assert_allclose(np.quantile(draws, [0.05, 0.95]), [lower, upper], rtol=0.01)


def test_interval_must_be_ordered():
    with pytest.raises(ValueError, match="lower < upper"):
        log_normal_from_interval("x", 5.0, 2.0)


def test_each_row_gives_its_familys_law_with_the_exact_median():
    laws = laws_from_table(TABLE)
    assert list(laws) == list(TABLE.index)
    logit_centre = (np.log(0.04 / 0.96) + np.log(0.20 / 0.80)) / 2
    expected = {
        "capacity": np.sqrt(140.0 * 450.0),
        "share": 1 / (1 + np.exp(-logit_centre)),
        "temperature": 22.0,
        "fraction": 1.0,
    }
    for name, law in laws.items():
        assert median(law) == pytest.approx(expected[name], rel=1e-12), name


def test_each_row_gives_its_familys_spread():
    with workflow_run(seed=1):
        draws = {
            name: np.asarray(sample(law, 100_000).raw)
            for name, law in laws_from_table(TABLE).items()
        }
    np.testing.assert_allclose(
        np.quantile(draws["capacity"], [0.025, 0.975]), [140, 450], rtol=0.02
    )
    np.testing.assert_allclose(np.quantile(draws["share"], [0.025, 0.975]), [0.04, 0.2], rtol=0.02)
    assert np.std(draws["temperature"]) == pytest.approx(2.5, rel=0.02)
    assert (draws["fraction"].min(), draws["fraction"].max()) == pytest.approx((0, 2), abs=1e-3)


def test_a_row_must_give_exactly_its_familys_numbers():
    with pytest.raises(ValueError, match=r"temperature: family 'normal' needs \['sd'\]"):
        laws_from_table(TABLE.assign(sd=np.nan))
    with pytest.raises(ValueError, match=r"capacity: family 'log_normal' does not use \['mean'\]"):
        laws_from_table(TABLE.assign(mean=1.0).loc[["capacity"]])
    with pytest.raises(ValueError, match="unknown family 'gamma'"):
        laws_from_table(TABLE.assign(family="gamma"))


def test_independent_prior_groups_the_laws():
    prior = independent_prior({"a": Normal("a", 0.0, 1.0), "b": Normal("b", 5.0, 1.0)})
    assert list(prior.event_spec.outputs) == ["theta"]
    with workflow_run(seed=2):
        draws = sample(prior, 1000)
    assert np.mean(np.asarray(draws["theta"]["b"].raw)) == pytest.approx(5.0, abs=0.2)


def test_shipped_prior_medians_are_the_designs():
    medians = {
        name: median(law)
        for name, law in laws_from_table(load_harvard_forest().prior_table).items()
    }
    assert len(medians) == 46
    assert medians["photosynthetic_capacity"] == pytest.approx(251.0, rel=1e-3)
    assert medians["respiration_share"] == pytest.approx(0.0926, rel=1e-3)
    assert medians["leaf_off_day"] == 285.0
    assert medians["frozen_soil_water_fraction"] == 0.5
