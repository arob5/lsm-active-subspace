import numpy as np
import pytest

from lsmas.data import load_harvard_forest
from lsmas.learning_rate import effective_sample_size_learning_rate


def test_rate_counts_months_not_features():
    # Six months with two features each: n = 12 features over M = 6 months.
    classes = np.array(["day"] * 12)
    years = np.full(12, 2015)
    months = np.repeat(np.arange(1, 7), 2)
    rates = effective_sample_size_learning_rate(
        classes, years, months, observations_per_class_year=3.0
    )
    np.testing.assert_allclose(rates, 3.0 * 6 / (12 * 12))


def test_rate_is_capped_at_one():
    rates = effective_sample_size_learning_rate(
        np.array(["night"]), np.array([2015]), np.array([1]), observations_per_class_year=24.0
    )
    assert rates.tolist() == [1.0]


def test_harvard_forest_rates_match_the_working_likelihood():
    data = load_harvard_forest()
    rates = effective_sample_size_learning_rate(
        data.feature_classes,
        data.features["year"],
        data.features["month"],
        observations_per_class_year=3.0,
    )
    assert rates[data.feature_classes == "day"][0] == pytest.approx(3 * 108 / (12 * 214))
    assert rates[data.feature_classes == "night"][0] == pytest.approx(3 * 102 / (12 * 186))
    assert round(rates[data.feature_classes == "day"][0], 3) == 0.126
    assert round(rates[data.feature_classes == "night"][0], 3) == 0.137
