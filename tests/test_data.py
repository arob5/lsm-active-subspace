import numpy as np
import pandas as pd

from lsmas.data import load_harvard_forest


def test_shipped_files_are_consistent():
    data = load_harvard_forest()
    assert data.operator.shape == (400, 37_992)
    assert data.step_ends[0] == pd.Timestamp("2012-01-01 00:00")
    assert data.step_ends[-1] == pd.Timestamp("2024-12-31 21:00")
    assert data.driver_path.is_file()


def test_feature_classes_match_the_likelihood_description():
    data = load_harvard_forest()
    classes, counts = np.unique(data.feature_classes, return_counts=True)
    assert dict(zip(classes, counts, strict=True)) == {"day": 214, "night": 186}
    assert data.observations.shape == (400,)
    assert np.all(np.isfinite(data.observations))


def test_each_feature_lies_inside_its_recorded_windows():
    data = load_harvard_forest()
    for row, feature in enumerate(data.features.itertuples()):
        columns = data.operator.indices[data.operator.indptr[row] : data.operator.indptr[row + 1]]
        window_ends = data.step_ends[columns]
        assert window_ends.min() == feature.first_step_end_utc
        assert window_ends.max() == feature.last_step_end_utc


def test_fixed_values_are_keyed_by_pysipnet_names():
    fixed = load_harvard_forest().fixed_sipnet_parameters
    assert fixed["leaf_carbon_fraction"] == 0.466075
    assert all(isinstance(value, float) for value in fixed.values())
