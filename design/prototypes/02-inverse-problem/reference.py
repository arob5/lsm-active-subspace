"""Shared pieces of the design-02 prototype: the reference point, SIPNET runs, and the features.

The reference point sets all 58 pySIPNET parameters: the shipped fixed values, the development
problem's representative values, its four calibration parameters at their prior medians, and the
initial soil wetness of the site's initial-condition ensemble (0.611), which the development problem
leaves at pySIPNET's Niwot default.
"""

from __future__ import annotations

import csv
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lsmas  # noqa: F401  (EnsKit first)
import numpy as np
from lsmas.data import load_harvard_forest
from lsmas.sipnet import SIMULATION_FAILURES, SIPNETSimulator

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "design" / "inputs" / "harvard_forest" / "parameter_analysis"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

DATA = load_harvard_forest()
_dev = importlib.import_module("lsmas.problems.harvard_forest_dev")

DEV_MEDIANS = {
    "photosynthetic_capacity": 252.0,
    "respiration_share": 0.0946,
    "optimum_photosynthesis_temperature": 22.0,
    "soil_respiration_q10": 2.08,
}


def reference_point() -> dict[str, float]:
    fixed = DATA.fixed_sipnet_parameters | _dev._REPRESENTATIVE_SIPNET_VALUES
    derived = _dev._sipnet_parameters_from(fixed)(DEV_MEDIANS)
    return fixed | derived | {"soil_wetness_fraction": 0.610879}


def parameter_table() -> list[dict[str, str]]:
    """The 58 pySIPNET parameters with units, domain, BETY quantiles and flags (phase 1)."""
    return list(csv.DictReader(open(INPUTS / "base_parameter_candidates.csv")))


SIMULATOR = SIPNETSimulator(DATA.driver_path, {})
SIGMA = np.sqrt(np.where(DATA.feature_classes == "day", 8.0, 1.5))
ETA = np.where(DATA.feature_classes == "day", 0.126, 0.137)


def features(parameters: dict[str, float]) -> np.ndarray:
    """G at a full set of SIPNET parameters; NaN row if SIPNET refuses or fails."""
    try:
        return DATA.operator @ SIMULATOR.nee_rate(parameters)
    except SIMULATION_FAILURES:
        return np.full(DATA.operator.shape[0], np.nan)


def features_batch(points: list[dict[str, float]], workers: int = 8) -> np.ndarray:
    with ThreadPoolExecutor(workers) as pool:
        return np.stack(list(pool.map(features, points)))
