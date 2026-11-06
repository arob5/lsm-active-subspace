"""Shared pieces of the design-02 prototype: the reference point, SIPNET runs, and the features.

The reference point sets all 58 pySIPNET parameters. It was assembled from the fixed values shipped
before design 02, the retired development problem's representative values, its four calibration
parameters at their prior medians, and the initial soil wetness of the site's initial-condition
ensemble (0.611), which the development problem left at pySIPNET's Niwot default. It is written out
below so that the scripts reproduce after both changed.
"""

from __future__ import annotations

import csv
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

#: All 58 SIPNET parameters at the reference point, keyed by pySIPNET's names.
REFERENCE_POINT = {
    "aerodynamic_resistance_constant": 1150.07,
    "base_coarse_root_respiration_rate": 0.0048344738532939925,
    "base_fine_root_respiration_rate": 0.185785,
    "base_soil_respiration_rate": 0.020384615384615383,
    "base_wood_respiration_rate": 0.008566186223361924,
    "coarse_root_fraction": 0.16,
    "coarse_root_respiration_q10": 3.20614,
    "coarse_root_turnover_rate": 0.056,
    "daily_mean_photosynthesis_fraction": 0.860623,
    "fast_flow_fraction": 0.0,
    "fine_root_allocation": 0.065,
    "fine_root_fraction": 0.03,
    "fine_root_respiration_q10": 3.19766,
    "fine_root_turnover_rate": 1.55048,
    "foliar_respiration_fraction": 0.0899215107134968,
    "frozen_soil_foliar_respiration_factor": 0.0,
    "frozen_soil_threshold": 0.0,
    "frozen_soil_water_fraction": 1.0,
    "growth_respiration_fraction": 0.272493,
    "half_saturation_light": 11.0,
    "interception_evaporation_fraction": 0.1,
    "leaf_allocation": 0.18,
    "leaf_area_index": 0.0,
    "leaf_carbon_fraction": 0.466075,
    "leaf_carbon_per_area": 30.7092,
    "leaf_off_day": 285.0,
    "leaf_off_fall_fraction": 0.995004,
    "leaf_on_day": 144.0,
    "leaf_on_growing_degree_days": 741.6,
    "leaf_on_growth": 94.9,
    "leaf_on_reallocation_fraction": 0.2,
    "leaf_on_soil_temperature": 12.0,
    "leaf_turnover_rate": 0.13,
    "leaf_water_pool_depth": 0.200384,
    "light_extinction_coefficient": 0.500493,
    "litter_breakdown_rate": 0.4,
    "litter_carbon": 0.0,
    "litter_respired_fraction": 0.5,
    "max_photosynthesis_rate": 123.56170455588568,
    "min_photosynthesis_temperature": -1.9579999999999984,
    "optimum_photosynthesis_temperature": 22.0,
    "snow_melt_rate": 0.15,
    "snow_water_equivalent": 0.0,
    "soil_carbon": 10000.0,
    "soil_resistance_intercept": 8.2,
    "soil_resistance_slope": 4.3,
    "soil_respiration_moisture_exponent": 1.0,
    "soil_respiration_q10": 2.08,
    "soil_water_holding_capacity": 47.4,
    "soil_wetness_fraction": 0.610879,
    "total_wood_carbon": 12345.67901234568,
    "vapor_pressure_deficit_exponent": 2.0,
    "vapor_pressure_deficit_slope": 0.049919,
    "water_removal_fraction": 0.088,
    "water_use_efficiency": 12.0976,
    "wood_allocation": 0.45,
    "wood_respiration_q10": 1.80944,
    "wood_turnover_rate": 0.004821,
}


def reference_point() -> dict[str, float]:
    return dict(REFERENCE_POINT)


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
