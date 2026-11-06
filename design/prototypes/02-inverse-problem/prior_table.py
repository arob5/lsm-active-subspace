"""Draft prior over the 46 calibration parameters, and their map to SIPNET's 58 parameters.

Each row: name, units, family, the family's two numbers, and where they come from. Families:

- ``lognormal``  : central 95% interval [lower, upper] of a log-normal law;
- ``logitnormal``: central 95% interval [lower, upper] of a logit-normal law on (0, 1);
- ``normal``     : mean and standard deviation;
- ``uniform``    : lower and upper bound.

Sources: P2 = the phase-2 prior (design/inputs/.../prior_proposal.csv); BETY = the
temperate-deciduous 2.5-97.5% after PEcAn's conversion (base_parameter_candidates.csv);
T = PEcAn's SIPNET template; IC = the site's initial-condition ensemble; R = reasoned here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

Z975 = stats.norm.ppf(0.975)


@dataclass(frozen=True)
class Row:
    name: str
    units: str
    family: str
    a: float
    b: float
    source: str

    def location_scale(self):
        if self.family in ("normal", "uniform"):
            return self.a, self.b
        forward = np.log if self.family == "lognormal" else special.logit
        low, high = forward(self.a), forward(self.b)
        return (low + high) / 2, (high - low) / (2 * Z975)

    def from_standard(self, z):
        if self.family == "uniform":
            return self.a + (self.b - self.a) * stats.norm.cdf(z)
        location, scale = self.location_scale()
        x = location + scale * z
        if self.family == "lognormal":
            return np.exp(x)
        if self.family == "logitnormal":
            return special.expit(x)
        return x

    def median(self):
        return float(self.from_standard(0.0))


ROWS = [
    # --- photosynthesis and canopy
    Row("photosynthetic_capacity", "nmol g-1 s-1", "lognormal", 140, 450, "P2"),
    Row("respiration_share", "1", "logitnormal", 0.04, 0.20, "P2"),
    Row("optimum_photosynthesis_temperature", "degC", "normal", 22.0, 2.5, "P2"),
    Row("photosynthesis_temperature_range", "degC", "lognormal", 17.0, 31.0, "BETY psnTMin 2.5-97.5% (-6.5, 6.6) below psnTOpt 24"),
    Row("half_saturation_light", "mol m-2 d-1", "lognormal", 4.6, 26.3, "BETY"),
    Row("vapor_pressure_deficit_slope", "kPa-1", "lognormal", 0.02, 0.12, "BETY median 0.050; R interval"),
    Row("vapor_pressure_deficit_exponent", "1", "lognormal", 1.0, 3.0, "Niwot range [1, 3]; BETY 1.5, Braswell 2"),
    Row("light_extinction_coefficient", "1", "lognormal", 0.38, 0.65, "BETY attenuation widened by BETY leafCSpWt spread (leafCSpWt fixed)"),
    # --- water
    Row("water_use_efficiency", "mg g-1 kPa", "lognormal", 7.9, 17.5, "BETY"),
    Row("water_removal_fraction", "d-1", "lognormal", 0.03, 0.25, "T 0.088; R interval"),
    Row("soil_water_holding_capacity", "cm", "lognormal", 15.0, 150.0, "P2"),
    Row("interception_evaporation_fraction", "1", "logitnormal", 0.03, 0.30, "T 0.1; R interval"),
    Row("fast_flow_fraction", "1", "logitnormal", 0.005, 0.30, "T 0; R interval"),
    Row("snow_melt_rate", "cm K-1 d-1", "lognormal", 0.05, 0.45, "T 0.15; R interval"),
    Row("aerodynamic_resistance_constant", "1", "lognormal", 941.0, 1377.0, "BETY"),
    Row("soil_resistance_intercept", "1", "normal", 8.2, 1.0, "T 8.2; R sd"),
    Row("soil_resistance_slope", "1", "lognormal", 2.0, 9.0, "T 4.3; R interval"),
    # --- phenology and leaves
    Row("leaf_on_growing_degree_days", "K d", "lognormal", 500.0, 1100.0, "P2"),
    Row("leaf_off_day", "d", "normal", 285.0, 7.0, "T 285; Harvard leaf fall DOY ~280-300 (approximate)"),
    Row("leaf_on_growth", "g m-2", "lognormal", 50.0, 180.0, "P2"),
    Row("leaf_off_fall_fraction", "1", "logitnormal", 0.990, 0.9997, "BETY"),
    Row("leaf_turnover_rate", "yr-1", "lognormal", 0.05, 0.35, "T 0.13; R interval (BETY 0.75 double-counts leaf fall)"),
    # --- allocation, by stick-breaking: leaf; wood share of the rest; fine-root share of the rest
    Row("leaf_allocation", "1", "logitnormal", 0.142, 0.217, "BETY"),
    Row("wood_share_of_nonleaf_allocation", "1", "logitnormal", 0.40, 0.70, "P2 centre 0.45/0.82 = 0.55; R interval"),
    Row("fine_root_share_of_root_allocation", "1", "logitnormal", 0.08, 0.35, "P2 centre 0.065/0.37 = 0.18; R interval"),
    # --- turnover
    Row("wood_turnover_rate", "yr-1", "lognormal", 0.00303, 0.00715, "BETY"),
    Row("fine_root_turnover_rate", "yr-1", "lognormal", 0.346, 7.19, "BETY"),
    Row("coarse_root_turnover_rate", "yr-1", "lognormal", 0.02, 0.15, "T 0.056; R interval"),
    # --- respiration, rates referenced to 10 C
    Row("wood_respiration_rate_at_10c", "yr-1", "lognormal", 0.006, 0.04, "P2"),
    Row("wood_respiration_q10", "1", "lognormal", 1.449, 2.211, "BETY (also foliar respiration's Q10)"),
    Row("fine_root_respiration_rate_at_10c", "yr-1", "lognormal", 0.15, 2.4, "BETY 0.186 yr-1 at 0 C x Q10 3.2 = 0.6; R factor-4 interval"),
    Row("fine_root_respiration_q10", "1", "lognormal", 1.511, 4.899, "BETY"),
    Row("coarse_root_respiration_rate_at_10c", "yr-1", "lognormal", 0.006, 0.04, "P2 wood rate (coarse roots are wood), untied"),
    Row("coarse_root_respiration_q10", "1", "lognormal", 1.521, 4.887, "BETY"),
    Row("soil_respiration_flux_at_10c", "g m-2 yr-1", "lognormal", 200.0, 900.0, "P2"),
    Row("soil_respiration_q10", "1", "lognormal", 1.3, 3.2, "P2"),
    Row("soil_respiration_moisture_exponent", "1", "lognormal", 0.3, 3.0, "T 1; R interval"),
    # --- frozen soil (live only in some prior draws: early leaf-on or late leaf-off over cold soil)
    Row("frozen_soil_threshold", "degC", "normal", 0.0, 1.5, "T 0; Niwot range [-5, 5]; R sd"),
    Row("frozen_soil_foliar_respiration_factor", "1", "uniform", 0.0, 1.0, "T 0; no information"),
    Row("frozen_soil_water_fraction", "1", "uniform", 0.0, 1.0, "T 1, Niwot 0; no information"),
    # --- initial conditions (1 January 2012)
    Row("initial_wood_carbon", "g m-2", "lognormal", 5000.0, 15000.0, "IC median 8633; R interval"),
    Row("initial_coarse_root_carbon", "g m-2", "lognormal", 800.0, 3500.0, "phase 1 1705; R interval"),
    Row("initial_fine_root_carbon", "g m-2", "lognormal", 100.0, 800.0, "phase 1 320; R interval"),
    Row("initial_soil_carbon", "g m-2", "lognormal", 7000.0, 30000.0, "IC median 15000; R interval"),
    Row("initial_soil_wetness", "1", "logitnormal", 0.30, 0.85, "IC median 0.61; R interval"),
    Row("initial_snow_water_equivalent", "cm", "lognormal", 0.1, 8.0, "R"),
]
NAMES = [row.name for row in ROWS]
assert len(ROWS) == 46, len(ROWS)

#: SIPNET parameters that are not calibrated: inert, or fixed to remove an exact degeneracy.
FIXED = {
    # exact degeneracies
    "daily_mean_photosynthesis_fraction": 0.860623,
    "leaf_carbon_fraction": 0.466075,
    "leaf_carbon_per_area": 30.7092,
    "leaf_area_index": 0.0,
    # inert under the model options
    "leaf_on_day": 144.0,
    "leaf_on_soil_temperature": 12.0,
    "growth_respiration_fraction": 0.272493,
    "litter_carbon": 0.0,
    "litter_breakdown_rate": 0.4,
    "litter_respired_fraction": 0.5,
    "leaf_water_pool_depth": 0.200384,
    # inert at this site: its cap on leaf-on growth never binds
    "leaf_on_reallocation_fraction": 0.2,
}


def to_sipnet(theta: dict) -> dict:
    """All 58 SIPNET parameters (pySIPNET names) at calibration parameters *theta*."""
    f = FIXED
    p, rho = theta["photosynthetic_capacity"], theta["respiration_share"]
    leaf = theta["leaf_allocation"]
    wood = (1 - leaf) * theta["wood_share_of_nonleaf_allocation"]
    fine = (1 - leaf - wood) * theta["fine_root_share_of_root_allocation"]
    pools = theta["initial_wood_carbon"], theta["initial_coarse_root_carbon"], theta["initial_fine_root_carbon"]
    total_wood = sum(pools)
    soil_c = theta["initial_soil_carbon"]
    copied = [
        "half_saturation_light", "vapor_pressure_deficit_slope", "vapor_pressure_deficit_exponent",
        "light_extinction_coefficient", "water_use_efficiency", "water_removal_fraction",
        "soil_water_holding_capacity", "interception_evaporation_fraction", "fast_flow_fraction",
        "snow_melt_rate", "aerodynamic_resistance_constant", "soil_resistance_intercept",
        "soil_resistance_slope", "leaf_on_growing_degree_days", "leaf_off_day", "leaf_on_growth",
        "leaf_off_fall_fraction", "leaf_turnover_rate", "leaf_allocation", "wood_turnover_rate",
        "fine_root_turnover_rate", "coarse_root_turnover_rate", "wood_respiration_q10",
        "fine_root_respiration_q10", "coarse_root_respiration_q10", "soil_respiration_q10",
        "soil_respiration_moisture_exponent", "frozen_soil_threshold",
        "frozen_soil_foliar_respiration_factor", "frozen_soil_water_fraction",
    ]
    out = dict(f) | {name: float(theta[name]) for name in copied} | {
        "max_photosynthesis_rate": p * f["leaf_carbon_fraction"] * (1 - rho) / f["daily_mean_photosynthesis_fraction"],
        "foliar_respiration_fraction": rho * f["daily_mean_photosynthesis_fraction"] / (1 - rho),
        "optimum_photosynthesis_temperature": theta["optimum_photosynthesis_temperature"],
        "min_photosynthesis_temperature": theta["optimum_photosynthesis_temperature"] - theta["photosynthesis_temperature_range"],
        "wood_allocation": wood,
        "fine_root_allocation": fine,
        "base_wood_respiration_rate": theta["wood_respiration_rate_at_10c"] / theta["wood_respiration_q10"],
        "base_fine_root_respiration_rate": theta["fine_root_respiration_rate_at_10c"] / theta["fine_root_respiration_q10"],
        "base_coarse_root_respiration_rate": theta["coarse_root_respiration_rate_at_10c"] / theta["coarse_root_respiration_q10"],
        "base_soil_respiration_rate": theta["soil_respiration_flux_at_10c"] / (soil_c * theta["soil_respiration_q10"]),
        "total_wood_carbon": total_wood,
        "coarse_root_fraction": pools[1] / total_wood,
        "fine_root_fraction": pools[2] / total_wood,
        "soil_carbon": soil_c,
        "soil_wetness_fraction": theta["initial_soil_wetness"],
        "snow_water_equivalent": theta["initial_snow_water_equivalent"],
    }
    assert len(out) == 58, len(out)
    return out


def draw(n: int, seed: int = 0) -> list[dict]:
    z = np.random.default_rng(seed).standard_normal((n, len(ROWS)))
    return [{row.name: float(row.from_standard(z[i, j])) for j, row in enumerate(ROWS)} for i in range(n)]


def medians() -> dict:
    return {row.name: row.median() for row in ROWS}
