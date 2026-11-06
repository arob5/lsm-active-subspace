"""Print the design doc's tables of calibration parameters and priors from prior_table.py."""

from prior_table import ROWS

REACHES = {
    "photosynthetic_capacity": "`aMax = P·cFracLeaf·(1−ρ)/aMaxFrac`",
    "respiration_share": "`baseFolRespFrac = ρ·aMaxFrac/(1−ρ)`",
    "optimum_photosynthesis_temperature": "`psnTOpt`",
    "photosynthesis_temperature_range": "`psnTMin = psnTOpt − range`",
    "half_saturation_light": "`halfSatPar`",
    "vapor_pressure_deficit_slope": "`dVpdSlope`",
    "vapor_pressure_deficit_exponent": "`dVpdExp`",
    "light_extinction_coefficient": "`attenuation` (with `leafCSpWt` fixed)",
    "water_use_efficiency": "`wueConst`",
    "water_removal_fraction": "`waterRemoveFrac`",
    "soil_water_holding_capacity": "`soilWHC`",
    "interception_evaporation_fraction": "`immedEvapFrac`",
    "fast_flow_fraction": "`fastFlowFrac`",
    "snow_melt_rate": "`snowMelt`",
    "aerodynamic_resistance_constant": "`rdConst`",
    "soil_resistance_intercept": "`rSoilConst1`",
    "soil_resistance_slope": "`rSoilConst2`",
    "leaf_on_growing_degree_days": "`gddLeafOn`",
    "leaf_off_day": "`leafOffDay`",
    "leaf_on_growth": "`leafGrowth`",
    "leaf_off_fall_fraction": "`fracLeafFall`",
    "leaf_turnover_rate": "`leafTurnoverRate`",
    "leaf_allocation": "`leafAllocation = ℓ`",
    "wood_share_of_nonleaf_allocation": "`woodAllocation = (1−ℓ)·w`",
    "fine_root_share_of_root_allocation": "`fineRootAllocation = (1−ℓ)(1−w)·f`; coarse roots get the rest",
    "wood_turnover_rate": "`woodTurnoverRate`",
    "fine_root_turnover_rate": "`fineRootTurnoverRate`",
    "coarse_root_turnover_rate": "`coarseRootTurnoverRate`",
    "wood_respiration_rate_at_10c": "`baseVegResp = r10/vegRespQ10`",
    "wood_respiration_q10": "`vegRespQ10` (wood and foliage)",
    "fine_root_respiration_rate_at_10c": "`baseFineRootResp = r10/fineRootQ10`",
    "fine_root_respiration_q10": "`fineRootQ10`",
    "coarse_root_respiration_rate_at_10c": "`baseCoarseRootResp = r10/coarseRootQ10`",
    "coarse_root_respiration_q10": "`coarseRootQ10`",
    "soil_respiration_flux_at_10c": "`baseSoilResp = F10/(soilInit·soilRespQ10)`",
    "soil_respiration_q10": "`soilRespQ10`",
    "soil_respiration_moisture_exponent": "`soilRespMoistEffect`",
    "frozen_soil_threshold": "`frozenSoilThreshold`",
    "frozen_soil_foliar_respiration_factor": "`frozenSoilFolREff`",
    "frozen_soil_water_fraction": "`frozenSoilEff`",
    "initial_wood_carbon": "`plantWoodInit = W + C + F`",
    "initial_coarse_root_carbon": "`coarseRootFrac = C/(W + C + F)`",
    "initial_fine_root_carbon": "`fineRootFrac = F/(W + C + F)`",
    "initial_soil_carbon": "`soilInit`",
    "initial_soil_wetness": "`soilWFracInit`",
    "initial_snow_water_equivalent": "`snowInit`",
}
assert set(REACHES) == {row.name for row in ROWS}


def prior_text(row):
    if row.family == "normal":
        return f"N({row.a:g}, {row.b:g}²)"
    if row.family == "uniform":
        return f"U({row.a:g}, {row.b:g})"
    kind = "log-normal" if row.family == "lognormal" else "logit-normal"
    return f"{kind}, 95% in [{row.a:g}, {row.b:g}]"


print("| # | parameter | units | reaches SIPNET as | prior | median | source |")
print("|---|---|---|---|---|---|---|")
for index, row in enumerate(ROWS, 1):
    print(f"| {index} | `{row.name}` | {row.units} | {REACHES[row.name]} | {prior_text(row)} | {row.median():.3g} | {row.source} |")
