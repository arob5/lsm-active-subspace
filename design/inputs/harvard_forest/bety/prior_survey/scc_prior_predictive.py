"""Small prior-predictive smoke test on the SCC: three flux-tower sites, one per
reanalysis PFT, driver member 1, (A) PEcAn template parameters x 10 IC members,
(B) 30 PEcAn PFT trait samples mapped onto SIPNET parameters, paired with IC
member of the same index. Writes annual metrics and 2016 daily NEE to CSV.
"""
from __future__ import annotations
import sys, time, os
import numpy as np, pandas as pd, xarray as xr

A = "/projectnb/dietzelab/arober/spatial-lsm-calibration"
sys.path.insert(0, A + "/src")
OUT = "/projectnb/dietzelab/arober/scratch/prior_survey"
SAMPLES = OUT

from pysipnet import ClimateDrivers, ModelFlags, SIPNETRunner, SIPNETModel
from pysipnet.runner import ClimateStaging, SIPNETRunError
from pysipnet.parameters import (
    AllocationParams, InitialConditions, LeafPhysiologyParams, PhenologyParams,
    PhotosynthesisParams, RespirationParams, SIPNETParameters, WaterParams,
)
from sipnet_calibration.initial_conditions import load_initial_conditions
from sipnet_calibration.drivers import driver_file

SITES = {4253: "temperate.deciduous.HPDA", 4480: "boreal.coniferous", 4102: "semiarid.grassland_HPDA"}
N_IC, N_TRAIT = 10, 30
FINE, COARSE = 0.2, 0.2   # template.param root fractions

def template_params(ic):
    return SIPNETParameters(
        initial_conditions=ic,
        photosynthesis=PhotosynthesisParams(max_photosynthesis_rate=112.0, daily_mean_photosynthesis_fraction=0.76,
            foliar_respiration_fraction=0.1, min_photosynthesis_temperature=2.0, optimum_photosynthesis_temperature=24.0,
            vapor_pressure_deficit_slope=0.05, vapor_pressure_deficit_exponent=2.0, half_saturation_light=17.0,
            light_extinction_coefficient=0.58),
        phenology=PhenologyParams(leaf_off_day=285.0, leaf_on_growing_degree_days=500.0, leaf_on_growth=126.0,
            leaf_off_fall_fraction=1.0, leaf_allocation=0.20, leaf_turnover_rate=0.13, leaf_on_reallocation_fraction=0.2),
        respiration=RespirationParams(base_wood_respiration_rate=0.006, wood_respiration_q10=2.0,
            frozen_soil_foliar_respiration_factor=0.0, frozen_soil_threshold=0.0, base_fine_root_respiration_rate=0.090,
            base_coarse_root_respiration_rate=0.006, fine_root_respiration_q10=2.6, coarse_root_respiration_q10=2.6,
            base_soil_respiration_rate=0.06, soil_respiration_q10=2.9, soil_respiration_moisture_exponent=1.0),
        allocation=AllocationParams(fine_root_allocation=0.40, wood_allocation=0.20, fine_root_turnover_rate=0.137,
            coarse_root_turnover_rate=0.056, wood_turnover_rate=0.014),
        water=WaterParams(water_removal_fraction=0.088, frozen_soil_water_fraction=1.0, water_use_efficiency=10.9,
            soil_water_holding_capacity=12.0, interception_evaporation_fraction=0.1, fast_flow_fraction=0.0,
            snow_melt_rate=0.15, aerodynamic_resistance_constant=300.0, soil_resistance_intercept=8.2,
            soil_resistance_slope=4.3),
        leaf=LeafPhysiologyParams(leaf_carbon_per_area=32.0, leaf_carbon_fraction=0.45),
    )

# trait -> SIPNET parameter (pySIPNET field names), per write.configs.SIPNET.R
def trait_overrides(row: pd.Series) -> dict:
    leafC = row.get("leafC", 48.0)
    sla = row["SLA"]
    conv = (44.0096 / 1e6) * (12.01 / 44.0096) / 1000 * 86400
    o = {
        "leaf_carbon_fraction": leafC * 0.01,
        "leaf_carbon_per_area": 10 * leafC / sla,
        "max_photosynthesis_rate": row["Amax"] * sla,
        "foliar_respiration_fraction": float(np.clip(row["leaf_respiration_rate_m2"] / row["Amax"], 0, 1)),
        "daily_mean_photosynthesis_fraction": row["AmaxFrac"],
        "min_photosynthesis_temperature": row["Vm_low_temp"],
        "optimum_photosynthesis_temperature": row["psnTOpt"],
        "light_extinction_coefficient": row["extinction_coefficient"],
        "half_saturation_light": row["half_saturation_PAR"],
        "vapor_pressure_deficit_slope": row["dVPDSlope"],
        "vapor_pressure_deficit_exponent": row["dVpdExp"],
        "wood_respiration_q10": row["veg_respiration_Q10"],
        "base_wood_respiration_rate": row["stem_respiration_rate"] * conv * row["veg_respiration_Q10"] ** -2.5,
        "fine_root_respiration_q10": row["fine_root_respiration_Q10"],
        "coarse_root_respiration_q10": row["coarse_root_respiration_Q10"],
        "base_fine_root_respiration_rate": row["root_respiration_rate"] * conv * row["fine_root_respiration_Q10"] ** -2.5,
        "fine_root_turnover_rate": row["root_turnover_rate"],
        "leaf_turnover_rate": row["leaf_turnover_rate"],
        "base_soil_respiration_rate": row["som_respiration_rate"],
        "leaf_off_fall_fraction": row["fracLeafFall"],
        "aerodynamic_resistance_constant": row["rdConst"],
    }
    for trait, field in (("wueConst", "water_use_efficiency"), ("wood_turnover_rate", "wood_turnover_rate"), ("wood_allocation_fraction", "wood_allocation"), ("leaf_allocation_fraction", "leaf_allocation"),
                         ("root_allocation_fraction", "fine_root_allocation"), ("leafGrowth", "leaf_on_growth"),
                         ("GDD", "leaf_on_growing_degree_days")):
        if trait in row.index:
            o[field] = row[trait]
    return {k: float(v) for k, v in o.items()}

def ic_from_product(ds, site, member, leaf_c_per_area, lai_fallback=0.0, swf_fallback=0.5):
    r = ds.sel(site=site, member=member)
    wood, soil = float(r.initial_wood_carbon), float(r.initial_soil_organic_carbon)
    leaf, swf = float(r.initial_leaf_carbon), float(r.initial_soil_moisture_saturation)
    return InitialConditions(
        total_wood_carbon=1000.0 * wood / (1 - FINE - COARSE),
        leaf_area_index=lai_fallback if np.isnan(leaf) else 1000.0 * leaf / leaf_c_per_area,
        soil_carbon=1000.0 * soil,
        soil_wetness_fraction=swf_fallback if np.isnan(swf) else swf / 100.0,
        fine_root_fraction=FINE, coarse_root_fraction=COARSE)

def summarize(res, leaf_c_per_area, site, exp, member, pft, annual, daily):
    out = res.outputs
    raw = out.pandas
    def col(*names):
        for n in names:
            if n in raw.columns: return raw[n]
        raise KeyError(f"{names} not in {list(raw.columns)}")
    df = pd.DataFrame({"year": col("year"), "day_of_year": col("day_of_year", "day"),
                       "nee": col("net_ecosystem_exchange", "nee"), "gpp": col("gross_primary_production", "gpp"),
                       "leaf_carbon": col("leaf_carbon", "plantLeafC"), "wood_carbon": col("wood_carbon", "plantWoodC"),
                       "soil_carbon": col("soil_carbon", "soil"), "soil_water": col("soil_water", "soilWater")})
    df["lai"] = df["leaf_carbon"] / leaf_c_per_area
    g = df.groupby("year")
    a = pd.DataFrame({"nee": g["nee"].sum(), "gpp": g["gpp"].sum(), "lai_mean": g["lai"].mean(), "lai_max": g["lai"].max(),
                      "wood_end": g["wood_carbon"].last(), "soil_end": g["soil_carbon"].last(),
                      "soil_water_mean": g["soil_water"].mean()}).reset_index()
    a.insert(0, "site", site); a.insert(1, "pft", pft); a.insert(2, "exp", exp); a.insert(3, "member", member)
    annual.append(a)
    d = df[df["year"] == 2016].copy()
    d["doy"] = d["day_of_year"]
    dd = d.groupby("doy")["nee"].sum().reset_index()
    dd.insert(0, "site", site); dd.insert(1, "exp", exp); dd.insert(2, "member", member)
    daily.append(dd)

def main():
    ds = load_initial_conditions(A + "/data/processed/initial_conditions.nc")
    runner = SIPNETRunner(flags=ModelFlags.standard(), climate_staging=ClimateStaging.SYMLINK,
                          cache_dir="/projectnb/dietzelab/arober/pysipnet/.sipnet_cache", timeout=600)
    annual, daily, log = [], [], []
    t_all = time.time()
    for site, pft in SITES.items():
        clim = ClimateDrivers.from_path(driver_file(A + "/data/raw/drivers", site, 1), n_columns=14)
        r = ds.sel(site=site)
        usable = np.where((r.initial_wood_carbon.values >= 0) & ~(r.initial_leaf_carbon.values < 0))[0]
        print(f"\n### site {site} {pft}: usable IC members {len(usable)}/100; "
              f"leaf present={not bool(r.initial_leaf_carbon.isnull().all())}, sm present={not bool(r.initial_soil_moisture_saturation.isnull().all())}",
              flush=True)
        # A: template params, IC members
        for m in usable[:N_IC]:
            ic = ic_from_product(ds, site, int(m), 32.0)
            try:
                res = runner.run(template_params(ic), clim, run_id=f"A_{site}_{m}")
                summarize(res, 32.0, site, "template", int(m), pft, annual, daily)
                log.append((site, "template", int(m), "ok", ""))
            except Exception as e:
                log.append((site, "template", int(m), "fail", f"{type(e).__name__}: {str(e)[:120]}"))
        # B: trait samples paired with IC members of the same index
        tr = pd.read_csv(f"{SAMPLES}/samples_{pft}.csv")
        for j in range(N_TRAIT):
            m = int(usable[j % len(usable)])
            ov = trait_overrides(tr.iloc[j])
            try:
                ic = ic_from_product(ds, site, m, ov["leaf_carbon_per_area"])
                model = SIPNETModel(runner, base_params=template_params(ic), base_climate=clim)
                res = model(**ov)
                summarize(res, ov["leaf_carbon_per_area"], site, "pecan_traits", j, pft, annual, daily)
                log.append((site, "pecan_traits", j, "ok", ""))
            except Exception as e:
                log.append((site, "pecan_traits", j, "fail", f"{type(e).__name__}: {str(e)[:160]}"))
        print(f"  done site {site} at {time.time()-t_all:.0f}s", flush=True)
    lg = pd.DataFrame(log, columns=["site", "exp", "member", "status", "error"])
    lg.to_csv(f"{OUT}/pp_log.csv", index=False)
    print(lg.groupby(["site", "exp", "status"]).size())
    print(lg[lg.status == "fail"].error.value_counts().head(10).to_string())
    if annual:
        pd.concat(annual).to_csv(f"{OUT}/pp_annual.csv", index=False)
        pd.concat(daily).to_csv(f"{OUT}/pp_daily2016.csv", index=False)
    print(f"total {time.time()-t_all:.0f}s")

if __name__ == "__main__":
    main()
