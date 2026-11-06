"""Phase 2 final: the proposed calibration as library objects, its prior predictive, and the CSVs.

Builds the ParameterVector / Prior / SIPNETParameterMap, samples theta with
Prior.sample, maps it with SIPNETParameterMap.sipnet_parameter_fields, runs the
fast path, and writes prior_predictive.npz, prior_proposal.csv,
fixed_parameters.csv and phase2/final_{coverage,plausibility,criteria}.csv.
Usage: phase2_final.py [J]
"""
import sys, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

#: Fixed values that differ from base_parameter_candidates.csv.
FIXED_CHANGES = {
    "base_fine_root_respiration_rate": (0.000509 * 365.0,
        "BETY root_respiration_rate median converted to SIPNET's yr-1 (sipnet.c:1902 divides by 365), "
        "undoing PEcAn's per-day slip (write.configs.SIPNET.R:461-468); pool is small, so inert in practice"),
}
PSN_T_RANGE = None  # set from the base: psnTOpt - psnTMin

def build():
    import jax.numpy as jnp, tensorflow_probability.substrates.jax as tfp
    import inputs, runs, phase2_common as pc
    from sipnet_calibration.parameter_vector import (POSITIVE, REAL, OPEN_UNIT_INTERVAL, SIMPLEX, Parameter,
                                                     DerivedParameter, ParameterVector)
    from sipnet_calibration.prior import (Prior, PriorTerm, log_normal_from_interval, logit_normal_from_interval,
                                          log_normal_from_samples, softmax_normal)
    from sipnet_calibration.sipnet_parameter_map import (ComputeInitialConditions, ComputePhotosynthesisRates,
                                                         Copy, CopySimplex, Fixed, SIPNETParameterMap)
    from stand_in_calibration import DECIDUOUS_BY_SITE_LABEL
    import stand_in_calibration as sic
    tdr = pc.BASE["optimum_photosynthesis_temperature"] - pc.BASE["min_photosynthesis_temperature"]
    q10_wood, q10_coarse = pc.BASE["wood_respiration_q10"], pc.BASE["coarse_root_respiration_q10"]
    ic = inputs.initial_condition_fields()
    members = {n: np.asarray(ic[n].squeeze()).ravel().astype(float) for n in ("initial_wood_carbon", "initial_soil_organic_carbon")}
    site_table = inputs.site_table()
    labels = forward_labels()
    P = lambda name, support, units, **kw: Parameter(name=name, support=support, units=units, **kw)
    parameters = [
        P("photosynthetic_capacity", POSITIVE, "nmol g-1 s-1"),
        P("respiration_share", OPEN_UNIT_INTERVAL, "1"),
        P("optimum_photosynthesis_temperature", REAL, "degC"),
        P("half_saturation_light", POSITIVE, "mol m-2 d-1"),
        P("soil_water_holding_capacity", POSITIVE, "cm"),
        P("leaf_on_growth", POSITIVE, "g m-2"),
        P("leaf_on_growing_degree_days", POSITIVE, "K d"),
        P("allocation", SIMPLEX, "1", natural_names=("leaf", "wood", "fine_root", "coarse_root")),
        P("wood_respiration_rate_at_10c", POSITIVE, "yr-1"),
        P("soil_respiration_flux_at_10c", POSITIVE, "g m-2 yr-1"),
        P("soil_respiration_q10", POSITIVE, "1"),
        P("initial_wood_carbon", POSITIVE, "kg m-2"),
        P("initial_soil_organic_carbon", POSITIVE, "kg m-2"),
    ]
    D = lambda name, units, frm, f, support=POSITIVE: DerivedParameter(name=name, units=units, derived_from=frm,
                                                                       compute=f, support=support)
    derived = [
        D("min_photosynthesis_temperature", "degC", ("optimum_photosynthesis_temperature",),
          lambda dim_index, site_table, optimum_photosynthesis_temperature: optimum_photosynthesis_temperature - tdr, REAL),
        D("base_wood_respiration_rate", "yr-1", ("wood_respiration_rate_at_10c",),
          lambda dim_index, site_table, wood_respiration_rate_at_10c: wood_respiration_rate_at_10c / q10_wood),
        D("base_coarse_root_respiration_rate", "yr-1", ("wood_respiration_rate_at_10c",),
          lambda dim_index, site_table, wood_respiration_rate_at_10c: wood_respiration_rate_at_10c / q10_coarse),
        D("base_soil_respiration_rate", "yr-1", ("soil_respiration_flux_at_10c", "initial_soil_organic_carbon", "soil_respiration_q10"),
          lambda dim_index, site_table, soil_respiration_flux_at_10c, initial_soil_organic_carbon, soil_respiration_q10:
              soil_respiration_flux_at_10c / (1000.0 * initial_soil_organic_carbon * soil_respiration_q10)),
    ]
    vector = ParameterVector(parameters=parameters, derived_parameters=derived, site_table=site_table,
                             site_labels={"pft": list(labels)})
    normal = tfp.distributions.Normal(jnp.float64(22.0), jnp.float64(2.5))
    T = lambda d, prov: PriorTerm(d, provenance=prov)
    prior = Prior(vector, {
        "photosynthetic_capacity": T(log_normal_from_interval(lower=140.0, upper=450.0), PROVENANCE["photosynthetic_capacity"]),
        "respiration_share": T(logit_normal_from_interval(lower=0.04, upper=0.20), PROVENANCE["respiration_share"]),
        "optimum_photosynthesis_temperature": T(normal, PROVENANCE["optimum_photosynthesis_temperature"]),
        "half_saturation_light": T(log_normal_from_interval(lower=4.6, upper=26.3), PROVENANCE["half_saturation_light"]),
        "soil_water_holding_capacity": T(log_normal_from_interval(lower=15.0, upper=150.0), PROVENANCE["soil_water_holding_capacity"]),
        "leaf_on_growth": T(log_normal_from_interval(lower=50.0, upper=180.0), PROVENANCE["leaf_on_growth"]),
        "leaf_on_growing_degree_days": T(log_normal_from_interval(lower=500.0, upper=1100.0), PROVENANCE["leaf_on_growing_degree_days"]),
        "allocation": T(softmax_normal(center=jnp.array([0.18, 0.45, 0.065, 0.305]), logit_sd=jnp.array([0.25, 0.30, 0.30])), PROVENANCE["allocation"]),
        "wood_respiration_rate_at_10c": T(log_normal_from_interval(lower=0.006, upper=0.04), PROVENANCE["wood_respiration_rate_at_10c"]),
        "soil_respiration_flux_at_10c": T(log_normal_from_interval(lower=200.0, upper=900.0), PROVENANCE["soil_respiration_flux_at_10c"]),
        "soil_respiration_q10": T(log_normal_from_interval(lower=1.3, upper=3.2), PROVENANCE["soil_respiration_q10"]),
        "initial_wood_carbon": T(log_normal_from_samples(members["initial_wood_carbon"]), PROVENANCE["initial_wood_carbon"]),
        "initial_soil_organic_carbon": T(log_normal_from_samples(members["initial_soil_organic_carbon"]), PROVENANCE["initial_soil_organic_carbon"]),
    })
    rules = [
        ComputePhotosynthesisRates(capacity_value_name="photosynthetic_capacity", respiration_share_value_name="respiration_share"),
        Copy(value_name="optimum_photosynthesis_temperature", sipnet_parameter_name="optimum_photosynthesis_temperature"),
        Copy(value_name="min_photosynthesis_temperature", sipnet_parameter_name="min_photosynthesis_temperature"),
        Copy(value_name="half_saturation_light", sipnet_parameter_name="half_saturation_light"),
        Copy(value_name="soil_water_holding_capacity", sipnet_parameter_name="soil_water_holding_capacity"),
        Copy(value_name="leaf_on_growth", sipnet_parameter_name="leaf_on_growth"),
        Copy(value_name="leaf_on_growing_degree_days", sipnet_parameter_name="leaf_on_growing_degree_days"),
        CopySimplex(value_name="allocation", sipnet_parameter_names=("leaf_allocation", "wood_allocation", "fine_root_allocation")),
        Copy(value_name="base_wood_respiration_rate", sipnet_parameter_name="base_wood_respiration_rate"),
        Copy(value_name="base_coarse_root_respiration_rate", sipnet_parameter_name="base_coarse_root_respiration_rate"),
        Copy(value_name="base_soil_respiration_rate", sipnet_parameter_name="base_soil_respiration_rate"),
        Copy(value_name="soil_respiration_q10", sipnet_parameter_name="soil_respiration_q10"),
        ComputeInitialConditions(deciduous=DECIDUOUS_BY_SITE_LABEL),
    ]
    written = {n for r in rules for n in r.sipnet_parameter_names_written}
    fixed_rows = []
    fixed = []
    base = pd.read_csv(HERE / "base_parameter_candidates.csv").set_index("flat_name")
    for name, value in pc.BASE.items():
        if name in written:
            continue
        v, why = FIXED_CHANGES.get(name, (value, None))
        if why is None:
            why = f"phase 1 base set: {base.loc[name, 'source']}"
        fixed.append(Fixed(sipnet_parameter_name=name, value=float(v), provenance=why))
        fixed_rows.append(dict(flat_name=name, sipnet_name=base.loc[name, "sipnet_name"], units=base.loc[name, "units"], value=float(v), justification=why))
    smap = SIPNETParameterMap(rules=rules, fixed=fixed)
    external = runs.initial_state()[["initial_leaf_carbon", "initial_soil_moisture_saturation"]]
    return vector, prior, smap, external, pd.DataFrame(fixed_rows), members

def forward_labels():
    import inputs
    from sipnet_calibration.site_labels import load_site_labels
    import stand_in_calibration as sic, config
    try:
        return list(load_site_labels(sic.SITE_LABELS_NAME).set_index("site_id").loc[[config.SITE]].iloc[:, 0])
    except Exception:
        return ["temperate.deciduous.HPDA"]

PROVENANCE = {
    "photosynthetic_capacity": "Reasoned. P = aMax(aMaxFrac+baseFolRespFrac)/cFracLeaf. Median 251 (aMax 123 nmol g-1 s-1, ~8 umol m-2 s-1 at the BETY SLA); "
        "2.5% 140 ~ BETY Amax*SLA 97.5% (P 170) and below template (228); 97.5% 450 ~ top-canopy oak leaf Amax ~13 umol m-2 s-1 (approximate literature). Upper end cut from 500 after i4's GPP tail.",
    "respiration_share": "Reasoned from BETY: area Rd 0.66 umol m-2 s-1 over leaf gross Amax 6-15 gives 0.04-0.10; upper end reaches BETY's baseFolRespFrac-based 0.16.",
    "optimum_photosynthesis_temperature": "Reasoned: temperate-deciduous canopy optimum ~20-27 C (approximate literature); BETY median 43 C rejected (phase 1). psnTMin follows at fixed range psnTOpt-psnTMin = 23.96 C.",
    "half_saturation_light": "BETY temperate.deciduous 2.5-97.5% (4.6-26.3).",
    "soil_water_holding_capacity": "Reasoned placeholder: from plant-available water of a ~1 m stony sandy-loam rooting zone (~15 cm, inferred) to past PEcAn's porosity x 2 m (88). Above ~88 the model is water-unlimited here (flat direction).",
    "leaf_on_growth": "BETY 2.5% (50) kept; upper end cut from BETY 249 to 180 (LAI 5.9 at leafCSpWt 30.7): the pulse should not exceed a full Harvard canopy (reasoned).",
    "leaf_on_growing_degree_days": "Driver-based: SIPNET GDD (base 0 C, sipnet.c:230-235) reaches 500 on DOY ~123, 700 on ~139, 800 on ~145; Harvard canopy develops mid-late May (approximate phenology literature; NEE-onset ~DOY 148 in the calibration data, GDD 847+-119). Median 742.",
    "allocation": "Simplex (leaf, wood, fine root, coarse root = SIPNET remainder). Leaf 0.18 and fine 0.065 BETY medians; wood raised 0.40->0.45 so coarse:wood allocation ~0.7 rather than 0.9 (reasoned; root:shoot ~0.2-0.3). Log-ratio sds 0.25/0.30/0.30 approximate BETY spreads.",
    "wood_respiration_rate_at_10c": "Reasoned: base rate referenced to 10 C (baseVegResp = r10/vegRespQ10, sipnet.c:1066-1067); median 0.0155 yr-1 ~ 130 g C m-2 yr-1 at 8.6 kg C m-2 of wood; BETY stem draws judged unconstrained (phase 1). Also sets coarse-root respiration (tied: coarse roots are wood).",
    "soil_respiration_flux_at_10c": "Reparameterization of the Rh ridge: F10 = k10*SOC0, g C m-2 yr-1 at 10 C and M=1; baseSoilResp = F10/(1000*SOC0*Q10). Reasoned: Harvard heterotrophic respiration ~300-700 g C m-2 yr-1 (approximate literature). Pushforward of k10 checked against BETY.",
    "soil_respiration_q10": "Reasoned: ecosystem-level Q10 converges near 1.4 (Mahecha et al. 2010, Science 329:838) while soil-chamber apparent Q10s are often 2-4; interval 1.3-3.2, median 2.04.",
    "initial_wood_carbon": "Site 4977 initial-condition ensemble (100 members), log-normal fit to the members (log_normal_from_samples).",
    "initial_soil_organic_carbon": "Site 4977 initial-condition ensemble, log-normal fit to the members; its upper tail (to ~100 kg m-2) is kept, the SoilGrids observation constrains it.",
}

def main():
    import jax, phase2_common as pc, fast_forward, config
    J = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    vector, prior, smap, external, fixed_table, members = build()
    theta = np.asarray(prior.sample(jax.random.key(20260928), J))
    natural = vector.dataset(theta)
    f = smap.sipnet_parameter_fields(vector, theta, external_inputs=external)
    sel = lambda a, j: float(a.sel(site=config.SITE).isel(sample=j)) if "sample" in a.dims else float(a.sel(site=config.SITE))
    batch = [{n: sel(f[n], j) for n in f.data_vars} for j in range(J)]
    missing = set(pc.BASE) - set(batch[0])
    assert not missing, missing
    cache = pc.load_cache(); t0 = time.perf_counter()
    res = fast_forward.run_batch(batch, cache["predictor"], n_workers=7)
    N = len(cache["sd"])
    pred = np.array([o["predictions"] if o["predictions"] is not None else np.full(N, np.nan) for o in res])
    cols = list(fast_forward.ANNUAL_SUMMARY_NAMES)
    years = next(o["annual"].index.to_numpy() for o in res if o["annual"] is not None)
    ann = np.array([o["annual"][cols].to_numpy() if o["annual"] is not None else np.full((len(years), len(cols)), np.nan) for o in res])
    err = np.array([o["error"] or "" for o in res])
    # natural values, flattened with component names
    names, cols_nat = [], []
    for v in natural.data_vars:
        names.append(str(v)); cols_nat.append(np.asarray(natural[v]).reshape(J, -1)[:, :1])
    nat = np.concatenate(cols_nat, axis=1)
    keys = sorted(batch[0]); vals = np.array([[b[k] for k in keys] for b in batch])
    iw = vals[:, keys.index("total_wood_carbon")] * (1 - vals[:, keys.index("fine_root_fraction")] - vals[:, keys.index("coarse_root_fraction")])
    isoil = vals[:, keys.index("soil_carbon")]
    summ = pc.summaries(ann, cols, vals[:, keys.index("leaf_carbon_per_area")], iw, isoil)
    lai_mask = cache["source"] == "modis_leaf_area_index"
    summ["jja_mean_lai"] = pred[:, lai_mask].mean(1)
    loglik = np.asarray(__import__("noise").calibration_likelihood.__wrapped__(None) if False else np.nan)
    np.savez_compressed(HERE / "prior_predictive.npz", theta=theta, theta_names=list(vector.index.to_flat_index().astype(str)),
                        natural_values=nat, natural_names=names, override_names=keys, override_values=vals,
                        predictions=pred, y=cache["y"], noise_sd=cache["sd"], observation_source=cache["source"],
                        observation_time=cache["time"].astype("datetime64[ns]").to_numpy(),
                        annual=ann, annual_columns=cols, years=years, errors=err,
                        **{f"summary_{k}": v for k, v in summ.items()})
    cov = pc.coverage_table(pred, cache); pl = pc.plausibility_table(summ)
    pl.loc["jja_mean_lai"] = np.nanpercentile(summ["jja_mean_lai"], [5, 50, 95])
    cov.to_csv(pc.OUT / "final_coverage.csv"); pl.to_csv(pc.OUT / "final_plausibility.csv")
    fixed_table.to_csv(HERE / "fixed_parameters.csv", index=False)
    proposal(vector, prior, smap, nat, names).to_csv(HERE / "prior_proposal.csv", index=False)
    print(f"{J} runs {time.perf_counter()-t0:.0f} s failures {(err!='').sum()} D={vector.dimension}")
    pd.set_option("display.width", 200)
    print(cov.round(2).to_string()); print(pl.round(3).to_string())
    k10 = nat[:, names.index("base_soil_respiration_rate")] * 1.0
    print("pushforward baseSoilResp (0 C, yr-1) 2.5/50/97.5%:", np.round(np.percentile(k10, [2.5, 50, 97.5]), 4), "BETY 0.004-0.020")

HOW = {
    "photosynthetic_capacity": "ComputePhotosynthesisRates -> max_photosynthesis_rate = P*cFracLeaf*(1-rho)/aMaxFrac (aMax)",
    "respiration_share": "ComputePhotosynthesisRates -> foliar_respiration_fraction = rho*aMaxFrac/(1-rho) (baseFolRespFrac)",
    "optimum_photosynthesis_temperature": "Copy -> optimum_photosynthesis_temperature (psnTOpt); DerivedParameter min = topt - 23.96 -> Copy min_photosynthesis_temperature (psnTMin)",
    "half_saturation_light": "Copy -> half_saturation_light (halfSatPar)",
    "soil_water_holding_capacity": "Copy -> soil_water_holding_capacity (soilWHC)",
    "leaf_on_growth": "Copy -> leaf_on_growth (leafGrowth)",
    "leaf_on_growing_degree_days": "Copy -> leaf_on_growing_degree_days (gddLeafOn)",
    "allocation": "CopySimplex -> leaf_allocation, wood_allocation, fine_root_allocation; coarse root = SIPNET remainder (sipnet.c:1113-1115)",
    "wood_respiration_rate_at_10c": "DerivedParameters r10/vegRespQ10 -> base_wood_respiration_rate (baseVegResp) and r10/coarseRootQ10 -> base_coarse_root_respiration_rate",
    "soil_respiration_flux_at_10c": "DerivedParameter F10/(1000*SOC0*Q10) -> base_soil_respiration_rate (baseSoilResp)",
    "soil_respiration_q10": "Copy -> soil_respiration_q10 (soilRespQ10); also enters baseSoilResp",
    "initial_wood_carbon": "ComputeInitialConditions -> total_wood_carbon = 1000*Cw/(1-fineRootFrac-coarseRootFrac)",
    "initial_soil_organic_carbon": "ComputeInitialConditions -> soil_carbon = 1000*Cs; also enters baseSoilResp",
}
FAMILY = {
    "photosynthetic_capacity": ("log_normal_from_interval", "lower=140, upper=450"),
    "respiration_share": ("logit_normal_from_interval", "lower=0.04, upper=0.20"),
    "optimum_photosynthesis_temperature": ("Normal", "loc=22, scale=2.5"),
    "half_saturation_light": ("log_normal_from_interval", "lower=4.6, upper=26.3"),
    "soil_water_holding_capacity": ("log_normal_from_interval", "lower=15, upper=150"),
    "leaf_on_growth": ("log_normal_from_interval", "lower=50, upper=180"),
    "leaf_on_growing_degree_days": ("log_normal_from_interval", "lower=500, upper=1100"),
    "allocation": ("softmax_normal", "center=(0.18, 0.45, 0.065, 0.305), logit_sd=(0.25, 0.30, 0.30)"),
    "wood_respiration_rate_at_10c": ("log_normal_from_interval", "lower=0.006, upper=0.04"),
    "soil_respiration_flux_at_10c": ("log_normal_from_interval", "lower=200, upper=900"),
    "soil_respiration_q10": ("log_normal_from_interval", "lower=1.3, upper=3.2"),
    "initial_wood_carbon": ("log_normal_from_samples", "site IC ensemble members"),
    "initial_soil_organic_carbon": ("log_normal_from_samples", "site IC ensemble members"),
}

def proposal(vector, prior, smap, nat, names):
    rows = []
    for p in vector.parameter_names:
        sub = [n for n in names if n == p or n.startswith(p + ".")]
        for n in sub:
            q = np.percentile(nat[:, names.index(n)], [5, 50, 95])
            par = vector[p]
            rows.append(dict(name=n, units=par.units, support=type(par.support).__name__ if not hasattr(par.support, "name") else par.support.name,
                             prior_family=FAMILY[p][0], prior_arguments=FAMILY[p][1],
                             q05=q[0], q50=q[1], q95=q[2], reaches_sipnet=HOW[p], provenance=PROVENANCE[p]))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    main()
