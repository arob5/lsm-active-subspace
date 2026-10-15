"""Phase 2 shared pieces: the base set, the candidate parameters, and natural values -> SIPNET overrides.

Calibration names are this analysis's; SIPNET overrides are pySIPNET flat names.
Re-referenced rates (``*_rate_at_10c``) are the rate at 10 C, and SIPNET's base
(0 C reference, sipnet.c:1066-1067, 1073-1078, depeffects.c) is rate / Q10.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys as _sys; _sys.path.insert(0, str(HERE.parents[1]))
OUT = HERE / "phase2"
BASE_TABLE = pd.read_csv(HERE / "base_parameter_candidates.csv")
BASE = dict(zip(BASE_TABLE.flat_name, BASE_TABLE.proposed_value.astype(float)))
SIPNET_NAME = dict(zip(BASE_TABLE.flat_name, BASE_TABLE.sipnet_name))

#: Fixed members of the degenerate photosynthesis group (BETY medians).
A_MAX_FRACTION = BASE["daily_mean_photosynthesis_fraction"]   # 0.8606
LEAF_CARBON_FRACTION = BASE["leaf_carbon_fraction"]           # 0.466
#: The site's initial state (runs.initial_state medians), kg m-2 and percent.
INITIAL_STATE = dict(initial_soil_organic_carbon=15.0, initial_wood_carbon=8.633,
                     initial_soil_moisture_saturation=61.0879)

def capacity_and_share(a_max, fol_frac, a_frac=A_MAX_FRACTION, c_frac=LEAF_CARBON_FRACTION):
    return a_max * (a_frac + fol_frac) / c_frac, fol_frac / (a_frac + fol_frac)

def a_max_and_fol_frac(capacity, share, a_frac=A_MAX_FRACTION, c_frac=LEAF_CARBON_FRACTION):
    total = a_frac / (1.0 - share)
    return capacity * c_frac / total, share * total

_P0, _RHO0 = capacity_and_share(BASE["max_photosynthesis_rate"], BASE["foliar_respiration_fraction"])
#: Calibration-name defaults: the base set in this parameterization.
DEFAULTS = dict(
    photosynthetic_capacity=_P0, respiration_share=_RHO0,
    wood_respiration_rate_at_10c=BASE["base_wood_respiration_rate"] * BASE["wood_respiration_q10"],
    soil_respiration_rate_at_10c=BASE["base_soil_respiration_rate"] * BASE["soil_respiration_q10"],
    coarse_root_respiration_rate_at_10c=BASE["base_coarse_root_respiration_rate"] * BASE["coarse_root_respiration_q10"],
    fine_root_respiration_rate_at_10c=BASE["base_fine_root_respiration_rate"] * BASE["fine_root_respiration_q10"],
    initial_wood_carbon=INITIAL_STATE["initial_wood_carbon"],
    initial_soil_organic_carbon=INITIAL_STATE["initial_soil_organic_carbon"],
    photosynthesis_temperature_range=BASE["optimum_photosynthesis_temperature"] - BASE["min_photosynthesis_temperature"],
)
_DERIVED = set(DEFAULTS)

def overrides(params: dict, base: dict | None = None) -> dict:
    """Full SIPNET overrides from calibration-name natural values (others at the base)."""
    o = dict(BASE if base is None else base)
    p = dict(DEFAULTS); p.update(params)
    for k, v in params.items():
        if k not in _DERIVED:
            if k not in o:
                raise KeyError(k)
            o[k] = float(v)
    o["max_photosynthesis_rate"], o["foliar_respiration_fraction"] = a_max_and_fol_frac(
        p["photosynthetic_capacity"], p["respiration_share"])
    o["base_wood_respiration_rate"] = p["wood_respiration_rate_at_10c"] / o["wood_respiration_q10"]
    o["base_soil_respiration_rate"] = p["soil_respiration_rate_at_10c"] / o["soil_respiration_q10"]
    o["base_coarse_root_respiration_rate"] = p["coarse_root_respiration_rate_at_10c"] / o["coarse_root_respiration_q10"]
    o["base_fine_root_respiration_rate"] = p["fine_root_respiration_rate_at_10c"] / o["fine_root_respiration_q10"]
    if "photosynthesis_temperature_range" in params or "optimum_photosynthesis_temperature" in params:
        o["min_photosynthesis_temperature"] = o["optimum_photosynthesis_temperature"] - p["photosynthesis_temperature_range"]
    f, c = o["fine_root_fraction"], o["coarse_root_fraction"]
    o["total_wood_carbon"] = 1000.0 * p["initial_wood_carbon"] / (1.0 - f - c)
    o["soil_carbon"] = 1000.0 * p["initial_soil_organic_carbon"]
    o["soil_wetness_fraction"] = INITIAL_STATE["initial_soil_moisture_saturation"] / 100.0
    o["leaf_area_index"] = 0.0
    return {k: float(v) for k, v in o.items()}

def load_cache():
    with open(OUT / "cache.pkl", "rb") as fh:
        return pickle.load(fh)

def step_ends():
    return load_cache()["predictor"]

SOURCES = ("nee_night_centered", "nee_day_centered", "modis_leaf_area_index",
           "landtrendr_aboveground_biomass", "soilgrids_soil_organic_carbon")

# ── summaries ──

SUMMARY_NAMES = ("gpp", "reco", "nee", "ra_over_gpp", "rroot_over_rsoil", "peak_lai",
                 "wood_change", "soil_change", "fine_root_end", "coarse_root_end")

def summaries(annual, columns, leaf_carbon_per_area, initial_wood, initial_soil):
    """Per-run scalar summaries from annual arrays (runs, years, columns).

    Fluxes: record means, g C m-2 yr-1. peak_lai: mean over years of the annual
    maximum. wood/soil change: last year-end minus the initial pool, g C m-2
    (aboveground wood). Root pools: record-mean year-end values.
    """
    c = {n: i for i, n in enumerate(columns)}
    a = lambda n: annual[:, :, c[n]]
    ra = a("above_ground_respiration") + a("root_respiration")
    out = dict(
        gpp=a("gpp").mean(1), reco=a("respiration").mean(1), nee=a("nee").mean(1),
        ra_over_gpp=ra.mean(1) / a("gpp").mean(1),
        rroot_over_rsoil=a("root_respiration").mean(1) / (a("root_respiration") + a("heterotrophic_respiration")).mean(1),
        peak_lai=(a("leaf_carbon_max") / np.asarray(leaf_carbon_per_area)[:, None]).mean(1),
        wood_change=a("wood_carbon_end")[:, -1] - initial_wood,
        soil_change=a("soil_carbon_end")[:, -1] - initial_soil,
        fine_root_end=a("fine_root_carbon_end").mean(1),
        coarse_root_end=a("coarse_root_carbon_end").mean(1),
    )
    return out

def misfits(predictions, cache):
    """Per-source 0.5 x sum of squared standardized residuals (diagonal noise), (runs,) each."""
    z = (predictions - cache["y"]) / cache["sd"]
    return {s: 0.5 * np.nansum(z[:, cache["source"] == s] ** 2, axis=1) for s in SOURCES}

# ── prior predictive checks ──

_SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

def coverage_table(predictions, cache, seed=0, level=0.90):
    """Fraction of observations inside / below / above the predictive interval of prediction + noise.

    Also the mean standardized residual (y - median prediction)/sd. Groups: each
    source, and each NEE source by season.
    """
    ok = np.all(np.isfinite(predictions), axis=1)
    p = predictions[ok]
    rng = np.random.default_rng(seed)
    sim = p + rng.standard_normal(p.shape) * cache["sd"]
    lo, hi = np.quantile(sim, [(1 - level) / 2, (1 + level) / 2], axis=0)
    med = np.median(p, axis=0)
    y = cache["y"]; src = cache["source"]
    season = np.array([_SEASON.get(m, "none") for m in cache["time"].month], dtype=object)
    groups = [(s, src == s) for s in SOURCES]
    for s in SOURCES[:2]:
        for q in ("DJF", "MAM", "JJA", "SON"):
            groups.append((f"{s[4:9]}_{q}", (src == s) & (season == q)))
    rows = []
    for g, m in groups:
        rows.append(dict(group=g, n=int(m.sum()), inside=np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])),
                         below=np.mean(y[m] < lo[m]), above=np.mean(y[m] > hi[m]),
                         mean_z_median=np.mean((y[m] - med[m]) / cache["sd"][m]),
                         width_over_sd=np.mean((hi[m] - lo[m]) / (2 * 1.645 * cache["sd"][m]))))
    return pd.DataFrame(rows).set_index("group")

def plausibility_table(summ):
    rows = []
    for n in SUMMARY_NAMES:
        v = summ[n][np.isfinite(summ[n])]
        rows.append(dict(summary=n, q05=np.percentile(v, 5), q50=np.median(v), q95=np.percentile(v, 95)))
    return pd.DataFrame(rows).set_index("summary")

PLAUSIBLE = dict(gpp=(1000, 1800), nee=(-700, 0), peak_lai=(3, 8))

def plausible_fraction(summ, initial_wood):
    ok = np.ones_like(summ["gpp"], dtype=bool)
    for k, (lo, hi) in PLAUSIBLE.items():
        ok &= (summ[k] >= lo) & (summ[k] <= hi)
    w = (summ["wood_change"] + initial_wood) / initial_wood
    ok &= (w > 0.8) & (w < 2.0)
    return ok
