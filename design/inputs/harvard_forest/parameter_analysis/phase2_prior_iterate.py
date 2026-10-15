"""Phase 2 task 4: prior predictive for one prior iteration, sampled in NumPy.

The priors here are the same families the library builds (log_normal_from_interval,
logit_normal_from_interval, Normal, softmax_normal, log_normal_from_samples);
phase2_final.py rebuilds the chosen one with the library and re-runs it.
Usage: phase2_prior_iterate.py <iteration> [J]
Writes phase2/iter_<iteration>.npz, _coverage.csv, _plausibility.csv.
"""
import sys, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

ALLOCATION = ("leaf_allocation", "wood_allocation", "fine_root_allocation")  # 4th = coarse remainder
Z = 1.959963984540054

def ic_members():
    import inputs
    f = inputs.initial_condition_fields()
    return {n: np.asarray(f[n].squeeze()).ravel().astype(float) for n in ("initial_wood_carbon", "initial_soil_organic_carbon")}

def draw(spec, n, rng, members):
    kind = spec[0]
    if kind == "lognormal_interval":
        lo, hi = spec[1:3]; m, s = (np.log(lo) + np.log(hi)) / 2, (np.log(hi) - np.log(lo)) / (2 * Z)
        return np.exp(m + s * rng.standard_normal(n))
    if kind == "lognormal":
        return spec[1] * np.exp(np.log(spec[2]) * rng.standard_normal(n))
    if kind == "lognormal_samples":
        v = np.log(members[spec[1]]); return np.exp(v.mean() + v.std(ddof=1) * rng.standard_normal(n))
    if kind == "logitnormal_interval":
        lo, hi = spec[1:3]; L = lambda p: np.log(p / (1 - p))
        m, s = (L(lo) + L(hi)) / 2, (L(hi) - L(lo)) / (2 * Z)
        return 1 / (1 + np.exp(-(m + s * rng.standard_normal(n))))
    if kind == "normal":
        return spec[1] + spec[2] * rng.standard_normal(n)
    if kind == "simplex":
        c = np.asarray(spec[1], float); sd = np.asarray(spec[2], float)
        t = np.log(c[:-1] / c[-1]) + sd * rng.standard_normal((n, len(c) - 1))
        e = np.concatenate([np.exp(t), np.ones((n, 1))], axis=1)
        return e / e.sum(1, keepdims=True)
    raise ValueError(kind)

BASE_PRIOR = {
    "photosynthetic_capacity": ("lognormal_interval", 100, 400),
    "respiration_share": ("logitnormal_interval", 0.05, 0.25),
    "optimum_photosynthesis_temperature": ("normal", 22.0, 2.5),
    "half_saturation_light": ("lognormal_interval", 4.6, 26.3),
    "soil_water_holding_capacity": ("lognormal_interval", 8, 100),
    "leaf_on_growth": ("lognormal_interval", 50.7, 248.6),
    "leaf_carbon_per_area": ("lognormal_interval", 27.4, 35.1),
    "allocation": ("simplex", (0.18, 0.45, 0.065, 0.305), (0.25, 0.3, 0.3)),
    "wood_respiration_rate_at_10c": ("lognormal_interval", 0.007, 0.06),
    "soil_respiration_rate_at_10c": ("lognormal_interval", 0.0117, 0.058),
    "initial_wood_carbon": ("lognormal_samples", "initial_wood_carbon"),
    "initial_soil_organic_carbon": ("lognormal_samples", "initial_soil_organic_carbon"),
}
#: Fixed values that differ from the base set, by iteration (flat names).
BASE_FIXED = {"base_fine_root_respiration_rate": 0.000509 * 365}
I2_PRIOR = dict(BASE_PRIOR)
del I2_PRIOR["soil_respiration_rate_at_10c"], I2_PRIOR["leaf_carbon_per_area"]
I2_PRIOR.update({
    "photosynthetic_capacity": ("lognormal_interval", 150, 500),
    "respiration_share": ("logitnormal_interval", 0.04, 0.20),
    "soil_respiration_flux_at_10c": ("lognormal_interval", 200, 900),
    "soil_respiration_q10": ("lognormal_interval", 1.5, 4.0),
})
I3_PRIOR = dict(I2_PRIOR)
I3_PRIOR.update({
    "leaf_on_growing_degree_days": ("lognormal_interval", 500, 1100),
    "soil_respiration_q10": ("lognormal_interval", 1.3, 3.2),
})
I4_PRIOR = dict(I3_PRIOR)
I4_PRIOR.update({
    "soil_water_holding_capacity": ("lognormal_interval", 15, 150),
    "leaf_on_growth": ("lognormal_interval", 50, 180),
    "wood_respiration_rate_at_10c": ("lognormal_interval", 0.006, 0.04),
})
I5_PRIOR = dict(I4_PRIOR)
I5_PRIOR.update({"photosynthetic_capacity": ("lognormal_interval", 120, 400)})
ITERATIONS = {"i1": (BASE_PRIOR, BASE_FIXED), "i2": (I2_PRIOR, BASE_FIXED), "i3": (I3_PRIOR, BASE_FIXED),
              "i4": (I4_PRIOR, BASE_FIXED), "i5": (I5_PRIOR, BASE_FIXED)}

def sample(prior, n, seed, members):
    rng = np.random.default_rng(seed)
    x = {}
    for name, spec in prior.items():
        v = draw(spec, n, rng, members)
        if name == "allocation":
            for i, a in enumerate(ALLOCATION):
                x[a] = v[:, i]
        else:
            x[name] = v
    return pd.DataFrame(x)

def to_overrides(row, fixed):
    import phase2_common as pc
    p = dict(row)
    if "soil_respiration_flux_at_10c" in p:   # k10 = F10 / SOC0 (g m-2 yr-1 over g m-2)
        p["soil_respiration_rate_at_10c"] = p.pop("soil_respiration_flux_at_10c") / (1000.0 * p["initial_soil_organic_carbon"])
    base = dict(pc.BASE); base.update(fixed)
    o = pc.overrides(p, base)
    # coarse-root respiration tied to the wood rate at 10 C (coarse roots are wood)
    if "wood_respiration_rate_at_10c" in p:
        o["base_coarse_root_respiration_rate"] = p["wood_respiration_rate_at_10c"] / o["coarse_root_respiration_q10"]
    return o

def evaluate(name, prior, fixed, J, seed=1):
    import phase2_common as pc, fast_forward
    cache = pc.load_cache()
    x = sample(prior, J, seed, ic_members())
    batch = [to_overrides(r, fixed) for _, r in x.iterrows()]
    t0 = time.perf_counter()
    res = fast_forward.run_batch(batch, cache["predictor"], n_workers=7)
    N = len(cache["sd"])
    pred = np.array([o["predictions"] if o["predictions"] is not None else np.full(N, np.nan) for o in res])
    cols = list(res[0]["annual"].columns) if res[0]["annual"] is not None else list(fast_forward.ANNUAL_SUMMARY_NAMES)
    years = next(o["annual"].index.to_numpy() for o in res if o["annual"] is not None)
    ann = np.array([o["annual"].to_numpy() if o["annual"] is not None else np.full((len(years), len(cols)), np.nan) for o in res])
    err = np.array([o["error"] or "" for o in res])
    lcpa = np.array([b["leaf_carbon_per_area"] for b in batch])
    iw = x["initial_wood_carbon"].to_numpy() * 1000; isoil = x["initial_soil_organic_carbon"].to_numpy() * 1000
    keys = sorted(batch[0]); vals = np.array([[b[k] for k in keys] for b in batch])
    np.savez_compressed(pc.OUT / f"iter_{name}.npz", theta_names=list(x.columns), theta=x.to_numpy(),
                        override_names=keys, override_values=vals, predictions=pred, annual=ann,
                        annual_columns=cols, years=years, errors=err)
    summ = pc.summaries(ann, cols, lcpa, iw, isoil)
    cov = pc.coverage_table(pred, cache); pl = pc.plausibility_table(summ)
    cov.to_csv(pc.OUT / f"iter_{name}_coverage.csv"); pl.to_csv(pc.OUT / f"iter_{name}_plausibility.csv")
    okp = pc.plausible_fraction(summ, iw)
    print(f"[{name}] {J} runs, {time.perf_counter()-t0:.0f} s, failures {(err!='').sum()}, all-plausible fraction {okp.mean():.2f}")
    for e in sorted(set(err[err != ""]))[:3]:
        print("  error:", e[:160])
    pd.set_option("display.width", 200)
    print(cov.round(2).to_string()); print(pl.round(3).to_string())
    return x, summ, pred

if __name__ == "__main__":
    name = sys.argv[1]; J = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    prior, fixed = ITERATIONS[name]
    evaluate(name, prior, fixed, J)
