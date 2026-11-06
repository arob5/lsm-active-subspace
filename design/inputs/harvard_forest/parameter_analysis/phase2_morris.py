"""Phase 2 task 1: Morris elementary-effects screening over the candidate parameters.

Writes phase2/morris_runs.npz (unit and natural designs, predictions, annual
summaries, errors). Analysis is phase2_morris_analyse.py.
Usage: .venv/bin/python experiments/.../output/analysis/phase2_morris.py [r]
"""
import sys, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np

#: name, low, high, log-scaled, provenance of the range
CANDIDATES = [
    ("photosynthetic_capacity", 120, 400, True, "aMax 59-196 nmol/g/s at the fixed aMaxFrac/cFracLeaf, rho=0.094; brackets BETY 58 and template 112 (reasoned)"),
    ("respiration_share", 0.05, 0.30, True, "BETY baseFolRespFrac 0.098-0.39 at aMaxFrac 0.86 -> 0.10-0.31, widened down to the base 0.094"),
    ("half_saturation_light", 4.6, 26.3, True, "BETY 2.5-97.5%"),
    ("light_extinction_coefficient", 0.39, 0.61, False, "BETY 2.5-97.5%"),
    ("leaf_carbon_per_area", 25, 40, False, "BETY leafC/SLA 27.4-35.1, widened"),
    ("optimum_photosynthesis_temperature", 18, 30, False, "reasoned (~20-27 C literature; BETY 43 implausible)"),
    ("vapor_pressure_deficit_slope", 0.01, 0.10, True, "BETY 0.003-0.097, floor raised"),
    ("soil_water_holding_capacity", 10, 100, True, "template 12 to PEcAn porosity formula 88 (reasoned)"),
    ("water_removal_fraction", 0.04, 0.20, True, "template 0.088 x/÷ 2.2 (reasoned)"),
    ("water_use_efficiency", 7.9, 17.5, True, "BETY 2.5-97.5%"),
    ("leaf_on_growing_degree_days", 350, 650, False, "BETY 458-542 widened (reasoned)"),
    ("leaf_off_day", 265, 300, False, "Harvard leaf fall DOY ~280-300 (reasoned)"),
    ("leaf_on_growth", 50, 250, True, "BETY 2.5-97.5%"),
    ("leaf_allocation", 0.08, 0.30, False, "BETY 0.14-0.22 widened"),
    ("leaf_turnover_rate", 0.03, 0.5, True, "reasoned around template 0.13"),
    ("wood_allocation", 0.24, 0.57, False, "BETY 2.5-97.5%"),
    ("fine_root_allocation", 0.04, 0.10, False, "BETY 2.5-97.5%"),
    ("wood_turnover_rate", 0.002, 0.02, True, "BETY 0.003-0.007 widened (mortality ~1%/yr)"),
    ("fine_root_turnover_rate", 0.35, 3.0, True, "BETY 0.35-7.2, capped"),
    ("coarse_root_turnover_rate", 0.02, 0.10, True, "template 0.056 x/÷ 2"),
    ("wood_respiration_rate_at_10c", 0.005, 0.06, True, "base 0.021 x/÷ ~3"),
    ("wood_respiration_q10", 1.45, 2.4, False, "BETY 1.45-2.21, widened up"),
    ("soil_respiration_rate_at_10c", 0.01, 0.06, True, "BETY som rate 0.004-0.020 x Q10 2.9"),
    ("soil_respiration_q10", 1.5, 3.5, False, "reasoned around template 2.9"),
    ("soil_respiration_moisture_exponent", 0.3, 2.0, False, "reasoned around template 1"),
    ("coarse_root_respiration_rate_at_10c", 0.005, 0.10, True, "template 0.019 x/÷ 4 (unit question open)"),
    ("coarse_root_fraction", 0.08, 0.25, False, "reasoned around 0.16"),
    ("initial_wood_carbon", 4.8, 12.3, False, "site IC ensemble 25-75%"),
    ("initial_soil_organic_carbon", 10.6, 25.5, True, "site IC ensemble 25-75%"),
]
NAMES = [c[0] for c in CANDIDATES]

def to_natural(u):
    x = np.empty_like(u)
    for j, (_, lo, hi, lg, _) in enumerate(CANDIDATES):
        x[:, j] = np.exp(np.log(lo) + u[:, j] * np.log(hi / lo)) if lg else lo + u[:, j] * (hi - lo)
    return x

def trajectories(r, k, seed=0, p=4):
    rng = np.random.default_rng(seed)
    delta = p / (2 * (p - 1))
    levels = np.arange(p) / (p - 1)
    out, steps = [], []
    for _ in range(r):
        x = rng.choice(levels, size=k)
        order = rng.permutation(k)
        pts = [x.copy()]
        for i in order:
            x = x.copy()
            x[i] = x[i] + delta if x[i] + delta <= 1 + 1e-9 else x[i] - delta
            pts.append(x)
        out.append(np.array(pts)); steps.append(order)
    return np.array(out), np.array(steps), delta

def main():
    import phase2_common as pc, fast_forward
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    k = len(NAMES)
    U, order, delta = trajectories(r, k)
    u = U.reshape(-1, k)
    x = to_natural(u)
    batch = [pc.overrides(dict(zip(NAMES, row))) for row in x]
    cache = pc.load_cache()
    t0 = time.perf_counter()
    res = fast_forward.run_batch(batch, cache["predictor"], n_workers=7)
    N = len(cache["sd"])
    pred = np.array([o["predictions"] if o["predictions"] is not None else np.full(N, np.nan) for o in res])
    years = res[0]["annual"].index.to_numpy()
    cols = list(res[0]["annual"].columns)
    ann = np.array([o["annual"].to_numpy() if o["annual"] is not None else np.full((len(years), len(cols)), np.nan) for o in res])
    err = np.array([o["error"] or "" for o in res])
    np.savez_compressed(pc.OUT / "morris_runs.npz", U=U, order=order, delta=delta, x=x, names=NAMES,
                        predictions=pred, annual=ann, years=years, annual_columns=cols, errors=err,
                        leaf_carbon_per_area=np.array([b["leaf_carbon_per_area"] for b in batch]),
                        initial_wood=np.array([b["total_wood_carbon"] * (1 - b["fine_root_fraction"] - b["coarse_root_fraction"]) for b in batch]),
                        initial_soil=np.array([b["soil_carbon"] for b in batch]))
    print(f"{len(batch)} runs in {time.perf_counter()-t0:.0f} s; failures {(err!='').sum()}")
    for e in sorted(set(err[err != ""]))[:5]:
        print("  ", e[:200])

if __name__ == "__main__":
    main()
