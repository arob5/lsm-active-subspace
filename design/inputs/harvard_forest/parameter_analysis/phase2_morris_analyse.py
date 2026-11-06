"""Morris mu*/sigma from phase2/morris_runs.npz; writes phase2/morris_mu_star.csv and prints a ranking."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import phase2_common as pc

d = np.load(pc.OUT / "morris_runs.npz", allow_pickle=True)
cache = pc.load_cache()
names = list(d["names"]); k = len(names); r = d["U"].shape[0]
m = pc.misfits(d["predictions"], cache)
s = pc.summaries(d["annual"], list(d["annual_columns"]), d["leaf_carbon_per_area"], d["initial_wood"], d["initial_soil"])
outputs = {**{"misfit_" + key: v for key, v in m.items()}, **s}
outputs["misfit_total"] = sum(m.values())
U = d["U"]; rows = []
for oname, f in outputs.items():
    f = f.reshape(r, k + 1)
    ee = np.full((r, k), np.nan)
    for t in range(r):
        for step in range(k):
            du = U[t, step + 1] - U[t, step]
            i = int(np.argmax(np.abs(du)))
            ee[t, i] = (f[t, step + 1] - f[t, step]) / du[i]
    for i, n in enumerate(names):
        e = ee[:, i][np.isfinite(ee[:, i])]
        rows.append(dict(output=oname, parameter=n, mu_star=np.abs(e).mean(), mu=e.mean(), sigma=e.std(ddof=1)))
t = pd.DataFrame(rows)
t.to_csv(pc.OUT / "morris_mu_star.csv", index=False)
mu = t.pivot(index="parameter", columns="output", values="mu_star")
rel = mu / mu.max()          # each output's mu* relative to its top parameter
key = ["misfit_nee_night_centered", "misfit_nee_day_centered", "misfit_modis_leaf_area_index",
       "misfit_landtrendr_aboveground_biomass", "misfit_soilgrids_soil_organic_carbon",
       "gpp", "reco", "nee", "peak_lai", "wood_change", "soil_change"]
rel = rel[key]
rel["score"] = rel.max(axis=1)
rel = rel.sort_values("score", ascending=False)
short = {c: c.replace("misfit_", "m_").replace("nee_night_centered", "night").replace("nee_day_centered", "day")
         .replace("modis_leaf_area_index", "lai").replace("landtrendr_aboveground_biomass", "agb")
         .replace("soilgrids_soil_organic_carbon", "soc") for c in rel.columns}
pd.set_option("display.width", 250)
print(rel.rename(columns=short).round(2).to_string())
rel.rename(columns=short).round(3).to_csv(pc.OUT / "morris_relative_mu_star.csv")
print("sigma/mu* (total misfit):")
tt = t[t.output == "misfit_total"].set_index("parameter")
print((tt.sigma / tt.mu_star).round(2).sort_values().to_string())
print("run summaries (median, 5-95%):")
for n, v in outputs.items():
    print(f"  {n}: {np.nanmedian(v):.3g} [{np.nanpercentile(v,5):.3g}, {np.nanpercentile(v,95):.3g}]")
