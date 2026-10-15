"""Final-prior diagnostics: plausibility criteria, residual scale of the best runs vs the noise sd, low MODIS LAI."""
import sys; from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import numpy as np, pandas as pd, phase2_common as pc
d = np.load(HERE / "prior_predictive.npz", allow_pickle=True); c = pc.load_cache()
S = {k[8:]: d[k] for k in d.files if k.startswith("summary_")}
on = list(d["override_names"]); ov = d["override_values"]
iw = ov[:, on.index("total_wood_carbon")] * (1 - ov[:, on.index("fine_root_fraction")] - ov[:, on.index("coarse_root_fraction")]); isoil = ov[:, on.index("soil_carbon")]
w = (S["wood_change"] + iw) / iw
crit = {"gpp<1000": S["gpp"] < 1000, "gpp>1800": S["gpp"] > 1800, "nee>0": S["nee"] > 0, "nee<-700": S["nee"] < -700,
        "jja_lai<3": S["jja_mean_lai"] < 3, "jja_lai>8": S["jja_mean_lai"] > 8, "peak_lai>8": S["peak_lai"] > 8,
        "wood<0.8x": w < 0.8, "wood>2x": w > 2, "ra/gpp>1": S["ra_over_gpp"] > 1, "soil|chg|>20%": np.abs(S["soil_change"]) / isoil > 0.2}
t = pd.Series({k: v.mean() for k, v in crit.items()}).round(3); t.to_csv(pc.OUT / "final_criteria.csv"); print(t.to_dict())
m = pc.misfits(d["predictions"], c); tot = sum(m.values()); best = np.argsort(tot)[: max(5, len(tot) // 20)]
rows = []
for s in pc.SOURCES:
    k = c["source"] == s; r = c["y"][k] - d["predictions"][best][:, k]
    rows.append(dict(source=s, rms_resid_best5pct=np.sqrt(np.mean(r ** 2)), mean_resid_best=np.mean(r), median_total_sd=np.median(c["sd"][k]),
                     chi2_per_obs_best=np.mean((r / c["sd"][k]) ** 2)))
print(pd.DataFrame(rows).round(2).to_string(index=False))
k = c["source"] == "modis_leaf_area_index"; y = c["y"][k]; tt = c["time"][k]; p = np.median(d["predictions"][:, k], 0)
low = y < 2.5
print("MODIS JJA obs:", k.sum(), "below 2.5:", low.sum(), "dates:", [str(x.date()) for x in tt[low]][:12])
print("low obs values:", np.round(y[low], 2), "prior-median prediction there:", np.round(p[low], 1))
print("best-run LAI residual excluding low obs rms:", np.sqrt(np.mean((y[~low] - d["predictions"][best][:, k][:, ~low]) ** 2)).round(2))
