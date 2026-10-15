"""Diagnose an iteration: monthly observed vs predictive NEE (day/night) and LAI, and theta correlations."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, scipy.stats as st
import phase2_common as pc
d = np.load(pc.OUT / f"iter_{sys.argv[1]}.npz", allow_pickle=True); c = pc.load_cache()
P = d["predictions"]; ok = np.all(np.isfinite(P), 1); P = P[ok]; th = pd.DataFrame(d["theta"][ok], columns=list(d["theta_names"]))
mon = c["time"].month; rows = []
for s, lab in [("nee_night_centered", "night"), ("nee_day_centered", "day"), ("modis_leaf_area_index", "lai")]:
    for m in sorted(set(mon[c["source"] == s])):
        k = (c["source"] == s) & (mon == m)
        per_run = P[:, k].mean(1)
        rows.append(dict(src=lab, month=m, n=int(k.sum()), obs=c["y"][k].mean(), q05=np.percentile(per_run, 5), q50=np.median(per_run), q95=np.percentile(per_run, 95)))
print(pd.DataFrame(rows).round(2).to_string(index=False))
k = (c["source"] == "nee_day_centered") & np.isin(mon, [6, 7, 8]); f = P[:, k].mean(1)
k2 = (c["source"] == "nee_night_centered") & np.isin(mon, [6, 7, 8]); g = P[:, k2].mean(1)
print("spearman with JJA day mean / JJA night mean:")
print(pd.DataFrame({"jja_day": [st.spearmanr(th[p], f)[0] for p in th], "jja_night": [st.spearmanr(th[p], g)[0] for p in th]}, index=th.columns).round(2).to_string())
