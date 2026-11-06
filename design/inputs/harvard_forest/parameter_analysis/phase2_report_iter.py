"""Re-print an iteration's tables from its saved npz (no runs)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import phase2_common as pc
d = np.load(pc.OUT / f"iter_{sys.argv[1]}.npz", allow_pickle=True)
cache = pc.load_cache()
names = list(d["theta_names"]); th = d["theta"]
on = list(d["override_names"]); ov = d["override_values"]
lcpa = ov[:, on.index("leaf_carbon_per_area")]
iw = th[:, names.index("initial_wood_carbon")] * 1000; isoil = th[:, names.index("initial_soil_organic_carbon")] * 1000
summ = pc.summaries(d["annual"], list(d["annual_columns"]), lcpa, iw, isoil)
cov = pc.coverage_table(d["predictions"], cache); pl = pc.plausibility_table(summ)
cov.to_csv(pc.OUT / f"iter_{sys.argv[1]}_coverage.csv"); pl.to_csv(pc.OUT / f"iter_{sys.argv[1]}_plausibility.csv")
ok = pc.plausible_fraction(summ, iw)
print("failures", (d["errors"] != "").sum(), "all-plausible", ok.mean().round(2))
pd.set_option("display.width", 200)
print(cov.round(2).to_string()); print(pl.round(3).to_string())
# which parameters drive implausibility: rank correlation with each summary
import scipy.stats as st
S = pd.DataFrame(summ); T = pd.DataFrame(th, columns=names)
m = np.isfinite(S["gpp"])
C = pd.DataFrame({s: [st.spearmanr(T[p][m], S[s][m])[0] for p in names] for s in ["gpp", "reco", "nee", "peak_lai", "wood_change", "soil_change"]}, index=names)
print(C.round(2).to_string())
