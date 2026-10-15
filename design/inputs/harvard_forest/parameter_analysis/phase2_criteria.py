"""Per-criterion plausibility failure fractions for an iteration."""
import sys; from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import numpy as np, pandas as pd, phase2_common as pc
d = np.load(pc.OUT / f"iter_{sys.argv[1]}.npz", allow_pickle=True); n = list(d["theta_names"]); th = pd.DataFrame(d["theta"], columns=n)
on = list(d["override_names"]); ov = d["override_values"]; iw = th.initial_wood_carbon.values * 1000
S = pd.DataFrame(pc.summaries(d["annual"], list(d["annual_columns"]), ov[:, on.index("leaf_carbon_per_area")], iw, th.initial_soil_organic_carbon.values * 1000))
w = (S.wood_change + iw) / iw
crit = {"gpp<1000": S.gpp < 1000, "gpp>1800": S.gpp > 1800, "nee>0": S.nee > 0, "nee<-700": S.nee < -700, "lai<3": S.peak_lai < 3,
        "lai>8": S.peak_lai > 8, "wood<0.8x": w < 0.8, "wood>2x": w > 2, "ra/gpp>1": S.ra_over_gpp > 1, "soil |change|>20%": (S.soil_change.abs() / (th.initial_soil_organic_carbon.values * 1000)) > 0.2}
t = pd.Series({k: v.mean() for k, v in crit.items()}).round(2); t.to_csv(pc.OUT / f"iter_{sys.argv[1]}_criteria.csv"); print(t.to_dict())
