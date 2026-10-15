"""Leaf-on evidence: SIPNET's GDD (base 0 C, sipnet.c:230-235) by year from the drivers, the DOY
GDD crosses thresholds, and the DOY observed daytime NEE (12-24 UTC mean) first stays below -5 umol m-2 s-1."""
import sys, warnings; warnings.simplefilter("ignore")
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import runs, phase2_common as pc
x = runs.climate_drivers().xarray
t = pd.DatetimeIndex(x["time"].values); L = x["timestep_length"].values / np.timedelta64(1, "D") if np.issubdtype(x["timestep_length"].dtype, np.timedelta64) else x["timestep_length"].values
df = pd.DataFrame({"gdd": np.clip(x["air_temperature"].values, 0, None) * L, "tsoil": x["soil_temperature"].values}, index=t)
c = pc.load_cache(); m = c["source"] == "nee_day_centered"
obs = pd.Series(c["y"][m], index=c["time"][m])
rows = []
for y, g in df.groupby(df.index.year):
    cum = g["gdd"].cumsum(); doy = g.index.dayofyear
    r = dict(year=y)
    for th in (400, 500, 600, 700, 800):
        r[f"doy_gdd{th}"] = int(doy[np.argmax(cum.values >= th)]) if (cum.values >= th).any() else np.nan
    o = obs[obs.index.year == y].rolling(3).max()   # three consecutive days below -5
    r["doy_uptake"] = int(o.index[np.argmax(o.values < -5)].dayofyear) if (o.values < -5).any() else np.nan
    if np.isfinite(r["doy_uptake"]):
        r["gdd_at_uptake"] = float(cum[doy <= r["doy_uptake"]].iloc[-1])
        ts = g["tsoil"].resample("1D").mean(); r["tsoil_at_uptake"] = float(ts[ts.index.dayofyear == r["doy_uptake"]].iloc[0])
    rows.append(r)
T = pd.DataFrame(rows).set_index("year"); print(T.round(1).to_string()); print(T.describe().loc[["mean", "std"]].round(1).to_string())
T.to_csv(pc.OUT / "phenology_evidence.csv")
