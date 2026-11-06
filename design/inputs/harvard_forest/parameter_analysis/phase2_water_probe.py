"""One-at-a-time probes around the i3 prior medians: JJA/May daytime NEE, JJA night, GPP, peak LAI."""
import sys, warnings; warnings.simplefilter("ignore")
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
def main():
    import phase2_common as pc, fast_forward, phase2_prior_iterate as it
    c = pc.load_cache(); prior = it.ITERATIONS["i3"][0]
    med = it.sample(prior, 4000, 0, it.ic_members()).median()
    probes = [("median", {})] + [(f"{k}={v}", {k: v}) for k, vs in [
        ("soil_water_holding_capacity", [10, 88, 300]), ("water_use_efficiency", [17.5, 30]),
        ("water_removal_fraction", [0.2, 0.5]), ("vapor_pressure_deficit_slope", [0.01]),
        ("respiration_share", [0.05]), ("photosynthetic_capacity", [450]), ("half_saturation_light", [6])] for v in vs]
    batch = []
    for _, ch in probes:
        r = med.copy()
        for k, v in ch.items(): r[k] = v
        batch.append(it.to_overrides(r, it.BASE_FIXED) if all(k in med for k in ch) else None)
        if batch[-1] is None:
            o = it.to_overrides(med, it.BASE_FIXED); o.update(ch); batch[-1] = o
    res = fast_forward.run_batch(batch, c["predictor"], n_workers=7)
    mon = c["time"].month; s = c["source"]; rows = []
    sel = {"day_may": (s == "nee_day_centered") & (mon == 5), "day_jja": (s == "nee_day_centered") & np.isin(mon, [6, 7, 8]),
           "night_jja": (s == "nee_night_centered") & np.isin(mon, [6, 7, 8]), "night_djf": (s == "nee_night_centered") & np.isin(mon, [12, 1, 2])}
    rows.append(dict(probe="OBSERVED", **{k: c["y"][m].mean() for k, m in sel.items()}))
    for (name, _), o in zip(probes, res):
        a = o["annual"]; rows.append(dict(probe=name, **{k: o["predictions"][m].mean() for k, m in sel.items()},
                    gpp=a.gpp.mean(), nee=a.nee.mean(), gpp_2016=a.gpp.loc[2016], gpp_2014=a.gpp.loc[2014]))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
if __name__ == "__main__":
    main()
