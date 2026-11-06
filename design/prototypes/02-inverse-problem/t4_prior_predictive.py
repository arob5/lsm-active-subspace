"""Prior predictive check of the draft 43-parameter prior.

For J prior draws: SIPNET's features G(theta) and the replicated data y_rep = G + sqrt(s2 v) e, with
s2 ~ IG(2, 1.678) per class. Reports

- failures (SIPNET refused or stopped);
- coverage: the fraction of observed features inside the 90% interval of y_rep, by class and
  season, with the fraction below and above, and the mean z = (y - median) / sd;
- plausibility of the runs: annual GPP, ecosystem respiration and NEE (2012-2020 means,
  g C m-2 yr-1), Ra/GPP, peak LAI, and the change in wood and soil carbon over 2012-2024.

Usage: python t4_prior_predictive.py [J] [tag]. Writes out/prior_predictive_<tag>.npz.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from prior_table import FIXED, NAMES, draw, to_sipnet
from reference import DATA, OUT, SIMULATOR
from lsmas.sipnet import SIMULATION_FAILURES

J = int(sys.argv[1]) if len(sys.argv) > 1 else 400
TAG = sys.argv[2] if len(sys.argv) > 2 else "draft"
STEP_TO_RATE = 1e6 / 12.011 / 10800.0
LEAF_C_PER_AREA = FIXED["leaf_carbon_per_area"]


def run(theta):
    try:
        frame = SIMULATOR._model(**to_sipnet(theta)).outputs.pandas
    except SIMULATION_FAILURES as failure:
        return None, type(failure).__name__
    features = DATA.operator @ (frame["net_ecosystem_exchange"].to_numpy() * STEP_TO_RATE)
    years = frame["year"]
    calibration = frame[years <= 2020]
    annual = calibration.groupby("year")[
        ["gross_primary_production", "ecosystem_respiration", "net_ecosystem_exchange", "autotrophic_respiration"]
    ].sum().mean()
    peak_lai = (frame.groupby("year")["leaf_carbon"].max() / LEAF_C_PER_AREA).mean()
    summary = {
        "gpp": annual["gross_primary_production"],
        "reco": annual["ecosystem_respiration"],
        "nee": annual["net_ecosystem_exchange"],
        "ra_over_gpp": annual["autotrophic_respiration"] / annual["gross_primary_production"],
        "peak_lai": peak_lai,
        "wood_change": frame["wood_carbon"].iloc[-1] - frame["wood_carbon"].iloc[0],
        "soil_change": frame["soil_carbon"].iloc[-1] - frame["soil_carbon"].iloc[0],
    }
    return features, summary


thetas = draw(J, seed=11)
start = time.time()
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(run, thetas))
print(f"{J} runs in {time.time() - start:.0f} s")

failures = [summary for features, summary in results if features is None]
ok = [i for i, (features, _) in enumerate(results) if features is not None]
print(f"failures: {len(failures)} ({', '.join(sorted(set(failures)))})")
G = np.stack([results[i][0] for i in ok])
summaries = pd.DataFrame([results[i][1] for i in ok])
theta_ok = np.array([[thetas[i][name] for name in NAMES] for i in ok])
nonfinite = ~np.isfinite(G).all(axis=1)
print(f"non-finite feature rows: {nonfinite.sum()}")

rng = np.random.default_rng(1)
classes = DATA.feature_classes
v = np.where(classes == "day", 8.0, 1.5)
s2 = {k: 1.678 / rng.gamma(2.0, size=len(ok)) for k in ("day", "night")}
s2_per_feature = np.where(classes == "day", s2["day"][:, None], s2["night"][:, None])
y_rep = G + np.sqrt(s2_per_feature * v) * rng.standard_normal(G.shape)
low, median, high = np.quantile(y_rep, [0.05, 0.5, 0.95], axis=0)
y = DATA.observations
z = (y - median) / y_rep.std(axis=0)
months = DATA.features["month"].to_numpy()
season = np.select([np.isin(months, [12, 1, 2]), np.isin(months, [3, 4, 5]), np.isin(months, [6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")

print("\ncoverage of the 90% interval    inside below above   z")
for k in ("day", "night"):
    for s in ("all", "DJF", "MAM", "JJA", "SON"):
        mask = (classes == k) & ((season == s) if s != "all" else True)
        print(f"  {k:5s} {s:3s} (n={mask.sum():3d})            {np.mean((y >= low) & (y <= high), where=mask):.2f}  "
              f"{np.mean(y < low, where=mask):.2f}  {np.mean(y > high, where=mask):.2f}  {np.mean(z, where=mask):+.2f}")

print("\nplausibility, 5 / 50 / 95%")
for column in summaries:
    q = summaries[column].quantile([0.05, 0.5, 0.95]).to_numpy()
    print(f"  {column:12s} {q[0]:9.3g} {q[1]:9.3g} {q[2]:9.3g}")
bounds = {
    "GPP < 1000": summaries.gpp < 1000, "GPP > 1800": summaries.gpp > 1800,
    "NEE > 0 (source)": summaries.nee > 0, "NEE < -700": summaries.nee < -700,
    "peak LAI > 8": summaries.peak_lai > 8, "peak LAI < 3": summaries.peak_lai < 3,
    "Ra > GPP": summaries.ra_over_gpp > 1,
}
print("\nfraction outside plausible bounds")
for label, mask in bounds.items():
    print(f"  {label:18s} {mask.mean():.2f}")
observed_nee = -np.nan  # annual NEE is not observed directly; the features are
np.savez(OUT / f"prior_predictive_{TAG}.npz", G=G, theta=theta_ok, names=np.array(NAMES), summaries=summaries.to_numpy(), summary_names=np.array(summaries.columns))
