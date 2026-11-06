"""Which parameters move the prior-median run's GPP and NEE away from the reference run?

Runs SIPNET at the prior medians, then with one SIPNET parameter at a time set back to its value at
the reference point, and reports the annual GPP and NEE (2012-2020) of each run.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from prior_table import medians, to_sipnet
from reference import SIMULATOR, reference_point


def annual(parameters):
    frame = SIMULATOR._model(**parameters).outputs.pandas
    frame = frame[frame["year"] <= 2020]
    sums = frame.groupby("year")[["gross_primary_production", "net_ecosystem_exchange"]].sum().mean()
    return sums["gross_primary_production"], sums["net_ecosystem_exchange"]


median_point = to_sipnet(medians())
reference = reference_point()
changed = [name for name in median_point if not np.isclose(median_point[name], reference[name], rtol=1e-3)]
points = [reference, median_point] + [median_point | {name: reference[name]} for name in changed]
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(annual, points))
print(f"reference: GPP {results[0][0]:.0f}, NEE {results[0][1]:.0f}")
print(f"prior median: GPP {results[1][0]:.0f}, NEE {results[1][1]:.0f}")
rows = sorted(zip(changed, results[2:]), key=lambda item: -abs(item[1][0] - results[1][0]))
for name, (gpp, nee) in rows:
    print(f"  {name:40s} {median_point[name]:>10.4g} -> {reference[name]:<10.4g} GPP {gpp - results[1][0]:+6.0f}  NEE {nee - results[1][1]:+6.0f}")
