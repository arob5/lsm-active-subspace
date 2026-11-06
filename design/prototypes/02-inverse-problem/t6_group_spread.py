"""How much of the prior predictive spread comes from each group of parameters?

Draws 200 prior samples in which only one group varies and the others sit at their prior medians,
and reports the 5/50/95% of annual GPP and NEE and the fraction of runs that make the site a source.
Groups: the 15 phase-2 parameters (P2) and the 28 others.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from prior_table import ROWS, draw, medians, to_sipnet
from t5_median_shift import annual

P2 = {row.name for row in ROWS if row.source == "P2"} | {"leaf_allocation", "photosynthesis_temperature_range"}
GROUPS = {"phase-2 parameters": P2, "the others": {row.name for row in ROWS} - P2, "all": {row.name for row in ROWS}}
centre = medians()
draws = draw(200, seed=5)
for label, group in GROUPS.items():
    points = [to_sipnet(centre | {name: theta[name] for name in group}) for theta in draws]
    with ThreadPoolExecutor(8) as pool:
        results = np.array(list(pool.map(annual, points)))
    gpp, nee = results[:, 0], results[:, 1]
    print(f"{label:20s} ({len(group):2d}) GPP {np.percentile(gpp, [5, 50, 95]).round()}  NEE {np.percentile(nee, [5, 50, 95]).round()}  source {np.mean(nee > 0):.2f}")
