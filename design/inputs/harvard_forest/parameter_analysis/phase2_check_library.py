"""Check: the library map's SIPNET overrides equal phase2_prior_iterate.to_overrides at the same natural values."""
import sys, warnings; warnings.simplefilter("ignore")
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, jax, config
import phase2_final as fin, phase2_prior_iterate as it
vector, prior, smap, external, _, _ = fin.build()
theta = np.asarray(prior.sample(jax.random.key(3), 5)); ds = vector.dataset(theta)
f = smap.sipnet_parameter_fields(vector, theta, external_inputs=external)
worst = 0.0
for j in range(5):
    lib = {n: float(f[n].sel(site=config.SITE).isel(sample=j)) if "sample" in f[n].dims else float(f[n].sel(site=config.SITE)) for n in f.data_vars}
    row = {p: float(np.asarray(ds[p]).reshape(5, -1)[j, 0]) for p in vector.parameter_names if p != "allocation"}
    a = [float(np.asarray(ds[f"allocation.{k}"]).reshape(5, -1)[j, 0]) for k in ("leaf", "wood", "fine_root")]
    row.update(dict(zip(it.ALLOCATION, a)))
    mine = it.to_overrides(row, it.BASE_FIXED)
    d = {k: abs(lib[k] - mine[k]) / max(abs(mine[k]), 1e-12) for k in mine}; bad = {k: (lib[k], mine[k]) for k, v in d.items() if v > 1e-9}
    if bad: print(j, bad)
    rel = max(d.values())
    worst = max(worst, rel)
print("max relative difference library vs numpy mapping:", worst, "; log_prob finite:", bool(np.all(np.isfinite(np.asarray(prior.log_prob(theta))))))
