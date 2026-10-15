import time, numpy as np, jax, jax.numpy as jnp
import lsmas_proto as L
from probpipe import sample, workflow_run, condition_on, mean, quantile
print("eta:", {k: round(v, 3) for k, v in L.ETA.items()}, "n:", L.N_DAY, L.N_NIGHT)
# 1. one forward run, and time-axis alignment with A's columns
th = {"photosynthetic_capacity": 251.0, "respiration_share": 0.09, "optimum_photosynthesis_temperature": 22.0, "soil_respiration_q10": 2.04}
t = time.time(); x = L.nee_rates(th); print("one run s:", round(time.time() - t, 2), x.shape)
out = L._MODEL(**L.FIXED, **L.to_sipnet(th)).outputs["nee"]
import json, pandas as pd
ax = json.load(open(L.DATA / "step_axis.json")); t = pd.DatetimeIndex(out.time.values)
print("SIPNET time axis == A's columns:", len(t) == ax["n_steps"] == L.A.shape[1] and str(t[0]) == ax["first_step_end"] and str(t[-1]) == ax["last_step_end"])
g = L.A @ x; r = L.Y - g
print("rms resid day/night:", np.sqrt(np.mean(r[L.IS_DAY]**2)).round(2), np.sqrt(np.mean(r[~L.IS_DAY]**2)).round(2))
# 2. kernel construction ran no SIPNET
print("calls so far:", L.CALLS["n"])
print("model:", L.model, "| components:", list(L.model.event_spec.components))
# 3. prior and IG median
with workflow_run(seed=0):
    d = sample(L.prior, sample_shape=(4000,))
print("IG s2_day median:", float(np.median(np.asarray(d["s2_day"].raw()))))
print("prior draw keys:", list(d.raw().keys()) if hasattr(d.raw(), "keys") else type(d.raw()))
