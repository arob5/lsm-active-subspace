import time, numpy as np
import enskit  # first: sets jax x64 before any ProbPipe object exists
import lsmas_proto as L
from lsmas_backend import GaussianSimulatorLikelihood, tempered_condition_on
from probpipe import workflow_run, condition_on, mean
lik = GaussianSimulatorLikelihood(L.nee_features, component="nee", theta_spec=L.THETA_SPEC, reference_variance=L.VREF,
    classes=np.where(L.IS_DAY, "day", "night"), scale_slots={"day": "s2_day", "night": "s2_night"})
model = lik * L.prior

L.CALLS["n"] = 0; t = time.time()
with workflow_run(seed=0):
    try:
        post = tempered_condition_on(model, L.DATA_OBS, learning_rate=L.ETA_VEC, method="blackjax_rwmh", method_options={"num_results": 20, "num_warmup": 10, "num_chains": 2})
        print("RWMH OK:", type(post).__name__, "| SIPNET runs:", L.CALLS["n"], "| s:", round(time.time()-t, 1))
    except Exception as e:
        print("RWMH FAILS:", type(e).__name__, str(e).splitlines()[0][:200])
