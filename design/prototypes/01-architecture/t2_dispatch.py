import time, traceback, numpy as np, jax, jax.numpy as jnp
import lsmas_proto as L
from probpipe import sample, workflow_run, condition_on, mean, NumericRecordBatch
# A. what does condition_on pick by itself?
rep = condition_on.check(L.model, L.DATA_OBS)
print("A. auto route:", rep.route, "| method:", rep.method, "| exact:", rep.exact)
# B. joint sampling of the model (prior predictive through the kernel)
with workflow_run(seed=1):
    try:
        sample(L.model, sample_shape=(2,)); print("B. sample(model): OK")
    except Exception as e:
        print("B. sample(model) FAILS:", type(e).__name__, str(e).splitlines()[0][:160])
# C. prior predictive the workaround way: draw the prior, push theta through G as a batch
with workflow_run(seed=2):
    draws = sample(L.prior, sample_shape=(32,))
theta_batch = draws["theta"]
print("C. theta batch type:", type(theta_batch).__name__)
L.CALLS["n"] = 0; t = time.time()
with workflow_run(seed=3):
    G = L.nee_features(theta_batch)
print("C. G type:", type(G).__name__, "shape:", np.asarray(G.raw()).shape, "| calls:", L.CALLS["n"], "| s:", round(time.time() - t, 2))
L.CALLS["n"] = 0; t = time.time()
with workflow_run(seed=3):
    Gs = L.nee_features.using(dispatch="sequential")(theta_batch)
print("C'. sequential s:", round(time.time() - t, 2), "calls:", L.CALLS["n"])
# D. lift over the prior law itself (pushforward)
L.CALLS["n"] = 0
with workflow_run(seed=4):
    try:
        push = L.nee_features.using(n_broadcast_samples=16)(L.prior["theta"])
        print("D. pushforward:", type(push).__name__, push.provenance.metadata.get("route"), "| calls:", L.CALLS["n"])
    except Exception as e:
        print("D. pushforward FAILS:", type(e).__name__, str(e).splitlines()[0][:200])
# E. identical call again: cached?
L.CALLS["n"] = 0
with workflow_run(seed=3):
    L.nee_features(theta_batch)
print("E. repeat identical batch -> calls:", L.CALLS["n"])
