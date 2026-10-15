import time, numpy as np, jax, jax.numpy as jnp
import enskit  # noqa: F401  first: JAX x64
import lsmas_proto as L
from lsmas_backend import GaussianSimulatorLikelihood, tempered_condition_on
from probpipe import condition_on, workflow_run, mean, quantile, sample
from enskit import maps
from enskit.algorithms import eki

lik = GaussianSimulatorLikelihood(
    L.nee_features, component="nee", theta_spec=L.THETA_SPEC, reference_variance=L.VREF,
    classes=np.where(L.IS_DAY, "day", "night"), scale_slots={"day": "s2_day", "night": "s2_night"},
    label="nee_likelihood")
model = (lik * L.prior).with_label("hf_nee_model")
print("model:", model)
rep = condition_on.check(model, L.DATA_OBS)
print("auto:", rep.method, "| ranked:", [m for m in getattr(rep, "candidates", [])][:5])

# callback: accumulate H = sum_t delta_t * A_t^T R^-1 A_t from per-step statistical linearization
H = {"sum": None, "log": []}
def slr_callback(*, step, beta, increment, evaluation, noise):
    fit = maps.statistical_linearization(evaluation.ensemble, inputs="u", output=eki.PREDICTION)
    A = np.asarray(fit.map.op.to_dense())
    Rinv = 1.0 / np.diag(np.asarray(noise.to_dense()))
    w = 1.0 if increment is None else float(increment)
    Ht = w * (A.T * Rinv) @ A
    H["sum"] = Ht if H["sum"] is None else H["sum"] + Ht
    H["log"].append((step, round(beta, 4), None if increment is None else round(float(increment), 4)))

L.CALLS["n"] = 0; t = time.time()
with workflow_run(seed=7):
    post = tempered_condition_on(model, L.DATA_OBS, learning_rate=L.ETA_VEC, method="enskit_eki",
                                 method_options={"n_particles": 40, "callbacks": [slr_callback]})
print("EKI: s", round(time.time() - t, 1), "| SIPNET runs", L.CALLS["n"], "| steps", H["log"])
print("posterior:", type(post).__name__, "| method:", post.provenance.metadata.get("method"), "| exact:", post.provenance.metadata.get("exact"))
print("post['theta/soil_respiration_q10'] mean:", round(float(mean(post["theta/soil_respiration_q10"]).raw()), 3))
print("posterior means:", {k: round(float(np.asarray(v)), 3) for k, v in mean(post["theta"]).raw().items()})
print("s2 means:", round(float(mean(post["s2_day"]).raw()), 3), round(float(mean(post["s2_night"]).raw()), 3))
ev, V = np.linalg.eigh(H["sum"]); order = ev.argsort()[::-1]
print("accumulated H eigenvalues:", np.round(ev[order], 2))
print("leading direction (unconstrained coords):", dict(zip(L.NAMES, np.round(V[:, order[0]], 3))))
