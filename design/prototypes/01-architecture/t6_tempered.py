"""The learning rate lives in conditioning, not in the model."""
import time, numpy as np, jax.numpy as jnp
import enskit  # noqa: F401  first: JAX x64 before any ProbPipe object
import lsmas_proto as L
from lsmas_backend import GaussianSimulatorLikelihood, tempered_condition_on
from probpipe import condition_on, workflow_run, mean, unnormalized_log_prob, log_prob, sample
from scipy import stats

lik = GaussianSimulatorLikelihood(
    L.nee_features, component="nee", theta_spec=L.THETA_SPEC, reference_variance=L.VREF,
    classes=np.where(L.IS_DAY, "day", "night"), scale_slots={"day": "s2_day", "night": "s2_night"})
model = (lik * L.prior).with_label("hf_nee_model")          # a proper generative model

# 1. the likelihood is one law: its draws and its density agree
th = {"photosynthetic_capacity": 251.0, "respiration_share": 0.09, "optimum_photosynthesis_temperature": 22.0, "soil_respiration_q10": 2.04}
given = {"theta": th, "s2_day": 1.3, "s2_night": 0.8}
at = condition_on(lik, given)
with workflow_run(seed=0):
    y_sim = np.asarray(sample(at).raw())
g = L.A @ L.nee_rates(th); sd = np.sqrt(np.where(L.IS_DAY, 1.3, 0.8) * L.VREF)
print("1. log_prob(sample) == scipy:", np.isclose(float(log_prob(at, y_sim).raw()), stats.norm(g, sd).logpdf(y_sim).sum()))
print("   standardized residuals of a draw: mean %.3f sd %.3f" % (((y_sim - g) / sd).mean(), ((y_sim - g) / sd).std()))

# 2. the tempered target's density is prior + sum_i eta_i log N_i (differences between two points)
target = tempered_condition_on(model, L.DATA_OBS, learning_rate=L.ETA_VEC, method="unnormalized")
plain = condition_on.using(method="unnormalized")(model, L.DATA_OBS)
pt = lambda q10, s2d: {"theta": {**th, "soil_respiration_q10": q10}, "s2_day": s2d, "s2_night": 0.8}
p1, p2 = pt(2.04, 1.3), pt(1.6, 0.7)
def manual(p, eta):
    gp = L.A @ L.nee_rates(p["theta"]); sdp = np.sqrt(np.where(L.IS_DAY, p["s2_day"], p["s2_night"]) * L.VREF)
    return float(np.sum(eta * stats.norm(gp, sdp).logpdf(L.Y)) + log_prob(L.prior, p).raw())
for name, t, eta in (("tempered", target, L.ETA_VEC), ("plain", plain, 1.0)):
    d_pp = float(unnormalized_log_prob(t, p1).raw()) - float(unnormalized_log_prob(t, p2).raw())
    d_mn = manual(p1, eta) - manual(p2, eta)
    print(f"2. {name}: target difference {d_pp:.6f} vs manual {d_mn:.6f} -> equal: {np.isclose(d_pp, d_mn)}")

# 3. EKI through the tempered conditioning
L.CALLS["n"] = 0; t0 = time.time()
with workflow_run(seed=7):
    post = tempered_condition_on(model, L.DATA_OBS, learning_rate=L.ETA_VEC,
                                 method="enskit_eki", method_options={"n_particles": 40})
print("3. EKI:", type(post).__name__, post.provenance.metadata.get("method"), "| runs", L.CALLS["n"], "| s", round(time.time() - t0, 1))
print("   means:", {k: round(float(v), 3) for k, v in mean(post["theta"]).raw().items()},
      "s2:", round(float(mean(post["s2_day"]).raw()), 3), round(float(mean(post["s2_night"]).raw()), 3))

# 4. ProbPipe's RWMH on the same tempered target
L.CALLS["n"] = 0; t0 = time.time()
with workflow_run(seed=0):
    rw = tempered_condition_on(model, L.DATA_OBS, learning_rate=L.ETA_VEC, method="blackjax_rwmh",
                               method_options={"num_results": 20, "num_warmup": 10, "num_chains": 2})
print("4. RWMH:", type(rw).__name__, "| runs", L.CALLS["n"], "| s", round(time.time() - t0, 1))
