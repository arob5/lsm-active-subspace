import numpy as np, lsmas_proto as L
from lsmas_backend import GaussianSimulatorLikelihood, tempered_condition_on
from probpipe import sample, workflow_run, condition_on
lik = GaussianSimulatorLikelihood(L.nee_features, component="nee", theta_spec=L.THETA_SPEC, reference_variance=L.VREF,
    classes=np.where(L.IS_DAY, "day", "night"), scale_slots={"day": "s2_day", "night": "s2_night"})
model = lik * L.prior
with workflow_run(seed=0):
    try:
        d = sample(model, sample_shape=(3,)); print("sample(model) OK:", np.asarray(d["nee"].raw()).shape, "calls", L.CALLS["n"])
    except Exception as e:
        print("sample(model) FAILS:", type(e).__name__, str(e).splitlines()[0][:150])
    try:
        d = sample(condition_on(lik, {"theta": {n: 1.0 for n in L.NAMES} | {"photosynthetic_capacity": 250.0, "optimum_photosynthesis_temperature": 22.0, "respiration_share": 0.09, "soil_respiration_q10": 2.0}, "s2_day": 1.0, "s2_night": 1.0}), sample_shape=(2,))
        print("sample(lik at fixed given) OK:", np.asarray(d.raw()).shape)
    except Exception as e:
        print("sample(lik|given) FAILS:", type(e).__name__, str(e).splitlines()[0][:150])
