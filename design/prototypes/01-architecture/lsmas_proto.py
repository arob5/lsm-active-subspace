"""Prototype of the lsm-active-subspace stack: SIPNET -> ProbPipe model -> EnsKit EKI method.

Scratch code to verify design-doc claims. Not the repo's code.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from probpipe import (
    BijectorTransformedDistribution,
    bijector_for,
    unit_interval,
    Distribution,
    InverseGamma,
    LogNormal,
    Normal,
    NumericArraySpec,
    NumericRecordSpec,
    OutputSpec,
    conditional_distribution,
    function,
)
from pysipnet import ClimateDrivers, ModelFlags, SIPNETModel, SIPNETRunner, niwot_reference_parameters

DATA = Path(__file__).parent / "data"
# The corrected ERA5 driver made by spatial-lsm-calibration's run/prepare_drivers.py (not copied here).
DRIVER = Path(os.environ.get("LSMAS_DRIVER", Path(
    "~/Desktop/git-repos/spatial-lsm-calibration/.claude/worktrees/harvard-forest-models/experiments/"
    "single_site_mcmc_vs_eki/output/drivers/ERA5_4977_1/ERA5.1.2011-12-31.2024-12-31.clim").expanduser()))
A = sp.load_npz(DATA / "A.npz").tocsr()
FEATURES = pd.read_csv(DATA / "features.csv")
Y = FEATURES["y"].to_numpy()
IS_DAY = (FEATURES["class"] == "day").to_numpy()
N_DAY, N_NIGHT = int(IS_DAY.sum()), int((~IS_DAY).sum())
STEP_TOTAL_TO_RATE = 1e6 / 12.011 / (3 * 3600)  # g C m-2 per 3-h step -> umol CO2 m-2 s-1

# ---- fixed SIPNET parameters (HF fixed table + representative values for the rest) ----
_FIXED_CSV = Path(
    "~/Desktop/git-repos/spatial-lsm-calibration/.claude/worktrees/harvard-forest-models/"
    "experiments/single_site_mcmc_vs_eki/model/fixed_sipnet_parameters.csv"
).expanduser()
FIXED = {r.sipnet_parameter_name: float(r.value) for r in pd.read_csv(_FIXED_CSV).itertuples()}
FIXED.update(
    leaf_area_index=0.0,
    total_wood_carbon=1000 * 10.0 / (1 - 0.03 - 0.16),
    soil_carbon=10_000.0,
    leaf_allocation=0.18,
    wood_allocation=0.45,
    fine_root_allocation=0.065,
    base_wood_respiration_rate=0.0155 / 1.80944,
    base_coarse_root_respiration_rate=0.0155 / 3.20614,
    soil_water_holding_capacity=47.4,
    leaf_on_growth=94.9,
    leaf_on_growing_degree_days=741.6,
)
F_FRAC, C_FRAC = FIXED["daily_mean_photosynthesis_fraction"], FIXED["leaf_carbon_fraction"]

# ---- calibration parameters (a 4-parameter subset for the prototype) ----
NAMES = ("photosynthetic_capacity", "respiration_share", "optimum_photosynthesis_temperature",
         "soil_respiration_q10")


def to_sipnet(theta) -> dict[str, float]:
    """Calibration parameters -> pySIPNET overrides (the HF photosynthesis map)."""
    P, rho = float(theta["photosynthetic_capacity"]), float(theta["respiration_share"])
    t_opt, q10 = float(theta["optimum_photosynthesis_temperature"]), float(theta["soil_respiration_q10"])
    F10, soc_kg = 424.0, 10.0
    return dict(
        max_photosynthesis_rate=P * C_FRAC * (1 - rho) / F_FRAC,
        foliar_respiration_fraction=rho * F_FRAC / (1 - rho),
        optimum_photosynthesis_temperature=t_opt,
        min_photosynthesis_temperature=t_opt - 23.958,
        half_saturation_light=11.0,
        soil_respiration_q10=q10,
        base_soil_respiration_rate=F10 / (1000 * soc_kg * q10),
    )


_MODEL = SIPNETModel(
    SIPNETRunner(flags=ModelFlags.standard()),
    base_params=niwot_reference_parameters(),
    base_climate=ClimateDrivers.from_path(DRIVER, time_zone="UTC"),
)
CALLS = {"n": 0}
_LOCK = threading.Lock()


def nee_rates(theta) -> np.ndarray:
    """SIPNET's 3-hourly NEE at theta, umol m-2 s-1, shape (37992,)."""
    with _LOCK:
        CALLS["n"] += 1
    out = _MODEL(**FIXED, **to_sipnet(theta)).outputs["nee"]
    return out.values * STEP_TOTAL_TO_RATE


@function(dispatch="thread", max_workers=8, label="nee_features")
def nee_features(theta):
    """G(theta) = A x(theta): the 400 half-month day/night NEE features."""
    try:
        return jnp.asarray(A @ nee_rates(theta))
    except Exception:  # a failed run is a NaN row (EnsKit's contract)
        return jnp.full(A.shape[0], jnp.nan)


# ---- the likelihood: Gaussian features, per-class IG scales, learning rate eta ----
V_DAY, V_NIGHT, ETA_C = 8.0, 1.5, 3.0
M = FEATURES.assign(ym=FEATURES.year * 100 + FEATURES.month).groupby("class")["ym"].nunique()
ETA = {k: min(1.0, ETA_C * M[k] / (12 * n)) for k, n in (("day", N_DAY), ("night", N_NIGHT))}
VREF = np.where(IS_DAY, V_DAY, V_NIGHT)
ETA_VEC = np.where(IS_DAY, ETA["day"], ETA["night"])


class NEEFeatureLaw(Distribution):
    """Law of the 400 features given theta and the class scales.

    STOPGAP: unnormalized log-prob is the *tempered* Gaussian (eta * log N),
    while _sample draws the untempered data model.
    """

    def __init__(self, theta, s2_day, s2_night):
        super().__init__(OutputSpec(nee=NumericArraySpec((A.shape[0],))), label="NEEFeatureLaw")
        self.theta, self.s2_day, self.s2_night = theta, s2_day, s2_night

    # the pieces an ensemble Kalman method needs
    def predicted(self):
        return nee_features(self.theta).raw()

    def variances(self):
        return np.where(IS_DAY, float(self.s2_day), float(self.s2_night)) * VREF

    def _unnormalized_log_prob(self, value):
        r = np.asarray(value) - np.asarray(self.predicted())
        var = self.variances()
        return jnp.asarray(np.sum(ETA_VEC * (-0.5 * r**2 / var - 0.5 * np.log(2 * np.pi * var))))

    def _sample(self, key, sample_shape=()):
        z = np.asarray(jax.random.normal(key, (*sample_shape, A.shape[0])))
        return jnp.asarray(np.asarray(self.predicted()) + np.sqrt(self.variances()) * z)


THETA_SPEC = NumericRecordSpec(**{n: NumericArraySpec(()) for n in NAMES})
SCALAR = NumericArraySpec(())

likelihood = conditional_distribution(
    lambda theta, s2_day, s2_night: NEEFeatureLaw(theta, s2_day, s2_night),
    given_spec={"theta": THETA_SPEC, "s2_day": SCALAR, "s2_night": SCALAR},
    label="nee_likelihood",
)

theta_prior = (
    LogNormal("photosynthetic_capacity", float(np.log(np.sqrt(140 * 450))), float(np.log(450 / 140) / 3.92))
    * BijectorTransformedDistribution(
        "respiration_share", Normal("z", float(np.log(0.09 / 0.91)), 0.45), bijector_for(unit_interval)
    )
    * Normal("optimum_photosynthesis_temperature", 22.0, 2.5)
    * LogNormal("soil_respiration_q10", float(np.log(np.sqrt(1.3 * 3.2))), float(np.log(3.2 / 1.3) / 3.92))
)

prior = theta_prior.with_path_names({n: f"theta/{n}" for n in NAMES}) * InverseGamma(
    "s2_day", 2.0, 1.678
) * InverseGamma("s2_night", 2.0, 1.678)
model = (likelihood * prior).with_label("hf_nee_model")
DATA_OBS = {"nee": jnp.asarray(Y)}
