"""Prototype of the toy problem: SIPNET's canopy equations in JAX, with smooth phenology.

The toy keeps what makes the SIPNET problem hard for parameter reduction (products of parameters,
a light response that saturates, temperature and VPD limits, Q10 respiration) and drops what makes
it expensive or non-differentiable (carbon and water pools, step-function phenology, output
rounding). It runs on the shipped driver and observation operator, so its 400 features are the
real problem's features.

Equations, per 3-hour step s with driver tair, tsoil, PAR, VPD (SIPNET's own, `sipnet.c:600-640`):

    leaf carbon  C_L = leaf_on_growth * sigmoid((GDD - gdd_on) / 30) * sigmoid((off - doy) / 5)
    GPP          = P * 1.2e-8 * 86400 * C_L * dTemp * dVpd * dLight        [g C m-2 d-1]
    foliar resp  = rho * P * 1.2e-8 * 86400 * C_L * 1.8 ** ((tair - Topt) / 10)
    wood resp    = r10_wood * WOOD * 1.8 ** ((tair - 10) / 10) / 365
    soil resp    = F10 * Q10 ** ((tsoil - 10) / 10) / 365
    NEE rate     = (resp - GPP) / 86400 * 1e6 / 12.011                      [umol m-2 s-1]

with dTemp, dVpd and dLight as in SIPNET (the 6-layer Simpson canopy for dLight). P and rho enter
as in SIPNET after the photosynthesis reparameterization: aMax*aMaxFrac + respPerGram = P*cFracLeaf
and respPerGram = rho*P*cFracLeaf, and SIPNET's conversion factor leafCSpWt/cFracLeaf*LAI is
C_L/cFracLeaf, so cFracLeaf cancels.
"""

from __future__ import annotations

import time

import lsmas  # noqa: F401  (EnsKit first: 64-bit JAX)
import jax
import jax.numpy as jnp
import numpy as np
from pysipnet import ClimateDrivers

from lsmas.data import load_harvard_forest
from lsmas.priors import laws_from_table, prior_medians

#: The toy's calibration parameters: rows of the shipped prior table.
TOY_PARAMETERS = (
    "photosynthetic_capacity",
    "respiration_share",
    "optimum_photosynthesis_temperature",
    "photosynthesis_temperature_range",
    "half_saturation_light",
    "vapor_pressure_deficit_slope",
    "vapor_pressure_deficit_exponent",
    "light_extinction_coefficient",
    "leaf_on_growing_degree_days",
    "leaf_off_day",
    "leaf_on_growth",
    "wood_respiration_rate_at_10c",
    "soil_respiration_flux_at_10c",
    "soil_respiration_q10",
)
WOOD_CARBON = 8660.0          # g C m-2, the prior median of the initial wood
FOLIAR_Q10 = 1.8
GDD_WIDTH, LEAF_OFF_WIDTH = 30.0, 5.0

DATA = load_harvard_forest()
TABLE = DATA.prior_table.loc[list(TOY_PARAMETERS)]
LAWS = laws_from_table(TABLE)
MEDIANS = prior_medians(TABLE)


def _driver():
    frame = ClimateDrivers.from_path(DATA.driver_path, time_zone="UTC").pandas
    years = frame["year"].to_numpy()
    tair = frame["air_temperature"].to_numpy()
    length = frame["timestep_length"].to_numpy()
    gdd = np.zeros_like(tair)
    for year in np.unique(years):  # growing degree days since 1 January, base 0 C
        inside = years == year
        gdd[inside] = np.cumsum(np.maximum(tair[inside], 0.0) * length[inside])
    return {
        "tair": jnp.asarray(tair),
        "tsoil": jnp.asarray(frame["soil_temperature"].to_numpy()),
        "par": jnp.asarray(np.maximum(frame["photosynthetically_active_radiation"].to_numpy(), 0.0) / length),
        "vpd": jnp.asarray(np.maximum(frame["vapor_pressure_deficit"].to_numpy() * 1e-3, 1e-12)),
        "doy": jnp.asarray(frame["day_of_year"].to_numpy() + frame["hour_of_day"].to_numpy() / 24),
        "gdd": jnp.asarray(gdd),
    }


DRIVER = _driver()
_OPERATOR = DATA.operator.tocoo()
ROWS, COLS, VALS = (jnp.asarray(_OPERATOR.row), jnp.asarray(_OPERATOR.col), jnp.asarray(_OPERATOR.data))
N_FEATURES = DATA.operator.shape[0]


def light_effect(lai, par, attenuation, half_saturation):
    """SIPNET's calcLightEff: Simpson's rule over 6 canopy layers."""
    layers = jnp.arange(7) / 6
    coefficients = jnp.array([1, 4, 2, 4, 2, 4, 1]) / 18
    intensity = par[:, None] * jnp.exp(-attenuation * lai[:, None] * layers[None, :])
    efficiency = 1 - 2.0 ** (-intensity / half_saturation)
    return jnp.where((lai > 0) & (par > 0), efficiency @ coefficients, 0.0)


def nee_rate(theta):
    d = DRIVER
    leaf_carbon = (
        theta["leaf_on_growth"]
        * jax.nn.sigmoid((d["gdd"] - theta["leaf_on_growing_degree_days"]) / GDD_WIDTH)
        * jax.nn.sigmoid((theta["leaf_off_day"] - d["doy"]) / LEAF_OFF_WIDTH)
    )
    lai = leaf_carbon / 30.7092
    t_opt = theta["optimum_photosynthesis_temperature"]
    t_min = t_opt - theta["photosynthesis_temperature_range"]
    t_max = 2 * t_opt - t_min
    d_temp = jnp.maximum((t_max - d["tair"]) * (d["tair"] - t_min) / ((t_max - t_min) / 2) ** 2, 0.0)
    d_vpd = jnp.maximum(
        1 - theta["vapor_pressure_deficit_slope"] * d["vpd"] ** theta["vapor_pressure_deficit_exponent"], 0.0
    )
    d_light = light_effect(lai, d["par"], theta["light_extinction_coefficient"], theta["half_saturation_light"])
    capacity = theta["photosynthetic_capacity"] * 12e-9 * 86400 * leaf_carbon
    gpp = capacity * d_temp * d_vpd * d_light
    foliar = theta["respiration_share"] * capacity * FOLIAR_Q10 ** ((d["tair"] - t_opt) / 10)
    wood = theta["wood_respiration_rate_at_10c"] * WOOD_CARBON * FOLIAR_Q10 ** ((d["tair"] - 10) / 10) / 365
    soil = (
        theta["soil_respiration_flux_at_10c"]
        * theta["soil_respiration_q10"] ** ((d["tsoil"] - 10) / 10)
        / 365
    )
    return (foliar + wood + soil - gpp) / 86400 * 1e6 / 12.011, gpp


@jax.jit
def features(theta):
    rate, _ = nee_rate(theta)
    return jax.ops.segment_sum(VALS * rate[COLS], ROWS, num_segments=N_FEATURES)


if __name__ == "__main__":
    theta = {name: jnp.asarray(value) for name, value in MEDIANS.items()}
    start = time.time()
    g = features(theta).block_until_ready()
    print(f"first call (compile) {time.time() - start:.2f} s")
    start = time.time()
    for _ in range(20):
        features(theta).block_until_ready()
    print(f"one evaluation {(time.time() - start) / 20 * 1e3:.1f} ms")
    jac = jax.jit(jax.jacfwd(features))
    jac(theta)
    start = time.time()
    J = jac(theta)
    jax.block_until_ready(J)
    print(f"Jacobian {(time.time() - start) * 1e3:.1f} ms")
    _, gpp = nee_rate(theta)
    years = 9  # 2012-2020 of the features; the driver runs to 2024
    rate, _ = nee_rate(theta)
    print("annual GPP at medians (whole driver mean):", float(jnp.sum(gpp) * 0.125 / 13), "g C m-2 yr-1")
    print("annual NEE at medians:", float(jnp.sum(rate) * 0.125 * 86400 * 12.011 / 1e6 / 13), "g C m-2 yr-1")
    y = DATA.observations
    print("features at medians vs observed: mean", float(g.mean()), y.mean(), "corr", np.corrcoef(np.asarray(g), y)[0, 1])
