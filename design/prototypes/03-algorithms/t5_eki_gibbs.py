"""EKI with Gibbs updates of the noise scales, built from EnsKit's step-level API, on the toy.

The tempered target is joint in the parameters and the class noise scales s2_k (day, night):

    pi_phi(z, s2) ∝ pi(z) prod_k IG(s2_k; a, b) prod_i N(y_i; G_i(z), s2_k(i) v_i)^(phi eta_i).

Each step, at an ensemble (z_j, s2_j) with predictions g_j at level phi:
  1. increment: the largest dphi <= 1 - phi keeping ESS >= J/2 for the weights of the
     scale-marginal targets (Botha et al. 2023, as in the Harvard Forest EKI-Gibbs),
        log w_j = sum_k [A_k(phi) log B_jk(phi) - A_k(phi') log B_jk(phi')],
        A_k(phi) = a + phi n_k / 2,  B_jk(phi) = b + phi q_jk / 2,
        n_k = sum_{i in k} eta_i,    q_jk = sum_{i in k} eta_i (y_i - g_ji)^2 / v_i;
  2. z step: EnsKit's assimilate with the common noise covariance R(sbar) / dphi,
        R(s)_ii = s_k(i) v_i / eta_i, sbar_k = 1 / mean_j(1 / s2_jk);
  3. evaluate g at the new particles (EnsKit's evaluate): the step's one batch of runs;
  4. Gibbs: s2_jk ~ IG(A_k(phi'), B_jk(phi')), exact for pi_phi' given z_j.

Compared with NUTS on the joint posterior (t1_toy_nuts.py observed inferred).
Usage: python t5_eki_gibbs.py [n_particles] [seed]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from enskit import kalman, linalg
from enskit.algorithms import eki
from enskit.distribution import Ensemble

from common import coordinates, forward_batch, problem, whitened
from lsmas._stopgaps.sampling import draw_with_key
from lsmas.problems.harvard_forest import NOISE_SCALE_PRIOR, REFERENCE_VARIANCE
from toy import DATA

J = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
A0, B0 = NOISE_SCALE_PRIOR["concentration"], NOISE_SCALE_PRIOR["scale"]
y = np.asarray(problem.data["nee"])
eta = np.asarray(problem.learning_rate["nee"])
v = np.vectorize(REFERENCE_VARIANCE.get)(DATA.feature_classes)
classes = {k: DATA.feature_classes == k for k in ("day", "night")}
n_k = {k: eta[m].sum() for k, m in classes.items()}
rng = np.random.default_rng(SEED)


def forward_z(z):
    return forward_batch(coordinates.whiten(np.asarray(z)))


def quadratic_forms(g):
    return {k: ((y - g) ** 2 * eta / v)[:, m].sum(1) for k, m in classes.items()}


def log_weights(q, phi, dphi):
    total = 0.0
    for k in classes:
        before = (A0 + phi * n_k[k] / 2) * np.log(B0 + phi * q[k] / 2)
        after = (A0 + (phi + dphi) * n_k[k] / 2) * np.log(B0 + (phi + dphi) * q[k] / 2)
        total = total + before - after
    return total


def increment(q, phi):
    def ess(dphi):
        w = log_weights(q, phi, dphi)
        w = np.exp(w - w.max())
        return w.sum() ** 2 / (w**2).sum()

    if ess(1 - phi) >= J / 2:
        return 1 - phi
    low, high = 0.0, 1 - phi
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if ess(mid) >= J / 2 else (low, mid)
    return low


def draw_scales(q, phi):
    return {k: 1 / rng.gamma(A0 + phi * n_k[k] / 2, 1 / (B0 + phi * q[k] / 2)) for k in classes}


def noise(scales_bar):
    s = np.where(classes["day"], scales_bar["day"], scales_bar["night"])
    return linalg.PSDDiagonal(jnp.asarray(s * v / eta))


start = time.time()
theta0 = coordinates.array_from_draws(draw_with_key(problem.prior, jax.random.key(SEED), (J,)))
state = eki.EKIState(Ensemble(u=jnp.asarray(coordinates.bijection.to_unconstrained(theta0))), key=jax.random.key(SEED + 100))
scales = {k: 1 / rng.gamma(A0, 1 / B0, size=J) for k in classes}
phi, steps = 0.0, 0
base = noise({"day": 1.0, "night": 1.0})
evaluation = eki.evaluate(state, forward_z, jnp.asarray(y), base, on_failure="repair")
while phi < 1.0:
    g = np.asarray(evaluation.ensemble[eki.PREDICTION])
    q = quadratic_forms(g)
    dphi = increment(q, phi)
    scales_bar = {k: 1 / np.mean(1 / scales[k]) for k in classes}
    state = eki.assimilate(state, evaluation, dphi, jnp.asarray(y), noise(scales_bar), update_rule=kalman.Matheron())
    evaluation = eki.evaluate(state, forward_z, jnp.asarray(y), base, on_failure="repair")
    phi = min(phi + dphi, 1.0)
    scales = draw_scales(quadratic_forms(np.asarray(evaluation.ensemble[eki.PREDICTION])), phi)
    steps += 1
print(f"EKI-Gibbs: J={J}, {steps} steps, {steps + 1} evaluations, {time.time() - start:.1f} s")

xi = whitened(coordinates.bijection.to_constrained(np.asarray(state.ensemble["u"])))
nuts = np.load("out/nuts_observed_inferred.npz")
xi_nuts = nuts["xi"].reshape(-1, xi.shape[1])
print("mean |mean error| / sd over parameters:", round(float(np.mean(np.abs(xi.mean(0) - xi_nuts.mean(0)) / xi_nuts.std(0))), 3))
print("mean |log sd ratio|:", round(float(np.mean(np.abs(np.log(xi.std(0) / xi_nuts.std(0))))), 3))
for k in classes:
    print(f"s2_{k}: EKI-Gibbs 5/50/95% {np.round(np.percentile(scales[k], [5, 50, 95]), 3)}")
np.savez(f"out/eki_gibbs_J{J}_s{SEED}.npz", xi=xi, **{f"s2_{k}": scales[k] for k in classes})
