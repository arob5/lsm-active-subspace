"""Importance of each calibration parameter averaged over the prior, and inertness across the prior.

Each calibration parameter is a transform of a standard normal z_j (the prior is independent), so a
derivative in z_j is a derivative in prior-whitened coordinates. At N prior draws, central
differences of width 2h in each z_j give the Jacobian J (400 x 43); with W = diag(eta / v),

    C = (1/N) sum_n J_n^T W J_n,

the prior average of the tempered likelihood's information in whitened coordinates. Its diagonal is
each parameter's activity score; its eigenvalues are the spectrum an active-subspace method
estimates. h = 0.2 so that the differences span at least one day of leaf_off_day (prior sd 7 d) and
stay well above SIPNET's output rounding.

At the same draws, every parameter fixed because it is inert is moved far from its value, to check
that it stays inert across the prior.

Usage: python t7_prior_importance.py [N] [seed]. Writes out/prior_importance.npz.
"""

import sys
import time

import numpy as np

from prior_table import FIXED, ROWS, to_sipnet
from reference import ETA, OUT, features_batch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
H = 0.2
#: Every parameter fixed because it is inert, with two far-apart values to try at each draw.
REGIME_INERT = {
    "leaf_on_reallocation_fraction": (0.05, 0.8),
    "leaf_on_day": (100.0, 200.0),
    "leaf_on_soil_temperature": (5.0, 20.0),
    "growth_respiration_fraction": (0.0, 0.6),
    "litter_carbon": (500.0, 5000.0),
    "litter_breakdown_rate": (0.05, 2.0),
    "litter_respired_fraction": (0.1, 0.9),
    "leaf_water_pool_depth": (0.05, 1.0),
}
v = np.where(ETA == 0.126, 8.0, 1.5)
weights = ETA / v
d = len(ROWS)
z_draws = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 21).standard_normal((N, d))


def theta_at(z):
    return {row.name: float(row.from_standard(z[j])) for j, row in enumerate(ROWS)}


C = np.zeros((d, d))
inert_violations = {name: 0 for name in REGIME_INERT}
start = time.time()
for n, z in enumerate(z_draws):
    points = [to_sipnet(theta_at(z))]
    for j in range(d):
        for sign in (1, -1):
            moved = z.copy()
            moved[j] += sign * H
            points.append(to_sipnet(theta_at(moved)))
    for name, values in REGIME_INERT.items():
        points.extend(points[0] | {name: value} for value in values)
    runs = features_batch(points)
    base, plus, minus = runs[0], runs[1 : 2 * d + 1 : 2], runs[2 : 2 * d + 2 : 2]
    J = ((plus - minus) / (2 * H)).T
    C += J.T @ (weights[:, None] * J) / N
    for k, name in enumerate(REGIME_INERT):
        checks = runs[2 * d + 1 + 2 * k : 2 * d + 3 + 2 * k]
        inert_violations[name] += int(not all(np.array_equal(check, base) for check in checks))
    print(f"draw {n + 1}/{N}, {time.time() - start:.0f} s", flush=True)

np.savez(OUT / "prior_importance.npz", C=C, names=np.array([row.name for row in ROWS]))
eigenvalues = np.linalg.eigvalsh(C)[::-1]
print("eigenvalues of C:", np.array2string(eigenvalues, precision=3, max_line_width=100))
scores = np.diag(C)
order = np.argsort(-scores)
print("activity (diag C), largest first:")
for j in order:
    print(f"  {ROWS[j].name:40s} {scores[j]:9.3g}")
print("regime-inert parameters that moved the features, out of", N, "draws:", inert_violations)
