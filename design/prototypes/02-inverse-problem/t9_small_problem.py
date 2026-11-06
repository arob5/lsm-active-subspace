"""Choosing the small problem: the information matrix of candidate subsets, with the rest fixed.

For a subset S of the calibration parameters, the small problem calibrates S and fixes every other
parameter at its prior median. In S's prior-whitened coordinates z_S (each parameter a transform of
a standard normal), with W = diag(eta / v), the prior average of the tempered likelihood's
information is

    C_S = (1/N) sum_n J_n^T W J_n,

with J_n the Jacobian of the 400 features in z_S at the n-th prior draw, by central differences of
width 2h. Two things are checked:

1. Whether a parameter reduction is more than a parameter selection. For r = 1, ..., 4 the share of
   tr C_S left outside the top r eigenvectors is compared with the share left outside the best r
   parameters (the r largest diagonal entries). A subset whose informative directions mix its
   parameters leaves much less outside r eigenvectors than outside r parameters.
2. Whether derivatives are well defined. Each parameter's activity score (diagonal of C_S) at
   h = 0.2 is compared with the score at h = 0.05, at the same draws. A smooth response gives about
   the same score at both widths; a step function (leaf-on and leaf-off timing switch on whole days)
   gives scores that depend on h.

The candidate is the proposed small problem plus the two leaf-timing parameters, so that both checks
run on one set of draws, and the proposed problem's C_S is the block without them.

Usage: python t9_small_problem.py [N] [seed]. Writes out/small_problem.npz.
"""

import sys
import time

import numpy as np

from prior_table import ROWS, to_sipnet
from reference import ETA, OUT, features_batch

PROPOSED = (
    "photosynthetic_capacity",
    "half_saturation_light",
    "light_extinction_coefficient",
    "leaf_on_growth",
    "vapor_pressure_deficit_slope",
    "vapor_pressure_deficit_exponent",
    "soil_water_holding_capacity",
    "water_use_efficiency",
    "respiration_share",
    "soil_respiration_flux_at_10c",
)
STEP = ("leaf_on_growing_degree_days", "leaf_off_day")
SUBSET = PROPOSED + STEP
WIDTHS = (0.2, 0.05)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rows = {row.name: row for row in ROWS}
medians = {row.name: row.median() for row in ROWS}
v = np.where(ETA == 0.126, 8.0, 1.5)
weights = ETA / v
k = len(SUBSET)
z_draws = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 5).standard_normal((N, k))


def sipnet_at(z):
    return to_sipnet(medians | {name: float(rows[name].from_standard(z[j])) for j, name in enumerate(SUBSET)})


def information(h):
    C = np.zeros((k, k))
    start = time.time()
    for n, z in enumerate(z_draws):
        points = []
        for j in range(k):
            for sign in (1, -1):
                moved = z.copy()
                moved[j] += sign * h
                points.append(sipnet_at(moved))
        runs = features_batch(points)
        J = ((runs[0::2] - runs[1::2]) / (2 * h)).T
        C += J.T @ (weights[:, None] * J) / N
        if (n + 1) % 10 == 0:
            print(f"h={h}: draw {n + 1}/{N}, {time.time() - start:.0f} s", flush=True)
    return C


def residuals(C, ranks=(1, 2, 3, 4)):
    trace = C.trace()
    eigenvalues = np.linalg.eigvalsh(C)[::-1]
    diagonal = np.sort(np.diag(C))[::-1]
    return [(r, 1 - eigenvalues[:r].sum() / trace, 1 - diagonal[:r].sum() / trace) for r in ranks]


matrices = {h: information(h) for h in WIDTHS}
np.savez(OUT / "small_problem.npz", names=np.array(SUBSET), **{f"C_h{h}": C for h, C in matrices.items()})

wide, narrow = matrices[WIDTHS[0]], matrices[WIDTHS[1]]
print(f"\nactivity score at h = {WIDTHS[0]} and h = {WIDTHS[1]} (others at prior medians, N = {N}):")
for j, name in enumerate(SUBSET):
    print(f"  {name:36s} {wide[j, j]:8.3g} {narrow[j, j]:8.3g}  ratio {narrow[j, j] / wide[j, j]:5.2f}")

p = len(PROPOSED)
for h, C in matrices.items():
    block = C[:p, :p]
    eigenvalues = np.linalg.eigvalsh(block)[::-1]
    print(f"\nproposed problem, h = {h}: trace {block.trace():.1f}; eigenvalues",
          np.array2string(eigenvalues, precision=3, max_line_width=120))
    print("  eigenvalue / trace", np.array2string(eigenvalues / block.trace(), precision=3))
    for r, outside_eigen, outside_parameters in residuals(block):
        print(f"  r = {r}: outside top-r eigenvectors {outside_eigen:.2f}, outside best r parameters {outside_parameters:.2f}")
    vectors = np.linalg.eigh(block)[1][:, ::-1]
    for i in range(3):
        loadings = ", ".join(f"{PROPOSED[j]} {vectors[j, i]:+.2f}" for j in np.argsort(-abs(vectors[:, i]))[:4])
        print(f"  eigenvector {i + 1}: {loadings}")
