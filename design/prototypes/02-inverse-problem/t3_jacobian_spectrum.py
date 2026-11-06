"""Look for null directions beyond the known three: the spectrum of the weighted Jacobian.

The Jacobian of the 400 features is taken by central differences in log coordinates for every
positive parameter, fractions included, and in the value itself for a temperature or a zero-valued
parameter, over the parameters t1 found live, with laiInit dropped (it is fixed at 0, a leafless
1 January). Log coordinates matter: SIPNET writes NEE to 0.001 g C m-2 per step, so differences of
separate runs carry rounding noise, but an exact multiplicative degeneracy f(a k, c k) = f(a, c)
makes the two columns exact negatives of each other, run for run, so it shows at round-off. Rows are weighted by sqrt(eta_i) / sigma_i, so the singular values are in units of the
tempered likelihood's information. An exact null direction shows as a singular value at the
finite-difference noise level, far below the rest.

Writes out/jacobian.npz (J, names, transforms).
"""

import csv

import numpy as np

from reference import ETA, OUT, SIGMA, features_batch, parameter_table, reference_point

STEP = 1e-2
base = reference_point()
domains = {row["flat_name"]: row["domain"] for row in parameter_table()}
live = [row["parameter"] for row in csv.DictReader(open(OUT / "active_set.csv")) if row["inert"] == "False"]
live = [name for name in live if name != "leaf_area_index"]


def transform(name):
    value = base[name]
    if domains[name] != "real" and value > 0:
        return "log"
    return "identity"


def moved(name, delta):
    value, kind = base[name], transform(name)
    if kind == "log":
        return value * np.exp(delta)
    return value + delta


def on_lower_boundary(name):
    """A zero-valued non-negative parameter: difference forwards from the boundary."""
    return transform(name) == "identity" and base[name] == 0 and domains[name] != "real"


def near_upper_boundary(name):
    """A fraction that a forward step would push past 1: difference backwards."""
    return domains[name] in ("unit_interval", "open_unit_interval") and moved(name, STEP) >= 1


def step_pair(name):
    """The (forward, backward) displacements of a central or one-sided difference of width 2h."""
    if on_lower_boundary(name):
        return 2 * STEP, 0.0
    if near_upper_boundary(name):
        return 0.0, -2 * STEP
    return STEP, -STEP


points = [
    base | {name: moved(name, delta)} if delta else dict(base)
    for name in live
    for delta in step_pair(name)
]
runs = features_batch(points)
plus, minus = runs[0::2], runs[1::2]
jacobian = ((plus - minus) / (2 * STEP)).T  # (400, d)
weighted = (np.sqrt(ETA) / SIGMA)[:, None] * jacobian
np.savez(OUT / "jacobian.npz", J=jacobian, names=np.array(live), transforms=np.array([transform(n) for n in live]))

_, singular, right = np.linalg.svd(weighted, full_matrices=False)
print(f"d = {len(live)}; singular values (largest to smallest):")
print(np.array2string(singular, precision=3, max_line_width=100))
for index in range(1, 9):
    vector = right[-index]
    order = np.argsort(-np.abs(vector))[:4]
    loadings = ", ".join(f"{live[j]} {vector[j]:+.2f} ({transform(live[j])})" for j in order)
    print(f"#{len(live) - index + 1}: sv {singular[-index]:.2e}: {loadings}")
