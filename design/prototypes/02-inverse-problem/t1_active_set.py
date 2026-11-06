"""Which of the 58 SIPNET parameters move the 400 NEE features at the reference point?

Each parameter is perturbed both ways: by ±10% for a positive value, ±0.05 (clipped to its domain)
for a fraction, and ±1 for a temperature or a zero-valued parameter. A parameter is *inert* when
both perturbed runs reproduce the reference features bit for bit. For the others the table gives the
largest feature change and the information-weighted size of the central difference,

    s_j = sqrt( sum_i eta_i (dG_i / sigma_i)^2 ),   sigma_i^2 = v_k(i),

the change in the tempered log-likelihood scale per perturbation (a rough "how many noise sds").

Writes out/active_set.csv.
"""

import csv
import time

import numpy as np

from reference import ETA, OUT, SIGMA, features, features_batch, parameter_table, reference_point

base = reference_point()
table = {row["flat_name"]: row for row in parameter_table()}


def perturbations(name, value):
    domain = table[name]["domain"]
    if domain in ("unit_interval", "open_unit_interval"):
        low, high = max(value - 0.05, 0.0 if domain == "unit_interval" else 1e-4), min(value + 0.05, 1.0 if domain == "unit_interval" else 1 - 1e-4)
    elif domain in ("positive", "non_negative") and value > 0:
        low, high = value * 0.9, value * 1.1
    else:  # real, or zero-valued non-negative
        low, high = (value - 1.0 if domain == "real" else 0.0), value + 1.0
    return low, high


names = list(table)
points, labels = [], []
for name in names:
    low, high = perturbations(name, base[name])
    for value in (low, high):
        points.append(base | {name: value})
        labels.append((name, value))

start = time.time()
reference = features(base)
runs = features_batch(points)
print(f"{len(points)} runs in {time.time() - start:.0f} s; reference finite: {np.isfinite(reference).all()}")

rows = []
for index, name in enumerate(names):
    low_run, high_run = runs[2 * index], runs[2 * index + 1]
    failed = not (np.isfinite(low_run).all() and np.isfinite(high_run).all())
    inert = (not failed) and np.array_equal(low_run, reference) and np.array_equal(high_run, reference)
    change = np.nan_to_num(high_run - low_run)
    rows.append({
        "parameter": name,
        "sipnet_name": table[name]["sipnet_name"],
        "value": base[name],
        "low": labels[2 * index][1],
        "high": labels[2 * index + 1][1],
        "failed": failed,
        "inert": inert,
        "max_abs_change": float(np.max(np.abs(change))),
        "weighted_change": float(np.sqrt(np.sum(ETA * (change / SIGMA) ** 2))),
    })

rows.sort(key=lambda row: -row["weighted_change"])
with open(OUT / "active_set.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    flag = "FAILED" if row["failed"] else ("inert" if row["inert"] else "")
    print(f"{row['parameter']:40s} {row['value']:>11.5g}  s={row['weighted_change']:9.4g}  max|dG|={row['max_abs_change']:9.3g}  {flag}")
