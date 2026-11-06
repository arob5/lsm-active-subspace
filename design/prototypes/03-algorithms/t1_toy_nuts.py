"""Does ProbPipe's NUTS give a trustworthy reference posterior for the toy, and how fast?

Runs blackjax_nuts through problem.condition on the toy (observed or synthetic data), reports time,
per-parameter posterior summaries against the prior, and R-hat across chains. Saves the draws.

Usage: python t1_toy_nuts.py [observed|synthetic] [fixed|inferred]
"""

import sys
import time

import numpy as np
from probpipe import workflow_run

from problem import toy_problem
from toy import MEDIANS, TOY_PARAMETERS

DATA_KIND = sys.argv[1] if len(sys.argv) > 1 else "observed"
SCALES = sys.argv[2] if len(sys.argv) > 2 else "fixed"
problem = toy_problem(DATA_KIND, SCALES)
start = time.time()
with workflow_run(seed=0):
    posterior = problem.condition(method="blackjax_nuts")
print(f"NUTS ({DATA_KIND}, scales {SCALES}): {time.time() - start:.0f} s")
print("annotations:", list(getattr(posterior, "annotations", {}) or {})[:5])
theta = problem.coordinates.array_from_draws(posterior.atoms)  # (chains, draws, d)
print("draws shape", theta.shape)
xi = problem.coordinates.whiten(problem.coordinates.bijection.to_unconstrained(theta))


def rhat(x):  # x: (chains, draws)
    m, n = x.shape
    chain_means = x.mean(1)
    b = n * chain_means.var(ddof=1)
    w = x.var(1, ddof=1).mean()
    return np.sqrt(((n - 1) / n * w + b / n) / w)


print(f"{'parameter':38s} {'median':>10s} {'post median':>12s} {'post sd/prior sd':>17s} {'rhat':>6s}")
for j, name in enumerate(TOY_PARAMETERS):
    sd_ratio = xi[..., j].std()
    print(f"{name:38s} {MEDIANS[name]:10.4g} {np.median(theta[..., j]):12.4g} {sd_ratio:17.3f} {rhat(xi[..., j]):6.3f}")
if SCALES == "inferred":
    for slot in ("s2_day", "s2_night"):
        values = np.asarray(posterior.atoms[slot].raw) if hasattr(posterior.atoms[slot], "raw") else None
        print(slot, None if values is None else np.percentile(values, [5, 50, 95]))
covariance = np.cov(xi.reshape(-1, xi.shape[-1]).T)
print("eigenvalues of the whitened posterior covariance:", np.round(np.linalg.eigvalsh(covariance), 4))
np.savez(f"out/nuts_{DATA_KIND}_{SCALES}.npz", theta=theta, xi=xi)
