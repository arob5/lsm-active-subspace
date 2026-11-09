"""Candidate metrics on the toy: approximations against the NUTS reference, and the noise floor.

Every metric compares draws of an approximation q with draws of the reference p, in whitened
coordinates xi (and, for noise scales, log s^2):

- mean error      |E_q xi_j - E_p xi_j| / sd_p(xi_j), averaged and maximal over j
- sd error        |log(sd_q(xi_j) / sd_p(xi_j))|, averaged and maximal over j
- marginal W1     the 1-Wasserstein distance between the marginals of xi_j, over sd_p(xi_j)
- coverage        the share of p's draws inside q's central 90% interval, per parameter (0.90 ideal)
- Gaussian W2     the 2-Wasserstein distance between Gaussian fits of q and p (joint structure)

The noise floor is each metric for two independent halves of the reference's chains.

Usage: python t2_metrics.py
"""

import time
from pathlib import Path

import jax
import numpy as np
from probpipe import workflow_run
from scipy import linalg, stats

from lsmas.algorithms import active_subspace_mcmc
from lsmas.problems import load

NUTS = Path(__file__).resolve().parents[1] / "03-algorithms" / "out" / "nuts_observed_fixed.npz"


def metrics(q, p):
    sd = p.std(0)
    mean_error = np.abs(q.mean(0) - p.mean(0)) / sd
    sd_error = np.abs(np.log(q.std(0) / sd))
    w1 = np.array([stats.wasserstein_distance(q[:, j], p[:, j]) for j in range(p.shape[1])]) / sd
    low, high = np.quantile(q, [0.05, 0.95], axis=0)
    coverage = np.mean((p >= low) & (p <= high), axis=0)
    return {
        "mean_error": (mean_error.mean(), mean_error.max()),
        "sd_error": (sd_error.mean(), sd_error.max()),
        "marginal_w1": (w1.mean(), w1.max()),
        "coverage90": (coverage.mean(), coverage.min()),
        "gaussian_w2": gaussian_w2(q, p),
    }


def gaussian_w2(q, p):
    mq, mp = q.mean(0), p.mean(0)
    cq, cp = np.cov(q.T), np.cov(p.T)
    root = linalg.sqrtm(cp)
    cross = linalg.sqrtm(root @ cq @ root).real
    return float(np.sqrt(np.sum((mq - mp) ** 2) + np.trace(cq + cp - 2 * cross)))


def show(label, values):
    cells = "  ".join(
        f"{name} {v[0]:.3f}/{v[1]:.3f}" if isinstance(v, tuple) else f"{name} {v:.3f}"
        for name, v in values.items()
    )
    print(f"{label:34s} {cells}")


toy = load("toy")
coordinates = toy.coordinates
reference = np.load(NUTS)["xi"]  # (4, 1000, 14)
p = reference.reshape(-1, 14)
show("reference halves (noise floor)", metrics(reference[:2].reshape(-1, 14), reference[2:].reshape(-1, 14)))
show("prior", metrics(np.random.default_rng(0).standard_normal((4000, 14)), p))

with workflow_run(seed=1):
    eki = toy.condition(method="enskit_eki", method_options={"n_particles": 100})
show("EKI J=100", metrics(coordinates.whitened_from_parameters(coordinates.array_from_draws(eki.atoms).reshape(-1, 14)), p))

with workflow_run(seed=3):
    rwmh = toy.condition(method="blackjax_rwmh")
show("full RWMH, defaults", metrics(coordinates.whitened_from_parameters(coordinates.array_from_draws(rwmh.atoms).reshape(-1, 14)), p))

for seed in (1, 2, 3):
    start = time.time()
    result = active_subspace_mcmc(toy, seed=seed)
    q = coordinates.whitened_from_parameters(coordinates.array_from_draws(result.posterior.atoms))
    show(f"active_subspace_mcmc seed {seed} r={result.subspace.rank}", metrics(q, p))
del jax
