"""Baseline: ProbPipe's RWMH on the full 14-parameter toy, at its defaults (4 chains x 1500 steps,
the budget of the reduced sampler), scored against NUTS like the reduced posteriors."""

import time

import numpy as np
from probpipe import workflow_run

from common import problem, whitened

start = time.time()
with workflow_run(seed=3):
    posterior = problem.condition(method="blackjax_rwmh")
print(f"blackjax_rwmh, full toy: {time.time() - start:.0f} s")
xi = whitened(problem.coordinates.array_from_draws(posterior.atoms).reshape(-1, 14))
nuts = np.load("out/nuts_observed_fixed.npz")["xi"].reshape(-1, 14)
print("mean |mean error| / sd:", round(float(np.mean(np.abs(xi.mean(0) - nuts.mean(0)) / nuts.std(0))), 3))
print("mean |log sd ratio|:", round(float(np.mean(np.abs(np.log(xi.std(0) / nuts.std(0))))), 3))
