"""Design 09: the statistical linearization as a property of a joint law, and the field-order bug.

Run with ``uv run python design/prototypes/09-estimator-bodies/linearization_and_order.py``.
"""

import time

import numpy as np
from probpipe import EmpiricalDistribution, cov, pushforward, sample, workflow_run

from lsmas.problems import load_inverse_problem
from lsmas.tools import linearization_subspace


def report(check, outcome):
    print(f"{check:<62} {outcome}")


workflow_run.set_default(seed=0)

# 1. The joint law of (z, G(theta)) over draws, and its covariance blocks.
toy = load_inverse_problem("toy")
draws = sample(toy.prior, 300)
weighted = EmpiricalDistribution(draws, weights=np.random.default_rng(0).uniform(size=300))


def whitened_and_predicted(draw):
    return {
        "z": toy.coordinates.whiten(dict(draw["theta"])),
        "predictions": toy.forward(draw["theta"]),
    }


start = time.time()
pairs = pushforward.using(max_evaluations=300)(EmpiricalDistribution(draws), whitened_and_predicted)
joint_covariance = np.asarray(cov(pairs).raw)
d = toy.coordinates.dimension
slope = joint_covariance[d:, :d] @ np.linalg.pinv(joint_covariance[:d, :d])
report(
    "1. joint law of (z, G) over 300 equal draws",
    f"{time.time() - start:.1f} s, cov shape {joint_covariance.shape}",
)
current = linearization_subspace(toy, EmpiricalDistribution(draws)).information
from_joint = slope.T @ (slope / np.asarray(toy.noise_variance())[:, None])
report(
    "   slope from cov(joint) matches the current estimator",
    np.allclose(from_joint, current, rtol=1e-6),
)

# 2. The joint law runs the forward map once per draw: in threads, or one at a time?
small = load_inverse_problem("harvard_forest_small")
small_draws = sample(small.prior, 8)
start = time.time()
small.forward(small_draws["theta"])
batch_time = time.time() - start
start = time.time()
pushforward(
    EmpiricalDistribution(small_draws),
    lambda draw: {
        "z": small.coordinates.whiten(dict(draw["theta"])),
        "predictions": small.forward(draw["theta"]),
    },
)
report(
    "2. SIPNET at 8 draws: one batch call vs inside a pushforward",
    f"{batch_time:.1f} s vs {time.time() - start:.1f} s",
)
