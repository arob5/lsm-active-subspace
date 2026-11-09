"""Does a proposal tuned on pilot draws mix better than EKI's? Continue the trial's chains.

Takes the chains of t1_reference_trial.py (harvard_forest_small), estimates the posterior covariance
of the unconstrained parameters from their pooled draws, and continues every chain from its last
state with that proposal (scaled by 2.38 / sqrt(d), no further warmup). Reports draws per effective
draw, to compare with the trial's EKI-shaped proposal.

Usage: python t4_pilot_proposal.py DRAWS
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from t1_reference_trial import OUT, one_chain

DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 400

if __name__ == "__main__":
    import arviz as az

    from lsmas.problems import load

    problem = load("harvard_forest_small")
    coordinates = problem.coordinates
    pilot = np.load(OUT / "reference_trial_harvard_forest_small_fixed.npz")["theta"]
    z = coordinates.bijection.to_unconstrained(pilot.reshape(-1, pilot.shape[-1]))
    sigma = np.linalg.cholesky(np.cov(z.T)) * 2.38 / np.sqrt(z.shape[1])
    jobs = [("harvard_forest_small", "fixed", pilot[c, -1], sigma, 0, DRAWS, 200 + c) for c in range(len(pilot))]
    start = time.time()
    with ProcessPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(one_chain, jobs))
    chains = np.stack([r[0] for r in results])
    xi = coordinates.whitened_from_parameters(chains)
    dataset = az.from_dict({"posterior": {"xi": xi}})
    ess = np.asarray(az.ess(dataset)["xi"]).ravel()
    rhat = np.asarray(az.rhat(dataset)["xi"]).ravel()
    report = {
        "wall_seconds": round(time.time() - start, 1),
        "acceptance": [round(r[1], 3) for r in results],
        "ess_min": round(float(ess.min()), 1),
        "rhat_max": round(float(rhat.max()), 3),
        "draws_per_effective_draw_worst": round(chains.shape[0] * DRAWS / float(ess.min()), 1),
    }
    print(json.dumps(report, indent=1))
    np.savez(OUT / "pilot_continued_small.npz", theta=chains)
