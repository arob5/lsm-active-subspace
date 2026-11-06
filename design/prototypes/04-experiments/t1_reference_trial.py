"""Trial of a reference posterior for a SIPNET problem: parallel random-walk chains, one per process.

ProbPipe's blackjax_rwmh runs untraceable chains one after another (PG-07) and starts them all at
one point (PG-17), so each chain here is its own process and its own `condition` call, started at
a distinct particle of a preliminary EKI run, with a random-walk proposal shaped by that run's
ensemble covariance in unconstrained coordinates (scaled by 2.38 / sqrt(d)).

Measures: seconds per step with every process busy, acceptance, R-hat and effective sample size
per parameter, and from them the steps a converged reference needs.

Usage: python t1_reference_trial.py PROBLEM N_CHAINS WARMUP DRAWS [scales]
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def one_chain(arguments):
    problem_name, scales, init, sigma, warmup, draws, seed = arguments
    import lsmas  # noqa: F401
    from probpipe import workflow_run

    from lsmas.problems import load

    problem = load(problem_name, scales=scales, max_workers=1)
    start = time.time()
    with workflow_run(seed=seed):
        posterior = problem.condition(
            method="blackjax_rwmh",
            method_options={
                "num_chains": 1,
                "num_warmup": warmup,
                "num_results": draws,
                "init": np.asarray(init),
                "proposal_cov": np.asarray(sigma),
            },
        )
    theta = problem.coordinates.array_from_draws(posterior.atoms).reshape(draws, -1)
    accepted = np.asarray(posterior.annotations["arviz"]["sample_stats"]["is_accepted"]).mean()
    return theta, float(accepted), time.time() - start


def main():
    problem_name, n_chains, warmup, draws = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    scales = sys.argv[5] if len(sys.argv) > 5 else "fixed"
    import lsmas  # noqa: F401
    from probpipe import workflow_run

    from lsmas.problems import load

    problem = load(problem_name, scales=scales)
    coordinates = problem.coordinates
    start = time.time()
    with workflow_run(seed=0):
        eki = problem.condition(method="enskit_eki", method_options={"n_particles": 100})
    eki_theta = coordinates.array_from_draws(eki.atoms).reshape(-1, coordinates.dimension)
    print(f"EKI: {time.time() - start:.0f} s")
    z = coordinates.bijection.to_unconstrained(eki_theta)
    d = z.shape[1]
    sigma = np.linalg.cholesky(np.cov(z.T)) * 2.38 / np.sqrt(d)
    starts = eki_theta[np.random.default_rng(1).choice(len(eki_theta), n_chains, replace=False)]
    jobs = [(problem_name, scales, starts[c], sigma, warmup, draws, 100 + c) for c in range(n_chains)]
    start = time.time()
    with ProcessPoolExecutor(n_chains) as pool:
        results = list(pool.map(one_chain, jobs))
    wall = time.time() - start
    chains = np.stack([r[0] for r in results])  # (chains, draws, d)
    xi = coordinates.whitened_from_parameters(chains)
    import arviz as az

    dataset = az.from_dict({"posterior": {"xi": xi}})
    summary = az.summary(dataset, kind="diagnostics") if hasattr(az, "summary") else None
    ess = np.asarray(az.ess(dataset)["xi"]).ravel()
    rhat = np.asarray(az.rhat(dataset)["xi"]).ravel()
    steps = n_chains * (warmup + draws)
    report = {
        "problem": problem_name,
        "chains": n_chains,
        "warmup": warmup,
        "draws": draws,
        "wall_seconds": round(wall, 1),
        "seconds_per_step_per_chain": round(wall / (warmup + draws), 3),
        "acceptance": [round(r[1], 3) for r in results],
        "ess_min": round(float(ess.min()), 1),
        "ess_median": round(float(np.median(ess)), 1),
        "rhat_max": round(float(rhat.max()), 3),
        "draws_per_effective_draw_worst": round(n_chains * draws / float(ess.min()), 1),
    }
    print(json.dumps(report, indent=1))
    print("ESS per parameter:", dict(zip(coordinates.names, np.round(ess, 0))))
    print("R-hat per parameter:", dict(zip(coordinates.names, np.round(rhat, 3))))
    np.savez(OUT / f"reference_trial_{problem_name}_{scales}.npz", theta=chains)
    (OUT / f"reference_trial_{problem_name}_{scales}.json").write_text(json.dumps(report, indent=1))
    del summary, steps


if __name__ == "__main__":
    main()
