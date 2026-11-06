"""The proposed algorithm end to end on harvard_forest_small: explore, estimate, reduce, sample.

1. EKI (J particles), with a callback accumulating H = sum_t dbeta_t A_t^T Lambda A_t in whitened
   coordinates (A_t the statistical linearization of step t).
2. Rank: the smallest r whose discarded eigenvalues sum to at most 1/2.
3. Compare W with the prior-averaged finite-difference matrix of design 02 (t9_small_problem.py).
4. The plug-in reduced problem (forward a -> G(theta(W a)), SIPNET, not traceable), sampled by
   ProbPipe's blackjax_rwmh with short chains, to time the eager path and check acceptance.

Usage: python t7_sipnet_small.py [n_particles] [rwmh_steps]
"""

import sys
import time

import jax.numpy as jnp
import numpy as np
from probpipe import Function, Normal, NumericArraySpec, NumericRecordSpec, workflow_run

from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas.priors import independent_prior
from lsmas.problems import load
from lsmas.problems.inverse_problem import InverseProblem

J = int(sys.argv[1]) if len(sys.argv) > 1 else 100
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
problem = load("harvard_forest_small")
coordinates = problem.coordinates
weights = problem.learning_rate["nee"] / np.asarray(problem.likelihood.noise_variances())
d = coordinates.dimension


class Accumulator:
    def __init__(self):
        self.H = np.zeros((d, d))
        self.n_runs = 0

    def __call__(self, record):
        xi = coordinates.whiten(coordinates.bijection.to_unconstrained(record.theta))
        g = record.predictions
        ok = np.all(np.isfinite(g), axis=1)
        xc, gc = xi[ok] - xi[ok].mean(0), g[ok] - g[ok].mean(0)
        A = np.linalg.lstsq(xc, gc, rcond=None)[0].T
        self.H += record.increment * A.T @ (weights[:, None] * A)
        self.n_runs += len(g)


accumulator = Accumulator()
start = time.time()
with workflow_run(seed=1):
    eki_posterior = problem.condition(
        method="enskit_eki", method_options={"n_particles": J, "callbacks": [accumulator]}
    )
print(f"EKI: {accumulator.n_runs} SIPNET runs, {time.time() - start:.0f} s")
values, vectors = np.linalg.eigh(accumulator.H)
values, vectors = values[::-1], vectors[:, ::-1]
print("eigenvalues of the ladder matrix:", np.round(values, 2))
tail = np.cumsum(values[::-1])[::-1]
rank = next(r for r in range(1, d + 1) if (tail[r] if r < d else 0.0) <= 0.5)
print("rank (discarded <= 1/2):", rank)
for i in range(3):
    top = np.argsort(-abs(vectors[:, i]))[:3]
    print(f"  direction {i + 1}:", ", ".join(f"{coordinates.names[j]} {vectors[j, i]:+.2f}" for j in top))

prior_C = np.load("../02-inverse-problem/out/small_problem.npz")["C_h0.2"][:d, :d]
best = np.sort(np.linalg.eigvalsh(prior_C))[::-1]
for r in (1, 2, 3, rank):
    W = vectors[:, :r]
    print(f"r={r}: share of the prior finite-difference matrix's best rank-r information captured: "
          f"{np.trace(W.T @ prior_C @ W) / best[:r].sum():.3f}")

W = vectors[:, :rank]
names = tuple(f"a{i + 1}" for i in range(rank))


def reduced_features(a):
    xi = W @ np.array([float(np.asarray(a[name])) for name in names])
    theta = coordinates.bijection.to_constrained(coordinates.unwhiten(xi)[None, :])
    return jnp.asarray(problem.forward(coordinates.bijection.record_batch(theta)).raw)[0]


forward = Function(
    reduced_features, label="reduced_features", output_spec=NumericArraySpec((400,), jnp.float64), traceable=False
)
likelihood = GaussianSimulatorLikelihood(
    forward,
    output_name="nee",
    theta_spec=NumericRecordSpec(**{name: NumericArraySpec(()) for name in names}),
    reference_variance=problem.likelihood.reference_variance,
    classes=problem.likelihood.classes,
    scales={"day": 1.0, "night": 1.0},
    label="reduced_likelihood",
)
prior = independent_prior({name: Normal(name, 0.0, 1.0) for name in names})
reduced = InverseProblem(
    name="small_reduced", description="", likelihood=likelihood, prior=prior, data=problem.data,
    learning_rate=problem.learning_rate, coordinates=Coordinates.from_prior(prior, "theta"),
)
start = time.time()
with workflow_run(seed=2):
    posterior = reduced.condition(
        method="blackjax_rwmh",
        method_options={"num_warmup": STEPS, "num_results": STEPS, "num_chains": 2},
    )
elapsed = time.time() - start
print(f"blackjax_rwmh, 2 chains x {2 * STEPS} steps: {elapsed:.0f} s, {elapsed / (4 * STEPS):.2f} s per step")
stats = posterior.annotations if hasattr(posterior, "annotations") else None
try:
    print("acceptance:", float(np.mean(np.asarray(stats["arviz"]["sample_stats"]["is_accepted"]))))
except Exception as error:  # the layout of the annotations is not essential to this check
    print("acceptance not read:", type(error).__name__)
