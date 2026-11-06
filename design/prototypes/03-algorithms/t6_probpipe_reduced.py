"""The plug-in reduced posterior as a ProbPipe problem, sampled by ProbPipe's own RWMH, on the toy.

The reduced problem's parameters are the active variables a = W^T xi, with independent N(0, 1)
priors; its forward map is a -> G(theta(W a)), the inactive variables at their prior mean; its
likelihood, data and learning rate are the full problem's. Conditioning it with blackjax_rwmh is
plain ProbPipe. The approximation of the full posterior lifts each draw: xi = W a + W_perp b with b
from the prior.

Usage: python t6_probpipe_reduced.py [rank] [estimators_file]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import Function, Normal, NumericArraySpec, NumericRecordSpec, workflow_run

from common import forward_xi, problem, top_eigenvectors
from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas.priors import independent_prior
from lsmas.problems.nee_likelihood import _prior_with_noise_scales, _likelihood_scales
from lsmas.problems.inverse_problem import InverseProblem
from toy import DATA, N_FEATURES

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 7
estimators = np.load(sys.argv[2] if len(sys.argv) > 2 else "out/estimators_J100_s1.npz")
SCALES = sys.argv[3] if len(sys.argv) > 3 else "fixed"
W = top_eigenvectors(estimators["Hslr_ladder"], RANK)
names = tuple(f"a{i + 1}" for i in range(RANK))
W_jax = jnp.asarray(W)


def reduced_features(a):
    return forward_xi(W_jax @ jnp.stack([a[name] for name in names]))


forward = Function(
    reduced_features,
    label="reduced_features",
    output_spec=NumericArraySpec((N_FEATURES,), jnp.float64),
    traceable=True,
)
likelihood = GaussianSimulatorLikelihood(
    forward,
    output_name="nee",
    theta_spec=NumericRecordSpec(**{name: NumericArraySpec(()) for name in names}),
    reference_variance=problem.likelihood.reference_variance,
    classes=DATA.feature_classes,
    scales=_likelihood_scales(SCALES),
    label="reduced_likelihood",
)
prior = _prior_with_noise_scales({name: Normal(name, 0.0, 1.0) for name in names}, likelihood.random_scale_slots)
reduced = InverseProblem(
    name="toy_reduced",
    description=f"plug-in reduced toy, r = {RANK}",
    likelihood=likelihood,
    prior=prior,
    data=problem.data,
    learning_rate=problem.learning_rate,
    coordinates=Coordinates.from_prior(prior, "theta"),
)
start = time.time()
with workflow_run(seed=3):
    posterior = reduced.condition(method="blackjax_rwmh")
print(f"blackjax_rwmh on the reduced toy (r = {RANK}): {time.time() - start:.0f} s")
a = reduced.coordinates.array_from_draws(posterior.atoms).reshape(-1, RANK)
print("draws:", a.shape)
d = W.shape[0]
q, _ = np.linalg.qr(np.hstack([W, np.eye(d)]))
b = np.random.default_rng(0).standard_normal((len(a), d - RANK))
xi = a @ W.T + b @ q[:, RANK:].T
nuts = np.load(f"out/nuts_observed_{SCALES}.npz")["xi"].reshape(-1, d)
print("mean |mean error| / sd:", round(float(np.mean(np.abs(xi.mean(0) - nuts.mean(0)) / nuts.std(0))), 3))
print("mean |log sd ratio|:", round(float(np.mean(np.abs(np.log(xi.std(0) / nuts.std(0))))), 3))
if SCALES == "inferred":
    for slot in ("s2_day", "s2_night"):
        print(slot, np.round(np.percentile(np.asarray(posterior.atoms[slot].raw).ravel(), [5, 50, 95]), 3))
