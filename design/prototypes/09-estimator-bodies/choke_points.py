"""Design 09: checks each choke point between the ideal estimator bodies and ProbPipe at the pin.

Run with ``uv run python design/prototypes/09-estimator-bodies/choke_points.py``. Each check
prints one line.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import (
    EmpiricalDistribution,
    cov,
    expectation,
    function,
    pushforward,
    sample,
    workflow_run,
)

from lsmas.coordinates import inverse
from lsmas.problems import load_inverse_problem
from lsmas.tools import exact_subspace

toy = load_inverse_problem("toy", scales="inferred")
names = toy.coordinates.names
whiten = toy.coordinates.whiten
to_theta = inverse(whiten)


def report(check, outcome):
    print(f"{check:<62} {outcome}")


def outcome_of(call):
    try:
        call()
        return "works"
    except Exception as error:
        return f"{type(error).__name__}: {str(error)[:90]}"


workflow_run.set_default(seed=0)
draws = sample(toy.prior, 300)
weights = np.random.default_rng(0).uniform(size=300)
weighted = EmpiricalDistribution(draws, weights=weights)

# 1. Field order: the integrand's draw["theta"] is passed straight to whiten.
report(
    "1. whiten(draw['theta']) inside expectation",
    outcome_of(
        lambda: expectation.using(num_samples=5)(toy.prior, lambda draw: whiten(draw["theta"]))
    ),
)
report(
    "   the same over draws (EmpiricalDistribution)",
    outcome_of(
        lambda: expectation(EmpiricalDistribution(draws[:5]), lambda draw: whiten(draw["theta"]))
    ),
)
report("   whiten(draws['theta']) on a batch, outside", outcome_of(lambda: whiten(draws["theta"])))
for route, law in [("prior", toy.prior), ("draws", EmpiricalDistribution(draws[:2]))]:
    seen = {}

    def note_fields(draw, seen=seen):
        seen.setdefault("fields", list(draw["theta"]))
        return 0.0

    expectation.using(num_samples=2)(law, note_fields)
    report(
        f"   fields the integrand receives, {route}",
        "sorted" if seen["fields"] == sorted(names) else "declared",
    )

# 2. PG-10: an expectation over more than 256 draws.
z_law = pushforward(weighted, lambda draw: whiten({n: draw["theta"][n] for n in names}))
report("2. pushforward over 300 weighted draws, default", f"{np.asarray(z_law.weights).size} atoms")
z_law = pushforward.using(max_evaluations=300)(
    weighted, lambda draw: whiten({n: draw["theta"][n] for n in names})
)
report("   with max_evaluations=300", f"{np.asarray(z_law.weights).size} atoms")
z = np.asarray(whiten(draws["theta"]).to_vector())
w = weights / weights.sum()
mean_z = w @ z
weighted_cov = ((z - mean_z) * w[:, None]).T @ (z - mean_z)
report(
    "   cov(...) is the weighted covariance (no correction)",
    np.allclose(np.asarray(cov(z_law).raw), weighted_cov),
)

# 3. Compilation: the integrand is vectorized, not compiled.
forward = toy.forward.using(raw=True)
to_theta_raw = to_theta.using(raw=True)


def jacobian_in_z(z_vector):
    return jax.jacfwd(lambda zv: forward(to_theta_raw(dict(zip(names, zv, strict=True)))))(z_vector)


def information_with(jacobian_function):
    def information_at(draw):
        z = whiten({n: draw["theta"][n] for n in names})
        jacobian = jacobian_function(jnp.stack([z[n] for n in names]))
        gamma = toy.likelihood.noise_variances(draw["s2"]) / toy.learning_rate["nee"]
        return jacobian.T @ (jacobian / gamma[:, None])

    return information_at


equal = EmpiricalDistribution(draws[:200])
for label, jacobian_function in [
    ("uncompiled", jacobian_in_z),
    ("compiled", jax.jit(jacobian_in_z)),
]:
    start = time.time()
    information = np.asarray(expectation(equal, information_with(jacobian_function)).raw)
    report(f"3. expectation over 200 draws, Jacobian {label}", f"{time.time() - start:.1f} s")
start = time.time()
current = exact_subspace(toy, draws[:200]).information
report(
    "   current exact_subspace",
    f"{time.time() - start:.1f} s; same H: {np.allclose(information, current, rtol=1e-6)}",
)

# 4. Our own: InverseProblem.noise_variance converts to numpy, so it cannot be traced.
report(
    "4. toy.noise_variance(draw['s2']) inside expectation",
    outcome_of(
        lambda: expectation.using(num_samples=5)(
            toy.prior, lambda draw: toy.noise_variance(draw["s2"])
        )
    ),
)
report(
    "   the same over draws (EmpiricalDistribution)",
    outcome_of(
        lambda: expectation(
            EmpiricalDistribution(draws[:5]), lambda draw: toy.noise_variance(draw["s2"])
        )
    ),
)


# 5. Composition: a Function whose body calls Functions.
@function(traceable=True)
def whitened_forward(z):
    return toy.forward(to_theta(z))


report(
    "5. Function composing forward and inverse(whiten)",
    outcome_of(lambda: whitened_forward(whiten(draws["theta"][:3]))),
)
report(
    "   jax.jacfwd through it",
    outcome_of(
        lambda: jax.jacfwd(
            lambda zv: whitened_forward.using(raw=True)(dict(zip(names, zv, strict=True)))
        )(jnp.zeros(14))
    ),
)

# 6. Finite differences through SIPNET, which JAX cannot trace, inside an expectation.
small = load_inverse_problem("harvard_forest_small")
small_names = small.coordinates.names
small_whiten = small.coordinates.whiten
small_to_theta = inverse(small_whiten)
step = 0.2
shifts = step * np.concatenate([np.eye(len(small_names)), -np.eye(len(small_names))])


def central_difference_jacobian(z_record):
    z_vector = jnp.stack([jnp.asarray(z_record[n]) for n in small_names])
    shifted = [dict(zip(small_names, z_vector + shift, strict=True)) for shift in shifts]
    predictions = jnp.stack([small.forward(small_to_theta(point)).raw for point in shifted])
    half = len(small_names)
    return ((predictions[:half] - predictions[half:]) / (2 * step)).T


def small_information_at(draw):
    jacobian = central_difference_jacobian(small_whiten(draw["theta"]))
    return jacobian.T @ (jacobian / jnp.asarray(small.noise_variance())[:, None])


start = time.time()
report(
    "6. finite differences through SIPNET in an expectation (2 draws)",
    outcome_of(lambda: expectation.using(num_samples=2)(small.prior, small_information_at))
    + f", {time.time() - start:.0f} s",
)

# 7. num_samples=None leaves the default (enumerate held draws).
report(
    "7. expectation.using(num_samples=None) over draws",
    outcome_of(
        lambda: expectation.using(num_samples=None)(
            EmpiricalDistribution(draws[:5]), lambda draw: 0.0
        )
    ),
)

# 8. The linearization body on ProbPipe values.
predictions = toy.forward(draws["theta"][:50])
report(
    "8. forward(draws['theta']) returns",
    f"{type(predictions).__name__} {np.asarray(predictions.raw).shape}",
)
