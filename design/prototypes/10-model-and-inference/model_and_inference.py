"""Design 10: what an estimator can read from the model ``likelihood & prior`` alone.

Run from this directory with ``uv run python model_and_inference.py``.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import condition_on, expectation, factor, mean, sample, variance, workflow_run
from scaled_noise_likelihood import ScaledNoiseLikelihood

from lsmas.coordinates import Coordinates, inverse
from lsmas.problems import load_inverse_problem


def report(check, outcome):
    print(f"{check:<70} {outcome}")


def outcome_of(call):
    try:
        return call()
    except Exception as error:
        return f"{type(error).__name__}: {str(error)[:160]}"


workflow_run.set_default(seed=0)
current = {scales: load_inverse_problem("toy", scales=scales) for scales in ("fixed", "inferred")}
old = current["fixed"].likelihood
likelihood = ScaledNoiseLikelihood(
    old.forward,
    output_name="nee",
    reference_variance=old.reference_variance,
    classes=old.classes,
    default_scales={"day": 1.0, "night": 1.0},
    label="nee_likelihood",
)
models = {scales: likelihood & current[scales].prior for scales in current}

# 1. The coordinates from the model: the model's output theta is the prior's.
for scales, model in models.items():
    start = time.time()
    coordinates = Coordinates.from_prior(factor(model, "theta"), "theta")
    theta = sample(current[scales].prior, 3)["theta"]
    z = coordinates.whiten(theta).to_vector()
    elapsed = time.time() - start
    expected = Coordinates.from_prior(current[scales].prior, "theta").whiten(theta).to_vector()
    report(
        f"1. Coordinates.from_prior(factor(model, 'theta')), {scales}",
        f"{elapsed:.2f} s with moments; same z: {np.allclose(z, expected)}",
    )

# 2. The conditional law of the observations at given parameters: its mean is G, its variance Gamma.
for scales, model in models.items():
    parameters = sample(current[scales].prior)
    law = condition_on(model, parameters)
    g = np.asarray(mean(law).raw)
    report(
        f"2. mean(condition_on(model, parameters)) is G(theta), {scales}",
        np.allclose(g, np.asarray(old.forward(parameters["theta"]).raw)),
    )
    report(
        f"   variance(condition_on(model, parameters)) is s2 sigma2, {scales}",
        np.allclose(
            np.asarray(variance(law).raw), np.asarray(likelihood.noise_variance(parameters))
        ),
    )

# 3. Given theta alone, the inferred model's law of the observations integrates over s2.
law = outcome_of(
    lambda: condition_on(models["inferred"], {"theta": sample(current["inferred"].prior)["theta"]})
)
report(
    "3. condition_on(inferred model, theta only)",
    law if isinstance(law, str) else f"{type(law).__name__}, event {law.event_spec}",
)

# 4. The Jacobian of the conditional mean in whitened coordinates, through ProbPipe operations.
model = models["fixed"]
coordinates = Coordinates.from_prior(factor(model, "theta"), "theta")
to_theta = inverse(coordinates.whiten).using(raw=True)
names = coordinates.names


def mean_observation_at(z_vector):
    theta = to_theta(dict(zip(names, z_vector, strict=True)))
    return mean(condition_on(model, {"theta": theta})).raw


report(
    "4. jax.jacfwd of mean(condition_on(model, theta(z)))",
    outcome_of(lambda: jax.jacfwd(mean_observation_at)(jnp.zeros(14)).shape),
)

# 5. Gamma at the parameters inside an expectation over the prior.
for scales, model in models.items():
    report(
        f"5. variance(condition_on(model, parameters)) in an expectation, {scales}",
        outcome_of(
            lambda m=model, s=scales: (
                np.asarray(
                    expectation.using(num_samples=4)(
                        current[s].prior, lambda p: variance(condition_on(m, p))
                    ).raw
                ).shape
            )
        ),
    )

# 6. Taking the likelihood back out of the model.
first, second = factor(models["fixed"], "nee"), factor(models["fixed"], "nee")
report("6. factor(model, 'nee') returns the same object each call", first is second)
report("   its forward map is the likelihood's", first.forward is likelihood.forward)
