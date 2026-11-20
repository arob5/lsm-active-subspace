"""Design 10: the Jacobian of the model's conditional mean, inside an expectation, compiled or not.

Run from this directory with ``uv run python model_jacobian_in_expectation.py``.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import condition_on, expectation, factor, mean, variance, workflow_run
from scaled_noise_likelihood import ScaledNoiseLikelihood

from lsmas.coordinates import Coordinates, inverse
from lsmas.problems import load_inverse_problem

workflow_run.set_default(seed=0)
toy = load_inverse_problem("toy", scales="inferred")
likelihood = ScaledNoiseLikelihood(
    toy.likelihood.forward,
    output_name="nee",
    reference_variance=toy.likelihood.reference_variance,
    classes=toy.likelihood.classes,
    default_scales={"day": 1.0, "night": 1.0},
    label="nee_likelihood",
)
model = likelihood & toy.prior
coordinates = Coordinates.from_prior(factor(model, "theta"), "theta")
names = coordinates.names
to_theta = inverse(coordinates.whiten).using(raw=True)
whiten = coordinates.whiten.using(raw=True)


def information_with(compile_jacobian):
    def information_at(parameters):
        def mean_observations(z_vector):
            theta = to_theta(dict(zip(names, z_vector, strict=True)))
            return mean(condition_on(model, parameters.replace({"theta": theta}))).raw

        differentiate = jax.jacfwd(mean_observations)
        if compile_jacobian:
            differentiate = jax.jit(differentiate)
        z = whiten(parameters["theta"])
        jacobian = differentiate(jnp.stack([jnp.asarray(z[name]) for name in names]))
        noise_variance = variance(condition_on(model, parameters)).raw
        return jacobian.T @ (jacobian / noise_variance[:, None])

    return information_at


for compile_jacobian in (False, True):
    start = time.time()
    try:
        information = expectation.using(num_samples=100)(
            toy.prior, information_with(compile_jacobian)
        )
        outcome = f"{time.time() - start:.1f} s, shape {np.asarray(information.raw).shape}"
    except Exception as error:
        outcome = f"{type(error).__name__}: {str(error)[:200]}"
    label = "compiled" if compile_jacobian else "uncompiled"
    print(f"expectation over 100 prior samples, Jacobian {label}: {outcome}")
