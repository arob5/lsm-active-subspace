"""Shared fixtures: a small linear-Gaussian model whose answers are known in closed form."""

import jax.numpy as jnp
import numpy as np
import pytest
from probpipe import Normal, NumericArraySpec, NumericRecordSpec, function

import lsmas  # noqa: F401  (sets JAX's precision before any test builds an array)
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood

#: The forward map G(θ) = H θ of the linear model, for θ = (a, b) and six observations.
FORWARD_MATRIX = np.array(
    [[1.0, 0.5], [0.2, 1.0], [1.5, -0.3], [0.7, 0.7], [-1.0, 0.4], [0.3, 2.0]]
)
CLASSES = np.array(["day"] * 3 + ["night"] * 3)
REFERENCE_VARIANCE = np.array([0.5, 0.8, 0.6, 0.3, 0.4, 0.2])
PRIOR_MEAN = np.array([0.5, -1.0])
PRIOR_SD = np.array([1.0, 2.0])
THETA_SPEC = NumericRecordSpec(a=NumericArraySpec(()), b=NumericArraySpec(()))


@function(dispatch="sequential", label="linear_forward")
def linear_forward(theta):
    return jnp.asarray(FORWARD_MATRIX @ jnp.stack([theta["a"], theta["b"]]))


def linear_likelihood(scales):
    return GaussianSimulatorLikelihood(
        linear_forward,
        component="y",
        theta_spec=THETA_SPEC,
        reference_variance=REFERENCE_VARIANCE,
        classes=CLASSES,
        scales=scales,
    )


def linear_theta_prior():
    theta = Normal("a", PRIOR_MEAN[0], PRIOR_SD[0]) * Normal("b", PRIOR_MEAN[1], PRIOR_SD[1])
    return theta.with_path_names({"a": "theta/a", "b": "theta/b"})


@pytest.fixture
def observations():
    return np.array([1.2, -2.0, 1.9, -0.4, -1.3, -2.2])
