"""Shared fixtures: a small linear-Gaussian model whose answers are known in closed form."""

import jax.numpy as jnp
import numpy as np
import pytest
from probpipe import Normal, NumericArraySpec, NumericRecordSpec, cov, function, mean

import lsmas  # noqa: F401  (sets JAX's precision before any test builds an array)
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas._stopgaps.quantiles import median
from lsmas.priors import laws_from_table

#: The forward map G(θ) = H θ of the linear model, for θ = (a, b) and six observations.
FORWARD_MATRIX = np.array(
    [[1.0, 0.5], [0.2, 1.0], [1.5, -0.3], [0.7, 0.7], [-1.0, 0.4], [0.3, 2.0]]
)
CLASSES = np.array(["day"] * 3 + ["night"] * 3)
REFERENCE_VARIANCE = np.array([0.5, 0.8, 0.6, 0.3, 0.4, 0.2])
PRIOR_MEAN = np.array([0.5, -1.0])
PRIOR_SD = np.array([1.0, 2.0])
THETA_SPEC = NumericRecordSpec(a=NumericArraySpec(()), b=NumericArraySpec(()))


@function(label="linear_forward", traceable=True)
def linear_forward(theta):
    return jnp.asarray(FORWARD_MATRIX @ jnp.stack([theta["a"], theta["b"]]))


def linear_likelihood(scales):
    return GaussianSimulatorLikelihood(
        linear_forward,
        output_name="y",
        theta_spec=THETA_SPEC,
        reference_variance=REFERENCE_VARIANCE,
        classes=CLASSES,
        scales=scales,
    )


def linear_theta_prior():
    theta = Normal("a", PRIOR_MEAN[0], PRIOR_SD[0]) & Normal("b", PRIOR_MEAN[1], PRIOR_SD[1])
    return theta.rename({"a": "theta/a", "b": "theta/b"})


def prior_medians(prior_table):
    """Each calibration parameter's prior median, by name."""
    return {name: median(law) for name, law in laws_from_table(prior_table).items()}


def exact_linear_posterior(observations, scales, learning_rate):
    r"""Mean and covariance of π(θ) ∏ N(y_i | (Hθ)_i, s²_k(i) v_i)^{η_i}, for fixed scales s²."""
    variances = np.where(CLASSES == "day", scales["day"], scales["night"]) * REFERENCE_VARIANCE
    precision_weights = learning_rate / variances
    prior_precision = np.diag(1 / PRIOR_SD**2)
    precision = prior_precision + FORWARD_MATRIX.T @ (precision_weights[:, None] * FORWARD_MATRIX)
    covariance = np.linalg.inv(precision)
    posterior_mean = covariance @ (
        prior_precision @ PRIOR_MEAN + FORWARD_MATRIX.T @ (precision_weights * observations)
    )
    return posterior_mean, covariance


def posterior_moments(posterior):
    """The mean and covariance of a posterior's ``theta``, as arrays."""
    posterior_mean = np.array([float(value) for value in mean(posterior["theta"]).raw.values()])
    return posterior_mean, np.asarray(cov(posterior["theta"]).raw)


@pytest.fixture
def observations():
    return np.array([1.2, -2.0, 1.9, -0.4, -1.3, -2.2])
