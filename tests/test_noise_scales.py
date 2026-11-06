import jax
import numpy as np
import pytest
from probpipe import InverseGamma
from scipy import integrate, stats

from conftest import CLASSES, REFERENCE_VARIANCE, linear_likelihood
from lsmas.inference.noise_scales import NoiseScaleConditionals

OBSERVATIONS = np.array([1.2, -2.0, 1.9, -0.4, -1.3, -2.2])
LEARNING_RATE = np.where(CLASSES == "day", 0.5, 0.8)
PREDICTIONS = np.array([[1.0, -1.5, 2.5, 0.0, -1.0, -2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def conditionals(night_scale):
    """Day scale inferred with an IG(3, 2) prior; night scale fixed at *night_scale*, or inferred
    with an IG(2, 1) prior when it is None."""
    night = "s2_night" if night_scale is None else night_scale
    priors = {
        "s2_day": InverseGamma("s2_day", 3.0, 2.0),
        "s2_night": InverseGamma("s2_night", 2.0, 1.0),
    }
    return NoiseScaleConditionals.from_problem(
        linear_likelihood({"day": "s2_day", "night": night}), priors, OBSERVATIONS, LEARNING_RATE
    )


def test_draws_follow_the_conjugate_inverse_gamma_law():
    model = conditionals(None)
    beta = 0.6
    forms = np.repeat(model.quadratic_forms(PREDICTIONS[:1]), 200_000, axis=0)
    draws = model.draw(forms, beta, jax.random.key(0))
    shape = model.concentrations + beta * model.effective_counts / 2
    rate = model.prior_scales + beta * forms[0] / 2
    for k in range(2):
        expected = stats.invgamma(shape[k], scale=rate[k])
        np.testing.assert_allclose(
            np.quantile(draws[:, k], [0.25, 0.5, 0.75]), expected.ppf([0.25, 0.5, 0.75]), rtol=0.01
        )


def test_fixed_scales_give_the_usual_tempering_weights():
    fixed_scales = np.where(CLASSES == "day", 1.5, 0.7)
    model = NoiseScaleConditionals.from_problem(
        linear_likelihood({"day": 1.5, "night": 0.7}), {}, OBSERVATIONS, LEARNING_RATE
    )
    residuals = OBSERVATIONS - PREDICTIONS
    misfits = 0.5 * np.sum(
        LEARNING_RATE * residuals**2 / (fixed_scales * REFERENCE_VARIANCE), axis=1
    )
    increment = 0.3
    weights = model.log_weights(model.quadratic_forms(PREDICTIONS), 0.2, increment)
    np.testing.assert_allclose(weights, -increment * misfits)


def test_only_scales_given_as_slots_are_inferred():
    assert conditionals(0.7).inferred.tolist() == [True, False]
    assert conditionals(None).inferred.tolist() == [True, True]


def test_weights_are_the_ratio_of_the_scale_marginal_likelihoods():
    model = conditionals(0.7)
    forms = model.quadratic_forms(PREDICTIONS)
    beta, increment = 0.2, 0.3
    day = CLASSES == "day"

    def log_marginal(particle, level):
        """log of the integral over s2_day of its prior times the tempered day likelihood."""

        def integrand(scale):
            prior = stats.invgamma(3.0, scale=2.0).pdf(scale)
            tempered = np.prod(
                stats.norm(
                    PREDICTIONS[particle, day], np.sqrt(scale * REFERENCE_VARIANCE[day])
                ).pdf(OBSERVATIONS[day])
                ** (level * LEARNING_RATE[day])
            )
            return prior * tempered

        night = -level / 2 * forms[particle, 1] / 0.7
        return np.log(integrate.quad(integrand, 0, np.inf)[0]) + night

    expected = [log_marginal(j, beta + increment) - log_marginal(j, beta) for j in range(2)]
    computed = model.log_weights(forms, beta, increment)
    assert computed[0] - computed[1] == pytest.approx(expected[0] - expected[1], rel=1e-6)


def test_largest_increment_keeps_the_effective_sample_size():
    model = conditionals(None)
    forms = model.quadratic_forms(np.random.default_rng(0).normal(size=(50, 6)) * 2)
    increment = model.largest_increment(forms, 0.0, 0.5)
    weights = np.exp(model.log_weights(forms, 0.0, increment))
    ess = weights.sum() ** 2 / (weights**2).sum()
    assert 0 < increment < 1
    assert ess == pytest.approx(25, rel=1e-3)


def test_common_scales_average_the_precisions():
    scales = np.array([[1.0, 2.0], [3.0, 2.0]])
    np.testing.assert_allclose(NoiseScaleConditionals.common_scales(scales), [1.5, 2.0])


def test_noise_variance_is_the_tempered_covariance():
    model = conditionals(0.7)
    expected = np.where(CLASSES == "day", 2.0, 0.7) * REFERENCE_VARIANCE / LEARNING_RATE
    np.testing.assert_allclose(model.noise_variance(np.array([2.0, 0.7])), expected)


def test_increment_never_falls_below_the_floor():
    model = conditionals(None)
    forms = model.quadratic_forms(np.random.default_rng(1).normal(size=(50, 6)) * 1e4)
    assert model.largest_increment(forms, 0.0, 0.5) == pytest.approx(1e-3)
    assert model.largest_increment(forms, 0.9995, 0.5) == pytest.approx(0.0005)
