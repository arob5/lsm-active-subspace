import numpy as np
import pytest

from lsmas.experiments.metrics import (
    captured_information,
    gaussian_wasserstein,
    interval_coverage,
    largest_principal_angle,
    marginal_wasserstein,
    mean_errors,
    sd_errors,
)

RNG = np.random.default_rng(0)


def test_marginal_wasserstein_of_shifted_normals_is_the_shift_over_the_sd():
    target = RNG.normal(0.0, 2.0, size=(200_000, 1))
    approximation = RNG.normal(1.0, 2.0, size=(200_000, 1))
    assert marginal_wasserstein(approximation, target)[0] == pytest.approx(0.5, abs=0.01)


def test_mean_and_sd_errors_are_standardized():
    target = np.array([[0.0], [2.0]])  # mean 1, sd 1
    approximation = np.array([[1.5], [5.5]])  # mean 3.5, sd 2
    assert mean_errors(approximation, target)[0] == pytest.approx(2.5)
    assert sd_errors(approximation, target)[0] == pytest.approx(np.log(2.0))


def test_gaussian_wasserstein_matches_the_closed_form():
    target = RNG.multivariate_normal([0.0, 0.0], np.eye(2), size=400_000)
    approximation = RNG.multivariate_normal([3.0, 4.0], 4 * np.eye(2), size=400_000)
    # |m|^2 + tr(4I + I - 2 * 2I) = 25 + 2
    assert gaussian_wasserstein(approximation, target) == pytest.approx(np.sqrt(27.0), rel=0.01)


def test_coverage_of_the_same_law_is_the_interval_probability():
    draws = RNG.normal(size=(100_000, 3))
    np.testing.assert_allclose(
        interval_coverage(draws, RNG.normal(size=(100_000, 3))), 0.9, atol=0.01
    )


def test_subspace_scores_of_the_top_eigenvectors_and_of_their_complement():
    information = np.diag([9.0, 4.0, 1.0])
    top = np.eye(3)[:, :2]
    assert captured_information(top, information) == pytest.approx(1.0)
    assert largest_principal_angle(top, information) == pytest.approx(0.0, abs=1e-6)
    tilted = np.eye(3)[:, [0, 2]]
    assert captured_information(tilted, information) == pytest.approx(10.0 / 13.0)
    assert largest_principal_angle(tilted, information) == pytest.approx(90.0)
