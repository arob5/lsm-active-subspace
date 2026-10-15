import numpy as np
import pytest
from probpipe import (
    BijectorTransformedDistribution,
    InverseGamma,
    LogNormal,
    Normal,
    bijector_for,
    function,
    unit_interval,
)
from scipy.special import logit

from lsmas._stopgaps.coordinates import Coordinates

LOCATIONS = np.array([1.0, -1.0, 22.0])  # of log(a), logit(r), t
SCALES = np.array([0.5, 0.4, 2.5])


@pytest.fixture(scope="module")
def prior():
    theta = (
        LogNormal("a", LOCATIONS[0], SCALES[0])
        * BijectorTransformedDistribution(
            "r", Normal("z", LOCATIONS[1], SCALES[1]), bijector_for(unit_interval)
        )
        * Normal("t", LOCATIONS[2], SCALES[2])
    )
    return theta.with_path_names({name: f"theta/{name}" for name in "art"}) * InverseGamma(
        "s2", 2.0, 1.678
    )


@pytest.fixture(scope="module")
def coordinates(prior):
    return Coordinates.from_prior(prior, "theta")


def test_unconstrained_coordinates_apply_each_support_bijector(coordinates):
    theta = np.array([[2.0, 0.3, 21.0], [0.5, 0.9, -3.0]])
    expected = np.column_stack([np.log(theta[:, 0]), logit(theta[:, 1]), theta[:, 2]])
    assert coordinates.names == ("a", "r", "t")
    np.testing.assert_allclose(coordinates.bijection.to_unconstrained(theta), expected, rtol=1e-12)
    np.testing.assert_allclose(coordinates.bijection.to_constrained(expected), theta, rtol=1e-12)


def test_whitening_matches_the_prior_moments(coordinates):
    # The prior is Gaussian in unconstrained coordinates, with independent components.
    np.testing.assert_allclose(coordinates.whitening_mean, LOCATIONS, atol=0.03)
    np.testing.assert_allclose(np.diag(coordinates.whitening_factor), SCALES, rtol=0.02)
    off_diagonal = coordinates.whitening_factor[np.tril_indices(3, k=-1)]
    np.testing.assert_allclose(off_diagonal, 0.0, atol=0.02)


def test_whitening_is_invertible_and_works_on_any_leading_shape(coordinates):
    z = np.random.default_rng(0).normal(size=(4, 5, 3))
    np.testing.assert_allclose(coordinates.unwhiten(coordinates.whiten(z)), z, atol=1e-12)


def test_coordinates_do_not_depend_on_where_they_are_built(prior):
    first = Coordinates.from_prior(prior, "theta", n_moment_draws=1000)
    second = Coordinates.from_prior(prior, "theta", n_moment_draws=1000)
    np.testing.assert_array_equal(first.whitening_factor, second.whitening_factor)


def test_draws_and_batches_round_trip(prior, coordinates):
    from probpipe import sample, workflow_run

    with workflow_run(seed=1):
        draws = sample(prior, sample_shape=(6,))
    theta = coordinates.array_from_draws(draws)
    assert theta.shape == (6, 3)

    @function(dispatch="sequential")
    def total(theta):
        return theta["a"] + theta["r"] + theta["t"]

    totals = np.asarray(total(coordinates.bijection.record_batch(theta)).raw())
    np.testing.assert_allclose(totals, theta.sum(axis=1))


def test_unknown_block_is_refused(prior):
    with pytest.raises(ValueError, match="no block"):
        Coordinates.from_prior(prior, "phi")
