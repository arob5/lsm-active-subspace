import jax
import numpy as np
import pytest
from probpipe import (
    BijectorTransformedDistribution,
    EmpiricalDistribution,
    InverseGamma,
    LogNormal,
    Normal,
    NumericRecordBatch,
    bijector_for,
    is_invertible,
    sample,
    unit_interval,
    workflow_run,
)
from scipy.special import logit

from lsmas.coordinates import Coordinates, inverse

LOCATIONS = np.array([1.0, -1.0, 22.0])  # of log(a), logit(r), t
SCALES = np.array([0.5, 0.4, 2.5])
THETA = np.array([[2.0, 0.3, 21.0], [0.5, 0.9, -3.0]])


@pytest.fixture(scope="module")
def prior():
    theta = (
        LogNormal("a", LOCATIONS[0], SCALES[0])
        & BijectorTransformedDistribution(
            "r", Normal("z", LOCATIONS[1], SCALES[1]), bijector_for(unit_interval)
        )
        & Normal("t", LOCATIONS[2], SCALES[2])
    )
    return theta.rename({name: f"theta/{name}" for name in "art"}) & InverseGamma("s2", 2.0, 1.678)


@pytest.fixture(scope="module")
def coordinates(prior):
    return Coordinates.from_prior(prior, "theta")


def records(coordinates, theta, levels="draw"):
    return NumericRecordBatch.from_vector(coordinates.spec, np.asarray(theta), level_names=levels)


def test_unconstrain_applies_each_support_bijector(coordinates):
    expected = np.column_stack([np.log(THETA[:, 0]), logit(THETA[:, 1]), THETA[:, 2]])
    phi = coordinates.unconstrain(records(coordinates, THETA))
    assert coordinates.names == ("a", "r", "t")
    assert list(phi.raw) == ["a", "r", "t"]
    np.testing.assert_allclose(phi.to_vector(), expected, rtol=1e-12)
    back = inverse(coordinates.unconstrain)(phi)
    np.testing.assert_allclose(back.to_vector(), THETA, rtol=1e-12)


def test_whitening_matches_the_prior_moments(coordinates):
    # The prior is Gaussian in unconstrained coordinates, with independent components.
    np.testing.assert_allclose(coordinates.whitening_mean, LOCATIONS, atol=0.03)
    np.testing.assert_allclose(np.diag(coordinates.whitening_factor), SCALES, rtol=0.02)
    off_diagonal = coordinates.whitening_factor[np.tril_indices(3, k=-1)]
    np.testing.assert_allclose(off_diagonal, 0.0, atol=0.02)


def test_whitened_prior_draws_have_mean_zero_and_covariance_identity(prior, coordinates):
    with workflow_run(seed=2):
        z = np.asarray(coordinates.whiten(sample(prior, 20_000)["theta"]).to_vector())
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=0.03)
    np.testing.assert_allclose(np.cov(z, rowvar=False), np.eye(3), atol=0.04)


def test_whitening_is_invertible_on_batches_of_several_levels(prior, coordinates):
    with workflow_run(seed=3):
        theta = sample(prior["theta"], sample_shape=(4, 5))
    z = coordinates.whiten(theta)
    assert z.to_vector().shape == (4, 5, 3)
    back = inverse(coordinates.whiten)(z)
    np.testing.assert_allclose(back.to_vector(), theta.to_vector(), rtol=1e-10)
    assert is_invertible(coordinates.whiten)
    twice = inverse(inverse(coordinates.whiten))(theta)
    np.testing.assert_allclose(twice.to_vector(), z.to_vector(), rtol=1e-12)


def test_a_distribution_maps_to_a_distribution_of_its_draws_mapped(prior, coordinates):
    with workflow_run(seed=4):
        draws = sample(prior, 50)
    law = coordinates.whiten(EmpiricalDistribution(draws)["theta"])
    assert isinstance(law, EmpiricalDistribution)
    np.testing.assert_allclose(
        np.sort(law.atoms.to_vector(), axis=0),
        np.sort(coordinates.whiten(draws["theta"]).to_vector(), axis=0),
        rtol=1e-12,
    )


def test_log_det_jacobians_match_the_jacobians(coordinates):
    point = {name: value for name, value in zip(coordinates.names, THETA[0], strict=True)}
    for change, kernel in [
        (coordinates.unconstrain, coordinates._unconstrained),
        (coordinates.whiten, coordinates._whitened),
    ]:
        expected = np.linalg.slogdet(np.asarray(jax.jacfwd(kernel)(THETA[0])))[1]
        assert float(change._log_det_jacobian(point)) == pytest.approx(expected, rel=1e-10)


def test_coordinates_do_not_depend_on_where_they_are_built(prior):
    first = Coordinates.from_prior(prior, "theta", n_moment_draws=1000)
    second = Coordinates.from_prior(prior, "theta", n_moment_draws=1000)
    np.testing.assert_array_equal(first.whitening_factor, second.whitening_factor)


def test_unknown_block_is_refused(prior):
    with pytest.raises(ValueError, match="no block"):
        Coordinates.from_prior(prior, "psi")


def test_the_whitening_can_first_be_needed_inside_a_trace(prior):
    fresh = Coordinates.from_prior(prior, "theta", n_moment_draws=1000)
    traced = jax.jit(fresh._from_whitened)(np.zeros(3))
    np.testing.assert_allclose(traced, fresh._from_whitened(np.zeros(3)), rtol=1e-12)
