import numpy as np
import pytest
from probpipe import NumericRecordBatch

from conftest import CLASSES, FORWARD_MATRIX, linear_problem
from lsmas.coordinates import inverse


@pytest.fixture
def problem(observations):
    return linear_problem(observations, {"day": 1.0, "night": 1.0}, np.ones(len(CLASSES)))


def test_the_forward_map_takes_parameters_mapped_back_from_whitened_coordinates(problem):
    coordinates = problem.coordinates
    z = np.random.default_rng(0).normal(size=(3, 2))
    z_records = NumericRecordBatch.from_vector(
        coordinates.whiten.output_spec.term_spec, z, level_names="member"
    )
    theta = inverse(coordinates.whiten)(z_records)
    predictions = np.asarray(problem.forward(theta).raw)
    np.testing.assert_allclose(predictions, theta.to_vector() @ FORWARD_MATRIX.T, rtol=1e-12)


def test_describe_names_the_parameters_and_observations(problem):
    assert "Calibration parameters (2): a, b" in problem.describe()


def test_misfit_is_half_the_weighted_squared_residual(problem, observations):
    assert problem.misfit(observations) == pytest.approx(0.0)
    shifted = np.stack([observations + 1.0, observations - 2.0])
    gamma = problem.noise_variance()
    expected = [0.5 * np.sum(1.0 / gamma), 0.5 * np.sum(4.0 / gamma)]
    np.testing.assert_allclose(problem.misfit(shifted), expected)
