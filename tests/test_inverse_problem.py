import numpy as np
import pytest

from conftest import CLASSES, FORWARD_MATRIX, linear_problem

THETA = np.array([[0.3, -1.2], [1.0, 0.5], [-0.4, 2.0]])


@pytest.fixture
def problem(observations):
    return linear_problem(observations, {"day": 1.0, "night": 1.0}, np.ones(len(CLASSES)))


def test_predicted_observations_evaluate_the_forward_map_at_each_row(problem):
    np.testing.assert_allclose(problem.predicted_observations(THETA), THETA @ FORWARD_MATRIX.T)


def test_one_setting_gives_one_prediction(problem):
    prediction = problem.predicted_observations(THETA[0])
    assert prediction.shape == (len(CLASSES),)
    np.testing.assert_allclose(prediction, FORWARD_MATRIX @ THETA[0])


def test_parameters_with_the_wrong_number_of_columns_are_refused(problem):
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        problem.predicted_observations(np.zeros((4, 3)))
