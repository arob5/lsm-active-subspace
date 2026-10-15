import numpy as np
import pytest
from probpipe import sample, workflow_run

from lsmas.priors import log_normal_from_interval, logit_normal_from_interval


@pytest.mark.parametrize(
    ("constructor", "lower", "upper"),
    [(log_normal_from_interval, 140.0, 450.0), (logit_normal_from_interval, 0.04, 0.20)],
)
def test_central_interval_holds_the_stated_probability(constructor, lower, upper):
    law = constructor("x", lower, upper, probability=0.9)
    with workflow_run(seed=0):
        draws = np.asarray(sample(law, 200_000).raw())
    np.testing.assert_allclose(np.quantile(draws, [0.05, 0.95]), [lower, upper], rtol=0.01)


def test_interval_must_be_ordered():
    with pytest.raises(ValueError, match="lower < upper"):
        log_normal_from_interval("x", 5.0, 2.0)
