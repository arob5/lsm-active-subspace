"""ProbPipe internals that inference methods need.

ProbPipe lets a package register its own inference method, but the objects such a method works
with are not public yet: the conditioning target, the split of a model into prior and likelihood,
the constructor of the posterior, and the workflow's seed for a run. This module is the one place
that imports them. Once ProbPipe exposes them publicly, import them from there and delete this
module.
"""

from probpipe.inference._approximate_distribution import make_posterior  # STOPGAP(PG-04)
from probpipe.inference._inference_utils import (
    model_factors,  # STOPGAP(PG-04)
    run_seed,  # STOPGAP(PG-04)
)
from probpipe.operations._condition import (
    _UnnormalizedConditional as UnnormalizedConditional,  # STOPGAP(PG-04)
)

__all__ = [
    "UnnormalizedConditional",
    "make_posterior",
    "model_factors",
    "run_seed",
]
