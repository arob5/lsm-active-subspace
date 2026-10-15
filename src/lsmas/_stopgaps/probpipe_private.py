"""ProbPipe internals that inference methods and coordinate maps need.

ProbPipe lets a package register its own inference method, but the objects such a method works
with are not public yet: the conditioning target, the split of a model into prior and likelihood,
the constructor of the posterior, the workflow's seed for a run, and the maps between a law's draws
and unconstrained coordinates. This module is the one place that imports them. Once ProbPipe
exposes them publicly, import them from there and delete this module.
"""

from probpipe.inference._approximate_distribution import make_posterior  # STOPGAP(PG-04)
from probpipe.inference._inference_utils import (
    flat_unflatten,  # STOPGAP(PG-05)
    model_factors,  # STOPGAP(PG-04)
    run_seed,  # STOPGAP(PG-04)
    unconstrained_coordinates,  # STOPGAP(PG-05)
)
from probpipe.operations._condition import (
    _UnnormalizedConditional as UnnormalizedConditional,  # STOPGAP(PG-04)
)

__all__ = [
    "UnnormalizedConditional",
    "flat_unflatten",
    "make_posterior",
    "model_factors",
    "run_seed",
    "unconstrained_coordinates",
]
