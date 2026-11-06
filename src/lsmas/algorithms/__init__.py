"""Complete parameter-reduction algorithms, each composed from the tools in :mod:`lsmas.tools`."""

from lsmas.algorithms.active_subspace_mcmc import DEFAULT_SAMPLER_OPTIONS, active_subspace_mcmc
from lsmas.algorithms.result import AlgorithmResult

__all__ = ["DEFAULT_SAMPLER_OPTIONS", "AlgorithmResult", "active_subspace_mcmc"]
