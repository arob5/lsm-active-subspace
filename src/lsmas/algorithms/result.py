r"""What every parameter-reduction algorithm returns."""

from __future__ import annotations

from dataclasses import dataclass

from probpipe import Distribution

from lsmas.tools.subspace import Subspace

__all__ = ["AlgorithmResult"]


@dataclass(frozen=True, eq=False)
class AlgorithmResult:
    """The result of a parameter-reduction algorithm, in a form every algorithm shares.

    Attributes
    ----------
    posterior : Distribution
        The approximation of the full posterior: draws of the parameter record ``theta`` and of
        any inferred noise scales.
    subspace : Subspace
        The subspace the algorithm estimated and used.
    forward_runs : dict of str to int
        How many times each stage ran the forward map, by stage name.
    exploration : Distribution
        The posterior of the exploration stage (for EKI, with its step history in
        ``annotations["arviz/eki_history"]``).
    reduced_posterior : Distribution
        The posterior of the active variables, before lifting.
    """

    posterior: Distribution
    subspace: Subspace
    forward_runs: dict[str, int]
    exploration: Distribution
    reduced_posterior: Distribution

    @property
    def total_forward_runs(self) -> int:
        """The forward runs of every stage together."""
        return sum(self.forward_runs.values())
