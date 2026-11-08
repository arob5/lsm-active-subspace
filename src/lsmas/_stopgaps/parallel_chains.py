r"""Markov chains run side by side, one process per chain, each from its own starting point.

ProbPipe's ``blackjax_rwmh`` runs the chains of a target it cannot trace one after another, and
starts every chain from one state. For a forward map as slow as SIPNET that serializes a long run,
and leaves R-hat unable to detect chains that disagree because of where they started. ProbPipe
would ideally advance the chains together, evaluating each step's proposals as one batch through
the forward map's executor, and accept one starting state per chain. Until then
:func:`run_in_processes` runs one single-chain ``condition`` call per process; once ProbPipe does
both, its callers can make one ``condition`` call with every chain and this module can be deleted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

__all__ = ["run_in_processes"]


# STOPGAP(PG-07, PG-17): replace with one condition call that runs every chain, each from its start.
def run_in_processes(
    run_one_chain: Callable[..., Any], chain_arguments: Sequence[tuple], max_workers: int
) -> list[Any]:
    """``run_one_chain(*arguments)`` for each entry of *chain_arguments*, in separate processes.

    Parameters
    ----------
    run_one_chain : callable
        A function defined at the top level of a module, so that it can be sent to another
        process.
    chain_arguments : sequence of tuple
        The arguments of each call.
    max_workers : int
        How many processes run at once.

    Returns
    -------
    list
        Each call's return value, in the order of *chain_arguments*.
    """
    with ProcessPoolExecutor(max_workers) as pool:
        return list(pool.map(run_one_chain, *zip(*chain_arguments, strict=True)))
