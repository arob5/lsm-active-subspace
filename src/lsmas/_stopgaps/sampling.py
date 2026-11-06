r"""Draws from a law with a random key of the caller's choosing.

ProbPipe's ``sample`` takes its random key from the enclosing ``workflow_run`` and accepts none of
its own. Some code must draw with a given key: an inference method, which ProbPipe hands the run's
key, and a computation that must neither depend on nor disturb the caller's random stream (opening
a nested ``workflow_run`` does disturb it). ProbPipe would ideally offer a public keyed draw, such
as a ``key`` control on ``sample``. :func:`draw_with_key` has that shape, so each call can be
replaced by the public one, after which this module can be deleted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from probpipe import Distribution

__all__ = ["draw_with_key"]


# STOPGAP(PG-15): replace calls with ProbPipe's public keyed draw once it exists.
def draw_with_key(law: Distribution, key: Any, sample_shape: Sequence[int]) -> Any:
    """Draws of *law* made from the random key *key*, as raw values.

    One key always gives the same draws, and no ``workflow_run`` is read or advanced.

    Parameters
    ----------
    law : Distribution
        A ProbPipe law that can be sampled.
    key : jax.Array
        A JAX random key.
    sample_shape : sequence of int
        The leading shape of the draws.

    Returns
    -------
    Any
        The draws as the law's sampler returns them: an array, or a mapping of arrays for a law
        over a record, each with leading shape *sample_shape*.
    """
    return law._sample(key, tuple(int(size) for size in sample_shape))
