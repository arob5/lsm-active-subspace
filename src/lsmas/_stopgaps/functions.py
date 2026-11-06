r"""Reading whether a ProbPipe ``Function`` was declared traceable.

A ``Function`` is built with ``traceable=True`` or ``False``, and ProbPipe uses the claim to choose
how to run it, but the claim cannot be read back publicly. Code that wraps one function in another,
such as a reduced problem's forward map around the full problem's, needs it to make the same claim.
ProbPipe would ideally expose the declared claims as a public property; :func:`declares_traceable`
has that shape, so its calls can be replaced by it, after which this module can be deleted.
"""

from __future__ import annotations

from probpipe import Function

__all__ = ["declares_traceable"]


# STOPGAP(PG-18): replace calls with ProbPipe's public reading of a Function's claims.
def declares_traceable(function: Function) -> bool:
    """Whether *function* was declared one that JAX can trace (``traceable=True``)."""
    return bool(function._execution_claims.traceable)
