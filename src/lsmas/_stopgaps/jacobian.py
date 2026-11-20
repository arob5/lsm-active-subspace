r"""The Jacobian of a function of a record, as ProbPipe's planned ``jacobian`` operation.

ProbPipe has no Jacobian operation yet; its plan ``plans/deferred/differentiability.md`` proposes
one. This function has that shape. It takes ``method`` and ``step`` as keywords rather than through
``.using``, since it is not a ProbPipe operation, and it compiles its JAX Jacobian, which ProbPipe's
evaluation rules do not do for a traceable integrand.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp

__all__ = ["jacobian"]


# STOPGAP(PG-08): ProbPipe has no Jacobian operation.
# STOPGAP(PG-22): ProbPipe does not compile a traceable integrand, so the Jacobian is compiled here.
def jacobian(
    f: Callable[[Any], Any],
    x: Any,
    *,
    method: Literal["autodiff", "central_differences"] = "autodiff",
    step: float = 0.2,
) -> jax.Array:
    r"""The Jacobian $Df(x)$ of a function of a record of scalars, at the record *x*.

    $$
    Df(x)_{ij} = \frac{\partial f_i}{\partial x_j}(x), \qquad i = 1, \dots, n,\quad
    j = 1, \dots, d,
    $$

    with $j$ running over the fields of *x* in their order. Two methods:

    - ``"autodiff"``: exact, by JAX's forward-mode differentiation, for an $f$ JAX can trace.
    - ``"central_differences"``: column $j$ is
      $\left(f(x + h e_j) - f(x - h e_j)\right) / (2h)$, with $h$ the *step*, from $2d$
      evaluations of $f$, for an $f$ JAX cannot differentiate. An evaluation that fails gives
      non-finite entries.

    Parameters
    ----------
    f : callable
        Maps a record with the fields of *x* to an array of shape $(n,)$, such as a ProbPipe value
        like the mean of a law.
    x : Record or Mapping
        The point: a record of scalars, such as a value of the whitened coordinates ``z``.
    method : {"autodiff", "central_differences"}
        How to compute the Jacobian.
    step : float
        $h$, for ``"central_differences"``.

    Returns
    -------
    jax.Array
        Shape $(n, d)$.
    """
    names = tuple(x)
    point = jnp.stack([jnp.asarray(x[name]) for name in names])

    def f_of_vector(vector: jax.Array) -> jax.Array:
        return jnp.asarray(f(dict(zip(names, vector, strict=True))))

    if method == "autodiff":
        return jax.jit(jax.jacfwd(f_of_vector))(point)
    if method == "central_differences":
        shifts = step * jnp.concatenate([jnp.eye(len(names)), -jnp.eye(len(names))])
        values = jax.vmap(f_of_vector)(point + shifts)
        return ((values[: len(names)] - values[len(names) :]) / (2 * step)).T
    raise ValueError(f"unknown method {method!r}")
