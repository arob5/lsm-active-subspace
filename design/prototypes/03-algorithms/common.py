"""Shared pieces of the design-03 toy checks: the toy problem in whitened coordinates.

``forward_batch(xi)`` and ``jacobian(xi)`` evaluate G and its Jacobian in whitened coordinates for a
batch of rows; ``information``, ``mean_jacobian_information`` and ``slr_information`` build the
matrices the checks compare; ``captured`` scores a subspace against a reference matrix.
"""

import jax
import jax.numpy as jnp
import numpy as np

from problem import toy_problem
from toy import TABLE, TOY_PARAMETERS, features

problem = toy_problem("observed")
coordinates = problem.coordinates
m = jnp.asarray(coordinates.whitening_mean)
L = jnp.asarray(coordinates.whitening_factor)
families = [TABLE.loc[name, "family"] for name in TOY_PARAMETERS]
weights = jnp.asarray(problem.learning_rate["nee"] / np.asarray(problem.likelihood.noise_variances()))


def theta_of_xi(xi):
    z = m + L @ xi
    values = [
        jnp.exp(z[j]) if f == "log_normal" else jax.nn.sigmoid(z[j]) if f == "logit_normal" else z[j]
        for j, f in enumerate(families)
    ]
    return dict(zip(TOY_PARAMETERS, values))


def forward_xi(xi):
    return features(theta_of_xi(xi))


_jacobian_chunk = jax.jit(jax.vmap(jax.jacfwd(forward_xi)))
_forward_chunk = jax.jit(jax.vmap(forward_xi))
CHUNK = 25


def _chunked(function, xi):
    xi = np.asarray(xi)
    return np.concatenate([np.asarray(function(jnp.asarray(xi[i : i + CHUNK]))) for i in range(0, len(xi), CHUNK)])


def jacobian(xi):
    return _chunked(_jacobian_chunk, xi)


def forward_batch(xi):
    return _chunked(_forward_chunk, xi)


def information(J):  # J: (N, n, d) -> E[J^T W J]
    return np.einsum("kni,n,knj->ij", J, np.asarray(weights), J) / J.shape[0]


def mean_jacobian_information(J):
    Jbar = J.mean(0)
    return Jbar.T @ (np.asarray(weights)[:, None] * Jbar)


def slr(xi, g):
    """Least-squares slope of g on xi (EnsKit's statistical linearization, unweighted ensemble)."""
    xc, gc = xi - xi.mean(0), g - g.mean(0)
    return np.linalg.lstsq(xc, gc, rcond=None)[0].T  # (n, d)


def slr_information(xi, g):
    A = slr(xi, g)
    return A.T @ (np.asarray(weights)[:, None] * A)


def whitened(theta):
    return coordinates.whiten(coordinates.bijection.to_unconstrained(theta))




def top_eigenvectors(matrix, rank):
    values, vectors = np.linalg.eigh(matrix)
    return vectors[:, ::-1][:, :rank]


def captured(matrix, reference, rank):
    """Share of the reference's best rank-r information that the estimate's top r directions hold."""
    W = top_eigenvectors(matrix, rank)
    best = np.sort(np.linalg.eigvalsh(reference))[::-1][:rank].sum()
    return float(np.trace(W.T @ reference @ W) / best)
