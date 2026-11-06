"""Reduced posteriors on the toy, against NUTS on the full posterior.

Whitened coordinates split as xi = W a + W_perp b, with a, b independent N(0, I) under the prior.
Every approximation keeps b at its prior and differs in the law of a:

- optimal:  a ~ the exact posterior's a-marginal, pi(a) E_b[L(W a + W_perp b)]. For a given W this
            is the approximation closest to the posterior in KL (Zahm et al. 2022). Here it is read
            off the NUTS draws as a = W^T xi.
- marginal: the same target, sampled by pseudo-marginal random-walk Metropolis: each proposal draws
            one fresh b from the prior, so L(W a + W_perp b) is an unbiased estimate of the a-marginal
            likelihood (the cost of one run per step, like the plug-in).
- plug-in:  a ~ pi(a) L(W a), b fixed at its prior mean 0 inside the likelihood (Constantine et al.).

Scores, in whitened coordinates against NUTS: the mean over parameters of |mean error| / posterior
sd, and of |log(sd ratio)|.

Usage: python t4_reduced.py [estimators_file]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from common import forward_xi, problem, top_eigenvectors

estimators = np.load(sys.argv[1] if len(sys.argv) > 1 else "out/estimators_J100_s1.npz")
H_post = np.load("out/H_post.npy")
nuts = np.load("out/nuts_observed_fixed.npz")["xi"].reshape(-1, 14)
d = nuts.shape[1]
y = jnp.asarray(problem.data["nee"])
weights = jnp.asarray(problem.learning_rate["nee"] / np.asarray(problem.likelihood.noise_variances()))


def log_likelihood(xi):
    residual = y - forward_xi(xi)
    return -0.5 * jnp.sum(weights * residual**2)


def complement(W):
    q, _ = np.linalg.qr(np.hstack([W, np.eye(d)]))
    return q[:, W.shape[1] :]


def rwmh(W, plug_in, proposal_cov, key, chains=8, steps=6000, burn=1000):
    """Random-walk Metropolis in a, vectorized over chains; pseudo-marginal unless plug_in."""
    r = W.shape[1]
    Wp = jnp.asarray(complement(W))
    W = jnp.asarray(W)
    chol = jnp.linalg.cholesky(jnp.asarray(proposal_cov) * 2.38**2 / r)

    def ll(a, b):
        return jax.vmap(lambda ai, bi: log_likelihood(W @ ai + Wp @ bi))(a, b)

    def new_b(key):
        return jnp.zeros((chains, d - r)) if plug_in else jax.random.normal(key, (chains, d - r))

    def step(state, key):
        a, b, current = state
        k1, k2, k3 = jax.random.split(key, 3)
        proposal = a + jax.random.normal(k1, a.shape) @ chol.T
        b_new = new_b(k2)
        proposed = ll(proposal, b_new)
        log_ratio = proposed - current - 0.5 * jnp.sum(proposal**2 - a**2, axis=1)
        accept = jnp.log(jax.random.uniform(k3, (chains,))) < log_ratio
        a = jnp.where(accept[:, None], proposal, a)
        b = jnp.where(accept[:, None], b_new, b)
        current = jnp.where(accept, proposed, current)
        return (a, b, current), (a, accept)

    k0, k1, key = jax.random.split(key, 3)
    a0 = jax.random.normal(k0, (chains, r)) * 0.1
    b0 = new_b(k1)
    state = (a0, b0, ll(a0, b0))
    _, (path, accepted) = jax.lax.scan(jax.jit(step), state, jax.random.split(key, steps))
    return np.asarray(path[burn:]).reshape(-1, r), float(accepted.mean())


def lift(a, W, key):
    b = np.asarray(jax.random.normal(key, (len(a), d - W.shape[1])))
    return a @ W.T + b @ complement(W).T


def scores(xi):
    mean_error = np.mean(np.abs(xi.mean(0) - nuts.mean(0)) / nuts.std(0))
    sd_error = np.mean(np.abs(np.log(xi.std(0) / nuts.std(0))))
    return mean_error, sd_error


print("prior as the approximation:", np.round(scores(np.random.default_rng(0).standard_normal((20000, d))), 3))
print("EKI final ensemble:", np.round(scores(estimators["eki_final_xi"]), 3))
key = jax.random.key(0)
for label, matrix in (("oracle W (H_post)", H_post), ("shipped W (SLR ladder)", estimators["Hslr_ladder"])):
    print(f"\n{label}")
    print(f"{'r':>3s} {'optimal':>16s} {'marginal (PM)':>16s} {'accept':>7s} {'plug-in':>16s} {'accept':>7s} {'Var_b loglik':>13s}")
    for r in (1, 2, 3, 4, 6, 8):
        W = top_eigenvectors(matrix, r)
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        optimal = lift(nuts @ W, W, k1)
        proposal_cov = np.cov((estimators["eki_final_xi"] @ W).T).reshape(r, r)
        start = time.time()
        a_pm, accept_pm = rwmh(W, False, proposal_cov, k2)
        a_pi, accept_pi = rwmh(W, True, proposal_cov, k3)
        a_mean = jnp.asarray((nuts @ W).mean(0))
        b = jax.random.normal(k4, (200, d - r))
        values = jax.vmap(lambda bi: log_likelihood(jnp.asarray(W) @ a_mean + jnp.asarray(complement(W)) @ bi))(b)
        cells = [scores(optimal), scores(lift(a_pm, W, k1)), scores(lift(a_pi, W, k1))]
        text = [f"{m:6.3f} / {s:6.3f}" for m, s in cells]
        print(f"{r:3d} {text[0]:>16s} {text[1]:>16s} {accept_pm:7.2f} {text[2]:>16s} {accept_pi:7.2f} {float(values.var()):13.2f}"
              f"   ({time.time() - start:.0f} s)")
