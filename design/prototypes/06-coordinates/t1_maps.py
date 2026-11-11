"""Prototype for design 06: the coordinate maps as invertible ProbPipe Functions on records.

Checks, on the toy problem:
  A. a Function subclass claiming SupportsInverse maps a record, a batch of records, a law;
  B. its inverse (our stopgap for ProbPipe's unimplemented ``inverse``) maps back, and the forward
     model takes the result;
  C. the inverse can be traced and differentiated with raw=True (for exact Jacobians);
  D. to_vector / from_vector give the (n, d) arrays the algorithms use.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import (
    EmpiricalDistribution,
    Function,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    SupportsInverse,
    SupportsLogDetJacobian,
    bijector_for,
    is_invertible,
    pushforward,
    quantile,
    sample,
    workflow_run,
)

from lsmas import plots
from lsmas.problems import load_inverse_problem

workflow_run.set_default(seed=0)


class RecordBijection(Function, SupportsInverse, SupportsLogDetJacobian):
    """An invertible map from a record of scalars to a record with the same fields."""

    def __init__(self, forward, backward, log_det, names, *, input_name, label):
        spec = NumericRecordSpec(**{n: NumericArraySpec(()) for n in names})
        super().__init__(
            forward, label=label, input_spec={input_name: spec}, output_spec=spec, traceable=True
        )
        object.__setattr__(self, "_backward", backward)
        object.__setattr__(self, "_log_det", log_det)
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "record_spec", spec)

    def _inverse(self, y):
        return self._backward(y)

    def _log_det_jacobian(self, x):
        return self._log_det(x)


def coordinate_maps(prior, block="theta", n=20_000):
    spec = prior.event_spec.outputs[block]
    names = tuple(spec.fields)
    bij = [bijector_for(spec.at_path(n).support) for n in names]
    vec = lambda rec: jnp.stack([jnp.asarray(rec[n]) for n in names], axis=-1)
    rec = lambda v: {n: v[..., j] for j, n in enumerate(names)}
    to_z = lambda t: jnp.stack([b._inverse(jnp.asarray(t[n])) for b, n in zip(bij, names)], -1)
    to_t = lambda z: {n: b.raw(z[..., j]) for j, (b, n) in enumerate(zip(bij, names))}
    theta = sample(prior[block], n).to_vector()
    z = np.asarray(to_z(rec(theta)))
    m, L = z.mean(0), np.linalg.cholesky(np.cov(z, rowvar=False))
    Linv = np.linalg.inv(L)
    unconstrain = RecordBijection(
        lambda theta: rec(to_z(theta)), lambda z: to_t(vec(z)), None, names,
        input_name="theta", label="unconstrain",
    )
    whiten = RecordBijection(
        lambda theta: rec((to_z(theta) - m) @ Linv.T),
        lambda xi: to_t(m + vec(xi) @ L.T), None, names, input_name="theta", label="whiten",
    )
    return unconstrain, whiten, spec


def inverse(f):
    """STOPGAP stand-in for probpipe.inverse: the map y ↦ f⁻¹(y) as a Function."""
    return Function(f._inverse, label=f"{f.label}⁻¹", input_spec={"y": f.record_spec},
                    output_spec=f.record_spec, traceable=True)


problem = load_inverse_problem("toy", scales="inferred")
prior = problem.prior
unconstrain, whiten, spec = coordinate_maps(prior)
print("A0 is_invertible:", is_invertible(whiten))

medians = quantile(prior, 0.5)
print("A1 record ->", type(whiten(medians["theta"])).__name__)
draws = sample(prior, 500)
t = time.time(); xi = whiten(draws["theta"]); print("A2 batch ->", type(xi).__name__, xi.to_vector().shape, f"{time.time()-t:.2f}s")
print("   mean |xi|:", float(np.abs(np.asarray(xi.to_vector()).mean(0)).max()))
law = EmpiricalDistribution(draws)
xi_law = whiten(law["theta"])
print("A3 law ->", type(xi_law).__name__, xi_law)
pf = pushforward(prior["theta"], whiten)
print("A4 pushforward ->", type(pf).__name__, sample(pf, 3).to_vector().shape)
fig = plots.plot_marginals(xi_law, columns=7); print("A5 plot panels:", len([a for a in fig.axes if a.get_visible()]))

to_theta = inverse(whiten)
back = to_theta(xi)
print("B1 round trip:", np.allclose(back.to_vector(), draws["theta"].to_vector(), atol=1e-8))
pred = problem.forward(back)
print("B2 forward on it:", np.asarray(pred.raw).shape)

f = lambda x: problem.forward.using(raw=True)(to_theta.using(raw=True)(dict(zip(spec.fields, x))))
J = jax.jacfwd(f)(jnp.zeros(len(spec.fields)))
print("C1 jacobian:", J.shape, bool(jnp.all(jnp.isfinite(J))))

batch = NumericRecordBatch.from_vector(spec, np.zeros((4, len(spec.fields))), level_names="member")
print("D1 from_vector:", type(batch).__name__, batch.to_vector().shape)
