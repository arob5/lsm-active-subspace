# Design 08: the subspace estimators take distributions of the parameters

Status: **implemented** (2026-11-19), as agreed in review of the tools notebook (design 07).

## 1. The problem

Before this change, each estimator took a different array:

| Function | Took |
|---|---|
| `jacobians(problem, z)` | whitened points |
| `exact_subspace(problem, z, noise_scales=...)` | whitened points, and scale values |
| `finite_difference_subspace(problem, z)` | whitened points |
| `covariance_subspace(problem, theta)` | parameters |
| `statistical_linearization(z, predictions)` | two arrays the caller had to pair |
| `LadderLinearization(problem.coordinates)` | coordinates |

A user had to draw from the distribution to average over, whiten the draws, turn them into
arrays, keep inputs and outputs paired, and pass noise-scale values by hand. In review, the
notebook's first attempt at the statistical linearization paired the whitened prior draws with
forward runs at *new* prior draws, a mistake this interface invited.

## 2. The design

**The subspace stays in whitened coordinates; the inputs are in the problem's own parameters**, as
ProbPipe values, and the estimators whiten internally, as `condition_on` transforms to
unconstrained coordinates internally.

```python
exact_subspace(problem, mu, *, num_samples=None, noise_scales=None, rank=None, ...)
finite_difference_subspace(problem, mu, *, num_samples=None, step=0.2, noise_scales=None, ...)
linearization_subspace(problem, mu, *, num_samples=None, noise_scales=None, ...)
covariance_subspace(problem, mu, *, num_samples=None, ...)
jacobians(problem, theta)
LadderLinearization(problem, weighting=...)
statistical_linearization(z, predictions)        # unchanged: the bare regression
```

**The distribution $\mu$** is one of:

- a distribution held as draws (`EmpiricalDistribution`), such as an MCMC or EKI posterior or
  `Reference.posterior`: every draw is used, with its weight;
- a batch of draws (`RecordBatch`), such as `sample(problem.prior, 500)`: every draw, equally
  weighted;
- any other distribution, such as `problem.prior`: `num_samples` draws are taken from it, from the
  current `workflow_run`'s stream. Without `num_samples` this is refused, so that the number of
  draws, and of forward runs, is always the caller's choice.

`num_samples` for the first two resamples them. Draws of the whole record (with `theta` and `s2`)
and of the record `theta` alone both work.

**Noise scales** come from the draws when they include the record `s2`, as a posterior's do; each
draw's own scales set its $\Gamma$. Otherwise, or to override them, `noise_scales={"day": ...,
"night": ...}`.

**`num_samples`**, not `n_draws`, after ProbPipe's convention. It is a keyword argument rather
than a `.using` control, since the estimators are plain functions returning a `Subspace`, which
is not a ProbPipe value; that can be revisited when $W$ becomes a ProbPipe linear operator.

**`linearization_subspace`** (new) draws $\mu$, runs the forward model at the same draws, whitens
them, fits $\widehat{DG}$, and returns the subspace of
$\widehat{DG}^\top \overline{\Gamma^{-1}} \widehat{DG}$, with $\overline{\Gamma^{-1}}$ the draws'
average noise precision (the ensemble's common scales, as `LadderLinearization` uses). It needs
equally weighted draws, and leaves out draws at which the forward map fails. The regression must
be on whitened coordinates: the slope estimates $\mathbb E[DG(z)]$ by Stein's lemma only for a
roughly Gaussian ensemble, which $z$ is under the prior, and a slope with respect to $\theta$ does
not give $H$ in $z$.

**`Subspace` methods** (`active_variables`, `inactive_variables`) keep taking whitened arrays: a
subspace is a linear-algebra object in $z$.

## 3. What the estimators read from the problem

The problem stays the first argument. The estimators read at most three things from it, never the
observed data:

| Estimator | `forward` | `coordinates` | $\Gamma$ |
|---|---|---|---|
| `jacobians` | ✓ (JAX-differentiable) | ✓ | |
| `exact_subspace`, `finite_difference_subspace`, `linearization_subspace` | ✓ | ✓ | ✓ |
| `covariance_subspace` | | ✓ | |
| `LadderLinearization` | | ✓ | from the run's step records |

These are the ingredients of $H(\mu) = \mathbb E_\mu[DG(z)^\top \Gamma^{-1} DG(z)]$, so
`(problem, μ)` is the natural signature, and a reduced problem works as well as a full one. A
surrogate forward map would be its own `InverseProblem`, rather than new arguments; nothing needs
one yet.

## 4. Callers

`active_subspace_mcmc` passes the problem to `LadderLinearization`. The reference runs pass their
thinned draws, with any noise scales, as a batch, so the estimators read the scales from them.
The tools notebook uses the new calls, and every docstring says what the function computes, what
it reads from the problem, how it uses $\mu$, and what it costs in forward runs.
