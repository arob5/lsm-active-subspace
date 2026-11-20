# Design 09: estimator bodies that read like the mathematics

Status: **proposed** (2026-11-20), from review of `lsmas/tools/estimators.py` after design 08.

## 1. The problem

Each public estimator in `estimators.py` reads as a chain of private helpers:

```python
draws, weights = _draws_to_average(distribution, num_samples)
z = _whitened_parameters(problem, draws)
noise_variance = _noise_variance_at_draws(problem, draws, noise_scales)
jacobians_at_draws = _whitened_jacobians(problem, z, batch_size)
```

The helpers exist because the estimators average over $\mu$ by hand. They take draws and weights
out of three kinds of input, turn the draws into arrays, build each draw's $\Gamma$ outside the
average, and batch the Jacobians themselves. ProbPipe already has the operation that does this:
$H(\mu) = \mathbb E_\mu[DG(z)^\top \Gamma^{-1} DG(z)]$ is `expectation(mu, f)`. It averages over
an empirical law's draws with their weights, draws `num_samples` samples from any other law, and
hands the integrand each draw whole, so each draw's own noise scales `s2` set its $\Gamma$.

This design rewrites the estimators so that their bodies read like the mathematics, names what
ProbPipe cannot yet do (§4), and proposes for each gap a stopgap, a ProbPipe change, or another
route.

## 2. What ProbPipe offers

Scanned at the pin (`9158a50e`):

| The estimators need | ProbPipe | Used today |
|---|---|---|
| draws of $\mu$ | `sample(mu, n)` | yes |
| $\mathbb E_\mu[f]$, weighted for held draws | `expectation(mu, f)`, `.using(num_samples=n)` | **no** |
| the law of $f(\theta)$ for $\theta \sim \mu$, draw by draw | `pushforward(mu, f)` | **no** |
| a law's covariance | `cov(mu)`, weighted for held draws | **no** (`np.cov`) |
| $\theta(z)$ | `inverse(whiten)` | yes (our stopgap of it, PG-05) |
| $G(\theta(z))$ as one map | a `Function` whose body calls Functions (`@function`) | **no** |
| $DG(z)$ | none: no Jacobian operation (PG-08) | `jax.jacfwd` by hand |
| compiling a traceable integrand | none: the evaluation rules vectorize it but do not compile it | `jax.jit` by hand |

## 3. The ideal code

Each body below is what the function would be if ProbPipe had no gaps. §4 says which lines need
a stopgap today.

```python
def gradient_based_subspace(problem, distribution, *, num_samples=None, rank=None, ...):
    def information_at(draw):
        z = problem.coordinates.whiten(draw["theta"])
        return information_matrix(jacobian(problem.whitened_forward, z), problem.noise_variance(draw))

    H = expectation.using(num_samples=num_samples)(distribution, information_at)
    return Subspace.from_information(H, parameter_names=problem.coordinates.names,
                                     estimator="gradient", rank=rank, ...)


def finite_difference_subspace(problem, distribution, *, num_samples=None, step=0.2, ...):
    def information_at(draw):
        z = problem.coordinates.whiten(draw["theta"])
        jacobian_at_z = jacobian.using(method="central_differences", step=step)(
            problem.whitened_forward, z
        )
        return information_matrix(jacobian_at_z, problem.noise_variance(draw))

    H = expectation.using(num_samples=num_samples)(distribution, information_at)
    return Subspace.from_information(H, ..., estimator="finite_differences")


def linearization_subspace(problem, distribution, *, num_samples=None, ...):
    def linearization_inputs(draw):
        return {
            "z": problem.coordinates.whiten(draw["theta"]),
            "predictions": problem.forward(draw["theta"]),
            "noise_precision": 1 / problem.noise_variance(draw),
        }

    joint = pushforward.using(num_samples=num_samples)(distribution, linearization_inputs)
    slope = statistical_linearization(joint, inputs="z", outputs="predictions")
    H = information_matrix(slope, 1 / mean(joint)["noise_precision"])
    return Subspace.from_information(H, ..., estimator="statistical_linearization")


def covariance_subspace(problem, distribution, *, num_samples=None, ...):
    z = pushforward.using(num_samples=num_samples)(
        distribution, lambda draw: problem.coordinates.whiten(draw["theta"])
    )
    H = np.linalg.inv(cov(z)) - np.eye(problem.coordinates.dimension)
    return Subspace.from_information(H, ..., estimator="covariance")
```

`eki_linearization_subspace` keeps its shape: it reads arrays that EnsKit stored, and sums
`increment * information_matrix(slope, noise_variance)` over the steps, each step's slope being
`statistical_linearization` of that step's particles and predictions.

**The building blocks** become:

- `jacobian(f, x)`: $Df(x)$ of a map at a record, by JAX or by central differences (a stopgap;
  §4, C1).
- `statistical_linearization(joint, inputs, outputs)`: the least-squares slope
  $\operatorname{Cov}(G, z)\operatorname{Cov}(z)^{-1}$ of one field of a law on another. It takes
  the joint law of the pairs, so a pair can never be split, and weighted draws are handled by the
  weighted covariances, which removes the old equal-weights restriction. Verified: it matches the
  current estimator on 300 prior draws of the toy.
- `information_matrix(jacobian, noise_variance)`: $J^\top \Gamma^{-1} J$ for one Jacobian. Weights
  and averaging move into `expectation`.
- `InverseProblem.whitened_forward`: $z \mapsto G(\theta(z))$, a `Function` whose body calls
  `forward` and `inverse(coordinates.whiten)`. `ReducedProblem` uses it too.

## 4. Choke points

Each was checked at the pin by
[`prototypes/09-estimator-bodies/`](prototypes/09-estimator-bodies/): `choke_points.py` and
`linearization_and_order.py`.

**C1. No Jacobian operation (PG-08).** There is no `jacobian` in ProbPipe; `jax.jacfwd` through a
traceable `Function` works.
*Recommendation:* a stopgap `jacobian(f, x, *, method="autodiff", step=None)` in
`lsmas/_stopgaps/jacobian.py`, shaped like the operation in ProbPipe's
`plans/deferred/differentiability.md`. Not being a ProbPipe operation, it takes `method` and `step`
as keywords rather than through `.using`. It takes a `Function` and a record, returns
$Df(x)$ with columns in the record's declared order, and has two methods:
- `"autodiff"`: `jax.jacfwd`, for a map JAX can trace;
- `"central_differences"`: $2d$ evaluations at $x \pm h e_j$ in one batch, for one it cannot.

It raises when an evaluation fails (agreed). Upstream: a ProbPipe PR implementing that plan.

**C2. A traceable integrand is vectorized but not compiled.** Over 200 draws of the toy, the
expectation took 33–42 s with an uncompiled Jacobian and 3.8–4.9 s with a compiled one. Today's
code takes 1.8 s.
*Recommendation:* the C1 stopgap compiles its autodiff Jacobian once per map, so nothing shows in
the bodies. Upstream: a ProbPipe PR in which the evaluation rules compile a traceable integrand.

**C3. Over held draws, the integrand receives a record whose fields are in sorted order, which
ProbPipe's own binding then refuses (new PG-21).** Over the prior, `whiten(draw["theta"])` inside
an expectation works. Over an `EmpiricalDistribution` it fails ("fields [...] are not in the
declared order"), because the enumeration route presents the fields alphabetically. This is a
ProbPipe bug.
*Recommendation:* a ProbPipe PR (bug fix). Until then, the stopgap of C4 presents each draw with its
fields in the declared order.

**C4. `pushforward` resamples more than 256 held draws silently, and ignores `num_samples`
(PG-10, #585).** `pushforward` over 300 weighted draws gave 256 equally weighted ones; the
covariance was wrong, with no warning. `.using(max_evaluations=300)` gave all 300, and then `cov`
matched the weighted covariance exactly. The two operations count draws with different controls:
`expectation` uses `num_samples` (an estimate from 4 draws of a normal had sd 0.55, from 4000 had
0.02) and ignores `max_evaluations`, while `pushforward` uses `max_evaluations` and ignores
`num_samples` without a warning (`controls.py`). `expectation` itself is not affected over held
draws: it takes the empirical law's closed form, which is the exact weighted mean of all 1000 draws
in the check.
*Recommendation:* one stopgap module, `lsmas/_stopgaps/expectation.py`, holding `expectation` and
`pushforward` that behave as ProbPipe's should:
- `pushforward` enumerates every held draw, and reads `num_samples` as `expectation` does;
- both present each draw's fields in their declared order (C3).

The bodies import the two from there, so they read exactly as in §3. When ProbPipe fixes #585 and
C3, the import changes and the module goes. Upstream: #585, plus one name for the draw count.

**C5. `cov` returns one flat matrix, with no blocks by path.** The linearization needs the blocks
$\operatorname{Cov}(G, z)$ and $\operatorname{Cov}(z)$ of the joint law. `cov` lays out the
event's leaves "in canonical order", with no way to ask for one block or a cross-covariance by
path.
*Recommendation:* a ProbPipe PR adding covariance by path (`cov(d, "predictions", "z")`). Until
then, `statistical_linearization` reads the two blocks from the joint's layout, in that one place.
*Alternative:* EnsKit's regression on the pushforward's atoms. It keeps pairs together, but
ignores weights, so the equal-weights restriction stays.

**C6. No tolerant failure mode (PG-07).** An expectation cannot leave out draws at which the
forward map fails. *Decided:* finite differences raise. *Proposed:* the linearization raises too,
for the same reason. A failure at a prior draw of SIPNET is then reported, not hidden in a smaller
average.

**C7. Ours, not ProbPipe's: `InverseProblem.noise_variance` converts to numpy,** so it cannot be
called inside a vectorized expectation (`TracerArrayConversionError` over held draws).
*Fix:* it returns a JAX array and takes a draw, `problem.noise_variance(draw)`. That is $\Gamma$ at
the draw's noise scales `draw["s2"]` when they are inferred, and the fixed $\Gamma$ otherwise, so
the bodies need no branch. Called with no argument, it still gives the fixed $\Gamma$.

## 5. Interface changes (agreed 2026-11-20)

- `exact_subspace` → **`gradient_based_subspace`**, `estimator="gradient"`: the Jacobians are
  exact, but the average over $\mu$ is an estimate, so "exact" misdescribed it.
- $\mu$ is a `Distribution`. Draws in hand are passed as `EmpiricalDistribution(draws)`; a bare
  batch, and draws of `theta` alone, are no longer accepted.
- Removed: `noise_scales=` (each draw's own `s2` sets its $\Gamma$; no caller passed it) and
  `batch_size=` (the stopgap vectorizes over all draws).
- Finite differences raise when the forward map fails (C6).
- The covariance estimate uses ProbPipe's weighted covariance, without the small-sample
  correction of `np.cov`.

## 6. Removed

`_draws_to_average`, `_parameter_draws`, `_whitened_parameters`, `_noise_variance_at_draws`,
`_drawn_noise_scales`, `_average_noise_variance`, `_predictions_at_whitened`, `_whitened_jacobians`
and `_whitened_jacobian_function`. `_step_weights` stays inside `eki_linearization_subspace`'s
weighting. What remains private is the two stopgap modules, each tagged and registered.

## 7. Decisions needed

1. **C3 and C4:** one stopgap module that behaves as ProbPipe's `expectation` and `pushforward`
   should, imported in place of ProbPipe's?
2. **C5:** read the covariance blocks in `statistical_linearization` (recommended), or EnsKit's
   regression with the equal-weights restriction?
3. **C6:** the linearization raises on failure, like finite differences?
4. **C7:** `noise_variance(draw)` in place of `noise_variance(noise_scales)`?
5. **Upstream:** for you to decide whether to file or fix in ProbPipe:
   - the Jacobian operation (C1, PG-08);
   - compiling traceable integrands (C2, new PG-22);
   - field order over held draws (C3, new PG-21, a bug);
   - `pushforward` enumerating every held draw and reading `num_samples` (C4, PG-10, #585);
   - covariance by path (C5, new PG-23).
