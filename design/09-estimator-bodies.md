# Design 09: estimator bodies that read like the mathematics

Status: **proposed** (2026-11-20), from review of `lsmas/tools/estimators.py` after design 08. It
is written on top of design 10 (the model and inference).

## 1. The problem

Each public estimator in `estimators.py` reads as a chain of private helpers:

```python
draws, weights = _draws_to_average(distribution, num_samples)
z = _whitened_parameters(problem, draws)
noise_variance = _noise_variance_at_draws(problem, draws, noise_scales)
jacobians_at_draws = _whitened_jacobians(problem, z, batch_size)
```

The helpers exist because the estimators average over $\mu$ by hand. They pull samples and weights
out of three kinds of input, turn them into arrays, build each sample's $\Gamma$ outside the
average, and batch the Jacobians themselves. ProbPipe already has the operation that does this:
$H(\mu) = \mathbb E_\mu[DG(z)^\top \Gamma^{-1} DG(z)]$ is `expectation(mu, f)`. It averages over an
empirical law's samples with their weights, draws `num_samples` samples from any other law, and
passes each value of the parameters whole to the integrand, so its own noise scales `s2` set its
$\Gamma$.

This design rewrites the estimators so that their bodies read like the mathematics, and names what
ProbPipe cannot yet do (§4).

**Words.** As in design 10: `theta` is a value of $\theta$, and `parameters` is a value of
everything the prior covers ($\theta$, plus $s^2$ when it is inferred).

## 2. What ProbPipe offers

Scanned at the pin (`9158a50e`):

| The estimators need | ProbPipe | Used today |
|---|---|---|
| samples of $\mu$ | `sample(mu, n)` | yes |
| $\mathbb E_\mu[f]$, weighted for an empirical law | `expectation(mu, f)`, `.using(num_samples=n)` | **no** |
| the law of $f(\text{parameters})$ under $\mu$ | `pushforward(mu, f)` | **no** |
| a law's covariance | `cov(mu)`, weighted for an empirical law | **no** (`np.cov`) |
| $G(\theta)$ and $\Gamma$ before $\eta$ | the mean and variance of `condition_on(model, parameters)` (design 10) | **no** |
| $\theta(z)$ | `inverse(whiten)` | yes (our stopgap of it, PG-05) |
| $DG(z)$ | none: no Jacobian operation (PG-08) | `jax.jacfwd` by hand |
| compiling a traceable integrand | none: the evaluation rules vectorize it but do not compile it | `jax.jit` by hand |

## 3. The ideal code

Each body below is what the function would be if ProbPipe had no gaps; §4 says which lines need
something else today. The estimators take the model and the learning rate (design 10, §3.5).

```python
def gradient_based_subspace(model, parameter_distribution, *, learning_rate=1.0, num_samples=None, rank=None, ...):
    coordinates = Coordinates.from_model(model)
    to_theta = inverse(coordinates.whiten)

    def information_at(parameters):
        def mean_observations(z):
            return mean(condition_on(model, parameters.replace({"theta": to_theta(z)})))

        jacobian_at_z = jacobian(mean_observations, coordinates.whiten(parameters["theta"]))
        noise_variance = variance(condition_on(model, parameters)) / learning_rate
        return information_matrix(jacobian_at_z, noise_variance)

    H = expectation.using(num_samples=num_samples)(parameter_distribution, information_at)
    return Subspace.from_information(H, parameter_names=coordinates.names,
                                     estimator="gradient", rank=rank, ...)
```

`finite_difference_subspace(model, parameter_distribution, *, learning_rate=1.0, num_samples=None, step=0.2, ...)` is
the same, with `jacobian(..., method="central_differences", step=step)`.

```python
def linearization_subspace(model, parameter_distribution, *, learning_rate=1.0, num_samples=None, ...):
    coordinates = Coordinates.from_model(model)

    def linearization_inputs(parameters):
        observations = condition_on(model, parameters)
        return {
            "z": coordinates.whiten(parameters["theta"]),
            "predictions": mean(observations),
            "noise_precision": learning_rate / variance(observations),
        }

    joint = pushforward.using(num_samples=num_samples)(parameter_distribution, linearization_inputs)
    slope = cov(joint["predictions"], joint["z"]) @ inv(cov(joint["z"]))
    H = information_matrix(slope, 1 / mean(joint["noise_precision"]))
    return Subspace.from_information(H, ..., estimator="statistical_linearization")


def covariance_subspace(model, parameter_distribution, *, num_samples=None, ...):
    coordinates = Coordinates.from_model(model)
    z = pushforward.using(num_samples=num_samples)(
        parameter_distribution, lambda parameters: coordinates.whiten(parameters["theta"])
    )
    H = inv(cov(z)) - eye(coordinates.dimension)
    return Subspace.from_information(H, ..., estimator="covariance")
```

`covariance_subspace` needs no learning rate. The $\eta$ that shaped the posterior is already in
its samples.

`eki_linearization_subspace(model, eki_posterior, *, weighting=...)` keeps its shape. It reads what
EKI stored, whose noise variances already include $\eta$, and sums
`increment * information_matrix(slope, noise_variance)` over the steps, each step's slope fitted to
that step's particles and predictions.

**The building blocks** become:

- `jacobian(f, x)`: $Df(x)$ of a function of a record, by JAX or by central differences. This is a
  stopgap (§4, C1).
- `statistical_linearization(joint, inputs="z", outputs="predictions")`: the line
  $\operatorname{Cov}(G, z)\operatorname{Cov}(z)^{+}$ above, with the pseudo-inverse for an
  ensemble with no more members than parameters. It is a named function because two estimators
  use it: `linearization_subspace`, and the EKI estimate, which fits each step's slope with the
  step's particles and predictions as an empirical law. Taking the joint law of the pairs means a pair can never be split, and weighted
  samples are handled by the weighted covariances, which removes the old equal-weights
  restriction. Verified: it matches the current estimator on 300 prior samples of the toy.
- `information_matrix(jacobian, noise_variance)`: $J^\top \Gamma^{-1} J$ for one Jacobian. Weights
  and averaging move into `expectation`.

## 4. Choke points

Each was checked at the pin by [`prototypes/09-estimator-bodies/`](prototypes/09-estimator-bodies/)
(`choke_points.py`, `linearization_and_order.py`, `controls.py`) and by design 10's prototypes.

**C1. No Jacobian operation (PG-08).** ProbPipe has no `jacobian`; `jax.jacfwd` works through
ProbPipe operations on the model (design 10).
*Proposed:* a stopgap `jacobian(f, x, *, method="autodiff", step=None)` in
`lsmas/_stopgaps/jacobian.py`, shaped like the operation planned in ProbPipe's
`plans/deferred/differentiability.md`. It takes a function of a record and a record, and returns
$Df(x)$ with columns in the record's declared order, by one of two methods:
- `"autodiff"`: `jax.jacfwd`, for a function JAX can trace;
- `"central_differences"`: $2d$ evaluations at $x \pm h e_j$ in one batch, for one it cannot.

Not being a ProbPipe operation, it takes `method` and `step` as keywords rather than through
`.using`. It raises when an evaluation fails (C6). Upstream: that plan.

**C2. A traceable integrand is vectorized but not compiled.** Over 100 prior samples of the toy
with inferred scales, the expectation took 17.6 s with an uncompiled Jacobian and 3.3 s with a
compiled one (design 10). Today's code takes about 1.8 s for 200.
*Proposed:* the C1 stopgap compiles its autodiff Jacobian, so nothing shows in the bodies.
Upstream: the evaluation rules could compile a traceable integrand.

**C3. Over held samples, the integrand receives records with their fields in alphabetical order,
which ProbPipe's own input check then refuses.** This is a ProbPipe bug, filed as ProbPipe #716.
It affects every estimator whenever $\mu$ is an empirical law, such as a posterior.

**C4. `pushforward` resamples more than 256 held samples silently, and ignores `num_samples`.**
Filed as ProbPipe #717, with #585. It affects the linearization and covariance estimators.
`expectation` is exact over held samples.

**C5. `cov` has no cross-covariance.** Filed as ProbPipe #718, as `cov(y, x)` for coupled
marginals of one joint, which is the form §3 uses.

**C6. No tolerant failure mode (PG-07).** *Decided:* finite differences and the linearization
raise when the forward map fails, rather than leaving samples out.

**C7. `InverseProblem.noise_variance` cannot be used in an integrand.** *Resolved by design 10:*
$\Gamma$ before $\eta$ is the variance of `condition_on(model, parameters)`, and the method is
removed.

## 5. Interface changes (agreed 2026-11-20)

- `exact_subspace` → **`gradient_based_subspace`**, `estimator="gradient"`. The Jacobians are
  exact, but the average over $\mu$ is an estimate, so "exact" misdescribed it.
- The estimators take `(model, parameter_distribution, *, learning_rate=...)` (design 10).
- $\mu$ is a `Distribution`. Samples in hand are passed as `EmpiricalDistribution(samples)`; a bare
  batch, and samples of `theta` alone, are no longer accepted.
- Removed: `noise_scales=` (each value's own `s2` sets its $\Gamma$; no caller passed it) and
  `batch_size=` (everything is vectorized).
- Finite differences and the linearization raise when the forward map fails (C6).
- The covariance estimate uses ProbPipe's weighted covariance, without the small-sample correction
  of `np.cov`.

## 6. Removed

`_draws_to_average`, `_parameter_draws`, `_whitened_parameters`, `_noise_variance_at_draws`,
`_drawn_noise_scales`, `_average_noise_variance`, `_predictions_at_whitened`, `_whitened_jacobians`
and `_whitened_jacobian_function`. `_step_weights` stays inside `eki_linearization_subspace`.

The public `jacobians(problem, theta)` goes too; `lsmas.tools` exports the `jacobian` stopgap
instead, which the tools notebook applies to the model's mean. `statistical_linearization` now takes
a joint law rather than two arrays.

**As implemented (2026-11-20).** The stopgap operations take `num_samples` as a keyword, so the
bodies read `expectation(parameter_distribution, information_at, num_samples=num_samples)`. Over a
law held as samples every sample is used, by `expectation` (ProbPipe's closed form) and by the
stopgap `pushforward` alike; `num_samples` applies only to a law that is sampled. This replaces
design 08's resampling of held draws.

## 7. Decisions

Decided 2026-11-20:
- C1 and C2: the `jacobian` stopgap, compiling its autodiff Jacobian;
- C3, C4 and C5 go to ProbPipe (#716, #717, #718). Until they land, interim stopgaps behave as the
  fixed ProbPipe will: `expectation` and `pushforward` present fields in declared order and use
  every held sample, and `cov(y, x)` gives the cross-covariance of two coupled marginals. They come
  out when the pin moves;
- C6: raise;
- C7: design 10;
- the argument $\mu$ is named `parameter_distribution`.
