# Design 10: the model and inference

Status: **proposed** (2026-11-20). It came out of design 09's choke point C7. Design 09's estimators
are written on top of it.

## 1. The problem

**The model and inference are blurred.** ProbPipe separates a probability model, which is a law,
from inference, which conditions it on data. `InverseProblem` mixes the two:
- it stores the likelihood and the prior next to the model `likelihood & prior` built from them;
- it computes model quantities with the learning rate $\eta$, an inference setting, built in
  (`noise_variance`, `misfit`).

**The noise model has two shapes.** $\Gamma = \operatorname{diag}(s^2_{k(i)}\,\sigma^2_i/\eta_i)$,
with one noise scale $s^2_k$ per class of observation:

| | The prior | The likelihood's `scales` | A value of the parameters |
|---|---|---|---|
| fixed | $\pi_0(\theta)$ | `{"day": 1.0, "night": 1.0}` | `theta` |
| inferred | $\pi_0(\theta)\,\pi_0(s^2)$ | `{"day": "inferred", "night": "inferred"}` | `theta`, `s2` |

Code that needs $\Gamma$ or the scales first asks which shape it has. That is the purpose of
`INFERRED`, `inferred_scale_classes`, `inferred_scale_draws` and `noise_variances`, and it puts
branches into ten modules. Inside an integrand this fails outright (design 09, C7).

**Words.** In this design, `theta` is a value of the calibration parameters $\theta$, and
`parameters` is a value of everything the prior covers: $\theta$, plus $s^2$ when it is inferred.
Either may come from a sample or from the user.

## 2. What ProbPipe offers

- **A model is a law; inference is conditioning.** `condition_on(model, data)` builds the
  `InferenceTarget` that an inference method normalizes, and the method reads the prior,
  likelihood and observed values through `target.factors()`. The learning rate's ideal home is a
  control of conditioning, `condition_on.using(learning_rate=...)` (PG-03).
- **Parts of a model.** `factor(model, "theta")` is the prior of $\theta$, and
  `marginal(model, outputs)` is the law of some outputs.
- **The law of the observations at given parameters.** `condition_on(model, parameters)`. Its mean
  is $G(\theta)$ and its variance is $s^2_{k(i)}\sigma^2_i$, when the likelihood provides them.
- **Optional inputs of a kernel.** "An optional input holds a constant of the model. In a joint, a
  factor that produces an output of its name meets it, so one kernel serves a model with the
  default and a model with a prior on the input." (`conditional_distribution`.) A subclass of
  `ConditionalDistribution` declares one with `InputSpec(...).with_optional(name)`.

## 3. The design

### 3.1 The model

`model = likelihood & prior`, a ProbPipe law. What the tools need is read from it with ProbPipe
operations:

| Quantity | Expression |
|---|---|
| $G(\theta)$ | `mean(condition_on(model, parameters))` |
| $s^2_{k(i)}\sigma^2_i$, which is $\Gamma$ before $\eta$ | `variance(condition_on(model, parameters))` |
| $DG(z)$ | the Jacobian, in $z$, of the first line at `parameters.replace({"theta": theta(z)})` |
| the coordinates | `Coordinates.from_model(model)` (§3.3) |

None of these needs the forward map as a separate object, or the name of the observed output.

### 3.2 The noise model

`GaussianSimulatorLikelihood` declares the record `s2` as an **optional input** with **default
scales**, in place of the `scales` mapping:

```python
likelihood = GaussianSimulatorLikelihood(
    forward,
    output_name="nee",
    reference_variance=...,
    classes=...,
    default_scales={"day": 1.0, "night": 1.0},
)
```

Its law at given parameters provides its mean $G(\theta)$ and its variance $s^2_{k(i)}\sigma^2_i$.
The variance needs no forward run.

**Fixed or inferred is a property of the prior alone.** `nee_problem(scales="fixed")` builds
$\pi_0(\theta)$, and `scales="inferred"` builds $\pi_0(\theta)\,\pi_0(s^2)$. The likelihood is the
same object in both cases, and a value of the parameters holds `s2` exactly when the prior has it.
Given only $\theta$, the inferred model's law of the observations is the joint law of $s^2$ and
$y$, with $s^2$ integrated over rather than set to its default, so nothing silently falls back to
the defaults.

**Reading the scales.** Code that stores or scores the scales reads them by path, as
`posterior.atoms["s2"]`. Whether a problem infers its scales is whether its prior has `s2`.

**Given up:** some classes fixed and others inferred within one problem. No problem uses it.

### 3.3 The coordinates

`Coordinates.from_model(model)` builds the coordinates of $\theta$ from the prior factor
`factor(model, "theta")`, **once per model**, and caches them. Both parts matter:

- **Once per model.** The whitening moments are Monte Carlo estimates. Building from different
  objects draws different streams: a build from `factor(model, "theta")` and one from today's
  `problem.prior` differ by up to 0.04 in $z$. One build per model keeps every subspace in the
  same $z$.
- **From the prior factor, not from the model.** Drawing from the model draws the observations
  too, which runs the forward map at every one of the 50,000 moment draws. In the prototype this
  ran out of memory.

### 3.4 Inference

`InverseProblem` becomes a named bundle of the inputs to generalized conditioning:

```python
InverseProblem(name, description, model, data, learning_rate)
problem.condition(method="blackjax_rwmh")   # tempered_condition_on(model, data, learning_rate, ...), PG-03
```

It computes nothing that involves $\eta$ outside conditioning. Two properties are kept for
convenience, as model-level derivations:

- `problem.prior`, the marginal of the model over its unobserved outputs,
  `marginal(model, [the outputs not in data])`. It draws without forward runs.
- `problem.coordinates`, which is `Coordinates.from_model(problem.model)`.

**Removed from `InverseProblem`:** the `likelihood` and `prior` fields, `forward`,
`noise_variance` and `misfit`.

**$\eta$ enters explicitly wherever it is used:**
- in conditioning, through `problem.condition`;
- in the estimators, through a `learning_rate` argument (§3.5);
- in the misfit, written out in the notebook and the one plot that use it.

### 3.5 The estimators

They take the model and the learning rate, because they estimate the active subspace of the
generalized posterior, whose information is
$H_\eta(\mu) = \mathbb E_\mu[DG^\top \operatorname{diag}(\eta)\,\Gamma^{-1} DG]$:

```python
gradient_based_subspace(model, mu, *, learning_rate=1.0, num_samples=None, ...)
gradient_based_subspace(problem.model, posterior, learning_rate=problem.learning_rate["nee"])
```

The default $\eta = 1$ gives the ordinary posterior. Design 09 §3 gives the bodies.

### 3.6 Each module

| Module | After |
|---|---|
| `_stopgaps/likelihood.py` | optional `s2` with `default_scales`; the law at given parameters provides its mean and variance. `INFERRED`, `scales`, `inferred_scale_classes`, `inferred_scale_draws` and `noise_variances` go; `NOISE_SCALES` stays as the record's name |
| `problems/inverse_problem.py` | the bundle of §3.4 |
| `problems/nee_likelihood.py` | builds the likelihood once, and the prior with or without `s2` |
| `coordinates.py` | `Coordinates.from_model`, cached per model |
| `tools/estimators.py` | design 09, on `(model, mu, learning_rate)` |
| `tools/reduced.py` | a bundle whose model is the reduced joint; `lift` carries `s2` by path |
| `inference/enskit_eki.py`, `inference/noise_scales.py` | fixed scales from the likelihood's `default_scales` when the prior has no `s2`; the inverse-gamma priors from the prior's `s2` factor, as now |
| `experiments/`, `algorithms/active_subspace_mcmc.py` | `atoms["s2"]`, and whether the prior has `s2` |
| `plots.py`, notebooks | $\Gamma$ and the misfit from the model, with $\eta$ written out |

## 4. Verified

By [`prototypes/10-model-and-inference/`](prototypes/10-model-and-inference/) on the toy problem (and
harvard_forest_small where stated) at the pin. Each script uses the prototype likelihood in
`scaled_noise_likelihood.py`, built as §3.2 describes.

| Check | Result |
|---|---|
| the likelihood's input spec | `theta`, and `s2` optional |
| a sample of `likelihood & prior`, fixed / inferred | `nee`, `theta` / `nee`, `theta`, `s2` |
| pointwise log-density against today's likelihood, both shapes | equal |
| the mean and variance of `condition_on(model, parameters)` against $G(\theta)$ and today's variances, both shapes | equal |
| `variance(likelihood, given=parameters)` against `variance(condition_on(likelihood, parameters))` | equal |
| the inferred model given $\theta$ alone | the joint law of `nee` and `s2` |
| `jax.jacfwd` of `mean(condition_on(model, theta(z)))`, compiled, against today's `jacobians` | equal; 0.3 s to compile, then 6 ms |
| the information at given parameters in `expectation` over 100 prior samples, inferred scales, the Jacobian compiled / not (`model_jacobian_in_expectation.py`) | 3.3 s / 17.6 s |
| $\Gamma$ before $\eta$ inside `expectation` over the prior, both shapes | works |
| the same over held draws | fails: ProbPipe #716 (design 09, C3) |
| conditioning `likelihood & prior` with NUTS and RWMH, both shapes | works; samples hold `s2` only when inferred |
| harvard_forest_small: the variance at given parameters against one SIPNET run | 0.005 s against 0.43 s |
| harvard_forest_small: 16 SIPNET runs as the mean of the model's law, in a `pushforward` | 1.8 s, against 1.3 s as one batch call and 0.35 s per run alone |
| `marginal(model, unobserved outputs)` against `problem.prior` | the same law; draws without forward runs |
| `Coordinates.from_prior(factor(model, "theta"))`, built twice | identical; up to 0.04 from a build on today's `problem.prior` |
| `Coordinates.from_prior(model)` | runs the forward map at every moment draw; ran out of memory |

## 5. Choke points

No new ProbPipe gap. As every call on a value inside an expectation over held draws does, reading
the model there fails on ProbPipe #716, which design 09 covers.

## 6. Decisions

Agreed 2026-11-20: the model and inference are separated as above; the estimators take
`(model, mu, learning_rate)`; `noise_variance` is removed, with $\eta$ written out where it is used.

Still open:

1. **`problem.prior` and `problem.coordinates`.** Keep them as model-level conveniences, or write
   `marginal(...)` and `Coordinates.from_model(...)` at each call site?
2. **Mixed fixed and inferred classes.** Drop support, since nothing uses it?
