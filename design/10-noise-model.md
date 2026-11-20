# Design 10: one noise model, with the noise scales as an optional input of the likelihood

Status: **proposed** (2026-11-20). It came out of design 09's choke point C7, and design 09's
estimators read $\Gamma$ as §3 describes.

## 1. The problem

The noise covariance is $\Gamma = \operatorname{diag}(s^2_{k(i)}\,\sigma^2_i / \eta_i)$, with one
noise scale $s^2_k$ per class of observation (day, night). Today the noise model has two shapes:

| | The prior | The likelihood's `scales` | A draw |
|---|---|---|---|
| fixed | $\pi_0(\theta)$ | `{"day": 1.0, "night": 1.0}` | `theta` |
| inferred | $\pi_0(\theta)\,\pi_0(s^2)$ | `{"day": "inferred", "night": "inferred"}` | `theta`, `s2` |

So code that needs $\Gamma$ or the scales first asks which shape it has. That is the purpose of
`INFERRED`, `inferred_scale_classes`, `inferred_scale_draws` and `noise_variances`, and it creates
branches in the estimators, metrics, reference runs, studies, the reduced problem, EKI, the
noise-scale conditionals and `active_subspace_mcmc`. $\Gamma$ itself is computed by
`problem.noise_variance(noise_scales)` from loose scale values, outside the model, as a numpy array.
Inside an integrand this fails twice (design 09, C7): numpy breaks under `jax.vmap`, and a draw holds
`s2` in one shape and not the other, so the integrand would have to branch.

## 2. What ProbPipe offers

- **Optional inputs of a kernel.** `conditional_distribution` documents the idiom: "An optional
  input holds a constant of the model. In a joint, a factor that produces an output of its name
  meets it, so one kernel serves a model with the default and a model with a prior on the input."
  A subclass of `ConditionalDistribution` declares one with `InputSpec(...).with_optional(name)`.
- **A kernel's moments at given values.** `variance(kernel, given=draw)`, the same as
  `variance(condition_on(kernel, draw))`, is the variance of the law the kernel gives at `draw`.
- **Path indexing.** `posterior.atoms["s2/day"]` reads a scale's draws.

## 3. The design

**One likelihood.** `GaussianSimulatorLikelihood` declares the record `s2` as an **optional input**
with **default scales**, which replace the `scales` mapping:

```python
likelihood = GaussianSimulatorLikelihood(
    forward,
    output_name="nee",
    reference_variance=...,
    classes=...,
    default_scales={"day": 1.0, "night": 1.0},
)
```

At given values that include `s2` it uses them, and otherwise the defaults. Its law at given values
claims its variance, $s^2_{k(i)}\,\sigma^2_i$, which needs no forward run.

**Fixed or inferred is a property of the prior alone.** `nee_problem(scales="fixed")` builds the
prior $\pi_0(\theta)$, and `scales="inferred"` builds $\pi_0(\theta)\,\pi_0(s^2)$. The likelihood is
the same object in both. A draw holds `s2` exactly when the prior has it.

**$\Gamma$ at a draw is the variance of the likelihood there**, divided by the learning rate:

```python
variance(problem.likelihood, given=draw)    # s2_k(i) sigma2_i, at the draw's own scales or the defaults
problem.noise_variance(draw)                # Γ: the line above divided by η
```

The line reads the same for both shapes, inside or outside an expectation. `problem.noise_variance`
is that one line. It refuses a draw without `s2` when the problem infers its scales, because the
likelihood would otherwise silently use its defaults. Called with no draw, it gives the fixed
$\Gamma$, and refuses when the scales are inferred, as today. `problem.misfit(predictions, draw)`
reads $\Gamma$ the same way.

**Reading the scales.** Code that stores or scores the scales reads them by path, as
`posterior.atoms["s2"]`. Whether a problem infers its scales, and for which classes, is read from
the prior: `problem.noise_scale_classes`, empty when they are fixed.

**Each consumer:**

| Module | Today | After |
|---|---|---|
| `_stopgaps/likelihood.py` | `scales` with `INFERRED`, `inferred_scale_classes`, `inferred_scale_draws`, `noise_variances` | optional `s2`, `default_scales`, the law's variance |
| `problems/nee_likelihood.py` | builds both the likelihood's scales and the prior | builds the prior only |
| `problems/inverse_problem.py` | `noise_variance(noise_scales)`, numpy | `noise_variance(draw)`, `noise_scale_classes` |
| `tools/estimators.py` | `_noise_variance_at_draws`, `_drawn_noise_scales` | `problem.noise_variance(draw)` in the integrand (design 09) |
| `tools/reduced.py` | copies `scales`; lifts by `inferred_scale_draws` | copies `default_scales`; lifts `s2` by path |
| `inference/enskit_eki.py`, `inference/noise_scales.py` | per-class fixed values from the likelihood's `scales` | fixed values from `default_scales` when the prior has no `s2`; inverse-gamma priors from the prior's `s2` factor, as now |
| `experiments/metrics.py`, `reference.py`, `study.py`, `algorithms/active_subspace_mcmc.py` | `inferred_scale_classes`, `inferred_scale_draws` | `noise_scale_classes`, `atoms["s2"]` |

**Removed:** `INFERRED`, the `scales` mapping, `inferred_scale_classes`, `inferred_scale_draws`,
`noise_variances`, and the `noise_scales=` arguments. `NOISE_SCALES` stays as the record's name.

**Given up:** some classes fixed and others inferred within one problem. The likelihood's
`scales` mapping allows it, but no problem uses it, and the record `s2` is now present or absent
whole.

## 4. Verified

By [`prototypes/10-noise-model/optional_scales.py`](prototypes/10-noise-model/optional_scales.py),
on the toy problem at the pin, with a likelihood built as above:

| Check | Result |
|---|---|
| the likelihood's input spec | `theta`, and `s2` optional |
| a draw of `likelihood & prior`, fixed / inferred | `nee`, `theta` / `nee`, `theta`, `s2` |
| pointwise log-density against today's likelihood, both shapes | equal |
| `variance(condition_on(likelihood, draw))` against today's `noise_variances`, both shapes | equal |
| `variance(likelihood, given=draw)` against the line above, both shapes | equal |
| the same inside `expectation` over the prior, both shapes | works |
| the same inside `expectation` over held draws | fails: ProbPipe #716 (design 09, C3) |
| conditioning `likelihood & prior` with NUTS and RWMH, both shapes | works; draws hold `s2` only when inferred |
| harvard_forest_small: one SIPNET run against the variance at a draw | 0.43 s against 0.005 s: no forward run |

## 5. Choke points

No new gap. Reading $\Gamma$ inside an expectation over held draws fails because of ProbPipe #716,
as every call on a draw there does; design 09 covers it.

## 6. Decisions needed

1. **`problem.noise_variance(draw)`.** Keep it as the one-line method, since $\Gamma$ includes
   $\eta$, which only the problem holds? Or write `variance(problem.likelihood, given=draw) / eta`
   in each estimator?
2. **With no draw.** Allow `problem.noise_variance()` for fixed scales (the notebooks use it), and
   refuse it for inferred ones, as today?
3. **`problem.noise_scale_classes`.** A tuple of the classes whose scales the prior holds. Is this
   the name and form you want?
4. **Mixed fixed and inferred classes.** Drop support, since nothing uses it?
