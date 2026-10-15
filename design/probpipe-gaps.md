# ProbPipe gap registry

This file is the single list of places where this repo needs something ProbPipe does not yet do.
Every stopgap in the code carries a tag naming one entry here, `# STOPGAP(PG-NN)`, and a test fails
if a tag names an id missing from this file. When a gap closes upstream, we delete its stopgap and
mark the entry **closed**. This is how the debt shrinks to zero.

**Ids.** Entries are numbered `PG-01`, `PG-02`, … and never renumbered. The `PG-` prefix keeps them
distinct from the gap numbers inside ProbPipe issues (for example "G3" in #580).

**Pinned ProbPipe.** `dev/overhaul` at `638f56bc` (2026-10-14). File and line references are to that
commit. "Verified" means a script in [`prototypes/`](prototypes/) reproduces the behaviour.

**Status values.** `open` (no upstream fix yet) · `planned` (an upstream issue designs the fix) ·
`closed` (fixed upstream and the stopgap removed).

## Summary

| Id | Gap | Severity here | Upstream | Status |
|---|---|---|---|---|
| [PG-01](#pg-01) | Non-JAX kernels cannot be sampled jointly | High: prior predictive | #580 G1 | planned |
| [PG-02](#pg-02) | No "simulator plus Gaussian noise" likelihood family | High: the EnsKit seam | to file | open |
| [PG-03](#pg-03) | No learning rate (power likelihood) on conditioning | High: the likelihood | to file | open |
| [PG-04](#pg-04) | The inference-method extension surface is private | Medium | #580 G8, design §9 | planned |
| [PG-05](#pg-05) | No public flat/unconstrained coordinates; views cannot be flattened | Medium | #522 (partly) | open |
| [PG-06](#pg-06) | No reuse of evaluations (caching) | Low today, high on a cluster | #580 G6, #352 | planned |
| [PG-07](#pg-07) | No pluggable backend; inference loops bypass `Function`; no failure contract | Low today | #580 G3, G5 | planned |
| [PG-08](#pg-08) | No differentiability claim or Jacobian operation | Medium: toy problem | #580 G4, #522 | planned |
| [PG-09](#pg-09) | No linear pushforward of a law, no `LinOp` as a `Function` | Medium: reduced models, `W` | #521, #522 | planned |
| [PG-10](#pg-10) | `dispatch="auto"` re-runs the body; empirical lifts thin silently | Low (footguns) | #580 G2, defect 6 | planned |
| [PG-11](#pg-11) | A normalized density needs both method names to be claimed | Low (footgun) | to file | open |
| [PG-12](#pg-12) | "Unnormalized conditional density" does not say what may depend on the given | Medium (semantics) | to file | open |

Not ProbPipe, but tracked the same way: **EK-01**, EnsKit switches JAX to 64-bit floats on import
(below).

---

## PG-01

**Non-JAX kernels cannot be sampled jointly.**

- **What happens.** `sample(likelihood * prior)` raises `TracerArrayConversionError` when the
  likelihood calls SIPNET. The joint maps each factor over its drawn givens with `jax.vmap`, on both
  its compiled path and its fallback (`distributions/_factored.py:828-834`). This holds whether the
  kernel is built with `conditional_distribution` or by subclassing `ConditionalDistribution`.
  Verified: `t2_dispatch.py`, `t4_joint.py`.
- **Effect here.** Prior predictive draws of the observables, which the walkthrough notebook needs,
  cannot come from the model directly.
- **Ideal ProbPipe.** A joint evaluates a factor that does not trace row by row, through the
  execution layer, so `sample(model, n)` works for any model.
- **Stopgap.** `lsmas.predictive.prior_predictive(problem, n)`: sample the prior, call the forward
  `Function` on the θ batch, add noise from the likelihood. It returns the record a joint draw would.
- **Removal.** Replace the helper's body with `sample(problem.model, n)`.

## PG-02

**No "simulator plus Gaussian noise" likelihood family.**

- **What happens.** A kernel built with `conditional_distribution(lambda theta, s2: Normal("y",
  G(theta), …))` exposes a density and draws, but an ensemble Kalman method needs the pieces: the
  forward map G as a batchable `Function`, the noise covariance, and which given slots scale which
  parts of it. Nothing in the kernel exposes them. (Separately, `conditional_distribution` refuses a
  function that does not trace at construction; see #580 G1.)
- **Effect here.** The EnsKit methods cannot read G and Γ from an arbitrary ProbPipe model.
- **Ideal ProbPipe.** A family, a generalization of `glm_likelihood`, built from a forward
  `Function` and a covariance with named scale slots. It claims conditional sampling, a normalized
  conditional density, a conditional mean (G) and a conditional covariance (Γ), and exposes the
  forward `Function` so methods can evaluate it on a batch.
- **Stopgap.** `lsmas._stopgaps.likelihood.GaussianSimulatorLikelihood`, a `ConditionalDistribution`
  subclass with exactly that shape. It is a proper law: its draws and its density agree (verified:
  `t6_tempered.py`, check 1).
- **Removal.** Construct the ProbPipe family instead; the methods' `check` reads its capabilities.

## PG-03

**No learning rate (power likelihood) on conditioning.**

- **What happens.** The working NEE likelihood is a generalized-Bayes update,
  π_η(θ, s² | y) ∝ π(θ, s²) ∏ᵢ N(yᵢ | Gᵢ(θ), s²vᵢ)^{ηᵢ}. Raising a likelihood to a power is a
  property of the *update*, not of the data model: N(y | G, s²v)^η is not the density of any law of
  y (as a function of s² it differs from the density of N(G, s²v/η), which would turn the conjugate
  update IG(a + ηn/2, ·) into IG(a + n/2, ·)). ProbPipe has no way to say "condition with learning
  rate η".
- **Effect here.** Without care, the tempering ends up inside the likelihood, and one object's
  sampler and density describe different things.
- **Ideal ProbPipe.** A learning rate on conditioning, for example
  `condition_on.using(learning_rate={"nee": eta}, method=...)(model, data)`, forming the
  unnormalized conditional whose log-density is log π(θ, s²) + Σᵢ ηᵢ log p(yᵢ | θ, s²). Provenance
  records η, and methods can read it from the target. η may be a scalar per observed component or
  an array over the component's elements. The model stays a proper generative model, so prior and
  posterior predictives are unaffected.
- **Stopgap.** `lsmas._stopgaps.tempering.tempered_condition_on(model, data, *, learning_rate,
  **using)`, which has that signature. Internally it swaps the likelihood factor for a private
  `TemperedLikelihood` that claims only the unnormalized conditional density Σᵢ ηᵢ log N(…) and
  cannot be sampled, then calls `condition_on`. The tempered object never leaves the function.
  Verified (`t6_tempered.py`): the target's log-density differences equal
  log π + Σ ηᵢ log Nᵢ exactly; EKI and ProbPipe's RWMH both run on it.
- **Removal.** Replace calls with `condition_on.using(learning_rate=..., ...)`.
- **Related.** PG-12, which is why the private object must not escape.

## PG-04

**The inference-method extension surface is private.**

- **What happens.** `InferenceMethod`, `Feasibility` and `inference_method_registry` are public,
  but every method that reads a model's structure needs `_UnnormalizedConditional`
  (`operations/_condition.py:481`), `model_factors` (`inference/_inference_utils.py:577`),
  `make_posterior` (`inference/_approximate_distribution.py:393`) and `run_seed`
  (`inference/_inference_utils.py:221`).
- **Ideal ProbPipe.** A documented public target interface (#580 design §9 item 3).
- **Stopgap.** `lsmas._stopgaps.probpipe_private` re-exports the four names; it is the only module
  that imports from a ProbPipe underscore module.
- **Removal.** Import the public names; delete the module.

## PG-05

**No public flat/unconstrained coordinates; views cannot be flattened.**

- **What happens.** Methods need a prior's parameter layout and the maps to unconstrained
  coordinates. The helpers exist but are private (`unconstrained_coordinates`, `flat_unflatten` in
  `inference/_inference_utils.py:1048, 422`). They also refuse a view:
  `unconstrained_coordinates(prior["theta"])` raises `TypeError: cannot pack the draws of FieldView
  …` (verified), so the coordinates of θ alone must be cut out of those of the whole prior.
- **Ideal ProbPipe.** A public `coordinates(law)` giving names, the flat layout, and the maps to and
  from unconstrained coordinates with their log-Jacobian, accepting views. #522's flat-vector law
  covers part of this.
- **Stopgap.** `lsmas._stopgaps.coordinates.Coordinates.from_prior(prior, block="theta")`, with
  `names`, `to_unconstrained`, `to_constrained`, `whiten`, `unwhiten`.
- **Removal.** Build `Coordinates` on the public function; the whitening stays ours.

## PG-06

**No reuse of evaluations.**

- **What happens.** An identical call re-runs the simulator (32 runs twice, verified
  `t2_dispatch.py`). Laws are immutable, so G(θ) cannot be memoized on a law either (verified; this
  is by design).
- **Effect here.** Small today: a run takes 0.3 s.
- **Ideal ProbPipe.** Reuse of evaluations in the workflow layer (#580 G6 and design §10; #352).
- **Stopgap.** None in code. Methods keep every evaluation in their results (the ensemble and its
  predictions at every step), so diagnostics never re-run SIPNET.

## PG-07

**No pluggable backend; inference loops bypass `Function`; no failure contract.**

- **What happens.** Dispatch modes are the closed set `auto`, `jax`, `sequential`, `thread`
  (`values/_function_base.py:1641-1643`). ProbPipe's own inference methods evaluate densities in
  their own loops, so RWMH runs SIPNET sequentially (2 chains × 30 steps: 67 runs, 26 s, verified
  `t5_rwmh.py`). The first failure in a batch aborts the call.
- **Ideal ProbPipe.** #580 G3 and G5: a backend registry selectable per call, inference methods that
  evaluate through it, and per-point failure outcomes.
- **Stopgap.** `dispatch="thread"` on the forward `Function` (8 workers: 32 runs in 2.1 s against
  8.6 s, verified); the forward body returns a NaN row for a failed run, which is EnsKit's
  contract. Our EnsKit methods always evaluate the ensemble through the forward `Function`.
- **Removal.** Register a backend; drop the NaN convention once failures are first-class.

## PG-08

**No differentiability claim or Jacobian operation.**

- **What happens.** `Function(differentiable=...)` raises `NotImplementedError`
  (`values/_function_base.py:1077-1080`). NUTS is selected because a density traces, not because
  gradients are declared. There is no `jacobian` operation.
- **Effect here.** The differentiable toy problem needs exact Jacobians to check averaged-gradient
  estimators.
- **Ideal ProbPipe.** #580 G4 and #522: the claim, and a Jacobian operation.
- **Stopgap.** The toy problem computes Jacobians with `jax.jacfwd` directly, in one function.

## PG-09

**No linear pushforward of a law; `LinOp` is not a `Function`.**

- **What happens.** `LinearPushforwardDistribution` and `LinearGaussianConditional` are not
  implemented (`families/_conditional.py:103-109`), and `LinOp` is a bare ABC.
- **Effect here.** A reduced model writes ξ = W a + W⊥ b; the subspace's `W` would naturally be a
  `LinOp`.
- **Ideal ProbPipe.** #521, #522.
- **Stopgap.** `W` is an array. Reduced models are built from a reparameterized forward `Function`
  (θ as a function of the active variables). Design 03 states exactly what is needed.

## PG-10

**`dispatch="auto"` re-runs the body; empirical lifts thin silently.**

- **What happens.** `auto` runs the body once more to probe traceability (one extra SIPNET run per
  call). Lifting a `Function` over an `EmpiricalDistribution` with more atoms than
  `n_broadcast_samples` (default 256) samples instead of enumerating, without warning.
- **Ideal ProbPipe.** #580 G2 and defect 6.
- **Stopgap.** Always set `dispatch` and `n_broadcast_samples` explicitly; a lint test checks every
  `@function` in `lsmas` sets `dispatch`.

## PG-11

**A normalized density needs both method names to be claimed.**

- **What happens.** A law that defines only `_log_prob` does not satisfy `SupportsLogProb`, and a
  kernel that defines only `_conditional_log_prob` does not satisfy `SupportsConditionalLogProb`.
  The protocols' default for the unnormalized method applies only to classes that subclass the
  protocol. `log_prob(law, y)` then fails with "does not implement SupportsLogProb". Verified while
  building `GaussianSimulatorLikelihood`.
- **Ideal ProbPipe.** Defining the normalized method claims both capabilities, or the docs say to
  subclass the protocol.
- **Stopgap.** Our classes define both names; the unnormalized one returns the normalized one.

## PG-12

**"Unnormalized conditional density" does not say what may depend on the given.**

- **What happens.** For a law, "unnormalized" means up to an additive constant. For a kernel the
  constant could depend on the given values, and then the joint's density, and so the posterior, is
  not determined. Inference reads a kernel's unnormalized density as exact in the given values
  (`likelihood_flat`: "up to a constant in the parameters"). Binding an unnormalized kernel's given
  values, however, normalizes it separately at each given value. The two readings disagree whenever
  the normalizer depends on the given. The tempered factor of PG-03 is exactly such an object.
  Verified while testing it: `condition_on(tempered_likelihood, given)` treats the factor as an
  unnormalized law of y and runs an inference method to normalize it (about 50 s on the six-point
  test model), where conditioning the *model* on y uses the factor as a likelihood term.
- **Ideal ProbPipe.** State that a kernel's unnormalized density is exact up to a constant that does
  not depend on the given; or add a separate "factor" (potential) kind, a function of the given and
  the event that enters only conditioning and claims no law. PG-03's learning rate removes our need
  for either.
- **Stopgap.** The tempered factor is private to `tempered_condition_on` and claims no sampling.

---

## EK-01

**EnsKit switches JAX to 64-bit floats on import.**

- **What happens.** `import enskit` enables `jax_enable_x64`. A ProbPipe prior built before that
  import and a JAX sampler run after it give a float32/float64 mismatch ("cond branches must have
  equal output types"; verified, `t5_rwmh.py`). #580's principle J6 asks the same of ProbPipe.
- **Stopgap.** `lsmas/__init__.py` imports EnsKit before anything else, so 64-bit is on everywhere.
