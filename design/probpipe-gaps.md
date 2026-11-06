# ProbPipe gap registry

This file is the single list of places where this repo needs something ProbPipe does not yet do.
Every stopgap in the code carries a tag naming one entry here, `# STOPGAP(PG-NN)`, and a test fails
if a tag names an id missing from this file. When a gap closes upstream, we delete its stopgap and
mark the entry **closed**. This is how the debt shrinks to zero.

**Ids.** Entries are numbered `PG-01`, `PG-02`, … and never renumbered. The `PG-` prefix keeps them
distinct from the gap numbers inside ProbPipe issues (for example "G3" in #580).

**Pinned ProbPipe.** `main` at `9158a50e` (2026-11-02). File and line references are to that
commit. "Verified" means a script in [`prototypes/`](prototypes/) reproduces the behaviour.

**Status values.** `open` (no upstream fix yet) · `planned` (an upstream issue or plan designs the
fix) · `partly` (fixed upstream in part; the stopgap shrank) · `closed` (fixed upstream and the
stopgap removed).

**Verification at the pin.** Every open entry was re-checked at `9158a50e` by
[`01c-probpipe-main/gaps_at_main.py`](prototypes/01c-probpipe-main/gaps_at_main.py), which prints one line per behaviour below.

## Summary

| Id | Gap | Severity here | Upstream | Status |
|---|---|---|---|---|
| [PG-01](#pg-01) | Non-JAX kernels cannot be sampled jointly | High: prior predictive | #580 G1 | closed |
| [PG-02](#pg-02) | No "simulator plus Gaussian noise" likelihood family | High: the EnsKit seam | to file | open |
| [PG-03](#pg-03) | No learning rate (power likelihood) on conditioning | High: the likelihood | to file | open |
| [PG-04](#pg-04) | The inference-method extension surface is private | Medium | #580 G8, design §9 | closed |
| [PG-05](#pg-05) | No public map to unconstrained coordinates | Medium | `plans/deferred/vector-interface.md` | partly |
| [PG-06](#pg-06) | No reuse of evaluations (caching) | Low today, high on a cluster | `plans/execution-layer.md` §6.5 | planned |
| [PG-07](#pg-07) | Inference loops evaluate one point at a time; no tolerant failure mode | Low today | #580 G3, G5 | partly |
| [PG-08](#pg-08) | No differentiability claim or Jacobian operation; selection assumes gradients | Medium: toy problem, method choice | `plans/deferred/differentiability.md` | planned |
| [PG-09](#pg-09) | No linear pushforward of a law, no `LinOp` as a `Function` | Medium: reduced models, `W` | #521, #522 | planned |
| [PG-10](#pg-10) | An undeclared `Function` runs once more to probe; empirical lifts thin silently | Low (footguns) | #580 G2, #585 | partly |
| [PG-11](#pg-11) | A normalized density needs both method names to be claimed | Low (footgun) | to file | open |
| [PG-12](#pg-12) | "Unnormalized conditional density" does not say what may depend on the given | Medium (semantics) | to file | open |
| [PG-13](#pg-13) | A renamed factor taken from a joint loses its path names | Medium (bug) | to file | open |
| [PG-14](#pg-14) | Fingerprinting a closure that captures a sparse matrix recurses without end | Low (bug) | to file | open |
| [PG-15](#pg-15) | No public operation draws from a law with a given key | Low | to file | open |

Not ProbPipe, but tracked the same way: **EK-01**, EnsKit switches JAX to 64-bit floats on import
(below).

---

## PG-01

**Non-JAX kernels cannot be sampled jointly.** *Closed at `9158a50e`.*

- **What happened.** At `638f56bc`, `sample(likelihood * prior)` raised
  `TracerArrayConversionError` when the likelihood called SIPNET: the joint maps each factor over
  its drawn givens with `jax.vmap`, and nothing let a factor opt out. `lsmas.predictive` drew the
  prior, called the forward `Function` on the batch, and added the noise itself.
- **Fixed upstream.** A `Function` declared `traceable=False`, with an output spec of concrete shape
  and dtype, runs through a host callback when traced code calls it, and its executor evaluates
  the points. A kernel whose conditional methods are JAX code around such a call can be sampled
  jointly.
- **Done here.** `GaussianSimulatorLikelihood`'s sampler and density are JAX code around the forward
  map, called with `raw=True` so that it returns plain arrays inside and outside a trace.
  `sample(problem.model, n)` gives prior predictive draws; on the development problem, 16 draws
  run SIPNET on the forward map's threads in about 2 s. `lsmas.predictive` is deleted, and its
  tests became tests of sampling the model.

## PG-02

**No "simulator plus Gaussian noise" likelihood family.**

- **What happens.** A kernel built with `conditional_distribution(lambda theta, s2: Normal("y",
  G(theta), …))` exposes a density and draws, but an ensemble Kalman method needs the pieces: the
  forward map G as a batchable `Function`, the noise covariance, and which given slots scale which
  parts of it. Nothing in the kernel exposes them. Since `9158a50e`, `conditional_distribution`
  accepts a G that JAX cannot trace (see PG-01), but the kernel still hides G and the covariance.
- **Effect here.** The EnsKit methods cannot read G and Γ from an arbitrary ProbPipe model.
- **Ideal ProbPipe.** A family, a generalization of `glm_likelihood`, built from a forward
  `Function` and a covariance with named scale slots. It claims conditional sampling, a normalized
  conditional density, a conditional mean (G) and a conditional covariance (Γ), and exposes the
  forward `Function` so methods can evaluate it on a batch.
- **Stopgap.** `lsmas._stopgaps.likelihood.GaussianSimulatorLikelihood`, a `ConditionalDistribution`
  subclass with exactly that shape. It is a proper law: its draws and its density agree (tests in
  `tests/test_likelihood.py`), and a model containing it samples jointly.
- **Removal.** Construct the ProbPipe family instead; the methods' `check` reads its capabilities.

## PG-03

**No learning rate (power likelihood) on conditioning.**

- **What happens.** The working NEE likelihood is a generalized-Bayes update,
  π_η(θ, s² | y) ∝ π(θ, s²) ∏ᵢ N(yᵢ | Gᵢ(θ), s²vᵢ)^{ηᵢ}. Raising a likelihood to a power is a
  property of the *update*, not of the data model: N(y | G, s²v)^η is not the density of any law of
  y (as a function of s² it differs from the density of N(G, s²v/η), which would turn the conjugate
  update IG(a + ηn/2, ·) into IG(a + n/2, ·)). ProbPipe has no way to say "condition with learning
  rate η": at `9158a50e`, `condition_on.using(learning_rate=...)` is refused as an unknown control
  (verified).
- **Effect here.** Without care, the tempering ends up inside the likelihood, and one object's
  sampler and density describe different things.
- **Ideal ProbPipe.** A learning rate on conditioning, for example
  `condition_on.using(learning_rate={"nee": eta}, method=...)(model, data)`, forming the
  unnormalized conditional whose log-density is log π(θ, s²) + Σᵢ ηᵢ log p(yᵢ | θ, s²). Provenance
  records η, and methods can read it from the target. η may be a scalar per observed output or an
  array over the output's elements. The model stays a proper generative model, so prior and
  posterior predictives are unaffected.
- **Stopgap.** `lsmas._stopgaps.tempering.tempered_condition_on(model, data, *, learning_rate,
  **using)`, which has that signature. Internally it swaps the likelihood factor for a private
  `TemperedLikelihood` that claims only the unnormalized conditional density Σᵢ ηᵢ log N(…) and
  cannot be sampled, then calls `condition_on`. The tempered object never leaves the function.
  Tested (`tests/test_tempering.py`, `tests/test_enskit_eki.py`): the target's log-density
  differences equal log π + Σ ηᵢ log Nᵢ exactly, and on the linear-Gaussian test model both EKI
  and ProbPipe's default sampler (NUTS, since that model's forward map traces) recover the exact
  tempered posterior.
- **Removal.** Replace calls with `condition_on.using(learning_rate=..., ...)`.
- **Related.** PG-12, which is why the private object must not escape.

## PG-04

**The inference-method extension surface is private.** *Closed at `9158a50e`.*

- **What happened.** At `638f56bc`, `InferenceMethod`, `Feasibility` and
  `inference_method_registry` were public, but a method that reads a model's structure needed four
  private names: the conditioning target class, `model_factors`, `make_posterior` and `run_seed`.
- **Fixed upstream.** `probpipe.inference` now exports `InferenceTarget`, whose `factors()` returns
  `ModelFactors(prior, likelihood, observed)`, and `make_posterior(target, chains, *, method,
  annotations, …)`, which reads chains in the `to_vector` layout of the target's event.
  `condition_on` passes the run's random key to `execute(target, /, *, key, **options)`. A method's
  `supported_types` is `(Distribution,)`, since every target is a law.
- **Done here.** `EnsKitEKI` uses only these names, and `lsmas._stopgaps.probpipe_private` is
  deleted. Drawing the initial ensemble from the prior with the run's key still needs a private
  method: [PG-15](#pg-15).

## PG-05

**No public map to unconstrained coordinates.**

- **What happens.** Methods need a prior's parameter layout and the maps to unconstrained
  coordinates. Since `9158a50e` the layout is public: `to_vector` on a record or a batch of records,
  `NumericRecord.from_vector`, and `vector_size` on a spec, and the draws of a view such as
  `prior["theta"]` lay out as vectors (verified). The maps are not: `bijector_for(support)` gives
  the forward map b, but `inverse(bijector)` raises `NotImplementedError`, so b⁻¹ is reachable only
  through the bijector's private `_inverse`. ProbPipe's own helper, `unconstrained_coordinates`
  (`inference/_inference_utils.py:1007`), is private and refuses a view (`TypeError: cannot lay out
  the draws of CoupledMarginal …`).
- **Ideal ProbPipe.** A public map from a law's values to unconstrained vectors and back, with its
  log-Jacobian, that accepts views. `plans/deferred/vector-interface.md` proposes it
  (`vector_bijector`, `unconstrained_law`); the plan is not yet approved.
- **Stopgap.** `lsmas._stopgaps.coordinates.ParameterBijection.from_prior(prior, block)` builds the
  map from the block's *declared* supports, using the public `bijector_for`, its `raw` forward
  map, and its `_inverse`. It works whatever internal form the prior takes. `Coordinates.from_prior`
  adds whitening (see PG-15 for its draws).
- **Removal.** Build `ParameterBijection` on the public map; the whitening stays ours.

## PG-06

**No reuse of evaluations.**

- **What happens.** An identical call re-runs the simulator: eight points, eight runs, twice
  (verified). Laws are immutable, so G(θ) cannot be memoized on a law either; this is by design.
- **Effect here.** Small today: a run takes 0.3 s.
- **Ideal ProbPipe.** Reuse of evaluations in the workflow layer, designed in
  `plans/execution-layer.md` §6.5 (the earlier issue, #492, closed as superseded by it).
- **Stopgap.** None in code. Methods keep every evaluation in their results (the ensemble and its
  predictions at every step), so diagnostics never re-run SIPNET.

## PG-07

**Inference loops evaluate one point at a time; no tolerant failure mode.**

- **Fixed at `9158a50e`.** Execution is pluggable: `executor_registry` holds `jax`, `sequential`,
  `thread` and `process`, a user can register an `EvaluationExecutor` of their own (for a cluster,
  say), and `probpipe.testing.check_executor` checks one. A failing point raises `EvaluationError`,
  whose `failures` name the failed points and whose `outputs` keep the others (verified).
- **What still happens.** ProbPipe's RWMH evaluates the density one point per chain at a time,
  whatever executor the call names: 2 chains × 30 steps ran 67 evaluations, at most 2 at once
  (verified). And there is no mode that carries on past failed points: `on_failure="drop"` is
  deferred.
- **Ideal ProbPipe.** #580 G3 and G5: inference methods that hand the executor batches of points,
  and a failure mode that returns the outcomes it has.
- **Stopgap.** The forward map runs on the `thread` executor (8 workers), and its body returns a
  row of NaN for a failed run, which is EnsKit's contract. Our EnsKit methods always evaluate the
  ensemble through the forward `Function`, so it runs in parallel.
- **Removal.** Drop the NaN convention once a tolerant failure mode exists.

## PG-08

**No differentiability claim or Jacobian operation; selection assumes gradients.**

- **What happens.** `Function(differentiable=...)` raises `NotImplementedError`
  (`values/_function_base.py:1275`), and there is no `jacobian` operation. `jax.jacfwd` works
  through a `Function` that JAX can trace, and a gradient through an opaque one is refused with a
  clear message (verified). Automatic method selection asks only whether a density traces, not
  whether it has gradients. A density written in JAX around an opaque `Function` traces (the
  `Function` runs through a host callback), so `condition_on` picks NUTS, which then fails:
  "gradients cannot pass through 'nee_features', which is not traceable and supplies no
  derivative" (verified with a `GaussianSimulatorLikelihood`). A kernel built with
  `conditional_distribution` around an opaque `Function` is recognized, and gets RWMH.
- **Effect here.** The differentiable toy problem needs exact Jacobians to check averaged-gradient
  estimators. And since the likelihood became JAX code (PG-01), conditioning a SIPNET problem
  without naming a method fails at once, where it used to pick RWMH.
- **Ideal ProbPipe.** The claim and a Jacobian operation, proposed in
  `plans/deferred/differentiability.md` (not yet approved).
- **Stopgap.** The toy problem will compute Jacobians with `jax.jacfwd` directly, in one function.
  `InverseProblem.condition` requires its `method` argument, so a forgotten method fails at the
  call with a plain `TypeError` rather than inside NUTS.

## PG-09

**No linear pushforward of a law; `LinOp` is not a `Function`.**

- **What happens.** `LinearPushforwardDistribution` and `LinearGaussianConditional` raise
  `NotImplementedError` (`families/_transformed.py:81`, `families/_conditional.py:109`).
  `probpipe.linalg` has concrete operators (`DenseLinOp`, `DiagonalLinOp`, …), but they are not
  `Function`s.
- **Effect here.** A reduced model writes ξ = W a + W⊥ b; the subspace's `W` would naturally be a
  `LinOp`.
- **Ideal ProbPipe.** #521, #522.
- **Stopgap.** `W` is an array. Reduced models are built from a reparameterized forward `Function`
  (θ as a function of the active variables). Design 03 states exactly what is needed.

## PG-10

**An undeclared `Function` runs once more to probe; empirical lifts thin silently.**

- **Fixed at `9158a50e`.** A `Function` that declares `traceable=False` is never probed. One that
  declares nothing is probed once per structure of its inputs, not on every call: eight points
  ran the body nine times, then eight (verified).
- **What still happens.** Lifting a `Function` over an empirical law with more atoms than
  `max_evaluations` (default 256) draws instead of enumerating, without a warning: 300 atoms gave
  256 (verified; #585).
- **Ideal ProbPipe.** #580 G2, #585.
- **Stopgap.** Every `Function` states whether JAX can trace it (`traceable=`), so ProbPipe never
  probes it. A repository test (`tests/test_repository.py`) checks this for every `Function` built
  in `lsmas` and in the tests. Nothing here lifts a `Function` over an empirical law yet; code
  that does must set `max_evaluations` to the number of atoms.

## PG-11

**A normalized density needs both method names to be claimed.**

- **What happens.** A law that defines only `_log_prob` does not satisfy `SupportsLogProb`, and a
  kernel that defines only `_conditional_log_prob` does not satisfy `SupportsConditionalLogProb`
  (verified). `log_prob(law, y)` then fails with "does not implement SupportsLogProb". ProbPipe's
  documentation says otherwise: "A class claims a capability by defining that method"
  (`docs/api/distributions.md:79`). Subclassing the protocol does work.
- **Ideal ProbPipe.** Defining the normalized method claims both capabilities, or the docs say to
  subclass the protocol.
- **Stopgap.** Our classes define both names; the unnormalized one returns the normalized one.

## PG-12

**"Unnormalized conditional density" does not say what may depend on the given.**

- **What happens.** For a law, "unnormalized" means up to an additive constant. For a kernel the
  constant could depend on the given values, and then the joint's density, and so the posterior, is
  not determined. Inference reads a kernel's unnormalized density as exact in the given values.
  Binding an unnormalized kernel's given values, however, normalizes it separately at each given
  value: `condition_on(kernel, given)` treats the kernel as an unnormalized law of y and runs an
  inference method to normalize it (verified). The two readings disagree whenever the normalizer
  depends on the given. The tempered factor of PG-03 is exactly such an object.
- **Ideal ProbPipe.** State that a kernel's unnormalized density is exact up to a constant that does
  not depend on the given; or add a separate "factor" (potential) kind, a function of the given and
  the event that enters only conditioning and claims no law. PG-03's learning rate removes our need
  for either.
- **Stopgap.** The tempered factor is private to `tempered_condition_on` and claims no sampling.

## PG-13

**A renamed factor taken from a joint loses its path names.**

- **What happens.** Build `theta = (Normal("a", …) & Normal("b", …)).rename({"a": "theta/a", "b":
  "theta/b"})` and `model = likelihood & theta`. The factor of `model.factors` that corresponds to
  `theta` declares the output `theta`, but `log_prob(factor, {"theta": {"a": …, "b": …}})` raises
  `ApplicabilityError: … missing ['a', 'b'], unexpected ['theta']`, and its draws have the fields
  `a` and `b`, ungrouped (verified). `log_prob(theta, …)` on the original object works. The prior
  that `target.factors()` gives an inference method is such a factor.
- **Effect here.** A model's prior cannot be recovered from its factors and then scored, and its
  draws need reading in either layout.
- **Ideal ProbPipe.** A factor taken from a joint keeps the renaming, so its declaration, its draws
  and its density agree.
- **Stopgap.** `InverseProblem` stores the likelihood and the prior it was built from and derives
  the model from them, rather than splitting the model. `lsmas._stopgaps.coordinates.block_values`
  reads a block's values from draws in either layout. (`tempered_condition_on` does rebuild a
  joint from `model.factors`; its tests confirm the resulting target density is correct.)
- **Removal.** Read draws in the grouped layout only. Storing the parts is the simpler design
  anyway, so `InverseProblem` keeps it.

## PG-14

**Fingerprinting a closure that captures a sparse matrix recurses without end.**

- **What happens.** `fingerprint(Function(body))`, where `body` is a closure over a
  `scipy.sparse.csr_matrix`, raises `RecursionError` (verified). Provenance catches it and logs
  "fingerprint() failed … maximum recursion depth exceeded" on every call, and the result's
  fingerprint is missing. A closure over an int, a dict or an arbitrary object is fine. The cause:
  `np.asarray` of a sparse matrix is a 0-d array of dtype object holding the matrix itself, and
  `_update_array` (`core/_fingerprint.py:483`) hashes each element of such an array with a fresh
  depth, so it meets the matrix again.
- **Effect here.** The forward map's provenance was weaker, with a warning per SIPNET run.
- **Ideal ProbPipe.** Sparse matrices hash by content or fall back to identity, and the traversal
  of an object array respects the depth guard.
- **Stopgap.** The forward map is a bound method of a small class (`lsmas.sipnet._FeatureEvaluator`),
  the pattern ProbPipe's documentation recommends for functions that use an object's state; a
  bound method is fingerprinted without walking the operator. The cost: two forward maps that
  differ only in their operator fingerprint alike (verified). That matters only once ProbPipe
  reuses evaluations by fingerprint (PG-06).
- **Removal.** Nothing to remove; recorded so the bug gets fixed.

## PG-15

**No public operation draws from a law with a given key.**

- **What happens.** `condition_on` hands an inference method the run's random key, and ProbPipe's
  guide says a method draws all its randomness from it. But `sample(law, n)` takes its key from the
  enclosing `workflow_run` and accepts none (`sample.using(key=...)` is refused as an unknown
  control). ProbPipe's own methods call the method a family implements, `law._sample(key, shape)`
  (for example `inference/_bayesflow_common.py:275`). Opening a nested `workflow_run` instead is no
  substitute: it shifts the outer run's random stream (verified).
- **Effect here.** Two places draw with a key of their own: `EnsKitEKI`'s initial ensemble (the
  run's key) and `Coordinates.from_prior`'s whitening draws (a fixed key, so the coordinates neither
  depend on nor disturb the caller's run).
- **Ideal ProbPipe.** A documented keyed draw for extension code, such as a `key` control on
  `sample`, or a public function `draw(law, key, sample_shape)`.
- **Stopgap.** `lsmas._stopgaps.sampling.draw_with_key(law, key, sample_shape)`, the one place
  that calls `_sample`. Both uses go through it.
- **Removal.** Replace its calls with the public operation; delete the module.

---

## EK-01

**EnsKit switches JAX to 64-bit floats on import.**

- **What happens.** `import enskit` enables `jax_enable_x64`. A ProbPipe law built before that
  import and a JAX sampler run after it give a float32/float64 mismatch ("cond branches must have
  equal output types"; verified). ProbPipe itself now changes no global JAX setting (its principle
  J6), so EnsKit is the one to fix.
- **Stopgap.** `lsmas/__init__.py` imports EnsKit before anything else, so 64-bit is on everywhere.
  The forward map's declared float64 output needs this too: ProbPipe requires 64-bit mode for a
  declared float64 output.
