# Design 01 — Architecture: how the pieces fit together

*Draft 2, 2026-10-15, updated after the first implementation round (see
[`01b-implementation-report.md`](01b-implementation-report.md)) and again on 2026-11-05 for the move
to ProbPipe `main` (see [`01c-probpipe-main-report.md`](01c-probpipe-main-report.md)).*

**Changes from draft 1.** The decisions in §11 are settled. "Blocks" became **tools**, "recipes"
became **algorithms**, "observers" became **callbacks**, and `methods/` became `inference/`. Two
implementation rules were added (§2). The learning rate moved out of the likelihood and into
conditioning (§4.3), so every object denotes one mathematical thing. The ProbPipe gap register now
lives in its own file, [`probpipe-gaps.md`](probpipe-gaps.md).

**Changes for ProbPipe `main`.** The code examples use ProbPipe's new names (`&`, `rename`,
`output_name`, `traceable=`, `executor=`). EKI uses ProbPipe's public inference-method interface
(§4.5). Prior predictives come from `sample(problem.model, n)` (§4.4). ProbPipe's automatic method
selection now fails on SIPNET problems, so conditioning calls name their method (§4.4).

This document fixes the shape of the repository: its layers, the handful of objects everything else
is built from, and how SIPNET, ProbPipe and EnsKit connect. It does **not** fix the inverse problem
(parameters, prior, likelihood details, data; design 02), the algorithms (the EKI update with noise
scales, the averaged-Jacobian estimator, the reduced posterior; design 03), or the experiment harness
(metrics, ablations, plots; design 04).

Every factual claim below was checked against the pinned packages. Claims first made at ProbPipe
`638f56bc` come from the prototype in [`prototypes/01-architecture/`](prototypes/01-architecture/);
the move to `main` re-checked them with the repository's tests and with
[`prototypes/01c-probpipe-main/`](prototypes/01c-probpipe-main/). Results quoted as "verified" come
from those scripts.

---

## 1. The picture

```
 data/                     shipped files: driver, NEE features, operator A, fixed SIPNET values, prior table
   │
   ▼
 problems/                 InverseProblem = ProbPipe model + data + learning rate + coordinates + baselines
   │                         harvard_forest · harvard_forest_small · toy
   ▼
 conditioning              condition_on(model, data, method=...), with the learning rate (§4.3)
   │
   ├── ProbPipe's own inference methods (NUTS, RWMH, …)
   └── inference/  EnsKit-backed inference methods registered with ProbPipe (enskit_eki, …)
                    │  calls callbacks after every ensemble step
                    ▼
 tools/                    subspace estimators, callbacks, reduced-model builders
   │
   ▼
 algorithms/               complete parameter-reduction algorithms assembled from the above
   │
   ▼
 experiments/, notebooks/  replicate runs, metrics, plots; the walkthrough and tutorials
```

**Inference method versus algorithm.** The names are close, so to be precise: an *inference method*
(`lsmas.inference`) approximates the posterior of whatever model it is given. EKI is one. An
*algorithm* (`lsmas.algorithms`) is the object of study: it combines inference methods and tools
to calibrate with parameter reduction, and returns a posterior together with a subspace. The split
mirrors ProbPipe, whose `probpipe.inference` holds inference methods.

Two rules hold the picture together:

1. **The probability model is written once, in ProbPipe, and never edited by an experiment.**
   Every algorithm receives it through conditioning, so changing algorithms never touches the model.
2. **Algorithms are assembled, not written.** An algorithm composes an inference method, callbacks,
   a subspace estimator and a reduced-model builder. Each of those has one small interface, so a
   variant is a one-line swap.

What using the repo looks like, once built:

```python
import lsmas
from probpipe import workflow_run

problem = lsmas.problems.load("harvard_forest")          # the fixed inverse problem
estimator = lsmas.tools.LinearizationAccumulator()       # a tool, used as a callback

with workflow_run(seed=1):
    ensemble = problem.condition(                         # conditioning with the problem's learning rate
        method="enskit_eki",
        method_options={"n_particles": 100, "callbacks": [estimator]},
    )

subspace = estimator.subspace(rank=5)                    # W, eigenvalues, parameter loadings
lsmas.plots.eigenvalues(subspace)
```

---

## 2. Principles and implementation rules

- **ProbPipe-native first.** Where ProbPipe has the concept, we use it. Where it does not, we write
  the smallest stopgap that has the shape ProbPipe will eventually have.
- **The inverse problem is data.** Nothing in the repo processes raw data. The driver, features,
  operator and prior ship as files; the code only reads them.
- **One interface per tool.** A subspace estimator, a callback, an algorithm: each is one protocol of
  a few methods, documented with an example.
- **Algorithms work in whitened coordinates** (§4.1), which puts every parameter on the same footing,
  whatever its units.
- **Every stopgap is labelled and self-explaining.** Code that works around a ProbPipe gap lives
  under `lsmas/_stopgaps/` and carries a `# STOPGAP(PG-NN)` comment naming its entry in
  [`probpipe-gaps.md`](probpipe-gaps.md). Its docstring says, for a user, what ProbPipe lacks, what it
  would ideally provide, and how the code simplifies once it does. Docstrings never reference design
  documents. For example:

  ```python
  # STOPGAP(PG-03): replace calls with condition_on.using(learning_rate=...) once available.
  def tempered_condition_on(model, data, *, learning_rate, **using):
      r"""Condition a model on data with a learning rate (a generalized-Bayes update).

      ProbPipe's ``condition_on`` does not yet accept a learning rate. Ideally one would write
      ``condition_on.using(learning_rate={"nee": eta}, method=...)(model, data)``; this function
      takes the same arguments, so it can be replaced by that call directly.
      ...
      """
  ```

- **Code standards.** Code is judged by its quality, not only its outputs. No redundancy (a single
  source of truth for every quantity); docstrings written for users, without design detail or
  jargon; public API first in each file, helpers after; names that say what things are, in helpers
  too; no massive functions or classes.
- **Mathematics is stated precisely in docstrings.** Any function or class that implements a
  mathematical object or computation states it in its docstring, with notation defined, in LaTeX.
  The format follows ProbPipe's: NumPy-style sections, raw docstrings (`r"""`), `$…$` inline and
  `$$…$$` displays (MathJax via mkdocs; EnsKit's Sphinx `:math:` form is not used). A `Notes`
  section holds derivations too long for the summary. For example:

  ```python
  class TemperedLikelihood(ConditionalDistribution):
      r"""The generalized-Bayes factor of a Gaussian simulator likelihood.

      For features $i = 1, \dots, n$ with classes $k(i)$, reference variances $v_i > 0$ and
      learning rates $\eta_i \in (0, 1]$, the factor's log-density is

      $$
      \ell_\eta(\theta, s^2; y)
        = \sum_{i=1}^{n} \eta_i \log \mathcal N\!\left(y_i \,\middle|\, G_i(\theta),\; s^2_{k(i)} v_i\right),
      $$

      which is not the log-density of a law of $y$: it is used only to form a conditioning target.
      ...
      """
  ```

---

## 3. Software stack and pinning

| Package | Pin | Why this pin |
|---|---|---|
| `probpipe-core` | `TARPS-group/prob-pipe@9158a50e` (`main`, 2026-11-02) | `main` after the overhaul merged (#468); no 0.2 release yet |
| `enskit` | `TARPS-group/EnsKit@f7dc7ee` (`main`, 2026-10-10) | Includes statistical linearization (PRs #87, #95) |
| `pysipnet` | `TARPS-group/pySIPNET@08bab7b` (`main`) | SIPNET `v2.2.0-alpha.1` (`41fa853e`) via its submodule |
| `jax` | resolved by the above (0.11.2 today) | |
| Python | ≥ 3.12 | ProbPipe's floor |

All three are git dependencies pinned to **commits**, not branches, in `pyproject.toml`, with
`uv.lock` committed. A branch pin would change under us: between `638f56bc` and `9158a50e` ProbPipe
renamed much of the API this repo uses (`*` became `&`, `x.raw()` became `x.raw`, `dispatch=`
became `traceable=` and `executor=`), and its deferred plans propose more. The README carries the
upgrade recipe:

```bash
# 1. edit the commit hash in pyproject.toml, then
uv lock --upgrade-package probpipe-core
uv sync
uv run pytest -m smoke        # the fast checks of the whole stack
```

When ProbPipe is released on PyPI, the pin becomes a version range.

**One import-order rule (verified, EK-01).** Importing EnsKit switches JAX to 64-bit floats. A
ProbPipe prior built before that switch and a JAX sampler run after it fail with a float32/float64
mismatch. So `lsmas/__init__.py` imports EnsKit first, and 64-bit is on everywhere. The forward
map relies on this too, since ProbPipe requires 64-bit mode for a declared float64 output.

---

## 4. The core objects

### 4.1 Parameters and the four coordinate systems

| Coordinates | Example | Who uses it |
|---|---|---|
| **SIPNET parameters** | `aMax = 123.7`, `baseFolRespFrac = 0.085` | pySIPNET only |
| **Calibration parameters** θ, named, in physical units | `photosynthetic_capacity = 251`, `respiration_share = 0.09` | the prior, the forward map, plots |
| **Unconstrained** z = b⁻¹(θ), from each prior factor's support bijector | `log 251`, `logit 0.09` | ProbPipe samplers, EnsKit |
| **Whitened** ξ = L⁻¹(z − m), with m and LLᵀ the prior mean and covariance of z | prior ≈ N(0, I) | subspace estimators, algorithms |

- **SIPNET ↔ calibration.** One pure function `to_sipnet(theta) -> dict[str, float]` maps
  calibration parameters to pySIPNET keyword arguments. It holds the reparameterizations (for example
  `aMax = P·cFracLeaf·(1−ρ)/aMaxFrac`) and nothing else. Design 02 fills it in.
- **Calibration ↔ unconstrained.** ProbPipe already knows this map: every family declares its
  support, and `bijector_for(support)` gives exp for positive values and the logistic function for
  the unit interval. We do not write our own transforms.
- **Unconstrained ↔ whitened.** Subspaces must be computed where the prior treats all directions
  alike. Otherwise the leading eigenvector partly reports units. In unconstrained coordinates the
  prototype's leading direction is 0.995 on `log photosynthetic_capacity` (`t3_eki.py`), next to a
  raw temperature in °C with prior sd 2.5. Whitening by the prior removes that arbitrariness and
  makes "eigenvalue > 1" mean "the data are more informative than the prior in this direction". m
  and L are estimated from 50,000 prior draws with a fixed key (about 0.3% error in each standard
  deviation), whatever the prior's families.

The prior in ProbPipe is a merge (`&`) of named families, grouped under `theta/`, and the
inverse-gamma scales (verified, `t1_basics.py`; `lsmas.priors` builds the interval-specified
families):

```python
theta_prior = (
    log_normal_from_interval("photosynthetic_capacity", 140.0, 450.0)
    & logit_normal_from_interval("respiration_share", 0.04, 0.20)   # ProbPipe has no LogitNormal family
    & Normal("optimum_photosynthesis_temperature", 22.0, 2.5)
    & ...
)
prior = (
    theta_prior.rename({name: f"theta/{name}" for name in NAMES})
    & InverseGamma("s2_day", 2.0, 1.678)                   # prior median 1 (verified: 0.991 from 4000 draws)
    & InverseGamma("s2_night", 2.0, 1.678)
)
```

ProbPipe lays a law's values out as vectors publicly (`to_vector`), but its maps to unconstrained
coordinates are private, and its own helper refuses views such as `prior["theta"]` (**PG-05**).
The stopgap has two parts: `ParameterBijection.from_prior(prior, "theta")` maps between θ and z
using each parameter's declared support, and `Coordinates.from_prior(prior, "theta")` adds the
whitening (`whiten`, `unwhiten`). The problem's `coordinates` are the single source of whitening.

### 4.2 The forward map

The forward map is a ProbPipe `Function` from calibration parameters to the 400 NEE features,
G(θ) = A x(θ), where x(θ) is SIPNET's 3-hourly NEE in µmol m⁻² s⁻¹ and A is the shipped feature
operator:

```python
class _FeatureEvaluator:                         # a bound method, not a closure (PG-14)
    def features(self, theta):
        try:
            rate = self.simulator.nee_rate(self.to_sipnet(theta))
        except SIMULATION_FAILURES:              # an invalid parameter set or a crashed run
            return jnp.full(self.operator.shape[0], jnp.nan)   # EnsKit's contract (PG-07)
        return jnp.asarray(self.operator @ rate)

nee_features = Function(
    _FeatureEvaluator(simulator, A, to_sipnet).features,
    label="nee_features",
    output_spec=NumericArraySpec((400,), jnp.float64),    # lets ProbPipe call it from traced code
    traceable=False,                                       # SIPNET is an external program
    executor="thread",
    executor_options={"max_workers": 8},
)
```

Verified facts that shape this:

- **One run takes 0.3 s** for the full 2012–2024 3-hourly record (37,992 steps).
- **The time axes line up exactly.** SIPNET's output time stamps are the 37,992 step ends that A's
  columns refer to (`t1_basics.py`), so G is a single sparse matrix product.
- **The thread executor is real parallelism.** pySIPNET runs SIPNET as a subprocess, so threads do
  not contend for the GIL during a run. A batch of 32 took 2.1 s on 8 threads, against 8.6 s in
  sequence (`t2_dispatch.py`). The executor applies wherever ProbPipe evaluates the function on
  many points, including joint sampling of the model (§4.4).
- **Calling it on a batch just works.** A `NumericRecordBatch` of θ's returns a `NumericArrayBatch`
  of shape (J, 400). A distribution returns the pushforward.
- **Always load the driver with `ClimateDrivers.from_path`.** A driver held in memory is rewritten to
  disk on every run, which was about 90% of run time in the review's profile.
- **Always declare `traceable`, and set `max_evaluations` when lifting over an ensemble** (PG-10).
  Nothing is cached (PG-06): methods therefore keep every evaluation they make in their results, so
  no diagnostic re-runs SIPNET.

### 4.3 The likelihood, and where the learning rate goes

Design 02 fixes the likelihood. Here we need its shape:

  y_k | θ, s²_k ~ N(G_k(θ), s²_k v_k I),  k ∈ {day, night},  s²_k ~ IG(2, 1.678),

and the update is generalized Bayes with learning rates η_k (0.126 day, 0.137 night at c = 3; the
prototype reproduces both and the 214/186 feature split):

  π_η(θ, s² | y) ∝ π(θ, s²) ∏ₖ N(y_k | G_k(θ), s²_k v_k I)^{η_k}.

**Two objects, each one thing.** Draft 1 put η inside the likelihood. That made one object sample
one law and score another, because the tempered factor N(y | G, s²vI)^η is not the density of any
law of y. (As a function of s² it differs from N(G, s²vI/η). Inflating the covariance instead would
change the conjugate scale update from IG(a + ηn/2, ·) to IG(a + n/2, ·).) The literature treats a
power likelihood as part of the belief update, not of the data model (Bissiri, Holmes & Walker
2016). So the design keeps them apart:

1. **The model is a proper generative model.** `GaussianSimulatorLikelihood` is the law
   N(G(θ), s²vI). Its draws and its density agree (verified, `t6_tempered.py`: `log_prob` of a draw
   equals SciPy's; standardized residuals have mean 0.005, sd 1.03; and `tests/test_likelihood.py`).
   Its sampler and density are JAX code around the forward `Function`, so prior and posterior
   predictives use it directly.
2. **The learning rate belongs to conditioning.** The problem carries η, and conditioning applies
   it.

```python
likelihood = GaussianSimulatorLikelihood(          # STOPGAP(PG-02): ProbPipe has no such family yet
    nee_features,                                  # the forward Function above
    output_name="nee",
    theta_spec=THETA_SPEC,
    reference_variance=v,                          # (400,) v_day or v_night per feature
    classes=feature_class,                         # (400,) "day" / "night"
    scales={"day": "s2_day", "night": "s2_night"}, # a slot name per class, or a fixed value
)
model = likelihood & prior                         # a merge over nee, theta, s2_day, s2_night

# ideal ProbPipe (PG-03):  condition_on.using(learning_rate={"nee": eta}, method=...)(model, data)
# today:
posterior = tempered_condition_on(
    model, data, learning_rate={"nee": eta}, method="enskit_eki", method_options={...}
)
```

`tempered_condition_on` has the signature the ideal call would have, so removing it is a
find-and-replace. Inside, it swaps the likelihood factor for a private `TemperedLikelihood` that
claims only the unnormalized density Σᵢ ηᵢ log N(yᵢ | …) and cannot be sampled. It then calls
`condition_on`. Verified (`t6_tempered.py`): the target's log-density differences between two
parameter settings equal those of log π + Σᵢ ηᵢ log Nᵢ exactly (0.386083 both ways), and with η = 1
they equal the ordinary posterior's (8.239780 both ways). EKI and ProbPipe's RWMH both run on it,
and on the linear-Gaussian test model EKI and ProbPipe's NUTS both recover the exact tempered
posterior (`tests/test_enskit_eki.py`, `tests/test_tempering.py`).

The tempered object never leaves the function, for a reason worth fixing in ProbPipe (PG-12). For a
kernel, "unnormalized" could allow a normalizer that depends on the given values. Inference ignores
such a normalizer while sampling respects it, so an object like this would mean different things to
different operations.

`InverseProblem.condition(method=..., **using)` (§4.9) calls `tempered_condition_on` with the problem's η, so
algorithms never handle η themselves.

**Why one likelihood factor, not one per class.** A factor per class (`nee_day`, `nee_night`) would
make η a scalar per factor. Without evaluation reuse (PG-06), however, each factor would run SIPNET
separately, doubling the cost. So η is an array over the one factor's 400 features. The ideal
`learning_rate` should accept either form.

### 4.4 What ProbPipe does with the model

- **Name the method.** `condition_on.check(model, data).method` is `'blackjax_nuts'`: the density
  is JAX code that traces, with SIPNET running through a host callback, and ProbPipe's selection
  does not ask whether gradients exist. NUTS then fails, because gradients cannot pass through
  SIPNET (verified; **PG-08**). `InverseProblem.condition` therefore requires a method.
- **RWMH runs, one point per chain at a time.** 2 chains × 30 steps: 67 runs, 24 s, whether or not
  the call names the thread executor (verified; PG-07). A 100,000-run reference chain would take
  about 8 hours. Design 04 decides how reference posteriors are produced.
- **Prior predictives go through `sample(model)`.** `sample(problem.model, n)` draws the parameters
  and then the observations, running SIPNET on the forward map's threads: 16 draws in about 2 s
  (verified). This needed ProbPipe `main`, which calls an opaque `Function` from traced code
  (PG-01, closed).

### 4.5 Inference methods: the EnsKit seam

EnsKit algorithms are registered as ProbPipe inference methods, following the pattern of ProbPipe's
own `pyabc_smcabc`:

```python
class EnsKitEKI(InferenceMethod):
    _method_options = ("n_particles", "ess_fraction", "callbacks", "max_steps")
    name = "enskit_eki"                                       # priority None: runs only when named

    def supported_types(self):
        return (Distribution,)                                # every target is a law

    def check(self, target, /, **options):
        factors = target.factors()                            # prior, likelihood, observed value
        # accepts a GaussianSimulatorLikelihood, or the tempered factor wrapping one (reads its eta)
        ...

    def execute(self, target, /, *, key, **options):         # condition_on passes the run's key
        # 1. draw the initial ensemble from the prior with key; move it to unconstrained coordinates
        # 2. run EnsKit's tempered EKI, evaluating the forward Function on the whole ensemble at once
        # 3. call each callback after every evaluation
        # 4. return make_posterior(target, [vectors], method=..., annotations=...): an
        #    EmpiricalDistribution over theta, with every step's ensemble and predictions annotated
        ...

inference_method_registry.register(EnsKitEKI())
```

Verified end to end on the development problem, at ProbPipe `main`:

```python
problem = load("harvard_forest_dev")                # from lsmas.problems
with workflow_run(seed=1):
    post = problem.condition(method="enskit_eki",
                             method_options={"n_particles": 40, "callbacks": [records.append]})
# 5 adaptive tempering steps (β = 0 → 0.046 → 0.189 → 0.452 → 0.947 → 1), 200 SIPNET runs, 17 s
# post: EmpiricalDistribution, provenance method 'enskit_eki', exact False
# post.annotations["arviz/eki_history"] holds every step's ensemble and predictions
```

Design points:

- **EnsKit's driver takes a plain ensemble.** `eki.EKIState(Ensemble(u=draws), key=...)` accepts
  prior draws directly, so ProbPipe owns the prior and EnsKit sees only particles.
- **The batch goes through ProbPipe.** The method evaluates the ensemble by calling the likelihood's
  forward `Function` on a `NumericRecordBatch`, never SIPNET directly. Whatever ProbPipe adds to
  `Function` calls (reuse, backends; #580) then applies to EKI for free.
- **Step-level API.** With fixed noise scales the method runs EnsKit's `eki.iterate`, which yields
  after every evaluation. EKI with inferred scales changes the noise covariance between steps,
  which `eki.run` cannot do, so that method will drive `eki.evaluate`, `schedule.next_increment` and
  `eki.assimilate` itself (verified). Design 03 specifies the update.
- **Named only.** Priority `None`: automatic selection never picks it, and experiments name their
  method explicitly.
- **Upstream path.** If ProbPipe gains native ensemble Kalman updates (#299), the method name and
  options stay and only `execute` changes.
- **Public ProbPipe names only.** `InferenceTarget`, `target.factors()`, `make_posterior` and the
  `key` argument are public at `main` (PG-04, closed). The one private call left is the keyed draw
  of the initial ensemble, made through the stopgap `draw_with_key` (PG-15), as ProbPipe's own
  methods call `prior._sample(key, shape)`.

### 4.6 Callbacks: how tools see a run

A **callback** is any callable that an inference method calls after each ensemble evaluation with one
`StepRecord`. It reads the step and must not change the run.

```python
@dataclass(frozen=True)
class StepRecord:
    step: int                 # 0, 1, 2, …
    beta: float               # tempering level before this step's update
    increment: float          # Δβ the step then took; the last one reaches β = 1
    parameter_names: tuple    # the order of theta's last axis
    theta: Array              # (J, d) particles, constrained coordinates
    predictions: Array        # (J, n) G at the particles
    noise_variance: Array     # (n,) the diagonal of the noise covariance at β = 1
```

The record carries θ rather than whitened ξ. Ensemble Kalman updates are unchanged by an affine
change of coordinates, so the method works in unconstrained coordinates, and a tool whitens with
`problem.coordinates.whiten(problem.coordinates.bijection.to_unconstrained(record.theta))`. That
keeps one source of whitening, the problem's. Noise-scale fields join the record when EKI infers
the scales (design 03).

Every callback reads the same record, so tools stay decoupled from the methods that feed them. A
statistical-linearization accumulator, for example:

```python
class LinearizationAccumulator:
    def __call__(self, record):
        xi = self.coordinates.whiten(self.coordinates.bijection.to_unconstrained(record.theta))
        A_t = statistical_linearization(xi, record.predictions)          # EnsKit, via a thin wrapper
        self.H += weight(record) * A_t.T @ (A_t / record.noise_variance[:, None])

    def subspace(self, rank):
        return Subspace.from_matrix(self.H, rank=rank, names=..., estimator="slr-accumulated")
```

Verified with EnsKit's `maps.statistical_linearization` inside `t3_eki.py`: on the 4-parameter
prototype the accumulated matrix has eigenvalues 323, 24.1, 5.05, 0.06. That shows the plumbing
works, not that the estimator is right. **What** to accumulate (the weights, which average of
Jacobians it estimates, and its bias) is design 03's central question.

Methods also return the full step history, so anything a callback could compute can be computed
afterwards too.

### 4.7 Subspace

```python
@dataclass(frozen=True)
class Subspace:
    W: Array                 # (d, r) orthonormal columns, whitened coordinates
    eigenvalues: Array       # (d,) all of them, for spectra and rank choice
    parameter_names: tuple[str, ...]
    estimator: str           # e.g. "slr-accumulated"
    diagnostics: dict        # anything estimator-specific (bootstrap intervals, …)

    def active(self, xi): ...            # Wᵀ ξ
    def inactive_basis(self): ...        # an orthonormal complement W⊥
    def loadings(self): ...              # W mapped back to named parameters, for plots
    def activity_scores(self): ...       # Constantine & Diaz per-parameter importance
```

`W` is an array for now and becomes a ProbPipe `LinOp` when that lands (PG-09), without changing the
interface. Every estimator, whether gradient-based, linearization-based or covariance-based, returns
this one type.

### 4.8 Algorithms

An algorithm is a function from a problem and a budget to a result:

```python
def as_eki(problem, *, n_particles, rank, seed) -> AlgorithmResult:
    """Active-subspace EKI: explore with EKI, estimate W along the way, then infer in the subspace."""
    estimator = LinearizationAccumulator()
    with workflow_run(seed=seed):
        explore = problem.condition(method="enskit_eki",
            method_options={"n_particles": n_particles, "callbacks": [estimator]})
    W = estimator.subspace(rank)
    reduced = reduced_problem(problem, W)            # design 03
    with workflow_run(seed=seed + 1):
        posterior = reduced.condition(method=...)
    return AlgorithmResult(posterior=posterior, subspace=W, cost=..., history=explore)
```

`AlgorithmResult` (posterior, subspace, cost in SIPNET runs, history) is the standard form the
experiment harness compares (design 04). How to define the reduced posterior, and how to express it
in ProbPipe, is design 03's question. The natural form, ξ = W a + W⊥ b with a prior on (a, b), needs
linear pushforwards of laws (PG-09). Until then, reduced models are built from a reparameterized
forward `Function`, which ProbPipe already supports.

### 4.9 InverseProblem

```python
@dataclass(frozen=True)
class InverseProblem:
    name: str
    description: str                 # one paragraph, shown by problem.describe()
    likelihood: ConditionalDistribution
    prior: Distribution
    data: dict[str, Array]           # {"nee": y}
    learning_rate: dict[str, Array]  # {"nee": eta}, (400,); all ones for an ordinary posterior
    coordinates: Coordinates         # §4.1
    # later: baselines, loaded lazily from baselines/ (design 04)

    forward = property(...)          # the likelihood's forward Function, θ -> features
    model = cached_property(...)     # likelihood & prior: a proper generative model (ProbPipe)

    def condition(self, *, method, **using) -> Distribution:
        """The (generalized) posterior, by the named inference method (required: PG-08)."""
        return tempered_condition_on(self.model, self.data, learning_rate=self.learning_rate,
                                     method=method, **using)
```

The problem stores the likelihood and the prior and derives the model, because a model's factors
cannot yet be split back out reliably (PG-13).

Three problems share the interface, so every algorithm runs on all of them:

| Problem | What it is for | Baseline |
|---|---|---|
| `harvard_forest` | the real target: all calibration parameters | the best affordable reference (design 04) |
| `harvard_forest_small` | the same model and data with most parameters fixed | long MCMC: a posterior we can trust |
| `toy` | a JAX-differentiable model of similar structure | NUTS via plain conditioning, and exact Jacobians to check averaged-gradient estimates (PG-08) |

Baselines are precomputed and shipped (format: design 04), never recomputed by a user.

---

## 5. Repository layout

```
lsm-active-subspace/
├── README.md                      setup, upgrade recipe, map of tutorials
├── pyproject.toml, uv.lock        commit-pinned ProbPipe, EnsKit, pySIPNET
├── data/harvard_forest/
│   ├── PROVENANCE.md              where every file came from, and how it was made (outside this repo)
│   ├── driver/                    corrected ERA5 member(s), .clim + provenance.json
│   ├── nee_features.csv           y, class, year, month, half, n_steps per feature (400 rows)
│   ├── nee_operator.npz           A, sparse 400 × 37,992
│   ├── sipnet_fixed.csv           fixed SIPNET values, each with a justification
│   └── prior.csv                  calibration parameters: family, hyperparameters, units, source
├── baselines/                     precomputed reference posteriors, one file per problem
├── src/lsmas/
│   ├── __init__.py                imports EnsKit first (EK-01), then the public API
│   ├── sipnet.py                  the SIPNET runner and the forward Function
│   ├── parameters.py              to_sipnet: calibration -> SIPNET parameters
│   ├── problems/                  InverseProblem and the three problems
│   ├── inference/                 inference methods backed by EnsKit, registered with ProbPipe
│   ├── tools/                     Subspace, subspace estimators, callbacks (StepRecord), reduced-model builders
│   ├── algorithms/                complete parameter-reduction algorithms
│   ├── experiments/               replicate runner, metrics (design 04)
│   ├── plots/                     standard figures
│   └── _stopgaps/                 every ProbPipe workaround; each names its PG id, the gap, the ideal, the removal
├── notebooks/
│   ├── 00_the_inverse_problem.ipynb    walkthrough: model, data, likelihood, prior, sample(problem.model)
│   └── tutorials/                      running SIPNET; ProbPipe models; EKI; subspaces; algorithms
├── design/                        these documents, the gap registry, and the prototypes
└── tests/                         unit tests, plus `-m smoke` end-to-end checks
```

A reader who builds algorithms touches `algorithms/`, `tools/` and the notebooks. `inference/` and
`_stopgaps/` are plumbing they never need to open.

---

## 6. What happens in one EKI call, end to end

1. `problem.condition(method="enskit_eki", …)` calls the conditioning stopgap with the problem's η,
   which forms the target (the model conditioned on `nee = y`, likelihood tempered by η) and asks
   ProbPipe's registry for `enskit_eki`.
2. `check` finds the prior, the likelihood and y, confirms the likelihood carries the structure EKI
   needs, and reads η.
3. `execute` draws J prior samples with the run's key, which `condition_on` derives from the
   workflow's seed, and maps θ to unconstrained coordinates.
4. Each step evaluates the forward `Function` on the J particles (threads, 8 at a time), calls the
   callbacks, picks the next Δβ, applies the Kalman update and refreshes the scales.
5. At β = 1 it maps the particles back to θ and returns an `EmpiricalDistribution` whose provenance
   records the method, its options and its parents, together with the step history.
6. The caller reads the result with ordinary ProbPipe operations (`mean`, `quantile`,
   `post["theta"]`), and every callback holds what it accumulated.

---

## 7. Cost, today

| Operation | Cost (verified on this machine) |
|---|---|
| One SIPNET run, 2012–2024 | 0.3 s |
| 32 runs, thread executor (8 workers) | 2.1 s |
| `sample(problem.model, 16)` | 16 runs, 2.1 s |
| EKI, J = 40, 4 parameters, to β = 1 | 200 runs, 17 s |
| RWMH, 2 chains × 30 steps | 67 runs, 24 s (one point per chain at a time) |

EKI on the full problem (J ≈ 100, perhaps 5–10 steps) should take a minute or two. Reference
posteriors by MCMC are the expensive part, which is why baselines are precomputed.

---

## 8. Testing

- **Unit tests** for each piece (coordinates round-trip, `to_sipnet` against known values, subspace
  algebra, callbacks on synthetic records), validated against independent baselines as ProbPipe's
  style guide asks (analytic results, not the code under test).
- **Smoke tests** (`pytest -m smoke`, under a minute today): one SIPNET run aligned with A; draws
  of the model; one short `enskit_eki` run on each problem; one algorithm end to end. These are
  what an upgrade of ProbPipe, EnsKit or pySIPNET must pass.
- **Repository tests** (`tests/test_repository.py`). Every `STOPGAP(PG-NN)` tag names an entry in
  `probpipe-gaps.md`, every stopgap module carries a tag, no module imports a private ProbPipe
  module, and every `Function` states whether JAX can trace it (PG-10).

---

## 9. ProbPipe gaps

The register is [`probpipe-gaps.md`](probpipe-gaps.md). It has fifteen entries (PG-01 to PG-15)
plus one for EnsKit (EK-01). Each entry gives what happens (with evidence), the effect here, the ideal
ProbPipe support, the stopgap, how to remove it, and the upstream issue or plan. At ProbPipe `main`,
two are closed: joint sampling of a model whose likelihood runs SIPNET (PG-01) and the public
inference-method interface (PG-04). Three are partly fixed (PG-05, PG-07, PG-10), and three more
are designed in ProbPipe issues or plans (PG-06, PG-08, PG-09). Seven have no upstream issue yet:
the simulator-Gaussian family (PG-02), the learning rate on conditioning (PG-03), the protocol
footgun (PG-11), the semantics of an unnormalized conditional density (PG-12), and three found while
implementing: renamed factors losing their path names (PG-13), fingerprinting recursing on sparse
matrices (PG-14), and the missing keyed draw (PG-15).

---

## 10. Open questions for design 02 onward

None block this design. Recorded so they are not lost:

- **Reference posterior for the full problem** (design 04): RWMH evaluates one point per chain at a
  time through ProbPipe today (PG-07), so the full-problem baseline may need many chains run in
  parallel, MCMC run outside ProbPipe, or a smaller claim than "ground truth". ProbPipe's executors
  now let a user register a cluster executor (a GridEngine one, say), which helps any method that
  hands the executor many points at once, as EKI does.
- **The reduced model's ProbPipe form** (design 03): depends on PG-09.

## 11. Decisions

Settled 2026-10-15:

1. The package is `lsmas`.
2. EnsKit-backed inference methods live in this repo (`lsmas.inference`), written so they could move
   to an adapter package or into ProbPipe without changes for users.
3. They run only when named.
4. Methods both call callbacks and return the full step history.
5. Subspaces use coordinates whitened by the prior's mean and covariance in unconstrained space.

---

## Appendix: what the prototypes verified

At ProbPipe `638f56bc`, by [`prototypes/01-architecture/`](prototypes/01-architecture/). Rows marked
† changed at `main`; the table after this one gives the new results.

| Claim | Script | Result |
|---|---|---|
| η from c = 3 | `t1_basics.py` | 0.126 day, 0.137 night; 214 + 186 features |
| A's columns = SIPNET's time axis | `t1_basics.py` | equal (37,992 steps, 2012-01-01T00 to 2024-12-31T21) |
| IG(2, 1.678) median ≈ 1 | `t1_basics.py` | 0.991 (4000 draws) |
| Model construction runs no simulation | `t1_basics.py` | 0 runs |
| Auto-selected method † | `t2_dispatch.py` | `blackjax_rwmh` |
| `sample(model)` fails † | `t2_dispatch.py`, `t4_joint.py` | `TracerArrayConversionError`, both likelihood styles |
| Batch call, thread vs sequential | `t2_dispatch.py` | 32 runs: 2.1 s vs 8.6 s |
| Pushforward of the prior | `t2_dispatch.py` | `EmpiricalDistribution` via `sampling_lift` |
| No caching | `t2_dispatch.py` | an identical batch re-ran 32 simulations |
| The likelihood is one law | `t6_tempered.py` | `log_prob` of a draw equals SciPy's; residuals mean 0.005, sd 1.03 |
| Tempered target density | `t6_tempered.py` | differences equal log π + Σ ηᵢ log Nᵢ (0.386083); plain: 8.239780 |
| `enskit_eki` through tempered conditioning † | `t3_eki.py`, `t6_tempered.py` | 4 steps, 160 runs, 16 s, `EmpiricalDistribution` |
| RWMH through tempered conditioning † | `t5_rwmh.py`, `t6_tempered.py` | runs (67 runs, 26 s); fails if EnsKit is imported after the prior is built |
| Linearization callback | `t3_eki.py` | accumulated eigenvalues 323, 24.1, 5.05, 0.06 |
| Laws are immutable | (session check) | a memo on a law raises `AttributeError` |
| Both method names needed for a density claim | (session check, PG-11) | `_log_prob` alone: "does not implement SupportsLogProb" |
| Corrected driver passes pySIPNET's strict checks | (session check) | `ClimateDrivers.from_file` OK; 0.29 s/run |

At ProbPipe `main` (`9158a50e`), by the repository's tests, by
[`prototypes/01c-probpipe-main/gaps_at_main.py`](prototypes/01c-probpipe-main/gaps_at_main.py), and by
the timings in [`01c-probpipe-main-report.md`](01c-probpipe-main-report.md):

| Claim | Evidence | Result |
|---|---|---|
| Auto-selected method | `gaps_at_main.py` (PG-08) | `blackjax_nuts`, which fails: no gradients through SIPNET |
| `sample(model)` works | `tests/test_likelihood.py`, `tests/test_problems.py` | draws follow the likelihood; 16 SIPNET draws in 2.1 s |
| `enskit_eki` on the development problem | report timing | 5 steps, 200 runs, 17 s, no failed runs |
| RWMH on the development problem | report timing | 67 runs, 24 s; 22 s with the thread executor named |
| NUTS on the tempered linear model | `tests/test_tempering.py` | recovers the exact tempered posterior |
| EnsKit import order | `gaps_at_main.py` (EK-01) | unchanged: still fails if EnsKit is imported after the prior is built |
