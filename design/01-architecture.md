# Design 01 — Architecture: how the pieces fit together

*Draft 2, 2026-10-15. Status: for discussion.*

**Changes from draft 1.** The decisions in §11 are settled. "Blocks" became **tools**, "recipes"
became **algorithms**, "observers" became **callbacks**, and `methods/` became `inference/`. Two
implementation rules were added (§2). The learning rate moved out of the likelihood and into
conditioning (§4.3), so every object denotes one mathematical thing. The ProbPipe gap register now
lives in its own file, [`probpipe-gaps.md`](probpipe-gaps.md).

This document fixes the shape of the repository: its layers, the handful of objects everything else
is built from, and how SIPNET, ProbPipe and EnsKit connect. It does **not** fix the inverse problem
(parameters, prior, likelihood details, data; design 02), the algorithms (the EKI update with noise
scales, the averaged-Jacobian estimator, the reduced posterior; design 03), or the experiment harness
(metrics, ablations, plots; design 04).

Every factual claim below was checked against the pinned packages with the prototype in
[`prototypes/01-architecture/`](prototypes/01-architecture/). Results quoted as "verified" come from
those scripts.

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
| `probpipe-core` | `TARPS-group/prob-pipe@638f56bc` (`dev/overhaul`, 2026-10-14) | Current tip of the overhaul branch |
| `enskit` | `TARPS-group/EnsKit@f7dc7ee` (`main`, 2026-10-10) | Includes statistical linearization (PRs #87, #95) |
| `pysipnet` | `TARPS-group/pySIPNET@08bab7b` (`main`) | SIPNET `v2.2.0-alpha.1` (`41fa853e`) via its submodule |
| `jax` | resolved by the above (0.11.2 today) | |
| Python | ≥ 3.12 | ProbPipe's floor |

All three are git dependencies pinned to **commits**, not branches, in `pyproject.toml`, with
`uv.lock` committed. A branch pin would change under us: `dev/overhaul` merged more than 40 PRs in the
week before this draft, and two pending PRs rename public API (`x.raw()` becomes `x.raw`, #575;
summary field names change, #576). The README carries the upgrade recipe:

```bash
# 1. edit the commit hash in pyproject.toml, then
uv lock --upgrade-package probpipe-core
uv sync
uv run pytest -m smoke        # the fast checks of the whole stack
```

When ProbPipe is released on PyPI, the pin becomes a version range.

> **Note.** The local clone at `~/Desktop/git-repos/prob-pipe-overhaul` is 248 commits behind
> `origin/dev/overhaul`. The prototype installs from GitHub, not from that clone.

**One import-order rule (verified, EK-01).** Importing EnsKit switches JAX to 64-bit floats. A
ProbPipe prior built before that switch and a JAX sampler run after it fail with a float32/float64
mismatch. So `lsmas/__init__.py` imports EnsKit first, and 64-bit is on everywhere.

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
  and L are exact for normal, log-normal and logit-normal factors (all Gaussian in z) and are moment
  estimates otherwise.

The prior in ProbPipe is a product of named families, grouped under `theta/`, times the
inverse-gamma scales (verified, `t1_basics.py`):

```python
theta_prior = (
    LogNormal("photosynthetic_capacity", loc, scale)
    * BijectorTransformedDistribution(                      # logit-normal; ProbPipe has no LogitNormal family
        "respiration_share", Normal("z", m, s), bijector_for(unit_interval))
    * Normal("optimum_photosynthesis_temperature", 22.0, 2.5)
    * ...
)
prior = (
    theta_prior.with_path_names({name: f"theta/{name}" for name in NAMES})
    * InverseGamma("s2_day", 2.0, 1.678)                   # prior median 1 (verified: 0.991 from 4000 draws)
    * InverseGamma("s2_night", 2.0, 1.678)
)
```

ProbPipe's coordinate helpers are private and refuse views such as `prior["theta"]` (**PG-05**). The
stopgap is one class, `Coordinates.from_prior(prior, block="theta")`, with `names`,
`to_unconstrained`, `to_constrained`, `whiten` and `unwhiten`.

### 4.2 The forward map

The forward map is a ProbPipe `Function` from calibration parameters to the 400 NEE features,
G(θ) = A x(θ), where x(θ) is SIPNET's 3-hourly NEE in µmol m⁻² s⁻¹ and A is the shipped feature
operator:

```python
@function(dispatch="thread", max_workers=8, label="nee_features")
def nee_features(theta):
    try:
        x = sipnet_model(**FIXED, **to_sipnet(theta)).outputs["nee"].values * STEP_TOTAL_TO_RATE
        return jnp.asarray(A @ x)
    except SIPNET_FAILURES:                      # an invalid parameter set or a crashed run
        return jnp.full(A.shape[0], jnp.nan)     # EnsKit's contract: a failed particle is a NaN row (PG-07)
```

Verified facts that shape this:

- **One run takes 0.3 s** for the full 2012–2024 3-hourly record (37,992 steps).
- **The time axes line up exactly.** SIPNET's output time stamps are the 37,992 step ends that A's
  columns refer to (`t1_basics.py`), so G is a single sparse matrix product.
- **Thread dispatch is real parallelism.** pySIPNET runs SIPNET as a subprocess, so threads do not
  contend for the GIL during a run. A batch of 32 took 2.1 s with `dispatch="thread",
  max_workers=8`, against 8.6 s with `"sequential"` (`t2_dispatch.py`).
- **Calling it on a batch just works.** A `NumericRecordBatch` of θ's returns a `NumericArrayBatch`
  of shape (J, 400). A distribution returns the pushforward.
- **Always load the driver with `ClimateDrivers.from_path`.** A driver held in memory is rewritten to
  disk on every run, which was about 90% of run time in the review's profile.
- **Always set `dispatch` and `n_broadcast_samples`** (PG-10). Nothing is cached (PG-06): methods
  therefore keep every evaluation they make in their results, so no diagnostic re-runs SIPNET.

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
   equals SciPy's; standardized residuals have mean 0.005, sd 1.03). Prior and posterior
   predictives use it directly.
2. **The learning rate belongs to conditioning.** The problem carries η, and conditioning applies
   it.

```python
likelihood = GaussianSimulatorLikelihood(          # STOPGAP(PG-02): ProbPipe has no such family yet
    nee_features,                                  # the forward Function above
    component="nee",
    theta_spec=THETA_SPEC,
    reference_variance=v,                          # (400,) v_day or v_night per feature
    classes=feature_class,                         # (400,) "day" / "night"
    scale_slots={"day": "s2_day", "night": "s2_night"},
)
model = likelihood * prior                         # a FactoredDistribution over nee, theta, s2_day, s2_night

# ideal ProbPipe (PG-03):  condition_on.using(learning_rate={"nee": eta}, method=...)(model, data)
# today:
posterior = tempered_condition_on(model, data, learning_rate=eta, method="enskit_eki", method_options={...})
```

`tempered_condition_on` has the signature the ideal call would have, so removing it is a
find-and-replace. Inside, it swaps the likelihood factor for a private `TemperedLikelihood` that
claims only the unnormalized density Σᵢ ηᵢ log N(yᵢ | …) and cannot be sampled. It then calls
`condition_on`. Verified (`t6_tempered.py`): the target's log-density differences between two
parameter settings equal those of log π + Σᵢ ηᵢ log Nᵢ exactly (0.386083 both ways), and with η = 1
they equal the ordinary posterior's (8.239780 both ways). EKI and ProbPipe's RWMH both run on it.

The tempered object never leaves the function, for a reason worth fixing in ProbPipe (PG-12). For a
kernel, "unnormalized" could allow a normalizer that depends on the given values. Inference ignores
such a normalizer while sampling respects it, so an object like this would mean different things to
different operations.

`InverseProblem.condition(**using)` (§4.9) calls `tempered_condition_on` with the problem's η, so
algorithms never handle η themselves.

**Why one likelihood factor, not one per class.** A factor per class (`nee_day`, `nee_night`) would
make η a scalar per factor. Without evaluation reuse (PG-06), however, each factor would run SIPNET
separately, doubling the cost. So η is an array over the one factor's 400 features. The ideal
`learning_rate` should accept either form.

### 4.4 What ProbPipe does with the model

- **Automatic selection.** `condition_on.check(model, data).method` is `'blackjax_rwmh'`, because
  the model has a density but is not JAX-traceable. RWMH runs (2 chains × 30 steps: 67 runs, 26 s,
  `t5_rwmh.py`) but sequentially (PG-07), so a 100,000-run reference chain would take about 8 hours.
  Design 04 decides how reference posteriors are produced.
- **Prior predictives do not go through `sample(model)`.** Joint sampling vmaps every kernel, which
  cannot run SIPNET (verified for both ways of building the likelihood, `t2_dispatch.py`,
  `t4_joint.py`; **PG-01**). The stopgap, `lsmas.predictive.prior_predictive(problem, n)`, samples
  the prior, calls the forward `Function` on the θ batch and adds the likelihood's noise. Sampling the
  likelihood at fixed parameters works today (`sample(condition_on(likelihood, given))`, verified).

### 4.5 Inference methods: the EnsKit seam

EnsKit algorithms are registered as ProbPipe inference methods, following the pattern of ProbPipe's
own `pyabc_smcabc`:

```python
class EnsKitEKI(InferenceMethod):
    _method_options = ("n_particles", "ess_fraction", "callbacks", "max_steps")
    name = "enskit_eki"                                       # priority None: runs only when named

    def supported_types(self):
        return (UnnormalizedConditional,)                     # STOPGAP(PG-04): private in ProbPipe

    def check(self, target, /, **options):
        factors = model_factors(target)                       # prior, likelihood, observed value
        # accepts a GaussianSimulatorLikelihood, or the tempered factor wrapping one (reads its eta)
        ...

    def execute(self, target, /, **options):
        # 1. draw the initial ensemble from the prior; move it to whitened coordinates
        # 2. run EnsKit's tempered EKI, evaluating the forward Function on the whole ensemble at once
        # 3. call each callback after every evaluation
        # 4. return make_posterior(...): an EmpiricalDistribution over theta and the scales,
        #    keeping every step's ensemble and predictions in the result
        ...

inference_method_registry.register(EnsKitEKI())
```

Verified end to end (`t3_eki.py`, `t6_tempered.py`):

```python
with workflow_run(seed=7):
    post = tempered_condition_on(model, data, learning_rate=eta, method="enskit_eki",
                                 method_options={"n_particles": 40, "callbacks": [accumulator]})
# 4 adaptive tempering steps (β = 0 → 0.062 → 0.319 → 0.990 → 1), 160 SIPNET runs, 16 s
# post: EmpiricalDistribution, provenance method 'enskit_eki', exact False
# mean(post["theta"]), mean(post["s2_day"]), post["theta/soil_respiration_q10"] all work
```

Design points:

- **EnsKit's driver takes a plain ensemble.** `eki.EKIState(Ensemble(u=draws), key=...)` accepts
  prior draws directly, so ProbPipe owns the prior and EnsKit sees only particles.
- **The batch goes through ProbPipe.** The method evaluates the ensemble by calling the likelihood's
  forward `Function` on a `NumericRecordBatch`, never SIPNET directly. Whatever ProbPipe adds to
  `Function` calls (reuse, backends; #580) then applies to EKI for free.
- **Step-level API.** EKI with noise scales changes the noise covariance between steps, which
  `eki.run` cannot do, so the method drives `eki.evaluate`, `schedule.next_increment` and
  `eki.assimilate` itself (verified). Design 03 specifies the update.
- **Named only.** Priority `None`: automatic selection never picks it, and experiments name their
  method explicitly.
- **Upstream path.** If ProbPipe gains native ensemble Kalman updates (#299), the method name and
  options stay and only `execute` changes.
- **Private ProbPipe names** (`UnnormalizedConditional`, `model_factors`, `make_posterior`,
  `run_seed`) come from one re-export module (PG-04), designed upstream in #580 §9.

### 4.6 Callbacks: how tools see a run

A **callback** is any callable that an inference method calls after each ensemble evaluation with one
`StepRecord`. It reads the step and must not change the run.

```python
@dataclass(frozen=True)
class StepRecord:
    step: int                 # 0, 1, 2, …
    beta: float               # tempering level before this step's update
    increment: float | None   # Δβ the schedule chose; None after the last evaluation
    xi: Array                 # (J, d) particles, whitened coordinates
    predictions: Array        # (J, n) G at the particles
    noise_variance: Array     # (n,) the step's noise variances (diagonal)
    scales: dict[str, Array]  # each scale slot's (J,) values
```

Every callback reads the same record, so tools stay decoupled from the methods that feed them. A
statistical-linearization accumulator, for example:

```python
class LinearizationAccumulator:
    def __call__(self, record):
        A_t = statistical_linearization(record.xi, record.predictions)   # EnsKit, via a thin wrapper
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
    model: Distribution              # likelihood * prior: a proper generative model (ProbPipe)
    data: dict[str, Array]           # {"nee": y}
    learning_rate: dict[str, Array]  # {"nee": eta}, (400,); all ones for an ordinary posterior
    coordinates: Coordinates         # §4.1
    forward: Function                # θ -> features
    baselines: Mapping[str, Callable[[], Distribution]]   # loaded lazily from baselines/
    description: str                 # one paragraph, shown by problem.describe()

    def condition(self, **using) -> Distribution:
        """The (generalized) posterior, by the inference method `using` names."""
        return tempered_condition_on(self.model, self.data, learning_rate=self.learning_rate["nee"], **using)
```

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
│   ├── predictive.py              prior and posterior predictive helpers
│   ├── experiments/               replicate runner, metrics (design 04)
│   ├── plots/                     standard figures
│   └── _stopgaps/                 every ProbPipe workaround; each names its PG id, the gap, the ideal, the removal
├── notebooks/
│   ├── 00_the_inverse_problem.ipynb    walkthrough: model, data, likelihood, prior, prior predictive
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
3. `execute` draws J prior samples under the workflow's seed and maps θ to whitened coordinates.
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
| 32 runs, thread dispatch (8 workers) | 2.1 s |
| EKI, J = 40, 4 parameters, to β = 1 | 160 runs, 16 s |
| RWMH, 2 chains × 30 steps | 67 runs, 26 s (sequential) |

EKI on the full problem (J ≈ 100, perhaps 5–10 steps) should take a minute or two. Reference
posteriors by MCMC are the expensive part, which is why baselines are precomputed.

---

## 8. Testing

- **Unit tests** for each piece (coordinates round-trip, `to_sipnet` against known values, subspace
  algebra, callbacks on synthetic records), validated against independent baselines as ProbPipe's
  style guide asks (analytic results, not the code under test).
- **Smoke tests** (`pytest -m smoke`, a few minutes): one SIPNET run aligned with A; prior
  predictive; one short `enskit_eki` run on each problem; one algorithm end to end. These are what an
  upgrade of ProbPipe, EnsKit or pySIPNET must pass.
- **Inventory tests.** Every `STOPGAP(PG-NN)` tag names an entry in `probpipe-gaps.md`, and every
  open entry is referenced by at least one tag. Every `@function` in `lsmas` sets `dispatch` (PG-10).

---

## 9. ProbPipe gaps

The register is [`probpipe-gaps.md`](probpipe-gaps.md). It has twelve entries (PG-01 to PG-12) plus
one for EnsKit (EK-01). Each entry gives what happens (with evidence), the effect here, the ideal
ProbPipe support, the stopgap, how to remove it, and the upstream issue. Seven are covered by open upstream
issues (#580, #521, #522, #352). Five are not: the simulator-Gaussian family (PG-02), the learning
rate on conditioning (PG-03), public coordinates (PG-05, only partly covered by #522), the protocol
footgun (PG-11), and the semantics of an unnormalized conditional density (PG-12).

---

## 10. Open questions for design 02 onward

None block this design. Recorded so they are not lost:

- **Reference posterior for the full problem** (design 04): RWMH is sequential through ProbPipe today
  (PG-07), so the full-problem baseline may need EnsKit-free MCMC run outside ProbPipe, or a smaller
  claim than "ground truth".
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

## Appendix: what the prototype verified

| Claim | Script | Result |
|---|---|---|
| η from c = 3 | `t1_basics.py` | 0.126 day, 0.137 night; 214 + 186 features |
| A's columns = SIPNET's time axis | `t1_basics.py` | equal (37,992 steps, 2012-01-01T00 to 2024-12-31T21) |
| IG(2, 1.678) median ≈ 1 | `t1_basics.py` | 0.991 (4000 draws) |
| Model construction runs no simulation | `t1_basics.py` | 0 runs |
| Auto-selected method | `t2_dispatch.py` | `blackjax_rwmh` |
| `sample(model)` fails | `t2_dispatch.py`, `t4_joint.py` | `TracerArrayConversionError`, both likelihood styles |
| Batch call, thread vs sequential | `t2_dispatch.py` | 32 runs: 2.1 s vs 8.6 s |
| Pushforward of the prior | `t2_dispatch.py` | `EmpiricalDistribution` via `sampling_lift` |
| No caching | `t2_dispatch.py` | an identical batch re-ran 32 simulations |
| The likelihood is one law | `t6_tempered.py` | `log_prob` of a draw equals SciPy's; residuals mean 0.005, sd 1.03 |
| Tempered target density | `t6_tempered.py` | differences equal log π + Σ ηᵢ log Nᵢ (0.386083); plain: 8.239780 |
| `enskit_eki` through tempered conditioning | `t3_eki.py`, `t6_tempered.py` | 4 steps, 160 runs, 16 s, `EmpiricalDistribution` |
| RWMH through tempered conditioning | `t5_rwmh.py`, `t6_tempered.py` | runs (67 runs, 26 s); fails if EnsKit is imported after the prior is built |
| Linearization callback | `t3_eki.py` | accumulated eigenvalues 323, 24.1, 5.05, 0.06 |
| Laws are immutable | (session check) | a memo on a law raises `AttributeError` |
| Both method names needed for a density claim | (session check, PG-11) | `_log_prob` alone: "does not implement SupportsLogProb" |
| Corrected driver passes pySIPNET's strict checks | (session check) | `ClimateDrivers.from_file` OK; 0.29 s/run |
