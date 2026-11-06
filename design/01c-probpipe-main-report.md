# Report on moving to ProbPipe `main`

*2026-11-05. Branch `feat/probpipe-main`.*

ProbPipe's `dev/overhaul` branch merged into `main` on 2026-10-26 (#468), and `main` has moved on
since then. This round moves the repository's pin from `638f56bc` (`dev/overhaul`, 2026-10-14) to
`9158a50e` (`main`, 2026-11-02). ProbPipe is still `0.2.0.dev0`, with no PyPI release, so the pin
stays a commit. EnsKit (`f7dc7ee`) and pySIPNET (`08bab7b`) are unchanged.

- `uv run pytest` passes: 53 tests in about 28 s.
- `uv run pytest -m smoke` passes: 13 tests in about 17 s.
- Two gaps closed, three are partly fixed, and one new gap was found (PG-15).

## What changed

| Commit | Contents |
|---|---|
| `6d8ef54` | The pin. ProbPipe's renames: `&` for merging laws, `.raw` as a property, `rename`, `event_spec.outputs`, and `traceable=` / `executor=` / `executor_options=` on every `Function`. `EnsKitEKI` on the public inference-method interface; the module of private ProbPipe names deleted (**PG-04 closed**) |
| `c12ca9f` | The likelihood's sampler and density in JAX around the forward map; `sample(problem.model, n)` replaces `lsmas.predictive` (**PG-01 closed**) |
| `5deab50` | ProbPipe's word "output" for a law's named draws: `output_name=` on the interval priors and the likelihood |
| `255c0eb` | A test that NUTS recovers the exact tempered posterior on the linear model; the closed form shared in `conftest.py` |
| `0453c77` | A correction to `StepRecord.increment`'s docstring |
| `e375ed8` | The gap registry re-verified at `9158a50e`, with [`prototypes/01c-probpipe-main/`](prototypes/01c-probpipe-main/) |
| `fa24032` | Design 01, the README and this report |
| `22c5273` | `InverseProblem.condition` requires `method` (see PG-08 below) |
| `4c46c0d` | One stopgap for keyed draws, `draw_with_key` (PG-15) |
| `2ab4736` | `StepRecord.noise_variance` kept a NumPy array, as documented |
| `3a3c093` | Tests: no private ProbPipe name imported in any form; model draws reproducible under a seed; the NUTS test names its method |
| (this commit) | This report brought up to date after the review |

The last four commits address a review of the branch; see "Review" below.

**How the main pieces changed:**

- **`EnsKitEKI`** imports nothing private. It declares `supported_types` as `(Distribution,)`,
  reads the model through `target.factors()`, receives the run's key in `execute(target, /, *, key,
  **options)`, and returns `make_posterior(target, [vectors], method=..., annotations=...)`. The
  particles reach `make_posterior` in the target's `to_vector` layout, built through a
  `NumericRecordBatch` with the target's spec, so the layout is ProbPipe's and not assumed.
- **`GaussianSimulatorLikelihood`** is JAX code around the forward map, which it calls with
  `raw=True`. Inside a JAX trace a `Function` returns a bare array; `raw=True` makes it return one
  outside a trace too. The forward map declares `traceable=False` and an output spec with a
  concrete shape and dtype (float64), which lets ProbPipe call it from traced code.
- **Prior predictives** are `sample(problem.model, n)`. Draws on the development problem run SIPNET
  on the forward map's 8 threads: 16 draws in 2.1 s.

## Results worth knowing

- **EKI is unchanged where it matters.** On the linear-Gaussian model it still matches the
  closed-form tempered posterior. On the development problem, 40 particles take 5 tempering steps
  (β = 0, 0.046, 0.189, 0.452, 0.947, then 1): 200 SIPNET runs in 17 s, none failed. The first
  round reported 4 steps and 160 runs, from a different seed, so the two are not comparable.
- **ProbPipe's own samplers now work on the tempered target** when the forward map traces: NUTS
  recovers the exact tempered posterior on the linear model.
- **RWMH on the development problem**, 2 chains × 30 steps: 67 runs in 24 s, or 22 s with the thread
  executor named. It still evaluates one point per chain at a time (PG-07).

## Earlier claims that turned out wrong

Each was corrected where it was made.

1. **"NUTS is skipped for opaque densities"** (the review before this round). True for a kernel
   built with `conditional_distribution`, which gets RWMH. Not true for our likelihood. Once the
   likelihood became JAX code, ProbPipe's automatic selection picks NUTS for every SIPNET problem,
   and NUTS fails at once: "gradients cannot pass through 'nee_features'". This is a regression
   from the first round, where selection picked RWMH. It is recorded under PG-08, and
   `InverseProblem.condition` now requires its `method` argument.
2. **`StepRecord.increment` "0 for the final evaluation".** EnsKit makes such a record only when a
   stopping rule ends a run early. A run that reaches β = 1 ends with its last update, so every
   record has a positive increment and the final particles are in none. A test now checks this.
3. **Design 01** said whitening moments are exact for Gaussian-in-z priors. They have been
   estimates since the first round. It also said a test checks that every open gap is referenced by
   a tag, which no test does, and it named the likelihood's scale argument `scale_slots`; the real
   argument is `scales`.
4. **PG-05's "also verified"** said ProbPipe's private coordinate helper refuses a fused Gaussian
   prior. At `main` it accepts one; it still refuses a view.

## ProbPipe gap registry: changes

[`probpipe-gaps.md`](probpipe-gaps.md) now has 15 entries plus EK-01. Each open entry was re-checked
at `9158a50e` by
[`prototypes/01c-probpipe-main/gaps_at_main.py`](prototypes/01c-probpipe-main/gaps_at_main.py), and
the registry gained a `partly` status.

| | Entries |
|---|---|
| **Closed** | PG-01 (joint sampling), PG-04 (public inference-method interface) |
| **Partly fixed** | PG-05 (vector layout public; unconstrained maps not), PG-07 (executors and per-point failures; RWMH still evaluates one point per chain at a time), PG-10 (no probe when `traceable` is declared; lifts still thin silently) |
| **Planned upstream** | PG-06 (`plans/execution-layer.md` §6.5), PG-08 (`plans/deferred/differentiability.md`), PG-09 (#521, #522) |
| **Open, unchanged** | PG-02, PG-03, PG-11, PG-12, PG-13, PG-14, EK-01 |
| **New** | PG-15: no public operation draws from a law with a given key. EKI's initial ensemble and the whitening draws go through one stopgap, `draw_with_key`, which calls `_sample(key, shape)` as ProbPipe's own methods do |

**New evidence:**

- **PG-08** (above): automatic selection assumes that a density that traces has gradients.
- **PG-11**: ProbPipe's documentation says "A class claims a capability by defining that method"
  (`docs/api/distributions.md:79`), which is what fails.
- **PG-14**: the cause is that `np.asarray` of a sparse matrix is a 0-d object array holding the
  matrix, which the fingerprint's array traversal hashes element by element with a fresh depth. Our
  stopgap, a bound method, means two forward maps differing only in their operator fingerprint
  alike. That only matters once ProbPipe reuses evaluations (PG-06).

Seven entries have no upstream issue yet: PG-02, PG-03, PG-11, PG-12, PG-13, PG-14 and PG-15.

## Review

A review of the whole branch found seven problems, all now fixed:

1. `problem.condition()` without a method failed inside NUTS (PG-08). `method` is now a required
   argument, so the mistake fails at the call.
2. The keyed draw of PG-15 was written out twice, once outside `_stopgaps/`. It is now one function.
3. Writing the likelihood in JAX turned `StepRecord.noise_variance` into a JAX array, against its
   documentation. EKI converts it back.
4. This report credited the change in EKI's step count to the pin, but the runs used different
   seeds.
5. Deleting the prior-predictive tests dropped the check that draws are reproducible under a seed.
   It is back, for `sample(problem.model, n)` on the SIPNET problem.
6. The private-import test missed `import probpipe.x._y` and `from probpipe.x import _y`.
7. The NUTS test asserted ProbPipe's automatic choice of method, so an upstream policy change could
   fail it with nothing in `lsmas` broken.

## Noticed, not acted on

- **Deferred ProbPipe plans that would change this repo again.** None is approved yet.
  - `plans/deferred/controls-and-options.md` would replace `_method_options` and `_check_options`,
    which `EnsKitEKI` uses, with an `options` declaration, and rename `num_results` to `num_draws`.
  - `plans/deferred/vector-interface.md` would add the public unconstrained map that could replace
    `ParameterBijection` (PG-05).
  - `plans/deferred/annotations.md` asks whether a part of a posterior keeps its annotations.
    Today it does not: `post.annotations` holds `arviz/eki_history`, but `post["theta"]` has none
    (verified). Code that wants the history should read it from the whole posterior.
- **A cluster executor.** ProbPipe now accepts user-registered executors and documents a
  Slurm/submitit example. A GridEngine executor would run EKI's ensembles on the cluster with no
  change to the method. It is a small, separate piece of work.
- **New ProbPipe features worth using later.** `mixture(likelihood, posterior)` for posterior
  predictives; Monte Carlo `marginal`; `summarize(d)`; and readable provenance
  (`print(post.provenance)`).
