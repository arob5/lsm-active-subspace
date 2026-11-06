# Implementation report for design 01

*2026-10-15. Branch `feat/skeleton`.*

The plan in [`01a-implementation-plan.md`](01a-implementation-plan.md) is complete. All four of its
"done when" conditions hold:

- `uv run pytest` passes: 52 tests in about 26 s.
- `uv run pytest -m smoke` passes: 11 tests in about 12 s.
- On the development problem, `problem.condition(method="enskit_eki", …)` and
  `prior_predictive(problem, n)` both work.
- Every stopgap carries a `PG-NN` tag and a gap / ideal / removal docstring, and tests enforce this.

## What was built

| Step | Commit | Contents |
|---|---|---|
| 1 | `bcc1729` | `pyproject.toml` with the three commit pins, `uv.lock`, the package, the README with the upgrade recipe |
| 2 | `5294332` | `data/harvard_forest/` (driver, A, features, step axis, fixed values) with `PROVENANCE.md` and checksums; `design/inputs/harvard_forest/` with the rescued design-02 inputs |
| 3 | `62b819d` | `lsmas._stopgaps`: private ProbPipe re-exports, coordinates, `GaussianSimulatorLikelihood`, `tempered_condition_on` |
| 4 | `5f75dbe` | `lsmas.sipnet` (simulator, forward map), `lsmas.parameters` (photosynthesis and soil respiration reparameterizations) |
| 5 | `3e3b68a` | `lsmas.learning_rate`, `lsmas.priors`, `lsmas.problems` (`InverseProblem`, `harvard_forest_dev`) |
| 6 | `5f14deb` | `lsmas.predictive.prior_predictive` |
| 7 | `a34a8cf` | `lsmas.inference`: `EnsKitEKI` (`"enskit_eki"`), `StepRecord`, history |
| 8 | `af17ec4` | Repository-rule tests; design and registry brought in step with the code |

**Results worth knowing:**

- **EKI is exact where it should be.** On a linear-Gaussian problem with per-element learning rates,
  `enskit_eki` with 4000 particles matches the closed-form tempered posterior. The means agree
  within 5 standard errors and the covariances within 10%.
- **The photosynthesis reparameterization holds in SIPNET itself.** Changing `cFracLeaf` while
  holding the photosynthetic capacity P and the share ρ fixed leaves SIPNET's NEE bit-for-bit
  unchanged.
- **The learning rates reproduce the likelihood note.** The rule gives 0.126 (day) and 0.137
  (night). The tempered target's density equals log π + Σ ηᵢ log Nᵢ exactly, both on the test model
  and on the development problem.
- **On the development problem,** EKI with 40 particles takes 4 tempering steps: 160 SIPNET runs in
  about 12 s.

## Deviations from the design and the plan

All of these are small, and design 01 has been updated to match.

1. **`StepRecord` carries θ, not whitened ξ.** EKI's update is unchanged by an affine change of
   coordinates, so the method works in unconstrained coordinates and reports constrained θ. Tools
   whiten with `problem.coordinates`, which makes the problem the single source of whitening.
   Otherwise the method's coordinates could drift from the problem's. The record also drops the
   noise-scale field until EKI infers scales (design 03).
2. **Coordinates are built from declared supports.** The plan was to reuse ProbPipe's private
   unconstrained-coordinate helper. It fails on priors that ProbPipe re-expresses internally (see
   PG-05 and PG-13). So `ParameterBijection` builds each parameter's map from its declared support
   with the public `bijector_for`, and `Coordinates` adds the whitening.
3. **Whitening moments are estimated, not exact.** Design 01 promised exact moments for normal,
   log-normal and logit-normal factors. Reading them exactly means walking ProbPipe's internal
   factor structure, which the bugs above make fragile. They are instead estimated from 50,000 prior
   draws with a fixed key, about 0.3% error in each standard deviation. These draws do not touch
   the caller's `workflow_run`: a nested `workflow_run` was found to shift the outer random stream.
4. **`InverseProblem` stores its likelihood and prior and derives `model` and `forward`.** Splitting
   a model back into factors is unreliable (PG-13), and storing the parts removes a redundancy.
5. **Callbacks and history.** Methods call callbacks and also return the history, under the
   posterior's annotations at `arviz/eki_history`. ProbPipe stores annotations under `arviz/`.
6. **EKI posteriors have one `chain` level.** `make_posterior` lays every posterior out as chains of
   draws, so an EKI ensemble of J particles appears as one chain of J draws, shape `(1, J, d)`.
   Harmless, but users should know.
7. **`enskit_eki`'s `max_steps` defaults to 1000, not 100.** EnsKit refuses a step budget that its
   adaptive schedule's smallest increment (0.001) could exceed.
8. **Failures are logged, not counted by kind.** The forward map logs each failed run with its
   parameters and reason. `StepRecord.n_failed` counts failures per step.
9. **Not created yet:** `tools/`, `algorithms/`, `experiments/`, `plots/`. Nothing fills them until
   designs 03 and 04, and empty packages would be noise.

## Hiccups fixed along the way

- **pySIPNET fixed values.** An allocation chosen for a test was invalid against the default fine-root
  allocation. Fixed in the test.
- **The log message for a failed run** showed only "1 validation error"; it now shows the reason.
- **NumPy 2.5's `assert_allclose`** cannot format a failure when `atol` is an array, which hid a
  passing check. The test now uses explicit z-scores.
- **A commit ran after a failing test.** It went through because the shell pipeline's exit status
  came from `tail`. The commit was amended before anything was pushed, and commits now check
  `pytest`'s own status.

## ProbPipe gap registry: changes

[`probpipe-gaps.md`](probpipe-gaps.md) now has 14 entries.

**New entries:**

- **PG-13 (bug).** A path-renamed factor taken back out of a joint declares its grouped component,
  but its `log_prob` and its draws use the ungrouped field names. This shows up in the prior that
  `model_factors` gives an inference method.
- **PG-14 (bug).** Fingerprinting a `Function` whose closure captures a scipy sparse matrix recurses
  without end. Provenance logs a warning on every call. The forward map is now a bound method, the
  pattern ProbPipe's docs recommend.

**New evidence:**

- **PG-05:** the private helper refuses a prior that ProbPipe re-expressed as a
  `FactoredMultivariateGaussian`.
- **PG-12:** `condition_on(tempered_likelihood, given)` treats the tempered factor as an unnormalized
  law of y and runs an inference method to normalize it. That took about 50 s on the six-point test
  model.

**New stopgap:** PG-10's dispatch rule is now enforced by `tests/test_repository.py`.

Seven entries have no upstream issue yet: PG-02, PG-03, PG-05, PG-11, PG-12, PG-13 and PG-14.

## Noticed, not acted on

- **ProbPipe's per-call overhead** when a `Function` sweeps a batch row by row is about 2 ms per row.
  That is negligible next to SIPNET (0.3 s), but it dominates cheap test models; the linear test
  model therefore uses `dispatch="jax"`.
- **Untracked inputs in `spatial-lsm-calibration`.** The corrected driver (now shipped here) and the
  NEE step table behind A were never committed there.
