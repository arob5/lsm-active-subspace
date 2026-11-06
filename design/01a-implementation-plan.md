# Implementation plan for design 01

*2026-10-15. Status: approved 2026-10-15.*

Build the parts of [design 01](01-architecture.md) that are settled, so that design 02 (the inverse
problem) can be worked out with real tools rather than the prototype. Everything here was already
demonstrated by the prototype; this turns it into tested package code.

**Done when:**
- `uv sync && uv run pytest` passes;
- `uv run pytest -m smoke` passes in a few minutes;
- on the development problem, `problem.condition(method="enskit_eki", …)` and
  `prior_predictive(problem, n)` both work;
- every stopgap carries its `PG-NN` tag and a gap / ideal / removal docstring, which the inventory
  test checks.

**Not in this round:** the real parameter set and prior (design 02); EKI that infers the noise
scales, the subspace estimators, the algorithms, and the final `StepRecord` contents (design 03); the
experiment runner, metrics and plots (design 04); notebooks and tutorials.

## Code standards

Code is judged by its quality, not only by its outputs: clean and well organized.

- **No redundancy.** Every quantity has a single source of truth.
- **Docstrings are written for users.** They explain what an object is and how to use it. They
  carry no unnecessary design detail, no references to design documents, and no LLM jargon.
  Stopgap tags (`# STOPGAP(PG-NN)`) are code comments, not docstring text. A stopgap's docstring
  says, in user terms, what ProbPipe lacks, what it would ideally provide, and how the code
  simplifies once it does.
- **Mathematics is stated precisely.** Anything that implements mathematics defines it in its
  docstring, in LaTeX: NumPy-style sections, raw docstrings, `$…$` inline and `$$…$$` display.
- **File order.** The public API comes first in a file; internal helpers follow.
- **Names say what things are,** in helpers too: no `_set`, `eval`, `tmp`, `do_it`.
- **Small units.** No massive functions or classes; split work into clearly named helpers.

## Steps

Each step is one commit on a branch `feat/skeleton`, with its tests.

1. **Scaffolding.**
   - `pyproject.toml` with the three commit pins from design 01 §3; `uv.lock`; Python ≥ 3.12.
   - The `src/lsmas/` layout of design 01 §5. `lsmas/__init__.py` imports EnsKit first (EK-01).
   - pytest with a `smoke` marker; `.gitignore`.
   - A minimal README: setup, the upgrade recipe, and a map of the layout.
2. **Shipped data** (`data/harvard_forest/`).
   - Driver: ERA5 member 1 (the corrected file and its `provenance.json`).
   - Observations: `nee_operator.npz` (A), `nee_features.csv` (y, class, year, month, half, n_steps)
     and `step_axis.json`.
   - `sipnet_fixed.csv`, from the Harvard Forest `fixed_sipnet_parameters.csv`.
   - `PROVENANCE.md`: for each file, its source path, the commit and script that made it, and a
     SHA-256.
   - Loader `lsmas.data.load_harvard_forest()`: returns the arrays and checks their shapes and
     alignment.
3. **`_stopgaps/`.** Four modules: `probpipe_private` (PG-04), `coordinates` (PG-05, with whitening),
   `likelihood` (PG-02, PG-11) and `tempering` (PG-03, PG-12). Each docstring states the gap, the
   ideal and the removal. Tests check against independent results:
   - the coordinates round-trip, and whitening is exact for a log-normal;
   - the likelihood's `log_prob` matches SciPy, and its draws have the right moments;
   - the tempered target's log-density differences equal log π + Σ ηᵢ log Nᵢ.
4. **SIPNET and the forward map.**
   - `lsmas.sipnet`: the runner (driver via `from_path`), the unit conversion, and the forward
     `Function` (`dispatch="thread"`).
   - Failures become NaN rows and are counted by kind: pySIPNET validation errors and SIPNET run
     errors.
   - `lsmas.parameters`: the `to_sipnet` interface.
   - Tests:
     - SIPNET's time axis equals A's;
     - one run matches a direct pySIPNET call;
     - an invalid parameter set gives a NaN row;
     - thread and sequential dispatch give identical results.
5. **Problems.**
   - `InverseProblem`, with `condition()` and `describe()`, and `problems.load(name)`.
   - The learning-rate rule (η = min(1, c·M / 12n)), with a LaTeX docstring. A test reproduces
     0.126 and 0.137.
   - One temporary problem, `harvard_forest_dev`: the prototype's 4 calibration parameters and
     representative fixed values, labelled as a placeholder for design 02.
6. **Predictive.** `prior_predictive(problem, n)` (the PG-01 stopgap) returns θ, the scales and the
   NEE features. Tests check shapes and noise moments.
7. **`enskit_eki`, simplest correct form.**
   - Tempered EKI with the noise scales **fixed**.
   - The likelihood accepts each class's scale either as a given slot (random, with an IG prior) or as
     a fixed number. `check` refuses random scales with a clear message.
   - Named only. Calls callbacks with a provisional `StepRecord`, and keeps the step history in the
     result.
   - Test: a linear-Gaussian problem, built with the same classes, whose posterior is known in
     closed form. With a large ensemble, the EKI mean and covariance must match it.
8. **Repo-wide tests.**
   - Inventory: every `STOPGAP(PG-NN)` names an open registry entry, and every open entry is used.
   - Every `@function` sets `dispatch`.
   - Smoke: one run, a prior predictive, and a short `enskit_eki` on `harvard_forest_dev`.

At the end I write `design/01b-implementation-report.md`: what was built, any deviations, small
hiccups fixed along the way, and gap-registry updates.

**I stop and report back**, instead of improvising, if:
- ProbPipe or EnsKit behaves in a way that breaks a stopgap's planned shape;
- an interface in design 01 has to change beyond renaming;
- a shipped file's provenance cannot be established.

## Decisions (settled 2026-10-15)

1. The untracked design-02 inputs are copied into `design/inputs/harvard_forest/`, with a README
   recording where each came from. Code never imports them.
2. The development problem fixes s² = 1 for EKI. A variant with inferred scales serves prior
   predictives and RWMH.
3. Work happens on the local branch `feat/skeleton`, one commit per step, with a GitHub pull request
   at the end.
