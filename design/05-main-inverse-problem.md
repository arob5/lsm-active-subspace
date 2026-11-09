# Design 05 — The walkthrough notebook: the Harvard Forest inverse problem

*Approved 2026-11-09 (draft 1), with the recommendations of §6.*

Designs 01–04 built the inverse problem, the algorithms and the experiment harness. This design
specifies the notebook that presents the inverse problem itself, `notebooks/main_inverse_problem.ipynb`.
It has two jobs:

1. **State the problem.** It introduces each part of the Bayesian inverse problem in words and in
   mathematics: the site and its weather, SIPNET, the calibration parameters and their prior, the
   data and the observation operator, the forward map, the likelihood with its noise scales and
   learning rates, and the generalized posterior.
2. **Teach the objects.** Each part is reached in code through the object that holds it, and the
   notebook shows how to use that object: load it, inspect it, evaluate it, sample it, plot it.

It covers the full problem, `harvard_forest`, only; the smaller problems belong to later tutorials.
The other tutorials will come in later rounds.

Every number below comes from the prototype in
[`prototypes/05-main-inverse-problem/`](prototypes/05-main-inverse-problem/), on this machine
(8 cores). The notation is that of [`docs/notation.md`](../docs/notation.md).

**In one paragraph.** The notebook has eleven sections, from the overall equation to the
coordinates the algorithms work in. It has twelve figures, all drawn in the prototype. As built, it
runs SIPNET about 550 times and executes in about 65 seconds, so it is committed with its outputs
and executed by a test. Most of its code calls the package directly. Five small additions make that
code short:
- `SIPNETSimulator.run`, for all of SIPNET's outputs;
- `InverseProblem.predicted_observations`, for $G$ at an array of parameters;
- `HarvardForestData.feature_times`;
- two plotting functions that later tutorials will reuse.

The prototype also found that ProbPipe's `summarize` gives Monte Carlo summaries of a prior that is
known exactly. That is a new gap, PG-19. Until it closes, the notebook takes its prior table from
the shipped specification.

---

## 1. What a reader leaves with

A reader who has worked through the notebook can:

- write down the generalized posterior
  $\pi_\eta(\theta, s^2 \mid y) \propto \pi(\theta)\, \pi(s^2) \prod_i \mathcal N\!\left(y_i \mid
  G_i(\theta), s^2_{k(i)} v_i\right)^{\eta_i}$, and say what each symbol is and where it comes from;
- load the problem and its inputs, and find each symbol in an object: $y$ in `problem.data`, $G$ in
  `problem.forward`, $\eta$ in `problem.learning_rate`, $\Gamma$ from `problem.noise_variance`, the
  prior in `problem.prior`, and $A$ in `data.operator`;
- run SIPNET at a parameter setting and see its outputs;
- evaluate $G$ at one parameter setting, at a batch, and at an array of settings;
- draw from the prior and the prior predictive, evaluate prior densities, and summarize draws;
- move parameters between constrained, unconstrained and whitened coordinates, and say why the
  algorithms use the whitened ones.

---

## 2. The outline

Each section opens with a short statement of what it introduces, gives the mathematics, then the
code, then a figure with a sentence saying what to look for. The SIPNET runs and times are the
prototype's (`out/t1.txt`).

| § | Section | Introduces | Figure | SIPNET runs | Time |
|---|---|---|---|---|---|
| 1 | The problem in one equation | the generalized posterior; `load`, `describe` | — | 0 | 2 s |
| 2 | The site and its weather | the driver; `load_harvard_forest` | F1 | 0 | 2 s |
| 3 | The calibration parameters and their prior | 58 → 46 parameters; four families; `marginal`, `log_prob`, `sample` | F2 | 0 | 4 s |
| 4 | SIPNET | $x(\theta)$; `calibration_to_sipnet`, `SIPNETSimulator` | F3 | 1 | 1 s |
| 5 | The data and the observation operator | $y$, $A$, $G = A x$ | F4, F5 | 0 | 1 s |
| 6 | The forward map | `problem.forward` on a record and a batch; failures | F6 | 20 | 3 s |
| 7 | The likelihood | $v_k$, $s^2_k$, $\eta$, $\Gamma$ | F7, F8 | 0 | 1 s |
| 8 | The prior predictive | `sample(problem.model, n)`, `predictive_check` | F9, F10 | 400 | 31 s |
| 9 | The posterior along one line | tempered against untempered | F11 | 82 | 8 s |
| 10 | Coordinates | $\theta$, $z$, $\xi$; `problem.coordinates` | F12 | 0 | 1 s |
| 11 | Summary | the table of objects; where to go next | — | 0 | — |
| | **Total** | | 12 | **≈ 550** | **≈ 65 s** |

The figure numbers here follow the notebook's order. The prototype's file names (`f01`–`f12`)
follow its own order. The runs and times are the prototype's, except §9 and the total, which are
the notebook's as built: §6 also shows a batch of 16 prior draws and a failed run, and §7 one
pointwise log-density, which together add about 20 runs.

### §1 The problem in one equation

The whole problem in one display, with each symbol named and the section that introduces it:

$$
\pi_\eta(\theta, s^2 \mid y) \;\propto\; \underbrace{\pi(\theta)\, \pi(s^2)}_{\text{§3, §7}}
\prod_{i=1}^{n} \mathcal N\!\Big(\underbrace{y_i}_{\text{§5}} \;\Big|\;
\underbrace{G_i(\theta)}_{\text{§4–6}},\; \underbrace{s^2_{k(i)} v_i}_{\text{§7}}\Big)^{\eta_i}.
$$

Then the code that holds it:

```python
from lsmas.problems import load

problem = load("harvard_forest", scales="inferred")
print(problem.describe())
```

`scales="inferred"` is used throughout, because it includes the noise-scale prior. One sentence
says that `scales="fixed"` sets $s^2_{\mathrm{day}} = s^2_{\mathrm{night}} = 1$ and changes
nothing else.

### §2 The site and its weather

Harvard Forest: the tower, its location, and the years used (2012–2020). The driver: 3-hourly ERA5
weather, corrected as `data/harvard_forest/PROVENANCE.md` describes. `load_harvard_forest()`
returns every shipped input as one `HarvardForestData`. This section shows its fields and reads the
driver through pySIPNET:

```python
from pysipnet import ClimateDrivers

data = load_harvard_forest()
driver = ClimateDrivers.from_path(data.driver_path, time_zone="UTC").xarray
```

**F1**: daily air temperature, PAR, VPD and precipitation for one year (2015).

### §3 The calibration parameters and their prior

- **From SIPNET's parameters to the calibration parameters.** pySIPNET writes 58 parameters, and
  12 are fixed: 8 have no effect on NEE at this site, 3 remove exact degeneracies, and 1 follows
  from the leafless start. `data.fixed_sipnet_parameters` holds their values, with each reason in
  `sipnet_fixed.csv`. The other 46 are calibrated, some through reparameterizations: photosynthetic
  capacity and the respiration share, rates at 10 °C, allocation as successive shares, and initial
  pools as amounts. Each reparameterization is stated as an equation, with
  `calibration_to_sipnet(theta, fixed)` as the map.
- **The prior.** $\pi(\theta) = \prod_{j=1}^{46} \pi_j(\theta_j)$, independent, in four families:
  - log-normal and logit-normal, stated by their central 95% interval;
  - normal;
  - uniform.

  `data.prior_table` is the specification, one row per parameter with its source, and
  `prior_medians(data.prior_table)` gives the medians. One parameter's law is
  `marginal(problem.prior, "theta/photosynthetic_capacity")`, which has exact `mean`, `quantile`
  and `log_prob`. A density over a grid needs the grid as a batch:

  ```python
  law = marginal(problem.prior, "theta/photosynthetic_capacity")
  grid = np.linspace(100, 600, 200)
  density = np.exp(log_prob(law, NumericArrayBatch(grid, level_names="grid", label="grid")))
  ```

  Draws come from `sample(problem.prior, n)` inside `workflow_run(seed=...)`, and
  `problem.coordinates.array_from_draws(draws)` gives them as an $(n, 46)$ array.

**F2**: the 46 prior densities in a grid of panels, from the new `plot_prior_marginals` (§4).

### §4 SIPNET

SIPNET as a map from SIPNET parameters to 3-hourly fluxes and pools, $x(\theta) \in \mathbb R^S$
with $S = 37{,}992$ steps. The section runs SIPNET once, at the prior medians, through the layers
that `problem.forward` hides:

```python
sipnet_parameters = calibration_to_sipnet(medians, data.fixed_sipnet_parameters)
simulator = SIPNETSimulator(data.driver_path, data.fixed_sipnet_parameters)
outputs = simulator.run(sipnet_parameters)    # new: every SIPNET output, as an xarray Dataset
x = simulator.nee_rate(sipnet_parameters)     # NEE in µmol m⁻² s⁻¹, one value per step
```

**F3**: for 2015, NEE (3-hourly and daily mean), GPP, leaf carbon and soil wetness. The figure shows
leaf-on and leaf-off as jumps in leaf carbon, and the summer drawdown of soil water. Both come back
in §6.

### §5 The data and the observation operator

- **The features.** Each feature $y_i$ is the mean observed NEE over the usable day (or night)
  3-hour windows $S_i$ of one half-month: 214 day and 186 night features, 2012–2020.
  `data.features` is the table.
- **The operator.** $A_{is} = 1/|S_i|$ for $s \in S_i$, and 0 otherwise, so $(A x)_i$ is the same
  average of SIPNET's output. $A$ is sparse: 400 × 37,992 with 7,588 nonzeros, and 4–54 windows per
  feature. The section checks in code that `data.operator @ x` equals `problem.forward(theta)`.
  This equality holds (verified).

**F4**: the 400 features over time, day and night. **F5**: one half-month (1–15 July 2015). It
shows SIPNET's 3-hourly NEE, marks the steps each of its two features averages, and draws
$(A x)_i$ and $y_i$ over them. The 28 day windows and the 4 night windows show why the two classes
differ in precision.

### §6 The forward map

$G(\theta) = A\, x(\theta)$ as a ProbPipe `Function`. The section covers four things:

- A record in gives one prediction of shape $(400,)$, in 0.4 s.
- A batch of records, such as `sample(problem.prior, 16)["theta"]`, gives one row per record. The
  runs execute on 8 threads, so 16 take 1.6 s.
- `problem.predicted_observations(theta)` (new, §4) takes an array of shape $(N, 46)$ and returns
  $(N, 400)$. It is the form an algorithm uses.
- A run SIPNET cannot complete gives a row of NaN and a logged warning, so one failure does not
  stop a batch.

**F6**: one parameter at a time. Four parameters each move through their prior's 2.5, 25, 50, 75
and 97.5% quantiles, with the rest at their medians, and the figure shows the day and night
features for 2015: 20 runs, as one batch.
- `photosynthetic_capacity` moves daytime uptake.
- `soil_water_holding_capacity` moves late-summer uptake.
- `leaf_on_growing_degree_days` moves the start of the season.
- `soil_respiration_flux_at_10c` moves night respiration.

The section ends by saying that these one-at-a-time sensitivities miss how parameters act
together. Finding the directions the data inform is what the algorithms do.

### §7 The likelihood

$$
y_i \mid \theta, s^2 \sim \mathcal N\!\left(G_i(\theta),\; s^2_{k(i)} v_i\right),
\qquad v_{\mathrm{day}} = 8,\; v_{\mathrm{night}} = 1.5,
\qquad s^2_k \sim \mathrm{IG}(2, 1.678).
$$

- **The reference variances** $v_k$ set each class's noise level, and the scales $s^2_k$ multiply
  them. The inverse-gamma prior on each scale has median 1.
- **The learning rate.** The errors of neighbouring half-months are correlated, so the 400
  features carry less information than 400 independent observations would. The learning rate
  tempers the likelihood to $c = 3$ effective observations per class-year:
  $$
  \eta_k = \min\!\left(1, \frac{c\, M_k}{12\, n_k}\right),
  $$
  with $M_k$ calendar months and $n_k$ features in class $k$. This gives 0.126 (day) and 0.137
  (night).
- **The tempered noise covariance** $\Gamma = \operatorname{diag}(s^2_{k(i)} v_i / \eta_i)$
  combines the three. `problem.noise_variance({"s2_day": 1, "s2_night": 1})` returns its diagonal;
  $\sqrt{\Gamma_{ii}}$ is 7.96 by day and 3.31 by night.

The code covers `problem.likelihood.reference_variance`, `classes`, `noise_variances(...)` and
`pointwise_log_prob(...)`, `problem.learning_rate` and `problem.noise_variance(...)`.

**F7**: the residuals $y - G(\theta_{\mathrm{median}})$ over time, by class, against bands of
$\pm 2\sqrt{v_k}$ (untempered) and $\pm 2\sqrt{\Gamma_{ii}}$ (tempered). **F8**: the IG(2, 1.678)
density, with its median.

### §8 The prior predictive

The model $p(y \mid \theta, s^2)\, \pi(\theta, s^2)$ is `problem.model`, and
`sample(problem.model, n)` draws the parameters and then the data. It takes 200 draws, in 15 s.
None failed.

- **Coverage.** The prototype gave 0.97 of day features and 0.91 of night features inside the 90%
  interval of the replications. The notebook prints its own.
- **Test statistics.** `predictive_check(problem.likelihood, problem.prior, [statistics],
  observed_data=..., num_replications=200)` compares statistics of the replications with those of
  the data. Mean summer daytime NEE gives $p = 0.92$: the prior's summer uptake is weaker than
  observed, as design 02 found. Mean night NEE gives $p = 0.29$.

**F9**: the 50% and 90% bands of the replicated features over time, by class, with the
observations; from the new `plot_features` (§4). **F10**: each statistic's replicated histogram
and its observed value.

### §9 The posterior along one line

Conditioning is the next notebooks' subject, but one line shows what tempering does. The figure
fixes every parameter but `photosynthetic_capacity` at its median, with $s^2 = 1$. It evaluates the
prior, the untempered posterior and the tempered posterior along that line. The prototype used 41
values (41 runs, 3.4 s):

$$
\log \pi(\theta_j) + \sum_i \eta_i \log \mathcal N\!\left(y_i \mid G_i(\theta), v_i\right)
\quad\text{against}\quad
\log \pi(\theta_j) + \sum_i \log \mathcal N\!\left(y_i \mid G_i(\theta), v_i\right).
$$

**F11**: the three, each normalized along the line. The tempered curve is several times wider than
the untempered one, and both are far narrower than the prior. The text says that this is a
conditional slice, not a marginal posterior. In the prototype the grid was too coarse at the
untempered peak. The notebook uses 21 values across the prior, then 61 within 1.5 grid spacings
of the tempered peak (82 runs), and shows the peak in a second panel.

### §10 Coordinates

The algorithms compare directions in parameter space, so every parameter must be on one scale. The
coordinates are those of `docs/notation.md`:

- constrained, $\theta$;
- unconstrained, $z = b^{-1}(\theta)$;
- whitened, $\xi = L^{-1}(z - m)$.

`problem.coordinates.bijection.to_unconstrained`, `whiten`, `whitened_from_parameters` and
`parameters_from_whitened` move between them. In whitened coordinates the prior has mean 0 and
covariance $I$; for 4000 prior draws, the largest deviations were 0.04 in the mean and 0.06 in the
covariance (verified).

**F12**: the prior draws of two parameters in $\theta$, $z$ and $\xi$: a log-normal one
(`photosynthetic_capacity`) and a logit-normal one (`respiration_share`).

### §11 Summary

A table maps each symbol to its object and section, and the notebook ends with a pointer to the
README's sections on the algorithm and on comparing algorithms.

---

## 3. Conventions

- **Notation** is `docs/notation.md`'s, and the markdown uses the same LaTeX as the docstrings.
- **Seeds.** Every random call runs inside `workflow_run(seed=...)`, so a rerun reproduces every
  figure.
- **Code cells** are short. A figure drawn once (F1, F3, F5–F8, F10–F12) is drawn inline, in at
  most about 15 lines of matplotlib. A figure the later tutorials will draw again (F2, F4, F9) comes
  from `lsmas.plots`.
- **Text** is plain and specific: what is introduced, the equation, the code, what the figure
  shows.
- **Outputs are committed**, so the notebook reads like a presentation on GitHub without running
  SIPNET. Executed, it is 1.4 MB; the prior marginals (F2) are the largest figure, about 270 KB.

---

## 4. Additions to the package

The prototype wrote the notebook's code against the package as it is. Five things made that code
longer or reached into private names. Each addition below fixes one of them.

| Addition | Why | Used elsewhere |
|---|---|---|
| `SIPNETSimulator.run(parameters) -> xarray.Dataset`: every SIPNET output, with `time` the step end | §4 needs GPP, leaf carbon and soil wetness; today they are reachable only through the private `_model` | `run_nee` and `nee_rate` are rewritten on it, as one source |
| `InverseProblem.predicted_observations(theta)`: $G$ at each row of an $(N, d)$ array, returned as an $(N, n)$ array; one row of shape $(d,)$ gives $(n,)$ | §6 and §9 evaluate $G$ at arrays; today that is `np.asarray(problem.forward(problem.coordinates.bijection.record_batch(theta)).raw)` | `tools.estimators.finite_difference_subspace` and the tests use the same expression, and would call this instead |
| `HarvardForestData.feature_times`: each feature's midpoint, a `DatetimeIndex` | F4, F5, F7, F9 plot features against time | `plot_features` |
| `plots.plot_features(data, predictions=None, *, levels=(0.5, 0.9))`: the observed features by class over time, with optional quantile bands of predicted or replicated features $(N, n)$ | F4, F9 | posterior predictive figures in the algorithm tutorials |
| `plots.plot_prior_marginals(problem, draws=None)`: each parameter's prior density, with an optional histogram of draws over it | F2 | prior against posterior in the algorithm tutorials |

`plot_prior_marginals` uses only exact quantities:
- The grid of each panel is $b_j(m_j \pm 3 L_{jj})$ from the coordinates, which are fixed for a
  given prior.
- The density comes from `log_prob`, and the median from `lsmas._stopgaps.quantiles.median`, which
  is exact (PG-16).

**Running the notebook.** These are added to the `dev` dependency group, which `uv sync` installs:
`jupyterlab` to open the notebook, and `nbclient`, `nbformat` and `ipykernel` to execute it. The
prototype's execution under `nbclient` in the project environment ran (verified, `t2`).
- **Executing and saving in place:**

  ```bash
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/main_inverse_problem.ipynb
  ```

  The README gives this command.
- **The test.** `tests/test_notebooks.py` executes a copy of the notebook and fails on any error.
  It carries the `smoke` mark, which already labels the end-to-end SIPNET tests, and it adds about
  a minute to the suite.

---

## 5. ProbPipe observations

- **PG-19 (new): `summarize` of a product prior is Monte Carlo.** `summarize(problem.prior)` reports
  "Monte Carlo, 1,024 draws" for every calibration parameter, though each is an independent law
  with exact moments and (except the logit-normals, PG-16) exact quantiles. Its median of
  `photosynthetic_capacity` was 253.7 in one call against the exact 251.0 from
  `quantile(marginal(...), 0.5)`, and outside `workflow_run` it changes on every call (verified).
  - **Ideal ProbPipe:** `summarize` of a `FactoredDistribution` uses each factor's exact
    statistics.
  - **Here:** the notebook shows `data.prior_table` (the specification) and the medians from
    `prior_medians`, and leaves `summarize` to the posterior tutorials, where Monte Carlo is the
    right method.
  - **Registered** as PG-19 in [`probpipe-gaps.md`](probpipe-gaps.md#pg-19). Rechecked while
    implementing: two calls gave medians of 251.8 and 250.6 and means of 258.6 and 261.9, against
    the exact 251.0 and 262.4; the two noise scales, top-level factors, are summarized exactly.
- **Densities on a grid.** `log_prob(law, array)` refuses an array of values for a scalar law: the
  value must match the event shape. The grid must be wrapped as
  `NumericArrayBatch(grid, level_names="grid", label="grid")`. The notebook shows the wrapper, and
  `plot_prior_marginals` hides it. This is ProbPipe's design, but it is a stumbling block for a
  newcomer (decision 6).
- **`predictive_check`'s statistics** come back as an `EmpiricalDistribution`, whose draws are
  `.atoms.raw`. With a single statistic its fields lose the statistic's name as a prefix, so one
  statistic and two are read differently. The notebook always passes a list.
- **A marginal's printed form** names every factor of the product it came from:
  `marginal(problem.prior, "theta/photosynthetic_capacity")` prints as
  `(LogNormal·BijectorTransformedDistribution·…·InverseGamma)(photosynthetic_capacity)`. The
  notebook does not print a law. This is a small readability issue, not recorded as a gap.
- **The driver's precipitation** is in mm per step (pySIPNET's `attrs`); the prototype's figure
  labelled it cm. The notebook reads `kind` from the `attrs` to choose daily sums or means.

---

## 6. Decisions (approved 2026-11-09)

1. **Scope:** the `harvard_forest` problem only, with inferred scales, and no mention of the
   smaller problems except the closing pointer to the README.
2. **The outline** of §2: eleven sections in that order, with prior before SIPNET so that the
   medians SIPNET runs at are already defined.
3. **The additions of §4,** including rewriting `finite_difference_subspace` on
   `predicted_observations`.
4. **Committed outputs, executed by a test** in the default suite (about a minute), with Jupyter in
   the `dev` group.
5. **PG-19** registered as above.
6. **Densities on a grid** (§5) are left as ProbPipe's design, with the wrapper shown in the
   notebook and hidden inside `plot_prior_marginals`; no gap is registered.

---

## Appendix: what the prototype verified

| Claim | Script | Result |
|---|---|---|
| Every section's code runs against the package, with the additions of §4 stood in by their current forms | `t1_sections.py` | all twelve figures drawn |
| The notebook's cost | `t1_sections.py` | about 460 SIPNET runs, 48 s in all; the prior predictive (400 runs) is 31 s of it |
| $A\,x(\theta)$ equals `problem.forward(theta)` | `t1_sections.py` | equal |
| One SIPNET run; a batch of 16; of 20 | `t1_sections.py`, session check | 0.40 s; 1.6 s; 1.9 s |
| The prior predictive has no failed runs and covers the data | `t1_sections.py` | 0 of 200 failed; 0.97 of day and 0.91 of night features in the 90% band |
| Prior predictive check statistics | `t1_sections.py` | mean summer day NEE $p = 0.92$, mean night NEE $p = 0.29$ |
| Whitened prior draws have mean 0 and covariance $I$ | `t1_sections.py` | largest deviations 0.043 and 0.062 over 4000 draws |
| $\eta$ and $\sqrt{\Gamma_{ii}}$ at $s^2 = 1$ | `t1_sections.py` | 0.126 and 7.96 by day; 0.137 and 3.31 by night |
| A notebook of these cells executes under `nbclient` in the project environment | `t2_execute_notebook.py` | 11.6 s for load and 16 draws; 76 KB with one figure |
| `summarize` of the prior is Monte Carlo (PG-19) | session check | "Monte Carlo, 1,024 draws"; median 253.7 against the exact 251.0 |
| The notebook as built executes | `tests/test_notebooks.py` | 61 s under the test, 65 s with `nbconvert`; 1.4 MB with outputs |
| The notebook's prior predictive (seeds 2 and 3) | the notebook | 0 of 200 failed; 0.95 of day and 0.95 of night features in the 90% band; $p = 0.89$ and $0.30$ |
