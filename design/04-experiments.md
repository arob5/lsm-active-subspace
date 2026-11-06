# Design 04 — Experiments: reference posteriors, metrics, studies and plots

*Draft 1, 2026-11-06.*

Designs 01–03 built the problems, the tools and the first algorithm. This design builds what is
needed to compare algorithms: **reference posteriors** computed once and stored, **metrics** that
score an approximation against a reference, a **study runner** that repeats algorithms over settings
and seeds and collects the scores in one table, and the standard **plots**. The walkthrough notebook
and the tutorials use all four, so they come in the next round.

Every number below comes from the prototype in
[`prototypes/04-experiments/`](prototypes/04-experiments/), on this machine (8 cores). The notation
is that of [`docs/notation.md`](../docs/notation.md).

**In one paragraph.** References for the toy are NUTS runs (11–15 minutes each). References for the
SIPNET problems are long random-walk chains, one process per chain, because ProbPipe runs
untraceable chains one after another; each chain starts at its own EKI particle and uses EKI's
ensemble covariance as its proposal. On `harvard_forest_small` that mixes at about 115 draws per
effective draw, so a reference with 1000 effective draws per parameter takes about 2 hours on 8
cores; on `harvard_forest` it is at least 265 draws per effective draw, 5–35 hours, a job for a
cluster. Each reference is stored with its draws, its information matrix (for scoring subspaces),
its convergence diagnostics and its provenance. The headline metric is the marginal 1-Wasserstein
distance in whitened coordinates, in reference standard deviations; its Monte Carlo floor is 0.04.
A study of 12 toy runs takes a minute and resumes from its cache.

---

## 1. Reference posteriors

### 1.1 What a reference holds

For each problem and noise-scale setting (`toy`, `harvard_forest_small`, `harvard_forest`; fixed or
inferred scales), one file `baselines/<problem>/<scales>/reference.nc` holds:

| Part | Contents |
|---|---|
| `posterior` | draws of $\theta$ (and of $s^2$ when inferred), by chain and draw, thinned to at most 4000 in all |
| `information` | $H_{\mathrm{ref}} = \mathbb E_{\mathrm{ref}}\!\left[DG(\xi)^\top \Gamma^{-1} DG(\xi)\right]$ at 200 thinned reference draws (with each draw's own $\Gamma$ when the scales are inferred): exact Jacobians for the toy, central differences ($h = 0.2$) for SIPNET |
| `diagnostics` | per coordinate: R-hat, bulk and tail effective sample size (Vehtari et al. 2021) |
| provenance (attributes) | the method and its settings, seeds, package pins, git commit, forward runs, wall time |

It is an xarray `DataTree` written as netCDF (verified: 4000 draws of 14 parameters take 445 KB),
which ArviZ reads directly. References are small enough to ship in the repository and are never
recomputed by a user; `lsmas.experiments.load_reference(problem)` reads the one matching the
problem's name and scales.

### 1.2 How the toy's references are computed

NUTS through `problem.condition(method="blackjax_nuts")`, ProbPipe's defaults (4 chains of 1000
draws): 677 s with fixed scales, 885 s with inferred scales, R-hat at most 1.001 (design 03).

### 1.3 How the SIPNET references are computed

ProbPipe's `blackjax_rwmh` runs an untraceable target's chains one after another (PG-07) and starts
them all at one point (PG-17). So each chain is its own process and its own `condition` call:

1. **Pilot.** EKI with $J = 100$ particles.
2. **Proposal.** The random-walk proposal's Cholesky factor is that of the EKI ensemble's covariance
   in unconstrained coordinates (with $\log s^2$ when the scales are inferred), scaled by
   $2.38/\sqrt d$. It stays fixed, so every chain is a time-homogeneous Markov chain.
3. **Chains.** $C$ chains, each started at a distinct EKI particle, each `blackjax_rwmh` with
   `num_chains=1` in its own process. A chain runs in segments (2000 steps by default), each segment
   started at the previous one's last state and saved, so an interrupted run resumes and a run that
   has not converged is extended.
4. **Stopping.** The first segment is discarded as warmup. The run stops when every coordinate has
   R-hat at most 1.01 and bulk and tail effective sample sizes of at least 1000.
5. **Information.** $H_{\mathrm{ref}}$ by central differences at 200 thinned draws: $2d$ runs per
   draw, 4000 runs for the small problem and 18,400 for the full one.

Each chain is one invocation of a command-line entry point, so the same code runs locally (a pool
of processes) or on a cluster (one array-job task per chain), followed by a merge.

**Verified** (`t1_reference_trial.py`, `t4_pilot_proposal.py`; 8 chains of 100 warmup steps and 400
draws, then 400 more):

| | `harvard_forest_small` (d = 10) | `harvard_forest` (d = 46) |
|---|---|---|
| seconds per step per chain, 8 at once | 0.53 | 0.55 |
| acceptance | 0.16–0.25 | 0.21–0.28 |
| draws per effective draw, worst coordinate | 112 (119 with a proposal tuned on the trial's draws) | 265, from chains not yet mixed (R-hat 1.77) |
| draws for 1000 effective draws per coordinate | ≈ 115,000 | ≥ 265,000, plausibly 1,000,000 |
| wall time on 8 cores | ≈ 2 h | 5–35 h |

A proposal tuned on the trial's own draws did not mix better (119 against 112), so the EKI
covariance stays. For the full problem the first numbers are a lower bound: its short chains had not
mixed, so it needs a cluster and is stored as the **best affordable** reference, with its
diagnostics, rather than a ground truth.

### 1.4 Which references

| Problem | Fixed scales | Inferred scales |
|---|---|---|
| `toy` | NUTS, 11 min | NUTS, 15 min |
| `harvard_forest_small` | parallel RWMH, ≈ 2 h | parallel RWMH, ≈ 2 h |
| `harvard_forest` | not computed | parallel RWMH on a cluster (decision 2) |

---

## 2. Metrics

Each metric compares draws of an approximation $q$ with the reference $p$, in whitened coordinates
$\xi_j$ (and $\log s^2_k$ for the scales). With $\sigma_j$ the reference standard deviation of
$\xi_j$:

| Metric | Definition | Role |
|---|---|---|
| **marginal $W_1$** | $W_1(q_j, p_j) / \sigma_j$, the 1-Wasserstein distance between marginals; reported as the mean and the maximum over $j$ | the headline: it sees both location and spread |
| mean error | $\lvert \mathbb E_q \xi_j - \mathbb E_p \xi_j \rvert / \sigma_j$ | where the location is wrong |
| sd error | $\lvert \log(\mathrm{sd}_q \xi_j / \sigma_j) \rvert$ | where the spread is wrong |
| Gaussian $W_2$ | the 2-Wasserstein distance between Gaussian fits of $q$ and $p$ | joint structure: correlations |
| captured information | $\operatorname{tr}(W^\top H_{\mathrm{ref}} W) / \sum_{i \le r} \lambda_i(H_{\mathrm{ref}})$ | how good the subspace is, at its rank |
| largest principal angle | between the span of $W$ and that of the top $r$ eigenvectors of $H_{\mathrm{ref}}$ | the same, as an angle |
| forward runs | by stage, from `AlgorithmResult.forward_runs` | cost |

90% interval coverage was tried and is not a headline: the prior alone covers 0.87 of the reference
on average, and an over-dispersed approximation cannot score above 1.

**Verified on the toy** (`t2_metrics.py`; marginal $W_1$ mean / max, mean error mean, Gaussian
$W_2$):

| Approximation | marginal $W_1$ | mean error | Gaussian $W_2$ |
|---|---|---|---|
| two halves of the NUTS reference (the noise floor) | 0.042 / 0.061 | 0.019 | 0.245 |
| the prior | 0.883 / 2.246 | 0.814 | 3.11 |
| full-dimensional RWMH, ProbPipe defaults | 0.387 / 1.797 | 0.338 | 2.06 |
| EKI, $J = 100$ | 0.268 / 0.494 | 0.236 | 1.07 |
| `active_subspace_mcmc`, seeds 1, 2, 3 | 0.151, 0.188, 0.137 / 0.456, 0.339, 0.300 | 0.133, 0.171, 0.116 | 0.67, 0.74, 0.59 |

The floor is small next to every difference that matters, and the spread over seeds is as large as
some differences between methods: studies need replicates.

---

## 3. Studies

A **study** runs one algorithm over a list of settings and a set of seeds on one problem:

```python
from lsmas.algorithms import active_subspace_mcmc
from lsmas.experiments import Study, run_study

study = Study(
    name="toy_rank",
    problem="toy",
    scales="fixed",
    algorithm=active_subspace_mcmc,
    settings=[{"rank": rank} for rank in (3, 5, 7, 9)],
    seeds=range(5),
)
table = run_study(study, directory="studies/toy_rank")   # one row per run
table.groupby("rank")["marginal_w1_mean"].agg(["mean", "std"])
```

- **One run** is an algorithm call on the problem with one setting and one seed. Its directory,
  named by a hash of the problem, the scales, the algorithm's name, the setting and the seed, holds
  the result (posterior draws, subspace, forward runs, wall time) as netCDF and its metrics as JSON.
- **Resuming.** A run whose directory is complete is read, not rerun, so an interrupted study
  continues where it stopped, and a study extended with more seeds runs only the new ones.
- **The table** has one row per run: the setting, the seed, the metrics of §2 against the stored
  reference, the forward runs and the wall time.
- Runs execute one after another; SIPNET runs already use every core within a run.

**Verified** (`t3_study.py`): a rank sweep on the toy, 4 ranks × 3 seeds with short chains, took 60
seconds and reran from its cache at no cost.

| rank | marginal $W_1$ (mean over 3 seeds ± sd) |
|---|---|
| 3 | 0.736 ± 0.009 |
| 5 | 0.344 ± 0.025 |
| 7 | 0.221 ± 0.099 |
| 9 | 0.252 ± 0.051 |

---

## 4. Plots

`lsmas.plots` holds one function per standard figure, each taking results or a table and returning
a matplotlib figure. All were drawn in the prototype (`out/figures/`):

| Figure | Shows | Data |
|---|---|---|
| spectrum | eigenvalues of each run's estimate, and the reference's, on a log scale with the line $\lambda = 1$ | subspaces, reference |
| error against rank or cost | a metric's mean and spread over seeds | the study table |
| QQ against the reference | per coordinate, the approximation's quantiles against the reference's | one run, reference |
| activity scores and loadings | per parameter, as bars | a subspace |
| sufficient summary | the misfit $\Phi$ of every EKI particle against $a_1$, $a_2$ and an inactive variable | EKI's step history, no extra runs |

The prototype's figures read as intended: the spectrum shows the ladder estimates' low bias against
the reference; the QQ plot shows where a reduced posterior errs (on the toy, `leaf_off_day` is too
narrow); the sufficient-summary plot shows a clear valley in $a_1$, weaker structure in $a_2$, and
none in an inactive variable.

---

## 5. Code shape

| Module | Holds |
|---|---|
| `lsmas.experiments.reference` | `Reference` (draws, information, diagnostics, provenance), `load_reference(problem)`, and the computation of §1: `nuts_reference(problem)`, `rwmh_reference_chain(problem, chain, ...)`, `merge_reference(...)` |
| `lsmas.experiments.metrics` | the metrics of §2, each a function of approximation draws and a `Reference`; `score(result, reference)` gives a row |
| `lsmas.experiments.study` | `Study`, `run_study(study, directory)` |
| `lsmas.plots` | the figures of §4 |
| `lsmas._stopgaps.parallel_chains` | one chain per process, each with its own start (PG-07, PG-17) |
| `scripts/compute_reference.py` | the command line for references: one chain, or merge |
| `baselines/` | the stored references |

New dependencies: `matplotlib`, `h5netcdf` and `h5py`.

---

## 6. Decisions for you

1. **References** (§1.4): toy and small problem with both scale settings; the full problem with
   inferred scales only, run on a cluster and labelled best affordable.
2. **The full problem's reference**: run it now on a cluster, or defer it until an algorithm study
   needs it.
3. **SIPNET reference recipe** (§1.3): parallel RWMH from EKI particles with EKI's covariance,
   stopping at R-hat 1.01 and 1000 effective draws per coordinate.
4. **The headline metric** (§2): marginal $W_1$ in whitened coordinates, mean and maximum, with the
   others reported alongside; coverage as a diagnostic only.
5. **Storage**: netCDF files shipped in `baselines/`, with the new dependencies.
6. **Scope**: the walkthrough notebook and the tutorials in the next round, on top of this.

---

## Appendix: what the prototype verified

| Claim | Script | Result |
|---|---|---|
| Parallel RWMH works through ProbPipe, one process per chain | `t1_reference_trial.py` | 8 chains, 0.53 s per step each; acceptance 0.16–0.28 |
| The small problem's reference costs about 2 hours | `t1`, `t4` | 112–119 draws per effective draw |
| The full problem's reference needs a cluster | `t1` | ≥ 265 draws per effective draw, R-hat 1.77 after 400 draws |
| A pilot-tuned proposal does not beat EKI's | `t4_pilot_proposal.py` | 119 against 112 |
| References store compactly as netCDF | session check | 445 KB for 4000 × 14; round trip exact |
| The metrics discriminate, with a small floor | `t2_metrics.py` | table of §2 |
| A study runs, caches and resumes | `t3_study.py` | 12 runs in 60 s; rerun free |
| The figures read as intended | `t3_study.py`, `t5_shadow_plot.py` | `out/figures/` |
