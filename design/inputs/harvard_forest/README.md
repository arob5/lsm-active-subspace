# Reference inputs for design 02

Material from the Harvard Forest work in `spatial-lsm-calibration` that design 02 (the parameter set
and the prior) will draw on. These files were **untracked** in that repository's
`single-site-mcmc-vs-eki` worktree (`experiments/single_site_mcmc_vs_eki/output/`, worktree at
commit `f8da99b`). They were copied here on 2026-10-15 so that they are not lost. Nothing in
`lsmas` imports them.

| Path here | Copied from (under `output/`) | What it is |
|---|---|---|
| `parameter_analysis/phase1_report.md` | `analysis/` | Structural analysis of SIPNET's parameters: inert parameters, ridges, step functions |
| `parameter_analysis/phase2_report.md` | `analysis/` | Screening (Morris) and prior iteration for the 13-parameter calibration |
| `parameter_analysis/base_parameter_candidates.csv` | `analysis/` | All 58 pySIPNET parameters: base values, sources, BETY quantiles after PEcAn's conversion, inert/step flags |
| `parameter_analysis/prior_proposal.csv`, `fixed_parameters.csv`, `run_*.csv` | `analysis/` | The proposed priors, the fixed values, and prior-predictive summaries of candidate settings |
| `parameter_analysis/phase2_*.py` | `analysis/` | The scripts behind phase 2. `phase2_morris.py` holds the Morris screening ranges for 29 candidate parameters, with log or linear scale and a source for each |
| `parameter_analysis/phase2/*.csv` | `analysis/phase2/` | Phase 2 results: Morris μ* (absolute and relative), coverage, criteria and plausibility per prior iteration |
| `bety/samples.Rdata` | `bety/` | BETY trait samples for the site's plant functional type |
| `bety/prior_survey/` | `bety/prior_survey/` | BETY samples for three plant functional types (100 × 30 traits each), the trait-to-SIPNET conversion (`scc_prior_predictive.py`) and its prior-predictive summaries |

Not copied: the cached run outputs under `analysis/phase2/` (`cache.pkl`, `morris_runs.npz`,
`iter_*.npz`, about 55 MB) and `analysis/prior_predictive.npz` (3.6 MB). The scripts above
regenerate them.

Some of these files are also committed, in summary form, under
`experiments/single_site_mcmc_vs_eki/exploration/parameter_analysis/` on that repository's branches.
