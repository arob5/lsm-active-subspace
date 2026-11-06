# Prototype for design 02 (the inverse problem)

Scratch code that backs the claims in [`../../02-inverse-problem.md`](../../02-inverse-problem.md).
It runs in the repository's own environment and is not the repository's code. Outputs go to `out/`.

| File | What it checks |
|---|---|
| `reference.py` | Shared pieces: the reference point (all 58 SIPNET parameters), threaded SIPNET runs, the features |
| `prior_table.py` | The draft prior over the 46 calibration parameters, and their map to the 58 SIPNET parameters |
| `t1_active_set.py` | Which SIPNET parameters move the 400 features at the reference point; which are exactly inert |
| `t2_degeneracies.py` | The three exact degeneracies, bit for bit, with controls |
| `t3_jacobian_spectrum.py` | No further exact ratio degeneracy: the weighted Jacobian's spectrum in log coordinates |
| `t4_prior_predictive.py` | Prior predictive check: failures, coverage of the observed features, plausibility of the runs |
| `t5_median_shift.py` | The prior-median run against the reference run, one parameter at a time |
| `t6_group_spread.py` | How much of the prior predictive spread comes from the phase-2 parameters and from the rest |
| `t7_prior_importance.py` | Prior-averaged information matrix in whitened coordinates (activity scores, spectrum); every fixed-as-inert parameter checked across the prior |
| `t8_frozen_soil.py` | The three frozen-soil parameters act in parts of the prior, though inert at the reference point |
| `make_tables.py` | Prints the design's table of calibration parameters and priors from `prior_table.py` |

```bash
cd design/prototypes/02-inverse-problem
uv run python -W ignore t1_active_set.py          # ~10 s
uv run python -W ignore t4_prior_predictive.py 400 final   # ~30 s
uv run python -W ignore t7_prior_importance.py 80 3   # ~8 min
```
