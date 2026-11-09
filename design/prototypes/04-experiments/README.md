# Prototype for design 04 (experiments)

Scratch code that backs the claims in [`../../04-experiments.md`](../../04-experiments.md). Arrays
in `out/` are not committed; printed results and figures are.

| File | What it checks |
|---|---|
| `t1_reference_trial.py` | A SIPNET reference by parallel random-walk chains, one process each: cost and mixing |
| `t2_metrics.py` | Candidate metrics on the toy against NUTS, with their noise floor |
| `t3_study.py` | A study runner with a cache, its table, and the standard plots |
| `t4_pilot_proposal.py` | Whether a proposal tuned on pilot draws mixes better than EKI's |
| `t5_shadow_plot.py` | Sufficient-summary plots from an EKI run's history |

```bash
cd design/prototypes/04-experiments
uv run python -W ignore t1_reference_trial.py harvard_forest_small 8 100 400     # ~5 min
uv run python -W ignore t2_metrics.py                                             # ~1 min; needs design 03's NUTS draws
uv run --with h5py --with h5netcdf --with matplotlib python -W ignore t3_study.py # ~1 min
```
