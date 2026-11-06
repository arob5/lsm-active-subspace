# Prototype for design 03 (the algorithms)

Scratch code that backs the claims in [`../../03-algorithms.md`](../../03-algorithms.md). It runs in
the repository's environment and is not the repository's code. Outputs go to `out/` (arrays are not
committed; the printed results are).

| File | What it checks |
|---|---|
| `toy.py` | The toy model: SIPNET's canopy equations in JAX; speed and realism at the prior medians |
| `problem.py` | The toy as an `InverseProblem`, built as `lsmas` builds the Harvard Forest problems |
| `common.py` | Shared pieces: the toy in whitened coordinates, its Jacobians, the information matrices |
| `t1_toy_nuts.py` | NUTS reference posteriors through `problem.condition` (fixed or inferred scales) |
| `t2_estimators.py` | Exact versus statistical-linearization information matrices: prior, ladder, last level |
| `t3_compare.py` | How much of the posterior's information each estimated subspace captures |
| `t4_reduced.py` | Reduced posteriors (optimal, pseudo-marginal, plug-in) against NUTS, by rank |
| `t5_eki_gibbs.py` | EKI with Gibbs noise-scale updates, from EnsKit's step API, against NUTS |
| `t6_probpipe_reduced.py` | The plug-in reduced problem sampled by ProbPipe's `blackjax_rwmh` |
| `t7_sipnet_small.py` | The proposed algorithm end to end on `harvard_forest_small` |
| `t8_full_rwmh.py` | Baseline: full-dimensional `blackjax_rwmh` on the toy at the same budget |

```bash
cd design/prototypes/03-algorithms
uv run python -W ignore t1_toy_nuts.py observed fixed      # ~11 min; needed by t3-t6, t8
uv run python -W ignore t1_toy_nuts.py observed inferred   # ~15 min; needed by t5, t6 inferred
uv run python -W ignore t2_estimators.py 100 1             # ~30 s
uv run python -W ignore t3_compare.py out/estimators_J100_s1.npz
uv run python -W ignore t4_reduced.py out/estimators_J100_s1.npz   # ~20 min
uv run python -W ignore t7_sipnet_small.py 100 60          # ~3 min
```
