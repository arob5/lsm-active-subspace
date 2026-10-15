# Prototype for design 01 (architecture)

Scratch code that backs the claims in [`../../01-architecture.md`](../../01-architecture.md).
It is **not** the repo's code: it exists to check that SIPNET, ProbPipe and EnsKit fit together
the way the design says, and to record the numbers quoted there.

## Contents

| File | What it checks |
|---|---|
| `lsmas_proto.py` | The pieces: SIPNET forward map as a ProbPipe `Function`, the prior, and (draft 1, superseded) a lazy-law likelihood with η inside it |
| `lsmas_backend.py` | The proper likelihood (`GaussianSimulatorLikelihood`), the private tempered factor and `tempered_condition_on`, and an EnsKit EKI `InferenceMethod` |
| `t1_basics.py` | One SIPNET run, time-axis alignment with A, learning rates, IG prior median, no simulation at model construction |
| `t2_dispatch.py` | What `condition_on` selects, joint sampling, batched prior predictive (thread vs sequential), pushforward, no caching |
| `t3_eki.py` | End to end: tempered conditioning with `method="enskit_eki"` and a per-step statistical-linearization callback |
| `t4_joint.py` | Joint sampling with the subclass likelihood; sampling the likelihood at fixed parameters |
| `t5_rwmh.py` | ProbPipe's built-in RWMH on the SIPNET model (needs EnsKit imported first) |
| `t6_tempered.py` | The likelihood is one law; the tempered target's density is exactly log π + Σ ηᵢ log Nᵢ; EKI and RWMH on it |
| `build_A.py` | Rebuilds the 400-feature operator from the Harvard Forest step table, with `hf_operators.py` |
| `data/A.npz`, `data/features.csv` | The operator and feature table (`y`, class, year, month, half, …) |
| `data/step_axis.json` | The 3-hourly step axis A's columns refer to |

## Running

The environment pins the exact commits the design uses (`pyproject.toml`):

```bash
uv sync
```

```bash
uv run python t3_eki.py
```

The scripts read the corrected Harvard Forest ERA5 driver from the `spatial-lsm-calibration`
worktree path in `lsmas_proto.py`. Set `LSMAS_DRIVER` to point elsewhere. `build_A.py` needs
`hf_operators.py` (a copy of `exploration/nee_error_model/operators.py` from the
`hf-nee-error-model` worktree) and that worktree's untracked `output/nee_error_model/steps.pkl`:

```bash
python build_A.py <path>/steps.pkl <dir containing hf_operators.py> data
```

## Caveats

- The calibration parameters here are a 4-parameter subset with placeholder priors and
  representative fixed values. The real parameter set and prior are design 02's job.
- The EKI method's handling of the inverse-gamma scales is placeholder plumbing. The algorithm
  is design 03's job.
