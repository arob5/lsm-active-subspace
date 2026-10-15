# lsm-active-subspace

Calibrating the SIPNET land-surface model to eddy-covariance data from Harvard Forest, with
algorithms that learn which directions in parameter space the data actually inform.

The inverse problem (the model, the data, the likelihood and the prior) is fixed and shipped with
the repository. The work done here is algorithmic: building, combining and comparing parameter
reduction methods.

> **Status:** under construction. The probability model, the SIPNET forward map and ensemble
> Kalman inversion work; parameter-reduction tools, algorithms and tutorials are next.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage its environment.

```bash
uv sync
```

Check that everything works:

```bash
uv run pytest
```

The first run builds SIPNET, which takes a minute.

## Layout

```
data/harvard_forest/   the shipped inputs: driver, NEE features, observation operator, fixed SIPNET values
src/lsmas/
    data.py            loads the shipped inputs
    sipnet.py          runs SIPNET and builds the forward map
    parameters.py      maps calibration parameters to SIPNET parameters
    priors.py          prior families specified by intervals
    learning_rate.py   the likelihood's learning rate
    problems/          inverse problems, ready to condition
    predictive.py      prior predictive draws
    inference/         ensemble Kalman inversion, registered with ProbPipe
    _stopgaps/         temporary code that fills gaps in ProbPipe
tests/                 unit tests; `-m smoke` selects the end-to-end checks that run SIPNET
design/                design documents
```

## Upgrading ProbPipe, EnsKit or pySIPNET

All three are pinned to exact commits in `pyproject.toml`, because they change quickly. To move to a
newer version:

1. Replace the commit hash after `@` in that package's line of `pyproject.toml`.
2. Update the lock file and the environment:

   ```bash
   uv lock --upgrade-package probpipe-core
   ```

   ```bash
   uv sync
   ```

   (Use `enskit` or `pysipnet` in place of `probpipe-core` as appropriate.)
3. Run the end-to-end checks:

   ```bash
   uv run pytest -m smoke
   ```
