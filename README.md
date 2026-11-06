# lsm-active-subspace

Calibrating the SIPNET land-surface model to eddy-covariance data from Harvard Forest, with
algorithms that learn which directions in parameter space the data actually inform.

The inverse problem (the model, the data, the likelihood and the prior) is fixed and shipped with
the repository. The work done here is algorithmic: building, combining and comparing parameter
reduction methods.

> **Status:** under construction. The probability model, the SIPNET forward map, ensemble Kalman
> inversion, the parameter-reduction tools and the first algorithm work; the experiment harness and
> tutorials are next.

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

## A first look

```python
from probpipe import sample, workflow_run

from lsmas.problems import load

problem = load("harvard_forest_small")
print(problem.describe())

with workflow_run(seed=1):
    draws = sample(problem.model, 16)              # prior predictive: runs SIPNET 16 times
    posterior = problem.condition(method="enskit_eki", method_options={"n_particles": 40})
```

Three problems ship. `harvard_forest` calibrates all 46 parameters. `harvard_forest_small`
calibrates ten of them and holds the rest at their prior medians, so that a long MCMC run, and with
it a trustworthy reference posterior, is affordable. `toy` is a simplified SIPNET written in JAX:
it can be differentiated exactly and sampled with NUTS, so it is where methods are checked.
[Design 02](design/02-inverse-problem.md) explains how the Harvard Forest problems were chosen.

## The first algorithm

```python
from lsmas.algorithms import active_subspace_mcmc

result = active_subspace_mcmc(load("toy", scales="inferred"), seed=1)
result.subspace.rank                    # how many directions the data inform
result.subspace.activity_scores()       # which parameters matter, and how much
result.posterior                        # draws of every parameter and the noise scales
result.forward_runs                     # {'exploration': 600, 'sampling': 6004}
```

It explores with ensemble Kalman inversion, estimates the active subspace from the ensemble's
linearizations along the way, samples the reduced posterior of the active variables with
random-walk Metropolis, and lifts the draws back to every parameter. Each step is a tool in
`lsmas.tools` that can be swapped: another estimator of the subspace, another rank, another
sampler. [Design 03](design/03-algorithms.md) explains the choices.

`condition` requires an inference method: ProbPipe's automatic choice assumes gradients, which
SIPNET does not have.

## Layout

```
data/harvard_forest/   the shipped inputs: driver, NEE features, observation operator, fixed SIPNET values, prior
src/lsmas/
    data.py            loads the shipped inputs
    sipnet.py          runs SIPNET and builds the forward map
    parameters.py      maps calibration parameters to SIPNET parameters
    priors.py          prior families specified by intervals; the prior from a table
    learning_rate.py   the likelihood's learning rate
    problems/          inverse problems, ready to condition: harvard_forest, harvard_forest_small, toy
    inference/         ensemble Kalman inversion, registered with ProbPipe
    tools/             subspaces, their estimators, and reduced problems
    algorithms/        complete parameter-reduction algorithms built from the tools
    _stopgaps/         temporary code that fills gaps in ProbPipe
tests/                 unit tests; `-m smoke` selects the end-to-end checks that run SIPNET
docs/notation.md       the mathematical notation every docstring uses
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
