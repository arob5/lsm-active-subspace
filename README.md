# lsm-active-subspace

Calibrating the SIPNET land-surface model to eddy-covariance data from Harvard Forest, with
algorithms that find the directions in parameter space that the data inform.

The inverse problem (the model, the data, the likelihood and the prior) is fixed and shipped with
the repository. This repository is for building and comparing parameter-reduction algorithms.

> **Status:** under construction. The tutorial on the inverse problem is ready; tutorials on
> the algorithms are next.

## Getting started

These steps work on macOS and Linux. On Windows, work inside
[WSL](https://learn.microsoft.com/windows/wsl/install) (Ubuntu) and follow the Linux steps.

**1. Install the tools.** You need [git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/), which installs Python and every package the project uses:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal, so that `uv` is on your path.

**2. Get the code and its environment.**

```bash
git clone https://github.com/arob5/lsm-active-subspace.git
```

```bash
cd lsm-active-subspace
```

```bash
uv sync
```

`uv sync` creates the environment in `.venv/`, with the right Python version. Run every command
below from the `lsm-active-subspace` folder.

**3. Install SIPNET.** The model itself is a separate program:

```bash
uv run pysipnet install-sipnet
```

On an Apple Silicon Mac or on x86-64 Linux this downloads a prebuilt copy. Elsewhere it compiles
SIPNET, which needs `make` and a C compiler (on a Mac, run `xcode-select --install` first).

**4. Check the setup** (a few minutes; it runs SIPNET and the tutorial):

```bash
uv run pytest -m smoke
```

## The tutorial

Start with [`notebooks/main_inverse_problem.ipynb`](notebooks/main_inverse_problem.ipynb). It
introduces the Harvard Forest inverse problem: the data, the prior, the forward model and the
likelihood, one at a time, and how to work with each in code. It is saved with its outputs, so you
can read it on GitHub first.

To run it, open it in JupyterLab:

```bash
uv run jupyter lab notebooks/main_inverse_problem.ipynb
```

Or in VS Code: open the `lsm-active-subspace` folder, install the Python and Jupyter extensions,
open the notebook, and choose the kernel `.venv/bin/python` (**Select Kernel** at the top right).

The whole notebook runs in under a minute. To experiment, work in a copy, so that updating the
code never conflicts with your changes:

```bash
cp notebooks/main_inverse_problem.ipynb notebooks/my_main_inverse_problem.ipynb
```

The notation is listed in [`docs/notation.md`](docs/notation.md).

## Staying up to date

```bash
git pull
```

```bash
uv sync
```

## A first look

```python
from probpipe import sample, workflow_run

from lsmas.problems import load_inverse_problem

problem = load_inverse_problem("harvard_forest_small")
print(problem.describe())

with workflow_run(seed=1):
    draws = sample(problem.model, 16)              # prior predictive: runs SIPNET 16 times
    posterior = problem.condition(method="enskit_eki", method_options={"n_particles": 40})
```

Three problems ship. `harvard_forest` calibrates all 46 parameters. `harvard_forest_small`
calibrates ten of them and holds the rest at their prior medians, so that a long MCMC run, and with
it a trustworthy reference posterior, is affordable. `toy` is a simplified SIPNET written in JAX:
it can be differentiated exactly and sampled with NUTS, so methods can be tested on it against
exact Jacobians and a reference posterior.
[Design 02](design/02-inverse-problem.md) explains how the Harvard Forest problems were chosen.

## The first algorithm

```python
from lsmas.algorithms import active_subspace_mcmc

result = active_subspace_mcmc(load_inverse_problem("toy", scales="inferred"), seed=1)
result.subspace.rank                    # the number of active directions
result.subspace.activity_scores()       # each parameter's activity score
result.posterior                        # draws of every parameter and the noise scales
result.forward_runs                     # {'exploration': 600, 'sampling': 6004}
```

It runs ensemble Kalman inversion (EKI), estimates the active subspace from statistical
linearizations of the EKI ensembles, samples the posterior of the active variables with
random-walk Metropolis, and maps the draws back to every parameter. Each step is a tool in
`lsmas.tools` that can be swapped: another estimator of the subspace, another rank, another
sampler. [Design 03](design/03-algorithms.md) explains the choices.

`condition` requires an inference method: ProbPipe's automatic choice assumes gradients, which
SIPNET does not have.

## Comparing algorithms

A study runs one algorithm with several settings and seeds and scores every run against a stored
reference posterior:

```python
from lsmas.algorithms import active_subspace_mcmc
from lsmas.experiments import Study, run_study
from lsmas.plots import plot_metric_against_setting

study = Study(
    name="toy_rank",
    problem="toy",
    scales="fixed",
    algorithm=active_subspace_mcmc,
    settings=[{"rank": rank} for rank in (3, 5, 7, 9)],
    seeds=range(5),
)
table = run_study(study, directory="runs/studies/toy_rank")   # one row per run; reruns are free
plot_metric_against_setting(table, "rank", "marginal_w1_mean")
```

The headline score, `marginal_w1_mean`, is the 1-Wasserstein distance between each parameter's
marginal and the reference's, in whitened coordinates and reference standard deviations, averaged
over parameters; about 0.04 is Monte Carlo noise. `lsmas.experiments.metrics` defines every score,
and `lsmas.plots` draws the standard figures. [Design 04](design/04-experiments.md) explains both.

Stored references (`baselines/`):

| Problem | Method | Draws | Cost of each noise-scale setting |
|---|---|---|---|
| `toy` | NUTS, 4 chains | 4000 | about 10 minutes |
| `harvard_forest_small` | parallel random-walk Metropolis, 8 chains | 4000 | about 97,000 SIPNET runs, 1.5 hours on 8 cores |
| `harvard_forest` | not computed yet (a cluster job) | | |

Each is stored for both the fixed and the inferred noise scales.

`scripts/compute_reference.py` computes a reference, on this machine or as cluster jobs (see its
help).

## Layout

```
data/harvard_forest/   the shipped inputs: driver, NEE features, observation operator, fixed SIPNET values, prior
notebooks/             the walkthrough notebook
src/lsmas/
    data.py            loads the shipped inputs
    sipnet.py          runs SIPNET and builds the forward map
    parameters.py      maps calibration parameters to SIPNET parameters
    priors.py          prior families specified by intervals; the prior from a table
    learning_rate.py   the likelihood's learning rate
    problems/          the inverse problems: harvard_forest, harvard_forest_small, toy
    inference/         ensemble Kalman inversion, registered with ProbPipe
    tools/             subspaces, their estimators, and reduced problems
    algorithms/        complete parameter-reduction algorithms built from the tools
    experiments/       reference posteriors, metrics, and studies
    plots.py           the standard figures
    _stopgaps/         temporary code that fills gaps in ProbPipe
baselines/             the stored reference posteriors
scripts/               command-line tools, such as computing a reference
tests/                 unit tests; `-m smoke` selects the end-to-end checks that run SIPNET or a notebook
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
