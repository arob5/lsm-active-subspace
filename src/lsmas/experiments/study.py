r"""Studies: one algorithm run over settings and seeds, each run scored against a reference.

Each run is saved in its own directory, named by a hash of the problem, the noise-scale setting,
the algorithm, the setting and the seed. A run whose directory is complete is read rather than
rerun, so an interrupted study resumes where it stopped, and a study given more seeds or settings
runs only the new ones. :func:`run_study` returns one table row per run.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from lsmas.algorithms.result import AlgorithmResult
from lsmas.experiments.metrics import score, score_names
from lsmas.experiments.reference import Reference, ScaleSetting, load_reference
from lsmas.problems import load_inverse_problem
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools.subspace import Subspace

__all__ = ["StoredRun", "Study", "load_run", "run_study"]

#: The file that marks a run as complete; it is written last.
_SCORES_FILE = "scores.json"
_RUN_FILE = "run.nc"


@dataclass(frozen=True)
class Study:
    """One algorithm, run with each setting and each seed on one problem.

    Attributes
    ----------
    name : str
        A short name for the study.
    problem : str
        The problem's name, such as ``"toy"``.
    scales : {"fixed", "inferred"}
        The problem's noise-scale setting.
    algorithm : callable
        Called as ``algorithm(problem, seed=seed, **setting)``; returns an ``AlgorithmResult``.
    settings : sequence of Mapping
        The keyword arguments of each setting; their values must be JSON-serializable.
    seeds : sequence of int
        The seeds every setting is run with.
    """

    name: str
    problem: str
    scales: ScaleSetting
    algorithm: Callable[..., AlgorithmResult]
    settings: Sequence[Mapping[str, Any]]
    seeds: Sequence[int]


@dataclass(frozen=True, eq=False)
class StoredRun:
    """One saved run of a study.

    Attributes
    ----------
    setting : dict
        The run's keyword arguments.
    seed : int
        Its seed.
    theta : numpy.ndarray
        Its posterior draws of the calibration parameters, shape $(N, d)$.
    noise_scales : dict of str to numpy.ndarray
        Its posterior draws of each inferred noise scale, shape $(N,)$.
    subspace : Subspace
        The subspace it estimated.
    forward_runs : dict of str to int
        Its forward runs by stage.
    wall_seconds : float
        Its wall time.
    scores : dict
        Its table row.
    """

    setting: dict[str, Any]
    seed: int
    theta: np.ndarray
    noise_scales: dict[str, np.ndarray]
    subspace: Subspace
    forward_runs: dict[str, int]
    wall_seconds: float
    scores: dict[str, Any]


def run_study(study: Study, directory: Path, *, reference: Reference | None = None) -> pd.DataFrame:
    """Run every setting with every seed, skipping runs already complete, and tabulate them.

    Parameters
    ----------
    study : Study
        The study.
    directory : Path
        Where the runs are saved, one subdirectory per run.
    reference : Reference, optional
        The reference to score against; by default the stored one for the problem.

    Raises
    ------
    ValueError
        If *reference* is for another problem or noise-scale setting, or a setting is named like
        another column of the table.

    Returns
    -------
    pandas.DataFrame
        One row per run: the setting's values, the seed, the scores of
        :func:`lsmas.experiments.metrics.score`, the wall time and the run's directory name.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    problem = load_inverse_problem(study.problem, scales=study.scales)
    reference = load_reference(problem) if reference is None else reference
    if (reference.problem_name, reference.scales) != (study.problem, study.scales):
        raise ValueError(
            f"the reference is for {reference.problem_name} ({reference.scales}), "
            f"the study for {study.problem} ({study.scales})"
        )
    reserved = {*score_names(problem), "seed", "wall_seconds", "run"}
    if clashing := sorted({name for setting in study.settings for name in setting} & reserved):
        raise ValueError(f"settings may not be named like the table's other columns: {clashing}")
    rows = []
    for setting in study.settings:
        for seed in study.seeds:
            run_directory = directory / _run_key(study, setting, seed)
            if not (run_directory / _SCORES_FILE).exists():
                _run_and_save(study, problem, reference, dict(setting), seed, run_directory)
            rows.append(json.loads((run_directory / _SCORES_FILE).read_text()))
    return pd.DataFrame(rows)


def load_run(run_directory: Path) -> StoredRun:
    """Read one saved run of a study, for plotting or further scoring."""
    run_directory = Path(run_directory)
    tree = xr.open_datatree(run_directory / _RUN_FILE, engine="h5netcdf")
    posterior = tree["posterior"].to_dataset()
    subspace = tree["subspace"].to_dataset()
    attributes = tree.attrs
    return StoredRun(
        setting=json.loads(attributes["setting"]),
        seed=int(attributes["seed"]),
        theta=posterior["theta"].values,
        noise_scales={
            str(name): posterior[name].values for name in posterior.data_vars if name != "theta"
        },
        subspace=Subspace(
            eigenvalues=subspace["eigenvalues"].values,
            eigenvectors=subspace["eigenvectors"].values,
            rank=int(subspace.attrs["rank"]),
            parameter_names=tuple(str(name) for name in posterior["parameter"].values),
            estimator=str(subspace.attrs["estimator"]),
        ),
        forward_runs=json.loads(attributes["forward_runs"]),
        wall_seconds=float(attributes["wall_seconds"]),
        scores=json.loads((run_directory / _SCORES_FILE).read_text()),
    )


def _run_key(study: Study, setting: Mapping[str, Any], seed: int) -> str:
    """A short hash of everything that determines a run."""
    algorithm = f"{study.algorithm.__module__}.{study.algorithm.__qualname__}"
    description = {
        "problem": study.problem,
        "scales": study.scales,
        "algorithm": algorithm,
        "setting": dict(setting),
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:16]


def _run_and_save(
    study: Study,
    problem: InverseProblem,
    reference: Reference,
    setting: dict[str, Any],
    seed: int,
    run_directory: Path,
) -> None:
    """Run the algorithm once, save its result, then its scores, which mark the run complete."""
    start = time.time()
    result = study.algorithm(problem, seed=seed, **setting)
    wall_seconds = time.time() - start
    run_directory.mkdir(parents=True, exist_ok=True)
    _result_tree(result, problem, setting, seed, wall_seconds).to_netcdf(
        run_directory / _RUN_FILE, engine="h5netcdf"
    )
    row = (
        setting
        | {"seed": seed}
        | score(result, reference, problem)
        | {"wall_seconds": round(wall_seconds, 1), "run": run_directory.name}
    )
    (run_directory / _SCORES_FILE).write_text(json.dumps(row))


def _result_tree(
    result: AlgorithmResult,
    problem: InverseProblem,
    setting: dict[str, Any],
    seed: int,
    wall_seconds: float,
) -> xr.DataTree:
    """A run's posterior draws and subspace as a DataTree, with the run's settings as attributes."""
    theta = problem.coordinates.array_from_draws(result.posterior.atoms).reshape(
        -1, problem.coordinates.dimension
    )
    posterior = xr.Dataset(
        {"theta": (("draw", "parameter"), theta)}
        | {
            slot: ("draw", np.asarray(result.posterior.atoms[slot].raw).reshape(-1))
            for slot in problem.likelihood.random_scale_slots
        },
        coords={"parameter": list(problem.coordinates.names)},
    )
    subspace = xr.Dataset(
        {
            "eigenvalues": ("index", result.subspace.eigenvalues),
            "eigenvectors": (("parameter", "index"), result.subspace.eigenvectors),
        },
        coords={"parameter": list(problem.coordinates.names)},
        attrs={"rank": result.subspace.rank, "estimator": result.subspace.estimator},
    )
    tree = xr.DataTree.from_dict({"posterior": posterior, "subspace": subspace})
    tree.attrs = {
        "setting": json.dumps(setting),
        "seed": seed,
        "forward_runs": json.dumps(result.forward_runs),
        "wall_seconds": wall_seconds,
    }
    return tree
