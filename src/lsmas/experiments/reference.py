r"""Reference posteriors: computed once by long MCMC, stored in ``baselines/``, to score algorithms.

A reference holds draws of the posterior of one problem with one noise-scale setting, the
information matrix averaged over those draws,

$$
H_{\mathrm{ref}} = \mathbb E_{\mathrm{ref}}\!\left[DG(\xi)^\top \Gamma^{-1} DG(\xi)\right],
$$

each draw with its own $\Gamma$ when the noise scales are inferred, the chains' convergence
diagnostics, and a record of how it was computed. :func:`load_reference` reads the stored one.

Two recipes compute them:

- :func:`nuts_reference`, NUTS through ProbPipe, for a forward map JAX can differentiate (the toy);
- :func:`rwmh_reference`, long random-walk Metropolis chains run side by side, for SIPNET. A
  preliminary EKI run gives each chain its own starting particle and gives the proposal, the EKI
  ensemble's covariance in unconstrained coordinates scaled by $2.38 / \sqrt{D}$ for $D$
  coordinates. Each chain runs in segments saved to disk, so a run resumes after an interruption
  and continues until every coordinate has R-hat at most 1.01 and at least 1000 effective draws.
  The first segment is discarded as warmup. :func:`start_rwmh_reference`,
  :func:`run_rwmh_segment`, :func:`rwmh_diagnostics` and :func:`finish_rwmh_reference` are the
  steps, so that a cluster can run each chain's segments as separate jobs.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from probpipe import workflow_run

from lsmas._stopgaps.functions import declares_traceable
from lsmas._stopgaps.parallel_chains import run_in_processes
from lsmas.problems import load
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools import exact_subspace, finite_difference_subspace

__all__ = [
    "BASELINE_DIRECTORY",
    "Reference",
    "convergence_diagnostics",
    "finish_rwmh_reference",
    "load_reference",
    "noise_scale_setting",
    "nuts_reference",
    "reference_path",
    "run_rwmh_segment",
    "rwmh_diagnostics",
    "rwmh_reference",
    "start_rwmh_reference",
]

#: Where the stored references live in a checkout of the repository.
BASELINE_DIRECTORY = Path(__file__).resolve().parents[3] / "baselines"

#: The most draws a stored reference keeps, thinned evenly from its chains.
MAX_STORED_DRAWS = 4000

#: How many thinned draws the reference information matrix averages over.
INFORMATION_DRAWS = 200

#: The convergence criteria: the largest R-hat and the smallest bulk and tail effective sample
#: sizes allowed for every coordinate.
MAX_RHAT, MIN_EFFECTIVE_DRAWS = 1.01, 1000

ScaleSetting = Literal["fixed", "inferred"]


@dataclass(frozen=True, eq=False)
class Reference:
    r"""A reference posterior of one problem with one noise-scale setting.

    Attributes
    ----------
    problem_name : str
        The problem, such as ``"toy"``.
    scales : {"fixed", "inferred"}
        Whether the noise scales are fixed or inferred.
    parameter_names : tuple of str
        The calibration parameters, in the order of the last axis of :attr:`theta`.
    theta : numpy.ndarray
        Draws of the calibration parameters, shape (chains, draws, $d$).
    noise_scales : dict of str to numpy.ndarray
        Draws of each inferred noise scale, shape (chains, draws); empty when the scales are fixed.
    information : numpy.ndarray
        $H_{\mathrm{ref}}$, shape $(d, d)$.
    diagnostics : pandas.DataFrame
        One row per coordinate (each whitened parameter, and the logarithm of each inferred
        scale), with columns ``rhat``, ``ess_bulk`` and ``ess_tail``, computed on the chains before
        thinning.
    provenance : dict
        How the reference was computed: the method and its settings, the seed, the package
        versions, the repository's commit, the forward runs and the wall time.
    """

    problem_name: str
    scales: ScaleSetting
    parameter_names: tuple[str, ...]
    theta: np.ndarray
    noise_scales: dict[str, np.ndarray]
    information: np.ndarray
    diagnostics: pd.DataFrame
    provenance: dict[str, Any]

    def parameter_draws(self) -> np.ndarray:
        r"""Every draw of $\theta$ with the chains pooled, shape $(N, d)$."""
        return self.theta.reshape(-1, self.theta.shape[-1])

    def noise_scale_draws(self) -> dict[str, np.ndarray]:
        """Every draw of each inferred noise scale with the chains pooled, shape $(N,)$."""
        return {slot: draws.reshape(-1) for slot, draws in self.noise_scales.items()}

    def save(self, path: Path | None = None) -> Path:
        """Write the reference as netCDF, by default to its place in ``baselines/``.

        Returns
        -------
        Path
            Where it was written.
        """
        path = reference_path(self.problem_name, self.scales) if path is None else Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _reference_tree(self).to_netcdf(path, engine="h5netcdf")
        return path

    @classmethod
    def load(cls, path: Path) -> Reference:
        """Read a reference written by :meth:`save`."""
        tree = xr.open_datatree(path, engine="h5netcdf")
        posterior = tree["posterior"].to_dataset()
        attributes = tree.attrs
        return cls(
            problem_name=str(attributes["problem_name"]),
            scales=str(attributes["scales"]),
            parameter_names=tuple(str(name) for name in posterior["parameter"].values),
            theta=posterior["theta"].values,
            noise_scales={
                str(name): posterior[name].values for name in posterior.data_vars if name != "theta"
            },
            information=tree["information"]["information"].values,
            diagnostics=tree["diagnostics"].to_dataset().to_dataframe(),
            provenance=json.loads(attributes["provenance"]),
        )


def reference_path(problem_name: str, scales: ScaleSetting) -> Path:
    """Where the reference of a problem and scale setting is stored."""
    return BASELINE_DIRECTORY / problem_name / scales / "reference.nc"


def noise_scale_setting(problem: InverseProblem) -> ScaleSetting:
    """``"inferred"`` if *problem* infers its noise scales, ``"fixed"`` otherwise."""
    return "inferred" if problem.likelihood.random_scale_slots else "fixed"


def load_reference(problem: InverseProblem) -> Reference:
    """The stored reference of *problem* (its name and noise-scale setting).

    Raises
    ------
    FileNotFoundError
        If no reference has been computed for it.
    """
    path = reference_path(problem.name, noise_scale_setting(problem))
    if not path.exists():
        raise FileNotFoundError(f"no reference for {problem.name} ({noise_scale_setting(problem)})")
    return Reference.load(path)


def convergence_diagnostics(
    problem: InverseProblem, theta: np.ndarray, noise_scales: dict[str, np.ndarray]
) -> pd.DataFrame:
    r"""R-hat and bulk and tail effective sample sizes of each coordinate (Vehtari et al. 2021).

    The coordinates are the whitened parameters $\xi_j$ and the logarithm of each inferred scale.

    Parameters
    ----------
    problem : InverseProblem
        The problem whose coordinates whiten the parameters.
    theta : numpy.ndarray
        Draws, shape (chains, draws, $d$).
    noise_scales : dict of str to numpy.ndarray
        Draws of each inferred scale, shape (chains, draws).
    """
    coordinates = problem.coordinates
    variables = {
        name: coordinates.whitened_from_parameters(theta)[..., j]
        for j, name in enumerate(coordinates.names)
    } | {f"log_{slot}": np.log(draws) for slot, draws in noise_scales.items()}
    tree = az.from_dict({"posterior": variables})
    return pd.DataFrame(
        {
            "rhat": _per_variable(az.rhat(tree), variables),
            "ess_bulk": _per_variable(az.ess(tree, method="bulk"), variables),
            "ess_tail": _per_variable(az.ess(tree, method="tail"), variables),
        },
        index=pd.Index(list(variables), name="coordinate"),
    )


def nuts_reference(problem: InverseProblem, *, seed: int = 0) -> Reference:
    """The reference of a problem whose forward map JAX can differentiate, by ProbPipe's NUTS.

    Parameters
    ----------
    problem : InverseProblem
        The problem, such as the toy.
    seed : int
        The seed of the NUTS run.
    """
    start = time.time()
    with workflow_run(seed=seed):
        posterior = problem.condition(method="blackjax_nuts")
    theta = problem.coordinates.array_from_draws(posterior.atoms)
    noise_scales = {
        slot: np.asarray(posterior.atoms[slot].raw)
        for slot in problem.likelihood.random_scale_slots
    }
    return _build_reference(
        problem,
        theta,
        noise_scales,
        method="blackjax_nuts",
        settings={"seed": seed},
        forward_runs=None,
        wall_seconds=time.time() - start,
    )


def start_rwmh_reference(
    problem: InverseProblem, run_directory: Path, *, n_chains: int = 8, seed: int = 0
) -> None:
    r"""The first step of an RWMH reference: a preliminary EKI run that sets the chains' starts and
    the proposal.

    Writes ``settings.json`` and ``pilot.npz`` (the starts, shape (chains, $D$), and the proposal's
    Cholesky factor, shape $(D, D)$) to *run_directory*. The $D$ coordinates are the parameters
    followed by the inferred noise scales.

    Parameters
    ----------
    problem : InverseProblem
        The problem.
    run_directory : Path
        Where the run's files go.
    n_chains : int
        How many chains to run.
    seed : int
        The seed of the EKI run and the root of the chains' seeds.
    """
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    n_particles = 100
    with workflow_run(seed=seed):
        eki = problem.condition(method="enskit_eki", method_options={"n_particles": n_particles})
    states = _flat_states(problem, eki.atoms)
    unconstrained = _unconstrained_states(problem, states)
    proposal = np.linalg.cholesky(np.cov(unconstrained, rowvar=False))
    proposal *= 2.38 / np.sqrt(states.shape[1])
    chosen = np.random.default_rng(seed).choice(len(states), n_chains, replace=False)
    np.savez(run_directory / "pilot.npz", starts=states[chosen], proposal_cholesky=proposal)
    n_steps = eki.annotations["arviz/eki_history"]["beta"].size
    final_evaluation = 1 if problem.likelihood.random_scale_slots else 0
    settings = {
        "problem_name": problem.name,
        "scales": noise_scale_setting(problem),
        "n_chains": n_chains,
        "seed": seed,
        "pilot_forward_runs": n_particles * (n_steps + final_evaluation),
    }
    (run_directory / "settings.json").write_text(json.dumps(settings, indent=1))


def run_rwmh_segment(run_directory: Path, chain: int, segment: int, steps: int) -> Path:
    """Run one segment of one chain of an RWMH reference, continuing from the previous segment.

    Segment 0 starts at the chain's pilot start; segment $k$ starts at the last draw of segment
    $k - 1$. The proposal is the pilot's and stays fixed, so the segments together form one chain.
    Safe to run in a separate process or as a cluster job.

    Parameters
    ----------
    run_directory : Path
        The run's directory, as :func:`start_rwmh_reference` left it.
    chain, segment : int
        Which chain, and which of its segments.
    steps : int
        The segment's length.

    Returns
    -------
    Path
        The file the segment's draws were written to.
    """
    run_directory = Path(run_directory)
    settings = json.loads((run_directory / "settings.json").read_text())
    pilot = np.load(run_directory / "pilot.npz")
    start_state = (
        pilot["starts"][chain]
        if segment == 0
        else np.load(_segment_path(run_directory, chain, segment - 1))["states"][-1]
    )
    problem = load(settings["problem_name"], scales=settings["scales"])
    seed = int(np.random.SeedSequence([settings["seed"], chain, segment]).generate_state(1)[0])
    with workflow_run(seed=seed):
        posterior = problem.condition(
            method="blackjax_rwmh",
            method_options={
                "num_chains": 1,
                "num_warmup": 0,
                "num_results": steps,
                "init": start_state,
                "proposal_cov": pilot["proposal_cholesky"],
            },
        )
    path = _segment_path(run_directory, chain, segment)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, states=_flat_states(problem, posterior.atoms))
    return path


def rwmh_diagnostics(run_directory: Path) -> pd.DataFrame:
    """The convergence diagnostics of an RWMH reference run so far, without its first segment.

    Raises
    ------
    ValueError
        If no chain has two segments yet.
    """
    problem, theta, noise_scales = _completed_chains(Path(run_directory))
    return convergence_diagnostics(problem, theta, noise_scales)


def finish_rwmh_reference(run_directory: Path, wall_seconds: float | None = None) -> Reference:
    """The reference from an RWMH run's segments, its first segment discarded as warmup.

    Parameters
    ----------
    run_directory : Path
        The run's directory.
    wall_seconds : float, optional
        The run's wall time, for the provenance.
    """
    run_directory = Path(run_directory)
    settings = json.loads((run_directory / "settings.json").read_text())
    problem, theta, noise_scales = _completed_chains(run_directory)
    segment_files = sorted(run_directory.glob("chain_*/segment_*.npz"))
    chain_runs = sum(len(np.load(path)["states"]) + 1 for path in segment_files)
    return _build_reference(
        problem,
        theta,
        noise_scales,
        method="parallel_rwmh",
        settings={key: settings[key] for key in ("n_chains", "seed")}
        | {"segments_per_chain": len(segment_files) // settings["n_chains"]},
        forward_runs=settings["pilot_forward_runs"] + chain_runs,
        wall_seconds=wall_seconds,
    )


def rwmh_reference(
    problem: InverseProblem,
    run_directory: Path,
    *,
    n_chains: int = 8,
    segment_steps: int = 2000,
    max_segments: int = 50,
    seed: int = 0,
) -> Reference:
    """Compute an RWMH reference on this machine, one process per chain, resuming a stopped run.

    Runs segments of every chain until the diagnostics without the first segment meet the
    criteria (R-hat at most 1.01, bulk and tail effective sample sizes at least 1000 for every
    coordinate) or *max_segments* is reached.

    Parameters
    ----------
    problem : InverseProblem
        The problem.
    run_directory : Path
        Where the run's files go; a run already there is continued.
    n_chains : int
        How many chains, which is also how many processes run at once.
    segment_steps : int
        Each segment's length.
    max_segments : int
        The most segments per chain.
    seed : int
        The root seed.
    """
    run_directory = Path(run_directory)
    start = time.time()
    if not (run_directory / "pilot.npz").exists():
        start_rwmh_reference(problem, run_directory, n_chains=n_chains, seed=seed)
    for segment in range(max_segments):
        missing = [
            (run_directory, chain, segment, segment_steps)
            for chain in range(n_chains)
            if not _segment_path(run_directory, chain, segment).exists()
        ]
        if missing:
            run_in_processes(run_rwmh_segment, missing, max_workers=n_chains)
        if segment >= 1 and _meets_criteria(rwmh_diagnostics(run_directory)):
            break
    return finish_rwmh_reference(run_directory, wall_seconds=time.time() - start)


def _build_reference(
    problem: InverseProblem,
    theta: np.ndarray,
    noise_scales: dict[str, np.ndarray],
    *,
    method: str,
    settings: dict[str, Any],
    forward_runs: int | None,
    wall_seconds: float | None,
) -> Reference:
    """Diagnose, thin, compute the information matrix and record the provenance."""
    diagnostics = convergence_diagnostics(problem, theta, noise_scales)
    keep = _evenly_spaced(theta.shape[1], max(1, MAX_STORED_DRAWS // theta.shape[0]))
    theta, noise_scales = (
        theta[:, keep],
        {slot: draws[:, keep] for slot, draws in noise_scales.items()},
    )
    return Reference(
        problem_name=problem.name,
        scales=noise_scale_setting(problem),
        parameter_names=problem.coordinates.names,
        theta=theta,
        noise_scales=noise_scales,
        information=_reference_information(problem, theta, noise_scales),
        diagnostics=diagnostics,
        provenance=_provenance(method, settings, forward_runs, wall_seconds),
    )


def _reference_information(
    problem: InverseProblem, theta: np.ndarray, noise_scales: dict[str, np.ndarray]
) -> np.ndarray:
    r"""$H_{\mathrm{ref}}$ at evenly thinned draws: exact Jacobians if the forward map can be
    differentiated, central differences otherwise."""
    pooled = theta.reshape(-1, theta.shape[-1])
    keep = _evenly_spaced(len(pooled), INFORMATION_DRAWS)
    xi = problem.coordinates.whitened_from_parameters(pooled[keep])
    scale_values = {slot: draws.reshape(-1)[keep] for slot, draws in noise_scales.items()}
    estimator = (
        exact_subspace if declares_traceable(problem.forward) else finite_difference_subspace
    )
    return estimator(problem, xi, scale_values=scale_values or None).information


def _provenance(
    method: str, settings: dict[str, Any], forward_runs: int | None, wall_seconds: float | None
) -> dict[str, Any]:
    """How a reference was computed, with the versions of the packages that computed it."""
    return {
        "method": method,
        "settings": settings,
        "forward_runs": forward_runs,
        "wall_seconds": None if wall_seconds is None else round(wall_seconds, 1),
        "created": datetime.date.today().isoformat(),
        "packages": {
            package: _package_version(package)
            for package in ("lsmas", "probpipe-core", "enskit", "pysipnet")
        },
        "commit": _repository_commit(),
    }


def _package_version(package: str) -> str:
    """A package's version, with the commit it was installed from when there is one."""
    distribution = metadata.distribution(package)
    direct_url = distribution.read_text("direct_url.json")
    commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None
    return distribution.version + (f" ({commit[:12]})" if commit else "")


def _repository_commit() -> str | None:
    """The commit of the checkout this package runs from, if it is a git repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASELINE_DIRECTORY.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _flat_states(problem: InverseProblem, atoms: Any) -> np.ndarray:
    """Draws as flat states, shape $(N, D)$: the parameters, then the inferred noise scales."""
    theta = problem.coordinates.array_from_draws(atoms).reshape(-1, problem.coordinates.dimension)
    scales = [
        np.asarray(atoms[slot].raw).reshape(-1, 1) for slot in problem.likelihood.random_scale_slots
    ]
    return np.concatenate([theta, *scales], axis=1)


def _unconstrained_states(problem: InverseProblem, states: np.ndarray) -> np.ndarray:
    r"""Flat states in unconstrained coordinates: $z = b^{-1}(\theta)$, then $\log s^2$."""
    dimension = problem.coordinates.dimension
    return np.concatenate(
        [
            problem.coordinates.bijection.to_unconstrained(states[:, :dimension]),
            np.log(states[:, dimension:]),
        ],
        axis=1,
    )


def _completed_chains(
    run_directory: Path,
) -> tuple[InverseProblem, np.ndarray, dict[str, np.ndarray]]:
    """The run's problem and its chains without their first segment, cut to a common length."""
    settings = json.loads((run_directory / "settings.json").read_text())
    problem = load(settings["problem_name"], scales=settings["scales"])
    chains = []
    for chain in range(settings["n_chains"]):
        segments = sorted((run_directory / f"chain_{chain:02d}").glob("segment_*.npz"))[1:]
        if segments:
            chains.append(np.concatenate([np.load(path)["states"] for path in segments]))
    if len(chains) < 2:
        raise ValueError("diagnostics need at least two chains with two segments each")
    length = min(len(states) for states in chains)
    states = np.stack([states[:length] for states in chains])
    dimension = problem.coordinates.dimension
    noise_scales = {
        slot: states[..., dimension + k]
        for k, slot in enumerate(problem.likelihood.random_scale_slots)
    }
    return problem, states[..., :dimension], noise_scales


def _segment_path(run_directory: Path, chain: int, segment: int) -> Path:
    """Where one segment of one chain is saved."""
    return Path(run_directory) / f"chain_{chain:02d}" / f"segment_{segment:03d}.npz"


def _meets_criteria(diagnostics: pd.DataFrame) -> bool:
    """Whether every coordinate has R-hat at most 1.01 and effective sample sizes of 1000."""
    return bool(
        (diagnostics["rhat"] <= MAX_RHAT).all()
        and (diagnostics[["ess_bulk", "ess_tail"]] >= MIN_EFFECTIVE_DRAWS).all(axis=None)
    )


def _evenly_spaced(length: int, count: int) -> np.ndarray:
    """At most *count* evenly spaced indices into a sequence of *length*."""
    return np.unique(np.linspace(0, length - 1, min(count, length)).round().astype(int))


def _per_variable(result: Any, variables: dict[str, Any]) -> list[float]:
    """One diagnostic's value for each variable, in order."""
    return [float(np.asarray(result[name])) for name in variables]


def _reference_tree(reference: Reference) -> xr.DataTree:
    """The reference as a DataTree: posterior, information and diagnostics, provenance as JSON."""
    dims = ("chain", "draw")
    posterior = xr.Dataset(
        {"theta": ((*dims, "parameter"), reference.theta)}
        | {slot: (dims, draws) for slot, draws in reference.noise_scales.items()},
        coords={"parameter": list(reference.parameter_names)},
    )
    information = xr.Dataset(
        {"information": (("parameter", "parameter_dual"), reference.information)},
        coords={
            "parameter": list(reference.parameter_names),
            "parameter_dual": list(reference.parameter_names),
        },
    )
    diagnostics = xr.Dataset.from_dataframe(reference.diagnostics)
    tree = xr.DataTree.from_dict(
        {"posterior": posterior, "information": information, "diagnostics": diagnostics}
    )
    tree.attrs = {
        "problem_name": reference.problem_name,
        "scales": reference.scales,
        "provenance": json.dumps(reference.provenance),
    }
    return tree
