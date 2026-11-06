"""A study runner, its cache, its metrics table and the standard plots, on the toy.

A study runs one algorithm over a list of settings and independent seeds on one problem. Each run
is saved under a key that hashes its problem, settings and seed, so an interrupted study resumes
where it stopped. Each run is scored against the stored reference posterior and the reference
information matrix, giving one row of a tidy table.

Usage: uv run --with h5py --with h5netcdf --with matplotlib python t3_study.py
"""

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402

from lsmas.algorithms import active_subspace_mcmc  # noqa: E402
from lsmas.problems import load  # noqa: E402
from lsmas.tools import exact_subspace  # noqa: E402

HERE = Path(__file__).resolve().parent
NUTS = HERE.parents[0] / "03-algorithms" / "out" / "nuts_observed_fixed.npz"
STUDY = HERE / "out" / "study_toy_rank"
SHORT_CHAINS = {"num_chains": 2, "num_warmup": 200, "num_results": 500}

toy = load("toy")
coordinates = toy.coordinates
reference_xi = np.load(NUTS)["xi"].reshape(-1, 14)
reference_information = exact_subspace(toy, reference_xi[::8]).information
reference_eigenvectors = np.linalg.eigh(reference_information)[1][:, ::-1]


def run_key(settings, seed):
    text = json.dumps({"problem": "toy", "scales": "fixed", "settings": settings, "seed": seed}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def run_once(settings, seed):
    directory = STUDY / run_key(settings, seed)
    if (directory / "metrics.json").exists():
        return json.loads((directory / "metrics.json").read_text())
    start = time.time()
    result = active_subspace_mcmc(toy, sampler_options=SHORT_CHAINS, seed=seed, **settings)
    seconds = time.time() - start
    xi = coordinates.whitened_from_parameters(coordinates.array_from_draws(result.posterior.atoms))
    directory.mkdir(parents=True, exist_ok=True)
    xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({"xi": (("draw", "parameter"), xi)}, coords={"parameter": list(coordinates.names)}),
            "subspace": xr.Dataset(
                {
                    "eigenvalues": ("index", result.subspace.eigenvalues),
                    "eigenvectors": (("parameter", "index"), result.subspace.eigenvectors),
                },
                attrs={"rank": result.subspace.rank, "estimator": result.subspace.estimator},
            ),
        }
    ).to_netcdf(directory / "result.nc", engine="h5netcdf")
    row = {"seed": seed, **settings, "rank_used": result.subspace.rank, "seconds": round(seconds, 1),
           "forward_runs": result.total_forward_runs, **score(xi, result.subspace)}
    (directory / "metrics.json").write_text(json.dumps(row))
    return row


def score(xi, subspace):
    sd = reference_xi.std(0)
    w1 = np.array([stats.wasserstein_distance(xi[:, j], reference_xi[:, j]) for j in range(14)]) / sd
    mean_error = np.abs(xi.mean(0) - reference_xi.mean(0)) / sd
    r = subspace.rank
    best = np.sort(np.linalg.eigvalsh(reference_information))[::-1][:r].sum()
    captured = np.trace(subspace.basis.T @ reference_information @ subspace.basis) / best
    cosines = np.linalg.svd(subspace.basis.T @ reference_eigenvectors[:, :r], compute_uv=False)
    return {
        "marginal_w1_mean": round(float(w1.mean()), 4),
        "marginal_w1_max": round(float(w1.max()), 4),
        "mean_error_mean": round(float(mean_error.mean()), 4),
        "captured_information": round(float(captured), 4),
        "largest_principal_angle_deg": round(float(np.degrees(np.arccos(np.clip(cosines.min(), -1, 1)))), 2),
    }


start = time.time()
rows = [run_once({"rank": rank}, seed) for rank in (3, 5, 7, 9) for seed in (1, 2, 3)]
table = pd.DataFrame(rows)
print(f"study: {len(rows)} runs, {time.time() - start:.0f} s (cached runs are free)")
print(table.groupby("rank")[["marginal_w1_mean", "marginal_w1_max", "captured_information", "largest_principal_angle_deg", "forward_runs"]].agg(["mean", "std"]).round(3))
table.to_csv(STUDY / "table.csv", index=False)

# --- plots -------------------------------------------------------------------------------------
figures = HERE / "out" / "figures"
figures.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 3.5))
for path in sorted(STUDY.glob("*/result.nc")):
    tree = xr.open_datatree(path, engine="h5netcdf")
    ax.semilogy(np.arange(1, 15), tree["subspace"]["eigenvalues"].values, color="0.6", lw=0.8)
ax.semilogy(np.arange(1, 15), np.sort(np.linalg.eigvalsh(reference_information))[::-1], "k", lw=2, label="reference $H$ (NUTS, exact)")
ax.axhline(1, ls=":", color="0.3")
ax.set(xlabel="index $i$", ylabel=r"eigenvalue $\lambda_i$", title="Spectra: ladder estimates (grey) and reference")
ax.legend()
fig.tight_layout()
fig.savefig(figures / "spectrum.png", dpi=120)

fig, ax = plt.subplots(figsize=(5, 3.5))
summary = table.groupby("rank")["marginal_w1_mean"].agg(["mean", "std"])
ax.errorbar(summary.index, summary["mean"], yerr=summary["std"], marker="o", capsize=3)
ax.set(xlabel="rank $r$", ylabel="mean marginal $W_1$ / reference sd", title="Error against rank (3 seeds)")
fig.tight_layout()
fig.savefig(figures / "error_vs_rank.png", dpi=120)

tree = xr.open_datatree(STUDY / run_key({"rank": 7}, 1) / "result.nc", engine="h5netcdf")
xi = tree["posterior"]["xi"].values
fig, axes = plt.subplots(2, 7, figsize=(14, 4))
quantiles = np.linspace(0.01, 0.99, 50)
for j, axis in enumerate(axes.ravel()):
    axis.plot(np.quantile(reference_xi[:, j], quantiles), np.quantile(xi[:, j], quantiles), ".", ms=3)
    lo, hi = np.quantile(reference_xi[:, j], [0.01, 0.99])
    axis.plot([lo, hi], [lo, hi], "k", lw=0.6)
    axis.set_title(coordinates.names[j][:22], fontsize=7)
    axis.tick_params(labelsize=6)
fig.suptitle("QQ: active_subspace_mcmc (r = 7) against NUTS, whitened coordinates")
fig.tight_layout()
fig.savefig(figures / "qq.png", dpi=110)

eigenvectors = tree["subspace"]["eigenvectors"].values
eigenvalues = tree["subspace"]["eigenvalues"].values
scores = (eigenvectors[:, :7] ** 2) @ eigenvalues[:7]
order = np.argsort(scores)
fig, ax = plt.subplots(figsize=(5, 4))
ax.barh(np.array(coordinates.names)[order], scores[order])
ax.set(xlabel="activity score", title="Activity scores (r = 7)")
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(figures / "activity_scores.png", dpi=120)
print("figures:", sorted(p.name for p in figures.glob("*.png")))
