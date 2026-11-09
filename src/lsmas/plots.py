r"""The standard figures for subspaces, posteriors and studies.

Functions that draw one panel take an optional matplotlib ``Axes`` and return it; functions that
draw several panels return a ``Figure``. Coordinates are whitened, $\xi = L^{-1}(z - m)$, so that
every parameter is shown on the same scale.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lsmas.experiments.reference import Reference
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools.subspace import Subspace

__all__ = [
    "plot_activity_scores",
    "plot_loadings",
    "plot_metric_against_setting",
    "plot_qq",
    "plot_spectra",
    "plot_sufficient_summary",
]


def plot_spectra(
    subspaces: Sequence[Subspace], reference: Reference | None = None, ax: Axes | None = None
) -> Axes:
    r"""The eigenvalues $\lambda_i$ of each subspace's information matrix, on a log scale.

    Each estimate is a thin grey line; the reference information matrix, if given, is a thick black
    line. The dotted line $\lambda = 1$ separates directions the data inform more than the prior
    does from the rest.
    """
    ax = plt.subplots(figsize=(5, 3.5))[1] if ax is None else ax
    for subspace in subspaces:
        ax.semilogy(np.arange(1, subspace.dimension + 1), subspace.eigenvalues, color="0.6", lw=0.8)
    if reference is not None:
        eigenvalues = np.sort(np.linalg.eigvalsh(reference.information))[::-1]
        ax.semilogy(np.arange(1, len(eigenvalues) + 1), eigenvalues, "k", lw=2, label="reference")
        ax.legend()
    ax.axhline(1.0, ls=":", color="0.3")
    ax.set(xlabel="index $i$", ylabel=r"eigenvalue $\lambda_i$")
    return ax


def plot_metric_against_setting(
    table: pd.DataFrame, setting: str, metric: str = "marginal_w1_mean", ax: Axes | None = None
) -> Axes:
    """A metric's mean over seeds, with one standard deviation as error bars, for each value of a
    setting in a study's table."""
    ax = plt.subplots(figsize=(5, 3.5))[1] if ax is None else ax
    summary = table.groupby(setting)[metric].agg(["mean", "std"])
    ax.errorbar(summary.index, summary["mean"], yerr=summary["std"], marker="o", capsize=3)
    ax.set(xlabel=setting, ylabel=metric)
    return ax


def plot_qq(
    theta: np.ndarray, reference: Reference, problem: InverseProblem, columns: int = 7
) -> Figure:
    r"""For each parameter, the quantiles of the draws *theta* against the reference's, in whitened
    coordinates; points on the diagonal mean the marginals agree.

    Parameters
    ----------
    theta : numpy.ndarray
        Draws of the calibration parameters, shape $(N, d)$.
    reference : Reference
        The reference.
    problem : InverseProblem
        The problem, whose coordinates whiten the draws.
    columns : int
        Panels per row.
    """
    approximation = problem.coordinates.whitened_from_parameters(theta)
    target = problem.coordinates.whitened_from_parameters(reference.parameter_draws())
    dimension = target.shape[1]
    rows = int(np.ceil(dimension / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    levels = np.linspace(0.01, 0.99, 50)
    for j, axis in enumerate(axes.ravel()):
        if j >= dimension:
            axis.set_visible(False)
            continue
        axis.plot(
            np.quantile(target[:, j], levels), np.quantile(approximation[:, j], levels), ".", ms=3
        )
        low, high = np.quantile(target[:, j], [0.01, 0.99])
        axis.plot([low, high], [low, high], "k", lw=0.6)
        axis.set_title(problem.coordinates.names[j], fontsize=7)
        axis.tick_params(labelsize=6)
    figure.supxlabel("reference quantile")
    figure.supylabel("approximation quantile")
    figure.tight_layout()
    return figure


def plot_activity_scores(subspace: Subspace, ax: Axes | None = None) -> Axes:
    r"""Each parameter's activity score $\alpha_j = \sum_{i \le r} \lambda_i W_{ji}^2$, as bars."""
    scores = subspace.activity_scores().iloc[::-1]
    ax = plt.subplots(figsize=(5, 0.25 * len(scores) + 1))[1] if ax is None else ax
    ax.barh(scores.index, scores.values)
    ax.set(xlabel="activity score", title=f"rank {subspace.rank}")
    ax.tick_params(axis="y", labelsize=7)
    return ax


def plot_loadings(subspace: Subspace, direction: int = 1, ax: Axes | None = None) -> Axes:
    """The weights of each parameter in one direction of the subspace (1 is the leading one)."""
    weights = subspace.loadings()[f"direction_{direction}"]
    ax = plt.subplots(figsize=(5, 0.25 * len(weights) + 1))[1] if ax is None else ax
    ax.barh(weights.index[::-1], weights.values[::-1])
    ax.axvline(0.0, color="0.3", lw=0.6)
    ax.set(xlabel=f"weight in direction {direction}")
    ax.tick_params(axis="y", labelsize=7)
    return ax


def plot_sufficient_summary(
    problem: InverseProblem, eki_history: xr.Dataset, subspace: Subspace
) -> Figure:
    r"""The misfit of every particle of an EKI run against its first two active variables and its
    last inactive variable.

    The misfit of particle $j$ is $\Phi_j = \tfrac12 \sum_i (y_i - G_i(\theta_j))^2 /
    \Gamma_{ii}$, with each step's $\Gamma$. If the subspace is good, the misfit varies with the
    active variables and shows no pattern against the inactive one. The data come from the run's
    history, so the plot costs no forward runs.

    Parameters
    ----------
    problem : InverseProblem
        The problem the run conditioned.
    eki_history : xarray.Dataset
        The run's history, ``posterior.annotations["arviz/eki_history"]``.
    subspace : Subspace
        The subspace whose variables to plot against.
    """
    theta = eki_history["theta"].values.reshape(-1, problem.coordinates.dimension)
    xi = problem.coordinates.whitened_from_parameters(theta)
    residuals = problem.data[problem.likelihood.output_name] - eki_history["predictions"].values
    noise_variance = eki_history["noise_variance"].values[:, None, :]
    misfit = (0.5 * np.sum(residuals**2 / noise_variance, axis=-1)).reshape(-1)
    active = subspace.active_variables(xi)
    panels = [(active[:, 0], "$a_1$")]
    if subspace.rank > 1:
        panels.append((active[:, 1], "$a_2$"))
    if subspace.rank < subspace.dimension:
        panels.append((subspace.inactive_variables(xi)[:, -1], "last inactive variable $b$"))
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5), squeeze=False)
    for axis, (values, label) in zip(axes[0], panels, strict=True):
        axis.semilogy(values, misfit, ".", ms=3, alpha=0.6)
        axis.set(xlabel=label, ylabel=r"misfit $\Phi$")
    figure.tight_layout()
    return figure
