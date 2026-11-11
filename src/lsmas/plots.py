r"""The standard figures: the data, the prior, subspaces, posteriors and studies.

Functions that draw one panel take an optional matplotlib ``Axes`` and return it; functions that
draw several panels return a ``Figure``. Figures of the data and of marginal distributions show
values in their own units. Figures of subspaces and of posteriors against a reference use whitened
coordinates,
$\xi = L^{-1}(z - m)$, so that every parameter is shown on the same scale.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from probpipe import (
    Distribution,
    EmpiricalDistribution,
    NumericArrayBatch,
    RecordBatch,
    ResolutionError,
    SupportsLogProb,
    log_prob,
    marginal,
    quantile,
)

from lsmas._stopgaps.sampling import draw_with_key
from lsmas.data import HarvardForestData
from lsmas.experiments.reference import Reference
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools.subspace import Subspace

__all__ = [
    "plot_activity_scores",
    "plot_features",
    "plot_loadings",
    "plot_marginals",
    "plot_metric_against_setting",
    "plot_qq",
    "plot_spectra",
    "plot_sufficient_summary",
]

#: Draws taken of a marginal that has neither a density nor draws of its own.
_PLOT_DRAWS = 4000

#: The colour of each feature class.
_CLASS_COLOURS = {"day": "tab:orange", "night": "tab:blue"}


def plot_features(
    data: HarvardForestData,
    predictions: np.ndarray | None = None,
    *,
    levels: Sequence[float] = (0.5, 0.9),
) -> Figure:
    r"""The observed features $y$ over time, one panel per class, with optional bands of
    predicted or replicated features.

    Each feature is plotted at the middle of its half-month. With *predictions*, each panel also
    shows, for every feature, the median of the predictions and their central intervals at each
    of *levels*: for level $\ell$, the band between the quantiles $(1 - \ell)/2$ and
    $(1 + \ell)/2$. Rows with a NaN, from failed forward runs, are left out of the bands.

    Parameters
    ----------
    data : HarvardForestData
        The features and their observed values.
    predictions : numpy.ndarray, optional
        Predicted or replicated features, shape $(N, n)$, such as prior predictive draws.
    levels : sequence of float
        The probabilities of the central intervals, each in $(0, 1)$.

    Raises
    ------
    ValueError
        If *predictions* does not have $n$ columns or has no row without a NaN.
    """
    times = data.feature_times
    classes = data.feature_classes
    if predictions is not None:
        predictions = _finite_rows(predictions, len(classes))
    figure, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for axis, feature_class in zip(axes, _CLASS_COLOURS, strict=True):
        columns = np.flatnonzero(classes == feature_class)
        columns = columns[np.argsort(times[columns])]
        colour = _CLASS_COLOURS[feature_class]
        if predictions is not None:
            _draw_bands(axis, times[columns], predictions[:, columns], levels, colour)
        observed = data.observations[columns]
        axis.plot(times[columns], observed, ".", color="k", ms=3, label="observed")
        axis.axhline(0.0, color="0.3", lw=0.5)
        axis.set_ylabel(f"{feature_class} NEE (µmol m⁻² s⁻¹)")
        axis.legend(fontsize=7, ncol=len(levels) + 2, loc="upper right")
    figure.tight_layout()
    return figure


def plot_marginals(
    *distributions: Distribution | RecordBatch,
    labels: Sequence[str] | None = None,
    columns: int = 6,
) -> Figure:
    r"""The one-dimensional marginal of every scalar field of one or more distributions.

    Each argument is a ProbPipe distribution, such as a prior or a posterior, or a batch of draws,
    such as the result of ``sample(prior, n)``, which is shown as the empirical distribution of its
    draws. There is one panel per scalar field, such as ``theta/photosynthetic_capacity``, in the
    order the fields first appear, and arguments that share a field are drawn on the same panel. A
    marginal with an exact density is drawn as that density; any other, such as a posterior held as
    draws, as a histogram of its (weighted) draws, normalized to a density. Each panel spans the
    central 99% of every marginal drawn on it. A panel is titled by the field's name, such as
    ``photosynthetic_capacity``, or by its whole path, such as ``s2/day``, when the field is not in
    the first panel's group or another field shares its name.

    A law with neither a density nor draws of its own is sampled, and a marginal without exact
    quantiles has its span estimated from draws; both use a fixed random key, so plotting never
    advances an enclosing ``workflow_run``.

    Parameters
    ----------
    *distributions : Distribution or RecordBatch
        The distributions, or batches of draws, to plot.
    labels : sequence of str, optional
        A legend label for each argument; by default a distribution's label, or ``"draws"``.
    columns : int
        Panels per row.

    Raises
    ------
    ValueError
        If no distribution is given, or *labels* has the wrong length.
    """
    if not distributions:
        raise ValueError("plot_marginals needs at least one distribution")
    if labels is not None and len(labels) != len(distributions):
        raise ValueError(f"{len(labels)} labels for {len(distributions)} distributions")
    laws = [_as_distribution(argument) for argument in distributions]
    labels = [_default_label(argument) for argument in distributions] if labels is None else labels
    paths_of_law = [_scalar_paths(law) for law in laws]
    paths = list(dict.fromkeys(path for law_paths in paths_of_law for path in law_paths))
    titles = _panel_titles(paths)
    rows = int(np.ceil(len(paths) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 1.8 * rows), squeeze=False)
    for axis in axes.ravel()[len(paths) :]:
        axis.set_visible(False)
    for axis, path in zip(axes.ravel(), paths, strict=False):
        marginals = {
            index: marginal(law, path)
            for index, (law, law_paths) in enumerate(zip(laws, paths_of_law, strict=True))
            if path in law_paths
        }
        low, high = _central_range(marginals.values())
        for index, law in marginals.items():
            _draw_marginal(axis, law, (low, high), colour=f"C{index}", label=labels[index])
        axis.set_title(titles[path], fontsize=7)
        axis.set_yticks([])
        axis.tick_params(labelsize=6)
    if len(laws) > 1:
        axes[0, 0].legend(fontsize=6)
    figure.tight_layout()
    return figure


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


def _finite_rows(predictions: np.ndarray, n_features: int) -> np.ndarray:
    """The rows of *predictions* without a NaN, after checking they have *n_features* columns."""
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != n_features:
        raise ValueError(f"predictions must have shape (N, {n_features}); got {predictions.shape}")
    finite = predictions[np.all(np.isfinite(predictions), axis=1)]
    if len(finite) == 0:
        raise ValueError("every row of the predictions has a NaN")
    return finite


def _draw_bands(
    axis: Axes,
    times: pd.DatetimeIndex,
    predictions: np.ndarray,
    levels: Sequence[float],
    colour: str,
) -> None:
    """The median of *predictions* over its rows, and its central intervals at each level.

    The intervals are shaded from light (the widest) to dark (the narrowest).
    """
    for index, level in enumerate(sorted(levels, reverse=True)):
        lower, upper = np.quantile(predictions, [(1 - level) / 2, (1 + level) / 2], axis=0)
        shade = 0.15 + 0.15 * index / max(len(levels) - 1, 1)
        axis.fill_between(
            times, lower, upper, color=colour, alpha=shade, lw=0, label=f"{level:.0%}"
        )
    axis.plot(times, np.median(predictions, axis=0), color=colour, lw=0.8, label="median")


def _as_distribution(argument: Distribution | RecordBatch) -> Distribution:
    """*argument* itself, or the empirical distribution of a batch of draws."""
    return EmpiricalDistribution(argument) if isinstance(argument, RecordBatch) else argument


def _default_label(argument: Distribution | RecordBatch) -> str:
    """A distribution's label, or ``"draws"`` for a batch of draws."""
    return "draws" if isinstance(argument, RecordBatch) else str(argument.label)


def _scalar_paths(law: Distribution) -> list[str]:
    """The paths of the scalar fields of *law*, such as ``theta/photosynthetic_capacity``, each
    starting with the name of its output, as ``marginal`` reads them."""
    paths = []
    for output, spec in law.event_spec.outputs.items():
        paths += _scalar_paths_below(spec, output)
    return paths


def _scalar_paths_below(spec: Any, prefix: str = "") -> list[str]:
    """The paths of the scalar leaves of a spec, each prefixed by *prefix*."""
    children = getattr(spec, "children", None)
    if children is None:
        return [prefix] if getattr(spec, "shape", None) == () else []
    paths = []
    for name, child in children.items():
        paths += _scalar_paths_below(child, f"{prefix}/{name}" if prefix else name)
    return paths


def _panel_titles(paths: Sequence[str]) -> dict[str, str]:
    """Each path's last segment, or the whole path when it is outside the first path's group or
    another path ends the same way."""
    groups = [path.rpartition("/")[0] for path in paths]
    names = [path.rpartition("/")[2] for path in paths]
    return {
        path: path if group != groups[0] or names.count(name) > 1 else name
        for path, group, name in zip(paths, groups, names, strict=True)
    }


def _central_range(marginals: Iterable[Distribution]) -> tuple[float, float]:
    """The smallest interval holding the central 99% of each marginal, widened if it is a point."""
    bounds = np.array([_central_bounds(law) for law in marginals])
    low, high = float(bounds[:, 0].min()), float(bounds[:, 1].max())
    if high > low:
        return low, high
    half_width = 0.05 * abs(low) or 0.5
    return low - half_width, high + half_width


def _central_bounds(law: Distribution) -> tuple[float, float]:
    """The 0.5% and 99.5% quantiles of a scalar marginal: exact if it has them, else from draws."""
    levels = np.array([0.005, 0.995])
    try:
        low, high = np.asarray(quantile.using(exact_only=True)(law, levels), dtype=float)
    except ResolutionError:
        draws, weights = _draws_and_weights(law)
        low, high = np.quantile(draws, levels, weights=weights, method="inverted_cdf")
    return float(low), float(high)


def _draw_marginal(
    axis: Axes, law: Distribution, span: tuple[float, float], *, colour: str, label: str
) -> None:
    """A scalar marginal on *axis*: its density if it has one, else a histogram of its draws."""
    if isinstance(law, SupportsLogProb):
        grid = np.linspace(*span, 200)
        log_density = log_prob(law, NumericArrayBatch(grid, level_names="grid", label="grid"))
        axis.plot(grid, np.exp(np.asarray(log_density)), color=colour, lw=1.2, label=label)
        return
    draws, weights = _draws_and_weights(law)
    bins = np.linspace(*span, 41)
    # Heights are densities of all the draws, so draws outside the span are not renormalized away.
    heights = weights / (weights.sum() * (bins[1] - bins[0]))
    axis.hist(draws, bins=bins, weights=heights, color=colour, alpha=0.4, label=label)


def _draws_and_weights(law: Distribution) -> tuple[np.ndarray, np.ndarray]:
    """A scalar marginal's draws and their weights: its own if it is empirical, else drawn."""
    if isinstance(law, EmpiricalDistribution):
        draws = np.asarray(law.atoms.raw, dtype=float).reshape(-1)
        return draws, np.asarray(law.weights, dtype=float).reshape(-1)
    # STOPGAP(PG-15): a fixed key, so that plotting does not advance an enclosing workflow_run.
    draws = np.asarray(draw_with_key(law, jax.random.key(0), (_PLOT_DRAWS,)), dtype=float)
    return draws.reshape(-1), np.ones(draws.size)
