"""Every planned section of the walkthrough notebook: its code, its figure, and its cost.

Run from the repository root:

    uv run python design/prototypes/05-main-inverse-problem/t1_sections.py

Figures go to out/figures/, timings and numbers to out/t1.txt.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from probpipe import (  # noqa: E402
    NumericArrayBatch,
    log_prob,
    marginal,
    predictive_check,
    quantile,
    sample,
    summarize,
    workflow_run,
)
from pysipnet import ClimateDrivers  # noqa: E402

from lsmas.data import load_harvard_forest  # noqa: E402
from lsmas.parameters import calibration_to_sipnet  # noqa: E402
from lsmas.priors import prior_medians  # noqa: E402
from lsmas.problems import load  # noqa: E402
from lsmas.sipnet import SIPNETSimulator  # noqa: E402

OUT = Path(__file__).parent / "out"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
LOG = []
YEAR = 2015


def log(line: str) -> None:
    print(line)
    LOG.append(line)


def timed(name):
    def wrap(section):
        def run(*args, **kwargs):
            start = time.time()
            result = section(*args, **kwargs)
            log(f"[{name}] {time.time() - start:.1f} s")
            return result

        return run

    return wrap


def save(figure, name: str) -> None:
    figure.savefig(FIGURES / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    log(f"  figure {name}.png ({(FIGURES / f'{name}.png').stat().st_size // 1024} KB)")


def feature_midpoints(data) -> pd.Series:
    f = data.features
    return f["first_step_end_utc"] + (f["last_step_end_utc"] - f["first_step_end_utc"]) / 2


@timed("0 load")
def section_load():
    problem = load("harvard_forest", scales="inferred")
    data = load_harvard_forest()
    log(problem.describe()[:300])
    return problem, data


@timed("1 driver")
def section_driver(data):
    driver = ClimateDrivers.from_path(data.driver_path, time_zone="UTC").xarray
    one_year = driver.sel(time=str(YEAR))
    daily = one_year.resample(time="1D")
    panels = [
        (daily.mean()["air_temperature"], "air temperature (°C)"),
        (daily.sum()["photosynthetically_active_radiation"], "PAR (mol m⁻² d⁻¹)"),
        (daily.mean()["vapor_pressure_deficit"] / 1000, "VPD (kPa)"),
        (daily.sum()["precipitation"], "precipitation (cm d⁻¹)"),
    ]
    figure, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    for axis, (series, label) in zip(axes, panels, strict=True):
        axis.plot(series["time"], series, lw=0.8)
        axis.set_ylabel(label, fontsize=8)
    figure.suptitle(f"Meteorological driver, {YEAR} (daily)")
    save(figure, "f01_driver")
    log(f"  driver variables: {list(driver.data_vars)}")


@timed("2 SIPNET at the prior medians")
def section_sipnet(data):
    medians = prior_medians(data.prior_table)
    sipnet_parameters = calibration_to_sipnet(medians, data.fixed_sipnet_parameters)
    log(f"  {len(medians)} calibration parameters -> {len(sipnet_parameters)} SIPNET parameters")
    simulator = SIPNETSimulator(data.driver_path, data.fixed_sipnet_parameters)
    start = time.time()
    outputs = simulator._model(**sipnet_parameters).outputs.xarray  # proposed: simulator.run(...)
    log(f"  one SIPNET run: {time.time() - start:.2f} s; outputs: {list(outputs.data_vars)}")
    x = simulator.nee_rate(sipnet_parameters)
    one_year = outputs.sel(time=str(YEAR))
    figure, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    rate = pd.Series(x, index=data.step_ends)[str(YEAR)]
    axes[0].plot(rate.index, rate.values, lw=0.3, color="0.6", label="3-hourly")
    axes[0].plot(rate.resample("1D").mean(), lw=1, label="daily mean")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("NEE (µmol m⁻² s⁻¹)", fontsize=8)
    axes[0].legend(fontsize=7)
    for axis, name, label in [
        (axes[1], "gross_primary_production", "GPP (g C m⁻² d⁻¹)"),
        (axes[2], "leaf_carbon", "leaf carbon (g C m⁻²)"),
        (axes[3], "soil_wetness_fraction", "soil wetness"),
    ]:
        series = one_year[name].resample(time="1D")
        series = series.sum() if name == "gross_primary_production" else series.mean()
        axis.plot(series["time"], series, lw=1)
        axis.set_ylabel(label, fontsize=8)
    figure.suptitle(f"SIPNET at the prior medians, {YEAR}")
    save(figure, "f02_sipnet_at_medians")
    return medians, simulator, sipnet_parameters, x


@timed("3 data and operator")
def section_data(data, x, problem, medians):
    f = data.features
    log(f"  features: {len(f)} ({(f['class'] == 'day').sum()} day, "
        f"{(f['class'] == 'night').sum()} night), years {f['year'].min()}-{f['year'].max()}")
    log(f"  operator {data.operator.shape}, nonzeros {data.operator.nnz}, "
        f"windows per feature {f['n_steps'].min()}-{f['n_steps'].max()}")
    middle = feature_midpoints(data)
    figure, axis = plt.subplots(figsize=(9, 3.5))
    for cls, colour in [("day", "tab:orange"), ("night", "tab:blue")]:
        rows = f["class"] == cls
        axis.plot(middle[rows], f.loc[rows, "nee"], ".", color=colour, ms=4, label=cls)
    axis.axhline(0, color="k", lw=0.5)
    axis.set(ylabel="observed NEE feature (µmol m⁻² s⁻¹)", title="The data y: 400 half-month means")
    axis.legend()
    save(figure, "f03_features")

    # The operator on one half-month: which steps each feature averages.
    july = (f["year"] == YEAR) & (f["month"] == 7) & (f["half"] == 1)
    rows = np.where(july)[0]
    figure, axis = plt.subplots(figsize=(9, 3.5))
    window = slice(*np.searchsorted(data.step_ends, [pd.Timestamp(f"{YEAR}-07-01"),
                                                     pd.Timestamp(f"{YEAR}-07-17")]))
    axis.plot(data.step_ends[window], x[window], color="0.6", lw=0.8, label="SIPNET x(θ), 3-hourly")
    G = data.operator @ x
    for row in rows:
        cls = f["class"].iloc[row]
        colour = "tab:orange" if cls == "day" else "tab:blue"
        steps = data.operator[row].indices
        axis.plot(data.step_ends[steps], x[steps], "o", ms=3, color=colour,
                  label=f"{cls} windows ({len(steps)})")
        span = [data.step_ends[steps[0]], data.step_ends[steps[-1]]]
        axis.hlines(G[row], *span, color=colour, lw=2, label=f"{cls}: (A x)_i")
        axis.hlines(f["nee"].iloc[row], *span, color=colour, lw=2, ls="--", label=f"{cls}: y_i")
    axis.legend(fontsize=7, ncol=3)
    axis.set(ylabel="NEE (µmol m⁻² s⁻¹)", title=f"The observation operator on 1–15 July {YEAR}")
    save(figure, "f04_operator")

    theta = {name: medians[name] for name in problem.coordinates.names}
    start = time.time()
    forward = np.asarray(problem.forward(theta).raw)
    log(f"  problem.forward(theta): {time.time() - start:.2f} s; equals A x: "
        f"{np.allclose(forward, G)}")
    return G


@timed("4 likelihood")
def section_likelihood(problem, data, G):
    f = data.features
    eta = problem.learning_rate["nee"]
    v = problem.likelihood.reference_variance
    log(f"  eta day {eta[f['class'] == 'day'][0]:.3f}, night {eta[f['class'] == 'night'][0]:.3f}")
    gamma = problem.noise_variance({"s2_day": 1.0, "s2_night": 1.0})
    log(f"  sqrt Gamma day {np.sqrt(gamma[f['class'] == 'day'][0]):.2f}, "
        f"night {np.sqrt(gamma[f['class'] == 'night'][0]):.2f}")
    residual = data.observations - G
    middle = feature_midpoints(data)
    figure, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for axis, cls in zip(axes, ["day", "night"], strict=True):
        rows = (f["class"] == cls).to_numpy()
        order = np.argsort(middle[rows].to_numpy())
        t = middle[rows].to_numpy()[order]
        axis.fill_between(t, -2 * np.sqrt(gamma[rows][order]), 2 * np.sqrt(gamma[rows][order]),
                          color="0.85", label="±2 √Γᵢᵢ (tempered)")
        axis.fill_between(t, -2 * np.sqrt(v[rows][order]), 2 * np.sqrt(v[rows][order]),
                          color="0.65", label="±2 √vₖ (untempered)")
        axis.plot(t, residual[rows][order], ".", ms=4, color="k", label="y − G(θ_median)")
        axis.set_ylabel(f"{cls} residual", fontsize=8)
        axis.legend(fontsize=7, loc="upper right", ncol=3)
    figure.suptitle("Residuals at the prior medians, against the noise (s² = 1)")
    save(figure, "f05_residuals")

    law = marginal(problem.prior, "s2_day")
    grid = np.linspace(0.05, 6, 300)
    density = np.exp(np.asarray(log_prob(law, NumericArrayBatch(grid, level_names="grid",
                                                                    label="s2 grid"))))
    figure, axis = plt.subplots(figsize=(4.5, 3))
    axis.plot(grid, density)
    axis.axvline(float(quantile(law, 0.5)), ls=":", color="0.3", label="median")
    axis.set(xlabel="s²", ylabel="density", title="IG(2, 1.678) prior of each noise scale")
    axis.legend()
    save(figure, "f06_noise_scale_prior")


@timed("5 prior")
def section_prior(problem, data):
    start = time.time()
    table = summarize(problem.prior)
    log(f"  summarize(prior): {time.time() - start:.1f} s")
    names = problem.coordinates.names
    columns = 6
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 1.7 * rows))
    for axis, name in zip(axes.ravel(), names, strict=False):
        law = marginal(problem.prior, f"theta/{name}")
        low, high = (float(quantile(law, q)) for q in (0.005, 0.995))
        grid = np.linspace(low, high, 200)
        density = np.exp(np.asarray(log_prob(law, NumericArrayBatch(grid, level_names="grid",
                                                                        label=name))))
        axis.plot(grid, density, lw=1)
        inner = (grid >= float(quantile(law, 0.025))) & (grid <= float(quantile(law, 0.975)))
        axis.fill_between(grid[inner], density[inner], alpha=0.3)
        axis.set_title(name, fontsize=6.5)
        axis.set_yticks([])
        axis.tick_params(labelsize=6)
    for axis in axes.ravel()[len(names):]:
        axis.set_visible(False)
    figure.suptitle("Prior marginals of the 46 calibration parameters (shaded: central 95%)")
    figure.tight_layout()
    save(figure, "f07_prior_marginals")
    return table


@timed("6 coordinates")
def section_coordinates(problem):
    with workflow_run(seed=0):
        draws = sample(problem.prior, 4000)
    theta = problem.coordinates.array_from_draws(draws)
    z = problem.coordinates.bijection.to_unconstrained(theta)
    xi = problem.coordinates.whiten(z)
    log(f"  xi mean |max| {np.abs(xi.mean(0)).max():.3f}, cov - I |max| "
        f"{np.abs(np.cov(xi, rowvar=False) - np.eye(xi.shape[1])).max():.3f}")
    j = problem.coordinates.names.index("photosynthetic_capacity")
    k = problem.coordinates.names.index("respiration_share")
    figure, axes = plt.subplots(2, 3, figsize=(10, 5))
    for column, (values, label) in enumerate([(theta, "θ"), (z, "z = b⁻¹(θ)"),
                                               (xi, "ξ = L⁻¹(z − m)")]):
        axes[0, column].hist(values[:, j], bins=60, color="tab:green")
        axes[0, column].set_title(f"photosynthetic_capacity, {label}", fontsize=8)
        axes[1, column].hist(values[:, k], bins=60, color="tab:purple")
        axes[1, column].set_title(f"respiration_share, {label}", fontsize=8)
    figure.tight_layout()
    save(figure, "f08_coordinates")


@timed("7 prior predictive")
def section_prior_predictive(problem, data, n=200):
    f = data.features
    with workflow_run(seed=1):
        start = time.time()
        replications = sample(problem.model, n)
        log(f"  sample(model, {n}): {time.time() - start:.1f} s")
    nee = np.asarray(replications["nee"].raw if hasattr(replications["nee"], "raw")
                     else replications["nee"])
    log(f"  replications {nee.shape}, NaN rows {np.isnan(nee).any(axis=1).sum()}")
    low, high = np.quantile(nee, [0.05, 0.95], axis=0)
    inside = (data.observations >= low) & (data.observations <= high)
    for cls in ("day", "night"):
        rows = (f["class"] == cls).to_numpy()
        log(f"  {cls}: {inside[rows].mean():.2f} of features inside the 90% interval")
    middle = feature_midpoints(data)
    figure, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for axis, cls in zip(axes, ["day", "night"], strict=True):
        rows = (f["class"] == cls).to_numpy()
        order = np.argsort(middle[rows].to_numpy())
        t = middle[rows].to_numpy()[order]
        q = np.quantile(nee[:, rows][:, order], [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        axis.fill_between(t, q[0], q[4], color="tab:blue", alpha=0.2, label="90%")
        axis.fill_between(t, q[1], q[3], color="tab:blue", alpha=0.35, label="50%")
        axis.plot(t, q[2], color="tab:blue", lw=0.8, label="median")
        axis.plot(t, data.observations[rows][order], ".", color="k", ms=3, label="observed")
        axis.set_ylabel(f"{cls} NEE", fontsize=8)
        axis.legend(fontsize=7, ncol=4, loc="upper right")
    figure.suptitle(f"Prior predictive of the features ({n} draws of sample(problem.model, n))")
    save(figure, "f09_prior_predictive")

    summer_day = jnp.asarray(np.where((f["class"] == "day") & f["month"].isin([6, 7, 8]))[0])
    night = jnp.asarray(np.where(f["class"] == "night")[0])

    def mean_summer_day_nee(y):
        return y[summer_day].mean()

    def mean_night_nee(y):
        return y[night].mean()

    with workflow_run(seed=2):
        start = time.time()
        check = predictive_check(problem.likelihood, problem.prior,
                                 [mean_summer_day_nee, mean_night_nee],
                                 observed_data={"nee": data.observations}, num_replications=n)
        log(f"  predictive_check({n}): {time.time() - start:.1f} s")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3))
    for axis, name in zip(axes, ["mean_summer_day_nee", "mean_night_nee"], strict=True):
        # The replicated statistics come back as an EmpiricalDistribution; its atoms are the draws.
        replicated = np.asarray(check[f"{name}/replicated_statistics"].atoms.raw)
        observed = float(np.asarray(getattr(check[f"{name}/observed_statistic"], "raw",
                                            check[f"{name}/observed_statistic"])))
        p_value = float(np.asarray(getattr(check[f"{name}/p_value"], "raw",
                                           check[f"{name}/p_value"])))
        axis.hist(replicated, bins=30, color="0.6")
        axis.axvline(observed, color="k", lw=2)
        axis.set_title(f"{name}: p = {p_value:.2f}", fontsize=8)
        log(f"  {name}: observed {observed:.2f}, p {p_value:.2f}")
    save(figure, "f10_predictive_check")


@timed("8 one at a time")
def section_one_at_a_time(problem, data, medians):
    f = data.features
    names = ["photosynthetic_capacity", "soil_water_holding_capacity",
             "leaf_on_growing_degree_days", "soil_respiration_flux_at_10c"]
    levels = [0.025, 0.25, 0.5, 0.75, 0.975]
    base = np.array([medians[name] for name in problem.coordinates.names])
    rows = []
    for name in names:
        law = marginal(problem.prior, f"theta/{name}")
        for level in levels:
            theta = base.copy()
            theta[problem.coordinates.names.index(name)] = float(quantile(law, level))
            rows.append(theta)
    start = time.time()
    G = np.asarray(problem.forward(problem.coordinates.bijection.record_batch(np.array(rows))).raw)
    log(f"  {len(rows)} runs as one batch: {time.time() - start:.1f} s")
    middle = feature_midpoints(data)
    in_year = (f["year"] == YEAR).to_numpy()
    figure, axes = plt.subplots(2, 4, figsize=(13, 5), sharex=True)
    colours = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    for column, name in enumerate(names):
        for row_axis, cls in enumerate(["day", "night"]):
            axis = axes[row_axis, column]
            mask = in_year & (f["class"] == cls).to_numpy()
            order = np.argsort(middle[mask].to_numpy())
            for index, level in enumerate(levels):
                axis.plot(middle[mask].to_numpy()[order],
                          G[column * len(levels) + index][mask][order],
                          color=colours[index], lw=1, label=f"{level:.1%} quantile")
            axis.plot(middle[mask].to_numpy()[order], data.observations[mask][order], ".k", ms=3)
            axis.set_title(f"{name} ({cls})", fontsize=7)
            axis.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    figure.suptitle(f"One parameter at a time through its prior, the rest at their medians ({YEAR})")
    figure.tight_layout()
    save(figure, "f11_one_at_a_time")


@timed("9 tempering slice")
def section_slice(problem, data, medians):
    name = "photosynthetic_capacity"
    law = marginal(problem.prior, f"theta/{name}")
    values = np.linspace(float(quantile(law, 0.005)), float(quantile(law, 0.995)), 41)
    base = np.array([medians[n] for n in problem.coordinates.names])
    theta = np.tile(base, (len(values), 1))
    theta[:, problem.coordinates.names.index(name)] = values
    start = time.time()
    G = np.asarray(problem.forward(problem.coordinates.bijection.record_batch(theta)).raw)
    log(f"  {len(values)} runs: {time.time() - start:.1f} s")
    v = problem.likelihood.noise_variances({"s2_day": 1.0, "s2_night": 1.0})
    eta = problem.learning_rate["nee"]
    pointwise = -0.5 * ((data.observations - G) ** 2 / v + np.log(2 * np.pi * v))
    log_prior = np.asarray(log_prob(law, NumericArrayBatch(values, level_names="grid",
                                                           label="grid")))
    curves = {"prior": log_prior, "posterior, η = 1": log_prior + pointwise.sum(1),
              "posterior, tempered η": log_prior + (eta * pointwise).sum(1)}
    figure, axis = plt.subplots(figsize=(6, 3.5))
    for label, curve in curves.items():
        density = np.exp(curve - curve.max())
        axis.plot(values, density / np.trapezoid(density, values), label=label)
    axis.set(xlabel=name, ylabel="density (normalized along the slice)",
             title="A slice through the prior medians (s² = 1)")
    axis.legend(fontsize=8)
    save(figure, "f12_tempering_slice")


def main():
    start = time.time()
    problem, data = section_load()
    section_driver(data)
    medians, simulator, sipnet_parameters, x = section_sipnet(data)
    G = section_data(data, x, problem, medians)
    section_likelihood(problem, data, G)
    section_prior(problem, data)
    section_coordinates(problem)
    section_prior_predictive(problem, data)
    section_one_at_a_time(problem, data, medians)
    section_slice(problem, data, medians)
    log(f"[total] {time.time() - start:.0f} s")
    (OUT / "t1.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
