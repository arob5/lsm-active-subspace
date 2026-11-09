import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from probpipe import workflow_run

from lsmas import plots
from lsmas.data import load_harvard_forest
from lsmas.experiments.reference import Reference
from lsmas.problems import load
from lsmas.tools import LadderLinearization, Subspace


@pytest.fixture(scope="module")
def toy_run():
    toy = load("toy")
    ladder = LadderLinearization(toy.coordinates)
    with workflow_run(seed=1):
        posterior = toy.condition(
            method="enskit_eki", method_options={"n_particles": 40, "callbacks": [ladder]}
        )
    theta = toy.coordinates.array_from_draws(posterior.atoms).reshape(-1, 14)
    reference = Reference(
        problem_name="toy",
        scales="fixed",
        parameter_names=toy.coordinates.names,
        theta=theta[None],
        noise_scales={},
        information=ladder.information,
        diagnostics=pd.DataFrame(),
        provenance={},
    )
    return toy, posterior, ladder.subspace(), reference, theta


def test_single_panel_plots_return_their_axes(toy_run):
    _, _, subspace, reference, _ = toy_run
    assert isinstance(plots.plot_spectra([subspace, subspace], reference), Axes)
    assert isinstance(plots.plot_activity_scores(subspace), Axes)
    assert isinstance(plots.plot_loadings(subspace, direction=2), Axes)
    table = pd.DataFrame({"rank": [2, 2, 4, 4], "marginal_w1_mean": [0.5, 0.6, 0.3, 0.2]})
    assert isinstance(plots.plot_metric_against_setting(table, "rank"), Axes)


def test_multi_panel_plots_return_a_figure(toy_run):
    toy, posterior, subspace, reference, theta = toy_run
    qq = plots.plot_qq(theta, reference, toy)
    assert isinstance(qq, Figure) and len([a for a in qq.axes if a.get_visible()]) == 14
    history = posterior.annotations["arviz/eki_history"]
    summary = plots.plot_sufficient_summary(toy, history, subspace)
    assert isinstance(summary, Figure) and len(summary.axes) == 3


def test_sufficient_summary_of_a_full_rank_subspace_has_no_inactive_panel(toy_run):
    toy, posterior, subspace, _, _ = toy_run
    full = Subspace.from_information(
        subspace.information, parameter_names=subspace.parameter_names, estimator="x", rank=14
    )
    figure = plots.plot_sufficient_summary(toy, posterior.annotations["arviz/eki_history"], full)
    assert len(figure.axes) == 2
    assert np.isfinite(full.eigenvalues).all()


def test_feature_plot_has_a_panel_per_class_and_skips_failed_rows():
    data = load_harvard_forest()
    predictions = data.observations + np.random.default_rng(0).normal(size=(20, 400))
    predictions[3] = np.nan
    figure = plots.plot_features(data, predictions, levels=(0.8,))
    assert [axis.get_ylabel().split()[0] for axis in figure.axes] == ["day", "night"]
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert labels == ["80%", "median", "observed"]
    with pytest.raises(ValueError, match=r"shape \(N, 400\)"):
        plots.plot_features(data, predictions[:, :10])


def test_prior_marginal_plot_has_a_panel_per_parameter(toy_run):
    toy, _, _, _, theta = toy_run
    figure = plots.plot_prior_marginals(toy, theta, columns=4)
    assert len([axis for axis in figure.axes if axis.get_visible()]) == 14
    assert figure.axes[0].get_title() == toy.coordinates.names[0]
    with pytest.raises(ValueError, match=r"shape \(N, 14\)"):
        plots.plot_prior_marginals(toy, theta[:, :3])
