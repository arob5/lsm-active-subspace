import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from probpipe import NumericRecordBatch, sample, workflow_run

from lsmas import plots
from lsmas.data import load_harvard_forest
from lsmas.experiments.reference import Reference
from lsmas.problems import load_inverse_problem
from lsmas.tools import LadderLinearization, Subspace


@pytest.fixture(scope="module")
def toy_run():
    toy = load_inverse_problem("toy")
    ladder = LadderLinearization(toy.coordinates)
    with workflow_run(seed=1):
        posterior = toy.condition(
            method="enskit_eki", method_options={"n_particles": 40, "callbacks": [ladder]}
        )
    theta = np.asarray(posterior.atoms["theta"].to_vector()).reshape(-1, 14)
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
    many_levels = plots.plot_features(data, predictions, levels=np.linspace(0.1, 0.9, 9))
    assert len(many_levels.axes) == 2
    with pytest.raises(ValueError, match=r"shape \(N, 400\)"):
        plots.plot_features(data, predictions[:, :10])


def test_marginal_plot_overlays_densities_and_draws_by_field(toy_run):
    toy, posterior, _, _, _ = toy_run
    with workflow_run(seed=3):
        draws = sample(toy.prior, 200)
    figure = plots.plot_marginals(toy.prior, posterior, draws, columns=4)
    panels = [axis for axis in figure.axes if axis.get_visible()]
    assert [axis.get_title() for axis in panels] == list(toy.coordinates.names)
    assert len(panels[0].get_lines()) == 1  # the prior's exact density
    assert len(panels[0].patches) == 2 * 40  # a histogram each for the posterior and the draws
    legend = [text.get_text() for text in panels[0].get_legend().get_texts()]
    assert legend == ["prior", str(posterior.label), "draws"]


def test_marginal_plot_shows_fields_that_only_some_distributions_have():
    inferred = load_inverse_problem("toy", scales="inferred")
    with workflow_run(seed=4):
        draws = sample(load_inverse_problem("toy").prior, 50)
    figure = plots.plot_marginals(inferred.prior, draws, labels=["inferred", "fixed"])
    titles = [axis.get_title() for axis in figure.axes if axis.get_visible()]
    assert titles[-2:] == ["s2/day", "s2/night"]
    record_view = plots.plot_marginals(inferred.prior["s2"], columns=2)
    assert [axis.get_title() for axis in record_view.axes] == ["day", "night"]
    with pytest.raises(ValueError, match="2 labels for 1"):
        plots.plot_marginals(inferred.prior, labels=["a", "b"])


def test_marginal_plot_of_a_constant_field_and_of_fields_sharing_a_name():
    draws = NumericRecordBatch(
        {"theta": {"scale": np.full(30, 2.0)}, "noise": {"scale": np.linspace(0.5, 1.5, 30)}},
        "sample",
    )
    figure = plots.plot_marginals(draws)
    assert [axis.get_title() for axis in figure.axes if axis.get_visible()] == [
        "theta/scale",
        "noise/scale",
    ]
    low, high = figure.axes[0].get_xlim()
    assert low < 2.0 < high
