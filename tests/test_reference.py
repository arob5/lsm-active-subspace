import json

import numpy as np
import pandas as pd
import pytest

from conftest import CLASSES, exact_linear_information, linear_problem
from lsmas._stopgaps.parallel_chains import run_in_processes
from lsmas.experiments.reference import (
    Reference,
    _meets_criteria,
    convergence_diagnostics,
    noise_scale_setting,
    nuts_reference,
    rwmh_diagnostics,
    rwmh_reference,
    start_rwmh_reference,
)
from lsmas.problems import load

SCALES = {"day": 1.5, "night": 0.7}
LEARNING_RATE = np.where(CLASSES == "day", 0.5, 0.8)


@pytest.fixture
def problem(observations):
    return linear_problem(observations, SCALES, LEARNING_RATE)


def test_nuts_reference_holds_draws_diagnostics_and_the_exact_information(problem):
    reference = nuts_reference(problem, seed=0)
    assert reference.theta.shape[0] == 4 and reference.theta.shape[-1] == 2
    assert reference.parameter_draws().shape[0] <= 4000
    np.testing.assert_allclose(
        reference.information,
        exact_linear_information(SCALES, LEARNING_RATE, problem.coordinates.whitening_factor),
        rtol=1e-6,
    )
    assert reference.diagnostics["rhat"].max() < 1.01
    assert reference.provenance["method"] == "blackjax_nuts"
    assert "probpipe-core" in reference.provenance["packages"]


def test_reference_round_trips_through_netcdf(problem, tmp_path):
    reference = Reference(
        problem_name="linear",
        scales="inferred",
        parameter_names=("a", "b"),
        theta=np.random.default_rng(0).normal(size=(2, 30, 2)),
        noise_scales={"s2_day": np.random.default_rng(1).gamma(2.0, size=(2, 30))},
        information=np.array([[2.0, 0.5], [0.5, 1.0]]),
        diagnostics=pd.DataFrame(
            {"rhat": [1.0, 1.0], "ess_bulk": [50.0, 60.0], "ess_tail": [40.0, 45.0]},
            index=pd.Index(["a", "b"], name="coordinate"),
        ),
        provenance={"method": "test", "settings": {"seed": 3}},
    )
    loaded = Reference.load(reference.save(tmp_path / "reference.nc"))
    np.testing.assert_array_equal(loaded.theta, reference.theta)
    np.testing.assert_array_equal(loaded.noise_scales["s2_day"], reference.noise_scales["s2_day"])
    np.testing.assert_array_equal(loaded.information, reference.information)
    pd.testing.assert_frame_equal(loaded.diagnostics, reference.diagnostics)
    assert loaded.provenance == reference.provenance
    assert (loaded.problem_name, loaded.scales, loaded.parameter_names) == (
        "linear",
        "inferred",
        ("a", "b"),
    )


def test_independent_chains_have_rhat_near_one_and_full_effective_size(problem):
    theta = problem.coordinates.parameters_from_whitened(
        np.random.default_rng(2).normal(size=(4, 2000, 2))
    )
    diagnostics = convergence_diagnostics(problem, theta, {})
    assert diagnostics["rhat"].max() < 1.01
    assert diagnostics["ess_bulk"].min() > 6000
    assert list(diagnostics.index) == ["a", "b"]


def test_criteria_need_every_coordinate_converged():
    good = pd.DataFrame({"rhat": [1.005, 1.0], "ess_bulk": [1200, 3000], "ess_tail": [1100, 2000]})
    assert _meets_criteria(good)
    assert not _meets_criteria(good.assign(rhat=[1.02, 1.0]))
    assert not _meets_criteria(good.assign(ess_tail=[999, 2000]))


def test_scale_setting_is_read_from_the_problem():
    assert noise_scale_setting(load("toy")) == "fixed"
    assert noise_scale_setting(load("toy", scales="inferred")) == "inferred"


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:the toy reference does not meet the convergence criteria")
def test_rwmh_reference_runs_segments_in_processes_and_resumes(tmp_path):
    toy = load("toy", scales="inferred")
    reference = rwmh_reference(toy, tmp_path, n_chains=2, segment_steps=20, max_segments=2)
    assert reference.theta.shape == (2, 20, 14)
    assert set(reference.noise_scales) == {"s2_day", "s2_night"}
    assert set(rwmh_diagnostics(tmp_path).index) >= {"log_s2_day", "log_s2_night"}
    segments = sorted(tmp_path.glob("chain_*/segment_*.npz"))
    assert len(segments) == 4
    modified = [path.stat().st_mtime for path in segments]
    rwmh_reference(toy, tmp_path, n_chains=2, segment_steps=20, max_segments=2)
    assert [path.stat().st_mtime for path in segments] == modified
    pilot_runs = json.loads((tmp_path / "settings.json").read_text())["pilot_forward_runs"]
    assert reference.provenance["forward_runs"] == pilot_runs + 4 * 21


@pytest.mark.smoke
def test_an_unconverged_run_is_marked_and_warned_about(tmp_path):
    toy = load("toy")
    with pytest.warns(UserWarning, match="does not meet the convergence criteria"):
        reference = rwmh_reference(toy, tmp_path, n_chains=2, segment_steps=10, max_segments=2)
    assert reference.provenance["converged"] is False
    assert reference.provenance["wall_seconds"] > 0
    with pytest.raises(ValueError, match="holds a run with 2 chains, not 3"):
        rwmh_reference(toy, tmp_path, n_chains=3, segment_steps=10, max_segments=2)


def test_a_reference_needs_two_chains(tmp_path):
    with pytest.raises(ValueError, match="at least two chains"):
        start_rwmh_reference(load("toy"), tmp_path, n_chains=1)


def test_no_chains_run_nothing():
    assert run_in_processes(print, [], max_workers=2) == []
