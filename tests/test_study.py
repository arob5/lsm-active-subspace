import numpy as np
import pandas as pd
import pytest
from probpipe import workflow_run

from lsmas.algorithms import active_subspace_mcmc
from lsmas.experiments.metrics import score_names
from lsmas.experiments.reference import Reference, convergence_diagnostics
from lsmas.experiments.study import Study, load_run, run_study
from lsmas.problems import load_inverse_problem
from lsmas.tools import exact_subspace

pytestmark = pytest.mark.smoke

SHORT_CHAINS = {"num_chains": 1, "num_warmup": 50, "num_results": 100}


@pytest.fixture(scope="module")
def stand_in_reference():
    """A cheap reference for the toy (EKI draws): enough to exercise the runner, not to judge."""
    toy = load_inverse_problem("toy")
    with workflow_run(seed=9):
        eki = toy.condition(method="enskit_eki", method_options={"n_particles": 60})
    theta = np.asarray(eki.atoms["theta"].to_vector()).reshape(2, 30, -1)
    return Reference(
        problem_name="toy",
        scales="fixed",
        parameter_names=toy.coordinates.names,
        theta=theta,
        noise_scales={},
        information=exact_subspace(toy, eki).information,
        diagnostics=convergence_diagnostics(toy, theta, {}),
        provenance={"method": "stand-in"},
    )


def test_study_runs_each_setting_and_seed_once_and_resumes(tmp_path, stand_in_reference):
    study = Study(
        name="toy_rank",
        problem="toy",
        scales="fixed",
        algorithm=active_subspace_mcmc,
        settings=[{"rank": rank, "sampler_options": SHORT_CHAINS} for rank in (2, 4)],
        seeds=[1, 2],
    )
    scores_table = run_study(study, tmp_path, reference=stand_in_reference)
    assert isinstance(scores_table, pd.DataFrame) and len(scores_table) == 4
    assert {"rank", "seed", "marginal_w1_mean", "captured_information", "forward_runs"} <= set(
        scores_table
    )
    assert sorted(scores_table["rank"].unique()) == [2, 4]
    score_files = sorted(tmp_path.glob("*/scores.json"))
    assert len(score_files) == 4
    modification_times = [path.stat().st_mtime for path in score_files]
    rerun_table = run_study(study, tmp_path, reference=stand_in_reference)
    assert [path.stat().st_mtime for path in score_files] == modification_times
    pd.testing.assert_frame_equal(rerun_table, scores_table)


def test_a_stored_run_reads_back_its_draws_and_subspace(tmp_path, stand_in_reference):
    study = Study(
        name="one_run",
        problem="toy",
        scales="fixed",
        algorithm=active_subspace_mcmc,
        settings=[{"rank": 3, "sampler_options": SHORT_CHAINS}],
        seeds=[5],
    )
    scores_table = run_study(study, tmp_path, reference=stand_in_reference)
    stored_run = load_run(tmp_path / scores_table["run"].iloc[0])
    assert stored_run.setting["rank"] == 3 and stored_run.seed == 5
    assert stored_run.theta.shape == (100, 14)
    assert stored_run.subspace.rank == 3
    assert stored_run.subspace.basis.shape == (14, 3)
    assert stored_run.scores["run"] == scores_table["run"].iloc[0]
    assert np.isfinite(stored_run.wall_seconds)


def test_a_reference_of_another_problem_is_refused(tmp_path, stand_in_reference):
    study = Study("s", "toy", "inferred", active_subspace_mcmc, [{"rank": 2}], [1])
    with pytest.raises(ValueError, match=r"reference is for toy \(fixed\)"):
        run_study(study, tmp_path, reference=stand_in_reference)


def test_a_setting_named_like_a_score_is_refused_before_any_run(tmp_path, stand_in_reference):
    study = Study("s", "toy", "fixed", active_subspace_mcmc, [{"subspace_rank": 2}], [1])
    with pytest.raises(ValueError, match="subspace_rank"):
        run_study(study, tmp_path, reference=stand_in_reference)
    assert not list(tmp_path.glob("*/scores.json"))


def test_score_names_are_the_scores_a_run_gets(tmp_path, stand_in_reference):
    study = Study(
        "s",
        "toy",
        "fixed",
        active_subspace_mcmc,
        [{"rank": 2, "sampler_options": SHORT_CHAINS}],
        [3],
    )
    score_row = run_study(study, tmp_path, reference=stand_in_reference).iloc[0]
    expected = score_names(load_inverse_problem("toy"))
    assert [name for name in score_row.index if name in expected] == expected
