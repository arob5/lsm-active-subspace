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
    z = np.asarray(toy.coordinates.whiten(eki.atoms["theta"]).to_vector()).reshape(60, -1)
    return Reference(
        problem_name="toy",
        scales="fixed",
        parameter_names=toy.coordinates.names,
        theta=theta,
        noise_scales={},
        information=exact_subspace(toy, z).information,
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
    table = run_study(study, tmp_path, reference=stand_in_reference)
    assert isinstance(table, pd.DataFrame) and len(table) == 4
    assert {"rank", "seed", "marginal_w1_mean", "captured_information", "forward_runs"} <= set(
        table
    )
    assert sorted(table["rank"].unique()) == [2, 4]
    runs = sorted(tmp_path.glob("*/scores.json"))
    assert len(runs) == 4
    modified = [path.stat().st_mtime for path in runs]
    again = run_study(study, tmp_path, reference=stand_in_reference)
    assert [path.stat().st_mtime for path in runs] == modified
    pd.testing.assert_frame_equal(again, table)


def test_a_stored_run_reads_back_its_draws_and_subspace(tmp_path, stand_in_reference):
    study = Study(
        name="one_run",
        problem="toy",
        scales="fixed",
        algorithm=active_subspace_mcmc,
        settings=[{"rank": 3, "sampler_options": SHORT_CHAINS}],
        seeds=[5],
    )
    table = run_study(study, tmp_path, reference=stand_in_reference)
    stored = load_run(tmp_path / table["run"].iloc[0])
    assert stored.setting["rank"] == 3 and stored.seed == 5
    assert stored.theta.shape == (100, 14)
    assert stored.subspace.rank == 3
    assert stored.subspace.basis.shape == (14, 3)
    assert stored.scores["run"] == table["run"].iloc[0]
    assert np.isfinite(stored.wall_seconds)


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
    row = run_study(study, tmp_path, reference=stand_in_reference).iloc[0]
    expected = score_names(load_inverse_problem("toy"))
    assert [name for name in row.index if name in expected] == expected
