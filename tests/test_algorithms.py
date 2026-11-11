import numpy as np
import pytest

from lsmas.algorithms import AlgorithmResult, active_subspace_mcmc
from lsmas.algorithms.active_subspace_mcmc import _independent_stage_seeds
from lsmas.problems import load_inverse_problem

pytestmark = pytest.mark.smoke

SHORT_CHAINS = {"num_chains": 2, "num_warmup": 100, "num_results": 200}


@pytest.fixture(scope="module")
def toy_result():
    return active_subspace_mcmc(
        load_inverse_problem("toy", scales="inferred"), sampler_options=SHORT_CHAINS, seed=1
    )


def test_result_holds_the_lifted_posterior_over_every_parameter(toy_result):
    assert isinstance(toy_result, AlgorithmResult)
    theta = np.asarray(toy_result.posterior.atoms["theta"].to_vector())
    assert theta.shape == (2 * 200, 14)
    assert np.all(np.isfinite(theta))
    assert set(toy_result.posterior.atoms.raw) == {"theta", "s2"}


def test_rank_follows_the_rule_and_counts_follow_the_stages(toy_result):
    subspace = toy_result.subspace
    discarded = subspace.eigenvalues[subspace.rank :].sum()
    assert discarded <= 0.5 < subspace.eigenvalues[subspace.rank - 1 :].sum()
    n_steps = toy_result.exploration_posterior.annotations["arviz/eki_history"]["beta"].size
    assert toy_result.forward_runs == {"exploration": 100 * (n_steps + 1), "sampling": 2 * 301}
    assert toy_result.total_forward_runs == sum(toy_result.forward_runs.values())


def test_a_given_rank_is_used():
    result = active_subspace_mcmc(
        load_inverse_problem("toy"), rank=2, sampler_options=SHORT_CHAINS, seed=2
    )
    assert result.subspace.rank == 2
    assert result.reduced_posterior.atoms.raw["theta"].keys() == {"active_1", "active_2"}


def test_runs_on_sipnet():
    result = active_subspace_mcmc(
        load_inverse_problem("harvard_forest_small"),
        n_particles=16,
        rank=2,
        sampler_options={"num_chains": 1, "num_warmup": 5, "num_results": 5},
        seed=3,
    )
    theta = np.asarray(result.posterior.atoms["theta"].to_vector())
    assert theta.shape == (5, 10)
    assert np.all(np.isfinite(theta))


def test_independent_stage_seeds_of_different_root_seeds_never_coincide():
    seeds = [seed for root in range(50) for seed in _independent_stage_seeds(root)]
    assert len(set(seeds)) == len(seeds)
    assert _independent_stage_seeds(3) == _independent_stage_seeds(3)
