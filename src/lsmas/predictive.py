r"""Draws of the observations predicted by the model."""

from __future__ import annotations

import numpy as np
from probpipe import Normal, NumericRecordBatch, sample

from lsmas.problems import InverseProblem

__all__ = ["prior_predictive"]


# STOPGAP(PG-01): becomes sample(problem.model, n_draws) once joints can sample non-JAX kernels.
def prior_predictive(problem: InverseProblem, n_draws: int) -> NumericRecordBatch:
    r"""Joint draws of the parameters and the observations from the model.

    Each draw is

    $$
    (\theta, s^2) \sim \pi, \qquad
    y_i = G_i(\theta) + \sqrt{s^2_{k(i)}\, v_i}\; \varepsilon_i, \quad
    \varepsilon_i \sim \mathcal N(0, 1) \text{ independently},
    $$

    so $y \mid \theta, s^2$ follows the problem's likelihood. The forward map runs once per draw, in
    parallel. Run it inside ``workflow_run(seed=...)`` to make the draws reproducible.

    ProbPipe's ``sample(problem.model, n_draws)`` would give the same draws directly, but ProbPipe
    cannot yet sample a model whose likelihood runs an external program; this function does the
    same thing in three steps.

    Parameters
    ----------
    problem : InverseProblem
        The problem whose model to draw from.
    n_draws : int
        How many draws.

    Returns
    -------
    NumericRecordBatch
        One record per draw, on the level ``sample``, holding the prior's components (``theta``
        and any noise scales) and the observations, as a draw of ``problem.model`` would.
    """
    likelihood = problem.likelihood
    parameters = sample(problem.prior, sample_shape=(n_draws,))
    theta = problem.coordinates.array_from_draws(parameters)
    predicted = np.asarray(problem.forward(problem.coordinates.record_batch(theta)).raw())
    variances = likelihood.noise_variances(
        {slot: np.asarray(parameters[slot].raw()) for slot in likelihood.random_scale_slots}
    )
    standard_noise = np.asarray(sample(Normal("noise", 0.0, 1.0), predicted.shape).raw())
    observations = predicted + np.sqrt(variances) * standard_noise
    return NumericRecordBatch(
        parameters.raw() | {likelihood.component: observations},
        "sample",
        label=f"prior predictive of {problem.name}",
    )
