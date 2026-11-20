"""Design 09: which control sets the number of draws of `expectation` and `pushforward` at the pin.

Run with ``uv run python design/prototypes/09-estimator-bodies/controls.py``.
"""

import jax.numpy as jnp
import numpy as np
from probpipe import EmpiricalDistribution, Normal, expectation, pushforward, workflow_run

x = Normal("x", 0.0, 1.0)
values = np.arange(1000.0)
weights = np.random.default_rng(0).uniform(size=1000)
held = EmpiricalDistribution(jnp.asarray(values), weights=weights, output_name="v")
weighted_mean = values @ weights / weights.sum()

with workflow_run(seed=0):
    for control in ({"num_samples": 7}, {"max_evaluations": 7}):
        n_draws = pushforward.using(**control)(x, lambda v: v).weights.size
        print(f"pushforward of a normal, {control}: {n_draws} draws")
    for control in ({"num_samples": 4}, {"num_samples": 4000}, {"max_evaluations": 4}):
        estimates = [
            float(np.asarray(expectation.using(**control)(x, lambda v: v).raw)) for _ in range(30)
        ]
        print(f"expectation of a normal, {control}: sd of the estimate {np.std(estimates):.3f}")
    is_exact = np.isclose(float(np.asarray(expectation(held, lambda v: v).raw)), weighted_mean)
    print(f"expectation over 1000 weighted held draws is their weighted mean: {is_exact}")
    n_draws = pushforward(held, lambda v: v).weights.size
    print(f"pushforward over 1000 held draws: {n_draws} draws")
