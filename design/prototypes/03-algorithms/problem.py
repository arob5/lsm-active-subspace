"""The toy as an InverseProblem, built the way lsmas builds the Harvard Forest problems.

Usage from the other scripts: ``from problem import toy_problem``. Data: ``"observed"`` (the real
Harvard Forest features) or ``"synthetic"`` (the toy at a fixed prior draw plus noise).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import Function, NumericArraySpec, NumericRecordSpec

from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas.priors import independent_prior
from lsmas.problems.nee_likelihood import REFERENCE_VARIANCE, _learning_rate, _scale_values, _prior
from lsmas.problems.inverse_problem import InverseProblem
from toy import DATA, LAWS, N_FEATURES, TOY_PARAMETERS, features

TRUTH_SEED = 11


def toy_forward():
    return Function(
        features,
        label="toy_features",
        output_spec=NumericArraySpec((N_FEATURES,), jnp.float64),
        traceable=True,
    )


def synthetic_truth_and_data():
    """A fixed prior draw, and the toy's features there plus the likelihood's noise (s2 = 1)."""
    rng = np.random.default_rng(TRUTH_SEED)
    from lsmas._stopgaps.sampling import draw_with_key

    truth = {
        name: float(np.asarray(draw_with_key(LAWS[name], jax.random.key(TRUTH_SEED + i), ())))
        for i, name in enumerate(TOY_PARAMETERS)
    }
    g = np.asarray(features({k: jnp.asarray(v) for k, v in truth.items()}))
    variance = np.vectorize(REFERENCE_VARIANCE.get)(DATA.feature_classes)
    return truth, g + rng.normal(size=g.shape) * np.sqrt(variance)


def toy_problem(data="observed", scales="fixed"):
    observations = DATA.observations if data == "observed" else synthetic_truth_and_data()[1]
    likelihood = GaussianSimulatorLikelihood(
        toy_forward(),
        output_name="nee",
        theta_spec=NumericRecordSpec(**{p: NumericArraySpec(()) for p in TOY_PARAMETERS}),
        reference_variance=np.vectorize(REFERENCE_VARIANCE.get)(DATA.feature_classes),
        classes=DATA.feature_classes,
        scales=_scale_values(scales),
        label="nee_likelihood",
    )
    prior = _prior({p: LAWS[p] for p in TOY_PARAMETERS}, likelihood.random_scale_slots)
    return InverseProblem(
        name="toy",
        description=f"toy, {data} data, scales {scales}",
        likelihood=likelihood,
        prior=prior,
        data={"nee": observations},
        learning_rate={"nee": _learning_rate(DATA)},
        coordinates=Coordinates.from_prior(prior, "theta"),
    )


_ = independent_prior  # (kept for readers: the prior is lsmas's independent prior)
