r"""The NEE likelihood shared by every problem, and a function that builds a problem from it.

Every problem calibrates a forward map $G$ against the same 400 half-monthly day and night NEE
features $y$, with the same likelihood,

$$
y_i \mid \theta, s^2 \sim \mathcal N\!\left(G_i(\theta),\; s^2_{k(i)}\, v_{k(i)}\right),
\qquad v_{\mathrm{day}} = 8,\; v_{\mathrm{night}} = 1.5,
$$

tempered by the learning rate of :func:`~lsmas.learning_rate.effective_sample_size_learning_rate`
with 3 observations per class-year. The noise scales $s^2_k$ are fixed at 1 or inferred with
independent $\mathrm{IG}(2, 1.678)$ priors. Problems differ only in $G$ and in which calibration
parameters they calibrate; :func:`nee_problem` builds one from those two.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from probpipe import Distribution, Function, InverseGamma, NumericArraySpec, NumericRecordSpec

from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas.data import HarvardForestData
from lsmas.learning_rate import effective_sample_size_learning_rate
from lsmas.priors import independent_prior
from lsmas.problems.inverse_problem import InverseProblem

__all__ = [
    "NOISE_SCALE_PRIOR",
    "OBSERVATIONS_PER_CLASS_YEAR",
    "REFERENCE_VARIANCE",
    "nee_problem",
]

#: The reference variances $v_k$ of the day and night features, in (µmol CO₂ m⁻² s⁻¹)².
REFERENCE_VARIANCE = {"day": 8.0, "night": 1.5}

#: Effective independent observations per class-year of data, which sets the learning rate.
OBSERVATIONS_PER_CLASS_YEAR = 3.0

#: The inverse-gamma prior on each class's noise scale: shape 2, scale 1.678, so median 1.
NOISE_SCALE_PRIOR = {"concentration": 2.0, "scale": 1.678}


def nee_problem(
    name: str,
    description: str,
    forward: Function,
    laws: Mapping[str, Distribution],
    *,
    scales: Literal["fixed", "inferred"],
    data: HarvardForestData,
) -> InverseProblem:
    r"""The problem that calibrates *forward* against the NEE features with the shared likelihood.

    Parameters
    ----------
    name : str
        The problem's short identifier.
    description : str
        A sentence saying what the problem is.
    forward : Function
        $G$: a ProbPipe ``Function`` of the record of calibration parameters, returning the 400
        predicted features.
    laws : Mapping of str to Distribution
        The prior of each calibration parameter, by name, in the order of the parameter record.
    scales : {"fixed", "inferred"}
        ``"fixed"`` sets $s^2_{\mathrm{day}} = s^2_{\mathrm{night}} = 1$. ``"inferred"`` makes them
        parameters ``s2_day`` and ``s2_night`` with independent $\mathrm{IG}(2, 1.678)$ priors.
    data : HarvardForestData
        The observations, their classes and dates.
    """
    likelihood = GaussianSimulatorLikelihood(
        forward,
        output_name="nee",
        theta_spec=NumericRecordSpec(**{parameter: NumericArraySpec(()) for parameter in laws}),
        reference_variance=np.vectorize(REFERENCE_VARIANCE.get)(data.feature_classes),
        classes=data.feature_classes,
        scales=_likelihood_scales(scales),
        label="nee_likelihood",
    )
    prior = _prior_with_noise_scales(laws, likelihood.random_scale_slots)
    return InverseProblem(
        name=name,
        description=description,
        likelihood=likelihood,
        prior=prior,
        data={"nee": data.observations},
        learning_rate={"nee": _feature_learning_rates(data)},
        coordinates=Coordinates.from_prior(prior, "theta"),
    )


def _prior_with_noise_scales(
    laws: Mapping[str, Distribution], random_scale_slots: Sequence[str]
) -> Any:
    """The prior of the calibration parameters and of any noise scales that are inferred."""
    prior = independent_prior(laws)
    for slot in random_scale_slots:
        prior = prior & InverseGamma(slot, **NOISE_SCALE_PRIOR)
    return prior


def _likelihood_scales(scales: str) -> dict[str, str | float]:
    """The likelihood's scale for each class: a slot name when inferred, 1 when fixed."""
    if scales == "fixed":
        return {"day": 1.0, "night": 1.0}
    if scales == "inferred":
        return {"day": "s2_day", "night": "s2_night"}
    raise ValueError(f"scales must be 'fixed' or 'inferred', got {scales!r}")


def _feature_learning_rates(data: HarvardForestData) -> np.ndarray:
    """The per-feature learning rate of the NEE features."""
    return effective_sample_size_learning_rate(
        data.feature_classes,
        data.features["year"].to_numpy(),
        data.features["month"].to_numpy(),
        observations_per_class_year=OBSERVATIONS_PER_CLASS_YEAR,
    )
