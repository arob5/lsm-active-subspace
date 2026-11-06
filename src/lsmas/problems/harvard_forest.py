r"""The Harvard Forest inverse problems: all 46 calibration parameters, or a small problem of ten.

Both calibrate SIPNET against the same 400 half-monthly day and night NEE features from Harvard
Forest, 2012–2020, with the same likelihood and the same independent prior, from the shipped prior
table. :func:`harvard_forest` calibrates every parameter. :func:`harvard_forest_small` calibrates
ten and holds the rest at their prior medians, so that a long MCMC run, and with it a trustworthy
reference posterior, is affordable.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from probpipe import Distribution, InverseGamma, NumericArraySpec, NumericRecordSpec

from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas.data import HarvardForestData, load_harvard_forest
from lsmas.learning_rate import effective_sample_size_learning_rate
from lsmas.parameters import calibration_to_sipnet
from lsmas.priors import independent_prior, laws_from_table, prior_medians
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.sipnet import SIPNETSimulator, feature_map

__all__ = ["SMALL_PROBLEM_PARAMETERS", "harvard_forest", "harvard_forest_small"]

#: The calibration parameters of the small problem: groups that act together through the model
#: (light use, the vapor pressure deficit, water, respiration), so that its most informed
#: directions mix parameters, and none that acts as a step function.
SMALL_PROBLEM_PARAMETERS = (
    "photosynthetic_capacity",
    "half_saturation_light",
    "light_extinction_coefficient",
    "leaf_on_growth",
    "vapor_pressure_deficit_slope",
    "vapor_pressure_deficit_exponent",
    "soil_water_holding_capacity",
    "water_use_efficiency",
    "respiration_share",
    "soil_respiration_flux_at_10c",
)

#: The reference variances $v_k$ of the day and night features, in (µmol CO₂ m⁻² s⁻¹)².
REFERENCE_VARIANCE = {"day": 8.0, "night": 1.5}

#: Effective independent observations per class-year of data, which sets the learning rate.
OBSERVATIONS_PER_CLASS_YEAR = 3.0

#: The inverse-gamma prior on each class's noise scale: shape 2, scale 1.678, so median 1.
NOISE_SCALE_PRIOR = {"concentration": 2.0, "scale": 1.678}


def harvard_forest(
    scales: Literal["fixed", "inferred"] = "fixed",
    *,
    data: HarvardForestData | None = None,
    max_workers: int = 8,
) -> InverseProblem:
    r"""The full problem: all 46 calibration parameters.

    The calibration parameters $\theta$ have independent priors, one per row of the shipped prior
    table (:attr:`~lsmas.data.HarvardForestData.prior_table`), and reach SIPNET through
    :func:`~lsmas.parameters.calibration_to_sipnet`. The likelihood of the 400 day and night NEE
    features is

    $$
    y_i \mid \theta, s^2 \sim \mathcal N\!\left(G_i(\theta),\; s^2_{k(i)}\, v_{k(i)}\right),
    \qquad v_{\mathrm{day}} = 8,\; v_{\mathrm{night}} = 1.5,
    $$

    tempered by the learning rate of
    :func:`~lsmas.learning_rate.effective_sample_size_learning_rate` with 3 observations per
    class-year.

    Parameters
    ----------
    scales : {"fixed", "inferred"}
        ``"fixed"`` sets $s^2_{\mathrm{day}} = s^2_{\mathrm{night}} = 1$. ``"inferred"`` makes them
        parameters ``s2_day`` and ``s2_night`` with independent $\mathrm{IG}(2, 1.678)$ priors.
    data : HarvardForestData, optional
        The inputs, :func:`~lsmas.data.load_harvard_forest` by default.
    max_workers : int
        How many SIPNET runs the forward map runs at a time.
    """
    data = load_harvard_forest() if data is None else data
    return _problem(
        "harvard_forest",
        f"Harvard Forest NEE, 2012-2020, with all {len(data.prior_table)} calibration parameters "
        f"(noise scales {scales}).",
        tuple(data.prior_table.index),
        scales=scales,
        data=data,
        max_workers=max_workers,
    )


def harvard_forest_small(
    scales: Literal["fixed", "inferred"] = "fixed",
    *,
    data: HarvardForestData | None = None,
    max_workers: int = 8,
) -> InverseProblem:
    r"""The small problem: ten calibration parameters, the rest held at their prior medians.

    The data, the likelihood and the priors of the ten are those of :func:`harvard_forest`. The
    ten, :data:`SMALL_PROBLEM_PARAMETERS`, come in groups that act together through the model, so
    that the directions the data inform most mix parameters rather than follow them one by one.
    None of them acts as a step function: the leaf-on and leaf-off timing parameters, which switch
    on whole days, are held fixed. So a finite difference of the forward map, once its width is
    well above the rounding of SIPNET's output (NEE is written to three decimals), estimates the
    same derivative whatever the width.

    Parameters
    ----------
    scales : {"fixed", "inferred"}
        As for :func:`harvard_forest`.
    data : HarvardForestData, optional
        The inputs, :func:`~lsmas.data.load_harvard_forest` by default.
    max_workers : int
        How many SIPNET runs the forward map runs at a time.
    """
    data = load_harvard_forest() if data is None else data
    return _problem(
        "harvard_forest_small",
        f"Harvard Forest NEE, 2012-2020, with {len(SMALL_PROBLEM_PARAMETERS)} of the "
        f"{len(data.prior_table)} calibration parameters; the others are held at their prior "
        f"medians (noise scales {scales}).",
        SMALL_PROBLEM_PARAMETERS,
        scales=scales,
        data=data,
        max_workers=max_workers,
    )


def _problem(
    name: str,
    description: str,
    calibrated: Sequence[str],
    *,
    scales: str,
    data: HarvardForestData,
    max_workers: int,
) -> InverseProblem:
    """The problem that calibrates *calibrated* and holds every other parameter at its median."""
    laws = laws_from_table(data.prior_table)
    if unknown := sorted(set(calibrated) - laws.keys()):
        raise ValueError(f"not calibration parameters: {unknown}")
    held = {
        parameter: value
        for parameter, value in prior_medians(data.prior_table).items()
        if parameter not in calibrated
    }
    forward = feature_map(
        SIPNETSimulator(data.driver_path, {}),
        data.operator,
        functools.partial(
            _sipnet_parameters,
            calibrated=tuple(calibrated),
            held=held,
            fixed=data.fixed_sipnet_parameters,
        ),
        max_workers=max_workers,
    )
    likelihood = GaussianSimulatorLikelihood(
        forward,
        output_name="nee",
        theta_spec=NumericRecordSpec(
            **{parameter: NumericArraySpec(()) for parameter in calibrated}
        ),
        reference_variance=np.vectorize(REFERENCE_VARIANCE.get)(data.feature_classes),
        classes=data.feature_classes,
        scales=_scale_values(scales),
        label="nee_likelihood",
    )
    prior = _prior(
        {parameter: laws[parameter] for parameter in calibrated}, likelihood.random_scale_slots
    )
    return InverseProblem(
        name=name,
        description=description,
        likelihood=likelihood,
        prior=prior,
        data={"nee": data.observations},
        learning_rate={"nee": _learning_rate(data)},
        coordinates=Coordinates.from_prior(prior, "theta"),
    )


def _sipnet_parameters(
    theta: Mapping[str, Any],
    *,
    calibrated: Sequence[str],
    held: Mapping[str, float],
    fixed: Mapping[str, float],
) -> dict[str, float]:
    """All SIPNET parameters, from the calibrated values in *theta* and the held values *held*."""
    return calibration_to_sipnet(
        held | {parameter: theta[parameter] for parameter in calibrated}, fixed
    )


def _prior(laws: Mapping[str, Distribution], random_scale_slots: Sequence[str]) -> Distribution:
    """The prior of the calibration parameters and of any noise scales that are inferred."""
    prior = independent_prior(laws)
    for slot in random_scale_slots:
        prior = prior & InverseGamma(slot, **NOISE_SCALE_PRIOR)
    return prior


def _scale_values(scales: str) -> dict[str, str | float]:
    """The likelihood's scale for each class: a slot name when inferred, 1 when fixed."""
    if scales == "fixed":
        return {"day": 1.0, "night": 1.0}
    if scales == "inferred":
        return {"day": "s2_day", "night": "s2_night"}
    raise ValueError(f"scales must be 'fixed' or 'inferred', got {scales!r}")


def _learning_rate(data: HarvardForestData) -> np.ndarray:
    """The per-feature learning rate of the NEE features."""
    return effective_sample_size_learning_rate(
        data.feature_classes,
        data.features["year"].to_numpy(),
        data.features["month"].to_numpy(),
        observations_per_class_year=OBSERVATIONS_PER_CLASS_YEAR,
    )
