r"""A small development version of the Harvard Forest problem: four calibration parameters.

This problem exists to develop and test the tools before the full problem's parameter set and prior
are settled. Its four calibration parameters and their priors are reasonable, and the many SIPNET
parameters it does not calibrate take representative fixed values; neither is a considered choice.
The data, the observation operator and the likelihood are those of the full problem.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from probpipe import InverseGamma, Normal, NumericArraySpec, NumericRecordSpec

from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood
from lsmas.data import HarvardForestData, load_harvard_forest
from lsmas.learning_rate import effective_sample_size_learning_rate
from lsmas.parameters import photosynthesis_to_sipnet, soil_respiration_to_sipnet
from lsmas.priors import log_normal_from_interval, logit_normal_from_interval
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.sipnet import SIPNETSimulator, feature_map

__all__ = ["harvard_forest_dev"]

#: The calibration parameters, in the order of the parameter vector.
PARAMETER_NAMES = (
    "photosynthetic_capacity",
    "respiration_share",
    "optimum_photosynthesis_temperature",
    "soil_respiration_q10",
)

#: The reference variances $v_k$ of the day and night features, in (µmol CO₂ m⁻² s⁻¹)².
REFERENCE_VARIANCE = {"day": 8.0, "night": 1.5}

#: Effective independent observations per class-year of data, which sets the learning rate.
OBSERVATIONS_PER_CLASS_YEAR = 3.0

#: The inverse-gamma prior on each class's noise scale: shape 2, scale 1.678, so median 1.
NOISE_SCALE_PRIOR = {"concentration": 2.0, "scale": 1.678}

#: SIPNET values this problem fixes beyond the shipped fixed values (representative, not fitted).
_REPRESENTATIVE_SIPNET_VALUES = {
    "leaf_area_index": 0.0,
    "total_wood_carbon": 10_000.0 / (1 - 0.03 - 0.16),
    "soil_carbon": 10_000.0,
    "leaf_allocation": 0.18,
    "wood_allocation": 0.45,
    "fine_root_allocation": 0.065,
    "base_wood_respiration_rate": 0.0155 / 1.80944,
    "base_coarse_root_respiration_rate": 0.0155 / 3.20614,
    "soil_water_holding_capacity": 47.4,
    "leaf_on_growth": 94.9,
    "leaf_on_growing_degree_days": 741.6,
    "half_saturation_light": 11.0,
}

#: The range psnTOpt − psnTMin, held fixed, in °C.
_PHOTOSYNTHESIS_TEMPERATURE_RANGE = 23.958

#: The soil respiration flux at 10 °C, held fixed, in g C m⁻² yr⁻¹.
_SOIL_RESPIRATION_AT_10C = 424.0


def harvard_forest_dev(
    scales: Literal["fixed", "inferred"] = "fixed",
    *,
    data: HarvardForestData | None = None,
    max_workers: int = 8,
) -> InverseProblem:
    r"""The development problem.

    The calibration parameters $\theta$ and their priors:

    | parameter | prior |
    |---|---|
    | ``photosynthetic_capacity`` (nmol g⁻¹ s⁻¹) | log-normal, 95% in [140, 450] |
    | ``respiration_share`` | logit-normal, 95% in [0.04, 0.20] |
    | ``optimum_photosynthesis_temperature`` (°C) | $\mathcal N(22, 2.5^2)$ |
    | ``soil_respiration_q10`` | log-normal, 95% in [1.3, 3.2] |

    The likelihood of the 400 day and night NEE features is

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
    simulator = SIPNETSimulator(
        data.driver_path, data.fixed_sipnet_parameters | _REPRESENTATIVE_SIPNET_VALUES
    )
    forward = feature_map(
        simulator, data.operator, _sipnet_parameters_from(data), max_workers=max_workers
    )
    scale_values = _scale_values(scales)
    likelihood = GaussianSimulatorLikelihood(
        forward,
        component="nee",
        theta_spec=NumericRecordSpec(**{name: NumericArraySpec(()) for name in PARAMETER_NAMES}),
        reference_variance=np.vectorize(REFERENCE_VARIANCE.get)(data.feature_classes),
        classes=data.feature_classes,
        scales=scale_values,
        label="nee_likelihood",
    )
    prior = _prior(random_scale_slots=likelihood.random_scale_slots)
    return InverseProblem(
        name="harvard_forest_dev",
        description=(
            "Harvard Forest NEE, 2012-2020, with four calibration parameters; a development "
            f"version of the full problem (noise scales {scales})."
        ),
        likelihood=likelihood,
        prior=prior,
        data={"nee": data.observations},
        learning_rate={"nee": _learning_rate(data)},
        forward=forward,
        coordinates=Coordinates.from_prior(prior, "theta"),
    )


def _theta_prior() -> Any:
    """The prior of the four calibration parameters, grouped under ``theta``."""
    factors = (
        log_normal_from_interval("photosynthetic_capacity", 140.0, 450.0)
        * logit_normal_from_interval("respiration_share", 0.04, 0.20)
        * Normal("optimum_photosynthesis_temperature", 22.0, 2.5)
        * log_normal_from_interval("soil_respiration_q10", 1.3, 3.2)
    )
    return factors.with_path_names({name: f"theta/{name}" for name in PARAMETER_NAMES})


def _prior(random_scale_slots: tuple[str, ...]) -> Any:
    """The prior of the calibration parameters and of any noise scales that are inferred."""
    prior = _theta_prior()
    for slot in random_scale_slots:
        prior = prior * InverseGamma(slot, **NOISE_SCALE_PRIOR)
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


def _sipnet_parameters_from(data: HarvardForestData) -> Any:
    """The map from this problem's calibration parameters to SIPNET parameters."""
    fixed = data.fixed_sipnet_parameters | _REPRESENTATIVE_SIPNET_VALUES

    def to_sipnet(theta: Mapping[str, Any]) -> dict[str, float]:
        optimum_temperature = float(theta["optimum_photosynthesis_temperature"])
        return (
            photosynthesis_to_sipnet(
                theta["photosynthetic_capacity"],
                theta["respiration_share"],
                daily_mean_photosynthesis_fraction=fixed["daily_mean_photosynthesis_fraction"],
                leaf_carbon_fraction=fixed["leaf_carbon_fraction"],
            )
            | soil_respiration_to_sipnet(
                _SOIL_RESPIRATION_AT_10C,
                theta["soil_respiration_q10"],
                soil_carbon=fixed["soil_carbon"],
            )
            | {
                "optimum_photosynthesis_temperature": optimum_temperature,
                "min_photosynthesis_temperature": optimum_temperature
                - _PHOTOSYNTHESIS_TEMPERATURE_RANGE,
            }
        )

    return to_sipnet
