r"""The Harvard Forest inverse problems: all 46 calibration parameters, or a small problem of ten.

Both calibrate SIPNET against the same 400 half-monthly day and night NEE features from Harvard
Forest, 2012–2020, with the likelihood every problem shares (:mod:`lsmas.problems.nee_likelihood`)
and the same independent prior, from the shipped prior table. :func:`harvard_forest` calibrates
every parameter. :func:`harvard_forest_small` calibrates ten and holds the rest at their prior
medians, so that a long MCMC run, and with it a trustworthy reference posterior, is affordable.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from probpipe import quantile

from lsmas.data import HarvardForestData, load_harvard_forest
from lsmas.parameters import calibration_to_sipnet
from lsmas.priors import independent_prior, laws_from_table
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.problems.nee_likelihood import nee_problem
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
    features is the one every problem shares (:mod:`lsmas.problems.nee_likelihood`).

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
    return _harvard_forest_problem(
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
    ten, :data:`SMALL_PROBLEM_PARAMETERS`, are chosen in groups whose parameters affect the
    predictions jointly, so that the leading directions of the subspace combine several parameters
    rather than lining up with single parameters.
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
    return _harvard_forest_problem(
        "harvard_forest_small",
        f"Harvard Forest NEE, 2012-2020, with {len(SMALL_PROBLEM_PARAMETERS)} of the "
        f"{len(data.prior_table)} calibration parameters; the others are held at their prior "
        f"medians (noise scales {scales}).",
        SMALL_PROBLEM_PARAMETERS,
        scales=scales,
        data=data,
        max_workers=max_workers,
    )


def _harvard_forest_problem(
    name: str,
    description: str,
    calibrated: Sequence[str],
    *,
    scales: Literal["fixed", "inferred"],
    data: HarvardForestData,
    max_workers: int,
) -> InverseProblem:
    """The problem that calibrates *calibrated* and holds every other parameter at its median."""
    laws = laws_from_table(data.prior_table)
    if unknown := sorted(set(calibrated) - laws.keys()):
        raise ValueError(f"not calibration parameters: {unknown}")
    medians = quantile(independent_prior(laws), 0.5)["theta"]
    held = {
        parameter: float(medians[parameter]) for parameter in laws if parameter not in calibrated
    }
    forward = feature_map(
        SIPNETSimulator(data.driver_path, {}),
        data.operator,
        functools.partial(
            _sipnet_parameters_with_held_values,
            calibrated=tuple(calibrated),
            held=held,
            fixed=data.fixed_sipnet_parameters,
        ),
        parameter_names=calibrated,
        max_workers=max_workers,
    )
    return nee_problem(
        name,
        description,
        forward,
        {parameter: laws[parameter] for parameter in calibrated},
        scales=scales,
        data=data,
    )


def _sipnet_parameters_with_held_values(
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
