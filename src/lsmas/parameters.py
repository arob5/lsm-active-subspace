r"""The map from calibration parameters to SIPNET's parameters.

Most calibration parameters are SIPNET parameters under the same name. The others reparameterize
SIPNET's, for one of two reasons. Some remove an exact redundancy: SIPNET's output depends on its
photosynthesis parameters only through two combinations, and those are calibrated instead. The rest
are bijective changes of coordinates, chosen so that each parameter's prior can be stated on its
own and in the units measurements report: respiration rates at 10 °C instead of 0 °C, allocation as
successive shares, initial carbon pools as amounts. Each function here returns pySIPNET keyword
arguments; :func:`calibration_to_sipnet` combines them into a complete set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from probpipe import NumericArraySpec, NumericRecordSpec
from pysipnet.parameters import SIPNET_PARAMS_BY_GROUP

__all__ = [
    "SIPNET_PARAMETERS",
    "allocation_to_sipnet",
    "calibration_to_sipnet",
    "initial_pools_to_sipnet",
    "parameter_record_spec",
    "photosynthesis_temperatures_to_sipnet",
    "photosynthesis_to_sipnet",
    "respiration_rates_to_sipnet",
    "soil_respiration_to_sipnet",
]

#: Every SIPNET parameter, by pySIPNET's name.
SIPNET_PARAMETERS = frozenset(name for names in SIPNET_PARAMS_BY_GROUP.values() for name in names)

#: Calibration parameters that are SIPNET parameters under the same name.
_SIPNET_PARAMETERS_CALIBRATED_AS_THEY_ARE = (
    "half_saturation_light",
    "vapor_pressure_deficit_slope",
    "vapor_pressure_deficit_exponent",
    "light_extinction_coefficient",
    "water_use_efficiency",
    "water_removal_fraction",
    "soil_water_holding_capacity",
    "interception_evaporation_fraction",
    "fast_flow_fraction",
    "snow_melt_rate",
    "aerodynamic_resistance_constant",
    "soil_resistance_intercept",
    "soil_resistance_slope",
    "leaf_on_growing_degree_days",
    "leaf_off_day",
    "leaf_on_growth",
    "leaf_off_fall_fraction",
    "leaf_turnover_rate",
    "wood_turnover_rate",
    "fine_root_turnover_rate",
    "coarse_root_turnover_rate",
    "soil_respiration_moisture_exponent",
    "frozen_soil_threshold",
    "frozen_soil_foliar_respiration_factor",
    "frozen_soil_water_fraction",
)

#: Initial conditions, by calibration name and SIPNET name.
_INITIAL_CONDITIONS = {
    "initial_soil_carbon": "soil_carbon",
    "initial_soil_wetness": "soil_wetness_fraction",
    "initial_snow_water_equivalent": "snow_water_equivalent",
}


def calibration_to_sipnet(theta: Mapping[str, Any], fixed: Mapping[str, float]) -> dict[str, float]:
    r"""Every SIPNET parameter, from the calibration parameters and the fixed SIPNET values.

    The calibration parameters $\theta$ reach SIPNET through the maps of this module:
    :func:`photosynthesis_to_sipnet`, :func:`photosynthesis_temperatures_to_sipnet`,
    :func:`respiration_rates_to_sipnet`, :func:`soil_respiration_to_sipnet`,
    :func:`allocation_to_sipnet` and :func:`initial_pools_to_sipnet`. The initial soil carbon,
    soil wetness and snow water equivalent are SIPNET's ``soil_carbon``,
    ``soil_wetness_fraction`` and ``snow_water_equivalent``, and the remaining calibration
    parameters are SIPNET parameters under the same name.

    Parameters
    ----------
    theta : Mapping of str to float
        The calibration parameters, by name.
    fixed : Mapping of str to float
        The SIPNET parameters that are not calibrated, by pySIPNET's name. They must include
        ``daily_mean_photosynthesis_fraction`` and ``leaf_carbon_fraction``, which the
        photosynthesis map reads.

    Returns
    -------
    dict of str to float
        A value for each of SIPNET's parameters, :data:`SIPNET_PARAMETERS`.

    Raises
    ------
    KeyError
        If *theta* lacks a calibration parameter.
    ValueError
        Unless the calibration parameters and *fixed* together set each SIPNET parameter exactly
        once.
    """
    calibrated = (
        {name: float(theta[name]) for name in _SIPNET_PARAMETERS_CALIBRATED_AS_THEY_ARE}
        | {sipnet: float(theta[name]) for name, sipnet in _INITIAL_CONDITIONS.items()}
        | photosynthesis_to_sipnet(
            theta["photosynthetic_capacity"],
            theta["respiration_share"],
            daily_mean_photosynthesis_fraction=fixed["daily_mean_photosynthesis_fraction"],
            leaf_carbon_fraction=fixed["leaf_carbon_fraction"],
        )
        | photosynthesis_temperatures_to_sipnet(
            theta["optimum_photosynthesis_temperature"], theta["photosynthesis_temperature_range"]
        )
        | respiration_rates_to_sipnet(
            wood_rate_at_10c=theta["wood_respiration_rate_at_10c"],
            wood_q10=theta["wood_respiration_q10"],
            fine_root_rate_at_10c=theta["fine_root_respiration_rate_at_10c"],
            fine_root_q10=theta["fine_root_respiration_q10"],
            coarse_root_rate_at_10c=theta["coarse_root_respiration_rate_at_10c"],
            coarse_root_q10=theta["coarse_root_respiration_q10"],
        )
        | soil_respiration_to_sipnet(
            theta["soil_respiration_flux_at_10c"],
            theta["soil_respiration_q10"],
            soil_carbon=float(theta["initial_soil_carbon"]),
        )
        | allocation_to_sipnet(
            theta["leaf_allocation"],
            theta["wood_share_of_nonleaf_allocation"],
            theta["fine_root_share_of_root_allocation"],
        )
        | initial_pools_to_sipnet(
            theta["initial_wood_carbon"],
            theta["initial_coarse_root_carbon"],
            theta["initial_fine_root_carbon"],
        )
    )
    _check_each_set_once(calibrated, fixed)
    return dict(fixed) | calibrated


def parameter_record_spec(names: Sequence[str]) -> NumericRecordSpec:
    r"""The ProbPipe declaration of a record of scalar calibration parameters $\theta$.

    A forward map declares its input with it, as ``input_spec={"theta": parameter_record_spec(
    names)}``, and the likelihood reads the record's declaration from there.

    Parameters
    ----------
    names : sequence of str
        The parameters, in the record's order.
    """
    return NumericRecordSpec(**{name: NumericArraySpec(()) for name in names})


def photosynthesis_to_sipnet(
    photosynthetic_capacity: float,
    respiration_share: float,
    *,
    daily_mean_photosynthesis_fraction: float,
    leaf_carbon_fraction: float,
) -> dict[str, float]:
    r"""SIPNET's photosynthesis parameters from the two combinations the output depends on.

    SIPNET's gross photosynthesis and foliar respiration depend on the maximum photosynthesis rate
    $A$, the daily-mean fraction $f$, the foliar respiration fraction $r$ and the leaf carbon
    fraction $c$ only through

    $$
    P = \frac{A\,(f + r)}{c} \qquad\text{and}\qquad \rho = \frac{r}{f + r},
    $$

    the photosynthetic capacity (nmol CO₂ per g leaf carbon per s) and the share of it respired by
    leaves. So $f$ and $c$ are fixed, and $P$ and $\rho$ are calibrated. Inverting,

    $$
    A = \frac{P\, c\, (1 - \rho)}{f}, \qquad r = \frac{\rho\, f}{1 - \rho}.
    $$

    Parameters
    ----------
    photosynthetic_capacity : float
        $P > 0$.
    respiration_share : float
        $\rho \in (0, 1)$.
    daily_mean_photosynthesis_fraction : float
        $f$, held fixed.
    leaf_carbon_fraction : float
        $c$, held fixed.

    Returns
    -------
    dict of str to float
        ``max_photosynthesis_rate`` ($A$) and ``foliar_respiration_fraction`` ($r$).
    """
    capacity, share = float(photosynthetic_capacity), float(respiration_share)
    return {
        "max_photosynthesis_rate": capacity
        * leaf_carbon_fraction
        * (1.0 - share)
        / daily_mean_photosynthesis_fraction,
        "foliar_respiration_fraction": share * daily_mean_photosynthesis_fraction / (1.0 - share),
    }


def photosynthesis_temperatures_to_sipnet(
    optimum_temperature: float, temperature_range: float
) -> dict[str, float]:
    r"""SIPNET's photosynthesis temperatures from the optimum and the range.

    SIPNET's temperature effect on photosynthesis is a parabola that peaks at the optimum
    $T_{\mathrm{opt}}$ and falls to zero at $T_{\mathrm{opt}} \pm \Delta$, where
    $\Delta = T_{\mathrm{opt}} - T_{\min}$. Calibrating the range $\Delta > 0$ instead of
    $T_{\min}$ keeps $T_{\min} < T_{\mathrm{opt}}$ for any values:

    $$
    T_{\min} = T_{\mathrm{opt}} - \Delta.
    $$

    Parameters
    ----------
    optimum_temperature : float
        $T_{\mathrm{opt}}$, in °C.
    temperature_range : float
        $\Delta > 0$, in °C.

    Returns
    -------
    dict of str to float
        ``optimum_photosynthesis_temperature`` and ``min_photosynthesis_temperature``.
    """
    optimum = float(optimum_temperature)
    return {
        "optimum_photosynthesis_temperature": optimum,
        "min_photosynthesis_temperature": optimum - float(temperature_range),
    }


def respiration_rates_to_sipnet(
    *,
    wood_rate_at_10c: float,
    wood_q10: float,
    fine_root_rate_at_10c: float,
    fine_root_q10: float,
    coarse_root_rate_at_10c: float,
    coarse_root_q10: float,
) -> dict[str, float]:
    r"""SIPNET's wood and root respiration parameters from their rates at 10 °C.

    SIPNET's maintenance respiration of a pool of carbon $C$ at temperature $T$ (°C) is
    $C\, k\, Q_{10}^{T/10}$, for a base rate $k$ (yr⁻¹, the rate at 0 °C). Wood respiration uses
    the air temperature, root respiration the soil temperature. Measurements report rates near the
    site's temperatures, so the rate at 10 °C, $k_{10} = k\, Q_{10}$, is calibrated instead of
    $k$, which lets its prior be stated independently of the $Q_{10}$:

    $$
    k = \frac{k_{10}}{Q_{10}}.
    $$

    The wood $Q_{10}$ is also the $Q_{10}$ of foliar respiration.

    Parameters
    ----------
    wood_rate_at_10c, fine_root_rate_at_10c, coarse_root_rate_at_10c : float
        $k_{10} > 0$ for wood, fine roots and coarse roots, in yr⁻¹.
    wood_q10, fine_root_q10, coarse_root_q10 : float
        $Q_{10} > 0$ for each.

    Returns
    -------
    dict of str to float
        ``base_wood_respiration_rate``, ``wood_respiration_q10``,
        ``base_fine_root_respiration_rate``, ``fine_root_respiration_q10``,
        ``base_coarse_root_respiration_rate`` and ``coarse_root_respiration_q10``.
    """
    wood_q10, fine_root_q10, coarse_root_q10 = (
        float(wood_q10),
        float(fine_root_q10),
        float(coarse_root_q10),
    )
    return {
        "base_wood_respiration_rate": float(wood_rate_at_10c) / wood_q10,
        "wood_respiration_q10": wood_q10,
        "base_fine_root_respiration_rate": float(fine_root_rate_at_10c) / fine_root_q10,
        "fine_root_respiration_q10": fine_root_q10,
        "base_coarse_root_respiration_rate": float(coarse_root_rate_at_10c) / coarse_root_q10,
        "coarse_root_respiration_q10": coarse_root_q10,
    }


def soil_respiration_to_sipnet(
    soil_respiration_flux_at_10c: float, soil_respiration_q10: float, *, soil_carbon: float
) -> dict[str, float]:
    r"""SIPNET's soil respiration parameters from the flux at 10 °C and the $Q_{10}$.

    SIPNET's heterotrophic respiration, before its moisture effect, is
    $R(T) = C\, k\, Q_{10}^{T/10}$ for soil carbon $C$ (g C m⁻²), base rate $k$ (yr⁻¹) and soil
    temperature $T$ (°C). Calibrating the flux at 10 °C, $F_{10} = R(10) = C\, k\, Q_{10}$, instead
    of $k$ decorrelates the level of respiration from its temperature sensitivity and from the
    size of the soil pool. Then

    $$
    k = \frac{F_{10}}{C\, Q_{10}}.
    $$

    Parameters
    ----------
    soil_respiration_flux_at_10c : float
        $F_{10} > 0$, in g C m⁻² yr⁻¹.
    soil_respiration_q10 : float
        $Q_{10} > 0$.
    soil_carbon : float
        $C > 0$, the initial soil carbon in g C m⁻².

    Returns
    -------
    dict of str to float
        ``base_soil_respiration_rate`` ($k$) and ``soil_respiration_q10``.
    """
    q10 = float(soil_respiration_q10)
    return {
        "base_soil_respiration_rate": float(soil_respiration_flux_at_10c) / (soil_carbon * q10),
        "soil_respiration_q10": q10,
    }


def allocation_to_sipnet(
    leaf_allocation: float,
    wood_share_of_nonleaf_allocation: float,
    fine_root_share_of_root_allocation: float,
) -> dict[str, float]:
    r"""SIPNET's allocation fractions from successive shares.

    SIPNET splits new growth among leaves, wood, fine roots and coarse roots, and stops with an
    error unless the first three fractions are non-negative and sum to at most 1. Stating them as
    successive shares, each in $(0, 1)$, makes every value valid: leaves take $\ell$, wood takes a
    share $w$ of the rest, fine roots a share $f$ of what remains, and coarse roots the remainder,

    $$
    a_{\mathrm{leaf}} = \ell, \qquad
    a_{\mathrm{wood}} = (1 - \ell)\, w, \qquad
    a_{\mathrm{fine}} = (1 - \ell)(1 - w)\, f, \qquad
    a_{\mathrm{coarse}} = (1 - \ell)(1 - w)(1 - f).
    $$

    Parameters
    ----------
    leaf_allocation : float
        $\ell \in (0, 1)$.
    wood_share_of_nonleaf_allocation : float
        $w \in (0, 1)$.
    fine_root_share_of_root_allocation : float
        $f \in (0, 1)$.

    Returns
    -------
    dict of str to float
        ``leaf_allocation``, ``wood_allocation`` and ``fine_root_allocation``; SIPNET gives coarse
        roots the remainder.
    """
    leaf, wood_share, fine_root_share = (
        float(leaf_allocation),
        float(wood_share_of_nonleaf_allocation),
        float(fine_root_share_of_root_allocation),
    )
    return {
        "leaf_allocation": leaf,
        "wood_allocation": (1.0 - leaf) * wood_share,
        "fine_root_allocation": (1.0 - leaf) * (1.0 - wood_share) * fine_root_share,
    }


def initial_pools_to_sipnet(
    wood_carbon: float, coarse_root_carbon: float, fine_root_carbon: float
) -> dict[str, float]:
    r"""SIPNET's initial woody carbon and its root fractions from the three pools.

    SIPNET starts from a total woody carbon $T$ and the fractions $c$ and $f$ of it in coarse and
    fine roots; the rest is above-ground wood. Stating the three pools $W$, $C$ and $F$ (g C m⁻²)
    instead lets each have its own prior, and gives

    $$
    T = W + C + F, \qquad c = \frac{C}{T}, \qquad f = \frac{F}{T}.
    $$

    Parameters
    ----------
    wood_carbon : float
        $W > 0$, above-ground wood.
    coarse_root_carbon : float
        $C > 0$.
    fine_root_carbon : float
        $F > 0$.

    Returns
    -------
    dict of str to float
        ``total_wood_carbon`` ($T$), ``coarse_root_fraction`` ($c$) and ``fine_root_fraction``
        ($f$).
    """
    coarse_roots, fine_roots = float(coarse_root_carbon), float(fine_root_carbon)
    total = float(wood_carbon) + coarse_roots + fine_roots
    return {
        "total_wood_carbon": total,
        "coarse_root_fraction": coarse_roots / total,
        "fine_root_fraction": fine_roots / total,
    }


def _check_each_set_once(calibrated: Mapping[str, float], fixed: Mapping[str, float]) -> None:
    """Raise ``ValueError`` unless *calibrated* and *fixed* share out SIPNET's parameters."""
    problems = []
    if both := sorted(calibrated.keys() & fixed.keys()):
        problems.append(f"both calibrated and fixed: {both}")
    if missing := sorted(SIPNET_PARAMETERS - calibrated.keys() - fixed.keys()):
        problems.append(f"neither calibrated nor fixed: {missing}")
    if unknown := sorted(fixed.keys() - SIPNET_PARAMETERS):
        problems.append(f"fixed but not SIPNET parameters: {unknown}")
    if problems:
        raise ValueError("each SIPNET parameter must be set exactly once; " + "; ".join(problems))
