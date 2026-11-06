r"""Reparameterizations that map calibration parameters to SIPNET's parameters.

SIPNET has exact redundancies among its parameters: some combinations change the output and others
provably do not. Calibrating the identifiable combinations and fixing the rest removes those flat
directions before any algorithm sees them. Each function here returns pySIPNET keyword arguments.
"""

from __future__ import annotations

__all__ = ["photosynthesis_to_sipnet", "soil_respiration_to_sipnet"]


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


def soil_respiration_to_sipnet(
    soil_respiration_at_10c: float, soil_respiration_q10: float, *, soil_carbon: float
) -> dict[str, float]:
    r"""SIPNET's soil respiration parameters from the flux at 10 °C and the $Q_{10}$.

    SIPNET's heterotrophic respiration, before its moisture effect, is
    $R(T) = C\, k\, Q_{10}^{T/10}$ for soil carbon $C$ (g C m⁻²), base rate $k$ (yr⁻¹) and soil
    temperature $T$ (°C). Calibrating the flux at 10 °C, $F_{10} = R(10) = C\, k\, Q_{10}$, instead
    of $k$ decorrelates the level of respiration from its temperature sensitivity. Then

    $$
    k = \frac{F_{10}}{C\, Q_{10}}.
    $$

    Parameters
    ----------
    soil_respiration_at_10c : float
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
        "base_soil_respiration_rate": float(soil_respiration_at_10c) / (soil_carbon * q10),
        "soil_respiration_q10": q10,
    }
