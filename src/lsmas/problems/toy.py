r"""The toy problem: a simplified SIPNET in JAX, differentiable, with 14 calibration parameters.

The toy calibrates against the real Harvard Forest data, with the likelihood every problem shares
(:mod:`lsmas.problems.nee_likelihood`), but its forward map is a small model written in JAX, so it
can be differentiated exactly and sampled with NUTS. It exists to check methods: exact Jacobians
show what a gradient-free estimate of a subspace estimates, and a NUTS posterior is a reference that
a reduced posterior can be scored against.

**The model.** At each 3-hour step of the shipped driver, with air temperature $T_a$ and soil
temperature $T_s$ (°C), photosynthetically active radiation $\mathrm{PAR}$ (mol m⁻² d⁻¹), vapor
pressure deficit $D$ (kPa), day of year $t$, and growing degree days $\Theta$ (°C d, the
year-to-date sum of $\max(T_a\, \Delta t, 0)$ over steps of length $\Delta t$ days):

$$
\begin{aligned}
C_L &= g\; \sigma\!\left(\frac{\Theta - \Theta_{\mathrm{on}}}{30}\right)
      \sigma\!\left(\frac{t_{\mathrm{off}} - t}{5}\right)
      && \text{leaf carbon, g C m}^{-2}, \\
P_C &= 1.2 \times 10^{-8} \cdot 86400 \cdot P\, C_L
      && \text{photosynthetic capacity of the canopy, g C m}^{-2}\,\text{d}^{-1}, \\
\mathrm{GPP} &= P_C\; f_T(T_a)\, f_D(D)\, f_L(\mathrm{PAR}, C_L / \ell), \\
R &= \rho\, P_C\, \left(Q_{10}^{\mathrm{veg}}\right)^{(T_a - T_{\mathrm{opt}})/10}
    + \frac{k_{10}\, C_W}{365} \left(Q_{10}^{\mathrm{veg}}\right)^{(T_a - 10)/10}
    + \frac{F_{10}}{365} \left(Q_{10}^{\mathrm{soil}}\right)^{(T_s - 10)/10}, \\
\mathrm{NEE} &= R - \mathrm{GPP},
\end{aligned}
$$

reported as a rate in µmol CO₂ m⁻² s⁻¹ and averaged into the 400 features by the observation
operator $A$. Here $\sigma$ is the logistic function, and the three effects are SIPNET's:

$$
f_T(T) = \max\!\left(0,\; 1 - \left(\frac{T - T_{\mathrm{opt}}}{\Delta}\right)^2\right),
$$

SIPNET's parabola, which vanishes at $T_{\min} = T_{\mathrm{opt}} - \Delta$ and
$T_{\max} = T_{\mathrm{opt}} + \Delta$, and

$$
f_D(D) = \max\!\left(0,\; 1 - \delta\, D^{\nu}\right),
\qquad
f_L(\mathrm{PAR}, \mathrm{LAI}) = \int_0^1 \left(1 - 2^{-\mathrm{PAR}\,
e^{-\kappa\, \mathrm{LAI}\, u} / \mathrm{PAR}_{1/2}}\right) du,
$$

the last by Simpson's rule on 6 canopy layers, as SIPNET computes it, and 0 without leaves or
light. The fixed constants are the leaf carbon per leaf area $\ell = 30.7092$ g C m⁻² (SIPNET's
fixed value), and the wood carbon $C_W$ and the foliar and wood $Q_{10}^{\mathrm{veg}}$, both at
their prior medians (8660 g C m⁻² and 1.79).

**The 14 calibration parameters** (:data:`TOY_PARAMETERS`) are rows of the shipped prior table,
with their priors: the photosynthetic capacity $P$, respiration share $\rho$, optimum temperature
$T_{\mathrm{opt}}$ and range $\Delta$, light half-saturation $\mathrm{PAR}_{1/2}$, VPD slope
$\delta$ and exponent $\nu$, light extinction coefficient $\kappa$, the leaf-on growing degree days
$\Theta_{\mathrm{on}}$, leaf-off day $t_{\mathrm{off}}$ and leaf-on growth $g$, the wood
respiration rate at 10 °C $k_{10}$, and the soil respiration flux at 10 °C $F_{10}$ with its
$Q_{10}^{\mathrm{soil}}$.

**How it differs from SIPNET.** The light, temperature and VPD effects, the form of gross
photosynthesis (with $P$ and $\rho$ entering as in
:func:`~lsmas.parameters.photosynthesis_to_sipnet`, so the leaf carbon fraction cancels) and the
three respiration terms are SIPNET's. The rest is simplified: leaf carbon follows smooth leaf-on and
leaf-off curves of fixed amplitude $g$, where SIPNET has a leaf pool that grows by allocation and
loses carbon to turnover and leaf fall; there is no water stress on photosynthesis and no moisture
effect on soil respiration; the wood carbon and the soil flux's reference pool are constant; roots,
frozen soil, snow and SIPNET's rounding of its output are left out. At the prior medians the toy's
NEE features correlate with SIPNET's at 0.98, and its gross photosynthesis is about 22% lower,
mostly because its leaves do not grow after leaf-on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import Function, NumericArraySpec
from pysipnet import ClimateDrivers

from lsmas.data import HarvardForestData, load_harvard_forest
from lsmas.priors import laws_from_table, prior_medians
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.problems.nee_likelihood import nee_problem

__all__ = ["TOY_PARAMETERS", "ToyCanopyModel", "toy"]

#: The toy's calibration parameters, in the order of the parameter record.
TOY_PARAMETERS = (
    "photosynthetic_capacity",
    "respiration_share",
    "optimum_photosynthesis_temperature",
    "photosynthesis_temperature_range",
    "half_saturation_light",
    "vapor_pressure_deficit_slope",
    "vapor_pressure_deficit_exponent",
    "light_extinction_coefficient",
    "leaf_on_growing_degree_days",
    "leaf_off_day",
    "leaf_on_growth",
    "wood_respiration_rate_at_10c",
    "soil_respiration_flux_at_10c",
    "soil_respiration_q10",
)

#: The widths of the smooth leaf-on (°C d) and leaf-off (d) transitions.
LEAF_ON_WIDTH, LEAF_OFF_WIDTH = 30.0, 5.0

#: Grams of carbon per nanomole of CO₂, times seconds per day: converts P C_L to g C m⁻² d⁻¹.
_CAPACITY_CONVERSION = 12e-9 * 86400

#: Micromoles of CO₂ per gram of carbon, over seconds per day: g C m⁻² d⁻¹ to µmol m⁻² s⁻¹.
_RATE_CONVERSION = 1e6 / 12.011 / 86400

#: Simpson's rule on SIPNET's 6 canopy layers: the weights of the 7 layer boundaries.
_CANOPY_DEPTHS = np.linspace(0.0, 1.0, 7)
_CANOPY_WEIGHTS = np.array([1.0, 4.0, 2.0, 4.0, 2.0, 4.0, 1.0]) / 18.0


def toy(
    scales: Literal["fixed", "inferred"] = "fixed", *, data: HarvardForestData | None = None
) -> InverseProblem:
    r"""The toy problem: :class:`ToyCanopyModel` calibrated against the Harvard Forest NEE features.

    Parameters
    ----------
    scales : {"fixed", "inferred"}
        ``"fixed"`` sets $s^2_{\mathrm{day}} = s^2_{\mathrm{night}} = 1$. ``"inferred"`` makes them
        parameters ``s2_day`` and ``s2_night`` with independent $\mathrm{IG}(2, 1.678)$ priors.
    data : HarvardForestData, optional
        The inputs, :func:`~lsmas.data.load_harvard_forest` by default.
    """
    data = load_harvard_forest() if data is None else data
    model = ToyCanopyModel.from_data(data)
    forward = Function(
        model.features,
        label="toy_features",
        output_spec=NumericArraySpec((model.n_features,), jnp.float64),
        traceable=True,
    )
    return nee_problem(
        "toy",
        f"A simplified SIPNET in JAX with {len(TOY_PARAMETERS)} calibration parameters, against "
        f"the Harvard Forest NEE features (noise scales {scales}).",
        forward,
        laws_from_table(data.prior_table.loc[list(TOY_PARAMETERS)]),
        scales=scales,
        data=data,
    )


@dataclass(frozen=True, eq=False)
class ToyCanopyModel:
    r"""The toy's canopy model at one site (see the module's description for the equations).

    Its methods take the record of the 14 calibration parameters, as a mapping from name to value,
    and are written in JAX, so they can be jitted, vectorized and differentiated.

    Attributes
    ----------
    air_temperature, soil_temperature : jax.Array
        $T_a$ and $T_s$ at each step, °C, shape $(S,)$.
    radiation : jax.Array
        $\mathrm{PAR}$, mol m⁻² d⁻¹.
    vapor_pressure_deficit : jax.Array
        $D$, kPa.
    day_of_year : jax.Array
        $t$, the day of year plus the hour over 24.
    growing_degree_days : jax.Array
        $\Theta$, °C d.
    leaf_carbon_per_area, wood_carbon, respiration_q10 : float
        $\ell$, $C_W$ and $Q_{10}^{\mathrm{veg}}$.
    operator_rows, operator_columns, operator_weights : jax.Array
        The observation operator $A$ in coordinate form: feature ``operator_rows[m]`` adds
        ``operator_weights[m]`` times the rate at step ``operator_columns[m]``.
    n_features : int
        $n$.
    """

    air_temperature: jax.Array
    soil_temperature: jax.Array
    radiation: jax.Array
    vapor_pressure_deficit: jax.Array
    day_of_year: jax.Array
    growing_degree_days: jax.Array
    leaf_carbon_per_area: float
    wood_carbon: float
    respiration_q10: float
    operator_rows: jax.Array
    operator_columns: jax.Array
    operator_weights: jax.Array
    n_features: int

    @classmethod
    def from_data(cls, data: HarvardForestData) -> ToyCanopyModel:
        """The model on the shipped driver and operator, with its constants from the shipped files.

        Parameters
        ----------
        data : HarvardForestData
            The inputs: the driver, the observation operator, the fixed SIPNET values and the
            prior table.
        """
        driver = ClimateDrivers.from_path(data.driver_path, time_zone="UTC").pandas
        medians = prior_medians(data.prior_table)
        operator = data.operator.tocoo()
        return cls(
            **_driver_series(driver),
            leaf_carbon_per_area=data.fixed_sipnet_parameters["leaf_carbon_per_area"],
            wood_carbon=medians["initial_wood_carbon"],
            respiration_q10=medians["wood_respiration_q10"],
            operator_rows=jnp.asarray(operator.row),
            operator_columns=jnp.asarray(operator.col),
            operator_weights=jnp.asarray(operator.data),
            n_features=operator.shape[0],
        )

    def features(self, theta: Mapping[str, Any]) -> jax.Array:
        r"""The predicted features $G(\theta) = A\, \mathrm{NEE}(\theta)$, shape $(n,)$."""
        rate = self.fluxes(theta)["nee"] * _RATE_CONVERSION
        contributions = self.operator_weights * rate[self.operator_columns]
        return jax.ops.segment_sum(contributions, self.operator_rows, num_segments=self.n_features)

    def fluxes(self, theta: Mapping[str, Any]) -> dict[str, jax.Array]:
        r"""Leaf carbon $C_L$ and the fluxes GPP, $R$ and NEE (g C m⁻² d⁻¹) at each step."""
        leaf_carbon = self.leaf_carbon(theta)
        capacity = _CAPACITY_CONVERSION * theta["photosynthetic_capacity"] * leaf_carbon
        gpp = (
            capacity
            * self.temperature_effect(theta)
            * self.vapor_pressure_deficit_effect(theta)
            * self.light_effect(theta, leaf_carbon / self.leaf_carbon_per_area)
        )
        respiration = self.respiration(theta, capacity)
        return {
            "leaf_carbon": leaf_carbon,
            "gpp": gpp,
            "respiration": respiration,
            "nee": respiration - gpp,
        }

    def leaf_carbon(self, theta: Mapping[str, Any]) -> jax.Array:
        r"""$C_L$: leaf-on growth times smooth leaf-on and leaf-off transitions."""
        leaf_on = jax.nn.sigmoid(
            (self.growing_degree_days - theta["leaf_on_growing_degree_days"]) / LEAF_ON_WIDTH
        )
        before_leaf_off = jax.nn.sigmoid(
            (theta["leaf_off_day"] - self.day_of_year) / LEAF_OFF_WIDTH
        )
        return theta["leaf_on_growth"] * leaf_on * before_leaf_off

    def temperature_effect(self, theta: Mapping[str, Any]) -> jax.Array:
        r"""$f_T(T_a)$, SIPNET's parabola between $T_{\mathrm{opt}} \mp \Delta$."""
        half_width = theta["photosynthesis_temperature_range"]
        distance = self.air_temperature - theta["optimum_photosynthesis_temperature"]
        return jnp.maximum(1.0 - (distance / half_width) ** 2, 0.0)

    def vapor_pressure_deficit_effect(self, theta: Mapping[str, Any]) -> jax.Array:
        r"""$f_D(D) = \max(0, 1 - \delta D^\nu)$."""
        reduction = (
            theta["vapor_pressure_deficit_slope"]
            * self.vapor_pressure_deficit ** theta["vapor_pressure_deficit_exponent"]
        )
        return jnp.maximum(1.0 - reduction, 0.0)

    def light_effect(self, theta: Mapping[str, Any], leaf_area_index: jax.Array) -> jax.Array:
        r"""$f_L$: the canopy's mean light saturation, by Simpson's rule on 6 layers."""
        attenuation = jnp.exp(
            -theta["light_extinction_coefficient"]
            * leaf_area_index[:, None]
            * jnp.asarray(_CANOPY_DEPTHS)[None, :]
        )
        light = self.radiation[:, None] * attenuation
        saturation = 1.0 - 2.0 ** (-light / theta["half_saturation_light"])
        mean_saturation = saturation @ jnp.asarray(_CANOPY_WEIGHTS)
        has_light = (leaf_area_index > 0) & (self.radiation > 0)
        return jnp.where(has_light, mean_saturation, 0.0)

    def respiration(self, theta: Mapping[str, Any], capacity: jax.Array) -> jax.Array:
        r"""$R$: foliar, wood and soil respiration, each with its $Q_{10}$ form."""
        foliar = (
            theta["respiration_share"]
            * capacity
            * self.respiration_q10
            ** ((self.air_temperature - theta["optimum_photosynthesis_temperature"]) / 10)
        )
        wood = (
            theta["wood_respiration_rate_at_10c"]
            * self.wood_carbon
            / 365
            * self.respiration_q10 ** ((self.air_temperature - 10) / 10)
        )
        soil = (
            theta["soil_respiration_flux_at_10c"]
            / 365
            * theta["soil_respiration_q10"] ** ((self.soil_temperature - 10) / 10)
        )
        return foliar + wood + soil


def _driver_series(driver: Any) -> dict[str, jax.Array]:
    """The model's driver series from pySIPNET's driver table, in the model's units."""
    step_length = driver["timestep_length"].to_numpy()
    air_temperature = driver["air_temperature"].to_numpy()
    return {
        "air_temperature": jnp.asarray(air_temperature),
        "soil_temperature": jnp.asarray(driver["soil_temperature"].to_numpy()),
        "radiation": jnp.asarray(
            driver["photosynthetically_active_radiation"].to_numpy() / step_length
        ),
        "vapor_pressure_deficit": jnp.asarray(
            np.maximum(driver["vapor_pressure_deficit"].to_numpy() * 1e-3, 1e-12)
        ),
        "day_of_year": jnp.asarray(
            driver["day_of_year"].to_numpy() + driver["hour_of_day"].to_numpy() / 24
        ),
        "growing_degree_days": jnp.asarray(
            _growing_degree_days(driver["year"].to_numpy(), air_temperature, step_length)
        ),
    }


def _growing_degree_days(
    year: np.ndarray, air_temperature: np.ndarray, step_length: np.ndarray
) -> np.ndarray:
    r"""$\Theta$: the year-to-date sum of $\max(T_a\, \Delta t, 0)$, as SIPNET accumulates it."""
    contribution = np.maximum(air_temperature * step_length, 0.0)
    total = np.zeros_like(contribution)
    for each_year in np.unique(year):
        in_year = year == each_year
        total[in_year] = np.cumsum(contribution[in_year])
    return total
