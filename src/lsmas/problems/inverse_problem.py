r"""The inverse-problem class: prior, likelihood, observed data and learning rates."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from probpipe import Distribution, Function

from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.tempering import tempered_condition_on

__all__ = ["InverseProblem"]


@dataclass(frozen=True, eq=False)
class InverseProblem:
    r"""An inverse problem: the prior, the likelihood and the observed data.

    The model is $p(y \mid \theta, s^2)\, \pi_0(\theta, s^2)$, a ProbPipe ``likelihood & prior``,
    and conditioning it on the observed $y$ with learning rates $\eta$ gives the generalized
    posterior

    $$
    \pi_\eta(\theta, s^2 \mid y) \propto
    \pi_0(\theta, s^2) \prod_i p(y_i \mid \theta, s^2)^{\eta_i}.
    $$

    With every $\eta_i = 1$ this is the ordinary posterior.

    Attributes
    ----------
    name : str
        A short identifier, such as ``"harvard_forest_small"``.
    description : str
        A paragraph saying what the problem is.
    likelihood : ConditionalDistribution
        $p(y \mid \theta, s^2)$, the law of the observations given the parameters.
    prior : Distribution
        $\pi_0(\theta, s^2)$, the law of the parameters, with the calibration parameters grouped
        under ``theta``.
    data : dict of str to numpy.ndarray
        The observed value of each observed output.
    learning_rate : dict of str to numpy.ndarray
        $\eta$ for each observed output, one rate per element.
    coordinates : Coordinates
        Constrained, unconstrained and whitened coordinates of ``theta``.
    """

    name: str
    description: str
    likelihood: Any
    prior: Distribution
    data: dict[str, np.ndarray]
    learning_rate: dict[str, np.ndarray]
    coordinates: Coordinates

    @property
    def forward(self) -> Function:
        """The forward map $G$ from the parameter record ``theta`` to the predicted observations."""
        return self.likelihood.forward

    def predicted_observations(self, theta: Any) -> np.ndarray:
        r"""$G(\theta)$ at each row of an array of constrained parameters.

        This is :attr:`forward` for parameters held as an array, with columns in the order of
        ``coordinates.names``. The rows are evaluated as one batch, so a forward map that runs
        SIPNET runs them in parallel. A row at which the forward map fails gives a row of NaN.

        Parameters
        ----------
        theta : array_like
            Constrained parameters, shape $(N, d)$, or one setting of shape $(d,)$.

        Returns
        -------
        numpy.ndarray
            The predicted observations, shape $(N, n)$, or $(n,)$ for one setting.

        Raises
        ------
        ValueError
            If *theta* does not have $d$ columns.
        """
        theta = np.asarray(theta, dtype=float)
        rows = np.atleast_2d(theta)
        if rows.ndim != 2 or rows.shape[1] != self.coordinates.dimension:
            raise ValueError(
                f"expected parameters of shape (N, {self.coordinates.dimension}) or "
                f"({self.coordinates.dimension},); got {theta.shape}"
            )
        predictions = np.asarray(self.forward(self.coordinates.bijection.record_batch(rows)).raw)
        return predictions[0] if theta.ndim == 1 else predictions

    @functools.cached_property
    def model(self) -> Distribution:
        """The generative model ``likelihood & prior``.

        Its draws are prior predictive draws: ``sample(problem.model, n)`` draws the parameters
        from the prior and then the observations from the likelihood, running the forward map
        once per draw.
        """
        return (self.likelihood & self.prior).with_label(self.name)

    # STOPGAP(PG-08): automatic method selection assumes gradients, so the method is required.
    def condition(self, *, method: str, **using: Any) -> Distribution:
        """The generalized posterior, computed by the inference method *method* and labelled
        ``"posterior"``.

        Parameters
        ----------
        method : str
            The inference method, such as ``"enskit_eki"`` or ``"blackjax_rwmh"``. It is required
            because ProbPipe's automatic choice assumes that a density it can trace has gradients,
            and a forward map that runs SIPNET has none.
        **using
            Further controls of ``condition_on``, such as ``method_options``.
        """
        posterior = tempered_condition_on(
            self.model, self.data, learning_rate=self.learning_rate, method=method, **using
        )
        return posterior.with_label("posterior")

    def noise_variance(self, scale_values: Mapping[str, Any] | None = None) -> np.ndarray:
        r"""The diagonal of the tempered noise covariance,
        $\Gamma = \operatorname{diag}\!\left(s^2_{k(i)}\, \sigma^2_i / \eta_i\right)$, shape $(n,)$.

        $\Gamma^{-1}$ weights the observations in the generalized posterior and in the
        information matrices of :mod:`lsmas.tools`.

        Parameters
        ----------
        scale_values : Mapping of str to float, optional
            A value for each inferred noise scale, such as ``{"s2_day": 1.0, "s2_night": 1.0}``.
            Not needed when every scale is fixed.

        Raises
        ------
        ValueError
            If an inferred scale has no value, or the problem has more than one observed output.
        """
        if missing := sorted(set(self.likelihood.random_scale_slots) - set(scale_values or {})):
            raise ValueError(f"the noise scales are inferred; give values for {missing}")
        if len(self.data) != 1:
            raise ValueError(f"defined for one observed output; this problem has {list(self.data)}")
        (observed,) = self.data
        variances = np.asarray(self.likelihood.noise_variances(scale_values))
        return variances / self.learning_rate[observed]

    def describe(self) -> str:
        """A short readable summary: the description, the parameters and the data."""
        observed = ", ".join(f"{name} ({len(value)} values)" for name, value in self.data.items())
        return "\n".join(
            [
                f"{self.name}: {self.description}",
                f"Calibration parameters ({self.coordinates.dimension}): "
                + ", ".join(self.coordinates.names),
                f"Observations: {observed}",
            ]
        )
