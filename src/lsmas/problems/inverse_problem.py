r"""A Bayesian inverse problem, ready to condition: model, data, learning rate and coordinates."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
from probpipe import Distribution, Function

from lsmas._stopgaps.coordinates import Coordinates
from lsmas._stopgaps.tempering import tempered_condition_on

__all__ = ["InverseProblem"]


@dataclass(frozen=True, eq=False)
class InverseProblem:
    r"""A fixed inverse problem: a generative model, observed data, and how much the data count.

    The model is $p(y \mid \theta, s^2)\, \pi(\theta, s^2)$, a ProbPipe ``likelihood * prior``, and
    conditioning it on the observed $y$ with learning rates $\eta$ gives the generalized posterior

    $$
    \pi_\eta(\theta, s^2 \mid y) \propto \pi(\theta, s^2) \prod_i p(y_i \mid \theta, s^2)^{\eta_i}.
    $$

    With every $\eta_i = 1$ this is the ordinary posterior.

    Attributes
    ----------
    name : str
        A short identifier, such as ``"harvard_forest_dev"``.
    description : str
        A paragraph saying what the problem is.
    likelihood : ConditionalDistribution
        $p(y \mid \theta, s^2)$, the law of the observations given the parameters.
    prior : Distribution
        $\pi(\theta, s^2)$, the law of the parameters, with the calibration parameters grouped
        under ``theta``.
    data : dict of str to numpy.ndarray
        The observed value of each observed component.
    learning_rate : dict of str to numpy.ndarray
        $\eta$ for each observed component, one rate per element.
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

    @functools.cached_property
    def model(self) -> Distribution:
        """The generative model ``likelihood * prior``."""
        return (self.likelihood * self.prior).with_label(self.name)

    def condition(self, **using: Any) -> Distribution:
        """The generalized posterior, computed as ``condition_on.using(**using)`` computes it.

        Parameters
        ----------
        **using
            Controls of ``condition_on``, such as ``method`` and ``method_options``.
        """
        return tempered_condition_on(
            self.model, self.data, learning_rate=self.learning_rate, **using
        )

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
