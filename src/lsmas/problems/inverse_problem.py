r"""The inverse-problem class: a model, observed data, and the learning rates of conditioning."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
from probpipe import Distribution, Function, factor, marginal

from lsmas._stopgaps.likelihood import NOISE_SCALES
from lsmas._stopgaps.tempering import tempered_condition_on
from lsmas.coordinates import Coordinates

__all__ = ["InverseProblem"]


@dataclass(frozen=True, eq=False)
class InverseProblem:
    r"""An inverse problem: a model, the observed data, and the learning rates of conditioning.

    The model is the joint law $p(y \mid \theta, s^2)\, \pi_0(\theta, s^2)$ of the parameters and
    the observations, a ProbPipe ``likelihood & prior``. Conditioning it on the observed $y$ with
    learning rates $\eta$ gives the generalized posterior

    $$
    \pi_\eta(\theta, s^2 \mid y) \propto
    \pi_0(\theta, s^2) \prod_i p(y_i \mid \theta, s^2)^{\eta_i}.
    $$

    With every $\eta_i = 1$ this is the ordinary posterior. The model is the law; the data and
    the learning rates are what :meth:`condition` conditions it with. The noise scales $s^2$ are
    parameters when the prior has the output ``s2``, and fixed otherwise.

    Attributes
    ----------
    name : str
        A short identifier, such as ``"harvard_forest_small"``.
    description : str
        A paragraph saying what the problem is.
    model : Distribution
        The joint law ``likelihood & prior``.
    data : dict of str to numpy.ndarray
        The observed value of each observed output.
    learning_rate : dict of str to numpy.ndarray
        $\eta$ for each observed output, one rate per element.
    """

    name: str
    description: str
    model: Distribution
    data: dict[str, np.ndarray]
    learning_rate: dict[str, np.ndarray]

    @functools.cached_property
    def prior(self) -> Distribution:
        r"""$\pi_0$: the model's law of its unobserved outputs, the calibration parameters
        ``theta`` and any inferred noise scales ``s2``. It draws without running the forward
        map."""
        unobserved = [name for name in self.model.event_spec.outputs if name not in self.data]
        return marginal(self.model, unobserved).with_label("prior")

    @functools.cached_property
    def likelihood(self) -> Any:
        r"""$p(y \mid \theta, s^2)$: the model's factor that produces the observed output."""
        if len(self.data) != 1:
            raise ValueError(f"defined for one observed output; this problem has {list(self.data)}")
        (observed,) = self.data
        return factor(self.model, observed)

    @property
    def forward(self) -> Function:
        """The forward map $G$ from the parameter record ``theta`` to the predicted observations."""
        return self.likelihood.forward

    @property
    def inferred_noise_classes(self) -> tuple[str, ...]:
        """The classes of observation whose noise scales are parameters: every class when the
        prior has the output ``s2``, and none when the scales are fixed."""
        inferred = NOISE_SCALES in self.prior.event_spec.outputs
        return self.likelihood.noise_classes if inferred else ()

    @property
    def coordinates(self) -> Coordinates:
        """The changes of coordinates of ``theta``: ``Coordinates.from_model(model)``."""
        return Coordinates.from_model(self.model)

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
