r"""Reduced problems: the posterior of the active variables, with the inactive ones at their prior.

Given a subspace with bases $W$ and $W_\perp$, whitened coordinates split as
$z = W a + W_\perp b$. A reduced posterior keeps the inactive variables $b$ at their prior and
approximates the law of the active variables $a$. The **plug-in** reduced posterior evaluates the
likelihood with $b$ at its prior mean $0$:

$$
\pi_r(a \mid y) \propto \mathcal N(a;\, 0, I_r)\;
p\!\left(y \mid \theta(W a), s^2\right).
$$

Its forward map $a \mapsto G(\theta(Wa))$ is deterministic, so the reduced problem is an ordinary
inverse problem in $r$ parameters that any inference method can condition. The approximation of
the full posterior then combines its draws of $a$ with draws of $b$ from the prior:
$\theta = \theta(W a + W_\perp b)$ (:meth:`ReducedProblem.lift`).

The prior $\mathcal N(0, I_r)$ of $a$ is exact when the whitened prior is Gaussian, as it is for
log-normal, logit-normal and normal priors, and an approximation otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericRecordBatch,
)

from lsmas._stopgaps.functions import declares_traceable
from lsmas._stopgaps.likelihood import NOISE_SCALES, GaussianSimulatorLikelihood
from lsmas._stopgaps.sampling import draw_with_key
from lsmas.coordinates import Coordinates, _parameter_vectors
from lsmas.parameters import parameter_record_spec
from lsmas.priors import factors_over, independent_prior
from lsmas.problems.inverse_problem import InverseProblem
from lsmas.tools.subspace import Subspace

__all__ = ["ReducedProblem", "reduced_problem"]

#: The prior's output that holds the parameters, in the full and in the reduced problem.
_PARAMETER_BLOCK = "theta"


def reduced_problem(problem: InverseProblem, subspace: Subspace) -> ReducedProblem:
    r"""The plug-in reduced problem of *problem* in the active variables of *subspace*.

    Its parameters are the active variables ``active_1``, ..., ``active_r``, each with a
    $\mathcal N(0, 1)$ prior; its forward map is $a \mapsto G(\theta(W a))$; its likelihood, data,
    learning rate and noise scales (fixed or inferred, with the same priors) are *problem*'s.

    Parameters
    ----------
    problem : InverseProblem
        The full problem.
    subspace : Subspace
        A subspace of *problem*'s whitened coordinates.

    Examples
    --------
    >>> reduced = reduced_problem(problem, subspace)                      # doctest: +SKIP
    >>> with workflow_run(seed=1):                                          # doctest: +SKIP
    ...     posterior = reduced.condition(method="blackjax_rwmh")
    >>> approximation = reduced.lift(posterior, key=jax.random.key(2))     # doctest: +SKIP
    """
    if subspace.parameter_names != problem.coordinates.names:
        raise ValueError("the subspace's parameters are not the problem's")
    names = tuple(f"active_{i + 1}" for i in range(subspace.rank))
    likelihood = _reduced_likelihood(problem, subspace, names)
    prior = independent_prior({name: Normal(name, 0.0, 1.0) for name in names})
    for scale_prior in factors_over(problem.prior, [NOISE_SCALES]).values():
        prior = prior & scale_prior
    return ReducedProblem(
        name=f"{problem.name}_reduced",
        description=(
            f"{problem.name} reduced to {subspace.rank} active variables, the inactive ones at "
            "their prior mean in the likelihood."
        ),
        likelihood=likelihood,
        prior=prior,
        data=problem.data,
        learning_rate=problem.learning_rate,
        coordinates=Coordinates.from_prior(prior, _PARAMETER_BLOCK),
        full_problem=problem,
        subspace=subspace,
    )


@dataclass(frozen=True, eq=False)
class ReducedProblem(InverseProblem):
    """A reduced problem, with the full problem and the subspace it was built from.

    Attributes
    ----------
    full_problem : InverseProblem
        The problem it reduces.
    subspace : Subspace
        The subspace whose active variables are its parameters.
    """

    full_problem: InverseProblem
    subspace: Subspace

    def lift(self, posterior: Any, *, key: Any) -> EmpiricalDistribution:
        r"""The approximation of the full posterior from draws of this problem's posterior.

        Each draw of the active variables $a$ is combined with the inactive part
        $b = W_\perp^\top z$ of an independent prior draw $z$, and mapped to the calibration
        parameters $\theta(W a + W_\perp b)$. Drawn noise scales are carried over unchanged.

        Parameters
        ----------
        posterior : Distribution
            A posterior of this problem with draws (``posterior.atoms``), such as the result of
            :meth:`condition`.
        key : jax.Array
            The random key of the prior draws.

        Returns
        -------
        EmpiricalDistribution
            Draws of the full problem's parameter record ``theta`` and of any inferred noise scales.
        """
        atoms = posterior.atoms
        active = _parameter_vectors(atoms, _PARAMETER_BLOCK, self.coordinates.names).reshape(
            -1, self.subspace.rank
        )
        full = self.full_problem.coordinates
        prior_draws = draw_with_key(self.full_problem.prior, key, (active.shape[0],))
        prior_z = np.asarray(
            full._whitened(_parameter_vectors(prior_draws, _PARAMETER_BLOCK, full.names))
        )
        z = self.subspace.whitened_from_active_and_inactive(
            active, self.subspace.inactive_variables(prior_z)
        )
        theta = np.asarray(full._from_whitened(z))
        fields: dict[str, Any] = {
            _PARAMETER_BLOCK: {
                name: jnp.asarray(theta[:, index]) for index, name in enumerate(full.names)
            }
        }
        if scales := self.likelihood.inferred_scale_draws(atoms):
            fields[NOISE_SCALES] = {
                noise_class: jnp.asarray(draws.reshape(-1)) for noise_class, draws in scales.items()
            }
        return EmpiricalDistribution(NumericRecordBatch(fields, "draw"))


@dataclass(frozen=True, eq=False)
class _ReducedForwardMap:
    r"""Evaluates $G(\theta(W a))$ at one record of active variables."""

    problem: InverseProblem
    basis: jax.Array
    names: tuple[str, ...]

    def features(self, theta: Any) -> Any:
        """$G(W a)$ at the record *theta* of active variables $a$."""
        coordinates = self.problem.coordinates
        z = self.basis @ jnp.stack([jnp.asarray(theta[name]) for name in self.names])
        parameters = coordinates._from_whitened(z)
        record = {name: parameters[j] for j, name in enumerate(coordinates.names)}
        return self.problem.forward.using(raw=True)(record)


def _reduced_likelihood(
    problem: InverseProblem, subspace: Subspace, names: tuple[str, ...]
) -> GaussianSimulatorLikelihood:
    """The full problem's likelihood around the reduced forward map."""
    full_likelihood = problem.likelihood
    evaluator = _ReducedForwardMap(problem, jnp.asarray(subspace.basis), names)
    traceable = declares_traceable(problem.forward)
    forward = Function(
        evaluator.features,
        label=f"{problem.forward.label}_reduced",
        input_spec={"theta": parameter_record_spec(names)},
        output_spec=problem.forward.output_spec,
        traceable=traceable,
        **({} if traceable else {"executor": "thread"}),
    )
    return GaussianSimulatorLikelihood(
        forward,
        output_name=full_likelihood.output_name,
        reference_variance=full_likelihood.reference_variance,
        classes=full_likelihood.classes,
        scales=full_likelihood.scales,
        label=f"{full_likelihood.label}_reduced",
    )
