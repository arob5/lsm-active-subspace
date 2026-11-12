r"""Constrained, unconstrained and whitened coordinates of the calibration parameters.

For a block of $d$ scalar parameters $\theta$, such as a prior's ``theta``:

- **constrained** $\theta$, each parameter in its own units and support;
- **unconstrained** $\phi = b^{-1}(\theta) \in \mathbb R^d$, where $b_j$ is the bijector ProbPipe
  assigns to the support of $\theta_j$ (the exponential for a positive parameter, the logistic
  function for one in $(0, 1)$, the identity for a real one);
- **whitened** $z = L^{-1}(\phi - m)$, where $m$ and $L L^\top$ are the prior mean and covariance
  of $\phi$, so that $z$ has prior mean $0$ and covariance $I$.

:class:`Coordinates` gives the changes of coordinates as invertible ProbPipe ``Function``\ s,
:attr:`~Coordinates.unconstrain` ($\theta \mapsto \phi$) and :attr:`~Coordinates.whiten`
($\theta \mapsto z$). Each takes the record ``theta`` and returns a record with the same field
names, so it maps a record, a batch of draws or a distribution, as any ``Function`` does:

>>> coordinates = problem.coordinates                            # doctest: +SKIP
>>> z = coordinates.whiten(draws["theta"])  # whitened draws      # doctest: +SKIP
>>> z.to_vector()  # as an array of shape (n, d)                   # doctest: +SKIP
>>> coordinates.whiten(posterior["theta"])  # a distribution        # doctest: +SKIP
>>> inverse(coordinates.whiten)(z)  # back to the parameters      # doctest: +SKIP
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import (
    Distribution,
    Function,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    SupportsInverse,
    SupportsLogDetJacobian,
    bijector_for,
)
from probpipe import inverse as probpipe_inverse

from lsmas._stopgaps.sampling import draw_with_key

__all__ = ["CoordinateMap", "Coordinates", "inverse"]


class CoordinateMap(Function, SupportsInverse, SupportsLogDetJacobian):
    r"""An invertible change of coordinates of a record of $d$ scalars.

    It is a ProbPipe ``Function`` of one record, its input, that returns a record with the same
    field names. It claims ``SupportsInverse`` and ``SupportsLogDetJacobian``, as ProbPipe's
    bijectors do, and is JAX code, so it maps batches in one vectorized call and can be
    differentiated. :func:`inverse` gives the map back.

    Built by :class:`Coordinates`; see :attr:`Coordinates.unconstrain` and
    :attr:`Coordinates.whiten`.

    Parameters
    ----------
    forward, backward : callable
        The map and its inverse on arrays whose last axis holds the $d$ fields in order.
    log_det_jacobian : callable
        $\log \lvert \det J \rvert$ of *forward* at an array of shape $(d,)$.
    input_spec, output_spec : NumericRecordSpec
        The records the map takes and returns, with the same fields.
    input_name, output_name : str
        The name of the map's input, and the name its inverse gives its input.
    label : str
        The map's label.
    """

    def __init__(
        self,
        forward: Callable[[Any], Any],
        backward: Callable[[Any], Any],
        log_det_jacobian: Callable[[Any], Any],
        *,
        input_spec: NumericRecordSpec,
        output_spec: NumericRecordSpec,
        input_name: str,
        output_name: str,
        label: str,
    ) -> None:
        names = tuple(input_spec.fields)

        def apply(record: Any) -> dict[str, Any]:
            return _as_record(names, forward(_as_vector(names, record)))

        # ProbPipe binds a Function's input by its parameter's name.
        apply.__signature__ = inspect.Signature(
            [inspect.Parameter(input_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )

        super().__init__(
            apply,
            label=label,
            input_spec={input_name: input_spec},
            output_spec=output_spec,
            traceable=True,
        )
        for attribute, value in {
            "_forward": forward,
            "_backward": backward,
            "_log_det": log_det_jacobian,
            "_input_spec": input_spec,
            "_output_spec": output_spec,
            "_input_name": input_name,
            "_output_name": output_name,
            "_names": names,
        }.items():
            object.__setattr__(self, attribute, value)

    def _inverse(self, y: Any) -> dict[str, Any]:
        """The record the map sends to the record *y*."""
        return _as_record(self._names, self._backward(_as_vector(self._names, y)))

    def _log_det_jacobian(self, x: Any) -> Any:
        r"""$\log \lvert \det J \rvert$ of the map at the record *x*."""
        return self._log_det(_as_vector(self._names, x))

    def _inverted(self) -> CoordinateMap:
        """The inverse map, itself invertible with this map as its inverse."""
        return CoordinateMap(
            self._backward,
            self._forward,
            lambda y: -self._log_det(self._backward(y)),
            input_spec=self._output_spec,
            output_spec=self._input_spec,
            input_name=self._output_name,
            output_name=self._input_name,
            label=f"{self.label}⁻¹",
        )


# STOPGAP(PG-05): import inverse from probpipe once its inverse operation is implemented.
def inverse(f: Function) -> Function:
    r"""The inverse of the invertible map *f*, as ProbPipe's ``inverse`` will give it.

    Parameters
    ----------
    f : Function
        An invertible map, such as :attr:`Coordinates.whiten`.

    Returns
    -------
    Function
        The map $y \mapsto f^{-1}(y)$, itself invertible with *f* as its inverse.
    """
    if isinstance(f, CoordinateMap):
        return f._inverted()
    return probpipe_inverse(f)


@dataclass(frozen=True, eq=False)
class Coordinates:
    r"""The coordinates of one block of a prior's parameters, such as ``theta``.

    The public maps :attr:`unconstrain` and :attr:`whiten` act on ProbPipe records, batches and
    distributions. $m$ and $L$ are estimated from a fixed number of prior draws with a fixed seed,
    so a given prior always yields the same coordinates; they are computed when first needed.

    Use :meth:`from_prior` to build one.

    Attributes
    ----------
    block : str
        The prior's output that holds the parameters, such as ``"theta"``.
    spec : NumericRecordSpec
        The record of the parameters: their names, in order, and supports.
    """

    block: str
    spec: NumericRecordSpec = field(repr=False)
    _prior: Distribution = field(repr=False)
    _bijectors: tuple[Any, ...] = field(repr=False)
    _n_moment_draws: int = field(repr=False)
    _seed: int = field(repr=False)

    @classmethod
    def from_prior(
        cls,
        prior: Distribution,
        block: str = "theta",
        *,
        n_moment_draws: int = 50_000,
        seed: int = 0,
    ) -> Coordinates:
        r"""The coordinates of the block *block* of *prior*.

        Parameters
        ----------
        prior : Distribution
            A ProbPipe law whose event declares *block* as a record of scalar parameters.
        block : str
            The output of *prior* to describe.
        n_moment_draws : int
            How many prior draws estimate $m$ and $L$.
        seed : int
            The seed of those draws. The draws do not touch any surrounding ``workflow_run``.

        Raises
        ------
        ValueError
            If *block* is not an output of *prior* holding a record, or a parameter is not a scalar.
        """
        spec = _block_spec(prior, block)
        for name in spec.fields:
            if spec.at_path(name).shape != ():
                shape = spec.at_path(name).shape
                raise ValueError(f"only scalar parameters are supported; {name!r} has {shape}")
        bijectors = tuple(bijector_for(spec.at_path(name).support) for name in spec.fields)
        return cls(block, spec, prior, bijectors, n_moment_draws, seed)

    def __repr__(self) -> str:
        return f"Coordinates({self.block!r}: {self.dimension} parameters)"

    @property
    def names(self) -> tuple[str, ...]:
        """The parameters, in the order of :meth:`~probpipe.NumericRecord.to_vector`."""
        return tuple(self.spec.fields)

    @property
    def dimension(self) -> int:
        """The number of parameters, $d$."""
        return len(self.names)

    @functools.cached_property
    def unconstrain(self) -> CoordinateMap:
        r"""$\theta \mapsto \phi = b^{-1}(\theta)$, a ``Function`` of the record ``theta``."""
        return CoordinateMap(
            self._unconstrained,
            self._constrained,
            self._unconstrain_log_det,
            input_spec=self.spec,
            output_spec=_real_record_spec(self.names),
            input_name=self.block,
            output_name="phi",
            label="unconstrain",
        )

    @functools.cached_property
    def whiten(self) -> CoordinateMap:
        r"""$\theta \mapsto z = L^{-1}\left(b^{-1}(\theta) - m\right)$, a ``Function`` of the
        record ``theta``."""
        log_det_l = float(np.sum(np.log(np.diag(self.whitening_factor))))
        return CoordinateMap(
            self._whitened,
            self._from_whitened,
            lambda theta: self._unconstrain_log_det(theta) - log_det_l,
            input_spec=self.spec,
            output_spec=_real_record_spec(self.names),
            input_name=self.block,
            output_name="z",
            label="whiten",
        )

    @property
    def whitening_mean(self) -> np.ndarray:
        r"""$m$, the prior mean of $\phi$, shape $(d,)$."""
        return self._moments[0]

    @property
    def whitening_factor(self) -> np.ndarray:
        r"""$L$, the lower-triangular Cholesky factor of the prior covariance of $\phi$, shape
        $(d, d)$."""
        return self._moments[1]

    # The array kernels below are the one implementation of the maps. lsmas's algorithms call them
    # on arrays of shape (..., d), with the parameters in the order of `names`; they are JAX code,
    # so they can be traced and differentiated.

    def _unconstrained(self, theta: Any) -> jax.Array:
        r"""$\phi = b^{-1}(\theta)$ for an array *theta* of shape $(\ldots, d)$."""
        theta = jnp.asarray(theta)
        # STOPGAP(PG-05): a support bijector's inverse is reachable only through its _inverse.
        columns = [b._inverse(theta[..., j]) for j, b in enumerate(self._bijectors)]
        return jnp.stack(columns, axis=-1)

    def _constrained(self, phi: Any) -> jax.Array:
        r"""$\theta = b(\phi)$ for an array *phi* of shape $(\ldots, d)$."""
        phi = jnp.asarray(phi)
        return jnp.stack([b.raw(phi[..., j]) for j, b in enumerate(self._bijectors)], axis=-1)

    def _whitened(self, theta: Any) -> jax.Array:
        r"""$z = L^{-1}(b^{-1}(\theta) - m)$ for an array *theta* of shape $(\ldots, d)$."""
        mean, _, inverse_factor = self._moments
        return (self._unconstrained(theta) - mean) @ inverse_factor.T

    def _from_whitened(self, z: Any) -> jax.Array:
        r"""$\theta = b(m + L z)$ for an array *z* of shape $(\ldots, d)$."""
        mean, factor, _ = self._moments
        return self._constrained(mean + jnp.asarray(z) @ factor.T)

    def _records(self, theta: Any, level: str = "member") -> NumericRecordBatch:
        """Constrained parameters *theta*, shape $(n, d)$, as a batch of records ``theta``."""
        return NumericRecordBatch.from_vector(
            self.spec, jnp.asarray(theta, dtype=float), level_names=level
        )

    def _unconstrain_log_det(self, theta: Any) -> jax.Array:
        r"""$\log \lvert \det \partial \phi / \partial \theta \rvert
        = -\sum_j \log b_j'(\phi_j)$."""
        phi = self._unconstrained(theta)
        return -sum(b._log_det_jacobian(phi[..., j]) for j, b in enumerate(self._bijectors))

    @functools.cached_property
    def _moments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """$m$, $L$ and $L^{-1}$, from prior draws made with a fixed key.

        Computed eagerly even when first needed inside a JAX trace, such as a Jacobian of the
        forward map in whitened coordinates.
        """
        with jax.ensure_compile_time_eval():
            key = jax.random.key(self._seed)
            draws = draw_with_key(self._prior, key, (self._n_moment_draws,))
            vectors = _parameter_vectors(draws, self.block, self.names)
            phi = np.asarray(self._unconstrained(vectors))
        factor = np.linalg.cholesky(np.atleast_2d(np.cov(phi, rowvar=False)))
        return phi.mean(axis=0), factor, np.linalg.inv(factor)


def _parameter_vectors(draws: Any, block: str, names: tuple[str, ...]) -> np.ndarray:
    """The values of the parameters *names* of *block* in raw or tracked *draws*, shape $(n, d)$."""
    raw = draws.raw if hasattr(draws, "raw") else draws
    # STOPGAP(PG-13): a prior taken back out of a joint may draw its block's fields ungrouped.
    block_draws = raw.get(block, raw)
    return np.stack([np.asarray(block_draws[name], dtype=float) for name in names], axis=-1)


def _as_vector(names: tuple[str, ...], record: Any) -> jax.Array:
    """The fields *names* of a record, stacked on a last axis."""
    return jnp.stack([jnp.asarray(record[name]) for name in names], axis=-1)


def _as_record(names: tuple[str, ...], vector: Any) -> dict[str, jax.Array]:
    """The last axis of *vector* as a record with the fields *names*."""
    return {name: vector[..., j] for j, name in enumerate(names)}


def _real_record_spec(names: tuple[str, ...]) -> NumericRecordSpec:
    """A record of real scalars with the fields *names*."""
    return NumericRecordSpec(**{name: NumericArraySpec(()) for name in names})


def _block_spec(prior: Distribution, block: str) -> NumericRecordSpec:
    """The record spec of *block* in *prior*'s event declaration."""
    outputs = prior.event_spec.outputs
    if block not in outputs or not hasattr(outputs[block], "fields"):
        raise ValueError(f"the prior has no block of named parameters called {block!r}")
    return outputs[block]
