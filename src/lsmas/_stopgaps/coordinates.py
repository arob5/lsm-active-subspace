r"""Coordinates for the calibration parameters: constrained, unconstrained and whitened.

ProbPipe knows how to move a prior's draws to unconstrained coordinates (each family's support has
a bijector), but it does not expose that map publicly, and it cannot take the map for one block of a
prior, such as ``prior["theta"]``, on its own. :class:`Coordinates` fills that gap for a block of
scalar parameters. Once ProbPipe offers public, view-aware coordinates,
:meth:`Coordinates.from_prior` can be built on them and the rest of the class stays as it is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import NumericRecordBatch
from scipy.linalg import solve_triangular

from lsmas._stopgaps.probpipe_private import flat_unflatten, unconstrained_coordinates

__all__ = ["Coordinates"]


# STOPGAP(PG-05): ProbPipe's coordinate maps are private and refuse a block of a prior.
@dataclass(frozen=True, eq=False)
class Coordinates:
    r"""Maps between three descriptions of one block of a prior's parameters.

    For a block of $d$ scalar parameters $\theta$:

    - **constrained** $\theta \in \Theta$, in the parameters' own units and supports;
    - **unconstrained** $z = b^{-1}(\theta) \in \mathbb{R}^d$, where $b$ applies each parameter's
      support bijector (the exponential for a positive parameter, the logistic function for one in
      $(0, 1)$, the identity for a real one);
    - **whitened** $\xi = L^{-1}(z - m)$, where $m$ and $L L^\top$ are the prior mean and
      covariance of $z$, so that $\xi$ has prior mean $0$ and covariance $I$.

    Whitened coordinates put every parameter on the same footing whatever its units, which is
    what subspace methods need. $m$ and $L$ are estimated from a fixed number of prior draws with a
    fixed seed, so a given prior always yields the same coordinates.

    Every map acts on the last axis, so arrays of shape $(\ldots, d)$ map row by row.

    Use :meth:`from_prior` to build one.

    Attributes
    ----------
    names : tuple of str
        The parameters, in the order of the last axis.
    block : str
        The prior component that holds them, such as ``"theta"``.
    whitening_mean : numpy.ndarray
        $m$, shape $(d,)$.
    whitening_factor : numpy.ndarray
        $L$, the lower-triangular Cholesky factor of the prior covariance of $z$, shape $(d, d)$.
    """

    names: tuple[str, ...]
    block: str
    whitening_mean: np.ndarray
    whitening_factor: np.ndarray
    _block_bijection: _BlockBijection

    @classmethod
    def from_prior(
        cls, prior: Any, block: str = "theta", *, n_moment_draws: int = 50_000, seed: int = 0
    ) -> Coordinates:
        r"""Build the coordinates of one block of *prior*.

        Parameters
        ----------
        prior : Distribution
            A ProbPipe law whose draws hold *block* as a record of scalar parameters.
        block : str
            The component of *prior* to describe.
        n_moment_draws : int
            How many prior draws estimate $m$ and $L$.
        seed : int
            The seed of those draws. The draws do not touch any surrounding ``workflow_run``.

        Returns
        -------
        Coordinates

        Raises
        ------
        ValueError
            If *block* is not a component of *prior*, or a parameter in it is not a scalar.
        """
        bijection = _BlockBijection.from_prior(prior, block)
        draws = prior._sample(jax.random.key(seed), (n_moment_draws,))
        unconstrained = bijection.to_unconstrained(_block_array(draws, block, bijection.names))
        covariance = np.atleast_2d(np.cov(unconstrained, rowvar=False))
        return cls(
            names=bijection.names,
            block=block,
            whitening_mean=unconstrained.mean(axis=0),
            whitening_factor=np.linalg.cholesky(covariance),
            _block_bijection=bijection,
        )

    @property
    def dimension(self) -> int:
        """The number of parameters, $d$."""
        return len(self.names)

    def to_unconstrained(self, theta: Any) -> np.ndarray:
        r"""$z = b^{-1}(\theta)$ for constrained parameters *theta*, shape $(\ldots, d)$."""
        return self._block_bijection.to_unconstrained(theta)

    def to_constrained(self, z: Any) -> np.ndarray:
        r"""$\theta = b(z)$ for unconstrained coordinates *z*, shape $(\ldots, d)$."""
        return self._block_bijection.to_constrained(z)

    def whiten(self, z: Any) -> np.ndarray:
        r"""$\xi = L^{-1}(z - m)$ for unconstrained coordinates *z*, shape $(\ldots, d)$."""
        centered = np.asarray(z, dtype=float) - self.whitening_mean
        rows = centered.reshape(-1, self.dimension)
        whitened = solve_triangular(self.whitening_factor, rows.T, lower=True).T
        return whitened.reshape(centered.shape)

    def unwhiten(self, xi: Any) -> np.ndarray:
        r"""$z = m + L \xi$ for whitened coordinates *xi*, shape $(\ldots, d)$."""
        return self.whitening_mean + np.asarray(xi, dtype=float) @ self.whitening_factor.T

    def array_from_draws(self, draws: Any) -> np.ndarray:
        """The block's values in *draws*, as an array of shape $(n, d)$ in :attr:`names` order.

        Parameters
        ----------
        draws : RecordBatch or Mapping
            Draws of the prior (or of a posterior over the same parameters), tracked or raw.
        """
        raw = draws.raw() if hasattr(draws, "raw") else draws
        return _block_array(raw, self.block, self.names)

    def record_batch(self, theta: Any, level: str = "member") -> NumericRecordBatch:
        """Constrained parameters *theta*, shape $(n, d)$, as a ProbPipe batch of records.

        Calling a ProbPipe ``Function`` of the block on the result evaluates it once per row.

        Parameters
        ----------
        theta : array_like
            Constrained parameters, one row per member.
        level : str
            The name of the batch's level.
        """
        theta = np.asarray(theta, dtype=float)
        fields = {name: jnp.asarray(theta[:, index]) for index, name in enumerate(self.names)}
        return NumericRecordBatch(fields, level)


@dataclass(frozen=True, eq=False)
class _BlockBijection:
    """The support bijectors of one block's parameters, cut out of the whole prior's.

    ProbPipe's unconstrained map acts on the whole prior's flat vector, one leaf at a time. With
    scalar leaves, a flat slot and an unconstrained slot coincide, so the block's map is the whole
    map applied with the other slots held at a reference point.
    """

    names: tuple[str, ...]
    block_slots: np.ndarray
    reference_unconstrained: np.ndarray
    to_constrained_flat: Any
    to_unconstrained_flat: Any

    @classmethod
    def from_prior(cls, prior: Any, block: str) -> _BlockBijection:
        maps = unconstrained_coordinates(prior)
        slot_of = _flat_slots(prior, maps.size)
        if block not in slot_of or not isinstance(slot_of[block], Mapping):
            raise ValueError(f"the prior has no block of named parameters called {block!r}")
        names = tuple(slot_of[block])
        return cls(
            names=names,
            block_slots=np.array([slot_of[block][name] for name in names]),
            reference_unconstrained=np.zeros(maps.size),
            to_constrained_flat=jax.vmap(maps.forward),
            to_unconstrained_flat=jax.vmap(maps.inverse),
        )

    def to_unconstrained(self, theta: Any) -> np.ndarray:
        flat = self._embed(np.asarray(theta, dtype=float), self._reference_constrained())
        return self._apply_rowwise(self.to_unconstrained_flat, flat)

    def to_constrained(self, z: Any) -> np.ndarray:
        flat = self._embed(np.asarray(z, dtype=float), self.reference_unconstrained)
        return self._apply_rowwise(self.to_constrained_flat, flat)

    def _reference_constrained(self) -> np.ndarray:
        return np.asarray(self.to_constrained_flat(self.reference_unconstrained[None, :]))[0]

    def _embed(self, block_values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Full flat vectors equal to *reference* except in the block's slots."""
        rows = block_values.reshape(-1, len(self.names))
        flat = np.tile(reference, (rows.shape[0], 1))
        flat[:, self.block_slots] = rows
        return flat.reshape(*block_values.shape[:-1], reference.size)

    def _apply_rowwise(self, flat_map: Any, flat: np.ndarray) -> np.ndarray:
        """*flat_map* on each row of *flat*, keeping only the block's slots."""
        rows = flat.reshape(-1, flat.shape[-1])
        mapped = np.asarray(flat_map(jnp.asarray(rows)))[:, self.block_slots]
        return mapped.reshape(*flat.shape[:-1], len(self.names))


def _flat_slots(prior: Any, size: int) -> dict[str, Any]:
    """Each leaf's slot in *prior*'s flat vector, as a nested mapping like the prior's draws.

    Unflattening the vector ``0, 1, …, size - 1`` puts each slot's index where its leaf goes.
    """
    probe = flat_unflatten(prior)(jnp.arange(size, dtype=jnp.float64))
    nested = probe.to_nested_dict() if hasattr(probe, "to_nested_dict") else probe
    return _integer_leaves(nested)


def _integer_leaves(tree: Any) -> Any:
    """*tree* with each leaf converted to a Python int, refusing leaves that are not scalars."""
    if isinstance(tree, Mapping):
        return {key: _integer_leaves(value) for key, value in tree.items()}
    leaf = np.asarray(tree)
    if leaf.shape != ():
        raise ValueError(f"only scalar parameters are supported; found one with shape {leaf.shape}")
    return int(leaf)


def _block_array(draws: Mapping[str, Any], block: str, names: tuple[str, ...]) -> np.ndarray:
    """The values of *names* under *block* in raw *draws*, stacked into shape $(n, d)$."""
    values = draws[block]
    return np.stack([np.asarray(values[name], dtype=float) for name in names], axis=-1)
