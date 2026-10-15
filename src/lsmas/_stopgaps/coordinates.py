r"""Coordinates for the calibration parameters: constrained, unconstrained and whitened.

ProbPipe attaches a support to every parameter and a bijector to every support, but it has no
public map between a prior's draws and unconstrained coordinates, and its internal map refuses one
block of a prior, such as ``prior["theta"]``. :class:`ParameterBijection` builds that map for a
block of scalar parameters from their declared supports, and :class:`Coordinates` adds whitening.
Once ProbPipe offers public, block-aware coordinates, :class:`ParameterBijection` can be replaced by
them; the whitening stays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from probpipe import NumericRecordBatch, bijector_for
from scipy.linalg import solve_triangular

__all__ = ["Coordinates", "ParameterBijection", "block_values"]


# STOPGAP(PG-05): ProbPipe has no public map to unconstrained coordinates for a block of a prior.
@dataclass(frozen=True, eq=False)
class ParameterBijection:
    r"""The map between a block of scalar parameters and unconstrained coordinates.

    For parameters $\theta = (\theta_1, \dots, \theta_d)$ with supports $\mathcal S_j$, the map is
    $z_j = b_j^{-1}(\theta_j)$, where $b_j : \mathbb{R} \to \mathcal S_j$ is the bijector ProbPipe
    assigns to $\mathcal S_j$ (the exponential for a positive parameter, the logistic function for
    one in $(0, 1)$, the identity for a real one). Maps act on the last axis of an array of shape
    $(\ldots, d)$.

    Attributes
    ----------
    names : tuple of str
        The parameters, in the order of the last axis.
    """

    names: tuple[str, ...]
    _bijectors: tuple[Any, ...]

    @classmethod
    def from_prior(cls, prior: Any, block: str = "theta") -> ParameterBijection:
        """The bijection of the parameters a prior declares under *block*.

        Parameters
        ----------
        prior : Distribution
            A ProbPipe law whose event declares *block* as a record of scalar parameters.
        block : str
            The component holding the parameters.

        Raises
        ------
        ValueError
            If *block* is not a record component of *prior*, or a parameter is not a scalar.
        """
        record_spec = _block_spec(prior, block)
        names = tuple(record_spec.fields)
        leaves = [record_spec.at_path(name) for name in names]
        for name, leaf in zip(names, leaves, strict=True):
            if leaf.shape != ():
                raise ValueError(f"only scalar parameters are supported; {name!r} has {leaf.shape}")
        return cls(names=names, _bijectors=tuple(bijector_for(leaf.support) for leaf in leaves))

    @property
    def dimension(self) -> int:
        """The number of parameters, $d$."""
        return len(self.names)

    def to_unconstrained(self, theta: Any) -> np.ndarray:
        r"""$z = b^{-1}(\theta)$ for constrained parameters *theta*, shape $(\ldots, d)$."""
        return self._apply_columnwise(theta, [bijector._inverse for bijector in self._bijectors])

    def to_constrained(self, z: Any) -> np.ndarray:
        r"""$\theta = b(z)$ for unconstrained coordinates *z*, shape $(\ldots, d)$."""
        return self._apply_columnwise(z, [bijector.raw() for bijector in self._bijectors])

    def record_batch(self, theta: Any, level: str = "member") -> NumericRecordBatch:
        """Constrained parameters *theta*, shape $(n, d)$, as a ProbPipe batch of records.

        Calling a ProbPipe ``Function`` of the parameter record on the result evaluates it once
        per row.

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

    def _apply_columnwise(self, values: Any, maps: list[Any]) -> np.ndarray:
        """Each column of *values* through its own map."""
        values = np.asarray(values, dtype=float)
        columns = [np.asarray(apply(jnp.asarray(values[..., j]))) for j, apply in enumerate(maps)]
        return np.stack(columns, axis=-1)


@dataclass(frozen=True, eq=False)
class Coordinates:
    r"""Constrained, unconstrained and whitened coordinates of a block of a prior's parameters.

    For a block of $d$ scalar parameters $\theta$:

    - **constrained** $\theta$, in the parameters' own units and supports;
    - **unconstrained** $z = b^{-1}(\theta) \in \mathbb{R}^d$ (see :class:`ParameterBijection`);
    - **whitened** $\xi = L^{-1}(z - m)$, where $m$ and $L L^\top$ are the prior mean and
      covariance of $z$, so that $\xi$ has prior mean $0$ and covariance $I$.

    Whitened coordinates put every parameter on the same footing whatever its units, which is
    what subspace methods need. $m$ and $L$ are estimated from a fixed number of prior draws with a
    fixed seed, so a given prior always yields the same coordinates. Every map acts on the last
    axis, so arrays of shape $(\ldots, d)$ map row by row.

    Use :meth:`from_prior` to build one.

    Attributes
    ----------
    block : str
        The prior component that holds the parameters, such as ``"theta"``.
    bijection : ParameterBijection
        The map between constrained and unconstrained coordinates.
    whitening_mean : numpy.ndarray
        $m$, shape $(d,)$.
    whitening_factor : numpy.ndarray
        $L$, the lower-triangular Cholesky factor of the prior covariance of $z$, shape $(d, d)$.
    """

    block: str
    bijection: ParameterBijection
    whitening_mean: np.ndarray
    whitening_factor: np.ndarray

    @classmethod
    def from_prior(
        cls, prior: Any, block: str = "theta", *, n_moment_draws: int = 50_000, seed: int = 0
    ) -> Coordinates:
        r"""Build the coordinates of one block of *prior*.

        Parameters
        ----------
        prior : Distribution
            A ProbPipe law whose event declares *block* as a record of scalar parameters.
        block : str
            The component of *prior* to describe.
        n_moment_draws : int
            How many prior draws estimate $m$ and $L$.
        seed : int
            The seed of those draws. The draws do not touch any surrounding ``workflow_run``.
        """
        bijection = ParameterBijection.from_prior(prior, block)
        draws = prior._sample(jax.random.key(seed), (n_moment_draws,))
        unconstrained = bijection.to_unconstrained(block_values(draws, block, bijection.names))
        covariance = np.atleast_2d(np.cov(unconstrained, rowvar=False))
        return cls(
            block=block,
            bijection=bijection,
            whitening_mean=unconstrained.mean(axis=0),
            whitening_factor=np.linalg.cholesky(covariance),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The parameters, in the order of the last axis."""
        return self.bijection.names

    @property
    def dimension(self) -> int:
        """The number of parameters, $d$."""
        return self.bijection.dimension

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
        """The parameters' values in *draws*, as an array of shape $(n, d)$ in :attr:`names` order.

        Parameters
        ----------
        draws : RecordBatch or Mapping
            Draws of the prior, or of a posterior over the same parameters, tracked or raw.
        """
        return block_values(draws, self.block, self.names)


def block_values(draws: Any, block: str, names: tuple[str, ...]) -> np.ndarray:
    """The values of the parameters *names* of *block* in *draws*, stacked into shape $(n, d)$.

    Parameters
    ----------
    draws : RecordBatch or Mapping
        Draws, tracked or raw.
    block : str
        The component holding the parameters.
    names : tuple of str
        The parameters, in the order of the result's last axis.
    """
    raw = draws.raw() if hasattr(draws, "raw") else draws
    # STOPGAP(PG-13): a prior taken back out of a joint draws its block's fields ungrouped.
    values = raw.get(block, raw)
    return np.stack([np.asarray(values[name], dtype=float) for name in names], axis=-1)


def _block_spec(prior: Any, block: str) -> Any:
    """The record spec of *block* in *prior*'s event declaration."""
    components = prior.event_spec.components
    if block not in components or not hasattr(components[block], "fields"):
        raise ValueError(f"the prior has no block of named parameters called {block!r}")
    return components[block]
