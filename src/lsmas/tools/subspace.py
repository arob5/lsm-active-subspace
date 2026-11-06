r"""Active subspaces: the leading eigenvectors of an information matrix, and the rank to keep.

Every estimator in :mod:`lsmas.tools.estimators` returns an estimate of an information matrix in
whitened coordinates,

$$
H(\mu) = \mathbb E_\mu\!\left[DG(\xi)^\top \Gamma^{-1} DG(\xi)\right],
$$

and :class:`Subspace` turns it into what algorithms and plots use: the eigenvalues
$\lambda_1 \ge \dots \ge \lambda_d$, the basis $W$ of the leading $r$ eigenvectors and the
basis $W_\perp$ of the rest, the active and inactive variables $a = W^\top \xi$ and
$b = W_\perp^\top \xi$, and each parameter's loadings and activity score. Because the prior's
information in whitened coordinates is the identity, $\lambda_i > 1$ means the data inform
direction $i$ more than the prior does.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Subspace", "suggest_rank"]

#: The default bound on the discarded eigenvalues, for :func:`suggest_rank`.
DEFAULT_DISCARDED_INFORMATION = 0.5


@dataclass(frozen=True, eq=False)
class Subspace:
    r"""An active subspace of rank $r$ in whitened coordinates.

    Build one with :meth:`from_information`.

    Attributes
    ----------
    eigenvalues : numpy.ndarray
        $\lambda_1 \ge \dots \ge \lambda_d$ of the information matrix, shape $(d,)$.
    eigenvectors : numpy.ndarray
        The matching orthonormal eigenvectors as columns, shape $(d, d)$; each is signed so that
        its largest entry in absolute value is positive.
    rank : int
        $r$, the number of leading eigenvectors that span the subspace.
    parameter_names : tuple of str
        The calibration parameters, in the order of the whitened coordinates.
    estimator : str
        Which estimator produced the information matrix, such as ``"ladder_linearization"``.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rank: int
    parameter_names: tuple[str, ...]
    estimator: str

    @classmethod
    def from_information(
        cls,
        information: np.ndarray,
        *,
        parameter_names: Sequence[str],
        estimator: str,
        rank: int | None = None,
        discarded_information: float = DEFAULT_DISCARDED_INFORMATION,
    ) -> Subspace:
        r"""The subspace of the leading eigenvectors of a symmetric information matrix $H$.

        Parameters
        ----------
        information : numpy.ndarray
            $H$, shape $(d, d)$, symmetric.
        parameter_names : sequence of str
            The calibration parameters, in the order of $H$'s rows.
        estimator : str
            Which estimator produced $H$.
        rank : int, optional
            $r$. By default the rank :func:`suggest_rank` gives for *discarded_information*.
        discarded_information : float
            The bound on the discarded eigenvalues when the rank is not given.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(information, dtype=float))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], _signed(eigenvectors[:, order])
        if rank is None:
            rank = suggest_rank(eigenvalues, discarded_information)
        _check_rank(rank, len(eigenvalues))
        return cls(eigenvalues, eigenvectors, rank, tuple(parameter_names), estimator)

    @property
    def dimension(self) -> int:
        """$d$, the number of calibration parameters."""
        return len(self.eigenvalues)

    @property
    def information(self) -> np.ndarray:
        r"""The information matrix the subspace came from,
        $H = \sum_i \lambda_i v_i v_i^\top$, shape $(d, d)$."""
        return (self.eigenvectors * self.eigenvalues) @ self.eigenvectors.T

    @property
    def basis(self) -> np.ndarray:
        """$W$: the leading $r$ eigenvectors as columns, shape $(d, r)$."""
        return self.eigenvectors[:, : self.rank]

    @property
    def complement(self) -> np.ndarray:
        r"""$W_\perp$: the remaining eigenvectors as columns, shape $(d, d - r)$."""
        return self.eigenvectors[:, self.rank :]

    def with_rank(self, rank: int) -> Subspace:
        """The same eigenvectors with a different rank."""
        _check_rank(rank, self.dimension)
        return dataclasses.replace(self, rank=rank)

    def active_variables(self, xi: np.ndarray) -> np.ndarray:
        r"""$a = W^\top \xi$ for whitened coordinates *xi*, shape $(\ldots, d) \to (\ldots, r)$."""
        return np.asarray(xi, dtype=float) @ self.basis

    def inactive_variables(self, xi: np.ndarray) -> np.ndarray:
        r"""$b = W_\perp^\top \xi$, shape $(\ldots, d) \to (\ldots, d - r)$."""
        return np.asarray(xi, dtype=float) @ self.complement

    def whitened(self, active: np.ndarray, inactive: np.ndarray) -> np.ndarray:
        r"""$\xi = W a + W_\perp b$ from active and inactive variables, shape $(\ldots, d)$."""
        return (
            np.asarray(active, dtype=float) @ self.basis.T
            + np.asarray(inactive, dtype=float) @ self.complement.T
        )

    def loadings(self) -> pd.DataFrame:
        """$W$ as a table: one row per parameter, one column per direction of the subspace."""
        return pd.DataFrame(
            self.basis,
            index=pd.Index(self.parameter_names, name="parameter"),
            columns=[f"direction_{i + 1}" for i in range(self.rank)],
        )

    def activity_scores(self) -> pd.Series:
        r"""Each parameter's activity score, $\alpha_j = \sum_{i=1}^{r} \lambda_i W_{ji}^2$.

        The score is the information the subspace holds about the parameter's own whitened
        coordinate (Constantine & Diaz 2017); with $r = d$ it is $H_{jj}$. Sorted, largest first.
        """
        scores = (self.basis**2) @ self.eigenvalues[: self.rank]
        series = pd.Series(scores, index=pd.Index(self.parameter_names, name="parameter"))
        return series.sort_values(ascending=False).rename("activity_score")


def suggest_rank(
    eigenvalues: Sequence[float], discarded_information: float = DEFAULT_DISCARDED_INFORMATION
) -> int:
    r"""The smallest rank whose discarded eigenvalues sum to at most *discarded_information*.

    For eigenvalues $\lambda_1 \ge \dots \ge \lambda_d$ of an information matrix, the rank is the
    smallest $r \ge 1$ with

    $$
    \sum_{i > r} \lambda_i \le \varepsilon .
    $$

    The rule follows the error of a reduced posterior rather than the share of the trace: one
    dominant eigenvalue can hold most of the trace while several weaker directions remain informed.
    For the posterior-averaged information of gradients of the log-likelihood, half the discarded
    eigenvalues bounds the Kullback–Leibler error of the best reduced posterior (Zahm et al. 2022).

    Parameters
    ----------
    eigenvalues : sequence of float
        $\lambda_1 \ge \dots \ge \lambda_d$.
    discarded_information : float
        $\varepsilon \ge 0$.
    """
    values = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    discarded_after = np.concatenate([np.cumsum(values[::-1])[::-1][1:], [0.0]])
    return int(np.argmax(discarded_after <= discarded_information)) + 1


def _signed(eigenvectors: np.ndarray) -> np.ndarray:
    """Each column multiplied by the sign of its largest entry in absolute value."""
    largest = eigenvectors[
        np.argmax(np.abs(eigenvectors), axis=0), np.arange(eigenvectors.shape[1])
    ]
    return eigenvectors * np.sign(largest)


def _check_rank(rank: int, dimension: int) -> None:
    r"""Raise ``ValueError`` unless $1 \le r \le d$."""
    if not 1 <= rank <= dimension:
        raise ValueError(f"the rank must be between 1 and {dimension}, got {rank}")
