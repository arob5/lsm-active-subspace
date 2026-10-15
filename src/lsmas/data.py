"""The shipped Harvard Forest inputs: driver, NEE observations, observation operator, fixed values.

The files live in ``data/harvard_forest/`` at the root of the repository; its ``PROVENANCE.md`` says
where each one came from. :func:`load_harvard_forest` reads them and checks that they agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["DEFAULT_DATA_DIRECTORY", "HarvardForestData", "load_harvard_forest"]

#: Where the shipped files live in a checkout of the repository.
DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "harvard_forest"


@dataclass(frozen=True)
class HarvardForestData:
    r"""The inputs of the Harvard Forest inverse problem.

    Attributes
    ----------
    driver_path : Path
        The meteorological driver, a SIPNET ``.clim`` file.
    step_ends : pandas.DatetimeIndex
        The end time (UTC) of each of SIPNET's output steps for that driver, length $S$.
    operator : scipy.sparse.csr_matrix
        The observation operator $A$, shape $(n, S)$: feature $f$ is the mean of SIPNET's NEE rate
        over its windows $S_f$, so $A_{fs} = 1/|S_f|$ for $s \in S_f$ and 0 otherwise.
    features : pandas.DataFrame
        One row per feature: ``class`` (``"day"`` or ``"night"``), ``year``, ``month``, ``half``,
        ``n_steps``, ``first_step_end_utc``, ``last_step_end_utc`` and the observed mean ``nee``
        (µmol CO₂ m⁻² s⁻¹).
    fixed_sipnet_parameters : dict of str to float
        Values of SIPNET parameters that are not calibrated, keyed by pySIPNET's names.
    """

    driver_path: Path
    step_ends: pd.DatetimeIndex
    operator: sp.csr_matrix
    features: pd.DataFrame
    fixed_sipnet_parameters: dict[str, float]

    @property
    def observations(self) -> np.ndarray:
        """The observed NEE features $y$, shape $(n,)$, in µmol CO₂ m⁻² s⁻¹."""
        return self.features["nee"].to_numpy()

    @property
    def feature_classes(self) -> np.ndarray:
        """Each feature's class, ``"day"`` or ``"night"``, shape $(n,)$."""
        return self.features["class"].to_numpy()


def load_harvard_forest(directory: Path | None = None) -> HarvardForestData:
    """Read the Harvard Forest inputs and check that they are consistent.

    Parameters
    ----------
    directory : Path, optional
        The directory holding the files, :data:`DEFAULT_DATA_DIRECTORY` by default.

    Returns
    -------
    HarvardForestData

    Raises
    ------
    ValueError
        If the operator's shape, the step axis and the feature table disagree.
    """
    directory = DEFAULT_DATA_DIRECTORY if directory is None else Path(directory)
    data = HarvardForestData(
        driver_path=directory / "driver" / "era5_member_1.clim",
        step_ends=_read_step_ends(directory / "step_axis.json"),
        operator=sp.load_npz(directory / "nee_operator.npz").tocsr(),
        features=pd.read_csv(
            directory / "nee_features.csv",
            parse_dates=["first_step_end_utc", "last_step_end_utc"],
        ),
        fixed_sipnet_parameters=_read_fixed_values(directory / "sipnet_fixed.csv"),
    )
    _check_consistent(data)
    return data


def _read_step_ends(path: Path) -> pd.DatetimeIndex:
    """The step end times that ``step_axis.json`` describes."""
    axis = json.loads(path.read_text())
    step_ends = pd.date_range(
        start=axis["first_step_end"], periods=axis["n_steps"], freq=f"{axis['step_hours']}h"
    )
    if str(step_ends[-1]) != axis["last_step_end"]:
        raise ValueError(
            f"{path.name}: the last step end implied by the first and the step length, "
            f"{step_ends[-1]}, differs from the recorded {axis['last_step_end']}"
        )
    return step_ends


def _read_fixed_values(path: Path) -> dict[str, float]:
    """The fixed SIPNET values in ``sipnet_fixed.csv``, keyed by pySIPNET's names."""
    table = pd.read_csv(path)
    return dict(zip(table["sipnet_parameter_name"], table["value"].astype(float), strict=True))


def _check_consistent(data: HarvardForestData) -> None:
    """Raise ``ValueError`` unless the operator matches the step axis and the feature table."""
    n_features, n_steps = data.operator.shape
    if n_steps != len(data.step_ends):
        raise ValueError(
            f"the operator has {n_steps} columns but the step axis has {len(data.step_ends)} steps"
        )
    if n_features != len(data.features):
        raise ValueError(
            f"the operator has {n_features} rows but the feature table has "
            f"{len(data.features)} features"
        )
    windows_per_feature = np.diff(data.operator.indptr)
    if not np.array_equal(windows_per_feature, data.features["n_steps"].to_numpy()):
        raise ValueError("the operator's windows per feature differ from the table's n_steps")
    if not np.allclose(np.asarray(data.operator.sum(axis=1)).ravel(), 1.0):
        raise ValueError("each row of the operator must average its windows (sum to 1)")
