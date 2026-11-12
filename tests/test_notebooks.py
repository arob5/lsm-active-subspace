"""Every tutorial notebook executes from start to finish without an error.

Personal copies (``notebooks/my_*.ipynb``, see the README) are left out.
"""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = pytest.mark.smoke

NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[1] / "notebooks"


TUTORIALS = sorted(
    path for path in NOTEBOOK_DIRECTORY.glob("*.ipynb") if not path.name.startswith("my_")
)


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda path: path.stem)
def test_notebook_executes(path):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIRECTORY)}},
    )
    client.execute()  # raises CellExecutionError, naming the cell, if any cell fails
