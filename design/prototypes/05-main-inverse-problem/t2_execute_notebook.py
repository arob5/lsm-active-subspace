"""A notebook built from the walkthrough's opening cells runs under nbclient in the project's
environment, and its executed size with one figure.

Run from the repository root:

    uv run --with nbformat --with nbclient --with ipykernel \
        python design/prototypes/05-main-inverse-problem/t2_execute_notebook.py
"""

from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

cells = [
    nbformat.v4.new_markdown_cell("# The Harvard Forest inverse problem (execution check)"),
    nbformat.v4.new_code_cell(
        "from lsmas.problems import load\n"
        "problem = load('harvard_forest', scales='inferred')\n"
        "print(problem.describe())"
    ),
    nbformat.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "from probpipe import sample, workflow_run\n"
        "with workflow_run(seed=1):\n"
        "    replications = sample(problem.model, 16)\n"
        "nee = np.asarray(replications['nee'].raw)\n"
        "plt.plot(nee.T, color='0.6', lw=0.5); plt.plot(problem.data['nee'], 'k.', ms=2)"
    ),
]
notebook = nbformat.v4.new_notebook(cells=cells)
start = time.time()
NotebookClient(notebook, timeout=600, kernel_name="python3").execute()
path = OUT / "t2_executed.ipynb"
nbformat.write(notebook, path)
line = f"executed in {time.time() - start:.1f} s; {path.stat().st_size // 1024} KB with one figure"
print(line)
(OUT / "t2.txt").write_text(line + "\n")
