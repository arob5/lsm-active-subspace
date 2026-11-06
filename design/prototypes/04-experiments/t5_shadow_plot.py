"""Sufficient-summary (shadow) plots from an EKI run's history, at no extra forward runs.

Every EKI step records its particles and their predictions. Plotting each particle's tempered
misfit, Phi = 1/2 sum_i (y_i - G_i)^2 / Gamma_ii, against its active variables a_1 and a_2 shows
whether the misfit depends on the active variables alone (a tight curve) and that it does not
depend on an inactive variable (no structure).

Usage: uv run --with matplotlib python t5_shadow_plot.py PROBLEM
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from probpipe import workflow_run  # noqa: E402

from lsmas.problems import load  # noqa: E402
from lsmas.tools import LadderLinearization  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "toy"
problem = load(name)
ladder = LadderLinearization(problem.coordinates)
records = []
with workflow_run(seed=1):
    problem.condition(method="enskit_eki", method_options={"n_particles": 100, "callbacks": [ladder, records.append]})
subspace = ladder.subspace()
xi = np.concatenate([problem.coordinates.whitened_from_parameters(r.theta) for r in records])
predictions = np.concatenate([r.predictions for r in records])
misfit = 0.5 * np.sum((problem.data["nee"] - predictions) ** 2 / problem.noise_variance(), axis=1)
active = subspace.active_variables(xi)
inactive = subspace.inactive_variables(xi)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for axis, values, label in ((axes[0], active[:, 0], "$a_1$"), (axes[1], active[:, 1], "$a_2$"), (axes[2], inactive[:, -1], "last inactive $b$")):
    axis.semilogy(values, misfit, ".", ms=3, alpha=0.6)
    axis.set(xlabel=label, ylabel=r"misfit $\Phi$")
fig.suptitle(f"Sufficient-summary plots, {name}, from EKI's history ({len(xi)} particles)")
fig.tight_layout()
fig.savefig(f"out/figures/shadow_{name}.png", dpi=110)
print("rank", subspace.rank, "particles", len(xi))
