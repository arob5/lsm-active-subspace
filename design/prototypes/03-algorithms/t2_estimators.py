"""What EKI's statistical linearizations estimate, checked against exact Jacobians on the toy.

In whitened coordinates xi, with Lambda = diag(eta_i / v_i) (noise scales fixed at 1), the
information matrices compared are

    H(mu)    = E_mu[ J(xi)^T Lambda J(xi) ]              (the active-subspace / LIS target)
    Hbar(mu) = E_mu[J(xi)]^T Lambda E_mu[J(xi)]          (what one statistical linearization gives)

for mu the prior, each EKI level, and the ladder average sum_t dbeta_t (.)_t. Statistical
linearization A = C_{G xi} C_{xi xi}^{-1} equals E_mu[J] when mu is Gaussian (Stein), so A^T Lambda A
estimates Hbar, not H; H - Hbar = E[(J - EJ)^T Lambda (J - EJ)] >= 0.

Writes out/estimators.npz with every matrix, for t3_compare.py.
Usage: python t2_estimators.py [n_particles] [seed]
"""

import sys
import time

import jax
import numpy as np
from probpipe import workflow_run

from common import (
    coordinates,
    forward_batch,
    information,
    jacobian,
    mean_jacobian_information,
    problem,
    slr_information,
    whitened,
)
from lsmas._stopgaps.sampling import draw_with_key
from toy import TOY_PARAMETERS

N_PARTICLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
start = time.time()
prior_theta = coordinates.array_from_draws(draw_with_key(problem.prior, jax.random.key(5), (4000,)))
prior_xi = whitened(prior_theta)
J_prior = np.asarray(jacobian(prior_xi))
results = {
    "H_prior": information(J_prior),
    "Hbar_prior": mean_jacobian_information(J_prior),
}
g_prior = np.asarray(forward_batch(prior_xi))
for size in (50, 100, 200, 1000, 4000):
    results[f"Hslr_prior_{size}"] = slr_information(prior_xi[:size], g_prior[:size])
print(f"prior matrices: {time.time() - start:.0f} s")

records = []
start = time.time()
with workflow_run(seed=SEED):
    posterior = problem.condition(
        method="enskit_eki", method_options={"n_particles": N_PARTICLES, "callbacks": [records.append]}
    )
print(f"EKI: {len(records)} steps, {time.time() - start:.0f} s; betas", [round(r.beta, 3) for r in records])
final_xi = whitened(coordinates.array_from_draws(posterior.atoms).reshape(-1, len(TOY_PARAMETERS)))
H_ladder = np.zeros((len(TOY_PARAMETERS),) * 2)
Hbar_ladder, Hslr_ladder, Hslr_uniform = (np.zeros_like(H_ladder) for _ in range(3))
for record in records:
    xi = whitened(record.theta)
    J = np.asarray(jacobian(xi))
    H_ladder += record.increment * information(J)
    Hbar_ladder += record.increment * mean_jacobian_information(J)
    Hslr_ladder += record.increment * slr_information(xi, record.predictions)
    Hslr_uniform += slr_information(xi, record.predictions) / len(records)
J_final = np.asarray(jacobian(final_xi))
results |= {
    "H_ladder": H_ladder,
    "Hbar_ladder": Hbar_ladder,
    "Hslr_ladder": Hslr_ladder,
    "Hslr_ladder_uniform": Hslr_uniform,
    "H_eki_final": information(J_final),
    "Hbar_eki_final": mean_jacobian_information(J_final),
    "Hslr_eki_final": slr_information(final_xi, np.asarray(forward_batch(final_xi))),
    "eki_final_xi": final_xi,
    "betas": np.array([r.beta for r in records]),
    "increments": np.array([r.increment for r in records]),
}
np.savez(f"out/estimators_J{N_PARTICLES}_s{SEED}.npz", **results)
for name in ("H_prior", "Hbar_prior", "Hslr_prior_100", "H_ladder", "Hslr_ladder", "H_eki_final", "Hslr_eki_final"):
    print(f"{name:16s} eigenvalues", np.round(np.linalg.eigvalsh(results[name])[::-1][:8], 2))
