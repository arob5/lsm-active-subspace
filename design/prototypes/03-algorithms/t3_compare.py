"""How well each estimated subspace captures the posterior's information, on the toy.

Reference: H_post = E_post[J^T Lambda J], from 1000 NUTS draws and exact Jacobians. For an estimated
subspace W (the top-r eigenvectors of a matrix), the score is the share of the best possible
captured information,

    captured_r(W) = tr(W^T H_post W) / (lambda_1 + ... + lambda_r)(H_post)   in [0, 1].

Also scored: the covariance-informed subspace, the top eigenvectors of Sigma^{-1} - I for the
whitened covariance Sigma of the final EKI ensemble (and of the NUTS draws); for a linear model
with a Gaussian prior, Sigma^{-1} - I is exactly the constant J^T Lambda J.

Usage: python t3_compare.py estimators_file [more estimator files, averaged scores]
"""

import sys

import numpy as np

from common import captured, information, jacobian

nuts = np.load("out/nuts_observed_fixed.npz")
xi_post = nuts["xi"].reshape(-1, nuts["xi"].shape[-1])[::4]
H_post = information(jacobian(xi_post))
np.save("out/H_post.npy", H_post)
print("H_post eigenvalues", np.round(np.sort(np.linalg.eigvalsh(H_post))[::-1], 2))
covariance_nuts = np.cov(xi_post.T)
candidates_nuts = {"cov_nuts": np.linalg.inv(covariance_nuts) - np.eye(len(covariance_nuts))}

RANKS = (1, 2, 3, 4, 6, 8)
rows = {}
for path in sys.argv[1:]:
    run = np.load(path)
    candidates = {
        name: run[name]
        for name in run.files
        if name.startswith(("H", "Hbar", "Hslr")) and run[name].ndim == 2 and run[name].shape[0] == H_post.shape[0]
    }
    final = run["eki_final_xi"]
    candidates["cov_eki_final"] = np.linalg.inv(np.cov(final.T)) - np.eye(final.shape[1])
    for name, matrix in (candidates | candidates_nuts).items():
        rows.setdefault(name, []).append([captured(matrix, H_post, r) for r in RANKS])

print(f"{'estimate':22s}" + "".join(f"  r={r:<4d}" for r in RANKS))
for name, scores in rows.items():
    mean = np.mean(scores, axis=0)
    print(f"{name:22s}" + "".join(f"  {s:6.3f}" for s in mean))
