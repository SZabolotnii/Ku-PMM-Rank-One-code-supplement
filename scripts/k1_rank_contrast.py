#!/usr/bin/env python3
"""K1 contrast: does the rank of the plug-in penalty change when the centring
means enter the estimating equations un-aggregated?

Linear model y = X theta + eps without intercept (k = 2, xbar != 0), integer
basis e^j, j = 1..3, i.i.d. centred Exp(1) errors.  Every estimator is a
one-step GMM from OLS with the expected Jacobian and W = Omega^{-1}; the oracle
uses the true means, the feasible one the means of the sample-centred OLS
residuals (the K1 procedure).  Three channel structures:

  A  aggregated  -- K1: one condition x_i * h'(phi(e_i) - m), h = F^{-1}b
  B  stacked     -- x_i (x) (phi(e_i) - m): all S means enter one by one,
                    instruments separable (design (x) basis)
  C  mixed       -- x_i (x) (z_1, z_2) and w_i z_3 with w_i = (x_1^2, x_2):
                    a different instrument for the third basis component

Records the eigenvalues of n(Cov_feas - Cov_or).  Expected: A and B rank one
(efficient weighting re-aggregates the stacked means through h); C is not
rank one, and the excess is indefinite -- the uncorrelatedness premise of the
general form (rem:rank) fails.
Output: results/k1_rank_contrast.csv (+ _metadata.json)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from k1_common import write_csv, write_metadata  # noqa: E402

N, REPS, S = 2000, 10000, 3
SEEDS = (20260924, 7)


def population_moments(rng):
    z = rng.exponential(1.0, 5_000_000) - 1.0
    mu = [float(np.mean(z**j)) for j in range(2 * S + 1)]
    F = np.array([[mu[j + l] - mu[j] * mu[l] for l in range(1, S + 1)] for j in range(1, S + 1)])
    b = np.array([j * mu[j - 1] for j in range(1, S + 1)])
    return mu, F, b


def one_structure(conds, X, mu, F, b, rng):
    """conds: list of (Z, w) -- condition Z_i * w'(phi(e_i) - m), w of length S."""
    n = X.shape[0]
    G = np.vstack([-(Z.T @ X / n) * float(w @ b) for Z, w in conds])
    Om = np.block([[(Zc.T @ Zd / n) * float(wc @ F @ wd) for Zd, wd in conds] for Zc, wc in conds])
    W = np.linalg.inv(Om)
    A = np.linalg.solve(G.T @ W @ G, G.T @ W)
    eps = rng.exponential(1.0, (REPS, n)) - 1.0
    y = eps + X @ np.ones(X.shape[1])
    th0 = np.linalg.solve(X.T @ X, X.T @ y.T).T
    r = y - th0 @ X.T
    rc = r - r.mean(axis=1, keepdims=True)
    P = np.stack([r**j for j in range(1, S + 1)])                      # S, REPS, n
    mh = np.stack([(rc**j).mean(axis=1) for j in range(1, S + 1)])     # S, REPS
    mtrue = np.array(mu[1:S + 1])[:, None] * np.ones((1, REPS))

    def gbar(m):
        return np.hstack([(np.einsum("j,jrn->rn", w, P - m[:, :, None]) @ Z) / n for Z, w in conds])

    th_fe = th0 - gbar(mh) @ A.T
    th_or = th0 - gbar(mtrue) @ A.T
    excess = n * (np.cov(th_fe.T) - np.cov(th_or.T))
    ev = np.sort(np.linalg.eigvalsh((excess + excess.T) / 2))[::-1]
    return ev, float(np.trace(n * np.cov(th_or.T)))


def main() -> int:
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mu, F, b = population_moments(rng)
        h = np.linalg.solve(F, b)
        X = rng.uniform(0.0, 2.0, (N, 2))
        Wz = np.column_stack([X[:, 0] ** 2, X[:, 1]])
        e = np.eye(S)
        structures = {
            "A_aggregated": [(X, h)],
            "B_stacked_separable": [(X, e[0]), (X, e[1]), (X, e[2])],
            "C_mixed_instruments": [(X, e[0]), (X, e[1]), (Wz, e[2])],
        }
        for name, conds in structures.items():
            ev, tr_or = one_structure(conds, X, mu, F, b, rng)
            rows.append({
                "seed": seed, "structure": name, "n": N, "reps": REPS,
                "excess_eig_1": float(ev[0]), "excess_eig_2": float(ev[1]),
                "eig2_over_eig1": float(ev[1] / ev[0]),
                "oracle_trace": tr_or,
                "xbar_1": float(X[:, 0].mean()), "xbar_2": float(X[:, 1].mean()),
                "wbar_1": float(Wz[:, 0].mean()), "wbar_2": float(Wz[:, 1].mean()),
                "h_1": float(h[0]), "h_2": float(h[1]), "h_3": float(h[2]),
            })
            print(f"seed={seed} {name:22s} eig={ev[0]:.5f}, {ev[1]:.5f}  eig2/eig1={ev[1] / ev[0]:+.3f}")
    write_csv(ROOT / "results/k1_rank_contrast.csv", rows)
    write_metadata(ROOT / "results/k1_rank_contrast_metadata.json", {
        "script": "scripts/k1_rank_contrast.py", "n": N, "reps": REPS, "seeds": list(SEEDS),
        "model": "y = X theta + eps, no intercept, X ~ U(0,2)^2, eps = Exp(1) - 1, basis e^j j=1..3",
        "claim": "stacked separable conditions keep the penalty rank one; mixed instruments break "
                 "rank one and make the excess indefinite",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
