"""K1 check — the constant signal: what decides whether the feasible gain survives.

Model ``y_i = C + eps_i`` (p = 1, Q = 1, dbar = 1, R_D = 1).  The preliminary estimate
is the sample mean, followed by one refinement step with an integer-power basis.  Two
ways to centre the basis are compared on the same draws:

* ``estimated`` — every centring mean is the sample mean of the basis function over
  the same residuals (the scheme of the article).  The centred score then sums to
  zero identically, the step is zero and the estimate is the sample mean, for any S.
* ``imposed``   — the centring means are fixed a priori.  For symmetric noise and
  S = 3 the odd centring mean E eps^3 = 0 is imposed by symmetry (classical PMM3);
  for asymmetric noise and S = 2 the centring mean of eps^2 is the variance, which is
  known only to an oracle.

Weights (F, b) are estimated from the residuals in both branches.
Output: results/k1_constant_signal_check.csv (+ _metadata.json).
"""

from __future__ import annotations

import argparse

import numpy as np

from k1_common import ROOT, write_csv, write_metadata

# label, order, basis powers, sampler of unit-variance zero-mean noise, population g
CASES = (
    ("S2_asymmetric_exponential", 2, (1, 2),
     lambda rng, size: rng.exponential(1.0, size) - 1.0, 1.0 - 4.0 / (2.0 + 6.0)),
    ("S3_symmetric_uniform", 3, (1, 3),
     lambda rng, size: rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size),
     1.0 - 1.2**2 / (6.0 - 9.0 * 1.2 + 48.0 / 7.0)),
)


def one_step(y: np.ndarray, powers: tuple[int, ...], imposed_means) -> np.ndarray:
    """One refinement step from the sample mean; rows of ``y`` are replications."""
    start = y.mean(axis=1, keepdims=True)
    res = y - start
    basis = np.stack([res**q for q in powers], axis=1)            # (M, S, n)
    sample_means = basis.mean(axis=2, keepdims=True)
    centred_for_weights = basis - sample_means
    f = np.einsum("msn,mtn->mst", centred_for_weights, centred_for_weights) / y.shape[1]
    b = np.stack([q * (res ** (q - 1)).mean(axis=1) for q in powers], axis=1)
    h = np.linalg.solve(f, b[..., None])[..., 0]
    kappa = np.einsum("ms,ms->m", h, b)
    if imposed_means is None:
        means = sample_means[..., 0]
    else:
        means = imposed_means(res)
    score = np.einsum("ms,ms->m", h, basis.mean(axis=2) - means)
    return start[:, 0] + score / kappa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--sample-sizes", default="200,1000,5000")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    sizes = [int(x) for x in args.sample_sizes.split(",")]

    rows = []
    for case_index, (label, order, powers, sampler, g_population) in enumerate(CASES):
        for n in sizes:
            rng = np.random.default_rng([args.seed, case_index, n])
            y = 0.5 + sampler(rng, (args.reps, n))
            mean_estimate = y.mean(axis=1)
            estimated = one_step(y, powers, None)
            if order == 2:
                # oracle centring: E eps = 0, E eps^2 = c2 = 1
                imposed = one_step(y, powers, lambda res: np.array([0.0, 1.0])[None, :])
                imposed_source = "oracle variance c2 = 1"
            else:
                # symmetry: E eps = 0, E eps^3 = 0
                imposed = one_step(y, powers, lambda res: np.zeros((1, 2)))
                imposed_source = "symmetry, E eps^3 = 0"
            var_mean = float(np.var(mean_estimate, ddof=1))
            rows.append({
                "case": label, "order": order, "sample_size": n, "replications": args.reps,
                "population_g": g_population,
                "g_estimated_centring": float(np.var(estimated, ddof=1)) / var_mean,
                "max_abs_step_estimated_centring": float(np.max(np.abs(estimated - mean_estimate))),
                "g_imposed_centring": float(np.var(imposed, ddof=1)) / var_mean,
                "imposed_centring_source": imposed_source,
            })
            print(f"{label:28s} n={n:5d}  g_pop={g_population:.4f}  "
                  f"g(estimated)={rows[-1]['g_estimated_centring']:.4f}  "
                  f"max|step|={rows[-1]['max_abs_step_estimated_centring']:.1e}  "
                  f"g(imposed)={rows[-1]['g_imposed_centring']:.4f}")

    write_csv(ROOT / "results/k1_constant_signal_check.csv", rows)
    write_metadata(ROOT / "results/k1_constant_signal_check_metadata.json", {
        "study_id": "k1_constant_signal_check",
        "script": "scripts/k1_constant_signal_check.py",
        "claim": "for a constant signal the feasible gain is lost exactly when the centring "
                 "means are estimated from the same residuals, for any S; it survives when "
                 "they are imposed a priori",
        "replications": args.reps, "sample_sizes": sizes, "seed": args.seed,
        "resampling_unit": "iid Monte Carlo replicate",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
