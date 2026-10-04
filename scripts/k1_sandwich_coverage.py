"""K1 check 3 — coverage of the two feasible covariance estimators (12.1) and (12.2).

Fresh Monte Carlo on the L2 bench (``--order 2``, the asymmetric ``A*`` manifest, or
``--order 3``, the symmetric ``S*`` one; standardised AEPD noise, fixed feasible
rho from the L2 optimum): per replication, NLS fit -> residual moments ->
feasible one-step; then two covariance estimates of theta_hat,

  (12.1) closed form   Sigma_hat = fixed_design_plugin_sandwich(grad(theta_hat), spec_hat).feasible / n
  (12.2) empirical     Sigma_hat = empirical_fixed_design_sandwich(grad(theta_hat), resid(theta_hat), spec_hat) / n
  (naive)              oracle form Q_hat^{-1}/kappa_hat / n   — ignores the centring penalty

and the coverage of the true theta by per-parameter Wald 95% intervals and the
joint chi2(3) ellipsoid.  ``--pilot`` runs 200 replications (default);
``--publication`` runs 1000.  Output:
results/k1_sandwich_coverage[_s3][_pilot].csv
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import chi2, norm

from k1_common import (
    ROOT,
    TRUE_THETA,
    feasible_spec_for,
    fixed_design_plugin_sandwich,
    scenarios,
    signal_design,
    signal_gradient,
    write_csv,
    write_metadata,
)
from aepd_core import standardized_aepd_rvs  # noqa: E402
from patp_adaptive_core import build_empirical_spec  # noqa: E402
from patp_experiment_core import fit_bounded_nls, one_step_estimate, signal  # noqa: E402
from patp_plugin_theory import empirical_fixed_design_sandwich  # noqa: E402

DEFAULT_SCENARIOS = {
    2: ("A0", "A2L", "A4L", "A3L"),
    # S=3 is the symmetric manifest: the closed form predicts the largest centring
    # penalties of the whole table there (S075 0.169, S500 0.086), so this is where
    # (12.1) is under the most pressure.  See K1_EXTERNAL_REPLICATION_2026-09-21 §3.2.
    3: ("S075", "S100", "S300", "S500"),
}
Z = norm.ppf(0.975)
CHI2_95 = chi2.ppf(0.95, df=3)


def cover(theta_hat, cov, truth):
    d = theta_hat - truth
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    per_param = np.abs(d) <= Z * se
    try:
        joint = float(d @ np.linalg.solve(cov, d)) <= CHI2_95
    except np.linalg.LinAlgError:
        joint = False
    return per_param, joint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--publication", action="store_true")
    ap.add_argument("--sample-sizes", default="200,500,1000,5000")
    ap.add_argument("--scenario", action="append", default=[])
    ap.add_argument("--order", type=int, default=2, choices=(2, 3))
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()
    order = args.order
    reps = args.reps or (1000 if args.publication else 200)
    tag = "" if args.publication else "_pilot"
    if order != 2:
        tag = f"_s{order}{tag}"
    sizes = [int(s) for s in args.sample_sizes.split(",")]
    wanted = tuple(args.scenario) or DEFAULT_SCENARIOS[order]
    rng = np.random.default_rng(args.seed)
    out = []
    for sc in scenarios(order):
        if sc.scenario_id not in wanted:
            continue
        rho, spec_pop = feasible_spec_for(sc, order)
        if spec_pop.operational_fallback:
            continue
        for n in sizes:
            t, env = signal_design(n)
            clean = signal(TRUE_THETA, t, env)
            counts = {k: np.zeros(3) for k in ("closed", "empirical", "naive")}
            joint = {k: 0 for k in counts}
            valid = 0
            se_ratio = {k: [] for k in counts}
            thetas = []
            for _ in range(reps):
                y = clean + standardized_aepd_rvs(n, *sc.aepd_shape, rng=rng)
                fit = fit_bounded_nls(y, t, env)
                resid = y - signal(fit.theta, t, env)
                emp = build_empirical_spec(rho, order, resid)
                if not emp.valid:
                    continue
                th, fb, _, _ = one_step_estimate(y, fit.theta, t, env, emp.spec)
                if fb:
                    continue
                g = signal_gradient(th, t, env)
                r_hat = y - signal(th, t, env)
                sw = fixed_design_plugin_sandwich(g, emp.spec)
                covs = {
                    "closed": sw.feasible / n,
                    "empirical": empirical_fixed_design_sandwich(g, r_hat, emp.spec) / n,
                    "naive": sw.oracle / n,
                }
                valid += 1
                thetas.append(th)
                for k, c in covs.items():
                    pp, jt = cover(th, c, TRUE_THETA)
                    counts[k] += pp
                    joint[k] += int(jt)
                    se_ratio[k].append(np.sqrt(np.clip(np.diag(c), 0, None)))
            if valid == 0:
                continue
            emp_sd = np.std(np.asarray(thetas), axis=0, ddof=1)
            for k in counts:
                mean_se = np.mean(np.asarray(se_ratio[k]), axis=0)
                out.append({
                    "order": order, "scenario_id": sc.scenario_id, "sample_size": n,
                    "rho_feasible": rho, "estimator": k, "reps": reps, "valid": valid,
                    "coverage_A": counts[k][0] / valid, "coverage_Omega": counts[k][1] / valid,
                    "coverage_Phi": counts[k][2] / valid, "coverage_joint_chi2": joint[k] / valid,
                    "mean_se_over_mc_sd_A": float(mean_se[0] / emp_sd[0]),
                    "mean_se_over_mc_sd_Omega": float(mean_se[1] / emp_sd[1]),
                    "mean_se_over_mc_sd_Phi": float(mean_se[2] / emp_sd[2]),
                })
            row = {r["estimator"]: r for r in out if r["scenario_id"] == sc.scenario_id and r["sample_size"] == n}
            print(f"{sc.scenario_id:4s} N={n:5d} valid={valid:4d}  joint cov: closed {row['closed']['coverage_joint_chi2']:.3f} "
                  f"empirical {row['empirical']['coverage_joint_chi2']:.3f} naive {row['naive']['coverage_joint_chi2']:.3f}  "
                  f"| se/sd(A,Om,Phi) closed=({row['closed']['mean_se_over_mc_sd_A']:.2f},{row['closed']['mean_se_over_mc_sd_Omega']:.2f},{row['closed']['mean_se_over_mc_sd_Phi']:.2f})")
    if not out:
        raise SystemExit("no rows")
    write_csv(ROOT / f"results/k1_sandwich_coverage{tag}.csv", out)
    write_metadata(ROOT / f"results/k1_sandwich_coverage{tag}_metadata.json", {
        "study_id": "k1_sandwich_coverage", "pilot": not args.publication, "reps": reps,
        "order": order, "sample_sizes": sizes, "scenarios": list(wanted),
        "seed": args.seed, "nominal": 0.95,
        "estimators": {"closed": "(12.1) Q^-1/kappa + (c2-1/kappa) Q^-1 dbar dbar' Q^-1 with plug-in moments",
                       "empirical": "(12.2) empirical same-sample sandwich", "naive": "oracle form Q^-1/kappa, ignores the centring penalty"},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
