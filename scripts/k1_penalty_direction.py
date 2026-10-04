"""K1 check 1 — the plug-in penalty lives in the single direction Q^{-1} dbar.

For every S=2 scenario and sample size in the L2 publication replications,
project the empirical covariance of the feasible fixed-rho PATP one-step onto
the unit vector u = Q^{-1}dbar/|Q^{-1}dbar| and onto an orthonormal basis of
its complement.  Theory (PATP_PLUGIN_SANDWICH_THEORY.md, 7.4):

    Var(u'theta_feas) / Var(u'theta_or)  ->  1 + (c2 - 1/kappa) |Q^{-1}dbar|^2 / (u'Q^{-1}u)
    Var(v'theta_feas) / Var(v'theta_or)  ->  1        for every v  _|_  Q^{-1}dbar

Oracle variances come from the closed form Q^{-1}/kappa (and, where the L2 run
carries the oracle one-step at the same rho, from its empirical covariance).
Output: results/k1_penalty_direction.csv
"""

from __future__ import annotations

import numpy as np

from k1_common import (
    ROOT,
    design_quantities,
    feasible_spec_for,
    fixed_design_plugin_sandwich,
    load_replications,
    scenarios,
    theta_matrix,
    write_csv,
    write_metadata,
)

ORDER = 2


def complement_basis(u: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the complement of u (columns)."""
    q, _ = np.linalg.qr(np.column_stack([u, np.eye(u.size)]))
    return q[:, 1:u.size]


def main() -> int:
    rows_all = load_replications(ORDER)
    sizes = sorted({int(r["sample_size"]) for r in rows_all})
    out = []
    for scenario in scenarios(ORDER):
        rho, spec = feasible_spec_for(scenario, ORDER)
        if spec.operational_fallback:
            continue
        for n in sizes:
            gradient, q, q_inv, dbar, direction, r_d = design_quantities(n)
            sw = fixed_design_plugin_sandwich(gradient, spec)
            u = direction / np.linalg.norm(direction)
            v_basis = complement_basis(u)
            feas = theta_matrix(rows_all, method="PATP Feasible fixed",
                                scenario_id=scenario.scenario_id, sample_size=n, rho=rho)
            orac = theta_matrix(rows_all, method="PATP Oracle one-step",
                                scenario_id=scenario.scenario_id, sample_size=n, rho=rho)
            if feas.shape[0] < 50:
                continue
            cov_feas = np.cov(feas, rowvar=False) * n
            cov_orac_emp = np.cov(orac, rowvar=False) * n if orac.shape[0] >= 50 else None
            pred_ratio_u = float((u @ sw.feasible @ u) / (u @ sw.oracle @ u))
            emp_ratio_u = float((u @ cov_feas @ u) / (u @ sw.oracle @ u))
            emp_ratio_u_vs_oracle_emp = (
                float((u @ cov_feas @ u) / (u @ cov_orac_emp @ u)) if cov_orac_emp is not None else float("nan")
            )
            # complement: average over the orthonormal complement directions
            pred_ratio_v = float(np.mean([
                (v @ sw.feasible @ v) / (v @ sw.oracle @ v) for v in v_basis.T]))
            emp_ratio_v = float(np.mean([
                (v @ cov_feas @ v) / (v @ sw.oracle @ v) for v in v_basis.T]))
            emp_ratio_v_vs_oracle_emp = (
                float(np.mean([(v @ cov_feas @ v) / (v @ cov_orac_emp @ v) for v in v_basis.T]))
                if cov_orac_emp is not None else float("nan")
            )
            # rank check on the empirical excess covariance
            excess = cov_feas - (cov_orac_emp if cov_orac_emp is not None else sw.oracle)
            eig = np.linalg.eigvalsh((excess + excess.T) / 2.0)
            out.append({
                "order": ORDER,
                "scenario_id": scenario.scenario_id,
                "sample_size": n,
                "rho_feasible": rho,
                "n_feasible_reps": int(feas.shape[0]),
                "n_oracle_reps": int(orac.shape[0]),
                "design_factor_R": r_d,
                "predicted_ratio_along_u": pred_ratio_u,
                "empirical_ratio_along_u_vs_oracle_theory": emp_ratio_u,
                "empirical_ratio_along_u_vs_oracle_empirical": emp_ratio_u_vs_oracle_emp,
                "predicted_ratio_complement": pred_ratio_v,
                "empirical_ratio_complement_vs_oracle_theory": emp_ratio_v,
                "empirical_ratio_complement_vs_oracle_empirical": emp_ratio_v_vs_oracle_emp,
                "excess_eig_max": float(eig[-1]),
                "excess_eig_second": float(eig[-2]),
                "excess_eig_min": float(eig[0]),
                "excess_second_over_max": float(abs(eig[-2]) / abs(eig[-1])) if abs(eig[-1]) > 0 else float("nan"),
            })
    if not out:
        raise SystemExit("no rows produced — check the replications file")
    write_csv(ROOT / "results/k1_penalty_direction.csv", out)
    write_metadata(ROOT / "results/k1_penalty_direction_metadata.json", {
        "study_id": "k1_penalty_direction",
        "source": "results/aepd_patp_s2_replications.csv (L2 publication run)",
        "claim": "feasible/oracle variance ratio exceeds 1 only along Q^{-1}dbar; equals 1 on its complement",
        "rows": len(out),
    })
    for r in out:
        if r["sample_size"] == max(sizes):
            print(f"{r['scenario_id']:5s} N={r['sample_size']}  u: pred {r['predicted_ratio_along_u']:.3f} "
                  f"emp {r['empirical_ratio_along_u_vs_oracle_theory']:.3f} (vs oracle-emp {r['empirical_ratio_along_u_vs_oracle_empirical']:.3f})"
                  f"  |  v: pred {r['predicted_ratio_complement']:.3f} emp {r['empirical_ratio_complement_vs_oracle_theory']:.3f} "
                  f"(vs oracle-emp {r['empirical_ratio_complement_vs_oracle_empirical']:.3f})  eig2/eig1={r['excess_second_over_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
