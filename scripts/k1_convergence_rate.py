"""K1 check 4 — how fast the empirical feasible trace factor reaches the first-order prediction.

From the L2 summary CSVs (S=2 and S=3): for every scenario and N, the empirical
trace_g of 'PATP Feasible fixed' (with its bootstrap CI) against (a) the oracle
theory_g and (b) the plug-in prediction g_or + (1 - g_or) R_D(N).  Reports the
gap, whether the prediction lies inside the CI, and the log-log slope of |gap|
in N per scenario.  Output: results/k1_convergence_rate.csv
"""

from __future__ import annotations

import numpy as np

from k1_common import (
    ROOT,
    design_quantities,
    feasible_spec_for,
    fixed_design_plugin_sandwich,
    load_summary,
    scenarios,
    write_csv,
    write_metadata,
)


def main() -> int:
    out = []
    slopes = []
    for order in (2, 3):
        summary = load_summary(order)
        for sc in scenarios(order):
            rho, spec = feasible_spec_for(sc, order)
            if spec.operational_fallback:
                continue
            rows = [r for r in summary if r["scenario_id"] == sc.scenario_id
                    and r["method"] == "PATP Feasible fixed"]
            gaps, ns = [], []
            for r in sorted(rows, key=lambda x: int(x["sample_size"])):
                n = int(r["sample_size"])
                gradient, q, q_inv, dbar, direction, r_d = design_quantities(n)
                sw = fixed_design_plugin_sandwich(gradient, spec)
                nls_trace = spec.c2 * np.trace(q_inv)
                g_or = float(np.trace(sw.oracle) / nls_trace)
                g_pred = float(np.trace(sw.feasible) / nls_trace)
                g_emp = float(r["trace_g"])
                lo, hi = float(r["trace_g_ci_low"]), float(r["trace_g_ci_high"])
                gap = g_emp - g_pred
                out.append({
                    "order": order, "scenario_id": sc.scenario_id, "sample_size": n,
                    "rho_feasible": rho, "oracle_g": g_or, "design_factor_R": r_d,
                    "plugin_g_predicted": g_pred, "empirical_trace_g": g_emp,
                    "ci_low": lo, "ci_high": hi,
                    "gap_empirical_minus_predicted": gap,
                    "gap_empirical_minus_oracle": g_emp - g_or,
                    "prediction_inside_ci": int(lo <= g_pred <= hi),
                    "oracle_inside_ci": int(lo <= g_or <= hi),
                    "fallback_rate": float(r.get("fallback_rate", "nan") or "nan"),
                })
                gaps.append(abs(gap)); ns.append(n)
            if len(ns) >= 3 and all(g > 0 for g in gaps):
                slope = float(np.polyfit(np.log(ns), np.log(gaps), 1)[0])
            else:
                slope = float("nan")
            slopes.append({"order": order, "scenario_id": sc.scenario_id, "loglog_slope_abs_gap_vs_N": slope})
    write_csv(ROOT / "results/k1_convergence_rate.csv", out)
    write_csv(ROOT / "results/k1_convergence_rate_slopes.csv", slopes)
    inside = {o: (sum(r["prediction_inside_ci"] for r in out if r["order"] == o and r["sample_size"] == 5000),
                  sum(1 for r in out if r["order"] == o and r["sample_size"] == 5000)) for o in (2, 3)}
    inside_or = {o: (sum(r["oracle_inside_ci"] for r in out if r["order"] == o and r["sample_size"] == 5000),
                     sum(1 for r in out if r["order"] == o and r["sample_size"] == 5000)) for o in (2, 3)}
    write_metadata(ROOT / "results/k1_convergence_rate_metadata.json", {
        "study_id": "k1_convergence_rate",
        "source": "results/aepd_patp_s{2,3}_summary.csv",
        "prediction_inside_ci_at_N5000": {str(k): list(v) for k, v in inside.items()},
        "oracle_inside_ci_at_N5000": {str(k): list(v) for k, v in inside_or.items()},
        "rows": len(out),
    })
    for o in (2, 3):
        print(f"S={o}: prediction inside bootstrap CI at N=5000 in {inside[o][0]}/{inside[o][1]} scenarios; "
              f"oracle inside CI in {inside_or[o][0]}/{inside_or[o][1]}")
    for s in slopes:
        print(f"  S={s['order']} {s['scenario_id']:5s} slope log|gap| vs log N = {s['loglog_slope_abs_gap_vs_N']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
