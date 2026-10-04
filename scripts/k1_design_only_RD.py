"""K1 check 2 — R_D is a design quantity: constant across noise laws, varying across designs.

For a grid of designs (window, envelope, N) compute R_D = |Q^{-1}dbar|^2 / tr Q^{-1}
at the true parameter, and for every S=2 / S=3 AEPD scenario verify the identity
centering_penalty_g / (1 - oracle_g) == R_D to machine precision (noise-invariance),
where both g's are trace ratios against NLS from the closed-form sandwich.
Output: results/k1_design_factor_RD.csv (designs x scenarios) and
results/k1_design_factor_RD_designs.csv (one row per design).
"""

from __future__ import annotations

import numpy as np

from k1_common import (
    ROOT,
    design_quantities,
    feasible_spec_for,
    fixed_design_plugin_sandwich,
    scenarios,
    write_csv,
    write_metadata,
)

DESIGNS = [
    # (label, t_start, t_end, envelope_amplitude, envelope_freq_pi)
    ("track_window_[0.01,2]_env0.3", 0.01, 2.0, 0.3, 1.0),
    ("no_envelope_[0.01,2]", 0.01, 2.0, 0.0, 1.0),
    ("strong_envelope_[0.01,2]_env0.6", 0.01, 2.0, 0.6, 1.0),
    ("short_window_[0.01,1]_env0.3", 0.01, 1.0, 0.3, 1.0),
    ("long_window_[0.01,4]_env0.3", 0.01, 4.0, 0.3, 1.0),
    ("long_window_[0.01,8]_env0.3", 0.01, 8.0, 0.3, 1.0),
    ("shifted_window_[2,4]_env0.3", 2.0, 4.0, 0.3, 1.0),
]
SIZES = (200, 1000, 5000)


def design(label, t0, t1, amp, freq_pi, n):
    t = np.linspace(t0, t1, n)
    envelope = 1.0 + amp * np.cos(freq_pi * np.pi * t)
    return t, envelope


def main() -> int:
    design_rows, cell_rows = [], []
    for label, t0, t1, amp, fpi in DESIGNS:
        for n in SIZES:
            t, env = design(label, t0, t1, amp, fpi, n)
            gradient, q, q_inv, dbar, direction, r_d = design_quantities(n, t=t, envelope=env)
            u = direction / np.linalg.norm(direction)
            design_rows.append({
                "design": label, "sample_size": n, "t_start": t0, "t_end": t1,
                "envelope_amplitude": amp, "design_factor_R": r_d,
                "dbar_A": float(dbar[0]), "dbar_Omega": float(dbar[1]), "dbar_Phi": float(dbar[2]),
                "u_A": float(u[0]), "u_Omega": float(u[1]), "u_Phi": float(u[2]),
                "cond_Q": float(np.linalg.cond(q)),
            })
            for order in (2, 3):
                for sc in scenarios(order):
                    rho, spec = feasible_spec_for(sc, order)
                    if spec.operational_fallback:
                        continue
                    sw = fixed_design_plugin_sandwich(gradient, spec)
                    nls_trace = spec.c2 * np.trace(q_inv)
                    g_or = float(np.trace(sw.oracle) / nls_trace)
                    g_pen = float(np.trace(sw.penalty) / nls_trace)
                    g_feas = float(np.trace(sw.feasible) / nls_trace)
                    ratio = g_pen / (1.0 - g_or) if abs(1.0 - g_or) > 1e-12 else float("nan")
                    cell_rows.append({
                        "design": label, "sample_size": n, "order": order,
                        "scenario_id": sc.scenario_id, "rho_feasible": rho,
                        "oracle_g": g_or, "penalty_g": g_pen, "plugin_g": g_feas,
                        "penalty_over_one_minus_oracle": ratio,
                        "design_factor_R": r_d,
                        "identity_gap": (abs(ratio - r_d) if np.isfinite(ratio) else float("nan")),
                    })
    write_csv(ROOT / "results/k1_design_factor_RD_designs.csv", design_rows)
    write_csv(ROOT / "results/k1_design_factor_RD.csv", cell_rows)
    gaps = np.array([r["identity_gap"] for r in cell_rows if np.isfinite(r["identity_gap"])])
    write_metadata(ROOT / "results/k1_design_factor_RD_metadata.json", {
        "study_id": "k1_design_factor_RD",
        "claim": "penalty_g/(1-oracle_g) equals R_D for every noise law; R_D changes only with the design",
        "designs": len(DESIGNS), "sizes": list(SIZES),
        "cells": len(cell_rows), "max_identity_gap": float(gaps.max()),
        "noise_invariant_cells": int(np.sum(gaps < 1e-10)),
    })
    print(f"cells={len(cell_rows)} max identity gap={gaps.max():.2e}")
    for r in design_rows:
        if r["sample_size"] == 5000:
            print(f"{r['design']:38s} R_D={r['design_factor_R']:.4f}  u=({r['u_A']:+.3f},{r['u_Omega']:+.3f},{r['u_Phi']:+.3f})  cond(Q)={r['cond_Q']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
