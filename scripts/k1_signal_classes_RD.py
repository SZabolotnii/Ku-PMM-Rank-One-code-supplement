"""K1 check — the design weight R_D for signal classes other than the harmonic bench.

Deterministic: no Monte Carlo.  For every class in ``signal_classes.SIGNAL_CLASSES`` on
the window [0.01, 2] it computes, at the true parameter,

* ``R_D = |Q^{-1} dbar|^2 / tr Q^{-1}`` (trace metric; depends on parameter scaling);
* ``projection_norm_sq = dbar' Q^{-1} dbar`` in [0, 1] (invariant under smooth
  reparametrisation): the squared design-norm of the projection of the constant
  function onto the span of the gradient columns;
* the gap between ``Q^{-1} dbar`` and the least-squares coefficients of the constant
  regressed on the gradient (the two must coincide);
* for classes with a constant component, the gap between ``Q^{-1} dbar`` and the unit
  vector of that component;
* the identity ``penalty_g / (1 - oracle_g) == R_D`` over every S=2 / S=3 noise law.

Output: results/k1_signal_classes_RD.csv (one row per class and sample size),
results/k1_signal_classes_RD_cells.csv (classes x noise laws), and metadata.
"""

from __future__ import annotations

import numpy as np

from k1_common import (
    ROOT,
    feasible_spec_for,
    fixed_design_plugin_sandwich,
    scenarios,
    write_csv,
    write_metadata,
)
from signal_classes import SIGNAL_CLASSES, design_summary

WINDOW = (0.01, 2.0)
SIZES = (200, 1000, 5000)


def main() -> int:
    specs = []
    for order in (2, 3):
        for sc in scenarios(order):
            rho, spec = feasible_spec_for(sc, order)
            if not spec.operational_fallback:
                specs.append((order, sc.scenario_id, rho, spec))

    class_rows, cell_rows = [], []
    for cls in SIGNAL_CLASSES:
        theta = np.asarray(cls.theta, dtype=float)
        for n in SIZES:
            t = np.linspace(*WINDOW, n)
            gradient = cls.gradient(theta, t)
            summary = design_summary(gradient)
            direction = summary["direction"]
            lstsq = np.linalg.lstsq(gradient.T, np.ones(n), rcond=None)[0]
            residual_ms = float(np.mean((1.0 - gradient.T @ direction) ** 2))
            if cls.has_constant_component:
                unit = np.zeros(theta.size)
                unit[-1] = 1.0
                constant_gap = float(np.max(np.abs(direction - unit)))
            else:
                constant_gap = float("nan")
            u = direction / np.linalg.norm(direction)
            class_rows.append({
                "signal_class": cls.label, "parameter_count": theta.size,
                "parameters": " ".join(cls.params), "sample_size": n,
                "t_start": WINDOW[0], "t_end": WINDOW[1],
                "design_factor_R": summary["design_factor_R"],
                "projection_norm_sq": summary["projection_norm_sq"],
                "projection_over_p": summary["projection_norm_sq"] / theta.size,
                "constant_residual_mean_square": residual_ms,
                "lstsq_gap": float(np.max(np.abs(direction - lstsq))),
                "constant_component_gap": constant_gap,
                "direction_unit": " ".join(f"{x:+.6f}" for x in u),
                "cond_Q": summary["cond_Q"],
            })
            nls_trace_unit = np.trace(summary["q_inv"])
            for order, scenario_id, rho, spec in specs:
                sw = fixed_design_plugin_sandwich(gradient, spec)
                g_or = float(np.trace(sw.oracle) / (spec.c2 * nls_trace_unit))
                g_pen = float(np.trace(sw.penalty) / (spec.c2 * nls_trace_unit))
                ratio = g_pen / (1.0 - g_or) if abs(1.0 - g_or) > 1e-12 else float("nan")
                eig = np.linalg.eigvalsh(sw.penalty)
                cell_rows.append({
                    "signal_class": cls.label, "sample_size": n, "order": order,
                    "scenario_id": scenario_id, "rho_feasible": rho,
                    "oracle_g": g_or, "penalty_g": g_pen,
                    "design_factor_R": summary["design_factor_R"],
                    "identity_gap": (abs(ratio - summary["design_factor_R"])
                                     if np.isfinite(ratio) else float("nan")),
                    "penalty_min_eig": float(eig.min()),
                    "penalty_rank": int(np.linalg.matrix_rank(sw.penalty, tol=1e-10)),
                })

    write_csv(ROOT / "results/k1_signal_classes_RD.csv", class_rows)
    write_csv(ROOT / "results/k1_signal_classes_RD_cells.csv", cell_rows)
    gaps = np.array([r["identity_gap"] for r in cell_rows if np.isfinite(r["identity_gap"])])
    write_metadata(ROOT / "results/k1_signal_classes_RD_metadata.json", {
        "study_id": "k1_signal_classes_RD",
        "script": "scripts/k1_signal_classes_RD.py",
        "claim": "Q^{-1} dbar is the projection of the constant onto the gradient span; "
                 "R_D is a design quantity for every smooth signal class",
        "monte_carlo": False,
        "window": list(WINDOW), "sizes": list(SIZES),
        "signal_classes": {c.label: dict(zip(c.params, c.theta)) for c in SIGNAL_CLASSES},
        "cells": len(cell_rows), "max_identity_gap": float(gaps.max()),
        "max_lstsq_gap": max(r["lstsq_gap"] for r in class_rows),
    })
    print(f"cells={len(cell_rows)} max identity gap={gaps.max():.2e}")
    for r in class_rows:
        if r["sample_size"] == 5000:
            print(f"{r['signal_class']:26s} p={r['parameter_count']}  R_D={r['design_factor_R']:.4f}  "
                  f"proj={r['projection_norm_sq']:.4f}  cond(Q)={r['cond_Q']:.1f}  "
                  f"lstsq gap={r['lstsq_gap']:.1e}  const gap={r['constant_component_gap']:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
