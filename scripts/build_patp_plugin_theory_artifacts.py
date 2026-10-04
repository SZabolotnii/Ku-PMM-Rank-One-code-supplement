#!/usr/bin/env python3
"""Write asymptotic plug-in PATP sandwich predictions for every AEPD DGP."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from aepd_patp_core import load_manifest, oracle_and_feasible_optima
    from patp_experiment_core import TRUE_THETA, signal_design, signal_gradient
    from patp_plugin_theory import fixed_design_plugin_sandwich
except ModuleNotFoundError:  # package-style execution
    from scripts.aepd_patp_core import load_manifest, oracle_and_feasible_optima
    from scripts.patp_experiment_core import TRUE_THETA, signal_design, signal_gradient
    from scripts.patp_plugin_theory import fixed_design_plugin_sandwich


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "patp_plugin_sandwich_predictions.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order in (2, 3):
        manifest = load_manifest(ROOT / f"configs/aepd_patp_s{order}.json")
        metadata_rows = read_csv(ROOT / f"results/aepd_patp_s{order}_summary.csv")
        largest_n = max(int(row["sample_size"]) for row in metadata_rows)
        t, envelope = signal_design(largest_n)
        gradient = signal_gradient(TRUE_THETA, t, envelope)
        q = gradient @ gradient.T / largest_n
        q_inv = np.linalg.inv(q)
        dbar = np.mean(gradient, axis=1)
        design_factor = float((q_inv @ dbar) @ (q_inv @ dbar) / np.trace(q_inv))
        for scenario in manifest.scenarios:
            _, _, rho, spec = oracle_and_feasible_optima(scenario, order)
            sandwich = fixed_design_plugin_sandwich(gradient, spec)
            nls_asymptotic_trace = spec.c2 * np.trace(q_inv)
            oracle_g = float(np.trace(sandwich.oracle) / nls_asymptotic_trace)
            plugin_g = float(np.trace(sandwich.feasible) / nls_asymptotic_trace)
            penalty_g = float(np.trace(sandwich.penalty) / nls_asymptotic_trace)
            empirical = next(
                row for row in metadata_rows
                if row["scenario_id"] == scenario.scenario_id
                and int(row["sample_size"]) == largest_n
                and row["method"] == "PATP Feasible fixed"
            )
            rows.append(
                {
                    "order": order,
                    "scenario_id": scenario.scenario_id,
                    "sample_size_for_design": largest_n,
                    "rho_feasible": rho,
                    "c2": spec.c2,
                    "sensitivity_kappa": spec.sensitivity,
                    "oracle_g_asymptotic": oracle_g,
                    "design_factor_R": design_factor,
                    "centering_penalty_g": penalty_g,
                    "plugin_g_asymptotic": plugin_g,
                    "identity_g_plus_one_minus_g_R": (
                        oracle_g + (1.0 - oracle_g) * design_factor
                    ),
                    "empirical_fixed_g_at_largest_n": float(empirical["trace_g"]),
                    "empirical_minus_asymptotic_plugin_g": (
                        float(empirical["trace_g"]) - plugin_g
                    ),
                }
            )
    return rows


def main() -> int:
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
