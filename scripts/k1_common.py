"""Shared helpers for the K1 article checks (stage C of the K1 pipeline).

Every K1 script reads existing L2 artifacts (``results/aepd_patp_s{2,3}_*``)
or draws fresh pilot samples; numbers enter the K1 manuscript only through the
CSV files written under ``results/k1_*``.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from aepd_patp_core import (  # noqa: E402
    AEPDScenario,
    build_aepd_patp_spec,
    load_manifest,
    oracle_and_feasible_optima,
)
from patp_experiment_core import (  # noqa: E402
    TRUE_THETA,
    signal_design,
    signal_gradient,
)
from patp_plugin_theory import fixed_design_plugin_sandwich  # noqa: E402

PARAMS = ("A", "Omega", "Phi")


def design_quantities(sample_size: int, *, t=None, envelope=None):
    """Gradient at the truth, Q, Q^{-1}, dbar, and the design factor R_D."""
    if t is None or envelope is None:
        t, envelope = signal_design(sample_size)
    gradient = signal_gradient(TRUE_THETA, t, envelope)
    q = gradient @ gradient.T / gradient.shape[1]
    q_inv = np.linalg.inv(q)
    dbar = np.mean(gradient, axis=1)
    direction = q_inv @ dbar
    r_d = float(direction @ direction / np.trace(q_inv))
    return gradient, q, q_inv, dbar, direction, r_d


def load_replications(order: int) -> list[dict[str, str]]:
    path = ROOT / f"results/aepd_patp_s{order}_replications.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} is not versioned; regenerate it with "
            f"scripts/run_mc_aepd_patp_s{order}.py (publication mode)"
        )
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_summary(order: int) -> list[dict[str, str]]:
    with (ROOT / f"results/aepd_patp_s{order}_summary.csv").open() as handle:
        return list(csv.DictReader(handle))


def scenarios(order: int) -> list[AEPDScenario]:
    return list(load_manifest(ROOT / f"configs/aepd_patp_s{order}.json").scenarios)


def feasible_spec_for(scenario: AEPDScenario, order: int):
    _, _, rho, spec = oracle_and_feasible_optima(scenario, order)
    return rho, spec


def theta_matrix(rows, *, method: str, scenario_id: str, sample_size: int, rho=None):
    out = []
    for row in rows:
        if row["method"] != method or row["scenario_id"] != scenario_id:
            continue
        if int(row["sample_size"]) != sample_size:
            continue
        if rho is not None and row["rho"] != "" and abs(float(row["rho"]) - rho) > 1e-12:
            continue
        if row["converged"] != "1":
            continue
        out.append([float(row[p]) for p in PARAMS])
    return np.asarray(out, dtype=float)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


__all__ = [
    "AEPDScenario", "PARAMS", "ROOT", "TRUE_THETA", "build_aepd_patp_spec",
    "design_quantities", "feasible_spec_for", "fixed_design_plugin_sandwich",
    "load_replications", "load_summary", "scenarios", "signal_design",
    "signal_gradient", "theta_matrix", "write_csv", "write_metadata",
]
