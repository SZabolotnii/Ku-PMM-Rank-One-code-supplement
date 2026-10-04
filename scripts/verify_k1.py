#!/usr/bin/env python3
"""Verification gate for the rank-one variance inflation article.

Reads the stored artifacts under ``results/`` and checks the claims the
manuscript makes from them; it does not re-run any Monte Carlo.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def k1_checks() -> list[Check]:
    """Stage-C artifacts of article K1 (results/k1_*): rank-one direction, R_D identity."""
    import csv as _csv

    checks: list[Check] = []
    required = [
        ROOT / "results/k1_design_factor_RD.csv",
        ROOT / "results/k1_design_factor_RD_designs.csv",
        ROOT / "results/k1_penalty_direction.csv",
        ROOT / "results/k1_convergence_rate.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    checks.append(
        Check(
            "K1 stage-C artifacts exist",
            not missing,
            "all present" if not missing else "missing=" + ", ".join(missing),
        )
    )
    if missing:
        return checks
    with (ROOT / "results/k1_design_factor_RD.csv").open() as handle:
        gaps = [
            float(row["identity_gap"])
            for row in _csv.DictReader(handle)
            if row["identity_gap"] != "nan"
        ]
    checks.append(
        Check(
            "K1: penalty_g/(1-oracle_g) equals R_D for every noise law and design",
            len(gaps) >= 100 and max(gaps) < 1e-10,
            f"cells={len(gaps)}, max gap={max(gaps):.2e}",
        )
    )
    with (ROOT / "results/k1_penalty_direction.csv").open() as handle:
        rows = [
            row for row in _csv.DictReader(handle)
            if row["sample_size"] == "5000"
            and row["empirical_ratio_complement_vs_oracle_empirical"] != "nan"
        ]
    complement_dev = max(
        abs(float(row["empirical_ratio_complement_vs_oracle_empirical"]) - 1.0)
        for row in rows
    )
    material = [row for row in rows if float(row["predicted_ratio_along_u"]) > 1.03]
    along_ok = all(
        float(row["empirical_ratio_along_u_vs_oracle_empirical"]) > 1.0
        and float(row["excess_second_over_max"]) < 0.05
        for row in material
    )
    checks.append(
        Check(
            "K1: plug-in penalty is confined to the direction Q^{-1} dbar at N=5000",
            len(rows) >= 4 and complement_dev < 0.02 and len(material) >= 2 and along_ok,
            f"complement max |ratio-1|={complement_dev:.3f}; material scenarios={len(material)}, "
            f"rank-one and positive along u={along_ok}",
        )
    )
    checks.extend(_k1_symmetric_coverage_checks(_csv))
    checks.extend(_k1_rho_panel_and_rank_contrast_checks(_csv))
    checks.extend(_k1_signal_class_checks(_csv))
    checks.extend(_k1_constant_signal_checks(_csv))
    return checks


def _k1_constant_signal_checks(_csv) -> list[Check]:
    """Constant signal: the gain is lost under estimated centring for any S.

    With every centring mean estimated from the same residuals the refinement step is
    exactly zero (S = 2 asymmetric and S = 3 symmetric alike); with centring imposed a
    priori the empirical g approaches its population value.  Negative control: the
    ``imposed`` column of the same artifact, which must stay well below one.
    """
    path = ROOT / "results/k1_constant_signal_check.csv"
    if not path.exists():
        return [Check("K1: constant-signal artifact exists", False,
                      "run scripts/k1_constant_signal_check.py")]
    with path.open() as handle:
        rows = list(_csv.DictReader(handle))
    step = max(float(r["max_abs_step_estimated_centring"]) for r in rows)
    largest = [r for r in rows if r["sample_size"] == "5000"]
    imposed_gap = max(
        abs(float(r["g_imposed_centring"]) - float(r["population_g"])) for r in largest
    )
    orders = sorted({r["order"] for r in rows})
    return [
        Check(
            "K1: constant signal -- estimated centring returns the sample mean for S=2 and S=3",
            orders == ["2", "3"] and step < 1e-12
            and all(abs(float(r["g_estimated_centring"]) - 1.0) < 1e-12 for r in rows),
            f"orders={orders}; max |step|={step:.1e} over {len(rows)} cells",
        ),
        Check(
            "K1: constant signal -- imposed centring keeps the gain",
            len(largest) == 2 and imposed_gap < 0.02,
            "; ".join(
                f"{r['case']} g={float(r['g_imposed_centring']):.4f} "
                f"(population {float(r['population_g']):.4f})" for r in largest
            ),
        ),
    ]


def _k1_signal_class_checks(_csv) -> list[Check]:
    """General signal model (tab:classes, rem:projection, cor:constant).

    Deterministic design quantities for signal classes other than the harmonic bench:
    Q^{-1} dbar is the least-squares projection of the constant onto the gradient span,
    a constant component absorbs the whole direction, and penalty_g/(1-oracle_g) equals
    R_D for every noise law.  Negative control: a class whose gradient is rescaled
    changes R_D but not the projection norm (tests/test_signal_classes.py).
    """
    classes = ROOT / "results/k1_signal_classes_RD.csv"
    cells = ROOT / "results/k1_signal_classes_RD_cells.csv"
    if not classes.exists() or not cells.exists():
        return [Check("K1: signal-class design artifacts exist", False,
                      "run scripts/k1_signal_classes_RD.py")]
    with classes.open() as handle:
        rows = list(_csv.DictReader(handle))
    with cells.open() as handle:
        cell_rows = list(_csv.DictReader(handle))
    lstsq = max(float(r["lstsq_gap"]) for r in rows)
    pythagoras = max(
        abs(float(r["projection_norm_sq"]) + float(r["constant_residual_mean_square"]) - 1.0)
        for r in rows
    )
    in_range = all(-1e-12 <= float(r["projection_norm_sq"]) <= 1.0 + 1e-12 for r in rows)
    with_constant = [r for r in rows if r["constant_component_gap"] != "nan"]
    constant_gap = max(float(r["constant_component_gap"]) for r in with_constant)
    constant_pi = max(abs(float(r["projection_norm_sq"]) - 1.0) for r in with_constant)
    bench = [
        float(r["design_factor_R"]) for r in rows
        if r["signal_class"] == "harmonic_envelope" and r["sample_size"] == "5000"
    ]
    gaps = [float(r["identity_gap"]) for r in cell_rows if r["identity_gap"] != "nan"]
    min_eig = min(float(r["penalty_min_eig"]) for r in cell_rows)
    max_rank = max(int(r["penalty_rank"]) for r in cell_rows)
    n_classes = len({r["signal_class"] for r in rows})
    return [
        Check(
            "K1: Q^{-1} dbar is the projection of the constant onto the gradient span",
            n_classes >= 6 and lstsq < 1e-10 and pythagoras < 1e-10 and in_range,
            f"classes={n_classes}; max |Q^-1 dbar - lstsq|={lstsq:.1e}; "
            f"max |pi + residual - 1|={pythagoras:.1e}",
        ),
        Check(
            "K1: a constant component absorbs the whole increment direction",
            len(with_constant) >= 3 and constant_gap < 1e-10 and constant_pi < 1e-10,
            f"rows={len(with_constant)}; max |Q^-1 dbar - e_C|={constant_gap:.1e}; "
            f"max |pi - 1|={constant_pi:.1e}",
        ),
        Check(
            "K1: penalty_g/(1-oracle_g) equals R_D for every signal class and noise law",
            len(gaps) >= 200 and max(gaps) < 1e-10 and min_eig > -1e-12 and max_rank <= 1
            and len(bench) == 1 and abs(bench[0] - 0.219077) < 1e-5,
            f"cells={len(gaps)}, max gap={max(gaps):.2e}, min eig={min_eig:.1e}, "
            f"max rank={max_rank}, harmonic R_D={bench[0]:.6f}",
        ),
    ]


def _k1_rho_panel_and_rank_contrast_checks(_csv) -> list[Check]:
    """Fixed-rho direction panel (tab:rho-panel) and the rank contrast (rem:rank).

    Panel: for every fixed rho off the collision band the along-u ratio covers its
    prediction and excludes one, the complement stays at one.  Contrast: stacked
    separable conditions keep the penalty rank one; mixed instruments break it and
    make the excess indefinite (negative second eigenvalue).  Negative control: if
    the contrast artifact is regenerated with structure C replaced by B, the second
    check fails because eig2/eig1 is no longer below -0.03.
    """
    out: list[Check] = []
    panel = ROOT / "results/k1_rho_direction_panel.csv"
    if panel.exists():
        with panel.open() as handle:
            rows = list(_csv.DictReader(handle))
        covered = sum(int(r["pred_inside_ci"]) for r in rows)
        excludes_one = sum(1 - int(r["one_inside_ci"]) for r in rows)
        comp = max(abs(float(r["empirical_ratio_complement"]) - 1.0) for r in rows)
        out.append(Check(
            "K1: fixed-rho panel -- increment along Q^{-1} dbar at every rho, none on the complement",
            len(rows) >= 20 and covered == len(rows) and excludes_one == len(rows) and comp < 0.02,
            f"cells={len(rows)}; prediction inside CI {covered}/{len(rows)}; "
            f"one outside CI {excludes_one}/{len(rows)}; complement max |ratio-1|={comp:.4f}",
        ))
    else:
        out.append(Check("K1: fixed-rho panel artifact exists", False,
                         "run scripts/k1_rho_direction_panel.py"))
    contrast = ROOT / "results/k1_rank_contrast.csv"
    if contrast.exists():
        with contrast.open() as handle:
            rows = list(_csv.DictReader(handle))
        sep = [abs(float(r["eig2_over_eig1"])) for r in rows if r["structure"] != "C_mixed_instruments"]
        mixed = [float(r["eig2_over_eig1"]) for r in rows if r["structure"] == "C_mixed_instruments"]
        out.append(Check(
            "K1: rank contrast -- separable stacking stays rank one, mixed instruments do not",
            len(sep) >= 4 and len(mixed) >= 2 and max(sep) < 0.01 and max(mixed) < -0.03,
            f"separable max |eig2/eig1|={max(sep) if sep else float('nan'):.4f}; "
            f"mixed eig2/eig1={', '.join(f'{m:.3f}' for m in mixed)}",
        ))
    else:
        out.append(Check("K1: rank-contrast artifact exists", False,
                         "run scripts/k1_rank_contrast.py"))
    return out


def _k1_symmetric_coverage_checks(_csv) -> list[Check]:
    """Symmetric S=3 coverage: the centring penalty is real under symmetric noise.

    The external replication of 2026-09-21 proposed a fifth oracle-equivalence
    condition -- symmetry of the noise -- transplanted from a mixture model where
    parity kills the cross-Jacobian.  It does not transplant: in this design the
    penalty is ``(1 - g_or) * R_D``, a noise factor times a design factor, with no
    slot for noise parity.  This check is the Monte Carlo evidence: on the fully
    symmetric ``S*`` manifest the naive oracle form undercovers, by an amount that
    tracks the predicted penalty, while the closed form (12.1) holds nominal 95%.
    """
    path = ROOT / "results/k1_sandwich_coverage_s3.csv"
    if not path.exists():
        return [
            Check(
                "K1: symmetric S=3 coverage artifact exists",
                False,
                "run k1_sandwich_coverage.py --order 3 --publication",
            )
        ]
    with path.open() as handle:
        rows = [row for row in _csv.DictReader(handle) if row["sample_size"] == "5000"]
    with (ROOT / "results/patp_plugin_sandwich_predictions.csv").open() as handle:
        penalty = {
            row["scenario_id"]: float(row["centering_penalty_g"])
            for row in _csv.DictReader(handle)
            if row["order"] == "3"
        }
    by_scenario: dict[str, dict[str, float]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_id"], {})[row["estimator"]] = float(
            row["coverage_joint_chi2"]
        )
    cells = {
        scenario_id: (penalty[scenario_id], covers["closed"], covers["naive"])
        for scenario_id, covers in by_scenario.items()
        if scenario_id in penalty and {"closed", "naive"} <= covers.keys()
    }
    if len(cells) < 4:
        return [
            Check(
                "K1: symmetric S=3 coverage artifact exists",
                False,
                f"only {len(cells)} usable scenarios at N=5000; rerun in publication mode",
            )
        ]
    closed_ok = all(0.92 <= closed <= 0.98 for _, closed, _ in cells.values())
    worst = max(cells.values())  # largest predicted penalty
    best = min(cells.values())  # smallest predicted penalty
    pens = [cell[0] for cell in cells.values()]
    gaps = [cell[1] - cell[2] for cell in cells.values()]
    mean_p = sum(pens) / len(pens)
    mean_g = sum(gaps) / len(gaps)
    cov_pg = sum((p - mean_p) * (g - mean_g) for p, g in zip(pens, gaps))
    var_p = sum((p - mean_p) ** 2 for p in pens)
    var_g = sum((g - mean_g) ** 2 for g in gaps)
    corr = cov_pg / (var_p * var_g) ** 0.5 if var_p > 0 and var_g > 0 else 0.0
    return [
        Check(
            "K1: closed form holds nominal 95% under symmetric noise at N=5000",
            closed_ok and all(closed >= naive - 0.01 for _, closed, naive in cells.values()),
            "closed coverage="
            + ", ".join(
                f"{scenario_id} {closed:.3f}"
                for scenario_id, (_, closed, _) in sorted(cells.items())
            ),
        ),
        Check(
            "K1: symmetry is not an oracle-equivalence condition -- naive deficit "
            "tracks the predicted penalty",
            corr >= 0.9 and (worst[1] - worst[2]) >= 0.05 and (best[1] - best[2]) <= 0.02,
            f"corr(penalty, closed-naive)={corr:.3f} over {len(cells)} symmetric scenarios; "
            f"deficit {worst[1] - worst[2]:.3f} at penalty {worst[0]:.4f} vs "
            f"{best[1] - best[2]:.3f} at penalty {best[0]:.4f}",
        ),
    ]


def main() -> int:
    checks = k1_checks()
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    failed = sum(not check.passed for check in checks)
    print(f"\nSummary: {len(checks) - failed}/{len(checks)} checks passed; {failed} blockers remain.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
