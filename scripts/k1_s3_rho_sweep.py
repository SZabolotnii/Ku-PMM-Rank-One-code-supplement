#!/usr/bin/env python3
"""Separate the two drivers of the S=3 feasible excess: conditioning of F vs
sampling noise of the fractional moments.

``k1_s3_gap_diagnosis.py`` established that the excess of the feasible PATP
one-step over its own theoretical g tracks the error of the estimator's own
plug-in g, and that the ill-conditioned rho = 0.3 cells carry it.  But the
naive rho = 0 cell has a *low* condition number and the *worst* plug-in g, so
conditioning cannot be the whole story.

This script sweeps rho on a fixed noise law and records, per rho:

  * the population condition number and theory_g (exact, no Monte Carlo);
  * the Monte Carlo spread of the plug-in g, i.e. how badly the estimator
    misjudges its own efficiency -- the direct measure of moment noise;
  * the empirical trace_g of the feasible one-step against the paired NLS fit.

The Gaussian cell is the clean instrument: there g = 1 for every rho, so the
target does not move with rho and the excess is attributable to the estimator
alone.  A non-Gaussian cell is run alongside as a control.

Reuses the project estimators unchanged (``build_empirical_spec``,
``one_step_estimate``, ``fit_bounded_nls``); nothing is reimplemented here.
Every replication shares one NLS fit across the whole rho grid, so the rho
comparison is paired on the same data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aepd_patp_core import (  # noqa: E402
    build_aepd_patp_spec,
    load_manifest,
    standardized_aepd_rvs,
    _scenario_probabilities,
)
from patp_adaptive_core import build_empirical_spec  # noqa: E402
from patp_experiment_core import (  # noqa: E402
    TRUE_THETA,
    fit_bounded_nls,
    one_step_estimate,
    signal,
    signal_design,
    signal_gradient,
)

DEFAULT_MANIFEST = ROOT / "configs" / "aepd_patp_s3.json"
OUT = ROOT / "results" / "k1_s3_rho_sweep.csv"
META = ROOT / "results" / "k1_s3_rho_sweep_metadata.json"

# rho = 0.5 is the exact PATP collision; the grid deliberately skips its band.
RHO_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)


def design_factor_RD(sample_size: int) -> float:
    """R_D = ||Q^{-1} dbar||^2 / tr Q^{-1} -- the design weight of the K1 rank-one
    increment.  Pure design quantity: no noise law enters."""
    t, envelope = signal_design(sample_size)
    grad = signal_gradient(TRUE_THETA, t, envelope)      # (p, n)
    d_bar = grad.mean(axis=1)
    Q = grad @ grad.T / grad.shape[1]
    Qi = np.linalg.inv(Q)
    return float((Qi @ d_bar) @ (Qi @ d_bar) / np.trace(Qi))


def trace_var(theta_rows: np.ndarray) -> float:
    return float(np.trace(np.cov(theta_rows, rowvar=False)))


def run(scenario, order, sample_size, reps, seed):
    from aepd_patp_core import scenario_shape_statistics
    gamma3 = scenario_shape_statistics(scenario, order).gamma3
    rng = np.random.default_rng(seed)
    t, envelope = signal_design(sample_size)
    clean = signal(TRUE_THETA, t, envelope)

    pop = {}
    for rho in RHO_GRID:
        try:
            spec = build_aepd_patp_spec(scenario, rho, order)
            pop[rho] = (float(spec.condition_number), float(spec.theory_g),
                        bool(spec.operational_fallback))
        except np.linalg.LinAlgError:
            pop[rho] = (float("nan"), float("nan"), True)

    nls_rows = np.empty((reps, TRUE_THETA.size))
    est = {rho: np.empty((reps, TRUE_THETA.size)) for rho in RHO_GRID}
    est_or = {rho: np.empty((reps, TRUE_THETA.size)) for rho in RHO_GRID}
    pop_spec = {}
    for rho in RHO_GRID:
        try:
            pop_spec[rho] = build_aepd_patp_spec(scenario, rho, order)
        except np.linalg.LinAlgError:
            pop_spec[rho] = None
    plug = {rho: np.full(reps, np.nan) for rho in RHO_GRID}
    cond_emp = {rho: np.full(reps, np.nan) for rho in RHO_GRID}
    fallbacks = {rho: 0 for rho in RHO_GRID}

    for rep in range(reps):
        probs = rng.random(sample_size)
        errors = standardized_aepd_rvs(
            sample_size, *scenario.aepd_shape,
            probabilities=_scenario_probabilities(scenario, probs))
        obs = clean + errors
        fit = fit_bounded_nls(obs, t, envelope)
        nls_rows[rep] = fit.theta
        residuals = obs - signal(fit.theta, t, envelope)
        for rho in RHO_GRID:
            ps = pop_spec[rho]
            if ps is None or ps.operational_fallback:
                est_or[rho][rep] = fit.theta
            else:
                th_or, _fb, _dm, _nm = one_step_estimate(
                    obs, fit.theta, t, envelope, ps)
                est_or[rho][rep] = th_or
            emp = build_empirical_spec(rho, order, residuals)
            if not emp.valid:
                est[rho][rep] = fit.theta
                fallbacks[rho] += 1
                continue
            plug[rho][rep] = float(emp.spec.theory_g)
            cond_emp[rho][rep] = float(emp.spec.condition_number)
            theta, fell_back, _damping, _norm = one_step_estimate(
                obs, fit.theta, t, envelope, emp.spec)
            est[rho][rep] = theta
            fallbacks[rho] += int(fell_back)

    base = trace_var(nls_rows)
    rd = design_factor_RD(sample_size)
    out = []
    for rho in RHO_GRID:
        cond_pop, g_pop, degenerate = pop[rho]
        g_emp = trace_var(est[rho]) / base
        g_emp_or = trace_var(est_or[rho]) / base
        pg = plug[rho][np.isfinite(plug[rho])]
        out.append({
            "scenario_id": scenario.scenario_id,
            "order": order,
            "sample_size": sample_size,
            "reps": reps,
            "rho": rho,
            "cond_F_population": cond_pop,
            "theory_g_population": g_pop,
            "population_degenerate": int(degenerate),
            "cond_F_empirical_mean": float(np.nanmean(cond_emp[rho])),
            "plugin_g_mean": float(pg.mean()) if pg.size else float("nan"),
            "plugin_g_sd": float(pg.std(ddof=1)) if pg.size > 1 else float("nan"),
            "plugin_g_bias": (float(pg.mean() - g_pop) if pg.size else float("nan")),
            "trace_g_empirical": g_emp,
            "R_D": rd,
            "trace_g_oracle_empirical": g_emp_or,
            "theory_g_feasible": g_pop + (1.0 - g_pop) * rd,
            "predicted_increment": (1.0 - g_pop) * rd,
            "measured_increment": g_emp - g_emp_or,
            "increment_gap": (g_emp - g_emp_or) - (1.0 - g_pop) * rd,
            "excess_over_theory": g_emp - g_pop,
            "excess_over_feasible_theory": g_emp - (g_pop + (1.0 - g_pop) * rd),
            "fallback_rate": fallbacks[rho] / reps,
            "gamma3": float(gamma3),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--sample-sizes", default="500,2000,8000")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--scenarios", default="S200,S100")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-prefix", default="k1_s3_rho_sweep")
    ap.add_argument("--allow-asymmetric-order3", action="store_true",
                    help="diagnostics only: PATP path is valid for asymmetric "
                         "laws, the classical PMM3 comparator is not")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_manifest(
        manifest_path,
        allow_asymmetric_order3=args.allow_asymmetric_order3)
    out_csv = ROOT / "results" / f"{args.out_prefix}.csv"
    out_meta = ROOT / "results" / f"{args.out_prefix}_metadata.json"
    wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    chosen = [s for s in manifest.scenarios if s.scenario_id in wanted]
    if not chosen:
        raise SystemExit(f"no scenarios matched {wanted}")

    sizes = [int(v) for v in args.sample_sizes.split(",") if v.strip()]
    rows = []
    for scenario in chosen:
      for n in sizes:
        print(f"== {scenario.scenario_id} (tau={scenario.tau}, "
              f"p=({scenario.p_left},{scenario.p_right})), "
              f"n={n}, reps={args.reps}")
        block = run(scenario, manifest.order, n, args.reps, args.seed)
        rows.extend(block)
        print("R_D = %.4f  (predicted g_fe = g_or + (1-g_or)*R_D)" % block[0]["R_D"])
        hdr = ("%5s %9s %9s %9s %11s %11s %10s"
               % ("rho", "g_or(th)", "g_or(MC)", "g_fe(MC)", "прир.(MC)",
                  "прир.(K1)", "розрив"))
        print(hdr)
        for r in block:
            print("%5.2f %9.4f %9.4f %9.4f %+11.4f %+11.4f %+10.4f"
                  % (r["rho"], r["theory_g_population"],
                     r["trace_g_oracle_empirical"], r["trace_g_empirical"],
                     r["measured_increment"], r["predicted_increment"],
                     r["increment_gap"]))
        print()

    # which candidate driver does the excess follow?
    print("== correlation of the excess with each candidate driver ==")
    print("%-8s %14s %14s %14s" % ("scenario", "log10 cond(F)", "sd(plug g)",
                                   "|bias(plug g)|"))
    for sid, n in [(a, b) for a in wanted for b in sizes]:
        blk = [r for r in rows if r["scenario_id"] == sid
               and r["sample_size"] == n]
        if len(blk) < 3:
            continue
        ex = np.array([r["increment_gap"] for r in blk])
        drivers = {
            "log10 cond(F)": np.log10(np.array([r["cond_F_population"] for r in blk])),
            "sd(plug g)": np.array([r["plugin_g_sd"] for r in blk]),
            "|bias(plug g)|": np.abs(np.array([r["plugin_g_bias"] for r in blk])),
        }
        cells = []
        for key in ("log10 cond(F)", "sd(plug g)", "|bias(plug g)|"):
            d = drivers[key]
            ok = np.isfinite(d) & np.isfinite(ex)
            cells.append(float(np.corrcoef(d[ok], ex[ok])[0, 1]) if ok.sum() > 2
                         else float("nan"))
        print("%-8s %14.3f %14.3f %14.3f" % (f"{sid}/{n}", *cells))
        peak_ex = blk[int(np.nanargmax(ex))]["rho"]
        peak_cond = blk[int(np.nanargmax(drivers["log10 cond(F)"]))]["rho"]
        peak_sd = blk[int(np.nanargmax(drivers["sd(plug g)"]))]["rho"]
        print("         peak rho: excess=%.2f  cond=%.2f  sd(plug g)=%.2f"
              % (peak_ex, peak_cond, peak_sd))
    print()

    fields = list(rows[0])
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    out_meta.write_text(json.dumps({
        "script": "scripts/k1_s3_rho_sweep.py",
        "manifest": str(manifest_path),
        "rho_grid": list(RHO_GRID),
        "reps": args.reps,
        "sample_sizes": sizes,
        "seed": args.seed,
        "scenarios": wanted,
        "note": "diagnostic sweep, not a publication run",
    }, indent=2), encoding="utf-8")
    print("written:", out_csv.relative_to(ROOT))


if __name__ == "__main__":
    main()
