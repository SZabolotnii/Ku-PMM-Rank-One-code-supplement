#!/usr/bin/env python3
"""K1 fixed-rho direction panel: is the rank-one increment a property of every
fixed basis parameter, or only of the population-optimal one used in the L2 run?

For each fixed rho on the sweep grid (the collision band around rho = 1/2 is
skipped), the feasible and oracle PATP one-steps are computed on the SAME
replications (one NLS fit shared across the grid), and their empirical
covariances are projected onto u = Q^{-1}dbar/|Q^{-1}dbar| and onto an
orthonormal basis of its complement -- the direction test of
k1_penalty_direction.py, repeated per rho.  Theory (fixed rho):

    Var(u'theta_feas)/Var(u'theta_or) -> ratio from fixed_design_plugin_sandwich
    Var(v'theta_feas)/Var(v'theta_or) -> 1  for v _|_ Q^{-1}dbar

A paired bootstrap over replications gives a 95% interval for the along-u
ratio.  Reuses the project estimators unchanged.
Output: results/k1_rho_direction_panel.csv (+ _metadata.json)
"""

from __future__ import annotations

import argparse
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
)
from patp_plugin_theory import fixed_design_plugin_sandwich  # noqa: E402
from k1_common import design_quantities, write_csv, write_metadata  # noqa: E402
from k1_penalty_direction import complement_basis  # noqa: E402

RHO_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)
ORDER = 2


def ratio_u(fe, orr, u):
    return float(np.var(fe @ u, ddof=1) / np.var(orr @ u, ddof=1))


def run(scenario, n, reps, seed, n_boot):
    rng = np.random.default_rng(seed)
    t, envelope = signal_design(n)
    clean = signal(TRUE_THETA, t, envelope)
    gradient, _q, _qi, _dbar, direction, r_d = design_quantities(n)
    u = direction / np.linalg.norm(direction)
    vb = complement_basis(u)
    specs = {}
    for rho in RHO_GRID:
        try:
            s = build_aepd_patp_spec(scenario, rho, ORDER)
            specs[rho] = None if s.operational_fallback else s
        except np.linalg.LinAlgError:
            specs[rho] = None
    fe = {r: np.empty((reps, TRUE_THETA.size)) for r in RHO_GRID}
    orr = {r: np.empty((reps, TRUE_THETA.size)) for r in RHO_GRID}
    fb = {r: 0 for r in RHO_GRID}
    for rep in range(reps):
        probs = rng.random(n)
        errors = standardized_aepd_rvs(n, *scenario.aepd_shape,
                                       probabilities=_scenario_probabilities(scenario, probs))
        obs = clean + errors
        fit = fit_bounded_nls(obs, t, envelope)
        res = obs - signal(fit.theta, t, envelope)
        for rho in RHO_GRID:
            ps = specs[rho]
            if ps is None:
                continue
            orr[rho][rep] = one_step_estimate(obs, fit.theta, t, envelope, ps)[0]
            emp = build_empirical_spec(rho, ORDER, res)
            if not emp.valid:
                fe[rho][rep] = fit.theta
                fb[rho] += 1
                continue
            th, fell, _d, _nm = one_step_estimate(obs, fit.theta, t, envelope, emp.spec)
            fe[rho][rep] = th
            fb[rho] += int(fell)
    boot_rng = np.random.default_rng(seed + 1)
    idx = boot_rng.integers(0, reps, size=(n_boot, reps))
    out = []
    for rho in RHO_GRID:
        ps = specs[rho]
        if ps is None:
            continue
        sw = fixed_design_plugin_sandwich(gradient, ps)
        pred_u = float((u @ sw.feasible @ u) / (u @ sw.oracle @ u))
        emp_u = ratio_u(fe[rho], orr[rho], u)
        boots = np.array([ratio_u(fe[rho][i], orr[rho][i], u) for i in idx])
        cf = np.cov(fe[rho], rowvar=False)
        co = np.cov(orr[rho], rowvar=False)
        emp_v = float(np.mean([(v @ cf @ v) / (v @ co @ v) for v in vb.T]))
        eig = np.linalg.eigvalsh((cf - co + (cf - co).T) / 2.0) * n
        out.append({
            "scenario_id": scenario.scenario_id, "order": ORDER, "sample_size": n,
            "reps": reps, "rho": rho, "theory_g_population": float(ps.theory_g),
            "R_D": r_d, "predicted_ratio_along_u": pred_u,
            "empirical_ratio_along_u": emp_u,
            "ratio_u_ci_low": float(np.quantile(boots, 0.025)),
            "ratio_u_ci_high": float(np.quantile(boots, 0.975)),
            "pred_inside_ci": int(np.quantile(boots, 0.025) <= pred_u <= np.quantile(boots, 0.975)),
            "one_inside_ci": int(np.quantile(boots, 0.025) <= 1.0 <= np.quantile(boots, 0.975)),
            "empirical_ratio_complement": emp_v,
            "excess_eig_max": float(eig[-1]), "excess_eig_second": float(eig[-2]),
            "excess_eig_min": float(eig[0]),
            "excess_second_over_max": float(abs(eig[-2]) / abs(eig[-1])) if eig[-1] > 0 else float("nan"),
            "fallback_rate": fb[rho] / reps,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--sample-sizes", default="5000")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--scenarios", default="A2L,A2R,A4L,A4R")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out-prefix", default="k1_rho_direction_panel")
    a = ap.parse_args()
    manifest = load_manifest(ROOT / "configs" / "aepd_patp_s2.json")
    wanted = a.scenarios.split(",")
    scen = [s for s in manifest.scenarios if s.scenario_id in wanted]
    rows = []
    for s in scen:
        for n in [int(x) for x in a.sample_sizes.split(",")]:
            rows += run(s, n, a.reps, a.seed, a.bootstrap)
    write_csv(ROOT / "results" / f"{a.out_prefix}.csv", rows)
    write_metadata(ROOT / "results" / f"{a.out_prefix}_metadata.json", {
        "script": "scripts/k1_rho_direction_panel.py", "manifest": "configs/aepd_patp_s2.json",
        "rho_grid": list(RHO_GRID), "reps": a.reps, "sample_sizes": a.sample_sizes,
        "seed": a.seed, "bootstrap": a.bootstrap, "scenarios": wanted,
        "claim": "for every fixed rho the feasible/oracle variance ratio exceeds 1 only along Q^{-1}dbar",
    })
    for r in rows:
        print(f"{r['scenario_id']:4s} n={r['sample_size']} rho={r['rho']:.1f} u: pred {r['predicted_ratio_along_u']:.4f} "
              f"emp {r['empirical_ratio_along_u']:.4f} [{r['ratio_u_ci_low']:.4f}, {r['ratio_u_ci_high']:.4f}]  "
              f"v: {r['empirical_ratio_complement']:.4f}  eig2/eig1={r['excess_second_over_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
