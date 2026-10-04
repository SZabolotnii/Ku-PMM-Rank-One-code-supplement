#!/usr/bin/env python3
"""Diagnose the S=3 deviation reported in the K1 manuscript (Table tab:s3).

Read-only analysis of the frozen publication summary
``results/aepd_patp_s3_summary.csv``. It runs no Monte Carlo and writes no new
evidence; it regenerates three tables that separate the candidate explanations
the manuscript lists in Section "Відхилення при S=3":

  (a) the one-step-with-damping artefact,
  (b) the finite-sample cost of estimating the weights,
  (c) the cost of choosing rho near basis degeneracy.

Output: results/k1_s3_gap_diagnosis.csv plus a printed summary.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "aepd_patp_s3_summary.csv"
OUT = ROOT / "results" / "k1_s3_gap_diagnosis.csv"

SCEN = ["S075", "S100", "S150", "S200", "S300", "S500"]
METHODS = ["PATP Oracle one-step", "PATP Feasible fixed", "PATP Naive rho=0",
           "PATP Adaptive", "Classical PMM3"]


def num(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", "nan", None) else float("nan")


def load():
    with SRC.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def table_a(rows):
    """Is the one-step / damping the source?  Look at S200, where g_or = 1 and
    the rank-one increment is exactly zero, so the theorem makes no non-trivial
    prediction and any excess is pure estimator artefact."""
    print("== A. S200 (гаусів нейтральний, g_or = 1 точно): чи винен один крок? ==")
    print("%-22s %6s %10s %10s %11s %9s" % ("метод", "n", "trace_g", "g-1",
                                            "cond(F)", "damping"))
    out = []
    for m in ("PATP Oracle one-step", "PATP Feasible fixed", "Classical PMM3"):
        for n in ("200", "1000", "5000"):
            for r in rows:
                if (r["scenario_id"] == "S200" and r["sample_size"] == n
                        and r["method"] == m):
                    g = num(r, "trace_g")
                    print("%-22s %6s %10.5f %+10.5f %11.2e %9s"
                          % (m, n, g, g - 1.0, num(r, "condition_number_mean"),
                             r["mean_damping"]))
                    out.append({"table": "A", "method": m, "scenario_id": "S200",
                                "sample_size": n, "trace_g": g, "excess": g - 1.0,
                                "cond_F": num(r, "condition_number_mean"),
                                "mean_damping": r["mean_damping"],
                                "fallback_rate": r["fallback_rate"]})
                    break
        print()
    return out


def table_b(rows):
    """Does the gap track the error of the estimator's own weight estimate?"""
    print("== B. n = 5000: розрив проти похибки самооцінки ваг ==")
    print("%-22s %6s %5s %9s %9s %10s %+9s %10s"
          % ("метод", "сцен.", "rho", "theory_g", "plugin_g", "|plug-th|",
             "розрив", "cond(F)"))
    out = []
    for m in METHODS:
        for s in SCEN:
            for r in rows:
                if (r["sample_size"] == "5000" and r["method"] == m
                        and r["scenario_id"] == s):
                    th, pg, eg = (num(r, "theory_g"), num(r, "plugin_g_mean"),
                                  num(r, "trace_g"))
                    print("%-22s %6s %5s %9.4f %9.4f %10.4f %+9.4f %10.2e"
                          % (m, s, r["rho_feasible"], th, pg, abs(pg - th),
                             eg - th, num(r, "condition_number_mean")))
                    out.append({"table": "B", "method": m, "scenario_id": s,
                                "sample_size": "5000", "rho_feasible": r["rho_feasible"],
                                "theory_g": th, "plugin_g_mean": pg,
                                "abs_plugin_error": abs(pg - th), "trace_g": eg,
                                "excess": eg - th,
                                "cond_F": num(r, "condition_number_mean"),
                                "mean_damping": r["mean_damping"],
                                "fallback_rate": r["fallback_rate"]})
                    break
        print()
    return out


def table_c(rows):
    """Split the feasible gap into an oracle-level part (one step from NLS) and
    a plug-in part on top of it."""
    print("== C. n = 5000: розклад розриву на оракульну й plug-in частини ==")
    print("%-7s %5s %11s %11s %12s" % ("сцен.", "rho", "оракульна", "здійсненна",
                                       "plug-in понад"))
    out = []
    for s in SCEN:
        o = f = None
        for r in rows:
            if r["sample_size"] == "5000" and r["scenario_id"] == s:
                if r["method"] == "PATP Oracle one-step":
                    o = r
                elif r["method"] == "PATP Feasible fixed":
                    f = r
        if not (o and f):
            continue
        go = num(o, "trace_g") - num(o, "theory_g")
        gf = num(f, "trace_g") - num(f, "theory_g")
        print("%-7s %5s %+11.4f %+11.4f %+12.4f"
              % (s, f["rho_feasible"], go, gf, gf - go))
        out.append({"table": "C", "scenario_id": s, "sample_size": "5000",
                    "rho_feasible": f["rho_feasible"], "oracle_excess": go,
                    "feasible_excess": gf, "plugin_excess": gf - go})
    print()
    return out


def main():
    rows = load()
    recs = table_a(rows) + table_b(rows) + table_c(rows)
    fields = sorted({k for r in recs for k in r})
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(recs)
    print("Записано:", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
