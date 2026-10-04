#!/usr/bin/env python3
"""Check every headline number printed in the manuscript against the stored results.

Expected values are transcribed from the manuscript at the precision it prints them
(decimal comma written here as a dot). Actual values are recomputed from results/.
A check passes when the recomputed value rounds to the printed one, i.e. lies within
half a unit of the last printed digit. A mismatch means the manuscript and the
results disagree; it is never a reason to edit the expected value.

No simulation, no network:  python scripts/verify_reported_values.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CHECKS: list[tuple[str, str, bool, str]] = []


def rows(name):
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def _decimals(text):
    return len(text.split(".")[1]) if "." in text else 0


def num(section, label, printed, actual):
    """printed: the value as the manuscript prints it, e.g. '0.2191'."""
    tol = 0.5 * 10 ** (-_decimals(printed)) + 1e-12
    ok = math.isfinite(actual) and abs(actual - float(printed)) <= tol
    CHECKS.append((section, label, ok, f"printed {printed}, recomputed {actual:.6g}"))


def rng(section, label, lo, hi, values):
    values = list(values)
    num(section, label + " (lower end)", lo, min(values))
    num(section, label + " (upper end)", hi, max(values))


def sci(section, label, mantissa, exponent, actual):
    """A value printed as mantissa * 10^exponent, e.g. 1.9e-14."""
    num(section, label, mantissa, actual / 10 ** exponent)


def fact(section, label, ok, detail):
    CHECKS.append((section, label, bool(ok), detail))


def f(x):
    return float(x)


# --------------------------------------------------------------------------- §6: R_D
def design_weight():
    names = {
        "track_window_[0.01,2]_env0.3": ("0.2243", "0.2200", "0.2191"),
        "no_envelope_[0.01,2]": ("0.2374", "0.2344", "0.2338"),
        "strong_envelope_[0.01,2]_env0.6": ("0.2208", "0.2152", "0.2141"),
        "short_window_[0.01,1]_env0.3": ("0.0119", "0.0118", "0.0117"),
        "long_window_[0.01,4]_env0.3": ("0.3386", "0.3432", "0.3441"),
        "long_window_[0.01,8]_env0.3": ("0.0739", "0.0738", "0.0738"),
        "shifted_window_[2,4]_env0.3": ("0.0990", "0.0964", "0.0959"),
    }
    by = {(r["design"], int(r["sample_size"])): f(r["design_factor_R"])
          for r in rows("k1_design_factor_RD_designs.csv")}
    for design, printed in names.items():
        for n, p in zip((200, 1000, 5000), printed):
            num("tab:rd", f"R_D {design}, n={n}", p, by[(design, n)])
    rng("sec:g", "R_D range over designs, n=5000", "0.0117", "0.3441",
        [v for (d, n), v in by.items() if n == 5000])
    num("sec:g", "share of oracle gain kept (1 - R_D), %", "78.1", 100 * (1 - by[("track_window_[0.01,2]_env0.3", 5000)]))

    cells = rows("k1_design_factor_RD.csv")
    gaps = [f(r["identity_gap"]) for r in cells if r["identity_gap"] != "nan"]
    fact("sec:rd-num", "identity cells: 273 defined of 315", len(gaps) == 273 and len(cells) == 315,
         f"{len(gaps)} of {len(cells)}")
    sci("sec:rd-num", "max identity deviation", "1.9", -14, max(gaps))
    fact("sec:rd-num", "grid 7 designs x 3 sizes x 15 noise laws",
         len({r["design"] for r in cells}) == 7 and len({r["sample_size"] for r in cells}) == 3
         and len({r["scenario_id"] for r in cells}) == 15, "7 x 3 x 15")


def signal_classes():
    printed = {
        "harmonic_envelope": ("3", "0.2191", "0.9786", "0.3262"),
        "harmonic_envelope_offset": ("4", "0.0068", "1.0000", "0.2500"),
        "two_tone": ("6", "0.0096", "1.0000", "0.1667"),
        "damped_sinusoid": ("4", "0.0092", "0.9995", "0.2499"),
        "linear_chirp": ("4", "0.2331", "0.5151", "0.1288"),
        "gaussian_pulse": ("3", "0.0946", "0.7888", "0.2629"),
        "linear_no_intercept": ("2", "0.2774", "0.8932", "0.4466"),
        "constant": ("1", "1.0000", "1.0000", "1.0000"),
    }
    data = {r["signal_class"]: r for r in rows("k1_signal_classes_RD.csv") if r["sample_size"] == "5000"}
    for cls, (p, rd, pi, pip) in printed.items():
        r = data[cls]
        fact("tab:classes", f"{cls}: p = {p}", r["parameter_count"] == p, r["parameter_count"])
        num("tab:classes", f"{cls}: R_D", rd, f(r["design_factor_R"]))
        num("tab:classes", f"{cls}: pi", pi, f(r["projection_norm_sq"]))
        num("tab:classes", f"{cls}: pi/p", pip, f(r["projection_over_p"]))
    rng("sec:intro", "R_D range over the other signal models", "0.0068", "0.2774",
        [f(r["design_factor_R"]) for c, r in data.items() if c not in ("harmonic_envelope", "constant")])
    sci("sec:classes", "Q^-1 dbar = e_C for the offset model", "1.4", -14,
        f(data["harmonic_envelope_offset"]["constant_component_gap"]))
    num("sec:signal-class", "cond Q, harmonic", "18", f(data["harmonic_envelope"]["cond_Q"]))
    sci("sec:signal-class", "cond Q, two harmonics", "2.4", 3, f(data["two_tone"]["cond_Q"]))
    cells = rows("k1_signal_classes_RD_cells.csv")
    gaps = [f(r["identity_gap"]) for r in cells if r["identity_gap"] != "nan"]
    fact("sec:classes", "identity cells: 312 defined of 360", len(gaps) == 312 and len(cells) == 360,
         f"{len(gaps)} of {len(cells)}")
    sci("sec:classes", "max identity deviation over signal classes", "1.0", -14, max(gaps))


def predictions():
    pen = {r["scenario_id"]: f(r["centering_penalty_g"]) for r in rows("patp_plugin_sandwich_predictions.csv")}
    num("rem:symmetry", "penalty S075", "0.169", pen["S075"])
    num("rem:symmetry", "penalty A1L", "0.0013", pen["A1L"])


def constant_signal():
    data = {(r["case"], r["sample_size"]): r for r in rows("k1_constant_signal_check.csv")}
    u = data[("S3_symmetric_uniform", "5000")]
    e = data[("S2_asymmetric_exponential", "5000")]
    num("cor:constant", "uniform, S=3, imposed centring, n=5000", "0.3008", f(u["g_imposed_centring"]))
    num("cor:constant", "uniform population g", "0.3000", f(u["population_g"]))
    num("cor:constant", "exponential, S=2, imposed centring, n=5000", "0.5036", f(e["g_imposed_centring"]))
    num("cor:constant", "exponential population g", "0.5000", f(e["population_g"]))
    fact("cor:constant", "20 000 replications", u["replications"] == "20000" and e["replications"] == "20000",
         f"{u['replications']}, {e['replications']}")


# --------------------------------------------------------------------------- §8
def direction():
    printed = {  # g_or, predicted along u, empirical along u, complement, lambda2/lambda1
        "A2L": ("0.9025", "1.0370", "1.0300", "1.0012", "0.011"),
        "A2R": ("0.9025", "1.0370", "1.0301", "1.0014", "0.010"),
        "A4L": ("0.7843", "1.0943", "1.0744", "1.0016", "0.006"),
        "A4R": ("0.7843", "1.0943", "1.0756", "1.0021", "0.008"),
        "A3L": ("0.9943", "1.0020", "0.9964", "0.9991", None),
        "A3R": ("0.9943", "1.0020", "0.9960", "0.9991", None),
    }
    g_or = {r["scenario_id"]: f(r["oracle_g_asymptotic"]) for r in rows("patp_plugin_sandwich_predictions.csv")}
    data = {r["scenario_id"]: r for r in rows("k1_penalty_direction.csv") if r["sample_size"] == "5000"}
    for sid, (go, pu, eu, ec, lam) in printed.items():
        r = data[sid]
        num("tab:dir", f"{sid}: g_or", go, g_or[sid])
        num("tab:dir", f"{sid}: predicted along u", pu, f(r["predicted_ratio_along_u"]))
        num("tab:dir", f"{sid}: empirical along u", eu, f(r["empirical_ratio_along_u_vs_oracle_empirical"]))
        num("tab:dir", f"{sid}: empirical on complement", ec, f(r["empirical_ratio_complement_vs_oracle_empirical"]))
        if lam is not None:
            num("tab:dir", f"{sid}: lambda2/lambda1", lam, f(r["excess_second_over_max"]))


def rho_panel():
    data = rows("k1_rho_direction_panel.csv")
    fact("sec:rho-panel", "40 cells, prediction inside every interval, one outside every interval",
         len(data) == 40 and all(r["pred_inside_ci"] == "1" and r["one_inside_ci"] == "0" for r in data),
         f"{len(data)} cells")
    num("sec:rho-panel", "smallest lower interval end", "1.0107", min(f(r["ratio_u_ci_low"]) for r in data))
    rng("sec:rho-panel", "complement ratio", "0.9885", "1.0071", [f(r["empirical_ratio_complement"]) for r in data])
    num("sec:rho-panel", "largest lambda2/lambda1", "0.102", max(f(r["excess_second_over_max"]) for r in data))
    by = {(r["scenario_id"], r["rho"]): f(r["empirical_ratio_along_u"]) for r in data}
    mirror = max(abs(by[("A2L", rho)] - by[("A2R", rho)]) for (s, rho) in by if s == "A2L")
    mirror = max(mirror, max(abs(by[("A4L", rho)] - by[("A4R", rho)]) for (s, rho) in by if s == "A4L"))
    fact("fig:rho-panel", "mirror regimes differ by at most 0.0012", mirror <= 0.00125, f"max {mirror:.5f}")


def rank_contrast():
    data = rows("k1_rank_contrast.csv")
    ab = [abs(f(r["eig2_over_eig1"])) for r in data if r["structure"] != "C_mixed_instruments"]
    c = [f(r["eig2_over_eig1"]) for r in data if r["structure"] == "C_mixed_instruments"]
    fact("sec:rank-contrast", "aggregated and stacked: |lambda2/lambda1| <= 0.001", max(ab) <= 0.001, f"max {max(ab):.5f}")
    rng("sec:rank-contrast", "mixed instruments lambda2/lambda1", "-0.060", "-0.057", c)


def trace():
    printed = {  # g_or, predicted, empirical, ci low, ci high, oracle inside
        "A0": ("1.0000", "1.0000", "1.0043", "0.9977", "1.0103", True),
        "A1L": ("0.9940", "0.9953", "0.9985", "0.9867", "1.0104", True),
        "A1R": ("0.9940", "0.9953", "0.9982", "0.9865", "1.0108", True),
        "A2L": ("0.9025", "0.9239", "0.9190", "0.8927", "0.9466", True),
        "A2R": ("0.9025", "0.9239", "0.9208", "0.8978", "0.9472", True),
        "A3L": ("0.9943", "0.9956", "0.9923", "0.9857", "0.9993", True),
        "A3R": ("0.9943", "0.9956", "0.9929", "0.9860", "1.0000", True),
        "A4L": ("0.7843", "0.8316", "0.8180", "0.7831", "0.8525", True),
        "A4R": ("0.7843", "0.8316", "0.8221", "0.7870", "0.8594", False),
    }
    data = {r["scenario_id"]: r for r in rows("k1_convergence_rate.csv")
            if r["order"] == "2" and r["sample_size"] == "5000"}
    for sid, (go, pg, eg, lo, hi, inside) in printed.items():
        r = data[sid]
        num("tab:trace", f"{sid}: g_or", go, f(r["oracle_g"]))
        num("tab:trace", f"{sid}: predicted g_fe", pg, f(r["plugin_g_predicted"]))
        num("tab:trace", f"{sid}: empirical", eg, f(r["empirical_trace_g"]))
        num("tab:trace", f"{sid}: interval low", lo, f(r["ci_low"]))
        num("tab:trace", f"{sid}: interval high", hi, f(r["ci_high"]))
        fact("tab:trace", f"{sid}: prediction inside interval", r["prediction_inside_ci"] in ("1", "True"), r["prediction_inside_ci"])
        fact("tab:trace", f"{sid}: oracle inside interval = {inside}",
             (r["oracle_inside_ci"] in ("1", "True")) == inside, r["oracle_inside_ci"])
    widths = [f(data[s]["ci_high"]) - f(data[s]["ci_low"]) for s in ("A2L", "A2R", "A4L", "A4R")]
    rng("sec:trace", "interval width in A2 and A4", "0.05", "0.07", widths)


def coverage():
    s2 = {(r["scenario_id"], int(r["sample_size"]), r["estimator"]): r for r in rows("k1_sandwich_coverage.csv")}
    s3 = {(r["scenario_id"], int(r["sample_size"]), r["estimator"]): r for r in rows("k1_sandwich_coverage_s3.csv")}
    cov = lambda d, k: f(d[k]["coverage_joint_chi2"])
    num("tab:coverage", "Monte Carlo SE at coverage 0.95, M=1000", "0.007", math.sqrt(0.95 * 0.05 / 1000))
    deficits = [cov(s2, (s, n, "closed")) - cov(s2, (s, n, "naive")) for s in ("A2L", "A4L") for n in (1000, 5000)]
    rng("sec:coverage", "naive deficit, A2L/A4L, n>=1000", "0.006", "0.015", deficits)
    rng("sec:coverage", "coverage of all three forms at n=200, S=2", "0.879", "0.925",
        [cov(s2, (s, 200, e)) for s in ("A0", "A3L", "A2L", "A4L") for e in ("closed", "empirical", "naive")])
    rng("sec:coverage", "closed form at n=5000, S=3", "0.943", "0.957",
        [cov(s3, (s, 5000, "closed")) for s in ("S075", "S100", "S300", "S500")])
    num("sec:coverage", "naive deficit S300, n=5000", "0.003", cov(s3, ("S300", 5000, "closed")) - cov(s3, ("S300", 5000, "naive")))
    num("sec:coverage", "naive deficit S075, n=5000", "0.152", cov(s3, ("S075", 5000, "closed")) - cov(s3, ("S075", 5000, "naive")))
    pen = {r["scenario_id"]: f(r["centering_penalty_g"]) for r in rows("patp_plugin_sandwich_predictions.csv")}
    sids = ("S075", "S100", "S300", "S500")
    d = [cov(s3, (s, 5000, "closed")) - cov(s3, (s, 5000, "naive")) for s in sids]
    p = [pen[s] for s in sids]
    md, mp = sum(d) / 4, sum(p) / 4
    corr = sum((a - md) * (b - mp) for a, b in zip(d, p)) / math.sqrt(
        sum((a - md) ** 2 for a in d) * sum((b - mp) ** 2 for b in p))
    num("sec:coverage", "correlation of deficit with increment", "0.973", corr)
    rng("sec:coverage", "all three forms, S075/S100, n<=500", "0.588", "0.891",
        [cov(s3, (s, n, e)) for s in ("S075", "S100") for n in (200, 500) for e in ("closed", "empirical", "naive")])
    num("sec:coverage", "closed form S100, n=1000", "0.888", cov(s3, ("S100", 1000, "closed")))
    ratios = [f(s3[(s, 5000, "closed")][k]) for s in sids
              for k in ("mean_se_over_mc_sd_A", "mean_se_over_mc_sd_Omega", "mean_se_over_mc_sd_Phi")]
    rng("sec:s3", "mean SE over Monte Carlo SD, S=3, n=5000", "0.96", "1.06", ratios)


def order_three():
    printed = {  # rho, g_or, predicted, empirical, gap
        "S075": ("0.3", "0.2273", "0.3966", "0.4352", "0.0386"),
        "S100": ("0.3", "0.5108", "0.6180", "0.6879", "0.0700"),
        "S150": ("0.3", "0.9126", "0.9318", "1.0228", "0.0911"),
        "S200": ("0.3", "1.0000", "1.0000", "1.0886", "0.0886"),
        "S300": ("1", "0.8833", "0.9089", "0.8823", "-0.0266"),
        "S500": ("1", "0.6085", "0.6943", "0.6498", "-0.0444"),
    }
    data = {r["scenario_id"]: r for r in rows("k1_convergence_rate.csv")
            if r["order"] == "3" and r["sample_size"] == "5000"}
    for sid, (rho, go, pg, eg, gap) in printed.items():
        r = data[sid]
        num("tab:s3", f"{sid}: rho", rho, f(r["rho_feasible"]))
        num("tab:s3", f"{sid}: g_or", go, f(r["oracle_g"]))
        num("tab:s3", f"{sid}: predicted", pg, f(r["plugin_g_predicted"]))
        num("tab:s3", f"{sid}: empirical", eg, f(r["empirical_trace_g"]))
        num("tab:s3", f"{sid}: gap", gap, f(r["gap_empirical_minus_predicted"]))
        fact("tab:s3", f"{sid}: prediction outside interval", r["prediction_inside_ci"] in ("0", "False"), r["prediction_inside_ci"])
    num("sec:s3", "S150: gap", "0.091", f(data["S150"]["gap_empirical_minus_predicted"]))
    num("sec:s3", "S150: increment g_fe - g_or", "0.019", f(data["S150"]["plugin_g_predicted"]) - f(data["S150"]["oracle_g"]))

    diag = rows("k1_s3_gap_diagnosis.csv")
    a = {r["method"]: r for r in diag if r["table"] == "A" and r["sample_size"] == "5000"}
    sci("sec:s3", "S200 oracle one-step deviation from g=1", "6", -8, f(a["PATP Oracle one-step"]["excess"]))
    num("sec:s3", "S200 oracle mean damping", "1.0", f(a["PATP Oracle one-step"]["mean_damping"]))
    num("sec:s3", "S200 classical PMM3", "0.9992", f(a["Classical PMM3"]["trace_g"]))
    b = {r["scenario_id"]: r for r in diag if r["table"] == "B" and r["method"] == "PATP Feasible fixed"
         and r["sample_size"] == "5000"}
    rng("sec:s3", "|g-hat bias| at rho=0.3", "0.07", "0.12",
        [f(b[s]["abs_plugin_error"]) for s in ("S075", "S100", "S150", "S200")])
    fact("sec:s3", "|g-hat bias| at rho=1 below 0.002",
         max(f(b[s]["abs_plugin_error"]) for s in ("S300", "S500")) < 0.002,
         f"{f(b['S300']['abs_plugin_error']):.5f}, {f(b['S500']['abs_plugin_error']):.6f}")
    c = {r["scenario_id"]: f(r["oracle_excess"]) for r in diag if r["table"] == "C"}
    num("sec:s3", "S300 oracle below population", "0.020", -c["S300"])
    num("sec:s3", "S500 oracle below population", "0.027", -c["S500"])

    sweep = rows("k1_s3_rho_sweep.csv")
    bias_corr, cond_corr = [], []
    for sid in ("S200", "S100"):
        for n in ("500", "2000", "8000"):
            blk = [r for r in sweep if r["scenario_id"] == sid and r["sample_size"] == n]
            ex = [f(r["excess_over_feasible_theory"]) for r in blk]
            bias_corr.append(_corr([abs(f(r["plugin_g_bias"])) for r in blk], ex))
            cond_corr.append(_corr([math.log10(f(r["cond_F_population"])) for r in blk], ex))
    rng("sec:s3", "correlation of excess with |g-hat bias|", "0.71", "0.97", bias_corr)
    rng("sec:s3", "correlation of excess with log cond F", "-0.20", "0.23", cond_corr)
    s100 = {r["sample_size"]: f(r["excess_over_feasible_theory"]) for r in sweep
            if r["scenario_id"] == "S100" and r["rho"] == "0.3"}
    fact("sec:s3", "S100, rho=0.3: excess does not decrease from n=2000 to n=8000",
         s100["8000"] >= s100["2000"], f"{s100['2000']:.3f} -> {s100['8000']:.3f}")


def _corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def main() -> int:
    for block in (design_weight, signal_classes, predictions, constant_signal,
                  direction, rho_panel, rank_contrast, trace, coverage, order_three):
        block()
    failed = [c for c in CHECKS if not c[2]]
    for section, label, ok, detail in CHECKS:
        if not ok or "-v" in sys.argv:
            print(f"{'ok  ' if ok else 'FAIL'} [{section}] {label}: {detail}")
    if failed:
        print(f"{len(failed)} of {len(CHECKS)} reported values do NOT match the stored results")
        return 1
    print(f"all {len(CHECKS)} reported values reproduced from stored results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
