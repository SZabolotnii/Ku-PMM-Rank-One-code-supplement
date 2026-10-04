"""Residual-moment adaptive PATP experiments.

The adaptive estimator follows the PMM plug-in pattern: bounded NLS supplies
preliminary residuals; the moments needed by F and b are estimated from those
residuals; a safeguarded PATP correction is then applied.  The adaptive branch
also selects alpha on a fixed grid and re-estimates the moments after its first
accepted correction.  The DGP family is never used by either feasible branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    from patp_experiment_core import (
        LOWER_BOUNDS,
        PARAMETER_NAMES,
        RESULTS_DIR,
        SINGULAR_ALPHA,
        SINGULAR_BAND,
        TRUE_THETA,
        UPPER_BOUNDS,
        OracleSpec,
        build_oracle_spec,
        fit_bounded_nls,
        generate_errors,
        one_step_estimate,
        patp_power,
        signal,
        signal_design,
        write_csv,
    )
except ModuleNotFoundError:  # package-style import used by tests
    from scripts.patp_experiment_core import (
        LOWER_BOUNDS,
        PARAMETER_NAMES,
        RESULTS_DIR,
        SINGULAR_ALPHA,
        SINGULAR_BAND,
        TRUE_THETA,
        UPPER_BOUNDS,
        OracleSpec,
        build_oracle_spec,
        fit_bounded_nls,
        generate_errors,
        one_step_estimate,
        patp_power,
        signal,
        signal_design,
        write_csv,
    )


ADAPTIVE_ALPHA_GRID = tuple(
    float(value)
    for value in np.round(
        np.concatenate((np.arange(0.0, 0.50, 0.05), np.arange(0.55, 1.001, 0.05))),
        2,
    )
)
MAX_SPEC_CONDITION = 1.0e7
MIN_PLUGIN_G = 0.02
MAX_PLUGIN_G = 1.05
SMOOTHING_RATE = 0.75
MAX_ADAPTIVE_UPDATES = 2
SUMMARY_BOOTSTRAP_REPS = 500
METHOD_ORDER = (
    "NLS",
    "Oracle-calibrated",
    "Naive plug-in oracle alpha",
    "Feasible fixed alpha",
    "Adaptive alpha",
)


@dataclass(frozen=True)
class ResidualDiagnostics:
    c2: float
    gamma3: float
    gamma4: float
    gamma6: float
    epsilon: float
    valid: bool


@dataclass(frozen=True)
class EmpiricalSpec:
    spec: OracleSpec
    diagnostics: ResidualDiagnostics
    valid: bool


def feasible_alpha_grid(order: int) -> tuple[float, ...]:
    """Grid where every derivative-moment estimator has finite variance.

    Estimating b_i needs E|xi|^(p_i-1).  Under a continuous density positive at
    zero, its sample variance is finite only when 2(p_i-1) > -1, i.e. p_i>1/2.
    """
    return tuple(
        alpha
        for alpha in ADAPTIVE_ALPHA_GRID
        if min(patp_power(index, alpha) for index in range(2, order + 1)) > 0.5
    )


def residual_diagnostics(residuals: np.ndarray) -> tuple[np.ndarray, ResidualDiagnostics]:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size < 20 or not np.all(np.isfinite(residuals)):
        invalid = ResidualDiagnostics(*(float("nan"),) * 5, valid=False)
        return residuals, invalid
    centred = residuals - float(np.mean(residuals))
    c2 = float(np.mean(centred**2))
    if not np.isfinite(c2) or c2 <= 1e-12:
        invalid = ResidualDiagnostics(c2, *(float("nan"),) * 4, valid=False)
        return centred, invalid
    m3 = float(np.mean(centred**3))
    m4 = float(np.mean(centred**4))
    m6 = float(np.mean(centred**6))
    gamma3 = m3 / c2**1.5
    standardized_m4 = m4 / c2**2
    gamma4 = standardized_m4 - 3.0
    gamma6 = m6 / c2**3 - 15.0 * standardized_m4 + 30.0
    epsilon = float(np.sqrt(c2) * residuals.size ** (-SMOOTHING_RATE))
    valid = bool(
        np.all(np.isfinite([gamma3, gamma4, gamma6, epsilon])) and epsilon > 0.0
    )
    return centred, ResidualDiagnostics(c2, gamma3, gamma4, gamma6, epsilon, valid)


def _ols_fallback_spec(alpha: float, order: int, c2: float) -> OracleSpec:
    safe_c2 = c2 if np.isfinite(c2) and c2 > 0.0 else 1.0
    exponents = np.array(
        [1.0] + [patp_power(index, alpha) for index in range(2, order + 1)]
    )
    covariance = np.full((order, order), np.nan)
    covariance[0, 0] = safe_c2
    sensitivity_vector = np.full(order, np.nan)
    sensitivity_vector[0] = 1.0
    weights = np.zeros(order)
    weights[0] = 1.0 / safe_c2
    return OracleSpec(
        alpha=alpha,
        order=order,
        exponents=exponents,
        means=np.zeros(order),
        covariance=covariance,
        sensitivity_vector=sensitivity_vector,
        weights=weights,
        sensitivity=1.0 / safe_c2,
        c2=safe_c2,
        theory_g=1.0,
        condition_number=float("inf"),
        operational_fallback=True,
    )


def build_spec_from_moments(
    alpha: float,
    order: int,
    absolute_moment: Callable[[float], float],
    signed_moment: Callable[[float], float],
    *,
    c2: float,
) -> OracleSpec:
    """Construct an externally centred Form-B normal system."""
    exponents = np.array(
        [1.0] + [patp_power(index, alpha) for index in range(2, order + 1)]
    )
    exact_collision = np.max(np.abs(exponents - 1.0)) < 1e-13
    protected = abs(alpha - SINGULAR_ALPHA) < SINGULAR_BAND
    if exact_collision or protected:
        return _ols_fallback_spec(alpha, order, c2)
    means = np.array([float(signed_moment(float(q))) for q in exponents])
    covariance = np.array(
        [
            [
                float(absolute_moment(float(q_i + q_j))) - means[i] * means[j]
                for j, q_j in enumerate(exponents)
            ]
            for i, q_i in enumerate(exponents)
        ]
    )
    sensitivity_vector = np.array(
        [float(q * absolute_moment(float(q - 1.0))) for q in exponents]
    )
    try:
        weights = np.linalg.solve(covariance, sensitivity_vector)
        sensitivity = float(sensitivity_vector @ weights)
        condition_number = float(np.linalg.cond(covariance))
        plugin_g = float(1.0 / (c2 * sensitivity))
    except np.linalg.LinAlgError:
        return _ols_fallback_spec(alpha, order, c2)
    if (
        not np.all(np.isfinite(weights))
        or not np.isfinite(plugin_g)
        or plugin_g <= 0.0
        or condition_number > MAX_SPEC_CONDITION
    ):
        return _ols_fallback_spec(alpha, order, c2)
    return OracleSpec(
        alpha=alpha,
        order=order,
        exponents=exponents,
        means=means,
        covariance=covariance,
        sensitivity_vector=sensitivity_vector,
        weights=weights,
        sensitivity=sensitivity,
        c2=c2,
        theory_g=plugin_g,
        condition_number=condition_number,
        operational_fallback=False,
    )


def build_empirical_spec(alpha: float, order: int, residuals: np.ndarray) -> EmpiricalSpec:
    """Estimate every required moment from preliminary residuals.

    Negative fractional derivative moments are epsilon-smoothed with
    epsilon = sample_sd * n^(-3/4).  Positive moments are not winsorised for
    these finite-moment DGPs, so the plug-in estimator remains consistent.
    """
    centred, diagnostics = residual_diagnostics(residuals)
    if not diagnostics.valid:
        fallback = _ols_fallback_spec(alpha, order, diagnostics.c2)
        return EmpiricalSpec(fallback, diagnostics, False)
    absolute = np.abs(centred)
    signs = np.sign(centred)

    def absolute_moment(q: float) -> float:
        if q < 0.0:
            return float(np.mean(np.maximum(absolute, diagnostics.epsilon) ** q))
        return float(np.mean(absolute**q))

    def signed_moment(q: float) -> float:
        return float(np.mean(signs * absolute**q))

    spec = build_spec_from_moments(
        alpha,
        order,
        absolute_moment,
        signed_moment,
        c2=diagnostics.c2,
    )
    valid = bool(
        not spec.operational_fallback
        and MIN_PLUGIN_G <= spec.theory_g <= MAX_PLUGIN_G
        and spec.condition_number <= MAX_SPEC_CONDITION
    )
    return EmpiricalSpec(spec, diagnostics, valid)


def select_adaptive_alpha(
    order: int, residuals: np.ndarray
) -> tuple[float, EmpiricalSpec, bool]:
    candidates: list[tuple[float, EmpiricalSpec]] = []
    last: EmpiricalSpec | None = None
    for alpha in feasible_alpha_grid(order):
        empirical = build_empirical_spec(alpha, order, residuals)
        last = empirical
        if empirical.valid:
            candidates.append((alpha, empirical))
    if not candidates:
        if last is None:
            last = build_empirical_spec(SINGULAR_ALPHA, order, residuals)
        return SINGULAR_ALPHA, last, True
    alpha, empirical = min(candidates, key=lambda item: item[1].spec.theory_g)
    return alpha, empirical, False


def _bound_hit(theta: np.ndarray) -> bool:
    margin = 1e-5 * (UPPER_BOUNDS - LOWER_BOUNDS)
    return bool(
        np.all(np.isfinite(theta))
        and (
            np.any(theta <= LOWER_BOUNDS + margin)
            or np.any(theta >= UPPER_BOUNDS - margin)
        )
    )


def _row(
    *,
    base: dict[str, object],
    method: str,
    theta: np.ndarray,
    alpha: float | str,
    converged: bool,
    fallback: bool,
    damping: float,
    correction_norm: float,
    diagnostics: ResidualDiagnostics | None,
    selection_fallback: bool,
    plugin_g: float,
    true_g: float,
    oracle_alpha: float,
    oracle_g: float,
    feasible_alpha: float,
    feasible_g: float,
    adaptive_updates: int,
    alpha_switched: bool,
) -> dict[str, object]:
    return {
        **base,
        "method": method,
        "alpha": alpha,
        "A": theta[0],
        "Omega": theta[1],
        "Phi": theta[2],
        "converged": int(converged),
        "fallback": int(fallback),
        "bound_hit": int(_bound_hit(theta)),
        "damping": damping,
        "correction_norm": correction_norm,
        "gamma3_hat": "" if diagnostics is None else diagnostics.gamma3,
        "gamma4_hat": "" if diagnostics is None else diagnostics.gamma4,
        "gamma6_hat": "" if diagnostics is None else diagnostics.gamma6,
        "smoothing_epsilon": "" if diagnostics is None else diagnostics.epsilon,
        "moment_valid": 1 if diagnostics is None else int(diagnostics.valid),
        "selection_fallback": int(selection_fallback),
        "plugin_g": plugin_g,
        "true_g_at_alpha": true_g,
        "oracle_alpha": oracle_alpha,
        "oracle_g": oracle_g,
        "feasible_alpha": feasible_alpha,
        "feasible_g": feasible_g,
        "adaptive_updates": adaptive_updates,
        "alpha_switched": int(alpha_switched),
    }


def run_adaptive_experiment(
    *,
    order: int,
    distribution: str,
    sample_sizes: Iterable[int],
    reps: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Paired NLS, oracle, fixed plug-in and iteratively adaptive PATP study."""
    rng = np.random.default_rng(seed)
    true_specs = {
        alpha: build_oracle_spec(alpha, order, distribution)
        for alpha in ADAPTIVE_ALPHA_GRID
    }
    oracle_alpha, oracle_spec = min(
        (
            (alpha, spec)
            for alpha, spec in true_specs.items()
            if not spec.operational_fallback
        ),
        key=lambda item: item[1].theory_g,
    )
    feasible_alpha, feasible_spec = min(
        ((alpha, true_specs[alpha]) for alpha in feasible_alpha_grid(order)),
        key=lambda item: item[1].theory_g,
    )
    rows: list[dict[str, object]] = []

    for sample_size in sample_sizes:
        sample_size = int(sample_size)
        t, envelope = signal_design(sample_size)
        clean_signal = signal(TRUE_THETA, t, envelope)
        print(
            f"adaptive order={order}, distribution={distribution}, N={sample_size}, reps={reps}",
            flush=True,
        )
        for replication in range(reps):
            errors = generate_errors(distribution, sample_size, rng)
            observations = clean_signal + errors
            fit = fit_bounded_nls(observations, t, envelope)
            base = {
                "order": order,
                "distribution": distribution,
                "sample_size": sample_size,
                "replication": replication,
            }
            rows.append(
                _row(
                    base=base,
                    method="NLS",
                    theta=fit.theta,
                    alpha="",
                    converged=fit.success,
                    fallback=False,
                    damping=1.0,
                    correction_norm=0.0,
                    diagnostics=None,
                    selection_fallback=False,
                    plugin_g=1.0,
                    true_g=1.0,
                    oracle_alpha=oracle_alpha,
                    oracle_g=oracle_spec.theory_g,
                    feasible_alpha=feasible_alpha,
                    feasible_g=feasible_spec.theory_g,
                    adaptive_updates=0,
                    alpha_switched=False,
                )
            )
            if not fit.success:
                failed = np.full(3, np.nan)
                for method in METHOD_ORDER[1:]:
                    rows.append(
                        _row(
                            base=base,
                            method=method,
                            theta=failed,
                            alpha=(
                                oracle_alpha
                                if method in {"Oracle-calibrated", "Naive plug-in oracle alpha"}
                                else feasible_alpha if method == "Feasible fixed alpha" else ""
                            ),
                            converged=False,
                            fallback=True,
                            damping=0.0,
                            correction_norm=float("nan"),
                            diagnostics=None,
                            selection_fallback=method == "Adaptive alpha",
                            plugin_g=float("nan"),
                            true_g=float("nan"),
                            oracle_alpha=oracle_alpha,
                            oracle_g=oracle_spec.theory_g,
                            feasible_alpha=feasible_alpha,
                            feasible_g=feasible_spec.theory_g,
                            adaptive_updates=0,
                            alpha_switched=False,
                        )
                    )
                continue

            oracle_theta, oracle_fallback, oracle_damping, oracle_norm = one_step_estimate(
                observations, fit.theta, t, envelope, oracle_spec
            )
            rows.append(
                _row(
                    base=base,
                    method="Oracle-calibrated",
                    theta=oracle_theta,
                    alpha=oracle_alpha,
                    converged=True,
                    fallback=oracle_fallback,
                    damping=oracle_damping,
                    correction_norm=oracle_norm,
                    diagnostics=None,
                    selection_fallback=False,
                    plugin_g=oracle_spec.theory_g,
                    true_g=oracle_spec.theory_g,
                    oracle_alpha=oracle_alpha,
                    oracle_g=oracle_spec.theory_g,
                    feasible_alpha=feasible_alpha,
                    feasible_g=feasible_spec.theory_g,
                    adaptive_updates=1,
                    alpha_switched=False,
                )
            )

            initial_residuals = observations - signal(fit.theta, t, envelope)
            naive = build_empirical_spec(oracle_alpha, order, initial_residuals)
            if naive.valid:
                naive_theta, naive_fallback, naive_damping, naive_norm = one_step_estimate(
                    observations, fit.theta, t, envelope, naive.spec
                )
            else:
                naive_theta, naive_fallback, naive_damping, naive_norm = (
                    fit.theta.copy(),
                    True,
                    0.0,
                    0.0,
                )
            rows.append(
                _row(
                    base=base,
                    method="Naive plug-in oracle alpha",
                    theta=naive_theta,
                    alpha=oracle_alpha,
                    converged=True,
                    fallback=naive_fallback,
                    damping=naive_damping,
                    correction_norm=naive_norm,
                    diagnostics=naive.diagnostics,
                    selection_fallback=not naive.valid,
                    plugin_g=naive.spec.theory_g if naive.valid else float("nan"),
                    true_g=oracle_spec.theory_g,
                    oracle_alpha=oracle_alpha,
                    oracle_g=oracle_spec.theory_g,
                    feasible_alpha=feasible_alpha,
                    feasible_g=feasible_spec.theory_g,
                    adaptive_updates=int(naive.valid and not naive_fallback),
                    alpha_switched=False,
                )
            )

            fixed = build_empirical_spec(feasible_alpha, order, initial_residuals)
            if fixed.valid:
                fixed_theta, fixed_fallback, fixed_damping, fixed_norm = one_step_estimate(
                    observations, fit.theta, t, envelope, fixed.spec
                )
            else:
                fixed_theta, fixed_fallback, fixed_damping, fixed_norm = (
                    fit.theta.copy(),
                    True,
                    0.0,
                    0.0,
                )
            rows.append(
                _row(
                    base=base,
                    method="Feasible fixed alpha",
                    theta=fixed_theta,
                    alpha=feasible_alpha,
                    converged=True,
                    fallback=fixed_fallback,
                    damping=fixed_damping,
                    correction_norm=fixed_norm,
                    diagnostics=fixed.diagnostics,
                    selection_fallback=not fixed.valid,
                    plugin_g=fixed.spec.theory_g if fixed.valid else float("nan"),
                    true_g=feasible_spec.theory_g,
                    oracle_alpha=oracle_alpha,
                    oracle_g=oracle_spec.theory_g,
                    feasible_alpha=feasible_alpha,
                    feasible_g=feasible_spec.theory_g,
                    adaptive_updates=int(fixed.valid and not fixed_fallback),
                    alpha_switched=False,
                )
            )

            adaptive_theta = fit.theta.copy()
            selected_alpha = SINGULAR_ALPHA
            selected_empirical = build_empirical_spec(
                SINGULAR_ALPHA, order, initial_residuals
            )
            selection_fallback = False
            adaptive_fallback = False
            adaptive_damping = 0.0
            adaptive_norm = 0.0
            accepted_updates = 0
            alpha_history: list[float] = []
            for _ in range(MAX_ADAPTIVE_UPDATES):
                residuals = observations - signal(adaptive_theta, t, envelope)
                alpha, empirical, failed_selection = select_adaptive_alpha(order, residuals)
                if failed_selection:
                    selection_fallback = True
                    adaptive_fallback = True
                    break
                candidate, step_fallback, damping, correction_norm = one_step_estimate(
                    observations, adaptive_theta, t, envelope, empirical.spec
                )
                selected_alpha = alpha
                selected_empirical = empirical
                adaptive_damping = damping
                adaptive_norm = correction_norm
                alpha_history.append(alpha)
                if step_fallback:
                    adaptive_fallback = True
                    break
                adaptive_theta = candidate
                accepted_updates += 1
                if damping * correction_norm < 1e-8:
                    break
            if accepted_updates == 0:
                adaptive_theta = fit.theta.copy()
            true_g_selected = (
                true_specs[selected_alpha].theory_g
                if selected_alpha in true_specs and not selection_fallback
                else 1.0
            )
            rows.append(
                _row(
                    base=base,
                    method="Adaptive alpha",
                    theta=adaptive_theta,
                    alpha=selected_alpha,
                    converged=True,
                    fallback=adaptive_fallback,
                    damping=adaptive_damping,
                    correction_norm=adaptive_norm,
                    diagnostics=selected_empirical.diagnostics,
                    selection_fallback=selection_fallback,
                    plugin_g=(
                        selected_empirical.spec.theory_g
                        if selected_empirical.valid and not selection_fallback
                        else float("nan")
                    ),
                    true_g=true_g_selected,
                    oracle_alpha=oracle_alpha,
                    oracle_g=oracle_spec.theory_g,
                    feasible_alpha=feasible_alpha,
                    feasible_g=feasible_spec.theory_g,
                    adaptive_updates=accepted_updates,
                    alpha_switched=len(set(alpha_history)) > 1,
                )
            )
        print(f"completed adaptive N={sample_size}", flush=True)
    return rows, summarise_adaptive_rows(rows)


def _paired_trace_ratio_ci(
    estimates: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    bootstrap_reps: int = SUMMARY_BOOTSTRAP_REPS,
) -> tuple[float, float]:
    if estimates.shape[0] < 20:
        return float("nan"), float("nan")
    if np.array_equal(estimates, baseline):
        return 1.0, 1.0
    rng = np.random.default_rng(seed)
    ratios = np.empty(bootstrap_reps)
    for index in range(bootstrap_reps):
        draw = rng.integers(0, estimates.shape[0], size=estimates.shape[0])
        numerator = float(np.var(estimates[draw], axis=0, ddof=1).sum())
        denominator = float(np.var(baseline[draw], axis=0, ddof=1).sum())
        ratios[index] = numerator / denominator
    return tuple(float(value) for value in np.quantile(ratios, [0.025, 0.975]))


def summarise_adaptive_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["sample_size"]), str(row["method"])), []).append(row)
    summaries: list[dict[str, object]] = []
    method_seed = {method: index for index, method in enumerate(METHOD_ORDER)}
    for sample_size in sorted({key[0] for key in grouped}):
        baseline_group = sorted(
            grouped[(sample_size, "NLS")], key=lambda row: int(row["replication"])
        )
        baseline_all = np.array(
            [[row[name] for name in PARAMETER_NAMES] for row in baseline_group], dtype=float
        )
        for method in METHOD_ORDER:
            group = sorted(
                grouped[(sample_size, method)], key=lambda row: int(row["replication"])
            )
            estimates_all = np.array(
                [[row[name] for name in PARAMETER_NAMES] for row in group], dtype=float
            )
            paired_valid = np.all(np.isfinite(estimates_all), axis=1) & np.all(
                np.isfinite(baseline_all), axis=1
            )
            estimates = estimates_all[paired_valid]
            baseline = baseline_all[paired_valid]
            variances = np.var(estimates, axis=0, ddof=1)
            biases = np.mean(estimates, axis=0) - TRUE_THETA
            mse = np.mean((estimates - TRUE_THETA) ** 2, axis=0)
            baseline_variance = np.var(baseline, axis=0, ddof=1)
            baseline_mse = np.mean((baseline - TRUE_THETA) ** 2, axis=0)
            trace_ratio = float(variances.sum() / baseline_variance.sum())
            trace_mse_ratio = float(mse.sum() / baseline_mse.sum())
            ci_low, ci_high = _paired_trace_ratio_ci(
                estimates,
                baseline,
                seed=20260830 + sample_size + method_seed[method],
            )
            alphas = np.array(
                [float(row["alpha"]) for row in group if row["alpha"] != ""], dtype=float
            )
            oracle_alpha = float(group[0]["oracle_alpha"])
            oracle_g = float(group[0]["oracle_g"])
            feasible_alpha = float(group[0]["feasible_alpha"])
            feasible_g = float(group[0]["feasible_g"])

            def finite_field(name: str) -> np.ndarray:
                values = [float(row[name]) for row in group if row[name] != ""]
                array = np.asarray(values, dtype=float)
                return array[np.isfinite(array)]

            gamma3 = finite_field("gamma3_hat")
            gamma4 = finite_field("gamma4_hat")
            gamma6 = finite_field("gamma6_hat")
            epsilon = finite_field("smoothing_epsilon")
            plugin_g = finite_field("plugin_g")
            true_g = finite_field("true_g_at_alpha")
            record: dict[str, object] = {
                "sample_size": sample_size,
                "method": method,
                "n_total": len(group),
                "n_valid": int(paired_valid.sum()),
                "convergence_rate": float(np.mean([int(row["converged"]) for row in group])),
                "fallback_rate": float(np.mean([int(row["fallback"]) for row in group])),
                "bound_hit_rate": float(np.mean([int(row["bound_hit"]) for row in group])),
                "selection_fallback_rate": float(
                    np.mean([int(row["selection_fallback"]) for row in group])
                ),
                "alpha_switch_rate": float(np.mean([int(row["alpha_switched"]) for row in group])),
                "mean_adaptive_updates": float(
                    np.mean([int(row["adaptive_updates"]) for row in group])
                ),
                "bias_A": biases[0],
                "bias_Omega": biases[1],
                "bias_Phi": biases[2],
                "var_A": variances[0],
                "var_Omega": variances[1],
                "var_Phi": variances[2],
                "mse_A": mse[0],
                "mse_Omega": mse[1],
                "mse_Phi": mse[2],
                "trace_variance": float(variances.sum()),
                "trace_g_vs_nls": trace_ratio,
                "trace_g_ci_low": ci_low,
                "trace_g_ci_high": ci_high,
                "trace_mse": float(mse.sum()),
                "trace_mse_ratio": trace_mse_ratio,
                "oracle_alpha": oracle_alpha,
                "oracle_g": oracle_g,
                "feasible_alpha": feasible_alpha,
                "feasible_g": feasible_g,
                "alpha_mean": float(np.mean(alphas)) if alphas.size else float("nan"),
                "alpha_sd": float(np.std(alphas, ddof=1)) if alphas.size > 1 else 0.0,
                "alpha_oracle_rate": float(np.mean(np.isclose(alphas, oracle_alpha))) if alphas.size else float("nan"),
                "alpha_mae": float(np.mean(np.abs(alphas - oracle_alpha))) if alphas.size else float("nan"),
                "alpha_feasible_rate": float(np.mean(np.isclose(alphas, feasible_alpha))) if alphas.size else float("nan"),
                "alpha_feasible_mae": float(np.mean(np.abs(alphas - feasible_alpha))) if alphas.size else float("nan"),
                "gamma3_mean": float(np.mean(gamma3)) if gamma3.size else float("nan"),
                "gamma3_sd": float(np.std(gamma3, ddof=1)) if gamma3.size > 1 else float("nan"),
                "gamma4_mean": float(np.mean(gamma4)) if gamma4.size else float("nan"),
                "gamma4_sd": float(np.std(gamma4, ddof=1)) if gamma4.size > 1 else float("nan"),
                "gamma6_mean": float(np.mean(gamma6)) if gamma6.size else float("nan"),
                "gamma6_sd": float(np.std(gamma6, ddof=1)) if gamma6.size > 1 else float("nan"),
                "smoothing_epsilon_mean": float(np.mean(epsilon)) if epsilon.size else float("nan"),
                "plugin_g_mean": float(np.mean(plugin_g)) if plugin_g.size else float("nan"),
                "true_g_at_selected_mean": float(np.mean(true_g)) if true_g.size else float("nan"),
                "selection_regret_vs_full_oracle": float(np.mean(true_g - oracle_g)) if method == "Adaptive alpha" and true_g.size else 0.0,
                "selection_regret_vs_feasible_oracle": float(np.mean(true_g - feasible_g)) if method == "Adaptive alpha" and true_g.size else 0.0,
            }
            for index, name in enumerate(PARAMETER_NAMES):
                record[f"g_{name}"] = variances[index] / baseline_variance[index]
            summaries.append(record)
    return summaries


def write_adaptive_metadata(
    path: Path,
    *,
    order: int,
    distribution: str,
    sample_sizes: Iterable[int],
    reps: int,
    seed: int,
) -> None:
    true_specs = {
        alpha: build_oracle_spec(alpha, order, distribution)
        for alpha in ADAPTIVE_ALPHA_GRID
    }
    oracle_alpha, oracle_spec = min(
        (
            (alpha, spec)
            for alpha, spec in true_specs.items()
            if not spec.operational_fallback
        ),
        key=lambda item: item[1].theory_g,
    )
    feasible_alpha, feasible_spec = min(
        ((alpha, true_specs[alpha]) for alpha in feasible_alpha_grid(order)),
        key=lambda item: item[1].theory_g,
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "order": order,
        "distribution_used_only_as_mc_dgp": distribution,
        "distribution_family_used_by_feasible_estimators": False,
        "unknown_noise_characteristics": "all fractional moments/cumulants entering F and b",
        "zero_mean_is_identifying_restriction": True,
        "sample_sizes": list(sample_sizes),
        "replications_per_cell": reps,
        "seed": seed,
        "adaptive_alpha_grid": list(ADAPTIVE_ALPHA_GRID),
        "regularity_aware_alpha_grid": list(feasible_alpha_grid(order)),
        "regularity_rule": "p_i(alpha) > 1/2 for every estimated derivative moment",
        "oracle_alpha_on_grid": oracle_alpha,
        "oracle_g_on_grid": oracle_spec.theory_g,
        "feasible_oracle_alpha_on_grid": feasible_alpha,
        "feasible_oracle_g_on_grid": feasible_spec.theory_g,
        "negative_moment_smoothing": "epsilon = residual_sd * n^(-3/4)",
        "maximum_adaptive_updates": MAX_ADAPTIVE_UPDATES,
        "paired_bootstrap_replications": SUMMARY_BOOTSTRAP_REPS,
        "estimators": {
            "NLS": "bounded nonlinear least squares baseline",
            "Oracle-calibrated": "true noise moments and oracle alpha; comparison ceiling only",
            "Naive plug-in oracle alpha": "all moments estimated once at the unrestricted oracle alpha; negative baseline when nuisance moments lack finite variance",
            "Feasible fixed alpha": "all moments estimated once at the regularity-aware oracle alpha",
            "Adaptive alpha": "moments and alpha estimated over the regularity-aware grid and re-estimated after the first accepted correction",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_adaptive_artifacts(
    *,
    prefix: str,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    order: int,
    distribution: str,
    sample_sizes: Iterable[int],
    reps: int,
    seed: int,
) -> None:
    write_csv(RESULTS_DIR / f"{prefix}_replications.csv", rows)
    write_csv(RESULTS_DIR / f"{prefix}_summary.csv", summary)
    write_adaptive_metadata(
        RESULTS_DIR / f"{prefix}_metadata.json",
        order=order,
        distribution=distribution,
        sample_sizes=sample_sizes,
        reps=reps,
        seed=seed,
    )
