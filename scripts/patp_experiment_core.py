"""Shared, verification-first machinery for the PATP radio-signal studies."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import gamma, gammaln


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TRUE_THETA = np.array([2.0, 1.5, 0.5])
PARAMETER_NAMES = ("A", "Omega", "Phi")
LOWER_BOUNDS = np.array([0.05, 0.10, -np.pi])
UPPER_BOUNDS = np.array([6.00, 3.00, np.pi])
TARGET_ALPHAS = (0.0, 0.2, 0.7, 1.0)
SINGULAR_ALPHA = 0.5
SINGULAR_BAND = 0.02


@dataclass(frozen=True)
class OracleSpec:
    alpha: float
    order: int
    exponents: np.ndarray
    means: np.ndarray
    covariance: np.ndarray
    sensitivity_vector: np.ndarray
    weights: np.ndarray
    sensitivity: float
    c2: float
    theory_g: float
    condition_number: float
    operational_fallback: bool


@dataclass(frozen=True)
class NLSFit:
    theta: np.ndarray
    success: bool
    bound_hit: bool
    nfev: int
    cost: float


def patp_power(index: int, alpha: float) -> float:
    """Quadratic PATP exponent map."""
    return float(
        1.0 / index
        + (4.0 - index - 3.0 / index) * alpha
        + (2.0 * index - 4.0 + 2.0 / index) * alpha**2
    )


def laplace_abs_moment(q: float) -> float:
    """Exact absolute moment for a unit-variance Laplace variable."""
    if q <= -1.0:
        raise ValueError("The unit-variance Laplace absolute moment needs q > -1")
    return float(2.0 ** (-q / 2.0) * gamma(q + 1.0))


@lru_cache(maxsize=None)
def centered_gamma_moment(q: float, signed: bool = False) -> float:
    """High-accuracy moment for Gamma(1.5,1)-1.5 by split quadrature."""
    shape = 1.5
    center = shape
    log_constant = -gammaln(shape)

    def integrand(y: float) -> float:
        if y <= 0.0:
            return 0.0
        residual = y - center
        parity = np.sign(residual) if signed else 1.0
        log_density = log_constant + (shape - 1.0) * np.log(y) - y
        return float(parity * abs(residual) ** q * np.exp(log_density))

    left = quad(
        integrand,
        0.0,
        center,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=500,
    )[0]
    right = quad(
        integrand,
        center,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=500,
    )[0]
    return float(left + right)


def moment_function(distribution: str) -> Callable[[float], float]:
    if distribution == "centered_gamma":
        return centered_gamma_moment
    if distribution == "laplace":
        return laplace_abs_moment
    raise ValueError(f"Unsupported distribution: {distribution}")


def build_oracle_spec(
    alpha: float,
    order: int,
    distribution: str,
    *,
    apply_singular_band: bool = True,
) -> OracleSpec:
    """Build F, b and h for the externally calibrated Form-B score."""
    exponents = np.array(
        [1.0] + [patp_power(index, alpha) for index in range(2, order + 1)]
    )
    moment = moment_function(distribution)
    c2 = float(moment(2.0))
    if distribution == "centered_gamma":
        means = np.array(
            [0.0]
            + [centered_gamma_moment(float(q), True) for q in exponents[1:]]
        )
    else:
        means = np.zeros(order)

    exact_collision = np.max(np.abs(exponents - 1.0)) < 1e-13
    protected = apply_singular_band and abs(alpha - SINGULAR_ALPHA) < SINGULAR_BAND
    if exact_collision or protected:
        covariance = np.full((order, order), np.nan)
        covariance[0, 0] = c2
        b = np.array([1.0] + [float("nan")] * (order - 1))
        weights = np.zeros(order)
        weights[0] = 1.0 / c2
        return OracleSpec(
            alpha=alpha,
            order=order,
            exponents=exponents,
            means=means,
            covariance=covariance,
            sensitivity_vector=b,
            weights=weights,
            sensitivity=1.0 / c2,
            c2=c2,
            theory_g=1.0,
            condition_number=float("inf"),
            operational_fallback=True,
        )

    covariance = np.array(
        [
            [
                moment(float(q_i + q_j)) - means[i] * means[j]
                for j, q_j in enumerate(exponents)
            ]
            for i, q_i in enumerate(exponents)
        ]
    )
    b = np.array([q * moment(float(q - 1.0)) for q in exponents])
    weights = np.linalg.solve(covariance, b)
    sensitivity = float(b @ weights)
    theory_g = float(1.0 / (c2 * sensitivity))
    condition_number = float(np.linalg.cond(covariance))
    return OracleSpec(
        alpha=alpha,
        order=order,
        exponents=exponents,
        means=means,
        covariance=covariance,
        sensitivity_vector=b,
        weights=weights,
        sensitivity=sensitivity,
        c2=c2,
        theory_g=theory_g,
        condition_number=condition_number,
        operational_fallback=False,
    )


def signal_design(sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.01, 2.0, sample_size)
    envelope = 1.0 + 0.3 * np.cos(np.pi * t)
    return t, envelope


def signal(theta: np.ndarray, t: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    amplitude, frequency, phase = theta
    return amplitude * envelope * np.cos(frequency * t + phase)


def signal_gradient(
    theta: np.ndarray, t: np.ndarray, envelope: np.ndarray
) -> np.ndarray:
    amplitude, frequency, phase = theta
    angle = frequency * t + phase
    return np.vstack(
        [
            envelope * np.cos(angle),
            -amplitude * envelope * t * np.sin(angle),
            -amplitude * envelope * np.sin(angle),
        ]
    )


def harmonic_initial_guess(
    observations: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
) -> np.ndarray:
    """Data-driven frequency grid plus linear harmonic regression."""
    best_sse = float("inf")
    best = np.array([1.0, 1.5, 0.0])
    for frequency in np.linspace(0.25, 2.75, 51):
        design = np.column_stack(
            [envelope * np.cos(frequency * t), envelope * np.sin(frequency * t)]
        )
        coefficients = np.linalg.lstsq(design, observations, rcond=None)[0]
        residual = observations - design @ coefficients
        sse = float(residual @ residual)
        if sse < best_sse:
            amplitude = float(np.hypot(coefficients[0], coefficients[1]))
            phase = float(np.arctan2(-coefficients[1], coefficients[0]))
            best_sse = sse
            best = np.array([amplitude, frequency, phase])
    return np.clip(best, LOWER_BOUNDS + 1e-9, UPPER_BOUNDS - 1e-9)


def fit_bounded_nls(
    observations: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
) -> NLSFit:
    """Fit the Gaussian/NLS baseline without using the true parameter as a start."""
    start = harmonic_initial_guess(observations, t, envelope)
    result = least_squares(
        lambda theta: observations - signal(theta, t, envelope),
        start,
        jac=lambda theta: -signal_gradient(theta, t, envelope).T,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=500,
    )
    theta = np.asarray(result.x, dtype=float)
    margin = 1e-5 * (UPPER_BOUNDS - LOWER_BOUNDS)
    bound_hit = bool(
        np.any(theta <= LOWER_BOUNDS + margin) or np.any(theta >= UPPER_BOUNDS - margin)
    )
    success = bool(result.success and np.all(np.isfinite(theta)))
    return NLSFit(
        theta=theta,
        success=success,
        bound_hit=bound_hit,
        nfev=int(result.nfev),
        cost=float(result.cost),
    )


def basis_matrix(residuals: np.ndarray, spec: OracleSpec) -> np.ndarray:
    return np.vstack(
        [
            np.sign(residuals) * np.abs(residuals) ** exponent - spec.means[index]
            for index, exponent in enumerate(spec.exponents)
        ]
    )


def estimating_score(
    theta: np.ndarray,
    observations: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
    spec: OracleSpec,
) -> np.ndarray:
    residuals = observations - signal(theta, t, envelope)
    psi = spec.weights @ basis_matrix(residuals, spec)
    return signal_gradient(theta, t, envelope) @ psi


def oracle_influence_estimate(
    errors: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
    spec: OracleSpec,
) -> np.ndarray:
    """Oracle influence-function draw used only to verify the theory factor."""
    gradient = signal_gradient(TRUE_THETA, t, envelope)
    gram = gradient @ gradient.T
    psi = spec.weights @ basis_matrix(errors, spec)
    delta = np.linalg.solve(gram, gradient @ psi) / spec.sensitivity
    return TRUE_THETA + delta


def one_step_estimate(
    observations: np.ndarray,
    nls_theta: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
    spec: OracleSpec,
) -> tuple[np.ndarray, bool, float, float]:
    """One-step oracle-calibrated estimate with damping and OLS fallback.

    Returns ``(theta, fallback, damping, correction_norm)``.  Population moments
    remain external/oracle inputs; the signal parameter itself is not used.
    """
    if spec.operational_fallback:
        return nls_theta.copy(), True, 0.0, 0.0
    residuals = observations - signal(nls_theta, t, envelope)
    gradient = signal_gradient(nls_theta, t, envelope)
    gram = gradient @ gradient.T
    psi = spec.weights @ basis_matrix(residuals, spec)
    try:
        correction = np.linalg.solve(gram, gradient @ psi) / spec.sensitivity
    except np.linalg.LinAlgError:
        return nls_theta.copy(), True, 0.0, float("inf")
    correction_norm = float(np.linalg.norm(correction))
    if not np.all(np.isfinite(correction)):
        return nls_theta.copy(), True, 0.0, correction_norm

    base_score = float(
        np.linalg.norm(
            estimating_score(nls_theta, observations, t, envelope, spec)
        )
    )
    for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
        candidate = nls_theta + damping * correction
        if np.any(candidate <= LOWER_BOUNDS) or np.any(candidate >= UPPER_BOUNDS):
            continue
        candidate_score = float(
            np.linalg.norm(
                estimating_score(candidate, observations, t, envelope, spec)
            )
        )
        if candidate_score <= base_score * (1.0 - 1e-6 * damping) + 1e-12:
            return candidate, False, damping, correction_norm
    return nls_theta.copy(), True, 0.0, correction_norm


def generate_errors(
    distribution: str, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    if distribution == "centered_gamma":
        return rng.gamma(shape=1.5, scale=1.0, size=sample_size) - 1.5
    if distribution == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=sample_size)
    raise ValueError(f"Unsupported distribution: {distribution}")


def run_experiment(
    *,
    order: int,
    distribution: str,
    sample_sizes: Iterable[int],
    reps: int,
    seed: int,
    alphas: Iterable[float] = TARGET_ALPHAS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run paired NLS, oracle-IF and operational one-step Monte Carlo."""
    rng = np.random.default_rng(seed)
    alpha_values = tuple(float(alpha) for alpha in alphas)
    specs = {
        alpha: build_oracle_spec(alpha, order, distribution) for alpha in alpha_values
    }
    rows: list[dict[str, object]] = []

    for sample_size in sample_sizes:
        t, envelope = signal_design(int(sample_size))
        clean_signal = signal(TRUE_THETA, t, envelope)
        print(
            f"order={order}, distribution={distribution}, N={sample_size}, reps={reps}",
            flush=True,
        )
        for replication in range(reps):
            errors = generate_errors(distribution, int(sample_size), rng)
            observations = clean_signal + errors
            fit = fit_bounded_nls(observations, t, envelope)
            true_gradient = signal_gradient(TRUE_THETA, t, envelope)
            ols_if = TRUE_THETA + np.linalg.solve(
                true_gradient @ true_gradient.T, true_gradient @ errors
            )
            base = {
                "order": order,
                "distribution": distribution,
                "sample_size": int(sample_size),
                "replication": replication,
            }
            rows.append(
                {
                    **base,
                    "method": "NLS",
                    "alpha": "",
                    "A": fit.theta[0],
                    "Omega": fit.theta[1],
                    "Phi": fit.theta[2],
                    "converged": int(fit.success),
                    "fallback": 0,
                    "bound_hit": int(fit.bound_hit),
                    "damping": 1.0,
                    "correction_norm": 0.0,
                }
            )
            rows.append(
                {
                    **base,
                    "method": "OLS IF",
                    "alpha": "",
                    "A": ols_if[0],
                    "Omega": ols_if[1],
                    "Phi": ols_if[2],
                    "converged": 1,
                    "fallback": 0,
                    "bound_hit": 0,
                    "damping": 1.0,
                    "correction_norm": float(np.linalg.norm(ols_if - TRUE_THETA)),
                }
            )
            for alpha, spec in specs.items():
                oracle = oracle_influence_estimate(errors, t, envelope, spec)
                rows.append(
                    {
                        **base,
                        "method": "Oracle IF",
                        "alpha": alpha,
                        "A": oracle[0],
                        "Omega": oracle[1],
                        "Phi": oracle[2],
                        "converged": 1,
                        "fallback": int(spec.operational_fallback),
                        "bound_hit": 0,
                        "damping": 1.0,
                        "correction_norm": float(np.linalg.norm(oracle - TRUE_THETA)),
                    }
                )
                if fit.success:
                    estimate, fallback, damping, correction_norm = one_step_estimate(
                        observations, fit.theta, t, envelope, spec
                    )
                    converged = True
                else:
                    estimate = np.full(3, np.nan)
                    fallback = True
                    damping = 0.0
                    correction_norm = float("nan")
                    converged = False
                margin = 1e-5 * (UPPER_BOUNDS - LOWER_BOUNDS)
                bound_hit = bool(
                    np.all(np.isfinite(estimate))
                    and (
                        np.any(estimate <= LOWER_BOUNDS + margin)
                        or np.any(estimate >= UPPER_BOUNDS - margin)
                    )
                )
                rows.append(
                    {
                        **base,
                        "method": "One-step",
                        "alpha": alpha,
                        "A": estimate[0],
                        "Omega": estimate[1],
                        "Phi": estimate[2],
                        "converged": int(converged),
                        "fallback": int(fallback),
                        "bound_hit": int(bound_hit),
                        "damping": damping,
                        "correction_norm": correction_norm,
                    }
                )
        print(f"completed N={sample_size}", flush=True)
    summary = summarise_rows(rows, specs)
    return rows, summary


def summarise_rows(
    rows: list[dict[str, object]], specs: dict[float, OracleSpec]
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        alpha_key = "" if row["alpha"] == "" else f"{float(row['alpha']):.10g}"
        key = (int(row["sample_size"]), str(row["method"]), alpha_key)
        grouped.setdefault(key, []).append(row)

    baseline_variances: dict[tuple[int, str], np.ndarray] = {}
    baseline_mse: dict[tuple[int, str], np.ndarray] = {}
    for (sample_size, method, _), group in grouped.items():
        if method not in {"NLS", "OLS IF"}:
            continue
        estimates = np.array(
            [[row[name] for name in PARAMETER_NAMES] for row in group], dtype=float
        )
        valid = np.all(np.isfinite(estimates), axis=1)
        baseline_variances[(sample_size, method)] = np.var(
            estimates[valid], axis=0, ddof=1
        )
        baseline_mse[(sample_size, method)] = np.mean(
            (estimates[valid] - TRUE_THETA) ** 2, axis=0
        )

    summary: list[dict[str, object]] = []
    for (sample_size, method, alpha_key), group in sorted(grouped.items()):
        estimates = np.array(
            [[row[name] for name in PARAMETER_NAMES] for row in group], dtype=float
        )
        valid = np.all(np.isfinite(estimates), axis=1)
        valid_estimates = estimates[valid]
        alpha = None if alpha_key == "" else float(alpha_key)
        if valid_estimates.shape[0] < 2:
            variances = np.full(3, np.nan)
            biases = np.full(3, np.nan)
            mse = np.full(3, np.nan)
        else:
            variances = np.var(valid_estimates, axis=0, ddof=1)
            biases = np.mean(valid_estimates, axis=0) - TRUE_THETA
            mse = np.mean((valid_estimates - TRUE_THETA) ** 2, axis=0)
        trace = float(np.nansum(variances))
        comparison_baseline = "OLS IF" if method in {"OLS IF", "Oracle IF"} else "NLS"
        baseline = baseline_variances[(sample_size, comparison_baseline)]
        mse_baseline = baseline_mse[(sample_size, comparison_baseline)]
        trace_ratio = trace / float(baseline.sum())
        trace_mse = float(np.nansum(mse))
        trace_mse_ratio = trace_mse / float(mse_baseline.sum())
        theory_g = 1.0 if alpha is None else specs[alpha].theory_g
        record: dict[str, object] = {
            "sample_size": sample_size,
            "method": method,
            "alpha": "" if alpha is None else alpha,
            "comparison_baseline": comparison_baseline,
            "n_total": len(group),
            "n_valid": int(valid.sum()),
            "convergence_rate": float(np.mean([int(row["converged"]) for row in group])),
            "fallback_rate": float(np.mean([int(row["fallback"]) for row in group])),
            "bound_hit_rate": float(np.mean([int(row["bound_hit"]) for row in group])),
            "mean_damping": float(np.mean([float(row["damping"]) for row in group])),
            "bias_A": biases[0],
            "bias_Omega": biases[1],
            "bias_Phi": biases[2],
            "var_A": variances[0],
            "var_Omega": variances[1],
            "var_Phi": variances[2],
            "mse_A": mse[0],
            "mse_Omega": mse[1],
            "mse_Phi": mse[2],
            "trace_variance": trace,
            "trace_g_vs_nls": trace_ratio,
            "trace_mse": trace_mse,
            "trace_mse_ratio": trace_mse_ratio,
            "theory_g": theory_g,
            "absolute_gap": abs(trace_ratio - theory_g),
            "condition_number": 1.0 if alpha is None else specs[alpha].condition_number,
        }
        for index, name in enumerate(PARAMETER_NAMES):
            record[f"g_{name}"] = variances[index] / baseline[index]
        summary.append(record)
    return summary


def theory_sweep(
    *, order: int, distribution: str, step: float = 0.01
) -> list[dict[str, object]]:
    rows = []
    for alpha in np.arange(0.0, 1.0 + step / 2.0, step):
        alpha = float(np.round(alpha, 10))
        operational = build_oracle_spec(
            alpha, order, distribution, apply_singular_band=True
        )
        try:
            punctured = build_oracle_spec(
                alpha, order, distribution, apply_singular_band=False
            )
            punctured_g = punctured.theory_g
            punctured_condition = punctured.condition_number
        except np.linalg.LinAlgError:
            punctured_g = float("nan")
            punctured_condition = float("inf")
        rows.append(
            {
                "alpha": alpha,
                **{
                    f"p{index}": patp_power(index, alpha)
                    for index in range(2, order + 1)
                },
                "operational_g": operational.theory_g,
                "punctured_g": punctured_g,
                "condition_number": punctured_condition,
                "ols_fallback": int(operational.operational_fallback),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    path: Path,
    *,
    order: int,
    distribution: str,
    sample_sizes: Iterable[int],
    reps: int,
    seed: int,
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "order": order,
        "distribution": distribution,
        "sample_sizes": list(sample_sizes),
        "replications_per_cell": reps,
        "seed": seed,
        "target_alphas": list(TARGET_ALPHAS),
        "singular_alpha": SINGULAR_ALPHA,
        "protected_band": SINGULAR_BAND,
        "estimators": {
            "NLS": "bounded nonlinear least squares with a data-driven harmonic-grid start",
            "OLS IF": "linearised least-squares influence-function baseline",
            "Oracle IF": "true-parameter influence-function draw; theory verification only",
            "One-step": "NLS start plus externally population-calibrated PATP one-step correction",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
