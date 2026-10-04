"""Shared AEPD--PATP machinery for the two publication-oriented studies.

The AEPD shape is used only to generate Monte Carlo data and to construct
oracle comparators.  Feasible PATP and classical PMM competitors estimate all
moments from bounded-NLS residuals.  The AEPD split probability is named
``tau`` and the PATP control is named ``rho`` so that the two unrelated shape
parameters cannot be confused in code or reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

try:
    from aepd_core import (
        aepd_fractional_moment,
        aepd_logpdf,
        aepd_moments,
        standardized_aepd,
        standardized_aepd_rvs,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution from scripts/
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aepd_core import (
        aepd_fractional_moment,
        aepd_logpdf,
        aepd_moments,
        standardized_aepd,
        standardized_aepd_rvs,
    )

try:
    from patp_adaptive_core import (
        ADAPTIVE_ALPHA_GRID,
        MAX_ADAPTIVE_UPDATES,
        build_empirical_spec,
        build_spec_from_moments,
        feasible_alpha_grid,
        select_adaptive_alpha,
    )
    from patp_experiment_core import (
        LOWER_BOUNDS,
        PARAMETER_NAMES,
        RESULTS_DIR,
        TRUE_THETA,
        UPPER_BOUNDS,
        OracleSpec,
        fit_bounded_nls,
        one_step_estimate,
        oracle_influence_estimate,
        patp_power,
        signal,
        signal_design,
        signal_gradient,
    )
except ModuleNotFoundError:  # package-style imports used by tests
    from scripts.patp_adaptive_core import (
        ADAPTIVE_ALPHA_GRID,
        MAX_ADAPTIVE_UPDATES,
        build_empirical_spec,
        build_spec_from_moments,
        feasible_alpha_grid,
        select_adaptive_alpha,
    )
    from scripts.patp_experiment_core import (
        LOWER_BOUNDS,
        PARAMETER_NAMES,
        RESULTS_DIR,
        TRUE_THETA,
        UPPER_BOUNDS,
        OracleSpec,
        fit_bounded_nls,
        one_step_estimate,
        oracle_influence_estimate,
        patp_power,
        signal,
        signal_design,
        signal_gradient,
    )


ROOT = Path(__file__).resolve().parents[1]
FIXED_RHOS = (0.0, 0.2, 0.7, 1.0)
DEFAULT_BOOTSTRAP_REPS = 1_000
METHOD_ORDER = (
    "NLS",
    "OLS IF",
    "PATP Oracle IF",
    "PATP Oracle one-step",
    "PATP Naive rho=0",
    "PATP Feasible fixed",
    "PATP Adaptive",
    "Classical PMM2",
    "Classical PMM3",
    "AEPD MLE",
)


@dataclass(frozen=True)
class AEPDScenario:
    scenario_id: str
    tau: float
    p_left: float
    p_right: float
    role: str

    @property
    def is_symmetric(self) -> bool:
        return bool(
            abs(self.tau - 0.5) < 1e-14
            and abs(self.p_left - self.p_right) < 1e-14
        )

    @property
    def has_standard_mle_asymptotics(self) -> bool:
        return self.p_left > 1.0 and self.p_right > 1.0

    @property
    def aepd_shape(self) -> tuple[float, float, float]:
        return (self.tau, self.p_left, self.p_right)


@dataclass(frozen=True)
class ScenarioManifest:
    study_id: str
    order: int
    description: str
    scenarios: tuple[AEPDScenario, ...]
    path: Path
    sha256: str


@dataclass(frozen=True)
class ShapeStatistics:
    gamma3: float
    gamma4: float
    gamma6: float
    classical_g: float


@dataclass(frozen=True)
class ClassicalSpec:
    order: int
    means: np.ndarray
    covariance: np.ndarray
    sensitivity_vector: np.ndarray
    weights: np.ndarray
    sensitivity: float
    c2: float
    theory_g: float
    condition_number: float
    valid: bool


@dataclass(frozen=True)
class JointMLEFit:
    theta: np.ndarray
    noise_sd: float
    tau: float
    p_left: float
    p_right: float
    success: bool
    bound_hit: bool
    n_successful_starts: int
    n_starts: int
    objective: float


def load_manifest(
    path: Path | str,
    *,
    allow_asymmetric_order3: bool = False,
) -> ScenarioManifest:
    path = Path(path).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    order = int(payload["order"])
    if order not in {2, 3}:
        raise ValueError("AEPD--PATP manifests support only S=2 or S=3")
    scenarios = tuple(
        AEPDScenario(
            scenario_id=str(item["id"]),
            tau=float(item["tau"]),
            p_left=float(item["p_left"]),
            p_right=float(item["p_right"]),
            role=str(item["role"]),
        )
        for item in payload["scenarios"]
    )
    # The S=3 symmetry guard protects the *classical PMM3 comparator*, not the PATP
    # spec.  ``scenario_shape_statistics`` computes classical_g at order 3 from
    # 1 - gamma4^2 / (6 + 9 gamma4 + gamma6), which is derived for symmetric errors
    # and is wrong once gamma3 != 0.  ``build_aepd_patp_spec`` takes signed moments
    # and is valid for asymmetric laws.  ``allow_asymmetric_order3=True`` is therefore
    # admissible for diagnostics that use the PATP path only; classical_g is returned
    # as NaN in that case so it cannot be consumed silently.
    if not scenarios or len({item.scenario_id for item in scenarios}) != len(scenarios):
        raise ValueError("Scenario ids must be non-empty and unique")
    for scenario in scenarios:
        standardized_aepd(*scenario.aepd_shape)
        if order == 3 and not scenario.is_symmetric and not allow_asymmetric_order3:
            raise ValueError("Every S=3 AEPD scenario must be symmetric")
    return ScenarioManifest(
        study_id=str(payload["study_id"]),
        order=order,
        description=str(payload["description"]),
        scenarios=scenarios,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def scenario_shape_statistics(scenario: AEPDScenario, order: int) -> ShapeStatistics:
    tau, p_left, p_right = scenario.aepd_shape
    m3 = aepd_fractional_moment(3.0, tau, p_left, p_right, True)
    m4 = aepd_fractional_moment(4.0, tau, p_left, p_right, False)
    m6 = aepd_fractional_moment(6.0, tau, p_left, p_right, False)
    gamma3 = float(m3)
    gamma4 = float(m4 - 3.0)
    gamma6 = float(m6 - 15.0 * m4 + 30.0)
    if order == 2:
        denominator = 2.0 + gamma4
        classical_g = 1.0 - gamma3**2 / denominator
    elif abs(gamma3) > 1e-12:
        # PMM3 comparator is symmetric-only; refuse to report a number.
        classical_g = float("nan")
    else:
        denominator = 6.0 + 9.0 * gamma4 + gamma6
        classical_g = 1.0 - gamma4**2 / denominator
    return ShapeStatistics(gamma3, gamma4, gamma6, float(classical_g))


def build_aepd_patp_spec(
    scenario: AEPDScenario,
    rho: float,
    order: int,
    *,
    apply_singular_band: bool = True,
) -> OracleSpec:
    tau, p_left, p_right = scenario.aepd_shape

    def absolute_moment(q: float) -> float:
        return aepd_fractional_moment(q, tau, p_left, p_right, False)

    def signed_moment(q: float) -> float:
        return aepd_fractional_moment(q, tau, p_left, p_right, True)

    if not apply_singular_band and abs(rho - 0.5) < 1e-13:
        raise np.linalg.LinAlgError("exact PATP collision")
    spec = build_spec_from_moments(
        float(rho),
        order,
        absolute_moment,
        signed_moment,
        c2=1.0,
    )
    if not apply_singular_band and spec.operational_fallback:
        exponents = np.array(
            [1.0] + [patp_power(index, rho) for index in range(2, order + 1)]
        )
        means = np.array([signed_moment(float(q)) for q in exponents])
        covariance = np.array(
            [
                [absolute_moment(float(qi + qj)) - means[i] * means[j]
                 for j, qj in enumerate(exponents)]
                for i, qi in enumerate(exponents)
            ]
        )
        b = np.array([q * absolute_moment(float(q - 1.0)) for q in exponents])
        weights = np.linalg.solve(covariance, b)
        sensitivity = float(b @ weights)
        return OracleSpec(
            alpha=float(rho),
            order=order,
            exponents=exponents,
            means=means,
            covariance=covariance,
            sensitivity_vector=b,
            weights=weights,
            sensitivity=sensitivity,
            c2=1.0,
            theory_g=float(1.0 / sensitivity),
            condition_number=float(np.linalg.cond(covariance)),
            operational_fallback=False,
        )
    return spec


def oracle_and_feasible_optima(
    scenario: AEPDScenario, order: int
) -> tuple[float, OracleSpec, float, OracleSpec]:
    specs = {
        rho: build_aepd_patp_spec(scenario, rho, order)
        for rho in ADAPTIVE_ALPHA_GRID
    }
    oracle_rho, oracle_spec = min(
        ((rho, spec) for rho, spec in specs.items() if not spec.operational_fallback),
        key=lambda item: (round(item[1].theory_g, 12), item[0]),
    )
    feasible_rho, feasible_spec = min(
        ((rho, specs[rho]) for rho in feasible_alpha_grid(order)),
        key=lambda item: (round(item[1].theory_g, 12), item[0]),
    )
    return oracle_rho, oracle_spec, feasible_rho, feasible_spec


def build_classical_oracle_spec(
    scenario: AEPDScenario, order: int
) -> ClassicalSpec:
    tau, p_left, p_right = scenario.aepd_shape
    m2 = 1.0
    m3 = aepd_fractional_moment(3.0, tau, p_left, p_right, True)
    m4 = aepd_fractional_moment(4.0, tau, p_left, p_right, False)
    if order == 2:
        means = np.array([0.0, m2])
        covariance = np.array([[m2, m3], [m3, m4 - m2**2]])
        b = np.array([1.0, 0.0])
    elif order == 3:
        m6 = aepd_fractional_moment(6.0, tau, p_left, p_right, False)
        means = np.array([0.0, m3])
        covariance = np.array([[m2, m4], [m4, m6 - m3**2]])
        b = np.array([1.0, 3.0 * m2])
    else:
        raise ValueError("Classical comparator supports only PMM2 or PMM3")
    weights = np.linalg.solve(covariance, b)
    sensitivity = float(b @ weights)
    return ClassicalSpec(
        order=order,
        means=means,
        covariance=covariance,
        sensitivity_vector=b,
        weights=weights,
        sensitivity=sensitivity,
        c2=m2,
        theory_g=float(1.0 / (m2 * sensitivity)),
        condition_number=float(np.linalg.cond(covariance)),
        valid=True,
    )


def build_classical_empirical_spec(residuals: np.ndarray, order: int) -> ClassicalSpec:
    residuals = np.asarray(residuals, dtype=float)
    centred = residuals - float(np.mean(residuals))
    if centred.size < 20 or not np.all(np.isfinite(centred)):
        return _invalid_classical_spec(order)
    m2 = float(np.mean(centred**2))
    m3 = float(np.mean(centred**3))
    m4 = float(np.mean(centred**4))
    if not np.isfinite(m2) or m2 <= 1e-12:
        return _invalid_classical_spec(order)
    if order == 2:
        means = np.array([0.0, m2])
        covariance = np.array([[m2, m3], [m3, m4 - m2**2]])
        b = np.array([1.0, 0.0])
    else:
        m6 = float(np.mean(centred**6))
        means = np.array([0.0, m3])
        covariance = np.array([[m2, m4], [m4, m6 - m3**2]])
        b = np.array([1.0, 3.0 * m2])
    try:
        weights = np.linalg.solve(covariance, b)
        sensitivity = float(b @ weights)
        condition = float(np.linalg.cond(covariance))
        theory_g = float(1.0 / (m2 * sensitivity))
    except np.linalg.LinAlgError:
        return _invalid_classical_spec(order)
    valid = bool(
        np.all(np.isfinite(weights))
        and np.isfinite(theory_g)
        and 0.0 < theory_g <= 1.1
        and condition < 1e8
    )
    return ClassicalSpec(
        order, means, covariance, b, weights, sensitivity, m2,
        theory_g, condition, valid,
    )


def _invalid_classical_spec(order: int) -> ClassicalSpec:
    return ClassicalSpec(
        order=order,
        means=np.full(2, np.nan),
        covariance=np.full((2, 2), np.nan),
        sensitivity_vector=np.full(2, np.nan),
        weights=np.full(2, np.nan),
        sensitivity=float("nan"),
        c2=float("nan"),
        theory_g=float("nan"),
        condition_number=float("inf"),
        valid=False,
    )


def classical_basis(residuals: np.ndarray, spec: ClassicalSpec) -> np.ndarray:
    if spec.order == 2:
        return np.vstack([residuals, residuals**2 - spec.means[1]])
    return np.vstack([residuals, residuals**3 - spec.means[1]])


def _generic_score(
    theta: np.ndarray,
    observations: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
    spec: ClassicalSpec,
) -> np.ndarray:
    residuals = observations - signal(theta, t, envelope)
    psi = spec.weights @ classical_basis(residuals, spec)
    return signal_gradient(theta, t, envelope) @ psi


def classical_oracle_influence(
    errors: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
    spec: ClassicalSpec,
) -> np.ndarray:
    gradient = signal_gradient(TRUE_THETA, t, envelope)
    psi = spec.weights @ classical_basis(errors, spec)
    delta = np.linalg.solve(gradient @ gradient.T, gradient @ psi) / spec.sensitivity
    return TRUE_THETA + delta


def classical_one_step(
    observations: np.ndarray,
    start: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
    spec: ClassicalSpec,
) -> tuple[np.ndarray, bool, float, float]:
    if not spec.valid:
        return start.copy(), True, 0.0, 0.0
    residuals = observations - signal(start, t, envelope)
    gradient = signal_gradient(start, t, envelope)
    psi = spec.weights @ classical_basis(residuals, spec)
    try:
        correction = np.linalg.solve(gradient @ gradient.T, gradient @ psi) / spec.sensitivity
    except np.linalg.LinAlgError:
        return start.copy(), True, 0.0, float("inf")
    norm = float(np.linalg.norm(correction))
    if not np.all(np.isfinite(correction)):
        return start.copy(), True, 0.0, norm
    base_score = float(np.linalg.norm(_generic_score(start, observations, t, envelope, spec)))
    for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
        candidate = start + damping * correction
        if np.any(candidate <= LOWER_BOUNDS) or np.any(candidate >= UPPER_BOUNDS):
            continue
        score = float(np.linalg.norm(_generic_score(candidate, observations, t, envelope, spec)))
        if score <= base_score * (1.0 - 1e-6 * damping) + 1e-12:
            return candidate, False, damping, norm
    return start.copy(), True, 0.0, norm


def _mean_zero_logpdf(
    residuals: np.ndarray,
    noise_sd: float,
    tau: float,
    p_left: float,
    p_right: float,
) -> np.ndarray:
    unit = standardized_aepd(tau, p_left, p_right)
    return np.asarray(
        aepd_logpdf(
            residuals,
            noise_sd * unit.mu,
            noise_sd * unit.sigma,
            tau,
            p_left,
            p_right,
        ),
        dtype=float,
    )


def fit_mean_zero_aepd_signal(
    observations: np.ndarray,
    t: np.ndarray,
    envelope: np.ndarray,
    nls_theta: np.ndarray,
    *,
    max_starts: int = 3,
) -> JointMLEFit:
    """Jointly fit signal parameters and a mean-zero AEPD nuisance law."""
    residuals = observations - signal(nls_theta, t, envelope)
    sd0 = max(float(np.std(residuals)), 1e-3)
    skew = float(np.mean((residuals - np.mean(residuals)) ** 3) / sd0**3)
    directional = (0.35, 1.2, 2.5) if skew < 0 else (0.65, 2.5, 1.2)
    starts = [
        np.r_[nls_theta, sd0, 0.50, 2.0, 2.0],
        np.r_[nls_theta, sd0, *directional],
        np.r_[nls_theta, sd0, 0.50, 1.0, 1.0],
    ][:max(1, int(max_starts))]
    bounds = [
        *list(zip(LOWER_BOUNDS, UPPER_BOUNDS)),
        (sd0 * 0.05, sd0 * 20.0),
        (0.02, 0.98),
        (0.25, 8.0),
        (0.25, 8.0),
    ]

    def objective(parameters: np.ndarray) -> float:
        theta = parameters[:3]
        noise_sd, tau, p_left, p_right = parameters[3:]
        model_residuals = observations - signal(theta, t, envelope)
        try:
            values = _mean_zero_logpdf(model_residuals, noise_sd, tau, p_left, p_right)
        except (ValueError, FloatingPointError, OverflowError):
            return float("inf")
        if np.any(~np.isfinite(values)):
            return float("inf")
        return float(-np.sum(values))

    candidates: list[OptimizeResult] = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-6},
        )
        if result.success and np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            candidates.append(result)
    if not candidates:
        return JointMLEFit(
            np.full(3, np.nan), *(float("nan"),) * 4, False, False,
            0, len(starts), float("inf"),
        )
    best = min(candidates, key=lambda item: float(item.fun))
    values = np.asarray(best.x, dtype=float)
    lower = np.asarray([item[0] for item in bounds])
    upper = np.asarray([item[1] for item in bounds])
    margin = 1e-5 * (upper - lower)
    bound_hit = bool(np.any(values <= lower + margin) or np.any(values >= upper - margin))
    return JointMLEFit(
        theta=values[:3],
        noise_sd=float(values[3]),
        tau=float(values[4]),
        p_left=float(values[5]),
        p_right=float(values[6]),
        success=True,
        bound_hit=bound_hit,
        n_successful_starts=len(candidates),
        n_starts=len(starts),
        objective=float(best.fun),
    )


def _bound_hit(theta: np.ndarray) -> bool:
    if not np.all(np.isfinite(theta)):
        return False
    margin = 1e-5 * (UPPER_BOUNDS - LOWER_BOUNDS)
    return bool(
        np.any(theta <= LOWER_BOUNDS + margin)
        or np.any(theta >= UPPER_BOUNDS - margin)
    )


def _result_row(
    base: dict[str, object],
    *,
    method: str,
    theta: np.ndarray,
    rho: float | str = "",
    converged: bool = True,
    fallback: bool = False,
    bound_hit: bool | None = None,
    damping: float = 1.0,
    correction_norm: float = 0.0,
    moment_source: str,
    comparison_baseline: str,
    theory_g: float = 1.0,
    condition_number: float = 1.0,
    plugin_g: float = float("nan"),
    rho_oracle: float = float("nan"),
    rho_feasible: float = float("nan"),
    selection_fallback: bool = False,
    adaptive_updates: int = 0,
    mle_fit: JointMLEFit | None = None,
) -> dict[str, object]:
    return {
        **base,
        "method": method,
        "rho": rho,
        "A": theta[0],
        "Omega": theta[1],
        "Phi": theta[2],
        "converged": int(converged),
        "fallback": int(fallback),
        "bound_hit": int(_bound_hit(theta) if bound_hit is None else bound_hit),
        "damping": damping,
        "correction_norm": correction_norm,
        "moment_source": moment_source,
        "comparison_baseline": comparison_baseline,
        "theory_g": theory_g,
        "condition_number": condition_number,
        "plugin_g": plugin_g,
        "rho_oracle": rho_oracle,
        "rho_feasible": rho_feasible,
        "selection_fallback": int(selection_fallback),
        "adaptive_updates": adaptive_updates,
        "mle_noise_sd": "" if mle_fit is None else mle_fit.noise_sd,
        "mle_tau": "" if mle_fit is None else mle_fit.tau,
        "mle_p_left": "" if mle_fit is None else mle_fit.p_left,
        "mle_p_right": "" if mle_fit is None else mle_fit.p_right,
        "mle_successful_starts": "" if mle_fit is None else mle_fit.n_successful_starts,
        "mle_starts": "" if mle_fit is None else mle_fit.n_starts,
        "mle_objective": "" if mle_fit is None else mle_fit.objective,
    }


def _scenario_probabilities(
    scenario: AEPDScenario, probabilities: np.ndarray
) -> np.ndarray:
    return 1.0 - probabilities if scenario.scenario_id.endswith("R") else probabilities


def run_aepd_patp_study(
    manifest: ScenarioManifest,
    *,
    sample_sizes: Iterable[int],
    reps: int,
    seed: int,
    include_mle: bool = True,
    mle_starts: int = 3,
    scenario_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Run paired AEPD scenarios and every pre-specified estimator."""
    selected = set(scenario_ids or ())
    scenarios = tuple(
        scenario for scenario in manifest.scenarios
        if not selected or scenario.scenario_id in selected
    )
    if not scenarios:
        raise ValueError("No scenarios selected")
    order = manifest.order
    rng = np.random.default_rng(seed)
    shape_stats = {
        scenario.scenario_id: scenario_shape_statistics(scenario, order)
        for scenario in scenarios
    }
    optima = {
        scenario.scenario_id: oracle_and_feasible_optima(scenario, order)
        for scenario in scenarios
    }
    classical_oracle = {
        scenario.scenario_id: build_classical_oracle_spec(scenario, order)
        for scenario in scenarios
    }
    oracle_check_specs: dict[str, dict[float, OracleSpec]] = {}
    for scenario in scenarios:
        oracle_rho = optima[scenario.scenario_id][0]
        rhos = tuple(dict.fromkeys((*FIXED_RHOS, oracle_rho)))
        oracle_check_specs[scenario.scenario_id] = {
            rho: build_aepd_patp_spec(scenario, rho, order) for rho in rhos
        }

    rows: list[dict[str, object]] = []
    for sample_size in (int(value) for value in sample_sizes):
        t, envelope = signal_design(sample_size)
        clean_signal = signal(TRUE_THETA, t, envelope)
        for replication in range(reps):
            probabilities = rng.random(sample_size)
            for scenario in scenarios:
                errors = standardized_aepd_rvs(
                    sample_size,
                    *scenario.aepd_shape,
                    probabilities=_scenario_probabilities(scenario, probabilities),
                )
                observations = clean_signal + errors
                fit = fit_bounded_nls(observations, t, envelope)
                stats = shape_stats[scenario.scenario_id]
                oracle_rho, oracle_spec, feasible_rho, feasible_spec = optima[
                    scenario.scenario_id
                ]
                base = {
                    "study_id": manifest.study_id,
                    "order": order,
                    "scenario_id": scenario.scenario_id,
                    "scenario_role": scenario.role,
                    "tau": scenario.tau,
                    "p_left": scenario.p_left,
                    "p_right": scenario.p_right,
                    "gamma3": stats.gamma3,
                    "gamma4": stats.gamma4,
                    "gamma6": stats.gamma6,
                    "classical_g_theory": stats.classical_g,
                    "mle_regular": int(scenario.has_standard_mle_asymptotics),
                    "sample_size": sample_size,
                    "replication": replication,
                }
                rows.append(
                    _result_row(
                        base,
                        method="NLS",
                        theta=fit.theta,
                        converged=fit.success,
                        bound_hit=fit.bound_hit,
                        moment_source="none",
                        comparison_baseline="NLS",
                        rho_oracle=oracle_rho,
                        rho_feasible=feasible_rho,
                    )
                )
                true_gradient = signal_gradient(TRUE_THETA, t, envelope)
                ols_if = TRUE_THETA + np.linalg.solve(
                    true_gradient @ true_gradient.T, true_gradient @ errors
                )
                rows.append(
                    _result_row(
                        base,
                        method="OLS IF",
                        theta=ols_if,
                        moment_source="oracle linearisation",
                        comparison_baseline="OLS IF",
                        rho_oracle=oracle_rho,
                        rho_feasible=feasible_rho,
                    )
                )
                for rho, spec in oracle_check_specs[scenario.scenario_id].items():
                    oracle_theta = oracle_influence_estimate(errors, t, envelope, spec)
                    rows.append(
                        _result_row(
                            base,
                            method="PATP Oracle IF",
                            theta=oracle_theta,
                            rho=rho,
                            fallback=spec.operational_fallback,
                            moment_source="true AEPD moments",
                            comparison_baseline="OLS IF",
                            theory_g=spec.theory_g,
                            condition_number=spec.condition_number,
                            rho_oracle=oracle_rho,
                            rho_feasible=feasible_rho,
                        )
                    )
                if not fit.success:
                    failed = np.full(3, np.nan)
                    for method in (
                        "PATP Oracle one-step",
                        "PATP Naive rho=0",
                        "PATP Feasible fixed",
                        "PATP Adaptive",
                        f"Classical PMM{order}",
                        "AEPD MLE",
                    ):
                        if method == "AEPD MLE" and not include_mle:
                            continue
                        rows.append(
                            _result_row(
                                base,
                                method=method,
                                theta=failed,
                                converged=False,
                                fallback=method != "AEPD MLE",
                                moment_source="unavailable after NLS failure",
                                comparison_baseline="NLS",
                                rho_oracle=oracle_rho,
                                rho_feasible=feasible_rho,
                            )
                        )
                    continue

                oracle_theta, fallback, damping, norm = one_step_estimate(
                    observations, fit.theta, t, envelope, oracle_spec
                )
                rows.append(
                    _result_row(
                        base,
                        method="PATP Oracle one-step",
                        theta=oracle_theta,
                        rho=oracle_rho,
                        fallback=fallback,
                        damping=damping,
                        correction_norm=norm,
                        moment_source="true AEPD moments",
                        comparison_baseline="NLS",
                        theory_g=oracle_spec.theory_g,
                        condition_number=oracle_spec.condition_number,
                        plugin_g=oracle_spec.theory_g,
                        rho_oracle=oracle_rho,
                        rho_feasible=feasible_rho,
                        adaptive_updates=int(not fallback),
                    )
                )
                residuals = observations - signal(fit.theta, t, envelope)
                naive = build_empirical_spec(0.0, order, residuals)
                naive_theta, naive_fallback, naive_damping, naive_norm = one_step_estimate(
                    observations, fit.theta, t, envelope, naive.spec
                ) if naive.valid else (fit.theta.copy(), True, 0.0, 0.0)
                rows.append(
                    _result_row(
                        base,
                        method="PATP Naive rho=0",
                        theta=naive_theta,
                        rho=0.0,
                        fallback=naive_fallback,
                        damping=naive_damping,
                        correction_norm=naive_norm,
                        moment_source="NLS residual moments",
                        comparison_baseline="NLS",
                        theory_g=build_aepd_patp_spec(scenario, 0.0, order).theory_g,
                        condition_number=naive.spec.condition_number,
                        plugin_g=naive.spec.theory_g if naive.valid else float("nan"),
                        rho_oracle=oracle_rho,
                        rho_feasible=feasible_rho,
                        selection_fallback=not naive.valid,
                        adaptive_updates=int(naive.valid and not naive_fallback),
                    )
                )
                fixed = build_empirical_spec(feasible_rho, order, residuals)
                fixed_theta, fixed_fallback, fixed_damping, fixed_norm = one_step_estimate(
                    observations, fit.theta, t, envelope, fixed.spec
                ) if fixed.valid else (fit.theta.copy(), True, 0.0, 0.0)
                rows.append(
                    _result_row(
                        base,
                        method="PATP Feasible fixed",
                        theta=fixed_theta,
                        rho=feasible_rho,
                        fallback=fixed_fallback,
                        damping=fixed_damping,
                        correction_norm=fixed_norm,
                        moment_source="NLS residual moments",
                        comparison_baseline="NLS",
                        theory_g=feasible_spec.theory_g,
                        condition_number=fixed.spec.condition_number,
                        plugin_g=fixed.spec.theory_g if fixed.valid else float("nan"),
                        rho_oracle=oracle_rho,
                        rho_feasible=feasible_rho,
                        selection_fallback=not fixed.valid,
                        adaptive_updates=int(fixed.valid and not fixed_fallback),
                    )
                )
                adaptive_theta = fit.theta.copy()
                selected_rho = feasible_rho
                selected_empirical = fixed
                adaptive_fallback = False
                selection_fallback = False
                accepted_updates = 0
                adaptive_damping = 0.0
                adaptive_norm = 0.0
                for _ in range(MAX_ADAPTIVE_UPDATES):
                    current_residuals = observations - signal(adaptive_theta, t, envelope)
                    selected_rho, selected_empirical, failed_selection = select_adaptive_alpha(
                        order, current_residuals
                    )
                    if failed_selection:
                        selection_fallback = True
                        adaptive_fallback = True
                        break
                    candidate, step_fallback, adaptive_damping, adaptive_norm = one_step_estimate(
                        observations, adaptive_theta, t, envelope, selected_empirical.spec
                    )
                    if step_fallback:
                        adaptive_fallback = True
                        break
                    adaptive_theta = candidate
                    accepted_updates += 1
                    if adaptive_damping * adaptive_norm < 1e-8:
                        break
                if accepted_updates == 0:
                    adaptive_theta = fit.theta.copy()
                selected_true = build_aepd_patp_spec(scenario, selected_rho, order)
                rows.append(
                    _result_row(
                        base,
                        method="PATP Adaptive",
                        theta=adaptive_theta,
                        rho=selected_rho,
                        fallback=adaptive_fallback,
                        damping=adaptive_damping,
                        correction_norm=adaptive_norm,
                        moment_source="iterated residual moments",
                        comparison_baseline="NLS",
                        theory_g=selected_true.theory_g,
                        condition_number=selected_empirical.spec.condition_number,
                        plugin_g=(
                            selected_empirical.spec.theory_g
                            if selected_empirical.valid else float("nan")
                        ),
                        rho_oracle=oracle_rho,
                        rho_feasible=feasible_rho,
                        selection_fallback=selection_fallback,
                        adaptive_updates=accepted_updates,
                    )
                )
                classical = build_classical_empirical_spec(residuals, order)
                classical_theta, classical_fallback, classical_damping, classical_norm = (
                    classical_one_step(observations, fit.theta, t, envelope, classical)
                )
                rows.append(
                    _result_row(
                        base,
                        method=f"Classical PMM{order}",
                        theta=classical_theta,
                        fallback=classical_fallback,
                        damping=classical_damping,
                        correction_norm=classical_norm,
                        moment_source="NLS residual integer moments",
                        comparison_baseline="NLS",
                        theory_g=classical_oracle[scenario.scenario_id].theory_g,
                        condition_number=classical.condition_number,
                        plugin_g=classical.theory_g,
                        rho_oracle=oracle_rho,
                        rho_feasible=feasible_rho,
                        selection_fallback=not classical.valid,
                        adaptive_updates=int(classical.valid and not classical_fallback),
                    )
                )
                if include_mle:
                    mle = fit_mean_zero_aepd_signal(
                        observations, t, envelope, fit.theta, max_starts=mle_starts
                    )
                    rows.append(
                        _result_row(
                            base,
                            method="AEPD MLE",
                            theta=mle.theta,
                            converged=mle.success,
                            bound_hit=mle.bound_hit,
                            moment_source="joint mean-zero likelihood",
                            comparison_baseline="NLS",
                            rho_oracle=oracle_rho,
                            rho_feasible=feasible_rho,
                            mle_fit=mle,
                        )
                    )
            progress_step = max(1, reps // 10)
            if (replication + 1) % progress_step == 0 or replication + 1 == reps:
                print(
                    f"{manifest.study_id}: N={sample_size}, "
                    f"replication={replication + 1}/{reps}",
                    flush=True,
                )
    return rows


def theory_sweep(
    manifest: ScenarioManifest, *, step: float = 0.01
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in manifest.scenarios:
        stats = scenario_shape_statistics(scenario, manifest.order)
        oracle_rho, oracle_spec, feasible_rho, feasible_spec = oracle_and_feasible_optima(
            scenario, manifest.order
        )
        for rho in np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10):
            rho = float(rho)
            operational = build_aepd_patp_spec(scenario, rho, manifest.order)
            try:
                punctured = build_aepd_patp_spec(
                    scenario, rho, manifest.order, apply_singular_band=False
                )
                punctured_g = punctured.theory_g
                punctured_condition = punctured.condition_number
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                punctured_g = float("nan")
                punctured_condition = float("inf")
            rows.append(
                {
                    "study_id": manifest.study_id,
                    "order": manifest.order,
                    "scenario_id": scenario.scenario_id,
                    "tau": scenario.tau,
                    "p_left": scenario.p_left,
                    "p_right": scenario.p_right,
                    "gamma3": stats.gamma3,
                    "gamma4": stats.gamma4,
                    "gamma6": stats.gamma6,
                    "classical_g": stats.classical_g,
                    "rho": rho,
                    **{
                        f"patp_p{index}": patp_power(index, rho)
                        for index in range(2, manifest.order + 1)
                    },
                    "operational_g": operational.theory_g,
                    "punctured_g": punctured_g,
                    "condition_number": punctured_condition,
                    "ols_fallback": int(operational.operational_fallback),
                    "regularity_feasible": int(rho in feasible_alpha_grid(manifest.order)),
                    "rho_oracle": oracle_rho,
                    "g_oracle": oracle_spec.theory_g,
                    "rho_feasible": feasible_rho,
                    "g_feasible": feasible_spec.theory_g,
                }
            )
    return rows


def _paired_trace_ratio_ci(
    estimates: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    bootstrap_reps: int,
) -> tuple[float, float]:
    if estimates.shape[0] < 20 or bootstrap_reps < 20:
        return float("nan"), float("nan")
    if np.array_equal(estimates, baseline):
        return 1.0, 1.0
    rng = np.random.default_rng(seed)
    ratios = np.empty(bootstrap_reps)
    for index in range(bootstrap_reps):
        draw = rng.integers(0, estimates.shape[0], estimates.shape[0])
        ratios[index] = np.var(estimates[draw], axis=0, ddof=1).sum() / np.var(
            baseline[draw], axis=0, ddof=1
        ).sum()
    return tuple(float(value) for value in np.quantile(ratios, [0.025, 0.975]))


def summarise_rows(
    rows: list[dict[str, object]], *, bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        rho_key = (
            f"{float(row['rho']):.10g}"
            if row["method"] == "PATP Oracle IF" and row["rho"] != ""
            else ""
        )
        key = (
            str(row["scenario_id"]), int(row["sample_size"]),
            str(row["method"]), rho_key,
        )
        grouped.setdefault(key, []).append(row)
    baseline_groups: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for (scenario_id, sample_size, method, rho_key), group in grouped.items():
        if method in {"NLS", "OLS IF"} and rho_key == "":
            baseline_groups[(scenario_id, sample_size, method)] = group
    summaries: list[dict[str, object]] = []
    method_seed = {method: index for index, method in enumerate(METHOD_ORDER)}
    for key in sorted(grouped):
        scenario_id, sample_size, method, rho_key = key
        group = sorted(grouped[key], key=lambda item: int(item["replication"]))
        baseline_name = str(group[0]["comparison_baseline"])
        baseline_group = sorted(
            baseline_groups[(scenario_id, sample_size, baseline_name)],
            key=lambda item: int(item["replication"]),
        )
        estimates_all = np.asarray(
            [[row[name] for name in PARAMETER_NAMES] for row in group], dtype=float
        )
        baseline_all = np.asarray(
            [[row[name] for name in PARAMETER_NAMES] for row in baseline_group], dtype=float
        )
        valid = np.all(np.isfinite(estimates_all), axis=1) & np.all(
            np.isfinite(baseline_all), axis=1
        )
        estimates = estimates_all[valid]
        baseline = baseline_all[valid]
        if estimates.shape[0] < 2:
            biases = variances = mse = np.full(3, np.nan)
            covariance = np.full((3, 3), np.nan)
            base_variance = base_mse = np.full(3, np.nan)
        else:
            biases = np.mean(estimates, axis=0) - TRUE_THETA
            variances = np.var(estimates, axis=0, ddof=1)
            mse = np.mean((estimates - TRUE_THETA) ** 2, axis=0)
            covariance = np.cov(estimates, rowvar=False, ddof=1)
            base_variance = np.var(baseline, axis=0, ddof=1)
            base_mse = np.mean((baseline - TRUE_THETA) ** 2, axis=0)
        trace_ratio = float(np.sum(variances) / np.sum(base_variance))
        trace_mse_ratio = float(np.sum(mse) / np.sum(base_mse))
        seed_value = (
            20260850 + sample_size + 101 * sum(map(ord, scenario_id))
            + method_seed.get(method, 99)
        )
        ci_low, ci_high = _paired_trace_ratio_ci(
            estimates, baseline, seed=seed_value, bootstrap_reps=bootstrap_reps
        )
        finite_rhos = np.asarray(
            [float(row["rho"]) for row in group if row["rho"] != ""], dtype=float
        )
        convergence_rate = float(np.mean([int(row["converged"]) for row in group]))
        mle_regular = bool(int(group[0]["mle_regular"]))
        if method != "AEPD MLE":
            mle_inference_status = "not-applicable"
        elif not mle_regular:
            mle_inference_status = "nonregular-boundary-no-normal-inference"
        elif convergence_rate < 0.95:
            mle_inference_status = "unstable-no-inference"
        else:
            mle_inference_status = "regular-converged"
        record: dict[str, object] = {
            "study_id": group[0]["study_id"],
            "order": group[0]["order"],
            "scenario_id": scenario_id,
            "scenario_role": group[0]["scenario_role"],
            "tau": group[0]["tau"],
            "p_left": group[0]["p_left"],
            "p_right": group[0]["p_right"],
            "gamma3": group[0]["gamma3"],
            "gamma4": group[0]["gamma4"],
            "gamma6": group[0]["gamma6"],
            "classical_g_theory": group[0]["classical_g_theory"],
            "mle_regular": group[0]["mle_regular"],
            "mle_stable": int(convergence_rate >= 0.95),
            "mle_inference_status": mle_inference_status,
            "sample_size": sample_size,
            "method": method,
            "rho": (
                float(rho_key)
                if rho_key != ""
                else float(np.mean(finite_rhos)) if finite_rhos.size else ""
            ),
            "comparison_baseline": baseline_name,
            "n_total": len(group),
            "n_valid": int(np.sum(valid)),
            "convergence_rate": convergence_rate,
            "fallback_rate": float(np.mean([int(row["fallback"]) for row in group])),
            "bound_hit_rate": float(np.mean([int(row["bound_hit"]) for row in group])),
            "selection_fallback_rate": float(
                np.mean([int(row["selection_fallback"]) for row in group])
            ),
            "mean_damping": float(np.mean([float(row["damping"]) for row in group])),
            "mean_adaptive_updates": float(
                np.mean([int(row["adaptive_updates"]) for row in group])
            ),
            "bias_A": biases[0],
            "bias_Omega": biases[1],
            "bias_Phi": biases[2],
            "var_A": variances[0],
            "var_Omega": variances[1],
            "var_Phi": variances[2],
            "cov_A_Omega": covariance[0, 1],
            "cov_A_Phi": covariance[0, 2],
            "cov_Omega_Phi": covariance[1, 2],
            "mse_A": mse[0],
            "mse_Omega": mse[1],
            "mse_Phi": mse[2],
            "trace_variance": float(np.sum(variances)),
            "trace_g": trace_ratio,
            "trace_g_ci_low": ci_low,
            "trace_g_ci_high": ci_high,
            "trace_mse": float(np.sum(mse)),
            "trace_mse_ratio": trace_mse_ratio,
            "theory_g": float(group[0]["theory_g"]),
            "absolute_theory_gap": abs(trace_ratio - float(group[0]["theory_g"])),
            "condition_number_mean": float(
                np.mean([float(row["condition_number"]) for row in group])
            ),
            "plugin_g_mean": float(
                np.nanmean([float(row["plugin_g"]) for row in group])
            ) if any(np.isfinite(float(row["plugin_g"])) for row in group) else float("nan"),
            "rho_mean": float(np.mean(finite_rhos)) if finite_rhos.size else float("nan"),
            "rho_sd": float(np.std(finite_rhos, ddof=1)) if finite_rhos.size > 1 else 0.0,
            "rho_oracle": group[0]["rho_oracle"],
            "rho_feasible": group[0]["rho_feasible"],
            "gain_confirmed": int(
                np.isfinite(ci_high) and ci_high < 1.0 and trace_mse_ratio < 1.0
            ),
        }
        for index, name in enumerate(PARAMETER_NAMES):
            record[f"g_{name}"] = variances[index] / base_variance[index]
        summaries.append(record)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    prefix: str,
    manifest: ScenarioManifest,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    sweep: list[dict[str, object]],
    *,
    sample_sizes: Sequence[int],
    reps: int,
    seed: int,
    bootstrap_reps: int,
    include_mle: bool,
    mle_starts: int,
    pilot: bool,
) -> None:
    write_csv(RESULTS_DIR / f"{prefix}_replications.csv", rows)
    write_csv(RESULTS_DIR / f"{prefix}_summary.csv", summary)
    write_csv(RESULTS_DIR / f"{prefix}_theory_sweep.csv", sweep)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": manifest.study_id,
        "order": manifest.order,
        "description": manifest.description,
        "scenario_manifest": str(manifest.path.relative_to(ROOT)),
        "scenario_manifest_sha256": manifest.sha256,
        "scenario_count": len(manifest.scenarios),
        "sample_sizes": list(sample_sizes),
        "replications_per_cell": reps,
        "seed": seed,
        "pilot": pilot,
        "paired_bootstrap_replications": bootstrap_reps,
        "fixed_rhos": list(FIXED_RHOS),
        "adaptive_rho_grid": list(ADAPTIVE_ALPHA_GRID),
        "regularity_aware_rho_grid": list(feasible_alpha_grid(manifest.order)),
        "regularity_rule": "p_i(rho) > 1/2 for every estimated derivative moment",
        "collision_fallback": "NLS/OLS fallback for |rho-0.5| < 0.02",
        "noise_standardisation": "exact mean zero and variance one for every AEPD scenario",
        "simulation_surface": "analytic radio signal only; no real-data claim",
        "include_joint_mean_zero_aepd_mle": include_mle,
        "aepd_mle_starts": mle_starts if include_mle else 0,
        "estimators": {
            "OLS IF and PATP Oracle IF": "linearised theory verification only",
            "PATP Oracle one-step": "NLS start plus true AEPD moments",
            "PATP Naive rho=0": "negative plug-in baseline when derivative moments are irregular",
            "PATP Feasible fixed": "NLS residual moments at the regularity-aware oracle rho",
            "PATP Adaptive": "family-blind residual moments and two safeguarded rho updates",
            "Classical PMM2/PMM3": "integer-moment residual plug-in comparator",
            "AEPD MLE": "joint signal and mean-zero AEPD nuisance likelihood",
        },
    }
    (RESULTS_DIR / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
