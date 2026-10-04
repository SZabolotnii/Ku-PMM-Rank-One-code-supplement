"""First-order sandwich theory for fixed-rho residual-moment PATP.

All covariance matrices returned here are asymptotic covariances of
``sqrt(n) * (theta_hat - theta_0)``.  The fixed-design PATP specialization
matches the implementation in ``patp_adaptive_core.build_empirical_spec``:
preliminary residuals are sample-centred, every signed basis mean is estimated
from those residuals, and the optimal weights converge to ``F^{-1} b``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from patp_experiment_core import OracleSpec
except ModuleNotFoundError:  # package-style imports used by tests
    from scripts.patp_experiment_core import OracleSpec


@dataclass(frozen=True)
class PluginSandwich:
    """Oracle covariance, plug-in covariance, and the centering penalty."""

    oracle: np.ndarray
    feasible: np.ndarray
    penalty: np.ndarray
    design_information: np.ndarray
    mean_gradient: np.ndarray
    sensitivity: float
    score_variance: float
    score_error_covariance: float


def two_step_sandwich(
    a: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    d: np.ndarray,
    stacked_covariance: np.ndarray,
) -> np.ndarray:
    """Generic two-step sandwich with a root-n preliminary estimator.

    Let ``u_i`` be the main estimating score, ``v_i`` the nuisance-moment
    score, and ``ell_i`` the influence contribution of the preliminary
    estimator.  With Jacobians

        A = E[dU/dtheta], C = E[dU/deta],
        H = E[dV/deta],   D = E[dV/dtheta_tilde],

    the effective score is

        u_i - C H^{-1} (v_i + D ell_i).

    ``stacked_covariance`` is Var([u_i, v_i, ell_i]).
    """
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    h = np.asarray(h, dtype=float)
    d = np.asarray(d, dtype=float)
    omega = np.asarray(stacked_covariance, dtype=float)
    k = a.shape[0]
    r = h.shape[0]
    q = d.shape[1]
    if a.shape != (k, k) or c.shape != (k, r) or h.shape != (r, r):
        raise ValueError("Incompatible A, C, and H dimensions")
    if d.shape[0] != r or omega.shape != (k + r + q, k + r + q):
        raise ValueError("Incompatible D or stacked covariance dimensions")
    h_inv = np.linalg.inv(h)
    loading = np.concatenate(
        (np.eye(k), -c @ h_inv, -c @ h_inv @ d), axis=1
    )
    a_inv = np.linalg.inv(a)
    covariance = a_inv @ loading @ omega @ loading.T @ a_inv.T
    return (covariance + covariance.T) / 2.0


def fixed_design_plugin_sandwich(
    gradient: np.ndarray,
    spec: OracleSpec,
) -> PluginSandwich:
    """Closed-form same-sample plug-in PATP sandwich for a fixed rho.

    ``gradient`` has shape ``(k, n)`` with columns
    ``d_i = dm_i(theta_0)/dtheta``.  The result assumes:

    * iid, mean-zero errors independent of the fixed design;
    * sample-centred preliminary residuals;
    * fixed rho away from the exponent collision;
    * consistent F, b, signed means, weights and sensitivity;
    * a full, undamped one-step update with asymptotically inactive safeguards.

    The exact first-order penalty is

        (c2 - 1/kappa) Q^{-1} dbar dbar' Q^{-1},

    where ``Q = n^{-1} sum d_i d_i'`` and ``kappa = b'F^{-1}b``.
    """
    gradient = np.asarray(gradient, dtype=float)
    if gradient.ndim != 2 or gradient.shape[1] < gradient.shape[0]:
        raise ValueError("gradient must have shape (parameter_count, sample_size)")
    if spec.operational_fallback:
        raise ValueError("The PATP collision/fallback point has no plug-in sandwich")
    q = gradient @ gradient.T / gradient.shape[1]
    q_inv = np.linalg.inv(q)
    dbar = np.mean(gradient, axis=1)
    kappa = float(spec.sensitivity)
    c2 = float(spec.c2)
    if not np.isfinite(kappa) or kappa <= 0.0 or not np.isfinite(c2) or c2 <= 0.0:
        raise ValueError("Positive finite sensitivity and error variance are required")

    # For h=F^{-1}b: Var(h'z)=h'Fh=h'b=kappa and
    # Cov(h'z, epsilon)=h'F e_1=b_1=1.
    score_variance = float(spec.weights @ spec.covariance @ spec.weights)
    score_error_covariance = float(
        spec.weights @ spec.covariance[:, 0]
    )
    oracle = q_inv / kappa
    direction = q_inv @ dbar
    penalty = (c2 - 1.0 / kappa) * np.outer(direction, direction)
    feasible = oracle + penalty
    return PluginSandwich(
        oracle=(oracle + oracle.T) / 2.0,
        feasible=(feasible + feasible.T) / 2.0,
        penalty=(penalty + penalty.T) / 2.0,
        design_information=q,
        mean_gradient=dbar,
        sensitivity=kappa,
        score_variance=score_variance,
        score_error_covariance=score_error_covariance,
    )


def centering_penalty_factor(spec: OracleSpec) -> float:
    """Scalar multiplying the rank-one design term in the plug-in penalty."""
    return float(spec.c2 - 1.0 / spec.sensitivity)


def empirical_fixed_design_sandwich(
    gradient: np.ndarray,
    residuals: np.ndarray,
    spec: OracleSpec,
) -> np.ndarray:
    """Empirical version of the same-sample fixed-rho sandwich.

    The returned matrix estimates the covariance of
    ``sqrt(n) * (theta_hat-theta_0)``.  Divide by ``n`` for the covariance of
    ``theta_hat``.  This function does not account for adaptive-rho selection.
    """
    gradient = np.asarray(gradient, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if gradient.ndim != 2 or residuals.ndim != 1:
        raise ValueError("gradient must be 2D and residuals must be 1D")
    if gradient.shape[1] != residuals.size:
        raise ValueError("gradient and residuals must use the same observations")
    if spec.operational_fallback:
        raise ValueError("The PATP collision/fallback point has no plug-in sandwich")
    signs = np.sign(residuals)
    basis = np.vstack(
        [
            signs * np.abs(residuals) ** exponent - spec.means[index]
            for index, exponent in enumerate(spec.exponents)
        ]
    )
    psi = spec.weights @ basis
    dbar = np.mean(gradient, axis=1)
    chi = (gradient - dbar[:, None]) * psi[None, :]
    chi += dbar[:, None] * spec.sensitivity * residuals[None, :]
    meat = chi @ chi.T / residuals.size
    q = gradient @ gradient.T / residuals.size
    q_inv = np.linalg.inv(q)
    covariance = q_inv @ meat @ q_inv / spec.sensitivity**2
    return (covariance + covariance.T) / 2.0
