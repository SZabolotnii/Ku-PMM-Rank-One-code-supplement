"""Pure mathematical kernel for the Zhu--Zinde-Walsh AEPD model.

The dashboard imports this module, while the verification suite tests it without
executing Streamlit.  The parameterisation follows Zhu and Zinde-Walsh (2009):
``sigma`` is the AEPD scale parameter and is not, in general, the standard
deviation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import OptimizeResult, minimize
from scipy.special import (
    gammainc,
    gammaincc,
    gammainccinv,
    gammaincinv,
    gammaln,
)


@dataclass(frozen=True)
class AEPDMoments:
    """Mean and variance implied by the five AEPD parameters."""

    mean: float
    variance: float

    @property
    def standard_deviation(self) -> float:
        return float(np.sqrt(self.variance))


@dataclass(frozen=True)
class StandardizedAEPD:
    """Mean-zero, unit-variance Zhu--Zinde-Walsh AEPD parameters.

    ``tau`` is the probability to the left of the mode.  It is deliberately
    not named ``alpha`` here because the PATP studies use a distinct control
    parameter for the transition-polynomial basis.
    """

    mu: float
    sigma: float
    tau: float
    p_left: float
    p_right: float

    @property
    def parameters(self) -> tuple[float, float, float, float, float]:
        return (self.mu, self.sigma, self.tau, self.p_left, self.p_right)


def _validate_parameters(
    mu: float,
    sigma: float,
    alpha: float,
    p1: float,
    p2: float,
) -> None:
    values = np.asarray([mu, sigma, alpha, p1, p2], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("All AEPD parameters must be finite")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if p1 <= 0.0 or p2 <= 0.0:
        raise ValueError("p1 and p2 must be positive")


def _scalar_or_array(values: np.ndarray, original: object) -> float | np.ndarray:
    if np.asarray(original).ndim == 0:
        return float(np.asarray(values).reshape(-1)[0])
    return np.asarray(values)


def log_kep(p: float) -> float:
    """Return log K_EP(p) without avoidable overflow."""
    p = float(p)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError("p must be positive and finite")
    return float(-np.log(2.0) - np.log(p) / p - gammaln(1.0 + 1.0 / p))


def kep(p: float) -> float:
    """Normalising multiplier K_EP(p)."""
    return float(np.exp(log_kep(p)))


def alpha_star(alpha: float, p1: float, p2: float) -> float:
    """Continuity-adjusted split parameter alpha-star."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    log_k1 = log_kep(p1)
    log_k2 = log_kep(p2)
    numerator = np.log(alpha) + log_k1
    denominator = np.logaddexp(numerator, np.log1p(-alpha) + log_k2)
    return float(np.exp(numerator - denominator))


def aepd_logpdf(
    x: object,
    mu: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    p1: float = 2.0,
    p2: float = 2.0,
) -> float | np.ndarray:
    """Evaluate the exact AEPD log-density."""
    _validate_parameters(mu, sigma, alpha, p1, p2)
    original = x
    x_arr = np.asarray(x, dtype=float)
    a_star = alpha_star(alpha, p1, p2)
    z = (x_arr - mu) / sigma
    out = np.empty_like(z, dtype=float)

    left = z <= 0.0
    if np.any(left):
        scaled = np.abs(z[left] / (2.0 * a_star))
        out[left] = (
            np.log(alpha)
            - np.log(a_star)
            + log_kep(p1)
            - np.log(sigma)
            - scaled**p1 / p1
        )
    if np.any(~left):
        scaled = np.abs(z[~left] / (2.0 * (1.0 - a_star)))
        out[~left] = (
            np.log1p(-alpha)
            - np.log1p(-a_star)
            + log_kep(p2)
            - np.log(sigma)
            - scaled**p2 / p2
        )
    return _scalar_or_array(out, original)


def aepd_pdf(
    x: object,
    mu: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    p1: float = 2.0,
    p2: float = 2.0,
) -> float | np.ndarray:
    """Evaluate the exact, normalised AEPD density."""
    values = np.exp(aepd_logpdf(x, mu, sigma, alpha, p1, p2))
    return _scalar_or_array(np.asarray(values), x)


def aepd_cdf(
    x: object,
    mu: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    p1: float = 2.0,
    p2: float = 2.0,
) -> float | np.ndarray:
    """Evaluate the AEPD CDF through regularised incomplete gamma functions."""
    _validate_parameters(mu, sigma, alpha, p1, p2)
    original = x
    x_arr = np.asarray(x, dtype=float)
    a_star = alpha_star(alpha, p1, p2)
    z = (x_arr - mu) / sigma
    out = np.empty_like(z, dtype=float)

    left = z <= 0.0
    if np.any(left):
        u = np.abs(z[left]) / (2.0 * a_star)
        out[left] = alpha * gammaincc(1.0 / p1, u**p1 / p1)
    if np.any(~left):
        u = z[~left] / (2.0 * (1.0 - a_star))
        out[~left] = alpha + (1.0 - alpha) * gammainc(
            1.0 / p2, u**p2 / p2
        )
    out = np.clip(out, 0.0, 1.0)
    return _scalar_or_array(out, original)


def aepd_ppf(
    q: object,
    mu: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    p1: float = 2.0,
    p2: float = 2.0,
) -> float | np.ndarray:
    """Evaluate the exact AEPD quantile function."""
    _validate_parameters(mu, sigma, alpha, p1, p2)
    original = q
    q_arr = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q_arr)) or np.any((q_arr < 0.0) | (q_arr > 1.0)):
        raise ValueError("q must contain finite probabilities in [0, 1]")

    a_star = alpha_star(alpha, p1, p2)
    out = np.empty_like(q_arr, dtype=float)
    out[q_arr == 0.0] = -np.inf
    out[q_arr == 1.0] = np.inf
    at_mu = q_arr == alpha
    out[at_mu] = mu

    left = (q_arr > 0.0) & (q_arr < alpha)
    if np.any(left):
        y = gammainccinv(1.0 / p1, q_arr[left] / alpha)
        u = (p1 * y) ** (1.0 / p1)
        out[left] = mu - 2.0 * a_star * sigma * u

    right = (q_arr > alpha) & (q_arr < 1.0)
    if np.any(right):
        y = gammaincinv(1.0 / p2, (q_arr[right] - alpha) / (1.0 - alpha))
        u = (p2 * y) ** (1.0 / p2)
        out[right] = mu + 2.0 * (1.0 - a_star) * sigma * u
    return _scalar_or_array(out, original)


def _tail_power_integral(p: float, order: float, y: np.ndarray, *, upper: bool) -> np.ndarray:
    """Integral of u**order exp(-u**p/p) over a lower or upper tail."""
    shape = (order + 1.0) / p
    log_scale = ((order + 1.0) / p - 1.0) * np.log(p) + gammaln(shape)
    regularised = gammaincc(shape, y) if upper else gammainc(shape, y)
    return np.exp(log_scale) * regularised


def aepd_partial_first_moment(
    x: object,
    mu: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    p1: float = 2.0,
    p2: float = 2.0,
) -> float | np.ndarray:
    """Return E[X 1{X <= x}] exactly."""
    _validate_parameters(mu, sigma, alpha, p1, p2)
    original = x
    x_arr = np.asarray(x, dtype=float)
    a_star = alpha_star(alpha, p1, p2)
    out = np.empty_like(x_arr, dtype=float)

    left = x_arr <= mu
    if np.any(left):
        u = (mu - x_arr[left]) / (2.0 * a_star * sigma)
        y = u**p1 / p1
        probability = alpha * gammaincc(1.0 / p1, y)
        first_tail = _tail_power_integral(p1, 1.0, y, upper=True)
        out[left] = (
            mu * probability
            - 4.0 * alpha * a_star * sigma * kep(p1) * first_tail
        )

    if np.any(~left):
        left_first = (
            mu * alpha
            - 4.0
            * alpha
            * a_star
            * sigma
            * kep(p1)
            * _tail_power_integral(p1, 1.0, np.asarray(0.0), upper=True)
        )
        u = (x_arr[~left] - mu) / (2.0 * (1.0 - a_star) * sigma)
        y = u**p2 / p2
        right_probability = (1.0 - alpha) * gammainc(1.0 / p2, y)
        first_body = _tail_power_integral(p2, 1.0, y, upper=False)
        out[~left] = (
            left_first
            + mu * right_probability
            + 4.0
            * (1.0 - alpha)
            * (1.0 - a_star)
            * sigma
            * kep(p2)
            * first_body
        )
    return _scalar_or_array(out, original)


def compute_risk_metrics(
    eta: float,
    mu: float,
    sigma: float,
    alpha: float,
    p1: float,
    p2: float,
) -> tuple[float, float]:
    """Return downside VaR and ES = E[X | X <= VaR_eta]."""
    if not np.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between zero and one")
    var_eta = float(aepd_ppf(eta, mu, sigma, alpha, p1, p2))
    es_eta = float(aepd_partial_first_moment(var_eta, mu, sigma, alpha, p1, p2) / eta)
    return var_eta, es_eta


def _conditional_abs_moment(p: float, order: int) -> float:
    return float(
        np.exp(
            (order / p) * np.log(p)
            + gammaln((order + 1.0) / p)
            - gammaln(1.0 / p)
        )
    )


def aepd_moments(
    mu: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    p1: float = 2.0,
    p2: float = 2.0,
) -> AEPDMoments:
    """Return the exact mean and variance implied by the AEPD parameters."""
    _validate_parameters(mu, sigma, alpha, p1, p2)
    a_star = alpha_star(alpha, p1, p2)
    centered_first = (
        -alpha * 2.0 * a_star * sigma * _conditional_abs_moment(p1, 1)
        + (1.0 - alpha)
        * 2.0
        * (1.0 - a_star)
        * sigma
        * _conditional_abs_moment(p2, 1)
    )
    centered_second = (
        alpha * (2.0 * a_star * sigma) ** 2 * _conditional_abs_moment(p1, 2)
        + (1.0 - alpha)
        * (2.0 * (1.0 - a_star) * sigma) ** 2
        * _conditional_abs_moment(p2, 2)
    )
    variance = centered_second - centered_first**2
    if variance <= 0.0 or not np.isfinite(variance):
        raise FloatingPointError("AEPD variance evaluation failed")
    return AEPDMoments(mean=float(mu + centered_first), variance=float(variance))


@lru_cache(maxsize=None)
def standardized_aepd(tau: float, p_left: float, p_right: float) -> StandardizedAEPD:
    """Return AEPD parameters whose exact mean is zero and variance is one."""
    tau = float(tau)
    p_left = float(p_left)
    p_right = float(p_right)
    base = aepd_moments(0.0, 1.0, tau, p_left, p_right)
    sigma = float(1.0 / np.sqrt(base.variance))
    mu = float(-sigma * base.mean)
    standardized = StandardizedAEPD(mu, sigma, tau, p_left, p_right)
    check = aepd_moments(*standardized.parameters)
    if abs(check.mean) > 5e-13 or abs(check.variance - 1.0) > 5e-12:
        raise FloatingPointError("AEPD standardisation failed")
    return standardized


def aepd_from_uniforms(
    probabilities: object,
    mu: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    p1: float = 2.0,
    p2: float = 2.0,
) -> np.ndarray:
    """Transform supplied uniforms through the exact AEPD quantile function."""
    values = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("probabilities must lie strictly between zero and one")
    return np.asarray(aepd_ppf(values, mu, sigma, alpha, p1, p2), dtype=float)


def standardized_aepd_rvs(
    size: int,
    tau: float,
    p_left: float,
    p_right: float,
    *,
    rng: np.random.Generator | None = None,
    probabilities: object | None = None,
) -> np.ndarray:
    """Generate exact mean-zero, unit-variance AEPD draws.

    Supplying ``probabilities`` enables common-random-number and mirror-pair
    experiments.  Otherwise uniforms are generated from ``rng``.
    """
    if size < 1:
        raise ValueError("size must be positive")
    if probabilities is None:
        generator = np.random.default_rng() if rng is None else rng
        probabilities = generator.random(size)
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size != size:
        raise ValueError("probabilities must contain exactly size values")
    spec = standardized_aepd(tau, p_left, p_right)
    return aepd_from_uniforms(values, *spec.parameters)


@lru_cache(maxsize=None)
def aepd_fractional_moment(
    order: float,
    tau: float,
    p_left: float,
    p_right: float,
    signed: bool = False,
) -> float:
    """Return a centred standardized AEPD fractional moment by quadrature.

    The returned quantity is ``E|X|^order`` or
    ``E[sign(X)|X|^order]`` for the exact mean-zero/unit-variance law.  Orders
    above ``-1`` exist locally at zero; all positive tail moments exist for an
    exponential-power law.  Splitting at zero and at the AEPD mode protects
    both the fractional singularity and the density branch change.
    """
    order = float(order)
    if not np.isfinite(order) or order <= -1.0:
        raise ValueError("fractional moment order must be finite and greater than -1")
    spec = standardized_aepd(float(tau), float(p_left), float(p_right))

    def integrand(x: float) -> float:
        magnitude = abs(x) ** order
        parity = np.sign(x) if signed else 1.0
        return float(parity * magnitude * aepd_pdf(x, *spec.parameters))

    cuts = sorted({0.0, spec.mu})
    bounds = [-np.inf, *cuts, np.inf]
    total = 0.0
    error = 0.0
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        if lower == upper:
            continue
        value, local_error = quad(
            integrand,
            lower,
            upper,
            epsabs=2e-10,
            epsrel=2e-10,
            limit=500,
        )
        total += value
        error += local_error
    if not np.isfinite(total) or error > 2e-7 * max(1.0, abs(total)):
        raise FloatingPointError(
            f"AEPD fractional-moment quadrature failed: value={total}, error={error}"
        )
    if signed and abs(total) < 5e-14:
        return 0.0
    return float(total)


def aepd_nll(params: object, data: object) -> float:
    """Negative log-likelihood evaluated directly in log space."""
    mu, sigma, alpha, p1, p2 = np.asarray(params, dtype=float)
    try:
        values = np.asarray(aepd_logpdf(data, mu, sigma, alpha, p1, p2), dtype=float)
    except (ValueError, FloatingPointError, OverflowError):
        return float("inf")
    if np.any(~np.isfinite(values)):
        return float("inf")
    return float(-np.sum(values))


def fit_aepd_mle(data: object) -> OptimizeResult:
    """Fit AEPD by multi-start bounded maximum likelihood.

    Only successful, finite optimiser results can be returned.  A RuntimeError
    is raised if every start fails, which prevents a failed fit from silently
    entering the dashboard's risk calculations.
    """
    sample = np.asarray(data, dtype=float).reshape(-1)
    sample = sample[np.isfinite(sample)]
    if sample.size < 20:
        raise ValueError("At least 20 finite observations are required for AEPD MLE")
    scale0 = max(float(np.std(sample, ddof=0)), np.finfo(float).eps ** 0.25)
    median = float(np.median(sample))
    mean = float(np.mean(sample))
    starts = [
        np.array([median, scale0, 0.50, 1.5, 1.5]),
        np.array([mean, scale0, 0.50, 2.0, 2.0]),
        np.array([median, scale0, 0.35, 1.2, 2.0]),
        np.array([median, scale0, 0.65, 2.0, 1.2]),
    ]
    bounds = [
        (None, None),
        (scale0 * 1e-4, scale0 * 1e4),
        (0.01, 0.99),
        (0.2, 10.0),
        (0.2, 10.0),
    ]
    candidates: list[OptimizeResult] = []
    diagnostics: list[str] = []
    for start in starts:
        result = minimize(
            aepd_nll,
            start,
            args=(sample,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2_000, "ftol": 1e-11, "gtol": 1e-7},
        )
        diagnostics.append(f"success={result.success}, fun={result.fun}, message={result.message}")
        if result.success and np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("AEPD MLE failed for every start: " + " | ".join(diagnostics))
    best = min(candidates, key=lambda result: float(result.fun))
    best.n_successful_starts = len(candidates)
    best.n_starts = len(starts)
    return best


def aepd_rvs(
    size: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    p1: float = 2.0,
    p2: float = 2.0,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate AEPD draws by exact inverse transform sampling."""
    if size < 1:
        raise ValueError("size must be positive")
    generator = np.random.default_rng() if rng is None else rng
    probabilities = generator.random(size)
    return np.asarray(aepd_ppf(probabilities, mu, sigma, alpha, p1, p2), dtype=float)
