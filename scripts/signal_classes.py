"""Smooth signal models r(t; theta) with analytic gradients, beyond the harmonic bench.

The rank-one theorem uses the signal only through its gradient at the truth:
``Q = mean(d d')`` and ``dbar = mean(d)``.  Each class here supplies the model and
that gradient, so the design quantities can be computed for signals other than the
one simulated in the Monte Carlo.  The harmonic class reproduces
``patp_experiment_core.signal`` / ``signal_gradient``; that module is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SignalClass:
    label: str
    params: tuple[str, ...]
    theta: tuple[float, ...]
    model: Callable[[np.ndarray, np.ndarray], np.ndarray]
    gradient: Callable[[np.ndarray, np.ndarray], np.ndarray]
    has_constant_component: bool = False


def _envelope(t: np.ndarray) -> np.ndarray:
    return 1.0 + 0.3 * np.cos(np.pi * t)


def _harmonic(theta, t):
    a, w, phi = theta
    return a * _envelope(t) * np.cos(w * t + phi)


def _harmonic_grad(theta, t):
    a, w, phi = theta
    e, angle = _envelope(t), w * t + phi
    return np.vstack([e * np.cos(angle), -a * e * t * np.sin(angle), -a * e * np.sin(angle)])


def _harmonic_offset(theta, t):
    return _harmonic(theta[:3], t) + theta[3]


def _harmonic_offset_grad(theta, t):
    return np.vstack([_harmonic_grad(theta[:3], t), np.ones_like(t)])


def _tone(theta, t):
    a, w, phi = theta
    return a * np.cos(w * t + phi)


def _tone_grad(theta, t):
    a, w, phi = theta
    angle = w * t + phi
    return np.vstack([np.cos(angle), -a * t * np.sin(angle), -a * np.sin(angle)])


def _two_tone(theta, t):
    return _tone(theta[:3], t) + _tone(theta[3:], t)


def _two_tone_grad(theta, t):
    return np.vstack([_tone_grad(theta[:3], t), _tone_grad(theta[3:], t)])


def _damped(theta, t):
    a, lam, w, phi = theta
    return a * np.exp(-lam * t) * np.cos(w * t + phi)


def _damped_grad(theta, t):
    a, lam, w, phi = theta
    decay, angle = np.exp(-lam * t), w * t + phi
    return np.vstack([
        decay * np.cos(angle),
        -a * t * decay * np.cos(angle),
        -a * t * decay * np.sin(angle),
        -a * decay * np.sin(angle),
    ])


def _chirp(theta, t):
    a, w, beta, phi = theta
    return a * np.cos(w * t + beta * t**2 + phi)


def _chirp_grad(theta, t):
    a, w, beta, phi = theta
    angle = w * t + beta * t**2 + phi
    return np.vstack([
        np.cos(angle),
        -a * t * np.sin(angle),
        -a * t**2 * np.sin(angle),
        -a * np.sin(angle),
    ])


def _pulse(theta, t):
    a, tau, width = theta
    return a * np.exp(-((t - tau) ** 2) / (2.0 * width**2))


def _pulse_grad(theta, t):
    a, tau, width = theta
    shape = np.exp(-((t - tau) ** 2) / (2.0 * width**2))
    return np.vstack([
        shape,
        a * shape * (t - tau) / width**2,
        a * shape * (t - tau) ** 2 / width**3,
    ])


def _linear(theta, t):
    return theta[0] * t + theta[1] * t**2


def _linear_grad(theta, t):
    return np.vstack([t, t**2])


def _constant(theta, t):
    return np.full_like(t, theta[0])


def _constant_grad(theta, t):
    return np.ones((1, t.size))


SIGNAL_CLASSES: tuple[SignalClass, ...] = (
    SignalClass("harmonic_envelope", ("A", "Omega", "Phi"), (2.0, 1.5, 0.5),
                _harmonic, _harmonic_grad),
    SignalClass("harmonic_envelope_offset", ("A", "Omega", "Phi", "C"), (2.0, 1.5, 0.5, 0.5),
                _harmonic_offset, _harmonic_offset_grad, has_constant_component=True),
    SignalClass("two_tone", ("A1", "Omega1", "Phi1", "A2", "Omega2", "Phi2"),
                (2.0, 1.5, 0.5, 1.0, 4.0, 1.0), _two_tone, _two_tone_grad),
    SignalClass("damped_sinusoid", ("A", "lambda", "Omega", "Phi"), (2.0, 0.5, 1.5, 0.5),
                _damped, _damped_grad),
    SignalClass("linear_chirp", ("A", "Omega", "beta", "Phi"), (2.0, 1.5, 1.0, 0.5),
                _chirp, _chirp_grad),
    SignalClass("gaussian_pulse", ("A", "tau", "w"), (2.0, 1.0, 0.3),
                _pulse, _pulse_grad),
    SignalClass("linear_no_intercept", ("theta1", "theta2"), (1.0, 0.5),
                _linear, _linear_grad),
    SignalClass("constant", ("C",), (0.5,), _constant, _constant_grad,
                has_constant_component=True),
)


def design_summary(gradient: np.ndarray) -> dict[str, object]:
    """Q, dbar, the penalty direction, R_D and the projection norm of the constant.

    ``direction = Q^{-1} dbar`` is the least-squares coefficient vector of the constant
    function 1 regressed on the gradient columns; ``projection = dbar' Q^{-1} dbar`` is
    the squared design-norm of that projection and lies in [0, 1].
    """
    n = gradient.shape[1]
    q = gradient @ gradient.T / n
    q_inv = np.linalg.inv(q)
    dbar = gradient.mean(axis=1)
    direction = q_inv @ dbar
    return {
        "q": q, "q_inv": q_inv, "dbar": dbar, "direction": direction,
        "design_factor_R": float(direction @ direction / np.trace(q_inv)),
        "projection_norm_sq": float(dbar @ direction),
        "cond_Q": float(np.linalg.cond(q)),
    }
