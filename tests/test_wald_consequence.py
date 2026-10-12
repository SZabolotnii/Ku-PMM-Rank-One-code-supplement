"""Independent quadrature checks of cor:wald, not finite-sample coverage tests."""
import unittest

import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm


def influence_covariances(gradient):
    # Uniform[-1,1], score in span {e,e^3}; quadrature integrates all products exactly.
    e, weights = np.polynomial.legendre.leggauss(12)
    weights = weights / 2
    basis = np.column_stack((e, e**3))
    body = basis.T @ (weights[:, None] * basis)
    b = np.array([1.0, 3 * np.dot(weights, e**2)])
    h = np.linalg.solve(body, b)
    kappa = b @ h
    psi = basis @ h
    c2 = np.dot(weights, e**2)
    q = gradient @ gradient.T / gradient.shape[1]
    mean = gradient.mean(axis=1)
    qi = np.linalg.inv(q)
    oracle = np.zeros_like(q)
    feasible = np.zeros_like(q)
    for d in gradient.T:
        influence_or = (qi @ d)[:, None] * psi[None, :] / kappa
        influence_fe = influence_or + (qi @ mean)[:, None] * (e - psi / kappa)
        oracle += (influence_or * weights) @ influence_or.T
        feasible += (influence_fe * weights) @ influence_fe.T
    return oracle / gradient.shape[1], feasible / gradient.shape[1], q, mean, c2, 1 / (c2 * kappa)


class WaldConsequenceTests(unittest.TestCase):
    def test_joint_wald_eigenvalues_including_degenerate_cases(self):
        x = np.linspace(-1, 2, 41)
        centered = np.linspace(-1, 1, 41)
        for gradient in (np.array([np.ones_like(x)]), np.array([np.ones_like(x), x]),
                         np.array([x, x**2]), np.array([centered, centered**3])):
            with self.subTest(dimension=gradient.shape[0], mean=gradient.mean(axis=1)):
                oracle, feasible, q, mean, _, g = influence_covariances(gradient)
                pi = mean @ np.linalg.solve(q, mean)
                expected = np.ones(gradient.shape[0])
                expected[-1] += (1 - g) * pi / g
                np.testing.assert_allclose(eigh(feasible, oracle, eigvals_only=True), expected,
                                           rtol=1e-11, atol=1e-11)

    def test_contrast_variance_and_coverage_direction(self):
        x = np.linspace(-1, 2, 41)
        oracle, feasible, q, mean, c2, g = influence_covariances(np.array([np.ones_like(x), x]))
        for contrast, affected in ((np.array([1., 0.]), True), (np.array([0., 1.]), False)):
            delta = c2 * (1 - g) * (contrast @ np.linalg.solve(q, mean))**2 / (contrast @ oracle @ contrast)
            self.assertAlmostEqual((contrast @ feasible @ contrast) / (contrast @ oracle @ contrast), 1 + delta)
            coverage = 2 * norm.cdf(norm.ppf(.975) / np.sqrt(1 + delta)) - 1
            if affected:
                self.assertLess(coverage, .95)
            else:
                self.assertAlmostEqual(coverage, .95)

    def test_joint_distortion_is_invariant_under_change_of_coordinates(self):
        x = np.linspace(-1, 2, 41)
        gradient = np.array([x, x**2])
        transform = np.array([[3., .5], [-.2, .4]])
        original = influence_covariances(gradient)
        changed = influence_covariances(transform.T @ gradient)
        for result in (original, changed):
            _, _, q, mean, _, _ = result
            self.assertGreater(mean @ np.linalg.solve(q, mean), 0)
        np.testing.assert_allclose(eigh(original[1], original[0], eigvals_only=True),
                                   eigh(changed[1], changed[0], eigvals_only=True), rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
