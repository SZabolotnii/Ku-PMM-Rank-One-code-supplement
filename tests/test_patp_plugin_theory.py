import unittest

import numpy as np

from scripts.aepd_patp_core import AEPDScenario, build_aepd_patp_spec
from scripts.patp_experiment_core import TRUE_THETA, signal_design, signal_gradient
from scripts.patp_plugin_theory import (
    empirical_fixed_design_sandwich,
    fixed_design_plugin_sandwich,
    two_step_sandwich,
)


class PATPPluginTheoryTests(unittest.TestCase):
    def test_optimal_score_identities_and_psd_rank_one_penalty(self):
        scenario = AEPDScenario("A2L", 0.5, 1.2, 3.0, "test")
        spec = build_aepd_patp_spec(scenario, 1.0, 2)
        t, envelope = signal_design(1000)
        gradient = signal_gradient(TRUE_THETA, t, envelope)
        result = fixed_design_plugin_sandwich(gradient, spec)
        self.assertAlmostEqual(result.score_variance, spec.sensitivity, places=11)
        self.assertAlmostEqual(result.score_error_covariance, 1.0, places=11)
        self.assertTrue(np.allclose(result.feasible, result.oracle + result.penalty))
        eigenvalues = np.linalg.eigvalsh(result.penalty)
        self.assertGreaterEqual(eigenvalues.min(), -1e-12)
        self.assertLessEqual(np.sum(eigenvalues > 1e-11), 1)

    def test_zero_mean_gradient_gives_oracle_equivalence(self):
        scenario = AEPDScenario("S500", 0.5, 5.0, 5.0, "test")
        spec = build_aepd_patp_spec(scenario, 1.0, 3)
        x = np.linspace(-2.0, 2.0, 1001)
        gradient = np.vstack((x, x**3, np.sin(np.pi * x)))
        result = fixed_design_plugin_sandwich(gradient, spec)
        self.assertTrue(np.allclose(result.mean_gradient, 0.0, atol=1e-14))
        self.assertTrue(np.allclose(result.feasible, result.oracle, atol=1e-13))

    def test_gaussian_neutrality_eliminates_centering_penalty(self):
        scenario = AEPDScenario("A0", 0.5, 2.0, 2.0, "test")
        spec = build_aepd_patp_spec(scenario, 0.0, 2)
        t, envelope = signal_design(400)
        result = fixed_design_plugin_sandwich(
            signal_gradient(TRUE_THETA, t, envelope), spec
        )
        self.assertAlmostEqual(spec.c2 * spec.sensitivity, 1.0, places=11)
        self.assertTrue(np.allclose(result.penalty, 0.0, atol=1e-12))

    def test_generic_two_step_sandwich_matches_effective_score_loading(self):
        a = np.array([[2.0, 0.2], [0.1, 1.5]])
        c = np.array([[0.3], [-0.2]])
        h = np.array([[1.25]])
        d = np.array([[0.4, -0.1]])
        raw = np.array(
            [
                [1.0, 0.2, 0.1, -0.1, 0.0],
                [0.2, 1.4, -0.2, 0.0, 0.1],
                [0.1, -0.2, 0.8, 0.2, -0.1],
                [-0.1, 0.0, 0.2, 0.7, 0.15],
                [0.0, 0.1, -0.1, 0.15, 0.6],
            ]
        )
        omega = raw @ raw.T
        result = two_step_sandwich(a, c, h, d, omega)
        loading = np.concatenate(
            (np.eye(2), -c / h[0, 0], -(c / h[0, 0]) @ d), axis=1
        )
        a_inv = np.linalg.inv(a)
        expected = a_inv @ loading @ omega @ loading.T @ a_inv.T
        self.assertTrue(np.allclose(result, expected))

    def test_empirical_sandwich_is_symmetric_psd(self):
        scenario = AEPDScenario("A2L", 0.5, 1.2, 3.0, "test")
        spec = build_aepd_patp_spec(scenario, 1.0, 2)
        t, envelope = signal_design(500)
        gradient = signal_gradient(TRUE_THETA, t, envelope)
        residuals = np.linspace(-2.0, 2.0, 500)
        covariance = empirical_fixed_design_sandwich(gradient, residuals, spec)
        self.assertTrue(np.allclose(covariance, covariance.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)


if __name__ == "__main__":
    unittest.main()
