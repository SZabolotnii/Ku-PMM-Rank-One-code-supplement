"""Tests for the signal classes behind tab:classes and the projection reading of Q^{-1} dbar."""

import csv
import unittest
from pathlib import Path

import numpy as np

from scripts.patp_experiment_core import TRUE_THETA, signal_design, signal_gradient
from scripts.signal_classes import SIGNAL_CLASSES, design_summary

ROOT = Path(__file__).resolve().parents[1]


class SignalClassGradientTests(unittest.TestCase):
    def test_analytic_gradients_match_central_differences(self):
        t = np.linspace(0.01, 2.0, 400)
        step = 1e-6
        for cls in SIGNAL_CLASSES:
            theta = np.asarray(cls.theta, dtype=float)
            numeric = np.vstack([
                (cls.model(theta + step * e, t) - cls.model(theta - step * e, t)) / (2 * step)
                for e in np.eye(theta.size)
            ])
            self.assertLess(np.max(np.abs(numeric - cls.gradient(theta, t))), 1e-7, cls.label)

    def test_harmonic_class_is_the_monte_carlo_bench(self):
        t, envelope = signal_design(1000)
        bench = signal_gradient(TRUE_THETA, t, envelope)
        self.assertTrue(np.array_equal(SIGNAL_CLASSES[0].gradient(TRUE_THETA, t), bench))


class ProjectionTests(unittest.TestCase):
    def test_direction_is_the_projection_of_the_constant(self):
        t = np.linspace(0.01, 2.0, 1000)
        for cls in SIGNAL_CLASSES:
            gradient = cls.gradient(np.asarray(cls.theta, dtype=float), t)
            summary = design_summary(gradient)
            lstsq = np.linalg.lstsq(gradient.T, np.ones(t.size), rcond=None)[0]
            self.assertLess(np.max(np.abs(summary["direction"] - lstsq)), 1e-9, cls.label)
            residual = np.mean((1.0 - gradient.T @ summary["direction"]) ** 2)
            self.assertAlmostEqual(summary["projection_norm_sq"] + residual, 1.0, places=10)
            self.assertGreaterEqual(summary["projection_norm_sq"], 0.0)
            self.assertLessEqual(summary["projection_norm_sq"], 1.0 + 1e-12)

    def test_constant_component_takes_the_whole_direction(self):
        t = np.linspace(0.01, 2.0, 1000)
        for cls in SIGNAL_CLASSES:
            if not cls.has_constant_component:
                continue
            summary = design_summary(cls.gradient(np.asarray(cls.theta, dtype=float), t))
            unit = np.zeros(len(cls.theta))
            unit[-1] = 1.0
            self.assertLess(np.max(np.abs(summary["direction"] - unit)), 1e-10, cls.label)
            self.assertAlmostEqual(summary["projection_norm_sq"], 1.0, places=10)

    def test_projection_norm_is_invariant_under_rescaling_but_R_D_is_not(self):
        t = np.linspace(0.01, 2.0, 1000)
        gradient = SIGNAL_CLASSES[0].gradient(np.asarray(SIGNAL_CLASSES[0].theta), t)
        scaled = np.diag([1.0, 100.0, 0.01]) @ gradient
        base, other = design_summary(gradient), design_summary(scaled)
        self.assertAlmostEqual(base["projection_norm_sq"], other["projection_norm_sq"], places=9)
        self.assertGreater(abs(base["design_factor_R"] - other["design_factor_R"]), 0.01)


class SignalClassArtifactTests(unittest.TestCase):
    def test_stored_table_matches_the_harmonic_bench_value(self):
        path = ROOT / "results/k1_signal_classes_RD.csv"
        if not path.exists():
            raise unittest.SkipTest("run scripts/k1_signal_classes_RD.py")
        with path.open() as handle:
            rows = {r["signal_class"]: r for r in csv.DictReader(handle) if r["sample_size"] == "5000"}
        self.assertEqual(len(rows), len(SIGNAL_CLASSES))
        self.assertAlmostEqual(float(rows["harmonic_envelope"]["design_factor_R"]), 0.219077, places=5)
        self.assertAlmostEqual(float(rows["constant"]["design_factor_R"]), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
