"""Regression tests on the K1 stage-C artifacts (results/k1_*).

They read the CSVs written by scripts/k1_*.py; they do not re-run the Monte
Carlo.  If an artifact is missing the test is skipped with a message naming the
script that regenerates it.
"""

import csv
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _rows(name):
    path = ROOT / "results" / name
    if not path.exists():
        raise unittest.SkipTest(f"{name} missing — run scripts/k1_*.py")
    with path.open() as handle:
        return list(csv.DictReader(handle))


class K1DesignFactorTests(unittest.TestCase):
    def test_identity_penalty_over_one_minus_oracle_equals_RD_for_every_noise_law(self):
        rows = _rows("k1_design_factor_RD.csv")
        gaps = [float(r["identity_gap"]) for r in rows if r["identity_gap"] != "nan"]
        self.assertGreater(len(gaps), 100)
        self.assertLess(max(gaps), 1e-10)

    def test_RD_varies_across_designs(self):
        rows = [r for r in _rows("k1_design_factor_RD_designs.csv") if r["sample_size"] == "5000"]
        values = {r["design"]: float(r["design_factor_R"]) for r in rows}
        self.assertAlmostEqual(values["track_window_[0.01,2]_env0.3"], 0.219077, places=4)
        self.assertGreater(max(values.values()) / min(values.values()), 5.0)


class K1PenaltyDirectionTests(unittest.TestCase):
    def test_complement_directions_are_oracle_precise_at_largest_n(self):
        rows = [r for r in _rows("k1_penalty_direction.csv") if r["sample_size"] == "5000"
                and r["empirical_ratio_complement_vs_oracle_empirical"] != "nan"]
        self.assertGreaterEqual(len(rows), 4)
        for r in rows:
            self.assertLess(abs(float(r["empirical_ratio_complement_vs_oracle_empirical"]) - 1.0), 0.02, r["scenario_id"])

    def test_penalty_along_u_matches_prediction_where_material(self):
        rows = [r for r in _rows("k1_penalty_direction.csv") if r["sample_size"] == "5000"
                and float(r["predicted_ratio_along_u"]) > 1.03]
        self.assertGreaterEqual(len(rows), 2)
        for r in rows:
            pred = float(r["predicted_ratio_along_u"]) - 1.0
            emp = float(r["empirical_ratio_along_u_vs_oracle_empirical"]) - 1.0
            self.assertGreater(emp, 0.0, r["scenario_id"])
            self.assertLess(abs(emp - pred) / pred, 0.35, r["scenario_id"])
            self.assertLess(float(r["excess_second_over_max"]), 0.05, r["scenario_id"])


class K1ConvergenceTests(unittest.TestCase):
    def test_s2_prediction_inside_bootstrap_ci_at_largest_n(self):
        rows = [r for r in _rows("k1_convergence_rate.csv") if r["order"] == "2" and r["sample_size"] == "5000"]
        self.assertEqual(sum(int(r["prediction_inside_ci"]) for r in rows), len(rows))

    def test_s3_deviation_is_recorded_not_hidden(self):
        # S=3 at N=5000: the first-order prediction lies OUTSIDE the bootstrap CI in
        # every scenario, and the sign of the gap is two-sided (above the prediction
        # near the Gaussian regime S075-S200, below it for platykurtic S300/S500).
        rows = [r for r in _rows("k1_convergence_rate.csv") if r["order"] == "3" and r["sample_size"] == "5000"]
        self.assertGreaterEqual(len(rows), 4)
        self.assertEqual(sum(int(r["prediction_inside_ci"]) for r in rows), 0)
        gaps = [float(r["gap_empirical_minus_predicted"]) for r in rows]
        self.assertTrue(any(g > 0 for g in gaps) and any(g < 0 for g in gaps))


if __name__ == "__main__":
    unittest.main()
