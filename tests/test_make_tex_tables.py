"""Tests for the generated manuscript artifacts of section 8.

The bold rule of the coverage tables, the generated files being up to date with
results/, and the facts that the captions of the two data figures state.
"""
import unittest

from scripts import make_paper_figures as figs
from scripts import make_tex_tables as tabs


class BoldRuleTests(unittest.TestCase):
    def test_row_within_noise_has_no_bold(self):
        self.assertEqual(tabs.bold_mask([0.951, 0.952, 0.939]), [False, False, False])

    def test_clear_loser_leaves_the_tied_winners_bold(self):
        self.assertEqual(tabs.bold_mask([0.946, 0.946, 0.931]), [True, True, False])

    def test_winner_is_closest_to_nominal_not_highest(self):
        # 0.975 is higher but 0.025 from 0.95; 0.945 is 0.005 away and wins by more than 0.014
        self.assertEqual(tabs.bold_mask([0.945, 0.975, 0.900]), [True, False, False])

    def test_difference_below_noise_is_a_tie(self):
        # 0.008 and 0.020 from 0.95 differ by 0.012 < 0.014, so both are bold
        self.assertEqual(tabs.bold_mask([0.958, 0.930, 0.900]), [True, True, False])

    def test_rule_works_on_printed_values(self):
        # 0.9395 and 0.9404 print as 0.940 and are tied at printed precision
        self.assertEqual(tabs.bold_mask([0.9395, 0.9404, 0.900]), [True, True, False])


class GeneratedTablesTests(unittest.TestCase):
    def test_files_are_up_to_date_with_results(self):
        for name, text in tabs.build().items():
            path = tabs.OUT / name
            self.assertTrue(path.exists(), f"{name} missing - run scripts/make_tex_tables.py")
            self.assertEqual(path.read_text(encoding="utf-8"), text,
                             f"{name} is stale - run scripts/make_tex_tables.py")

    def test_english_tables_bold_the_same_cells(self):
        built = tabs.build()
        for name in tabs.TABLES:
            uk = built[name].splitlines()
            en = built[name.replace(".tex", "_en.tex")].splitlines()
            self.assertEqual([line.count("mathbf") for line in uk],
                             [line.count("mathbf") for line in en], name)
            self.assertNotIn("{,}", "\n".join(en))

    def test_naive_form_never_wins_at_n5000_in_symmetric_regimes_with_increment(self):
        body = tabs.build()["coverage_s3.tex"]
        rows = [line for line in body.splitlines() if "& $5000$" in line]
        self.assertEqual(len(rows), 4)
        for line in rows:
            naive = line.split("&")[-1]
            self.assertNotIn("mathbf", naive, line)


class FigureDataTests(unittest.TestCase):
    def test_rho_panel_prediction_inside_interval_and_unity_outside(self):
        for sid, d in figs.rho_panel_data().items():
            self.assertEqual(len(d["rho"]), 10, sid)
            self.assertNotIn(0.5, d["rho"])
            for p, lo, hi in zip(d["pred"], d["lo"], d["hi"]):
                self.assertTrue(lo <= p <= hi, sid)
                self.assertGreater(lo, 1.0, sid)

    def test_s3_sweep_excess_is_large_only_below_one_half(self):
        data = figs.s3_sweep_data()
        x, y = data[("S200", 8000)]
        below = [e for r, e in zip(x, y) if r <= 0.2]
        above = [e for r, e in zip(x, y) if r >= 0.6]
        self.assertGreaterEqual(min(below), 0.1)
        self.assertLess(max(abs(e) for e in above), 0.004)


if __name__ == "__main__":
    unittest.main()
