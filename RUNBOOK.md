# RUNBOOK

Execution notes for the Ku-PMM-Rank-One verification supplement.
Last updated: 2026-10-12 (manuscript revision; stored simulation results unchanged).

Three levels of checking, in increasing cost. Level 1 is enough to verify the
manuscript; levels 2 and 3 exist so that "verify" does not have to mean "trust the
stored files". Run everything from the repository root.

---

## Level 1 — the printed numbers (seconds, no simulation)

```sh
pip install -r requirements.txt
python scripts/verify_reported_values.py
```

Expected: `all 241 reported values reproduced from stored results`, exit 0.

Each check names the manuscript label it comes from (`tab:dir`, `sec:coverage`, ...).
The expected values are transcribed from the manuscript at the precision it prints
them; the actual values are recomputed from `results/`. Add `-v` to list every check.

```sh
python scripts/verify_k1.py                  # 12/12 claim-level checks
python -m unittest discover -s tests         # 30 tests
```

The October 12 revision adds three deterministic checks for Corollary 8:

```sh
python -m unittest discover -s tests -p test_wald_consequence.py -v
```

Quadrature of influence-function products in a uniform-noise example checks the
joint Wald eigenvalues, contrast coverage distortion, and invariance under a linear
coordinate change. This does not regenerate simulation results or establish
finite-sample coverage. The test is synchronized with the research repository.

## Level 1b — regenerate the tables and figures (seconds)

```sh
python scripts/make_tex_tables.py            # -> output/tables/coverage_s{2,3}{,_en}.tex
python scripts/make_tex_tables.py --check    # exit 1 if output/ is stale against results/
python scripts/make_paper_figures.py         # -> output/figures/fig_*_{uk,en}.{pdf,png}
```

Both scripts read `results/` only and contain no typed-in numbers. Tables and figures exist
in a Ukrainian version (decimal comma; figures `_uk`) and an English one (`_en`).

## Level 2 — deterministic recomputation (seconds to minutes, no Monte Carlo)

| Command | Writes | Manuscript element |
|---|---|---|
| `python scripts/build_patp_plugin_theory_artifacts.py` | `results/patp_plugin_sandwich_predictions.csv` | closed-form `g_or`, `g_fe` and the centring increment per scenario |
| `python scripts/k1_design_only_RD.py` | `results/k1_design_factor_RD*.csv` | Table 1 (`R_D` for seven designs); 315 cells, 273 with a defined penalty-to-gain ratio (§8.7) |
| `python scripts/k1_signal_classes_RD.py` | `results/k1_signal_classes_RD*.csv` | Table 2 (eight signal models); 360 cells, 312 with a defined penalty-to-gain ratio |

## Level 3 — Monte Carlo (minutes to hours)

The seeds and replication counts of the stored runs are recorded in
`results/*_metadata.json` and are the scripts' defaults unless stated below. Rerunning
with the same seeds is expected to reproduce the stored files up to floating-point
differences between platforms; bit-for-bit agreement is not guaranteed.

**Publication run of the benchmark.** It writes the replication files that are not
shipped, plus the summaries that are:

```sh
python scripts/run_mc_aepd_patp_s2.py        # seed 20260841, M = 1000, n in {200, 500, 1000, 5000}
python scripts/run_mc_aepd_patp_s3.py        # seed 20260842
```

Scripts that read this run:

| Command | Manuscript element |
|---|---|
| `python scripts/k1_penalty_direction.py` | Table 4: variance ratio along and off the increment direction (needs the replication files) |
| `python scripts/k1_convergence_rate.py` | Table 5 (trace metric, S = 2) and Table 8 (S = 3) |
| `python scripts/k1_s3_gap_diagnosis.py` | §8.8: the S200 checks and the bias of the estimated `g` |

**Independent Monte Carlo runs:**

| Command | Manuscript element |
|---|---|
| `python scripts/k1_rho_direction_panel.py` | Figure 2: direction of the increment at fixed `rho` |
| `python scripts/k1_rank_contrast.py` | §8.4: rank of the excess under three schemes of moment conditions |
| `python scripts/k1_sandwich_coverage.py --publication` | Table 6: coverage, S = 2 |
| `python scripts/k1_sandwich_coverage.py --order 3 --publication` | Table 7: coverage, S = 3 |
| `python scripts/k1_s3_rho_sweep.py --reps 250` | Figure 3: diagnostic sweep over `rho` at S = 3 (the stored run used 250 replications; the default is 400) |
| `python scripts/k1_constant_signal_check.py` | the constant-signal corollary (estimated against imposed centring) |

After a Level 3 run, repeat Level 1: `verify_reported_values.py` then compares the
manuscript with the regenerated results.

## Scope notes

- The order-3 manifest is symmetric. `load_manifest` refuses asymmetric scenarios at
  S = 3 unless `allow_asymmetric_order3=True`; that opt-in is for diagnostics only.
- All results are for a fixed basis parameter `rho`; choosing `rho` from the data is not
  covered by the theorem or by these scripts.
