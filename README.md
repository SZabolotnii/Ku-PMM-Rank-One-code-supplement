# Ku-PMM-Rank-One Code Supplement

Public verification supplement for the manuscript

**Rank-one variance inflation in polynomial estimation of signal parameters with a
sign-preserving fractional-power basis**
*Serhii Zabolotnii* (manuscript in preparation)

## The result in one line

When the moment characteristics of non-Gaussian noise are estimated from the residuals
of a preliminary least-squares fit, instead of being known, the asymptotic covariance of
the polynomial (PMM) estimator grows by exactly one term,

    Sigma_fe - Sigma_or = c2 (1 - g_or) Q^{-1} dbar dbar' Q^{-1},

a positive semi-definite matrix of rank at most one. In the trace metric this gives
`g_fe = g_or + (1 - g_or) R_D` with `R_D = |Q^{-1} dbar|^2 / tr Q^{-1}`, a weight that
depends on the design only.

## Start here

```sh
pip install -r requirements.txt
python scripts/verify_reported_values.py
```

It recomputes **241 numbers printed in the manuscript** from the stored results in
`results/` and compares each with the printed value at the precision the manuscript
prints it. It needs no simulation and no network, and runs in about a second. A
mismatch means the manuscript and the stored results disagree.

Two further checks, also seconds:

```sh
python scripts/verify_k1.py                  # 12 claim-level checks over results/
python -m unittest discover -s tests         # 27 unit and regression tests
```

## Layout

| Path | What it holds |
|---|---|
| `scripts/verify_reported_values.py` | every printed number against the stored results |
| `scripts/verify_k1.py` | claim-level gate: rank one, direction, identity for `R_D`, coverage, rank contrast |
| `aepd_core.py` | asymmetric exponential-power (AEPD) noise generator |
| `scripts/patp_experiment_core.py` | harmonic signal model, NLS fit, one-step refinement |
| `scripts/patp_plugin_theory.py` | closed-form sandwich of the feasible estimator |
| `scripts/aepd_patp_core.py`, `patp_adaptive_core.py` | scenario manifests and the Monte Carlo study |
| `scripts/signal_classes.py` | signal models beyond the harmonic bench (design quantities only) |
| `scripts/run_mc_aepd_patp_s{2,3}.py` | the publication Monte Carlo run of the benchmark |
| `scripts/k1_*.py` | one script per manuscript table, figure or check (see `RUNBOOK.md`) |
| `scripts/make_tex_tables.py`, `make_paper_figures.py` | regenerate the coverage tables and the figures into `output/` |
| `configs/aepd_patp_s{2,3}.json` | the nine asymmetric (`A*`, S = 2) and six symmetric (`S*`, S = 3) noise scenarios |
| `results/` | stored results of every run cited by the manuscript, with metadata (seeds, replications) |
| `output/` | tables and figures regenerated from `results/` |
| `tests/` | unit tests and regression tests on the stored results |

The prefix `k1_` is the internal code of this article; it is kept because the metadata
files in `results/` refer to the scripts by these names.

## What is not shipped

The replication-level output of the publication run (`results/aepd_patp_s2_replications.csv`,
145 MB, and `..._s3_replications.csv`, 89 MB) exceeds the repository size limits. Only one
script needs it: `k1_penalty_direction.py`, which projects the empirical covariance on the
increment direction (Table 4). The summary files of the same run are shipped, and
`scripts/run_mc_aepd_patp_s{2,3}.py` regenerates the replication files; see `RUNBOOK.md`.

## Bold in the coverage tables

`scripts/make_tex_tables.py` marks in bold the covariance form whose joint coverage is
closest to the nominal 0.95, together with every form within two Monte Carlo standard
errors of it (2 x 0.007). A row in which all three forms lie within that band is a tie at
noise level and carries no bold. The rule lives in the generator, never in hand-edited
LaTeX, and `tests/test_make_tex_tables.py` checks it.

## Requirements

numpy, scipy and matplotlib (`requirements.txt`). Checked in clean environments with
Python 3.13 and 3.14 (numpy 2.5, scipy 1.18, matplotlib 3.11); older versions are untested.

## Licence and citation

MIT licence (`LICENSE`). Until the article is published, cite this repository using
`CITATION.cff`.
