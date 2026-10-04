#!/usr/bin/env python3
"""Run the asymmetric AEPD--PATP S=2 publication study."""

from __future__ import annotations

import argparse
from pathlib import Path

from aepd_patp_core import (
    load_manifest,
    run_aepd_patp_study,
    summarise_rows,
    theory_sweep,
    write_artifacts,
)


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size < 20 for size in sizes):
        raise argparse.ArgumentTypeError("sample sizes must be integers >=20")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1_000)
    parser.add_argument("--sample-sizes", type=parse_sizes, default=(200, 500, 1_000, 5_000))
    parser.add_argument("--seed", type=int, default=20260841)
    parser.add_argument("--bootstrap-reps", type=int, default=1_000)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--skip-mle", action="store_true")
    parser.add_argument("--mle-starts", type=int, default=3)
    parser.add_argument("--scenario", action="append", default=[])
    args = parser.parse_args()
    reps = 100 if args.pilot and args.reps == 1_000 else args.reps
    bootstrap_reps = 200 if args.pilot and args.bootstrap_reps == 1_000 else args.bootstrap_reps
    if reps < 4:
        parser.error("--reps must be at least 4")
    manifest = load_manifest(Path(__file__).resolve().parents[1] / "configs/aepd_patp_s2.json")
    rows = run_aepd_patp_study(
        manifest,
        sample_sizes=args.sample_sizes,
        reps=reps,
        seed=args.seed,
        include_mle=not args.skip_mle,
        mle_starts=args.mle_starts,
        scenario_ids=args.scenario,
    )
    summary = summarise_rows(rows, bootstrap_reps=bootstrap_reps)
    sweep = theory_sweep(manifest)
    prefix = "aepd_patp_s2_pilot" if args.pilot else "aepd_patp_s2"
    write_artifacts(
        prefix, manifest, rows, summary, sweep,
        sample_sizes=args.sample_sizes, reps=reps, seed=args.seed,
        bootstrap_reps=bootstrap_reps, include_mle=not args.skip_mle,
        mle_starts=args.mle_starts, pilot=args.pilot,
    )
    print(f"Wrote {prefix} replication, summary, sweep and metadata artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
