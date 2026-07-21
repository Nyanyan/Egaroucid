#!/usr/bin/env python3
"""Generate clustered 95% confidence intervals from a completed raw log."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

from strength_elo_confidence import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    analyze_clustered_elos,
    default_worker_count,
    load_clustered_elo_data,
    write_clustered_elo_outputs,
)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reanalyze a completed strength_games.jsonl with an XOT/set-index "
            "cluster BCa bootstrap. Existing tournament logs are not changed."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
    )
    parser.add_argument(
        "--expected-raw-sha256",
        help="Refuse analysis unless strength_games.jsonl has this SHA-256.",
    )
    parser.add_argument(
        "--expected-results-sha256",
        help="Refuse analysis unless strength_results.json has this SHA-256.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = make_argparser().parse_args(argv)
    if args.bootstrap_replicates < 1_000:
        raise ValueError("at least 1,000 bootstrap replicates are required")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not math.isclose(
        args.confidence,
        DEFAULT_CONFIDENCE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("this report supports only a 95% confidence level")
    data = load_clustered_elo_data(args.output_dir)
    if (
        args.expected_raw_sha256
        and data.raw_results_sha256.lower()
        != args.expected_raw_sha256.lower()
    ):
        raise ValueError(
            "strength_games.jsonl SHA-256 does not match "
            "--expected-raw-sha256"
        )
    if (
        args.expected_results_sha256
        and data.aggregate_sha256.lower()
        != args.expected_results_sha256.lower()
    ):
        raise ValueError(
            "strength_results.json SHA-256 does not match "
            "--expected-results-sha256"
        )
    print("experiment_id", data.experiment_id, flush=True)
    print("raw_results_sha256", data.raw_results_sha256, flush=True)
    print("aggregate_results_sha256", data.aggregate_sha256, flush=True)
    print("participants", len(data.names), flush=True)
    print("player_pairs", len(data.pairs), flush=True)
    print("XOT_clusters", len(data.set_indices), flush=True)
    print("bootstrap_replicates", args.bootstrap_replicates, flush=True)
    print("bootstrap_seed", args.bootstrap_seed, flush=True)
    print("workers", args.workers, flush=True)
    analysis = analyze_clustered_elos(
        data,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
        workers=args.workers,
        progress=True,
    )
    write_clustered_elo_outputs(
        args.output_dir,
        data,
        analysis,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
    print("name\tElo\tCI low\tCI high\tmethod", flush=True)
    for name, point, interval in zip(
        data.names,
        analysis.point_elos,
        analysis.intervals,
    ):
        print(
            f"{name}\t{point:.3f}\t{interval.low:.3f}\t"
            f"{interval.high:.3f}\t{interval.method}",
            flush=True,
        )


if __name__ == "__main__":
    main()
