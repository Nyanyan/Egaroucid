#!/usr/bin/env python3
"""
Run policy-network training configs one by one and save raw logs.

Raw logs are written to train_log/, which is intentionally git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from resource_monitor import run_monitored_command


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_configs(text: str) -> list[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run policy network training experiments with raw log capture.")
    parser.add_argument("--configs", default="64x3,96x3,128x3,96x4")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-train-samples", type=int, default=2_000_000)
    parser.add_argument("--max-val-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--record-end", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=script_dir() / "trained" / "selected_v2")
    parser.add_argument("--log-dir", type=Path, default=script_dir() / "train_log")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in parse_configs(args.configs):
        log_path = args.log_dir / f"train_{config.replace(':', '_')}.log"
        run_output_dir = args.output_dir / config.replace(":", "_")
        cmd = [
            sys.executable,
            str(script_dir() / "train_policy_network.py"),
            "--configs",
            config,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-val-samples",
            str(args.max_val_samples),
            "--seed",
            str(args.seed),
            "--record-start",
            str(args.record_start),
            "--record-end",
            str(args.record_end),
            "--output-dir",
            str(run_output_dir),
        ]
        if args.data_root is not None:
            cmd.extend(["--data-root", str(args.data_root)])

        resource = run_monitored_command(cmd, log_path)
        row = {
            "config": config,
            "returncode": resource["returncode"],
            "elapsed_sec": resource["elapsed_sec"],
            "peak_rss_mib": resource["peak_rss_mib"],
            "resource_samples": resource["resource_samples"],
            "log": str(log_path),
            "output_dir": str(run_output_dir),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if resource["returncode"] != 0:
            break

    with (args.log_dir / "training_runs.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (args.log_dir / "training_runs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "returncode", "elapsed_sec", "peak_rss_mib", "resource_samples", "log", "output_dir"])
        writer.writeheader()
        writer.writerows(rows)
    if rows and rows[-1]["returncode"] != 0:
        raise SystemExit(rows[-1]["returncode"])


if __name__ == "__main__":
    main()
