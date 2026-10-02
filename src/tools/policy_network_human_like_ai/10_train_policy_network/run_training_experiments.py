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
from prepared_policy_data import add_data_arguments


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_configs(text: str) -> list[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run policy network training experiments with raw log capture.")
    parser.add_argument("--configs", default="64x3,96x3,128x3,96x4")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=None, help="Default: use the trainer's automatic batch size for each architecture.")
    parser.add_argument("--seed", type=int, default=613)
    add_data_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=script_dir() / "trained" / "training_runs")
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
            "--seed",
            str(args.seed),
            "--dataset",
            args.dataset,
            "--output-dir",
            str(run_output_dir),
        ]
        if args.data_root is not None:
            cmd.extend(["--data-root", str(args.data_root)])
        if args.board_split_dir is not None:
            cmd.extend(["--board-split-dir", str(args.board_split_dir)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        for name in ("train", "val", "test"):
            cap = getattr(args, f"max_{name}_samples")
            if cap is not None:
                cmd.extend([f"--max-{name}-samples", str(cap)])

        resource = run_monitored_command(cmd, log_path)
        row = {
            "config": config,
            "returncode": resource["returncode"],
            "elapsed_sec": resource["elapsed_sec"],
            "max_main_memory_mib": resource["max_main_memory_mib"],
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
        writer = csv.DictWriter(f, fieldnames=["config", "returncode", "elapsed_sec", "max_main_memory_mib", "resource_samples", "log", "output_dir"])
        writer.writeheader()
        writer.writerows(rows)
    if rows and rows[-1]["returncode"] != 0:
        raise SystemExit(rows[-1]["returncode"])


if __name__ == "__main__":
    main()
