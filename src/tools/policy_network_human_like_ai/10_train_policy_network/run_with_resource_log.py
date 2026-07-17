#!/usr/bin/env python3
"""Run one command and save stdout/stderr plus elapsed time and maximum resident memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from resource_monitor import run_monitored_command


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a command with raw log and resource summary.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command")
    summary = run_monitored_command(command, args.log, args.poll_interval_sec)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["returncode"] != 0:
        raise SystemExit(summary["returncode"])


if __name__ == "__main__":
    main()
