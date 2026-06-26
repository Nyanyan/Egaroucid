#!/usr/bin/env python3
"""
Build and run the bounded real-book Umigame regression verifier.

Defaults are intentionally finite:
- check at most 100000 reachable book nodes;
- stop after the first 10 mismatches.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_DIR = Path(__file__).resolve().parent
PROJECT = TOOL_DIR / "UmigameRegression.vcxproj"
RUN_DIR = REPO_ROOT / "bin" / "build_verify" / "umigame_regression"
EXE = RUN_DIR / "umigame_regression.exe"
DEFAULT_BOOK = REPO_ROOT / "bin" / "document" / "book.egbk3"
POLL_INTERVAL_SECONDS = 2
PROGRESS_INTERVAL_SECONDS = 30


@dataclass
class ShardProcess:
    shard: int
    process: subprocess.Popen[str]
    log_path: Path
    log_file: object

    def close_log(self) -> None:
        self.log_file.close()


def find_msbuild() -> str:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "MSBuild"


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def terminate_running(processes: list[ShardProcess]) -> None:
    for shard_proc in processes:
        if shard_proc.process.poll() is None:
            shard_proc.process.terminate()
    for shard_proc in processes:
        if shard_proc.process.poll() is None:
            try:
                shard_proc.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                shard_proc.process.kill()
                shard_proc.process.wait()


def print_shard_log(log_path: Path) -> None:
    print(log_path.read_text(encoding="utf-8", errors="replace"), flush=True)


def run_parallel_shards(base_cmd: list[str], shard_count: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    processes: list[ShardProcess] = []
    for shard in range(shard_count):
        cmd = base_cmd + [
            "--shard-index",
            str(shard),
            "--shard-count",
            str(shard_count),
        ]
        print("+ " + " ".join(cmd), flush=True)
        log_path = RUN_DIR / f"shard_{shard:02d}.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT / "bin",
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(ShardProcess(shard, process, log_path, log_file))

    failed = False
    next_progress = time.monotonic() + PROGRESS_INTERVAL_SECONDS
    try:
        while processes:
            time.sleep(POLL_INTERVAL_SECONDS)
            now = time.monotonic()
            if now >= next_progress:
                running = ", ".join(f"{shard_proc.shard:02d}" for shard_proc in processes)
                print(f"still running shards: {running}", flush=True)
                next_progress = now + PROGRESS_INTERVAL_SECONDS

            still_running: list[ShardProcess] = []
            for shard_proc in processes:
                code = shard_proc.process.poll()
                if code is None:
                    still_running.append(shard_proc)
                    continue

                shard_proc.close_log()
                print(f"shard {shard_proc.shard} finished with code {code}; log={shard_proc.log_path}", flush=True)
                if code != 0:
                    failed = True
                    print_shard_log(shard_proc.log_path)
                    terminate_running(still_running)
                    break

            if failed:
                break
            processes = still_running
    finally:
        terminate_running(processes)
        for shard_proc in processes:
            if not shard_proc.log_file.closed:
                shard_proc.close_log()

    if failed:
        raise SystemExit(1)

    for shard in range(shard_count):
        print_shard_log(RUN_DIR / f"shard_{shard:02d}.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default=str(DEFAULT_BOOK), help="book file to test")
    parser.add_argument("--depths", default="8,12,20,60", help="comma-separated Umigame depths")
    parser.add_argument("--ranges", default="-64:64,0:6,3:5,-6:0", help="comma-separated black-score ranges min:max")
    parser.add_argument("--integration-errors", default="0,2,6", help="comma-separated cumulative local-loss budgets")
    parser.add_argument("--node-limit", type=int, default=100000)
    parser.add_argument("--fail-limit", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--parallel-shards", type=int, default=1)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    if not args.no_build:
        run(
            [
                find_msbuild(),
                str(PROJECT),
                "/m",
                "/p:Configuration=Generic",
                "/p:Platform=x64",
                "/v:minimal",
            ],
            REPO_ROOT,
        )

    if not EXE.exists():
        raise SystemExit(f"verifier executable not found: {EXE}")

    base_cmd = [
        str(EXE),
        "--book",
        str(Path(args.book)),
        "--depths",
        args.depths,
        "--ranges",
        args.ranges,
        "--integration-errors",
        args.integration_errors,
        "--node-limit",
        str(args.node_limit),
        "--fail-limit",
        str(args.fail_limit),
        "--threads",
        str(args.threads),
    ]

    if args.parallel_shards <= 1:
        run(base_cmd, REPO_ROOT / "bin")
        return

    run_parallel_shards(base_cmd, args.parallel_shards)


if __name__ == "__main__":
    main()
