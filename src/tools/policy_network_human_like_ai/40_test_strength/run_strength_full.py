#!/usr/bin/env python3
"""
Wrapper for the full resumable strength test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BLEND_DIR = SCRIPT_DIR.parents[0] / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import default_egaroucid_exe, default_weights_file  # noqa: E402


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def display_command(command: Sequence[str]) -> List[str]:
    result = []
    for part in command:
        try:
            if Path(part).resolve() == Path(sys.executable).resolve():
                result.append("python")
                continue
        except (OSError, RuntimeError):
            pass
        path = Path(part)
        if path.is_absolute() or path.exists():
            result.append(repo_relative(path))
        else:
            result.append(part)
    return result


def run_command(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(str(part) for part in display_command(cmd)) + "\n")
        log.write("start_time " + time.strftime("%Y-%m-%dT%H:%M:%S") + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write("\nend_time " + time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        log.write(f"returncode {proc.returncode}\n")
        return proc.returncode


def make_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "battle_blend_strength.py"),
        "--random-seed",
        str(args.random_seed),
        "--baseline-levels",
        args.baseline_levels,
        "--blend-params",
        args.blend_params,
        "--games-per-pair",
        str(args.games_per_pair),
        "--parallel-matches",
        str(args.parallel_matches),
        "--processes-per-player",
        str(args.processes_per_player),
        "--baseline-processes-per-player",
        str(args.baseline_processes_per_player),
        "--blend-processes-per-player",
        str(args.blend_processes_per_player),
        "--engine-threads",
        str(args.engine_threads),
        "--status-every-match-sets",
        str(args.status_every_match_sets),
        "--task-retries",
        str(args.task_retries),
        "--weights",
        str(args.weights),
        "--egaroucid-exe",
        str(args.egaroucid_exe),
        "--blend-egaroucid-level",
        str(args.blend_egaroucid_level),
        "--egaroucid-timeout-sec",
        str(args.egaroucid_timeout_sec),
        "--score-temperature",
        str(args.score_temperature),
        "--openings",
        str(args.openings),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
        "--policy-backend",
        args.policy_backend,
        "--policy-server-host",
        args.policy_server_host,
        "--policy-server-timeout-sec",
        str(args.policy_server_timeout_sec),
        "--policy-server-startup-timeout-sec",
        str(args.policy_server_startup_timeout_sec),
        "--policy-max-batch-size",
        str(args.policy_max_batch_size),
        "--policy-batch-wait-ms",
        str(args.policy_batch_wait_ms),
        "--policy-inference-threads",
        str(args.policy_inference_threads),
        "--performance-sample-interval-sec",
        str(args.performance_sample_interval_sec),
        "--minimum-available-memory-mib",
        str(args.minimum_available_memory_mib),
        "--estimated-engine-memory-mib",
        str(args.estimated_engine_memory_mib),
    ]
    if args.policy_model is not None:
        cmd.extend(["--policy-model", str(args.policy_model)])
    if args.policy_server_port is not None:
        cmd.extend(["--policy-server-port", str(args.policy_server_port)])
    if args.hint_cache_db is not None:
        cmd.extend(["--hint-cache-db", str(args.hint_cache_db)])
    if args.time_limit_sec is not None:
        cmd.extend(["--time-limit-sec", str(args.time_limit_sec)])
    if args.max_match_sets is not None:
        cmd.extend(["--max-match-sets", str(args.max_match_sets)])
    if args.no_shuffle_openings:
        cmd.append("--no-shuffle-openings")
    if args.no_random_player:
        cmd.append("--no-random-player")
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.no_blend_cache:
        cmd.append("--no-blend-cache")
    if args.no_native_alpha_zero:
        cmd.append("--no-native-alpha-zero")
    if args.no_policy_batch_server:
        cmd.append("--no-policy-batch-server")
    if args.no_performance_monitor:
        cmd.append("--no-performance-monitor")
    if args.same_openings_for_all_pairs:
        cmd.append("--same-openings-for-all-pairs")
    if args.close_processes_after_game:
        cmd.append("--close-processes-after-game")
    return cmd


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or resume the full human-like policy strength study.")
    parser.add_argument("--no-random-player", action="store_true")
    parser.add_argument("--random-seed", type=int, default=613)
    parser.add_argument("--baseline-levels", default="1,3,5,7,9,11,13,15,17,19")
    parser.add_argument("--blend-params", "--alphas", dest="blend_params", default="0.0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--games-per-pair", type=int, default=50)
    parser.add_argument("--max-match-sets", "--max-games", dest="max_match_sets", type=int, default=None)
    parser.add_argument("--parallel-matches", type=int, default=16)
    parser.add_argument("--processes-per-player", type=int, default=2)
    parser.add_argument("--baseline-processes-per-player", type=int, default=None)
    parser.add_argument("--blend-processes-per-player", type=int, default=None)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--status-every-match-sets", "--status-every-games", dest="status_every_match_sets", type=int, default=200)
    parser.add_argument("--task-retries", type=int, default=2)
    parser.add_argument("--time-limit-sec", type=float, default=None)
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--blend-egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=300.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--openings", type=Path, default=SCRIPT_DIR.parents[3] / "bin" / "problem" / "xot" / "openingslarge.txt")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output" / "strength_full")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--no-shuffle-openings", action="store_true")
    parser.add_argument("--no-blend-cache", action="store_true", help="Disable per-process Egaroucid hint caching in blended engines.")
    parser.add_argument("--hint-cache-db", type=Path, default=None, help="Shared SQLite cache for Egaroucid hint scores.")
    parser.add_argument("--no-native-alpha-zero", action="store_true", help="Use the Python blend engine even for alpha=0.0.")
    parser.add_argument("--no-policy-batch-server", action="store_true")
    parser.add_argument("--policy-model", type=Path, default=None)
    parser.add_argument("--policy-backend", choices=("auto", "tensorflow", "numpy"), default="auto")
    parser.add_argument("--policy-server-host", default="127.0.0.1")
    parser.add_argument("--policy-server-port", type=int, default=None)
    parser.add_argument("--policy-server-timeout-sec", type=float, default=30.0)
    parser.add_argument("--policy-server-startup-timeout-sec", type=float, default=120.0)
    parser.add_argument("--policy-max-batch-size", type=int, default=128)
    parser.add_argument("--policy-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--policy-inference-threads", type=int, default=4)
    parser.add_argument("--no-performance-monitor", action="store_true")
    parser.add_argument("--performance-sample-interval-sec", type=float, default=2.0)
    parser.add_argument("--minimum-available-memory-mib", type=float, default=24576.0)
    parser.add_argument("--estimated-engine-memory-mib", type=float, default=1400.0)
    parser.add_argument("--same-openings-for-all-pairs", action="store_true", help="Use the same opening sequence for every pair.")
    parser.add_argument("--close-processes-after-game", action="store_true", help="Close engines after each game instead of keeping them in per-player pools.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    if args.baseline_processes_per_player is None:
        args.baseline_processes_per_player = args.processes_per_player
    if args.blend_processes_per_player is None:
        args.blend_processes_per_player = args.processes_per_player
    cmd = make_command(args)
    print(" ".join(str(part) for part in display_command(cmd)))
    returncode = run_command(cmd, args.output_dir / "run_strength_full.log")
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
