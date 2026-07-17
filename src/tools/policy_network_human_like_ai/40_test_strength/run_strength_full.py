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
        "--engine-threads",
        str(args.engine_threads),
        "--status-every-games",
        str(args.status_every_games),
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
    ]
    if args.time_limit_sec is not None:
        cmd.extend(["--time-limit-sec", str(args.time_limit_sec)])
    if args.max_games is not None:
        cmd.extend(["--max-games", str(args.max_games)])
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.no_blend_cache:
        cmd.append("--no-blend-cache")
    if args.close_processes_after_game:
        cmd.append("--close-processes-after-game")
    return cmd


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or resume the full human-like policy strength study.")
    parser.add_argument("--baseline-levels", default="1,5,10,15,21")
    parser.add_argument("--blend-params", "--alphas", dest="blend_params", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--games-per-pair", type=int, default=1000)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--parallel-matches", type=int, default=32)
    parser.add_argument("--processes-per-player", type=int, default=32)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--status-every-games", type=int, default=200)
    parser.add_argument("--task-retries", type=int, default=2)
    parser.add_argument("--time-limit-sec", type=float, default=None)
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--blend-egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=300.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--openings", type=Path, default=SCRIPT_DIR.parents[3] / "bin" / "problem" / "xot" / "openingslarge.txt")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output" / "strength_full")
    parser.add_argument("--no-blend-cache", action="store_true", help="Disable per-process Egaroucid hint caching in blended engines.")
    parser.add_argument("--close-processes-after-game", action="store_true", help="Close engines after each game instead of keeping them in per-player pools.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    cmd = make_command(args)
    print(" ".join(str(part) for part in display_command(cmd)))
    returncode = run_command(cmd, args.output_dir / "run_strength_full.log")
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
