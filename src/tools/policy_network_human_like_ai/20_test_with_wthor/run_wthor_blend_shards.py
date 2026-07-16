#!/usr/bin/env python3
"""
Run WTHOR blended-policy agreement evaluation in resumable shards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional, Sequence, Tuple

from evaluate_wthor_blend_human_match import count_position_samples, default_board_data_dir, discover_dat_files, split_absolute_ranges
from blend_policy_with_egaroucid import default_egaroucid_exe, default_weights_file


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]


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


def shard_done(output_dir: Path) -> bool:
    result_path = output_dir / "wthor_blend_human_match.json"
    if not result_path.exists():
        return False
    try:
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        return bool(result.get("topn"))
    except Exception:
        return False


def make_ranges(total_positions: int, num_shards: int, positions_per_shard: Optional[int]) -> List[Tuple[int, int]]:
    if positions_per_shard is not None:
        if positions_per_shard <= 0:
            raise ValueError("--positions-per-shard must be positive when set")
        return [(start, min(total_positions, start + positions_per_shard)) for start in range(0, total_positions, positions_per_shard)]
    return split_absolute_ranges(0, total_positions, num_shards)


def write_progress_summary(
    output_dir: Path,
    shard_dirs: Sequence[Path],
    ranges: Sequence[Tuple[int, int]],
    start_time: float,
    stop_reason: str,
) -> dict:
    done = [idx for idx, path in enumerate(shard_dirs) if shard_done(path)]
    completed_positions = sum(ranges[idx][1] - ranges[idx][0] for idx in done)
    total_positions = sum(end - start for start, end in ranges)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": time.time() - start_time,
        "stop_reason": stop_reason,
        "completed_shards": len(done),
        "total_shards": len(shard_dirs),
        "completed_positions": completed_positions,
        "total_positions": total_positions,
        "done_shards": done,
    }
    with (output_dir / "progress_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_command(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(part) for part in display_command(cmd)) + "\n")
        log.write("start_time " + time.strftime("%Y-%m-%dT%H:%M:%S") + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write("\nend_time " + time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        log.write(f"returncode {proc.returncode}\n")
        return proc.returncode


def make_shard_command(args: argparse.Namespace, shard_dir: Path, range_start: int, range_end: int) -> List[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_wthor_blend_human_match.py"),
        "--board-data-dir",
        str(args.board_data_dir),
        "--weights",
        str(args.weights),
        "--egaroucid-exe",
        str(args.egaroucid_exe),
        "--egaroucid-level",
        str(args.egaroucid_level),
        "--egaroucid-threads",
        str(args.egaroucid_threads),
        "--egaroucid-timeout-sec",
        str(args.egaroucid_timeout_sec),
        "--score-temperature",
        str(args.score_temperature),
        "--blend-params",
        args.blend_params,
        "--top-n",
        args.top_n,
        "--batch-size",
        str(args.batch_size),
        "--range-start",
        str(range_start),
        "--range-end",
        str(range_end),
        "--jobs",
        str(args.jobs_per_shard),
        "--raw-hint-samples",
        str(args.raw_hint_samples),
        "--output-dir",
        str(shard_dir),
    ]
    if args.max_positions is not None:
        cmd.extend(["--max-positions", str(args.max_positions)])
    if args.hint_cache_db is not None:
        cmd.extend(["--hint-cache-db", str(args.hint_cache_db)])
    return cmd


def merge_shards(output_dir: Path, shard_dirs: Sequence[Path]) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "merge_wthor_blend_results.py"),
        *[str(path) for path in shard_dirs],
        "--output-dir",
        str(output_dir / "merged"),
    ]
    returncode = run_command(cmd, output_dir / "merge.log")
    if returncode != 0:
        raise SystemExit(returncode)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WTHOR blend agreement in resumable shards.")
    parser.add_argument("--board-data-dir", type=Path, default=default_board_data_dir())
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-threads", type=int, default=1)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=300.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--blend-params", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--top-n", default="1,2,3,4,5,8,10,16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=None, help="Optional cap for smoke/benchmark sharded runs.")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--positions-per-shard", type=int, default=None, help="Split by fixed position_samples per shard instead of --num-shards.")
    parser.add_argument("--jobs-per-shard", type=int, default=4)
    parser.add_argument("--raw-hint-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output" / "blend_wthor_full_sharded")
    parser.add_argument("--hint-cache-db", type=Path, default=None, help="Shared SQLite cache for Egaroucid hint scores.")
    parser.add_argument("--no-hint-cache", action="store_true", help="Disable the default shared hint cache.")
    parser.add_argument("--max-shards-to-run", type=int, default=None, help="Optional smoke/benchmark cap.")
    parser.add_argument("--time-limit-sec", type=float, default=None, help="Stop launching new shards after this many seconds.")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    start_time = time.time()
    args = make_argparser().parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.jobs_per_shard <= 0:
        raise ValueError("--jobs-per-shard must be positive")
    if args.time_limit_sec is not None and args.time_limit_sec <= 0.0:
        raise ValueError("--time-limit-sec must be positive when set")
    if args.hint_cache_db is None and not args.no_hint_cache:
        args.hint_cache_db = args.output_dir / "hint_score_cache.sqlite3"

    dat_files = discover_dat_files(args.board_data_dir)
    total_positions = count_position_samples(dat_files, args.max_positions)
    ranges = make_ranges(total_positions, args.num_shards, args.positions_per_shard)
    shard_dirs = [args.output_dir / f"shard_{idx:03d}_{start}_{end}" for idx, (start, end) in enumerate(ranges)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "board_data_dir": str(args.board_data_dir),
        "weights": str(args.weights),
        "egaroucid_exe": str(args.egaroucid_exe),
        "total_positions": total_positions,
        "max_positions": args.max_positions,
        "num_shards": args.num_shards,
        "positions_per_shard": args.positions_per_shard,
        "jobs_per_shard": args.jobs_per_shard,
        "time_limit_sec": args.time_limit_sec,
        "hint_cache_db": str(args.hint_cache_db) if args.hint_cache_db is not None else None,
        "shards": [
            {"index": idx, "range_start": start, "range_end": end, "output_dir": str(shard_dirs[idx])}
            for idx, (start, end) in enumerate(ranges)
        ],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        write_progress_summary(args.output_dir, shard_dirs, ranges, start_time, "dry_run")
        return

    stop_reason = "finished"
    if not args.merge_only:
        ran = 0
        for idx, (range_start, range_end) in enumerate(ranges):
            if args.time_limit_sec is not None and ran > 0 and time.time() - start_time >= args.time_limit_sec:
                stop_reason = "time_limit"
                break
            shard_dir = shard_dirs[idx]
            if shard_done(shard_dir):
                print(f"skip shard {idx}: already done")
                continue
            if args.max_shards_to_run is not None and ran >= args.max_shards_to_run:
                stop_reason = "max_shards_to_run"
                break
            cmd = make_shard_command(args, shard_dir, range_start, range_end)
            print(f"run shard {idx}: {range_start}..{range_end}")
            returncode = run_command(cmd, args.output_dir / f"shard_{idx:03d}.log")
            if returncode != 0:
                raise SystemExit(returncode)
            ran += 1

    done_dirs = [path for path in shard_dirs if shard_done(path)]
    print(f"completed_shards {len(done_dirs)}/{len(shard_dirs)}")
    if len(done_dirs) == len(shard_dirs):
        merge_shards(args.output_dir, shard_dirs)
        print("merged", args.output_dir / "merged")
    else:
        if args.merge_only:
            stop_reason = "merge_only_incomplete"
        elif stop_reason == "finished":
            stop_reason = "incomplete"
    progress = write_progress_summary(args.output_dir, shard_dirs, ranges, start_time, stop_reason)
    print("completed_positions", progress["completed_positions"], "/", progress["total_positions"])
    print("stop_reason", progress["stop_reason"])


if __name__ == "__main__":
    main()
