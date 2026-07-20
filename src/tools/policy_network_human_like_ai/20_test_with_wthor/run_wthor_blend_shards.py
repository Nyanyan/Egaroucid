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
from typing import Iterable, List, Optional, Sequence, Tuple

from evaluate_wthor_blend_human_match import (
    ALL_LEGAL_HINT_COUNT,
    count_position_samples,
    default_board_data_dir,
    discover_dat_files,
    split_absolute_ranges,
)
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
        return (
            bool(result.get("topn"))
            and int(result.get("console_hint_count", -1))
            == ALL_LEGAL_HINT_COUNT
            and result.get("egaroucid_policy_support") == "all_legal_moves"
        )
    except Exception:
        return False


class ShardSchedule:
    def __init__(self, range_start: int, range_end: int, num_shards: int, positions_per_shard: Optional[int]) -> None:
        self.range_start = range_start
        self.range_end = range_end
        self.num_shards = num_shards
        self.positions_per_shard = positions_per_shard
        if positions_per_shard is not None:
            if positions_per_shard <= 0:
                raise ValueError("--positions-per-shard must be positive when set")
            total = max(0, range_end - range_start)
            self._ranges: Optional[List[Tuple[int, int]]] = None
            self.shard_count = (total + positions_per_shard - 1) // positions_per_shard
        else:
            self._ranges = split_absolute_ranges(range_start, range_end, num_shards)
            self.shard_count = len(self._ranges)

    @property
    def scheduled_position_samples(self) -> int:
        return max(0, self.range_end - self.range_start)

    def range_at(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= self.shard_count:
            raise IndexError(idx)
        if self.positions_per_shard is None:
            assert self._ranges is not None
            return self._ranges[idx]
        start = self.range_start + idx * self.positions_per_shard
        return start, min(self.range_end, start + self.positions_per_shard)

    def iter_ranges(self) -> Iterable[Tuple[int, int, int]]:
        for idx in range(self.shard_count):
            start, end = self.range_at(idx)
            yield idx, start, end

    def index_for_range(self, start: int, end: int) -> Optional[int]:
        if self.positions_per_shard is None:
            assert self._ranges is not None
            for idx, (range_start, range_end) in enumerate(self._ranges):
                if start == range_start and end == range_end:
                    return idx
            return None
        if start < self.range_start or end > self.range_end:
            return None
        offset = start - self.range_start
        if offset % self.positions_per_shard != 0:
            return None
        idx = offset // self.positions_per_shard
        if idx < 0 or idx >= self.shard_count:
            return None
        expected_start, expected_end = self.range_at(idx)
        if start == expected_start and end == expected_end:
            return idx
        return None


def shard_dir_for(output_dir: Path, idx: int, range_start: int, range_end: int) -> Path:
    return output_dir / f"shard_{idx:03d}_{range_start}_{range_end}"


def shard_manifest_entry(output_dir: Path, idx: int, range_start: int, range_end: int) -> dict:
    return {
        "index": idx,
        "range_start": range_start,
        "range_end": range_end,
        "output_dir": str(shard_dir_for(output_dir, idx, range_start, range_end)),
    }


def make_shard_preview(schedule: ShardSchedule, output_dir: Path, preview_limit: int) -> dict:
    if preview_limit < 0:
        raise ValueError("--manifest-shard-preview must be non-negative")
    shard_count = schedule.shard_count
    first_count = min(preview_limit, shard_count)
    first_shards = [
        shard_manifest_entry(output_dir, idx, *schedule.range_at(idx))
        for idx in range(first_count)
    ]
    last_shards = []
    if preview_limit > 0 and shard_count > first_count:
        last_start = max(first_count, shard_count - preview_limit)
        last_shards = [
            shard_manifest_entry(output_dir, idx, *schedule.range_at(idx))
            for idx in range(last_start, shard_count)
        ]
    omitted = shard_count - len(first_shards) - len(last_shards)
    return {
        "shard_count": shard_count,
        "manifest_shard_preview": preview_limit,
        "shards_truncated": omitted > 0,
        "shards_omitted": omitted,
        "shards": first_shards,
        "last_shards": last_shards,
    }


def parse_shard_range(path: Path) -> Optional[Tuple[int, int]]:
    parts = path.name.split("_")
    if len(parts) < 4 or parts[0] != "shard":
        return None
    try:
        return int(parts[-2]), int(parts[-1])
    except ValueError:
        return None


def discover_completed_shards(output_dir: Path) -> List[Tuple[int, int, Path]]:
    items = []
    for path in output_dir.glob("shard_*_*_*"):
        if not path.is_dir() or not shard_done(path):
            continue
        parsed = parse_shard_range(path)
        if parsed is None:
            continue
        start, end = parsed
        items.append((start, end, path))
    return sorted(items, key=lambda item: (item[0], item[1], item[2].name))


def completed_shard_dirs_for_merge(output_dir: Path) -> List[Path]:
    items = discover_completed_shards(output_dir)
    last_end: Optional[int] = None
    for start, end, path in items:
        if last_end is not None and start < last_end:
            raise ValueError(f"completed shard ranges overlap around {path}; cannot merge all completed shards")
        last_end = end
    return [path for _, _, path in items]


def completed_prefix_end(output_dir: Path, range_start: int) -> int:
    prefix_end = range_start
    for start, end, _ in discover_completed_shards(output_dir):
        if end <= prefix_end:
            continue
        if start > prefix_end:
            break
        prefix_end = end
    return prefix_end


def write_progress_summary(
    output_dir: Path,
    schedule: ShardSchedule,
    start_time: float,
    stop_reason: str,
) -> dict:
    all_done = discover_completed_shards(output_dir)
    scheduled_done = []
    for start, end, _ in all_done:
        idx = schedule.index_for_range(start, end)
        if idx is not None:
            scheduled_done.append((idx, start, end))
    scheduled_done.sort()
    completed_position_samples = sum(end - start for _, start, end in scheduled_done)
    all_completed_position_samples = sum(end - start for start, end, _ in all_done)
    contiguous_completed_prefix_end = completed_prefix_end(output_dir, 0)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": time.time() - start_time,
        "stop_reason": stop_reason,
        "completed_shards": len(scheduled_done),
        "total_shards": schedule.shard_count,
        "completed_position_samples": completed_position_samples,
        "scheduled_position_samples": schedule.scheduled_position_samples,
        "done_shards": [idx for idx, _, _ in scheduled_done],
        "all_completed_shards": len(all_done),
        "all_completed_position_samples": all_completed_position_samples,
        "contiguous_completed_prefix_end": contiguous_completed_prefix_end,
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


def merge_shards(output_dir: Path, shard_dirs: Sequence[Path], merge_dir_name: str = "merged", log_name: str = "merge.log") -> None:
    input_list_path = output_dir / f"{merge_dir_name}_inputs.txt"
    with input_list_path.open("w", encoding="utf-8") as f:
        for path in shard_dirs:
            f.write(str(path) + "\n")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "merge_wthor_blend_results.py"),
        "--input-list",
        str(input_list_path),
        "--output-dir",
        str(output_dir / merge_dir_name),
    ]
    returncode = run_command(cmd, output_dir / log_name)
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
    parser.add_argument("--blend-params", "--alphas", dest="blend_params", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--top-n", default="1,2,3,4,5,8,10,16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=None, help="Optional cap for smoke/benchmark sharded runs.")
    parser.add_argument("--range-start", type=int, default=0, help="First global WTHOR position_sample index to schedule.")
    parser.add_argument("--range-end", type=int, default=None, help="Exclusive global WTHOR position_sample index to schedule.")
    parser.add_argument("--resume-from-completed-prefix", action="store_true", help="Start at the end of the completed contiguous shard prefix.")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--positions-per-shard", type=int, default=None, help="Split by fixed position_samples per shard instead of --num-shards.")
    parser.add_argument("--jobs-per-shard", type=int, default=4)
    parser.add_argument("--raw-hint-samples", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            SCRIPT_DIR
            / "output"
            / f"blend_wthor_full_sharded_hint{ALL_LEGAL_HINT_COUNT}_all_legal"
        ),
    )
    parser.add_argument("--hint-cache-db", type=Path, default=None, help="Shared SQLite cache for Egaroucid hint scores.")
    parser.add_argument("--no-hint-cache", action="store_true", help="Disable the default shared hint cache.")
    parser.add_argument("--max-shards-to-run", type=int, default=None, help="Optional smoke/benchmark cap.")
    parser.add_argument("--time-limit-sec", type=float, default=None, help="Stop launching new shards after this many seconds.")
    parser.add_argument("--merge-completed", action="store_true", help="Merge all completed shards even if the full run is incomplete.")
    parser.add_argument("--manifest-shard-preview", type=int, default=32, help="Number of first/last shard ranges to show in manifest.json.")
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
    if args.manifest_shard_preview < 0:
        raise ValueError("--manifest-shard-preview must be non-negative")
    if args.hint_cache_db is None and not args.no_hint_cache:
        args.hint_cache_db = args.output_dir / "hint_score_cache.sqlite3"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dat_files = discover_dat_files(args.board_data_dir)
    available_position_samples = count_position_samples(dat_files, args.max_positions)
    range_start = int(args.range_start)
    range_end = available_position_samples if args.range_end is None else int(args.range_end)
    if range_start < 0 or range_start > available_position_samples:
        raise ValueError(f"--range-start must be within 0..{available_position_samples}")
    if range_end < range_start or range_end > available_position_samples:
        raise ValueError(f"--range-end must be within {range_start}..{available_position_samples}")
    requested_range_start = range_start
    if args.resume_from_completed_prefix:
        range_start = completed_prefix_end(args.output_dir, range_start)
        if range_start > range_end:
            range_start = range_end
    schedule = ShardSchedule(range_start, range_end, args.num_shards, args.positions_per_shard)

    manifest = {
        "board_data_dir": str(args.board_data_dir),
        "weights": str(args.weights),
        "egaroucid_exe": str(args.egaroucid_exe),
        "available_position_samples": available_position_samples,
        "range_start": range_start,
        "range_end": range_end,
        "requested_range_start": requested_range_start,
        "resume_from_completed_prefix": args.resume_from_completed_prefix,
        "scheduled_position_samples": schedule.scheduled_position_samples,
        "max_positions": args.max_positions,
        "num_shards": args.num_shards,
        "positions_per_shard": args.positions_per_shard,
        "jobs_per_shard": args.jobs_per_shard,
        "time_limit_sec": args.time_limit_sec,
        "merge_completed": args.merge_completed,
        "console_hint_count": ALL_LEGAL_HINT_COUNT,
        "egaroucid_policy_support": "all_legal_moves",
        "hint_cache_db": str(args.hint_cache_db) if args.hint_cache_db is not None else None,
    }
    manifest.update(make_shard_preview(schedule, args.output_dir, args.manifest_shard_preview))
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        write_progress_summary(args.output_dir, schedule, start_time, "dry_run")
        return

    stop_reason = "finished"
    if not args.merge_only:
        ran = 0
        skipped = 0
        completed_ranges = {(start, end) for start, end, _ in discover_completed_shards(args.output_dir)}
        for idx, shard_range_start, shard_range_end in schedule.iter_ranges():
            if args.time_limit_sec is not None and ran > 0 and time.time() - start_time >= args.time_limit_sec:
                stop_reason = "time_limit"
                break
            shard_dir = shard_dir_for(args.output_dir, idx, shard_range_start, shard_range_end)
            if (shard_range_start, shard_range_end) in completed_ranges or shard_done(shard_dir):
                skipped += 1
                continue
            if args.max_shards_to_run is not None and ran >= args.max_shards_to_run:
                stop_reason = "max_shards_to_run"
                break
            if skipped:
                print(f"skip {skipped} already-done shards before shard {idx}")
                skipped = 0
            cmd = make_shard_command(args, shard_dir, shard_range_start, shard_range_end)
            print(f"run shard {idx}: {shard_range_start}..{shard_range_end}")
            shard_log = args.output_dir / f"shard_{idx:03d}_{shard_range_start}_{shard_range_end}.log"
            returncode = run_command(cmd, shard_log)
            if returncode != 0:
                raise SystemExit(returncode)
            if shard_done(shard_dir):
                completed_ranges.add((shard_range_start, shard_range_end))
            ran += 1

    scheduled_done = []
    for shard_range_start, shard_range_end, path in discover_completed_shards(args.output_dir):
        idx = schedule.index_for_range(shard_range_start, shard_range_end)
        if idx is not None:
            scheduled_done.append((idx, path))
    scheduled_done.sort(key=lambda item: item[0])
    done_dirs = [path for _, path in scheduled_done]
    print(f"completed_shards {len(done_dirs)}/{schedule.shard_count}")
    if len(done_dirs) == schedule.shard_count:
        merge_shards(args.output_dir, done_dirs)
        print("merged", args.output_dir / "merged")
    else:
        if args.merge_only:
            stop_reason = "merge_only_incomplete"
        elif stop_reason == "finished":
            stop_reason = "incomplete"
        if args.merge_completed:
            completed_for_merge = completed_shard_dirs_for_merge(args.output_dir)
            if completed_for_merge:
                merge_shards(args.output_dir, completed_for_merge, "partial_merged", "partial_merge.log")
                print("partial_merged", args.output_dir / "partial_merged")
    progress = write_progress_summary(args.output_dir, schedule, start_time, stop_reason)
    print("completed_position_samples", progress["completed_position_samples"], "/", progress["scheduled_position_samples"])
    print("all_completed_position_samples", progress["all_completed_position_samples"])
    print("stop_reason", progress["stop_reason"])


if __name__ == "__main__":
    main()
