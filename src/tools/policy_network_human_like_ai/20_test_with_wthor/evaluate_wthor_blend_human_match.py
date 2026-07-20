#!/usr/bin/env python3
"""
Evaluate WTHOR human-move agreement for blended policy/Egaroucid output.

This script is intentionally slower than pure policy evaluation because it asks
Egaroucid for Console for hint scores on each position.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import sqlite3
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BLEND_DIR = SCRIPT_DIR.parents[0] / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import (  # noqa: E402
    BLACK,
    POLICY_SIZE,
    WHITE,
    BlendedPolicy,
    BoardState,
    default_egaroucid_exe,
    default_weights_file,
    geometric_blend_distribution,
    parse_hint_move_order,
)
from evaluate_wthor_human_match import (  # noqa: E402
    BOARD_DTYPE,
    BOARD_SAMPLE_SIZE,
    equivalent_policy_mask,
    move_bucket,
)

ALL_LEGAL_HINT_COUNT = POLICY_SIZE


def default_board_data_dir() -> Path:
    return Path(os.environ["EGAROUCID_DATA"]) / "train_data" / "board_data" / "records1"


def discover_dat_files(board_data_dir: Path) -> List[Path]:
    files = sorted(board_data_dir.glob("*.dat"), key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
    if not files:
        raise FileNotFoundError(f"no .dat files found in {board_data_dir}")
    return files


def count_position_samples(dat_files: Sequence[Path], max_positions: Optional[int]) -> int:
    total = 0
    for path in dat_files:
        n = path.stat().st_size // BOARD_SAMPLE_SIZE
        if max_positions is not None:
            n = min(n, max_positions - total)
        if n <= 0:
            break
        total += n
    return total


def choose_global_positions(total_positions: int, sample_positions: Optional[int], seed: int) -> Optional[np.ndarray]:
    if sample_positions is None:
        return None
    if total_positions <= 0:
        return np.empty(0, dtype=np.int64)
    n = min(sample_positions, total_positions)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_positions, size=n, replace=False).astype(np.int64, copy=False))


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("total split size must be positive")
    if ratio_sum <= 0.0:
        raise ValueError("train/validation/test ratios must have a positive sum")
    n_train = int(np.floor(total * (train_ratio / ratio_sum)))
    n_val = int(np.floor(total * (val_ratio / ratio_sum)))
    n_test = total - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"split produced an empty part: train={n_train} val={n_val} test={n_test}")
    return n_train, n_val, n_test


def choose_data_split_positions(
    total_positions: int,
    data_split: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    sample_positions: Optional[int],
    sample_seed: int,
) -> Tuple[Optional[np.ndarray], int]:
    if data_split == "all":
        return choose_global_positions(total_positions, sample_positions, sample_seed), total_positions

    n_train, n_val, n_test = split_counts(total_positions, train_ratio, val_ratio, test_ratio)
    order = np.random.default_rng(split_seed).permutation(total_positions)
    split_ranges = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }
    start, end = split_ranges[data_split]
    selected = order[start:end]
    split_positions = len(selected)
    if sample_positions is not None and sample_positions < split_positions:
        sample_rng = np.random.default_rng(sample_seed)
        selected = selected[sample_rng.choice(split_positions, size=sample_positions, replace=False)]
    return np.sort(selected.astype(np.int64, copy=False)), split_positions


def split_ranges(total_positions: int, jobs: int) -> List[Tuple[int, int]]:
    return split_absolute_ranges(0, total_positions, jobs)


def split_absolute_ranges(range_start: int, range_end: int, jobs: int) -> List[Tuple[int, int]]:
    jobs = max(1, int(jobs))
    ranges = []
    total_positions = max(0, range_end - range_start)
    for job in range(jobs):
        start = range_start + (total_positions * job) // jobs
        end = range_start + (total_positions * (job + 1)) // jobs
        if start < end:
            ranges.append((start, end))
    return ranges


def iter_position_batches(path: Path, batch_size: int, max_positions: Optional[int]) -> Iterable[np.ndarray]:
    size = path.stat().st_size
    n_position_samples = size // BOARD_SAMPLE_SIZE
    if size % BOARD_SAMPLE_SIZE != 0:
        print(f"warning: {path} has trailing bytes and will be truncated")
    if max_positions is not None:
        n_position_samples = min(n_position_samples, max_positions)
    mmap = np.memmap(path, dtype=BOARD_DTYPE, mode="r", shape=(size // BOARD_SAMPLE_SIZE,))
    for start in range(0, n_position_samples, batch_size):
        end = min(n_position_samples, start + batch_size)
        yield np.asarray(mmap[start:end])


def iter_position_batches_for_global_range(
    path: Path,
    file_global_start: int,
    available_positions: int,
    range_start: int,
    range_end: int,
    batch_size: int,
) -> Iterable[np.ndarray]:
    size = path.stat().st_size
    file_positions = min(size // BOARD_SAMPLE_SIZE, max(0, available_positions - file_global_start))
    file_global_end = file_global_start + file_positions
    overlap_start = max(file_global_start, range_start)
    overlap_end = min(file_global_end, range_end)
    if overlap_start >= overlap_end:
        return
    mmap = np.memmap(path, dtype=BOARD_DTYPE, mode="r", shape=(size // BOARD_SAMPLE_SIZE,))
    local_start = overlap_start - file_global_start
    local_end = overlap_end - file_global_start
    for start in range(local_start, local_end, batch_size):
        end = min(local_end, start + batch_size)
        yield np.asarray(mmap[start:end])


def iter_position_batches_for_selected_globals(
    path: Path,
    file_global_start: int,
    available_positions: int,
    selected_global_positions: np.ndarray,
    batch_size: int,
) -> Iterable[np.ndarray]:
    size = path.stat().st_size
    file_positions = min(size // BOARD_SAMPLE_SIZE, max(0, available_positions - file_global_start))
    file_global_end = file_global_start + file_positions
    left = int(np.searchsorted(selected_global_positions, file_global_start, side="left"))
    right = int(np.searchsorted(selected_global_positions, file_global_end, side="left"))
    if left == right:
        return
    local_indices = selected_global_positions[left:right] - file_global_start
    mmap = np.memmap(path, dtype=BOARD_DTYPE, mode="r", shape=(size // BOARD_SAMPLE_SIZE,))
    for start in range(0, len(local_indices), batch_size):
        yield np.asarray(mmap[local_indices[start : start + batch_size]])


class PositionSampleReader:
    def __init__(self, dat_files: Sequence[Path], available_positions: int):
        self.entries: List[Tuple[int, int, Path, int]] = []
        self.global_ends: List[int] = []
        self.mmaps: Dict[Path, np.memmap] = {}
        global_start = 0
        for path in dat_files:
            total_file_positions = path.stat().st_size // BOARD_SAMPLE_SIZE
            file_positions = min(
                total_file_positions,
                max(0, available_positions - global_start),
            )
            if file_positions <= 0:
                break
            global_end = global_start + file_positions
            self.entries.append(
                (global_start, global_end, path, total_file_positions)
            )
            self.global_ends.append(global_end)
            global_start = global_end

    def get(self, global_position: int):
        entry_index = int(
            np.searchsorted(self.global_ends, global_position, side="right")
        )
        if entry_index >= len(self.entries):
            raise IndexError(f"global position is out of range: {global_position}")
        global_start, global_end, path, total_file_positions = self.entries[
            entry_index
        ]
        if global_position < global_start or global_position >= global_end:
            raise IndexError(f"global position is out of range: {global_position}")
        mmap = self.mmaps.get(path)
        if mmap is None:
            mmap = np.memmap(
                path,
                dtype=BOARD_DTYPE,
                mode="r",
                shape=(total_file_positions,),
            )
            self.mmaps[path] = mmap
        return mmap[global_position - global_start]

    def close(self) -> None:
        self.mmaps.clear()


def iter_position_task_queue(position_task_queue) -> Iterable[int]:
    while True:
        global_position = position_task_queue.get()
        if global_position is None:
            return
        yield int(global_position)


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def hint_cache_key(level: int, state: BoardState, side: int, hint_count: int) -> str:
    return (
        f"level={int(level)}:hint={int(hint_count)}:"
        f"{state.black:016x}:{state.white:016x}:{int(side)}"
    )


class HintScoreCache:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path is not None else None
        self.conn: Optional[sqlite3.Connection] = None
        self.lookups = 0
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), timeout=60.0, isolation_level=None)
            self.conn.execute("PRAGMA busy_timeout = 60000")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hint_scores (
                    key TEXT PRIMARY KEY,
                    scores_json TEXT NOT NULL,
                    raw_hint TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def get(self, key: str) -> Optional[Tuple[Dict[int, float], str]]:
        if self.conn is None:
            return None
        self.lookups += 1
        row = self.conn.execute("SELECT scores_json, raw_hint FROM hint_scores WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        scores = {int(k): float(v) for k, v in json.loads(row[0]).items()}
        return scores, str(row[1])

    def put(self, key: str, scores: Dict[int, float], raw_hint: str) -> None:
        if self.conn is None:
            return
        scores_json = json.dumps({str(k): float(v) for k, v in scores.items()}, sort_keys=True)
        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO hint_scores(key, scores_json, raw_hint, created_at) VALUES (?, ?, ?, ?)",
            (key, scores_json, raw_hint, time.time()),
        )
        self.writes += int(cursor.rowcount > 0)

    def row_count(self) -> Optional[int]:
        if self.conn is None:
            return None
        row = self.conn.execute("SELECT COUNT(*) FROM hint_scores").fetchone()
        return int(row[0])

    def summary(self) -> dict:
        return {
            "path": str(self.path) if self.path is not None else None,
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "rows": self.row_count(),
        }


def hint_scores_with_cache(
    blender: BlendedPolicy,
    state: BoardState,
    side: int,
    hint_count: int,
    hint_cache: Optional[HintScoreCache],
) -> Tuple[Dict[int, float], str]:
    if hint_cache is None:
        return blender.hint_runner.hint_scores(state, side, hint_count)
    key = hint_cache_key(blender.hint_runner.level, state, side, hint_count)
    cached = hint_cache.get(key)
    if cached is not None:
        return cached
    scores, raw_hint = blender.hint_runner.hint_scores(state, side, hint_count)
    hint_cache.put(key, scores, raw_hint)
    return scores, raw_hint


def position_sample_to_state(position_sample) -> Tuple[BoardState, int, int, int, int, int]:
    player = int(position_sample["player"])
    opponent = int(position_sample["opponent"])
    side = int(position_sample["color"])
    policy = int(position_sample["policy"])
    if side == BLACK:
        black, white = player, opponent
    elif side == WHITE:
        black, white = opponent, player
    else:
        raise ValueError(f"invalid side color: {side}")
    move_number = popcount(player | opponent) - 3
    return BoardState(black, white, side), side, policy, player, opponent, move_number


def rank_for_distribution(
    distribution: np.ndarray,
    legal_policies: Sequence[int],
    target_policies: Sequence[int],
    tie_break_order: Optional[Sequence[int]] = None,
) -> int:
    if not legal_policies:
        return POLICY_SIZE + 1
    legal_set = set(legal_policies)
    targets = set(policy for policy in target_policies if policy in legal_set)
    if not targets:
        return POLICY_SIZE + 1

    ordered_ties: List[int] = []
    seen = set()
    for policy in tie_break_order or ():
        if policy in legal_set and policy not in seen:
            ordered_ties.append(policy)
            seen.add(policy)
    for policy in legal_policies:
        if policy not in seen:
            ordered_ties.append(policy)
            seen.add(policy)
    tie_rank = {policy: rank for rank, policy in enumerate(ordered_ties)}
    ranked = sorted(
        legal_policies,
        key=lambda policy: (-float(distribution[policy]), tie_rank[policy]),
    )
    return min(rank for rank, policy in enumerate(ranked, start=1) if policy in targets)


def equivalent_targets(player: int, opponent: int, policy: int) -> List[int]:
    mask = equivalent_policy_mask(
        np.array([player], dtype=np.uint64),
        np.array([opponent], dtype=np.uint64),
        np.array([policy], dtype=np.int64),
    )[0]
    return [int(i) for i in np.nonzero(mask)[0]]


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(text: str) -> List[float]:
    return [float(token) for token in text.split(",") if token.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def make_metrics(blend_params: Sequence[float], n_values: Sequence[int], bucket_names: Sequence[str]) -> Dict[float, dict]:
    return {
        blend: {
            "positions": 0,
            "hits": {n: 0 for n in n_values},
            "bucket_positions": {bucket: 0 for bucket in bucket_names},
            "bucket_hits": {bucket: {n: 0 for n in n_values} for bucket in bucket_names},
        }
        for blend in blend_params
    }


def update_metrics_for_position_sample(
    position_sample,
    blender: BlendedPolicy,
    blend_params: Sequence[float],
    n_values: Sequence[int],
    metrics: Dict[float, dict],
    raw_hint_samples: List[dict],
    raw_hint_limit: int,
    sample_index: int,
    hint_cache: Optional[HintScoreCache],
) -> Tuple[int, int, int]:
    state, side, policy, player, opponent, move_number = position_sample_to_state(position_sample)
    legal_policies = state.legal_policies(side)
    if policy < 0 or policy >= POLICY_SIZE:
        return 0, 1, 0
    if policy not in legal_policies:
        return 0, 0, 1

    policy_dist = blender.policy_distribution(state, side, legal_policies)
    if all(float(blend) >= 1.0 for blend in blend_params):
        raw_hint = ""
        hint_move_order: List[int] = []
        egaroucid_dist = np.zeros(POLICY_SIZE, dtype=np.float32)
    else:
        scores, raw_hint = hint_scores_with_cache(
            blender,
            state,
            side,
            ALL_LEGAL_HINT_COUNT,
            hint_cache,
        )
        hint_move_order = parse_hint_move_order(raw_hint)
        egaroucid_dist = blender.egaroucid_distribution(scores, legal_policies)
    if raw_hint_limit > 0 and len(raw_hint_samples) < raw_hint_limit:
        raw_hint_samples.append({"index": sample_index, "raw_hint": raw_hint})

    targets = equivalent_targets(player, opponent, policy)
    bucket = move_bucket(move_number)

    for blend in blend_params:
        distribution = geometric_blend_distribution(policy_dist, egaroucid_dist, legal_policies, blend)
        tie_break_order = hint_move_order if float(blend) == 0.0 else None
        rank = rank_for_distribution(
            distribution,
            legal_policies,
            targets,
            tie_break_order,
        )
        m = metrics[blend]
        m["positions"] += 1
        m["bucket_positions"][bucket] += 1
        for n in n_values:
            if rank <= n:
                m["hits"][n] += 1
                m["bucket_hits"][bucket][n] += 1
    return 1, 0, 0


def merge_worker_result(target: Dict[float, dict], worker_metrics: Dict[float, dict], blend_params: Sequence[float], n_values: Sequence[int], bucket_names: Sequence[str]) -> None:
    for blend in blend_params:
        dst = target[blend]
        src = worker_metrics[blend]
        dst["positions"] += src["positions"]
        for n in n_values:
            dst["hits"][n] += src["hits"][n]
        for bucket in bucket_names:
            dst["bucket_positions"][bucket] += src["bucket_positions"][bucket]
            for n in n_values:
                dst["bucket_hits"][bucket][n] += src["bucket_hits"][bucket][n]


def merge_hint_cache_stats(stats_rows: Sequence[dict]) -> dict:
    merged = {
        "path": None,
        "lookups": 0,
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "rows": None,
    }
    for stats in stats_rows:
        if not stats:
            continue
        merged["path"] = stats.get("path") or merged["path"]
        for key in ("lookups", "hits", "misses", "writes"):
            merged[key] += int(stats.get(key) or 0)
        if stats.get("rows") is not None:
            merged["rows"] = max(int(stats["rows"]), int(merged["rows"] or 0))
    return merged


def make_worker_progress_event(
    worker_id: int,
    attempted_positions: int,
    metrics: Dict[float, dict],
    blend_params: Sequence[float],
    n_values: Sequence[int],
    start_time: float,
) -> dict:
    rows = []
    for blend in blend_params:
        metric = metrics[blend]
        row = {
            "blend_param": blend,
            "positions": metric["positions"],
        }
        for n in n_values:
            row[f"top{n}_hits"] = metric["hits"][n]
        rows.append(row)
    return {
        "worker_id": worker_id,
        "attempted_positions": attempted_positions,
        "elapsed_sec": time.time() - start_time,
        "results": rows,
    }


def emit_worker_progress(
    progress_queue,
    worker_id: int,
    attempted_positions: int,
    metrics: Dict[float, dict],
    blend_params: Sequence[float],
    n_values: Sequence[int],
    start_time: float,
) -> None:
    if progress_queue is None:
        return
    try:
        progress_queue.put(
            make_worker_progress_event(
                worker_id,
                attempted_positions,
                metrics,
                blend_params,
                n_values,
                start_time,
            )
        )
    except (BrokenPipeError, EOFError, OSError):
        pass


def monitor_worker_progress(progress_queue, stop_event: threading.Event, callback: Callable[[dict], None]) -> None:
    while not stop_event.is_set():
        try:
            callback(progress_queue.get(timeout=0.2))
        except queue.Empty:
            pass
    while True:
        try:
            callback(progress_queue.get_nowait())
        except queue.Empty:
            return


def evaluate_worker(worker_args: dict) -> dict:
    start_time = time.time()
    blend_params = worker_args["blend_params"]
    n_values = worker_args["n_values"]
    bucket_names = worker_args["bucket_names"]
    dat_files = [Path(path) for path in worker_args["dat_files"]]
    available_positions = worker_args["available_positions"]
    selected_positions = worker_args["selected_positions"]
    position_task_queue = worker_args.get("position_task_queue")
    if selected_positions is not None:
        selected_positions = np.array(selected_positions, dtype=np.int64)
    metrics = make_metrics(blend_params, n_values, bucket_names)
    raw_hint_samples: List[dict] = []
    blender = BlendedPolicy(
        weights=Path(worker_args["weights"]),
        egaroucid_exe=Path(worker_args["egaroucid_exe"]),
        egaroucid_level=worker_args["egaroucid_level"],
        egaroucid_threads=worker_args["egaroucid_threads"],
        egaroucid_timeout_sec=worker_args["egaroucid_timeout_sec"],
        score_temperature=worker_args["score_temperature"],
        legal_mask_policy=worker_args["legal_mask_policy"],
        hint_command_stagger_sec=worker_args["hint_command_stagger_sec"],
        hint_command_stagger_lock=worker_args["hint_command_stagger_lock"],
        hint_command_stagger_state=worker_args["hint_command_stagger_state"],
        hint_command_semaphore=worker_args["hint_command_semaphore"],
    )
    invalid_policy = 0
    illegal_label = 0
    processed = 0
    attempted = 0
    file_global_start = 0
    hint_cache = HintScoreCache(Path(worker_args["hint_cache_db"])) if worker_args["hint_cache_db"] else None
    progress_queue = worker_args.get("progress_queue")
    progress_interval_sec = float(worker_args.get("progress_interval_sec") or 0.0)
    last_progress_time = time.monotonic()
    position_reader = (
        PositionSampleReader(dat_files, available_positions)
        if position_task_queue is not None
        else None
    )
    hint_cache_stats = {}

    def process_position_sample(position_sample, sample_index: int) -> None:
        nonlocal attempted
        nonlocal illegal_label
        nonlocal invalid_policy
        nonlocal last_progress_time
        nonlocal processed
        ok, bad_policy, bad_label = update_metrics_for_position_sample(
            position_sample,
            blender,
            blend_params,
            n_values,
            metrics,
            raw_hint_samples,
            worker_args["raw_hint_limit"],
            sample_index,
            hint_cache,
        )
        attempted += 1
        processed += ok
        invalid_policy += bad_policy
        illegal_label += bad_label
        now = time.monotonic()
        if (
            progress_queue is not None
            and now - last_progress_time >= progress_interval_sec
        ):
            emit_worker_progress(
                progress_queue,
                worker_args["worker_id"],
                attempted,
                metrics,
                blend_params,
                n_values,
                start_time,
            )
            last_progress_time = now

    try:
        if position_task_queue is not None:
            for global_position in iter_position_task_queue(position_task_queue):
                process_position_sample(
                    position_reader.get(global_position),
                    global_position,
                )
        else:
            for dat_file in dat_files:
                file_positions = dat_file.stat().st_size // BOARD_SAMPLE_SIZE
                if file_global_start >= available_positions:
                    break
                if selected_positions is None:
                    batches = iter_position_batches_for_global_range(
                        dat_file,
                        file_global_start,
                        available_positions,
                        worker_args["range_start"],
                        worker_args["range_end"],
                        worker_args["batch_size"],
                    )
                else:
                    batches = iter_position_batches_for_selected_globals(
                        dat_file,
                        file_global_start,
                        available_positions,
                        selected_positions,
                        worker_args["batch_size"],
                    )
                for position_samples in batches:
                    for position_sample in position_samples:
                        process_position_sample(
                            position_sample,
                            processed,
                        )
                file_global_start += file_positions
        hint_cache_stats = hint_cache.summary() if hint_cache is not None else {}
    finally:
        emit_worker_progress(
            progress_queue,
            worker_args["worker_id"],
            attempted,
            metrics,
            blend_params,
            n_values,
            start_time,
        )
        if hint_cache is not None:
            hint_cache.close()
        if position_reader is not None:
            position_reader.close()
        blender.hint_runner.close()
    return {
        "worker_id": worker_args["worker_id"],
        "elapsed_sec": time.time() - start_time,
        "attempted_positions": attempted,
        "processed_positions": processed,
        "invalid_policy_samples": invalid_policy,
        "illegal_label_samples": illegal_label,
        "metrics": metrics,
        "raw_hint_samples": raw_hint_samples,
        "hint_cache_stats": hint_cache_stats,
    }


def finalize_result(args: argparse.Namespace, blend_params: Sequence[float], n_values: Sequence[int], bucket_names: Sequence[str], metrics: Dict[float, dict], invalid_policy: int, illegal_label: int, raw_hint_samples: Sequence[dict], available_positions: int, range_start: int, range_end: int, worker_summaries: Sequence[dict], hint_cache_stats: dict) -> dict:
    topn_rows = []
    bucket_rows = []
    for blend in blend_params:
        m = metrics[blend]
        positions = m["positions"]
        for n in n_values:
            hits = m["hits"][n]
            accuracy = hits / positions if positions else 0.0
            topn_rows.append(
                {
                    "blend_param": blend,
                    "top_n": n,
                    "hits": hits,
                    "positions": positions,
                    "accuracy": accuracy,
                }
            )
        for bucket in bucket_names:
            bucket_positions = m["bucket_positions"][bucket]
            for n in n_values:
                hits = m["bucket_hits"][bucket][n]
                bucket_rows.append(
                    {
                        "blend_param": blend,
                        "move_bucket": bucket,
                        "top_n": n,
                        "hits": hits,
                        "positions": bucket_positions,
                        "accuracy": hits / bucket_positions if bucket_positions else 0.0,
                    }
                )

    result = {
        "board_data_dir": str(args.board_data_dir),
        "weights": str(args.weights),
        "egaroucid_exe": str(args.egaroucid_exe),
        "egaroucid_level": args.egaroucid_level,
        "console_hint_count": ALL_LEGAL_HINT_COUNT,
        "egaroucid_policy_support": "all_legal_moves",
        "hint_command_stagger_sec": float(
            getattr(args, "hint_command_stagger_sec", 0.0)
        ),
        "maximum_concurrent_hint_commands": getattr(
            args,
            "max_concurrent_hints",
            None,
        ),
        "blend_params": list(blend_params),
        "data_split": args.data_split,
        "split_seed": args.split_seed if args.data_split != "all" else None,
        "split_ratios": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": args.test_ratio,
        },
        "split_positions": args.split_positions,
        "available_positions": available_positions,
        "range_start": range_start,
        "range_end": range_end,
        "sample_positions": args.sample_positions,
        "sample_seed": args.sample_seed if args.sample_positions is not None else None,
        "jobs": args.jobs,
        "position_scheduling": getattr(
            args,
            "position_scheduling",
            {
                "strategy": "single_process",
                "task_size_positions": None,
            },
        ),
        "hint_cache_db": str(args.hint_cache_db) if args.hint_cache_db is not None else None,
        "hint_cache_stats": hint_cache_stats,
        "shared_computation": {
            "policy_inferences_per_position": 1,
            "hint_score_lookups_per_position": 1,
            "blend_params_evaluated_from_shared_distributions": len(blend_params),
            "note": "各局面のPolicy Network出力とhint出力を1回だけ計算し、すべてのalphaで共用する。",
        },
        "agreement_definition": {
            "metric": "board_symmetry_aware",
            "description": (
                "手番側と相手側の石配置をそれぞれ不変に保つ盤面対称変換で、"
                "WTHORの実着手から移る合法手を同値手とする。"
                "同値手のいずれかが上位N手に入れば一致と数える。"
            ),
        },
        "ranking_definition": {
            "top_n": (
                "確率の降順で手を一意に並べ、WTHORの実着手または盤面対称性により"
                "同値な手のいずれかが先頭N手に含まれるかを数える。"
            ),
            "alpha_zero_tie_break": "Egaroucid for Consoleのhint出力順を使う。",
            "other_tie_break": "確率が同じ場合は合法手の固定順を使う。",
        },
        "invalid_policy_samples": invalid_policy,
        "illegal_label_samples": illegal_label,
        "topn": topn_rows,
        "move_bucket_topn": bucket_rows,
        "raw_hint_samples": list(raw_hint_samples),
        "worker_summaries": list(worker_summaries),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wthor_blend_human_match.json").open("w") as f:
        json.dump(result, f, indent=2)
    write_csv(
        args.output_dir / "wthor_blend_human_match_topn.csv",
        topn_rows,
        ["blend_param", "top_n", "hits", "positions", "accuracy"],
    )
    write_csv(
        args.output_dir / "wthor_blend_human_match_by_move10.csv",
        bucket_rows,
        ["blend_param", "move_bucket", "top_n", "hits", "positions", "accuracy"],
    )
    return result


def evaluate(
    args: argparse.Namespace,
    progress_callback: Optional[Callable[[dict], None]] = None,
    progress_interval_sec: float = 30.0,
) -> dict:
    blend_params = parse_float_list(args.blend_params)
    n_values = sorted(set(parse_int_list(args.top_n)))
    if not n_values or any(n <= 0 for n in n_values):
        raise ValueError("--top-nには1以上の整数を1つ以上指定してください")
    if args.sample_positions is not None and args.sample_positions <= 0:
        raise ValueError("--sample-positions must be positive when set")
    if progress_callback is not None and progress_interval_sec <= 0.0:
        raise ValueError("progress_interval_sec must be positive")
    dat_files = discover_dat_files(args.board_data_dir)
    available_positions = count_position_samples(dat_files, args.max_positions)
    range_start = int(args.range_start)
    range_end = available_positions if args.range_end is None else int(args.range_end)
    if range_start < 0 or range_start > available_positions:
        raise ValueError(f"--range-start must be within 0..{available_positions}")
    if range_end < range_start or range_end > available_positions:
        raise ValueError(f"--range-end must be within {range_start}..{available_positions}")
    if args.data_split != "all" and (range_start != 0 or range_end != available_positions):
        raise ValueError("--data-split cannot be combined with --range-start/--range-end")
    if args.sample_positions is not None and (range_start != 0 or range_end != available_positions):
        raise ValueError("--sample-positions cannot be combined with --range-start/--range-end")
    selected_global_positions, args.split_positions = choose_data_split_positions(
        available_positions,
        args.data_split,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.split_seed,
        args.sample_positions,
        args.sample_seed,
    )
    bucket_names = [move_bucket(i) for i in range(1, 61, 10)]
    metrics = make_metrics(blend_params, n_values, bucket_names)

    invalid_policy = 0
    illegal_label = 0
    raw_hint_samples = []
    hint_cache_stats_rows: List[dict] = []
    if args.jobs > 1:
        process_manager = None
        progress_queue = None
        progress_stop_event = None
        progress_thread = None
        position_task_queue = None
        use_dynamic_position_queue = selected_global_positions is not None
        if progress_callback is not None or use_dynamic_position_queue:
            process_manager = multiprocessing.Manager()
        if progress_callback is not None:
            progress_queue = process_manager.Queue()
            progress_stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=monitor_worker_progress,
                args=(progress_queue, progress_stop_event, progress_callback),
                name="wthor-progress-monitor",
                daemon=True,
            )
            progress_thread.start()
        if selected_global_positions is None:
            work_ranges = split_absolute_ranges(range_start, range_end, args.jobs)
            selected_chunks = [None for _ in work_ranges]
            args.position_scheduling = {
                "strategy": "fixed_contiguous_ranges",
                "task_size_positions": None,
            }
        else:
            worker_count = min(args.jobs, len(selected_global_positions))
            work_ranges = [(range_start, range_end) for _ in range(worker_count)]
            selected_chunks = [None for _ in range(worker_count)]
            position_task_queue = process_manager.Queue()
            for global_position in selected_global_positions:
                position_task_queue.put(int(global_position))
            for _ in range(worker_count):
                position_task_queue.put(None)
            args.position_scheduling = {
                "strategy": "dynamic_shared_position_queue",
                "task_size_positions": 1,
            }
        worker_summaries = []
        raw_hint_remaining = args.raw_hint_samples
        worker_args = []
        for worker_id, ((worker_range_start, worker_range_end), selected_chunk) in enumerate(zip(work_ranges, selected_chunks)):
            worker_args.append(
                {
                    "worker_id": worker_id,
                    "dat_files": [str(path) for path in dat_files],
                    "available_positions": available_positions,
                    "range_start": worker_range_start,
                    "range_end": worker_range_end,
                    "selected_positions": selected_chunk,
                    "position_task_queue": position_task_queue,
                    "weights": str(args.weights),
                    "egaroucid_exe": str(args.egaroucid_exe),
                    "egaroucid_level": args.egaroucid_level,
                    "egaroucid_threads": args.egaroucid_threads,
                    "egaroucid_timeout_sec": args.egaroucid_timeout_sec,
                    "score_temperature": args.score_temperature,
                    "legal_mask_policy": not args.no_legal_mask_policy,
                    "blend_params": blend_params,
                    "n_values": n_values,
                    "bucket_names": bucket_names,
                    "batch_size": args.batch_size,
                    "raw_hint_limit": max(0, raw_hint_remaining),
                    "hint_cache_db": str(args.hint_cache_db) if args.hint_cache_db is not None else None,
                    "progress_queue": progress_queue,
                    "progress_interval_sec": progress_interval_sec,
                    "hint_command_stagger_sec": float(
                        getattr(args, "hint_command_stagger_sec", 0.0)
                    ),
                    "hint_command_stagger_lock": getattr(
                        args,
                        "hint_command_stagger_lock",
                        None,
                    ),
                    "hint_command_stagger_state": getattr(
                        args,
                        "hint_command_stagger_state",
                        None,
                    ),
                    "hint_command_semaphore": getattr(
                        args,
                        "hint_command_semaphore",
                        None,
                    ),
                }
            )
            raw_hint_remaining = 0
        try:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = [executor.submit(evaluate_worker, item) for item in worker_args]
                for future in as_completed(futures):
                    worker_result = future.result()
                    merge_worker_result(metrics, worker_result["metrics"], blend_params, n_values, bucket_names)
                    invalid_policy += worker_result["invalid_policy_samples"]
                    illegal_label += worker_result["illegal_label_samples"]
                    raw_hint_samples.extend(worker_result["raw_hint_samples"])
                    hint_cache_stats_rows.append(worker_result.get("hint_cache_stats", {}))
                    worker_summaries.append(
                        {
                            "worker_id": worker_result["worker_id"],
                            "elapsed_sec": worker_result["elapsed_sec"],
                            "attempted_positions": worker_result["attempted_positions"],
                            "processed_positions": worker_result["processed_positions"],
                            "invalid_policy_samples": worker_result["invalid_policy_samples"],
                            "illegal_label_samples": worker_result["illegal_label_samples"],
                            "hint_cache_stats": worker_result.get("hint_cache_stats", {}),
                        }
                    )
                    if args.verbose:
                        print(
                            f"worker {worker_result['worker_id']} finished "
                            f"{worker_result['processed_positions']} positions in {worker_result['elapsed_sec']:.3f} sec",
                            flush=True,
                        )
        finally:
            if progress_stop_event is not None:
                progress_stop_event.set()
            if progress_thread is not None:
                progress_thread.join()
            if process_manager is not None:
                process_manager.shutdown()
        worker_summaries.sort(key=lambda row: row["worker_id"])
        hint_cache_stats = merge_hint_cache_stats(hint_cache_stats_rows)
        return finalize_result(args, blend_params, n_values, bucket_names, metrics, invalid_policy, illegal_label, raw_hint_samples[: args.raw_hint_samples], available_positions, range_start, range_end, worker_summaries, hint_cache_stats)

    blender = BlendedPolicy(
        weights=args.weights,
        egaroucid_exe=args.egaroucid_exe,
        egaroucid_level=args.egaroucid_level,
        egaroucid_threads=args.egaroucid_threads,
        egaroucid_timeout_sec=args.egaroucid_timeout_sec,
        score_temperature=args.score_temperature,
        legal_mask_policy=not args.no_legal_mask_policy,
        hint_command_stagger_sec=float(
            getattr(args, "hint_command_stagger_sec", 0.0)
        ),
        hint_command_stagger_lock=getattr(args, "hint_command_stagger_lock", None),
        hint_command_stagger_state=getattr(args, "hint_command_stagger_state", None),
        hint_command_semaphore=getattr(args, "hint_command_semaphore", None),
    )

    total_seen = 0
    total_attempted = 0
    global_offset = 0
    hint_cache = HintScoreCache(args.hint_cache_db) if args.hint_cache_db is not None else None
    single_start_time = time.time()
    last_progress_time = time.monotonic()
    for dat_file in dat_files:
        file_positions = dat_file.stat().st_size // BOARD_SAMPLE_SIZE
        if global_offset >= available_positions:
            break
        if selected_global_positions is None:
            batches = iter_position_batches_for_global_range(dat_file, global_offset, available_positions, range_start, range_end, args.batch_size)
        else:
            batches = iter_position_batches_for_selected_globals(dat_file, global_offset, available_positions, selected_global_positions, args.batch_size)
        for position_samples in batches:
            if len(position_samples) == 0:
                continue
            for position_sample in position_samples:
                ok, bad_policy, bad_label = update_metrics_for_position_sample(
                    position_sample,
                    blender,
                    blend_params,
                    n_values,
                    metrics,
                    raw_hint_samples,
                    args.raw_hint_samples,
                    total_seen,
                    hint_cache,
                )
                total_attempted += 1
                total_seen += ok
                invalid_policy += bad_policy
                illegal_label += bad_label
                now = time.monotonic()
                if progress_callback is not None and now - last_progress_time >= progress_interval_sec:
                    progress_callback(
                        make_worker_progress_event(
                            0,
                            total_attempted,
                            metrics,
                            blend_params,
                            n_values,
                            single_start_time,
                        )
                    )
                    last_progress_time = now

            if args.verbose and total_seen and total_seen % args.progress_interval < len(position_samples):
                print(f"seen {total_seen} position_samples")
        global_offset += file_positions

    hint_cache_stats = hint_cache.summary() if hint_cache is not None else {}
    if hint_cache is not None:
        hint_cache.close()
    blender.hint_runner.close()
    if progress_callback is not None:
        progress_callback(
            make_worker_progress_event(
                0,
                total_attempted,
                metrics,
                blend_params,
                n_values,
                single_start_time,
            )
        )
    return finalize_result(args, blend_params, n_values, bucket_names, metrics, invalid_policy, illegal_label, raw_hint_samples[: args.raw_hint_samples], available_positions, range_start, range_end, [], hint_cache_stats)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WTHOR agreement for blended policy/Egaroucid output.")
    parser.add_argument("--board-data-dir", type=Path, default=default_board_data_dir())
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-threads", type=int, default=1)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=30.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--no-legal-mask-policy", action="store_true")
    parser.add_argument("--blend-params", "--alphas", dest="blend_params", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--top-n", default="1,2,3,4,5,8,10,16")
    parser.add_argument("--data-split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--split-seed", type=int, default=613)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--range-start", type=int, default=0)
    parser.add_argument("--range-end", type=int, default=None)
    parser.add_argument("--sample-positions", type=int, default=None, help="Randomly sample this many positions from the selected data split.")
    parser.add_argument("--sample-seed", type=int, default=613)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--hint-cache-db", type=Path, default=None)
    parser.add_argument("--raw-hint-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output" / "wthor_blend_human_match")
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    result = evaluate(args)
    print("weights", result["weights"])
    print("board_data_dir", result["board_data_dir"])
    print("egaroucid_level", result["egaroucid_level"])
    print("data_split", result["data_split"])
    print("split_positions", result["split_positions"])
    print("available_positions", result["available_positions"])
    print("range_start", result["range_start"])
    print("range_end", result["range_end"])
    if result["sample_positions"] is not None:
        print("sample_positions", result["sample_positions"])
        print("sample_seed", result["sample_seed"])
    print("jobs", result["jobs"])
    if result["hint_cache_db"] is not None:
        print("hint_cache_db", result["hint_cache_db"])
        print("hint_cache_stats", json.dumps(result["hint_cache_stats"], sort_keys=True))
    print("invalid_policy_samples", result["invalid_policy_samples"])
    print("illegal_label_samples", result["illegal_label_samples"])
    for row in result["topn"]:
        if row["top_n"] == 1:
            print(
                f"blend {row['blend_param']:g} top-1 symmetry-aware "
                f"{row['accuracy'] * 100.0:.3f}% "
                f"({row['positions']} positions)"
            )
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
