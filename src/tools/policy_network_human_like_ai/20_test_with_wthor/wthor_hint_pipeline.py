#!/usr/bin/env python3
"""WTHOR人間着手一致率実験のhint探索、キャッシュ、並列実行を担う。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
BLEND_DIR = SCRIPT_DIR.parents[0] / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import (  # noqa: E402
    ALL_LEGAL_HINT_COUNT,
    BoardState,
    EgaroucidHintRunner,
    parse_hint_move_order,
    require_complete_egaroucid_scores,
)
import evaluate_wthor_blend_human_match as wthor  # noqa: E402
from wthor_human_match_evaluation import (  # noqa: E402
    CONSOLE_ONLY_HINT_COUNT,
    CONSOLE_REFERENCE_LEVEL,
    HintData,
    PositionGroup,
    StateKey,
)


_HINT_CANCEL_EVENT = None


@dataclass(frozen=True)
class HintTask:
    level: int
    hint_count: int
    states: Tuple[StateKey, ...]
    egaroucid_exe: str
    egaroucid_threads: int
    timeout_sec: float
    max_retries: int


def initialize_hint_worker(cancel_event) -> None:
    global _HINT_CANCEL_EVENT
    _HINT_CANCEL_EVENT = cancel_event


def hint_worker_cancelled() -> bool:
    return (
        _HINT_CANCEL_EVENT is not None
        and _HINT_CANCEL_EVENT.is_set()
    )


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_hint_data(
    state: BoardState,
    side: int,
    scores: Dict[int, float],
    raw_hint: str,
    hint_count: int,
) -> Tuple[int, ...]:
    legal = state.legal_policies(side)
    move_order = tuple(parse_hint_move_order(raw_hint))
    if hint_count == ALL_LEGAL_HINT_COUNT:
        require_complete_egaroucid_scores(scores, legal)
        expected = set(legal)
    else:
        expected_count = min(hint_count, len(legal))
        expected = set(move_order)
        if (
            len(move_order) != expected_count
            or not expected.issubset(set(legal))
        ):
            raise ValueError(
                "Egaroucid hint order is not a legal top-N result: "
                f"legal={sorted(legal)}, order={list(move_order)}, "
                f"hint_count={hint_count}"
            )
    if (
        set(scores) != expected
        or set(move_order) != expected
        or len(move_order) != len(expected)
    ):
        raise ValueError(
            "Egaroucid hint scores and move order differ: "
            f"scores={sorted(scores)}, order={list(move_order)}"
        )
    return move_order


class HintCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path),
            timeout=60.0,
            isolation_level=None,
        )
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

    @staticmethod
    def cache_key(
        level: int,
        hint_count: int,
        state_key: StateKey,
    ) -> str:
        state = BoardState(*state_key)
        return wthor.hint_cache_key(
            level,
            state,
            state_key[2],
            hint_count,
        )

    def load(
        self,
        level: int,
        hint_count: int,
        state_keys: Sequence[StateKey],
    ) -> Dict[StateKey, HintData]:
        cache_to_state = {
            self.cache_key(level, hint_count, state_key): state_key
            for state_key in state_keys
        }
        result: Dict[StateKey, HintData] = {}
        cache_keys = list(cache_to_state)
        for start in range(0, len(cache_keys), 400):
            batch = cache_keys[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                (
                    "SELECT key, scores_json, raw_hint FROM hint_scores "
                    f"WHERE key IN ({placeholders})"
                ),
                batch,
            )
            for cache_key, scores_json, raw_hint in rows:
                state_key = cache_to_state[str(cache_key)]
                scores = {
                    int(policy): float(score)
                    for policy, score in json.loads(scores_json).items()
                }
                state = BoardState(*state_key)
                move_order = validate_hint_data(
                    state,
                    state_key[2],
                    scores,
                    str(raw_hint),
                    hint_count,
                )
                result[state_key] = HintData(scores, move_order)
        return result

    def put_many(
        self,
        level: int,
        hint_count: int,
        rows: Sequence[Tuple[StateKey, Dict[int, float], str]],
    ) -> int:
        if not rows:
            return 0
        created_at = time.time()
        values = [
            (
                self.cache_key(level, hint_count, state_key),
                json.dumps(
                    {
                        str(policy): float(score)
                        for policy, score in scores.items()
                    },
                    sort_keys=True,
                ),
                raw_hint,
                created_at,
            )
            for state_key, scores, raw_hint in rows
        ]
        before = self.conn.total_changes
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.executemany(
                (
                    "INSERT OR REPLACE INTO hint_scores"
                    "(key, scores_json, raw_hint, created_at) "
                    "VALUES (?, ?, ?, ?)"
                ),
                values,
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return int(self.conn.total_changes - before)

    def row_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM hint_scores"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        conn = self.conn
        self.conn = None
        if conn is not None:
            conn.close()


class CpuMonitor:
    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.samples: List[dict] = []

    def start(self) -> None:
        try:
            import psutil  # type: ignore
        except ImportError:
            return
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(
            target=self._run,
            name="cpu-monitor",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        import psutil  # type: ignore

        root = psutil.Process(os.getpid())
        while not self.stop_event.wait(self.interval_sec):
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.Error:
                processes = [root]
            tree_rss = 0
            egaroucid_processes = 0
            for process in processes:
                try:
                    tree_rss += process.memory_info().rss
                    if "egaroucid" in process.name().lower():
                        egaroucid_processes += 1
                except psutil.Error:
                    pass
            memory = psutil.virtual_memory()
            self.samples.append(
                {
                    "cpu_percent": float(
                        psutil.cpu_percent(interval=None)
                    ),
                    "available_memory_mib": float(
                        memory.available / (1024.0 * 1024.0)
                    ),
                    "process_tree_memory_mib": float(
                        tree_rss / (1024.0 * 1024.0)
                    ),
                    "egaroucid_processes": egaroucid_processes,
                }
            )

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_sec * 2.0 + 1.0)
        return {
            "available": bool(self.samples),
            "samples": len(self.samples),
            "average_cpu_percent": (
                sum(row["cpu_percent"] for row in self.samples)
                / len(self.samples)
                if self.samples
                else None
            ),
            "maximum_cpu_percent": (
                max(row["cpu_percent"] for row in self.samples)
                if self.samples
                else None
            ),
            "minimum_available_memory_mib": (
                min(
                    row["available_memory_mib"]
                    for row in self.samples
                )
                if self.samples
                else None
            ),
            "maximum_process_tree_memory_mib": (
                max(
                    row["process_tree_memory_mib"]
                    for row in self.samples
                )
                if self.samples
                else None
            ),
            "maximum_egaroucid_processes": (
                max(
                    row["egaroucid_processes"]
                    for row in self.samples
                )
                if self.samples
                else None
            ),
        }


def run_hint_task(task: HintTask) -> dict:
    started_at = time.time()
    start = time.perf_counter()
    runner = None
    rows = []
    try:
        for state_key in task.states:
            if hint_worker_cancelled():
                raise InterruptedError("hint task cancelled")
            state = BoardState(*state_key)
            side = state_key[2]
            for attempt in range(task.max_retries + 1):
                if runner is None:
                    runner = EgaroucidHintRunner(
                        Path(task.egaroucid_exe),
                        level=task.level,
                        threads=task.egaroucid_threads,
                        timeout_sec=task.timeout_sec,
                        persistent=True,
                        cancel_event=_HINT_CANCEL_EVENT,
                    )
                try:
                    scores, raw_hint = runner.hint_scores(
                        state,
                        side,
                        task.hint_count,
                    )
                    validate_hint_data(
                        state,
                        side,
                        scores,
                        raw_hint,
                        task.hint_count,
                    )
                    break
                except (OSError, RuntimeError, ValueError):
                    runner.close()
                    runner = None
                    if hint_worker_cancelled():
                        raise InterruptedError(
                            "hint task cancelled"
                        )
                    if attempt >= task.max_retries:
                        raise
            rows.append((state_key, scores, raw_hint))
    finally:
        if runner is not None:
            runner.close()
    return {
        "level": task.level,
        "rows": rows,
        "states": len(rows),
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "elapsed_sec": time.perf_counter() - start,
    }


def make_hint_tasks(
    levels: Sequence[int],
    states_by_level: Dict[int, Sequence[StateKey]],
    workers: int,
    egaroucid_exe: Path,
    egaroucid_threads: int,
    timeout_sec: float,
    max_retries: int,
    cached_by_level: Optional[
        Dict[int, Dict[StateKey, HintData]]
    ] = None,
) -> List[HintTask]:
    tasks = []
    for level in sorted(levels, reverse=True):
        states = list(states_by_level[level])
        shard_count = min(workers, len(states))
        if shard_count == 0:
            continue
        for shard in range(shard_count):
            shard_states = tuple(states[shard::shard_count])
            cached = (
                cached_by_level.get(level, {})
                if cached_by_level is not None
                else {}
            )
            if shard_states and not all(
                state_key in cached
                for state_key in shard_states
            ):
                tasks.append(
                    HintTask(
                        level=level,
                        hint_count=(
                            ALL_LEGAL_HINT_COUNT
                            if level == CONSOLE_REFERENCE_LEVEL
                            else CONSOLE_ONLY_HINT_COUNT
                        ),
                        states=shard_states,
                        egaroucid_exe=str(egaroucid_exe),
                        egaroucid_threads=egaroucid_threads,
                        timeout_sec=timeout_sec,
                        max_retries=max_retries,
                    )
                )
    return tasks


def collect_hints(
    groups: Sequence[PositionGroup],
    levels: Sequence[int],
    output_dir: Path,
    workers: int,
    egaroucid_exe: Path,
    egaroucid_threads: int,
    timeout_sec: float,
    max_retries: int,
    progress_interval_sec: float,
) -> Tuple[
    Dict[int, Dict[StateKey, HintData]],
    Dict[int, dict],
    Dict[int, dict],
    dict,
]:
    state_keys = [group.key for group in groups]
    caches: Dict[int, HintCache] = {}
    hints: Dict[int, Dict[StateKey, HintData]] = {}
    missing_by_level: Dict[int, List[StateKey]] = {}
    cache_stats: Dict[int, dict] = {}
    level_timing: Dict[int, dict] = {
        level: {
            "task_count": 0,
            "engine_seconds": 0.0,
            "started_at_unix": None,
            "finished_at_unix": None,
            "elapsed_sec": 0.0,
        }
        for level in levels
    }

    try:
        for level in levels:
            cache = HintCache(
                output_dir
                / (
                    f"hint_score_cache_level{level}"
                    f"_hint{ALL_LEGAL_HINT_COUNT if level == CONSOLE_REFERENCE_LEVEL else CONSOLE_ONLY_HINT_COUNT}"
                    ".sqlite3"
                )
            )
            caches[level] = cache
            hint_count = (
                ALL_LEGAL_HINT_COUNT
                if level == CONSOLE_REFERENCE_LEVEL
                else CONSOLE_ONLY_HINT_COUNT
            )
            cached = cache.load(level, hint_count, state_keys)
            hints[level] = cached
            missing_by_level[level] = [
                state_key
                for state_key in state_keys
                if state_key not in cached
            ]
            cache_stats[level] = {
                "path": str(cache.path),
                "lookups": len(state_keys),
                "hits": len(cached),
                "misses": len(missing_by_level[level]),
                "writes": 0,
                "scheduled_computations": 0,
                "rows": None,
            }

        tasks = make_hint_tasks(
            levels,
            {
                level: state_keys
                for level in levels
            },
            workers,
            egaroucid_exe,
            egaroucid_threads,
            timeout_sec,
            max_retries,
            cached_by_level=hints,
        )
        for task in tasks:
            cache_stats[task.level][
                "scheduled_computations"
            ] += len(task.states)
        total_states = sum(len(task.states) for task in tasks)
        completed_states = 0
        started = time.perf_counter()
        last_report = started

        def accept_result(future, result: dict) -> None:
            nonlocal completed_states, last_report
            task = future_to_task[future]
            level = int(result["level"])
            rows = result["rows"]
            cache_stats[level]["writes"] += caches[level].put_many(
                level,
                int(task.hint_count),
                rows,
            )
            for state_key, scores, raw_hint in rows:
                state_key = tuple(state_key)
                state = BoardState(*state_key)
                move_order = validate_hint_data(
                    state,
                    state_key[2],
                    scores,
                    raw_hint,
                    int(task.hint_count),
                )
                hints[level][state_key] = HintData(
                    scores,
                    move_order,
                )
            completed_states += int(result["states"])
            timing = level_timing[level]
            timing["task_count"] += 1
            timing["engine_seconds"] += float(result["elapsed_sec"])
            started_at = float(result["started_at_unix"])
            finished_at = float(result["finished_at_unix"])
            timing["started_at_unix"] = (
                started_at
                if timing["started_at_unix"] is None
                else min(timing["started_at_unix"], started_at)
            )
            timing["finished_at_unix"] = (
                finished_at
                if timing["finished_at_unix"] is None
                else max(timing["finished_at_unix"], finished_at)
            )

            now = time.perf_counter()
            if (
                now - last_report >= progress_interval_sec
                or completed_states == total_states
            ):
                elapsed = now - started
                rate = (
                    completed_states / elapsed
                    if elapsed > 0.0
                    else 0.0
                )
                remaining = total_states - completed_states
                eta = remaining / rate if rate > 0.0 else 0.0
                print(
                    "hint進捗 "
                    f"{completed_states:,}/{total_states:,} "
                    f"({completed_states / total_states:.1%}) "
                    f"経過 {format_duration(elapsed)} "
                    f"ETA {format_duration(eta)}",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = now

        monitor = CpuMonitor()
        monitor.start()
        try:
            if tasks:
                context = multiprocessing.get_context("spawn")
                cancel_event = context.Event()
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=initialize_hint_worker,
                    initargs=(cancel_event,),
                )
                future_to_task = {}
                processed_futures = set()
                try:
                    for task in tasks:
                        future = executor.submit(run_hint_task, task)
                        future_to_task[future] = task
                    for future in as_completed(future_to_task):
                        processed_futures.add(future)
                        result = future.result()
                        accept_result(future, result)
                except BaseException:
                    cancel_event.set()
                    for future in future_to_task:
                        future.cancel()
                    executor.shutdown(
                        wait=True,
                        cancel_futures=True,
                    )
                    for future in future_to_task:
                        if (
                            future in processed_futures
                            or future.cancelled()
                            or not future.done()
                        ):
                            continue
                        try:
                            result = future.result()
                            accept_result(future, result)
                        except BaseException:
                            pass
                    raise
                else:
                    executor.shutdown(wait=True)
        finally:
            cpu_stats = monitor.stop()

        for level in levels:
            timing = level_timing[level]
            if (
                timing["started_at_unix"] is not None
                and timing["finished_at_unix"] is not None
            ):
                timing["elapsed_sec"] = (
                    timing["finished_at_unix"]
                    - timing["started_at_unix"]
                )
            if len(hints[level]) != len(state_keys):
                raise RuntimeError(
                    f"level {level}: expected {len(state_keys)} hint rows, "
                    f"got {len(hints[level])}"
                )
            cache_stats[level]["rows"] = caches[level].row_count()
        return hints, cache_stats, level_timing, cpu_stats
    finally:
        for cache in caches.values():
            cache.close()


def available_memory_mib() -> Optional[float]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    return float(psutil.virtual_memory().available / (1024.0 * 1024.0))


def check_memory(
    workers: int,
    estimated_engine_memory_mib: float,
    minimum_remaining_memory_mib: float,
) -> Optional[float]:
    available = available_memory_mib()
    if available is None:
        return None
    required = (
        workers * estimated_engine_memory_mib
        + minimum_remaining_memory_mib
    )
    if available < required:
        raise RuntimeError(
            "空き物理メモリが不足しています: "
            f"available={available:.0f} MiB, required={required:.0f} MiB "
            f"({workers} workers x {estimated_engine_memory_mib:.0f} MiB "
            f"+ reserve {minimum_remaining_memory_mib:.0f} MiB)"
        )
    return available
