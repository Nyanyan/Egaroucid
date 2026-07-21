#!/usr/bin/env python3
"""WTHOR人間着手一致率実験のhint探索、キャッシュ、並列実行を担う。"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import queue
import sqlite3
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple


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
_HINT_PROGRESS_QUEUE = None

HINT_RESULT_BATCH_SIZE = 8
HINT_RESULT_BATCH_INTERVAL_SEC = 5.0
HINT_EVENT_POLL_INTERVAL_SEC = 0.5


@dataclass(frozen=True)
class HintTask:
    level: int
    hint_count: int
    states: Tuple[StateKey, ...]
    egaroucid_exe: str
    egaroucid_threads: int
    timeout_sec: float
    max_retries: int


HintRowsCallback = Callable[
    [int, Sequence[Tuple[StateKey, HintData]]],
    None,
]
HintProgressCallback = Callable[[dict], None]


def hint_task_id(task: HintTask) -> tuple:
    return (
        task.level,
        task.states[0] if task.states else None,
        len(task.states),
    )


def initialize_hint_worker(cancel_event, progress_queue=None) -> None:
    """Initialize spawn workers without passing a Queue in every task."""
    global _HINT_CANCEL_EVENT, _HINT_PROGRESS_QUEUE
    _HINT_CANCEL_EVENT = cancel_event
    _HINT_PROGRESS_QUEUE = progress_queue
    if progress_queue is not None:
        try:
            # Events are best-effort and every row is also in the Future.
            # Do not let a full event pipe prevent a worker from exiting while
            # the parent is cancelling and waiting in executor.shutdown().
            progress_queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass


def hint_worker_cancelled() -> bool:
    return (
        _HINT_CANCEL_EVENT is not None
        and _HINT_CANCEL_EVENT.is_set()
    )


def emit_hint_worker_event(event: dict) -> None:
    """Best-effort worker-to-parent event delivery.

    Every result row is also returned by :func:`run_hint_task`, so a broken
    progress pipe cannot lose experiment data.  It only reduces live
    visibility until the task future completes.
    """
    if _HINT_PROGRESS_QUEUE is None:
        return
    try:
        _HINT_PROGRESS_QUEUE.put(event)
    except (BrokenPipeError, EOFError, OSError, ValueError):
        pass


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
        self.lock = threading.Lock()

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
            sample = {
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
            with self.lock:
                self.samples.append(sample)

    def snapshot(self) -> dict:
        with self.lock:
            if not self.samples:
                return {"available": False}
            return {"available": True, **self.samples[-1]}

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_sec * 2.0 + 1.0)
        with self.lock:
            samples = list(self.samples)
        return {
            "available": bool(samples),
            "samples": len(samples),
            "average_cpu_percent": (
                sum(row["cpu_percent"] for row in samples)
                / len(samples)
                if samples
                else None
            ),
            "maximum_cpu_percent": (
                max(row["cpu_percent"] for row in samples)
                if samples
                else None
            ),
            "minimum_available_memory_mib": (
                min(
                    row["available_memory_mib"]
                    for row in samples
                )
                if samples
                else None
            ),
            "maximum_process_tree_memory_mib": (
                max(
                    row["process_tree_memory_mib"]
                    for row in samples
                )
                if samples
                else None
            ),
            "maximum_egaroucid_processes": (
                max(
                    row["egaroucid_processes"]
                    for row in samples
                )
                if samples
                else None
            ),
        }


def run_hint_task(task: HintTask) -> dict:
    started_at = time.time()
    start = time.perf_counter()
    runner = None
    rows = []
    pending_rows = []
    last_batch_at = start
    task_id = hint_task_id(task)
    final_attempt_error_sent = False
    failure = None

    def flush_rows() -> None:
        nonlocal last_batch_at
        if not pending_rows:
            return
        emit_hint_worker_event(
            {
                "kind": "rows",
                "task_id": task_id,
                "level": task.level,
                "hint_count": task.hint_count,
                "rows": tuple(pending_rows),
                "worker_pid": os.getpid(),
                "emitted_at_unix": time.time(),
            }
        )
        pending_rows.clear()
        last_batch_at = time.perf_counter()

    emit_hint_worker_event(
        {
            "kind": "task_started",
            "task_id": task_id,
            "level": task.level,
            "states": len(task.states),
            "worker_pid": os.getpid(),
            "emitted_at_unix": started_at,
        }
    )
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
                except (OSError, RuntimeError, ValueError) as exc:
                    will_retry = attempt < task.max_retries
                    emit_hint_worker_event(
                        {
                            "kind": (
                                "retry"
                                if will_retry
                                else "attempt_error"
                            ),
                            "task_id": task_id,
                            "level": task.level,
                            "state_key": state_key,
                            "attempt": attempt + 1,
                            "maximum_attempts": task.max_retries + 1,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "worker_pid": os.getpid(),
                            "emitted_at_unix": time.time(),
                        }
                    )
                    final_attempt_error_sent = not will_retry
                    runner.close()
                    runner = None
                    if hint_worker_cancelled():
                        raise InterruptedError(
                            "hint task cancelled"
                        )
                    if attempt >= task.max_retries:
                        raise
            rows.append((state_key, scores, raw_hint))
            pending_rows.append((state_key, scores, raw_hint))
            now = time.perf_counter()
            if (
                len(pending_rows) >= HINT_RESULT_BATCH_SIZE
                or now - last_batch_at
                >= HINT_RESULT_BATCH_INTERVAL_SEC
            ):
                flush_rows()
    except BaseException as exc:
        failure = exc
        if not final_attempt_error_sent and not (
            isinstance(exc, InterruptedError)
            and hint_worker_cancelled()
        ):
            emit_hint_worker_event(
                {
                    "kind": "fatal",
                    "task_id": task_id,
                    "level": task.level,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "worker_pid": os.getpid(),
                    "emitted_at_unix": time.time(),
                }
            )
        raise
    finally:
        flush_rows()
        if runner is not None:
            runner.close()
        emit_hint_worker_event(
            {
                "kind": "task_finished",
                "task_id": task_id,
                "level": task.level,
                "states": len(rows),
                "status": "failed" if failure is not None else "ok",
                "worker_pid": os.getpid(),
                "emitted_at_unix": time.time(),
            }
        )
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
    tasks_by_level: Dict[int, List[HintTask]] = {}
    for level in sorted(levels, reverse=True):
        level_tasks = []
        states = list(states_by_level[level])
        shard_count = (
            min(workers, len(states))
            if level == CONSOLE_REFERENCE_LEVEL
            else min(1, len(states))
        )
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
                level_tasks.append(
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
        if level_tasks:
            tasks_by_level[level] = level_tasks

    # Levels 1--19 each keep one Console alive for their complete state stream.
    # Level 21 is the expensive reference and uses all remaining process slots
    # through fixed shards.  Once a lower level finishes, the bounded submitter
    # below starts another queued level-21 shard in the released slot.
    level_order = sorted(tasks_by_level, reverse=True)
    frontier = [tasks_by_level[level][0] for level in level_order]
    remaining = [
        task
        for level in level_order
        for task in tasks_by_level[level][1:]
    ]
    return [*frontier, *remaining]


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
    on_rows: Optional[HintRowsCallback] = None,
    progress_callback: Optional[HintProgressCallback] = None,
) -> Tuple[
    Dict[int, Dict[StateKey, HintData]],
    Dict[int, dict],
    Dict[int, dict],
    dict,
]:
    if progress_interval_sec <= 0.0:
        raise ValueError("progress_interval_sec must be positive")

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
        scheduled_keys_by_level = {
            level: set()
            for level in levels
        }
        for task in tasks:
            cache_stats[task.level][
                "scheduled_computations"
            ] += len(task.states)
            scheduled_keys_by_level[task.level].update(task.states)

        notified_keys_by_level = {
            level: set()
            for level in levels
        }

        def notify_rows(
            level: int,
            rows: Sequence[Tuple[StateKey, HintData]],
        ) -> None:
            fresh_rows = []
            notified = notified_keys_by_level[level]
            for state_key, hint_data in rows:
                if state_key in notified:
                    continue
                notified.add(state_key)
                fresh_rows.append((state_key, hint_data))
            if fresh_rows and on_rows is not None:
                on_rows(level, tuple(fresh_rows))

        # A cached row in an incomplete fixed shard will be recomputed.  Do
        # not publish its old value to live metrics before its replacement.
        for level in levels:
            scheduled_keys = scheduled_keys_by_level[level]
            notify_rows(
                level,
                tuple(
                    (state_key, hint_data)
                    for state_key, hint_data in hints[level].items()
                    if state_key not in scheduled_keys
                ),
            )

        total_states = sum(len(task.states) for task in tasks)
        completed_states = 0
        completed_work_keys = set()
        completed_tasks = 0
        retry_count = 0
        attempt_error_count = 0
        active_task_ids = set()
        reported_fatal_task_ids = set()
        started = time.perf_counter()
        last_success_at = None

        def accept_rows(
            level: int,
            hint_count: int,
            rows: Sequence[Tuple[StateKey, Dict[int, float], str]],
        ) -> int:
            nonlocal completed_states, last_success_at
            if level not in caches:
                raise RuntimeError(
                    f"received hint rows for unexpected level {level}"
                )
            expected_hint_count = (
                ALL_LEGAL_HINT_COUNT
                if level == CONSOLE_REFERENCE_LEVEL
                else CONSOLE_ONLY_HINT_COUNT
            )
            if hint_count != expected_hint_count:
                raise RuntimeError(
                    f"level {level}: expected hint_count "
                    f"{expected_hint_count}, got {hint_count}"
                )

            accepted_cache_rows = []
            accepted_hint_rows = []
            for raw_state_key, raw_scores, raw_hint in rows:
                state_key = tuple(raw_state_key)
                work_key = (level, state_key)
                if work_key in completed_work_keys:
                    continue
                if state_key not in scheduled_keys_by_level[level]:
                    raise RuntimeError(
                        f"level {level}: received an unscheduled state "
                        f"{state_key}"
                    )
                scores = {
                    int(policy): float(score)
                    for policy, score in raw_scores.items()
                }
                raw_hint = str(raw_hint)
                state = BoardState(*state_key)
                move_order = validate_hint_data(
                    state,
                    state_key[2],
                    scores,
                    raw_hint,
                    hint_count,
                )
                hint_data = HintData(scores, move_order)
                completed_work_keys.add(work_key)
                accepted_cache_rows.append(
                    (state_key, scores, raw_hint)
                )
                accepted_hint_rows.append((state_key, hint_data))

            if not accepted_cache_rows:
                return 0
            cache_stats[level]["writes"] += caches[level].put_many(
                level,
                hint_count,
                accepted_cache_rows,
            )
            for state_key, hint_data in accepted_hint_rows:
                hints[level][state_key] = hint_data
            completed_states += len(accepted_cache_rows)
            last_success_at = time.perf_counter()
            notify_rows(level, accepted_hint_rows)
            return len(accepted_cache_rows)

        def accept_task_result(task: HintTask, result: dict) -> None:
            nonlocal completed_tasks
            active_task_ids.discard(hint_task_id(task))
            level = int(result["level"])
            if level != task.level:
                raise RuntimeError(
                    f"hint task for level {task.level} returned level {level}"
                )
            rows = result["rows"]
            if int(result["states"]) != len(rows):
                raise RuntimeError(
                    f"level {level}: task state count does not match rows"
                )
            accept_rows(level, task.hint_count, rows)
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
            completed_tasks += 1

        monitor = CpuMonitor()
        monitor.start()
        next_report_at = started

        def progress_snapshot(final: bool = False) -> dict:
            now = time.perf_counter()
            elapsed = now - started
            rate = (
                completed_states / elapsed
                if elapsed > 0.0 and completed_states
                else 0.0
            )
            remaining = max(0, total_states - completed_states)
            eta = (
                0.0
                if remaining == 0
                else remaining / rate
                if rate > 0.0
                else None
            )
            active_tasks_by_level = {}
            for task_id in active_task_ids:
                level = int(task_id[0])
                active_tasks_by_level[level] = (
                    active_tasks_by_level.get(level, 0) + 1
                )
            return {
                "kind": "hint_progress",
                "final": final,
                "completed_states": completed_states,
                "total_states": total_states,
                "progress_fraction": (
                    completed_states / total_states
                    if total_states
                    else 1.0
                ),
                "elapsed_sec": elapsed,
                "rate_states_per_sec": rate,
                "eta_sec": eta,
                "seconds_since_last_hint": (
                    now - last_success_at
                    if last_success_at is not None
                    else None
                ),
                "cache_hits": sum(
                    row["hits"] for row in cache_stats.values()
                ),
                "available_hints": sum(
                    len(level_hints)
                    for level_hints in hints.values()
                ),
                "reported_hints_by_level": {
                    level: len(notified_keys_by_level[level])
                    for level in levels
                },
                "target_hints_by_level": {
                    level: len(state_keys)
                    for level in levels
                },
                "target_hints": len(state_keys) * len(levels),
                "completed_tasks": completed_tasks,
                "total_tasks": len(tasks),
                "active_tasks": len(active_task_ids),
                "active_tasks_by_level": active_tasks_by_level,
                "retry_count": retry_count,
                "attempt_error_count": attempt_error_count,
                "cpu": monitor.snapshot(),
            }

        def report_progress(final: bool = False) -> None:
            snapshot = progress_snapshot(final=final)
            eta = snapshot["eta_sec"]
            eta_text = (
                format_duration(eta)
                if eta is not None
                else "算出待ち"
            )
            since_hint = snapshot["seconds_since_last_hint"]
            last_hint_status = (
                f"{format_duration(since_hint)}前"
                if since_hint is not None
                else "全件キャッシュ済み"
                if snapshot["total_states"] == 0
                else "初回結果待ち"
            )
            cpu = snapshot["cpu"]
            resource_line = None
            if cpu.get("available"):
                resource_line = (
                    f"  [資源] CPU {cpu['cpu_percent']:.0f}%"
                    f" | 空きRAM {cpu['available_memory_mib']:.0f} MiB"
                    f" | 関連メモリ "
                    f"{cpu['process_tree_memory_mib'] / 1024.0:.1f} GiB"
                    f" | Egaroucid {cpu['egaroucid_processes']}"
                )
            progress_kind = "hint進捗・完了" if final else "hint進捗"
            progress_label = (
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
                f"[{progress_kind}] "
            )
            active_level_text = ", ".join(
                f"level {level}:{count}"
                for level, count in sorted(
                    snapshot["active_tasks_by_level"].items(),
                    reverse=True,
                )
            )
            if active_level_text:
                active_level_text = f" | {active_level_text}"
            lines = [
                f"{progress_label}{snapshot['completed_states']:,}/"
                f"{snapshot['total_states']:,}件のhint計算 "
                f"({snapshot['progress_fraction']:.1%})"
                f" | 経過 {format_duration(snapshot['elapsed_sec'])}"
                f" | 概算残り {eta_text}"
                f" | {snapshot['rate_states_per_sec']:.2f}件/秒",
                f"  [実行状況] task完了 {snapshot['completed_tasks']}/"
                f"{snapshot['total_tasks']}"
                f" | 稼働 {snapshot['active_tasks']}"
                f"{active_level_text}"
                f" | cache検出 {snapshot['cache_hits']:,}"
                f" | 再試行 {snapshot['retry_count']}"
                f" | 最終結果 {last_hint_status}",
            ]
            if resource_line is not None:
                lines.append(resource_line)
            sys.stderr.write(
                "\n".join(lines) + "\n"
            )
            sys.stderr.flush()
            if progress_callback is not None:
                progress_callback(snapshot)

        def format_state_key(state_key) -> str:
            try:
                black, white, side = state_key
                return f"({int(black):#018x},{int(white):#018x},{side})"
            except (TypeError, ValueError):
                return repr(state_key)

        def handle_worker_event(event: dict) -> None:
            nonlocal retry_count, attempt_error_count
            kind = event.get("kind")
            task_id = event.get("task_id")
            if kind == "rows":
                accept_rows(
                    int(event["level"]),
                    int(event["hint_count"]),
                    event["rows"],
                )
            elif kind == "task_started":
                active_task_ids.add(task_id)
            elif kind == "task_finished":
                active_task_ids.discard(task_id)
            elif kind in ("retry", "attempt_error"):
                if kind == "retry":
                    retry_count += 1
                    label = "再試行"
                else:
                    attempt_error_count += 1
                    label = "エラー"
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
                    f"[hint{label}] level {event['level']} "
                    f"状態 {format_state_key(event.get('state_key'))} "
                    f"試行 {event['attempt']}/"
                    f"{event['maximum_attempts']}: "
                    f"{event['error_type']}: {event['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "fatal":
                reported_fatal_task_ids.add(task_id)
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
                    f"[hint致命的エラー] level {event['level']}: "
                    f"{event['error_type']}: {event['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"unknown hint worker event: {kind!r}"
                )

        def drain_worker_events(progress_queue) -> None:
            while True:
                try:
                    event = progress_queue.get_nowait()
                except queue.Empty:
                    return
                handle_worker_event(event)

        try:
            report_progress(final=not tasks)
            next_report_at = started + progress_interval_sec
            if tasks:
                context = multiprocessing.get_context("spawn")
                cancel_event = context.Event()
                progress_queue = context.Queue()
                try:
                    executor = ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=context,
                        initializer=initialize_hint_worker,
                        initargs=(cancel_event, progress_queue),
                    )
                except BaseException:
                    progress_queue.close()
                    progress_queue.join_thread()
                    raise
                future_to_task = {}
                processed_futures = set()
                try:
                    pending_futures = set()
                    next_task_index = 0

                    def submit_next_task() -> bool:
                        nonlocal next_task_index
                        if next_task_index >= len(tasks):
                            return False
                        task = tasks[next_task_index]
                        next_task_index += 1
                        future = executor.submit(run_hint_task, task)
                        future_to_task[future] = task
                        pending_futures.add(future)
                        return True

                    for _ in range(min(workers, len(tasks))):
                        submit_next_task()
                    while pending_futures:
                        drain_worker_events(progress_queue)
                        now = time.perf_counter()
                        if now >= next_report_at:
                            report_progress()
                            while next_report_at <= now:
                                next_report_at += progress_interval_sec
                        timeout = min(
                            HINT_EVENT_POLL_INTERVAL_SEC,
                            max(0.0, next_report_at - now),
                        )
                        done, _ = wait(
                            pending_futures,
                            timeout=timeout,
                            return_when=FIRST_COMPLETED,
                        )
                        drain_worker_events(progress_queue)
                        for future in done:
                            pending_futures.remove(future)
                            processed_futures.add(future)
                            task = future_to_task[future]
                            try:
                                result = future.result()
                            except BaseException as exc:
                                task_id = hint_task_id(task)
                                if task_id not in reported_fatal_task_ids:
                                    print(
                                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
                                        f"[hint致命的エラー] level {task.level}: "
                                        f"{type(exc).__name__}: {exc}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                raise
                            accept_task_result(task, result)
                            submit_next_task()
                    drain_worker_events(progress_queue)
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
                            accept_task_result(
                                future_to_task[future],
                                result,
                            )
                        except BaseException:
                            pass
                    drain_worker_events(progress_queue)
                    raise
                else:
                    executor.shutdown(wait=True)
                    drain_worker_events(progress_queue)
                finally:
                    progress_queue.close()
                    progress_queue.join_thread()
            if tasks:
                report_progress(final=True)
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
