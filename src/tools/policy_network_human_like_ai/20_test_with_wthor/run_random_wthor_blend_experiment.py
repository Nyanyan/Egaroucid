#!/usr/bin/env python3
"""N個のWTHOR局面でブレンド方策とConsole各レベルの着手一致率を測定する。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence

import evaluate_wthor_blend_human_match as evaluator


SCRIPT_DIR = Path(__file__).resolve().parent
BLEND_PARAMS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CONSOLE_LEVELS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
TOP_N_VALUES = (1, 3)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SUMMARY_FIELDS = (
    "alpha",
    "positions",
    "top1_hits",
    "top1_accuracy",
    "top1_ci95_lower",
    "top1_ci95_upper",
    "top3_hits",
    "top3_accuracy",
)
CONSOLE_SUMMARY_FIELDS = (
    "level",
    "positions",
    "top1_hits",
    "top1_accuracy",
    "top1_ci95_lower",
    "top1_ci95_upper",
    "top3_hits",
    "top3_accuracy",
    "elapsed_sec",
)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}日 {text}" if days else text


class CombinedProgressReporter:
    def __init__(
        self,
        total_positions: int,
        report_interval_sec: float,
        blend_level: int,
        blend_params: Sequence[float],
        console_levels: Sequence[int],
    ):
        self.total_positions = total_positions
        self.report_interval_sec = report_interval_sec
        self.blend_level = blend_level
        self.blend_params = tuple(blend_params)
        self.console_levels = tuple(console_levels)
        self.start_time = time.perf_counter()
        self.blend_worker_states: Dict[int, dict] = {}
        self.console_level_states: Dict[int, dict] = {}
        self.blend_rows: Optional[List[dict]] = None
        self.console_rows: Dict[int, dict] = {}
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self) -> None:
        with self.lock:
            self._print(time.perf_counter())
        self.thread = threading.Thread(
            target=self._run,
            name="combined-progress-reporter",
            daemon=True,
        )
        self.thread.start()

    def update_blend(self, event: dict) -> None:
        with self.lock:
            self.blend_worker_states[int(event["worker_id"])] = event

    def update_console(self, event: dict) -> None:
        with self.lock:
            self.console_level_states[int(event["console_level"])] = event

    def complete_blend(self, rows: Sequence[dict]) -> None:
        with self.lock:
            self.blend_rows = list(rows)

    def complete_console_level(self, level: int, row: dict) -> None:
        with self.lock:
            self.console_rows[level] = row

    def finish(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.report_interval_sec + 1.0))
        with self.lock:
            self._print(time.perf_counter())

    def _run(self) -> None:
        while not self.stop_event.wait(self.report_interval_sec):
            with self.lock:
                self._print(time.perf_counter())

    @staticmethod
    def _accuracy(hits: int, positions: int) -> float:
        return hits / positions if positions else 0.0

    def _blend_snapshot(self) -> tuple[int, Dict[float, dict], bool]:
        if self.blend_rows is not None:
            rows = {
                round(float(row["alpha"]), 10): {
                    "positions": int(row["positions"]),
                    "top1_accuracy": float(row["top1_accuracy"]),
                    "top3_accuracy": float(row["top3_accuracy"]),
                }
                for row in self.blend_rows
            }
            return self.total_positions, rows, True
        aggregate = {
            alpha: {"positions": 0, "top1_hits": 0, "top3_hits": 0}
            for alpha in self.blend_params
        }
        attempted = 0
        for state in self.blend_worker_states.values():
            attempted += int(state["attempted_positions"])
            for row in state["results"]:
                alpha = round(float(row["blend_param"]), 10)
                if alpha not in aggregate:
                    continue
                aggregate[alpha]["positions"] += int(row["positions"])
                aggregate[alpha]["top1_hits"] += int(row.get("top1_hits", 0))
                aggregate[alpha]["top3_hits"] += int(row.get("top3_hits", 0))
        rows = {
            alpha: {
                "positions": row["positions"],
                "top1_accuracy": self._accuracy(row["top1_hits"], row["positions"]),
                "top3_accuracy": self._accuracy(row["top3_hits"], row["positions"]),
            }
            for alpha, row in aggregate.items()
        }
        return attempted, rows, False

    def _console_snapshot(self) -> tuple[int, Dict[int, dict]]:
        attempted_total = 0
        rows = {}
        for level in self.console_levels:
            completed = self.console_rows.get(level)
            if completed is not None:
                attempted_total += self.total_positions
                rows[level] = {
                    "status": "完了",
                    "positions": int(completed["positions"]),
                    "attempted": self.total_positions,
                    "top1_accuracy": float(completed["top1_accuracy"]),
                    "top3_accuracy": float(completed["top3_accuracy"]),
                }
                continue
            state = self.console_level_states.get(level)
            if state is None:
                rows[level] = {
                    "status": "待機",
                    "positions": 0,
                    "attempted": 0,
                    "top1_accuracy": 0.0,
                    "top3_accuracy": 0.0,
                }
                continue
            attempted = int(state["attempted_positions"])
            attempted_total += attempted
            result = next(
                (
                    row
                    for row in state["results"]
                    if round(float(row["blend_param"]), 10) == 0.0
                ),
                None,
            )
            positions = int(result["positions"]) if result is not None else 0
            rows[level] = {
                "status": "暫定",
                "positions": positions,
                "attempted": attempted,
                "top1_accuracy": self._accuracy(
                    int(result.get("top1_hits", 0)) if result is not None else 0,
                    positions,
                ),
                "top3_accuracy": self._accuracy(
                    int(result.get("top3_hits", 0)) if result is not None else 0,
                    positions,
                ),
            }
        return attempted_total, rows

    def _eta_text(self, elapsed: float, attempted: int, total: int) -> str:
        if attempted <= 0:
            return "計算中"
        eta = elapsed * max(0, total - attempted) / attempted
        return format_duration(eta)

    def _print(self, now: float) -> None:
        elapsed = max(0.0, now - self.start_time)
        blend_attempted, blend_rows, blend_completed = self._blend_snapshot()
        console_attempted, console_rows = self._console_snapshot()
        overall_attempted = blend_attempted + console_attempted
        overall_total = self.total_positions * (1 + len(self.console_levels))
        print(
            f"[進捗][ブレンド・Console統合] "
            f"{overall_attempted:,}/{overall_total:,}件の局面評価 "
            f"({overall_attempted * 100.0 / overall_total:.2f}%) "
            f"経過 {format_duration(elapsed)} "
            f"全体ETA {self._eta_text(elapsed, overall_attempted, overall_total)}",
            file=sys.stderr,
        )
        blend_status = "完了" if blend_completed else "暫定"
        print(
            f"  ブレンド方策 (Egaroucid level {self.blend_level}): "
            f"{blend_attempted:,}/{self.total_positions:,}局面 "
            f"ETA {self._eta_text(elapsed, blend_attempted, self.total_positions)}",
            file=sys.stderr,
        )
        for alpha in self.blend_params:
            row = blend_rows[alpha]
            print(
                f"    alpha={alpha:.1f}: 1位 "
                f"{row['top1_accuracy'] * 100.0:.3f}% "
                f"上位3手 {row['top3_accuracy'] * 100.0:.3f}% "
                f"({row['positions']:,}局面、状態: {blend_status})",
                file=sys.stderr,
            )
        console_total = self.total_positions * len(self.console_levels)
        print(
            f"  Console単体: {len(self.console_rows)}/{len(self.console_levels)}レベル完了、"
            f"{console_attempted:,}/{console_total:,}件の局面評価 "
            f"ETA {self._eta_text(elapsed, console_attempted, console_total)}",
            file=sys.stderr,
        )
        for level in self.console_levels:
            row = console_rows[level]
            if row["status"] == "待機":
                print(
                    f"    level {level:>2}: 初回結果待ち "
                    f"(0/{self.total_positions:,}局面)",
                    file=sys.stderr,
                )
                continue
            print(
                f"    level {level:>2}: 1位 "
                f"{row['top1_accuracy'] * 100.0:.3f}% "
                f"上位3手 {row['top3_accuracy'] * 100.0:.3f}% "
                f"({row['attempted']:,}/{self.total_positions:,}局面、"
                f"状態: {row['status']})",
                file=sys.stderr,
            )
        sys.stderr.flush()


def monitor_console_progress(
    progress_queue,
    stop_event: threading.Event,
    reporter: CombinedProgressReporter,
) -> None:
    def dispatch(event: dict) -> None:
        reporter.update_console(event)

    while not stop_event.is_set():
        try:
            dispatch(progress_queue.get(timeout=0.2))
        except queue.Empty:
            pass
    while True:
        try:
            dispatch(progress_queue.get_nowait())
        except queue.Empty:
            return


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("N must be a positive integer")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("the value must be positive")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("0以上の値を指定してください")
    return value


def wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    rate = hits / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((rate * (1.0 - rate) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def default_output_dir(args: argparse.Namespace) -> Path:
    return (
        SCRIPT_DIR
        / "output"
        / f"random_wthor_{args.data_split}_n{args.positions}_seed{args.sample_seed}_level{args.egaroucid_level}"
    )


def split_position_count(total: int, data_split: str) -> int:
    if data_split == "all":
        return total
    n_train, n_val, n_test = evaluator.split_counts(total, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    return {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }[data_split]


def available_memory_mib() -> Optional[float]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    return float(psutil.virtual_memory().available / (1024.0 * 1024.0))


def default_max_concurrent_hints(egaroucid_threads: int) -> int:
    logical_cpus = os.cpu_count() or 1
    return max(1, logical_cpus // egaroucid_threads)


class ResourceMonitor:
    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.samples: List[dict] = []
        self.enabled = False

    def start(self) -> None:
        try:
            import psutil  # type: ignore
        except ImportError:
            return
        self.enabled = True
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        import psutil  # type: ignore

        root = psutil.Process(os.getpid())
        while not self.stop_event.wait(self.interval_sec):
            memory = psutil.virtual_memory()
            tree_rss = 0
            egaroucid_processes = 0
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.Error:
                processes = [root]
            for process in processes:
                try:
                    tree_rss += process.memory_info().rss
                    if "egaroucid" in process.name().lower():
                        egaroucid_processes += 1
                except psutil.Error:
                    pass
            self.samples.append(
                {
                    "time": time.time(),
                    "cpu_percent": float(psutil.cpu_percent(interval=None)),
                    "available_memory_mib": float(memory.available / (1024.0 * 1024.0)),
                    "process_tree_memory_mib": float(tree_rss / (1024.0 * 1024.0)),
                    "egaroucid_processes": egaroucid_processes,
                }
            )

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.interval_sec * 2.0))
        if not self.samples:
            return {
                "available": False,
                "samples": 0,
                "minimum_available_memory_mib": None,
                "maximum_process_tree_memory_mib": None,
                "average_cpu_percent": None,
                "maximum_egaroucid_processes": None,
            }
        return {
            "available": True,
            "samples": len(self.samples),
            "minimum_available_memory_mib": min(row["available_memory_mib"] for row in self.samples),
            "maximum_process_tree_memory_mib": max(row["process_tree_memory_mib"] for row in self.samples),
            "average_cpu_percent": sum(row["cpu_percent"] for row in self.samples) / len(self.samples),
            "maximum_egaroucid_processes": max(
                row["egaroucid_processes"] for row in self.samples
            ),
        }


def check_startup_memory(args: argparse.Namespace) -> Optional[float]:
    available = available_memory_mib()
    if available is None:
        return None
    blend_engine_count = min(args.jobs, args.positions)
    engine_count = blend_engine_count + len(CONSOLE_LEVELS)
    required = args.minimum_remaining_memory_mib + engine_count * args.estimated_engine_memory_mib
    if available < required:
        raise RuntimeError(
            "空き物理メモリが不足しています: "
            f"現在は{available:.0f} MiBですが、少なくとも{required:.0f} MiB必要です "
            f"(実行中に残すメモリ{args.minimum_remaining_memory_mib:.0f} MiB + "
            f"同時起動する最大{engine_count}個のConsoleプロセス x "
            f"1プロセス当たりの推定使用量{args.estimated_engine_memory_mib:.0f} MiB)"
        )
    return available


def make_evaluator_args(
    args: argparse.Namespace,
    output_dir: Path,
    egaroucid_level: int,
    blend_params: Sequence[float],
    hint_cache_db: Path,
    jobs: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        board_data_dir=args.board_data_dir,
        weights=args.weights,
        egaroucid_exe=args.egaroucid_exe,
        egaroucid_level=egaroucid_level,
        egaroucid_threads=args.egaroucid_threads,
        egaroucid_timeout_sec=args.egaroucid_timeout_sec,
        score_temperature=args.score_temperature,
        no_legal_mask_policy=False,
        blend_params=",".join(f"{alpha:.1f}" for alpha in blend_params),
        top_n=",".join(str(n) for n in TOP_N_VALUES),
        data_split=args.data_split,
        split_seed=args.split_seed,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        batch_size=1,
        max_positions=None,
        range_start=0,
        range_end=None,
        sample_positions=args.positions,
        sample_seed=args.sample_seed,
        jobs=jobs,
        hint_cache_db=hint_cache_db,
        raw_hint_samples=0,
        output_dir=output_dir,
        progress_interval=1000,
        verbose=args.verbose,
        hint_command_stagger_sec=args.hint_command_stagger_sec,
        hint_command_stagger_lock=args.hint_command_stagger_lock,
        hint_command_stagger_state=args.hint_command_stagger_state,
        hint_command_semaphore=args.hint_command_semaphore,
        max_concurrent_hints=args.max_concurrent_hints,
    )


def evaluate_console_level_worker(worker_args: dict) -> dict:
    level = int(worker_args["level"])
    progress_queue = worker_args["progress_queue"]

    def relay_progress(event: dict) -> None:
        forwarded = dict(event)
        forwarded["console_level"] = level
        try:
            progress_queue.put(forwarded)
        except (BrokenPipeError, EOFError, OSError):
            pass

    start_time = time.perf_counter()
    result = evaluator.evaluate(
        worker_args["evaluator_args"],
        progress_callback=relay_progress,
        progress_interval_sec=worker_args["progress_interval_sec"],
    )
    return {
        "level": level,
        "output_dir": worker_args["output_dir"],
        "result": result,
        "elapsed_sec": time.perf_counter() - start_time,
    }


def make_blend_summary_rows(result: dict) -> List[dict]:
    topn: Dict[tuple[float, int], dict] = {
        (round(float(row["blend_param"]), 10), int(row["top_n"])): row
        for row in result["topn"]
    }
    rows = []
    for alpha in BLEND_PARAMS:
        top1 = topn[(round(alpha, 10), 1)]
        top3 = topn[(round(alpha, 10), 3)]
        lower, upper = wilson_interval(int(top1["exact_hits"]), int(top1["positions"]))
        rows.append(
            {
                "alpha": alpha,
                "positions": int(top1["positions"]),
                "top1_hits": int(top1["exact_hits"]),
                "top1_accuracy": float(top1["exact_accuracy"]),
                "top1_ci95_lower": lower,
                "top1_ci95_upper": upper,
                "top3_hits": int(top3["exact_hits"]),
                "top3_accuracy": float(top3["exact_accuracy"]),
            }
        )
    return rows


def make_console_level_summary_row(level: int, result: dict, elapsed_sec: float) -> dict:
    topn = {
        int(row["top_n"]): row
        for row in result["topn"]
        if round(float(row["blend_param"]), 10) == 0.0
    }
    top1 = topn[1]
    top3 = topn[3]
    lower, upper = wilson_interval(int(top1["exact_hits"]), int(top1["positions"]))
    return {
        "level": level,
        "positions": int(top1["positions"]),
        "top1_hits": int(top1["exact_hits"]),
        "top1_accuracy": float(top1["exact_accuracy"]),
        "top1_ci95_lower": lower,
        "top1_ci95_upper": upper,
        "top3_hits": int(top3["exact_hits"]),
        "top3_accuracy": float(top3["exact_accuracy"]),
        "elapsed_sec": elapsed_sec,
    }


def run_console_level_experiments(
    args: argparse.Namespace,
    output_dir: Path,
    progress: CombinedProgressReporter,
) -> tuple[List[dict], dict]:
    started_at_unix = time.time()
    start_time = time.perf_counter()
    console_root = output_dir / "console_levels"
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    progress_stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=monitor_console_progress,
        args=(progress_queue, progress_stop_event, progress),
        name="console-level-progress-monitor",
        daemon=True,
    )
    progress_thread.start()

    worker_args = []
    for level in CONSOLE_LEVELS:
        level_output_dir = console_root / f"level_{level}"
        level_hint_cache = level_output_dir / f"hint_score_cache_level{level}.sqlite3"
        worker_args.append(
            {
                "level": level,
                "output_dir": level_output_dir,
                "progress_queue": progress_queue,
                "progress_interval_sec": min(10.0, args.progress_interval_sec),
                "evaluator_args": make_evaluator_args(
                    args,
                    level_output_dir,
                    level,
                    (0.0,),
                    level_hint_cache,
                    1,
                ),
            }
        )

    console_runs = []
    try:
        with ProcessPoolExecutor(max_workers=len(CONSOLE_LEVELS)) as executor:
            futures = [executor.submit(evaluate_console_level_worker, item) for item in worker_args]
            for future in as_completed(futures):
                worker_result = future.result()
                level = int(worker_result["level"])
                level_summary = make_console_level_summary_row(
                    level,
                    worker_result["result"],
                    worker_result["elapsed_sec"],
                )
                progress.complete_console_level(level, level_summary)
                console_runs.append(
                    {
                        "output_dir": Path(worker_result["output_dir"]),
                        "result": worker_result["result"],
                        "summary": level_summary,
                    }
                )
    finally:
        progress_stop_event.set()
        progress_thread.join()
        manager.shutdown()

    console_runs.sort(key=lambda run: run["summary"]["level"])
    return console_runs, {
        "started_at_unix": started_at_unix,
        "finished_at_unix": time.time(),
        "elapsed_sec": time.perf_counter() - start_time,
    }


def run_blend_experiment(
    evaluator_args: argparse.Namespace,
    progress: CombinedProgressReporter,
    progress_interval_sec: float,
) -> tuple[dict, List[dict], dict]:
    started_at_unix = time.time()
    start_time = time.perf_counter()
    result = evaluator.evaluate(
        evaluator_args,
        progress_callback=progress.update_blend,
        progress_interval_sec=min(10.0, progress_interval_sec),
    )
    elapsed_sec = time.perf_counter() - start_time
    rows = make_blend_summary_rows(result)
    progress.complete_blend(rows)
    return result, rows, {
        "started_at_unix": started_at_unix,
        "finished_at_unix": time.time(),
        "elapsed_sec": elapsed_sec,
    }


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    blend_result: dict,
    blend_rows: Sequence[dict],
    console_runs: Sequence[dict],
    blend_elapsed_sec: float,
    console_levels_elapsed_sec: float,
    elapsed_sec: float,
    startup_available_memory_mib: Optional[float],
    resources: dict,
    command_stagger_stats: dict,
    concurrent_phase_timing: dict,
) -> None:
    csv_path = output_dir / "random_wthor_blend_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(blend_rows)

    console_rows = [run["summary"] for run in console_runs]
    console_csv_path = output_dir / "random_wthor_console_level_summary.csv"
    with console_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONSOLE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(console_rows)

    summary = {
        "requested_positions": args.positions,
        "evaluated_positions": blend_rows[0]["positions"] if blend_rows else 0,
        "data_split": args.data_split,
        "split_positions": blend_result["split_positions"],
        "split_seed": args.split_seed,
        "sample_seed": args.sample_seed,
        "blend_params": list(BLEND_PARAMS),
        "blend_egaroucid_level": args.egaroucid_level,
        "console_levels": list(CONSOLE_LEVELS),
        "console_hint_count": max(TOP_N_VALUES),
        "console_level_process_isolation": {
            "separate_process_per_level": True,
            "concurrent_level_evaluation": True,
            "concurrent_with_blend_evaluation": True,
            "persistent_process_for_all_sampled_positions": True,
            "separate_hint_cache_per_level": True,
            "note": "ブレンド版とConsole単体版を同時実行し、各Consoleプロセスは担当する全局面の完了まで常駐させる。",
        },
        "hint_command_stagger": command_stagger_stats,
        "concurrent_phase_timing": concurrent_phase_timing,
        "blend_jobs": args.jobs,
        "console_level_processes": len(CONSOLE_LEVELS),
        "logical_cpu_threads": os.cpu_count(),
        "maximum_concurrent_console_processes": (
            min(args.jobs, args.positions) + len(CONSOLE_LEVELS)
        ),
        "egaroucid_threads_per_process": args.egaroucid_threads,
        "progress_interval_sec": args.progress_interval_sec,
        "elapsed_sec": elapsed_sec,
        "blend_elapsed_sec": blend_elapsed_sec,
        "console_levels_elapsed_sec": console_levels_elapsed_sec,
        "startup_available_memory_mib": startup_available_memory_mib,
        "resources": resources,
        "hint_cache_stats": blend_result["hint_cache_stats"],
        "invalid_policy_samples": blend_result["invalid_policy_samples"],
        "illegal_label_samples": blend_result["illegal_label_samples"],
        "results": list(blend_rows),
        "console_level_results": console_rows,
        "console_level_runs": [
            {
                "level": run["summary"]["level"],
                "hint_cache_stats": run["result"]["hint_cache_stats"],
                "invalid_policy_samples": run["result"]["invalid_policy_samples"],
                "illegal_label_samples": run["result"]["illegal_label_samples"],
                "detailed_result": str(run["output_dir"] / "wthor_blend_human_match.json"),
            }
            for run in console_runs
        ],
        "detailed_result": str(output_dir / "wthor_blend_human_match.json"),
    }
    with (output_dir / "random_wthor_blend_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_configuration(
    args: argparse.Namespace,
    output_dir: Path,
    available_positions: int,
    split_positions: int,
    startup_available_memory_mib: Optional[float],
) -> None:
    print("WTHORブレンド方策・無作為抽出実験")
    print("抽出元", args.data_split)
    print("抽出元の局面数", f"{split_positions:,}")
    print("無作為抽出する局面数", f"{args.positions:,}")
    print("データ分割seed", args.split_seed)
    print("無作為抽出seed", args.sample_seed)
    print("alpha", ", ".join(f"{alpha:.1f}" for alpha in BLEND_PARAMS))
    print("ブレンド方策で使うEgaroucidレベル", args.egaroucid_level)
    print("単体で評価するEgaroucidレベル", ", ".join(str(level) for level in CONSOLE_LEVELS))
    print("Consoleへ要求する候補手数", max(TOP_N_VALUES))
    print("評価の実行方式", "ブレンド版とConsole単体版を同時実行する")
    print("Console起動方式", "全プロセスを常駐させ、hintコマンドの送信時刻を共通管理する")
    print("hintコマンドの送信間隔", f"{args.hint_command_stagger_sec:g}秒")
    print("論理CPUスレッド数", os.cpu_count() or 1)
    print("同時にhint探索を行うConsoleプロセスの上限", args.max_concurrent_hints)
    print("ブレンド方策の同時評価処理数", args.jobs)
    print("Console level別の同時常駐プロセス数", len(CONSOLE_LEVELS))
    print(
        "同時に常駐する最大Consoleプロセス数",
        min(args.jobs, args.positions) + len(CONSOLE_LEVELS),
    )
    print("1 Consoleプロセス当たりのEgaroucidスレッド数", args.egaroucid_threads)
    print("進捗表示間隔", f"{args.progress_interval_sec:g}秒")
    if args.hint_cache_db is not None:
        print("共有hintキャッシュ", args.hint_cache_db)
    print("WTHOR全局面数", f"{available_positions:,}")
    if startup_available_memory_mib is not None:
        print("実行開始前の空き物理メモリ", f"{startup_available_memory_mib:.0f} MiB")
    print("出力先", output_dir)


def print_results(
    blend_rows: Sequence[dict],
    console_runs: Sequence[dict],
    console_levels_elapsed_sec: float,
    elapsed_sec: float,
    resources: dict,
    command_stagger_stats: dict,
    concurrent_phase_timing: dict,
    output_dir: Path,
) -> None:
    print()
    print("ブレンド方策の結果")
    print("alpha  1位一致率 (95%信頼区間)       上位3手一致率")
    for row in blend_rows:
        print(
            f"{row['alpha']:>4.1f}   "
            f"{row['top1_accuracy'] * 100.0:>7.3f}% "
            f"({row['top1_ci95_lower'] * 100.0:.3f}% - {row['top1_ci95_upper'] * 100.0:.3f}%)   "
            f"{row['top3_accuracy'] * 100.0:>7.3f}%"
        )
    print()
    print("Egaroucid for Console単体の結果")
    print("level  1位一致率 (95%信頼区間)       上位3手一致率    実行時間")
    for run in console_runs:
        row = run["summary"]
        print(
            f"{row['level']:>5}   "
            f"{row['top1_accuracy'] * 100.0:>7.3f}% "
            f"({row['top1_ci95_lower'] * 100.0:.3f}% - {row['top1_ci95_upper'] * 100.0:.3f}%)   "
            f"{row['top3_accuracy'] * 100.0:>7.3f}%   "
            f"{row['elapsed_sec']:.3f}秒"
        )
    print("Console level別評価の並列実行時間", f"{console_levels_elapsed_sec:.3f}秒")
    print(
        "ブレンド版とConsole単体版の重複実行時間",
        f"{concurrent_phase_timing['overlap_sec']:.3f}秒",
    )
    print("総実行時間", f"{elapsed_sec:.3f}秒")
    print("hintコマンド送信回数", f"{command_stagger_stats['command_count']:,}")
    print(
        "同時にhint探索を行ったConsoleプロセスの最大数",
        command_stagger_stats["maximum_simultaneous_hints_observed"],
    )
    if command_stagger_stats["minimum_observed_spacing_sec"] is not None:
        print(
            "実測したhintコマンドの最小送信間隔",
            f"{command_stagger_stats['minimum_observed_spacing_sec']:.6f}秒",
        )
    if resources.get("available"):
        print("実行中の最小空き物理メモリ", f"{resources['minimum_available_memory_mib']:.0f} MiB")
        print("関連プロセスの最大使用メモリ", f"{resources['maximum_process_tree_memory_mib']:.0f} MiB")
        print("実行中の平均CPU使用率", f"{resources['average_cpu_percent']:.1f}%")
        print(
            "同時に存在したConsoleプロセスの最大数",
            resources["maximum_egaroucid_processes"],
        )
    print("集計CSV", output_dir / "random_wthor_blend_summary.csv")
    print("Console level別集計CSV", output_dir / "random_wthor_console_level_summary.csv")
    print("集計JSON", output_dir / "random_wthor_blend_summary.json")


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WTHORからN局面をseed付きで無作為抽出し、"
            "alpha=0.0から1.0まで0.2刻みのブレンド方策と、"
            "Egaroucid level 1から19までの奇数levelの人間着手一致率をまとめて測定します。"
        )
    )
    parser.add_argument("positions", metavar="N", type=positive_int, help="無作為抽出する局面数")
    parser.add_argument("--data-split", choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=613)
    parser.add_argument("--sample-seed", type=int, default=613)
    parser.add_argument("--board-data-dir", type=Path, default=evaluator.default_board_data_dir())
    parser.add_argument("--weights", type=Path, default=evaluator.default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=evaluator.default_egaroucid_exe())
    parser.add_argument("--egaroucid-level", type=int, default=21)
    parser.add_argument("--jobs", type=positive_int, default=4)
    parser.add_argument("--egaroucid-threads", type=positive_int, default=8)
    parser.add_argument("--egaroucid-timeout-sec", type=positive_float, default=1800.0)
    parser.add_argument("--score-temperature", type=positive_float, default=1.0)
    parser.add_argument(
        "--hint-command-stagger-sec",
        type=nonnegative_float,
        default=0.1,
        help="全Consoleプロセスへhintコマンドを送る開始時刻の最小間隔（秒）",
    )
    parser.add_argument(
        "--max-concurrent-hints",
        type=positive_int,
        default=None,
        help="同時にhint探索を行うConsoleプロセス数。省略時は論理CPUスレッド数から決定する",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=positive_float,
        default=60.0,
        help="暫定一致率とETAを標準エラーへ表示する間隔（秒）",
    )
    parser.add_argument("--minimum-remaining-memory-mib", type=positive_float, default=16384.0)
    parser.add_argument("--estimated-engine-memory-mib", type=positive_float, default=1400.0)
    parser.add_argument(
        "--hint-cache-db",
        type=Path,
        default=None,
        help="ブレンド方策用の既存hintキャッシュを共有する場合のSQLiteファイル",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="設定だけを表示し、評価を実行しない")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    if args.max_concurrent_hints is None:
        args.max_concurrent_hints = default_max_concurrent_hints(args.egaroucid_threads)
    dat_files = evaluator.discover_dat_files(args.board_data_dir)
    available_positions = evaluator.count_position_samples(dat_files, None)
    selected_split_positions = split_position_count(available_positions, args.data_split)
    if args.positions > selected_split_positions:
        raise ValueError(
            f"N={args.positions} exceeds the {args.data_split} split size "
            f"of {selected_split_positions}"
        )

    output_dir = args.output_dir or default_output_dir(args)
    startup_available_memory_mib = available_memory_mib()
    print_configuration(
        args,
        output_dir,
        available_positions,
        selected_split_positions,
        startup_available_memory_mib,
    )
    if args.dry_run:
        return

    startup_available_memory_mib = check_startup_memory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    command_manager = multiprocessing.Manager()
    args.hint_command_stagger_lock = command_manager.Lock()
    args.hint_command_semaphore = command_manager.BoundedSemaphore(
        args.max_concurrent_hints
    )
    args.hint_command_stagger_state = command_manager.Namespace()
    args.hint_command_stagger_state.next_time = time.time()
    args.hint_command_stagger_state.last_time = 0.0
    args.hint_command_stagger_state.minimum_spacing = float("inf")
    args.hint_command_stagger_state.count = 0
    args.hint_command_stagger_state.active_count = 0
    args.hint_command_stagger_state.maximum_active_count = 0
    blend_hint_cache = args.hint_cache_db
    if blend_hint_cache is None:
        blend_hint_cache = output_dir / f"hint_score_cache_level{args.egaroucid_level}.sqlite3"
    blend_evaluator_args = make_evaluator_args(
        args,
        output_dir,
        args.egaroucid_level,
        BLEND_PARAMS,
        blend_hint_cache,
        args.jobs,
    )

    monitor = ResourceMonitor()
    monitor.start()
    experiment_start_time = time.perf_counter()
    try:
        progress = CombinedProgressReporter(
            args.positions,
            args.progress_interval_sec,
            args.egaroucid_level,
            BLEND_PARAMS,
            CONSOLE_LEVELS,
        )
        progress.start()
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        run_blend_experiment,
                        blend_evaluator_args,
                        progress,
                        args.progress_interval_sec,
                    ): "blend",
                    executor.submit(
                        run_console_level_experiments,
                        args,
                        output_dir,
                        progress,
                    ): "console",
                }
                concurrent_results = {}
                for future in as_completed(futures):
                    concurrent_results[futures[future]] = future.result()
        finally:
            progress.finish()
        blend_result, blend_rows, blend_timing = concurrent_results["blend"]
        console_runs, console_timing = concurrent_results["console"]
        blend_elapsed_sec = float(blend_timing["elapsed_sec"])
        console_levels_elapsed_sec = float(console_timing["elapsed_sec"])
        concurrent_phase_timing = {
            "blend": blend_timing,
            "console_levels": console_timing,
            "start_time_difference_sec": abs(
                float(blend_timing["started_at_unix"])
                - float(console_timing["started_at_unix"])
            ),
            "overlap_sec": max(
                0.0,
                min(
                    float(blend_timing["finished_at_unix"]),
                    float(console_timing["finished_at_unix"]),
                )
                - max(
                    float(blend_timing["started_at_unix"]),
                    float(console_timing["started_at_unix"]),
                ),
            ),
        }
    finally:
        elapsed_sec = time.perf_counter() - experiment_start_time
        resources = monitor.stop()
        command_count = int(args.hint_command_stagger_state.count)
        minimum_spacing = float(args.hint_command_stagger_state.minimum_spacing)
        command_stagger_stats = {
            "configured_spacing_sec": args.hint_command_stagger_sec,
            "configured_maximum_concurrent_hints": args.max_concurrent_hints,
            "command_count": command_count,
            "maximum_simultaneous_hints_observed": int(
                args.hint_command_stagger_state.maximum_active_count
            ),
            "minimum_observed_spacing_sec": (
                minimum_spacing if command_count >= 2 and math.isfinite(minimum_spacing) else None
            ),
        }
        command_manager.shutdown()

    write_summary(
        output_dir,
        args,
        blend_result,
        blend_rows,
        console_runs,
        blend_elapsed_sec,
        console_levels_elapsed_sec,
        elapsed_sec,
        startup_available_memory_mib,
        resources,
        command_stagger_stats,
        concurrent_phase_timing,
    )
    print_results(
        blend_rows,
        console_runs,
        console_levels_elapsed_sec,
        elapsed_sec,
        resources,
        command_stagger_stats,
        concurrent_phase_timing,
        output_dir,
    )


if __name__ == "__main__":
    main()
