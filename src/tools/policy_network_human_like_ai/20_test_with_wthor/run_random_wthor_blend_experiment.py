#!/usr/bin/env python3
"""N個のWTHOR局面でブレンド方策とConsole各レベルの着手一致率を測定する。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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


class ProgressReporter:
    def __init__(
        self,
        total_positions: int,
        report_interval_sec: float,
        phase_name: str,
        blend_params: Sequence[float],
        labels: Dict[float, str],
    ):
        self.total_positions = total_positions
        self.report_interval_sec = report_interval_sec
        self.phase_name = phase_name
        self.blend_params = tuple(blend_params)
        self.labels = labels
        self.start_time = time.perf_counter()
        self.last_report_time = self.start_time
        self.worker_states: Dict[int, dict] = {}
        self.final_reported = False
        self.lock = threading.Lock()

    def update(self, event: dict) -> None:
        with self.lock:
            self.worker_states[int(event["worker_id"])] = event
            now = time.perf_counter()
            attempted = sum(int(state["attempted_positions"]) for state in self.worker_states.values())
            completed = attempted >= self.total_positions
            if completed and self.final_reported:
                return
            if not completed and now - self.last_report_time < self.report_interval_sec:
                return
            self._print_worker_states(now, attempted)
            self.last_report_time = now
            self.final_reported = completed

    def _print_worker_states(self, now: float, attempted: int) -> None:
        elapsed = max(0.0, now - self.start_time)
        completed = attempted >= self.total_positions
        if attempted > 0:
            eta = elapsed * max(0, self.total_positions - attempted) / attempted
            eta_text = format_duration(eta)
        else:
            eta_text = "計算中"
        percent = 100.0 * attempted / self.total_positions
        print(
            f"[進捗][{self.phase_name}] {attempted:,}/{self.total_positions:,}局面 "
            f"({percent:.2f}%) 経過 {format_duration(elapsed)} ETA {eta_text}",
            file=sys.stderr,
        )

        aggregate = {
            alpha: {"positions": 0, "top1_hits": 0, "top3_hits": 0}
            for alpha in self.blend_params
        }
        for state in self.worker_states.values():
            for row in state["results"]:
                alpha = round(float(row["blend_param"]), 10)
                if alpha not in aggregate:
                    continue
                aggregate[alpha]["positions"] += int(row["positions"])
                aggregate[alpha]["top1_hits"] += int(row.get("top1_hits", 0))
                aggregate[alpha]["top3_hits"] += int(row.get("top3_hits", 0))

        for alpha in self.blend_params:
            row = aggregate[alpha]
            positions = row["positions"]
            top1 = row["top1_hits"] / positions if positions else 0.0
            top3 = row["top3_hits"] / positions if positions else 0.0
            prefix = "" if completed else "暫定"
            print(
                f"  {self.labels[alpha]}: {prefix}1位一致率 {top1 * 100.0:.3f}% "
                f"{prefix}上位3手一致率 {top3 * 100.0:.3f}% ({positions:,}局面)",
                file=sys.stderr,
            )
        sys.stderr.flush()

    def finish(self, rows: Sequence[dict], elapsed_sec: float) -> None:
        with self.lock:
            if self.final_reported:
                return
            print(
                f"[進捗][{self.phase_name}] {self.total_positions:,}/{self.total_positions:,}局面 "
                f"(100.00%) 経過 {format_duration(elapsed_sec)} ETA 00:00:00",
                file=sys.stderr,
            )
            for row in rows:
                blend_param = round(float(row["blend_param"]), 10)
                print(
                    f"  {self.labels[blend_param]}: 1位一致率 {row['top1_accuracy'] * 100.0:.3f}% "
                    f"上位3手一致率 {row['top3_accuracy'] * 100.0:.3f}% "
                    f"({row['positions']:,}局面)",
                    file=sys.stderr,
                )
            sys.stderr.flush()
            self.final_reported = True


def monitor_console_progress(
    progress_queue,
    stop_event: threading.Event,
    reporters: Dict[int, ProgressReporter],
) -> None:
    def dispatch(event: dict) -> None:
        reporters[int(event["console_level"])].update(event)

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
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.Error:
                processes = [root]
            for process in processes:
                try:
                    tree_rss += process.memory_info().rss
                except psutil.Error:
                    pass
            self.samples.append(
                {
                    "time": time.time(),
                    "cpu_percent": float(psutil.cpu_percent(interval=None)),
                    "available_memory_mib": float(memory.available / (1024.0 * 1024.0)),
                    "process_tree_memory_mib": float(tree_rss / (1024.0 * 1024.0)),
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
            }
        return {
            "available": True,
            "samples": len(self.samples),
            "minimum_available_memory_mib": min(row["available_memory_mib"] for row in self.samples),
            "maximum_process_tree_memory_mib": max(row["process_tree_memory_mib"] for row in self.samples),
            "average_cpu_percent": sum(row["cpu_percent"] for row in self.samples) / len(self.samples),
        }


def check_startup_memory(args: argparse.Namespace) -> Optional[float]:
    available = available_memory_mib()
    if available is None:
        return None
    engine_count = max(min(args.jobs, args.positions), len(CONSOLE_LEVELS))
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
        top_n="1,3",
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


def run_console_level_experiments(args: argparse.Namespace, output_dir: Path) -> List[dict]:
    console_root = output_dir / "console_levels"
    reporters = {
        level: ProgressReporter(
            args.positions,
            args.progress_interval_sec,
            f"Egaroucid level {level}",
            (0.0,),
            {0.0: f"Egaroucid level {level}"},
        )
        for level in CONSOLE_LEVELS
    }
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    progress_stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=monitor_console_progress,
        args=(progress_queue, progress_stop_event, reporters),
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
                reporters[level].finish(
                    console_row_for_progress(level_summary),
                    worker_result["elapsed_sec"],
                )
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
    return console_runs


def blend_rows_for_progress(rows: Sequence[dict]) -> List[dict]:
    return [
        {
            "blend_param": row["alpha"],
            "positions": row["positions"],
            "top1_accuracy": row["top1_accuracy"],
            "top3_accuracy": row["top3_accuracy"],
        }
        for row in rows
    ]


def console_row_for_progress(row: dict) -> List[dict]:
    return [
        {
            "blend_param": 0.0,
            "positions": row["positions"],
            "top1_accuracy": row["top1_accuracy"],
            "top3_accuracy": row["top3_accuracy"],
        }
    ]


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
        "console_level_process_isolation": {
            "separate_process_per_level": True,
            "concurrent_level_evaluation": True,
            "persistent_process_for_all_sampled_positions": True,
            "separate_hint_cache_per_level": True,
            "note": "各levelのConsoleプロセスを個別に同時起動し、担当levelの全局面が終わるまで常駐させる。",
        },
        "blend_jobs": args.jobs,
        "console_level_processes": len(CONSOLE_LEVELS),
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
    print("単体で評価するEgaroucidレベル", ", ".join(str(level) for level in CONSOLE_LEVELS))
    print("Console起動方式", "level別の10プロセスを同時起動し、各プロセスを全N局面で常駐させる")
    print("ブレンド方策の同時評価処理数", args.jobs)
    print("Console level別の同時常駐プロセス数", len(CONSOLE_LEVELS))
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
    print("総実行時間", f"{elapsed_sec:.3f}秒")
    if resources.get("available"):
        print("実行中の最小空き物理メモリ", f"{resources['minimum_available_memory_mib']:.0f} MiB")
        print("関連プロセスの最大使用メモリ", f"{resources['maximum_process_tree_memory_mib']:.0f} MiB")
        print("実行中の平均CPU使用率", f"{resources['average_cpu_percent']:.1f}%")
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
        blend_progress = ProgressReporter(
            args.positions,
            args.progress_interval_sec,
            "ブレンド方策",
            BLEND_PARAMS,
            {alpha: f"alpha={alpha:.1f}" for alpha in BLEND_PARAMS},
        )
        blend_start_time = time.perf_counter()
        blend_result = evaluator.evaluate(
            blend_evaluator_args,
            progress_callback=blend_progress.update,
            progress_interval_sec=min(10.0, args.progress_interval_sec),
        )
        blend_elapsed_sec = time.perf_counter() - blend_start_time
        blend_rows = make_blend_summary_rows(blend_result)
        blend_progress.finish(blend_rows_for_progress(blend_rows), blend_elapsed_sec)

        console_levels_start_time = time.perf_counter()
        console_runs = run_console_level_experiments(args, output_dir)
        console_levels_elapsed_sec = time.perf_counter() - console_levels_start_time
    finally:
        elapsed_sec = time.perf_counter() - experiment_start_time
        resources = monitor.stop()

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
    )
    print_results(
        blend_rows,
        console_runs,
        console_levels_elapsed_sec,
        elapsed_sec,
        resources,
        output_dir,
    )


if __name__ == "__main__":
    main()
