#!/usr/bin/env python3
"""Evaluate six policy-blend alpha values on N random WTHOR positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence

import evaluate_wthor_blend_human_match as evaluator


SCRIPT_DIR = Path(__file__).resolve().parent
BLEND_PARAMS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}日 {text}" if days else text


class ProgressReporter:
    def __init__(self, total_positions: int, report_interval_sec: float):
        self.total_positions = total_positions
        self.report_interval_sec = report_interval_sec
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
            f"[進捗] {attempted:,}/{self.total_positions:,}局面 "
            f"({percent:.2f}%) 経過 {format_duration(elapsed)} ETA {eta_text}",
            file=sys.stderr,
        )

        aggregate = {
            alpha: {"positions": 0, "top1_hits": 0, "top3_hits": 0}
            for alpha in BLEND_PARAMS
        }
        for state in self.worker_states.values():
            for row in state["results"]:
                alpha = round(float(row["blend_param"]), 10)
                if alpha not in aggregate:
                    continue
                aggregate[alpha]["positions"] += int(row["positions"])
                aggregate[alpha]["top1_hits"] += int(row.get("top1_hits", 0))
                aggregate[alpha]["top3_hits"] += int(row.get("top3_hits", 0))

        for alpha in BLEND_PARAMS:
            row = aggregate[alpha]
            positions = row["positions"]
            top1 = row["top1_hits"] / positions if positions else 0.0
            top3 = row["top3_hits"] / positions if positions else 0.0
            prefix = "" if completed else "暫定"
            print(
                f"  alpha={alpha:.1f}: {prefix}1位一致率 {top1 * 100.0:.3f}% "
                f"{prefix}上位3手一致率 {top3 * 100.0:.3f}% ({positions:,}局面)",
                file=sys.stderr,
            )
        sys.stderr.flush()

    def finish(self, rows: Sequence[dict], elapsed_sec: float) -> None:
        with self.lock:
            if self.final_reported:
                return
            print(
                f"[進捗] {self.total_positions:,}/{self.total_positions:,}局面 "
                f"(100.00%) 経過 {format_duration(elapsed_sec)} ETA 00:00:00",
                file=sys.stderr,
            )
            for row in rows:
                print(
                    f"  alpha={row['alpha']:.1f}: 1位一致率 {row['top1_accuracy'] * 100.0:.3f}% "
                    f"上位3手一致率 {row['top3_accuracy'] * 100.0:.3f}% "
                    f"({row['positions']:,}局面)",
                    file=sys.stderr,
                )
            sys.stderr.flush()
            self.final_reported = True


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
    engine_count = min(args.jobs, args.positions)
    required = args.minimum_remaining_memory_mib + engine_count * args.estimated_engine_memory_mib
    if available < required:
        raise RuntimeError(
            "insufficient available memory: "
            f"{available:.0f} MiB available, but at least {required:.0f} MiB is required "
            f"({args.minimum_remaining_memory_mib:.0f} MiB reserve plus "
            f"{engine_count} engines x {args.estimated_engine_memory_mib:.0f} MiB)"
        )
    return available


def make_evaluator_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    hint_cache_db = args.hint_cache_db
    if hint_cache_db is None:
        hint_cache_db = output_dir / f"hint_score_cache_level{args.egaroucid_level}.sqlite3"
    return argparse.Namespace(
        board_data_dir=args.board_data_dir,
        weights=args.weights,
        egaroucid_exe=args.egaroucid_exe,
        egaroucid_level=args.egaroucid_level,
        egaroucid_threads=args.egaroucid_threads,
        egaroucid_timeout_sec=args.egaroucid_timeout_sec,
        score_temperature=args.score_temperature,
        no_legal_mask_policy=False,
        blend_params=",".join(f"{alpha:.1f}" for alpha in BLEND_PARAMS),
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
        jobs=args.jobs,
        hint_cache_db=hint_cache_db,
        raw_hint_samples=0,
        output_dir=output_dir,
        progress_interval=1000,
        verbose=args.verbose,
    )


def make_summary_rows(result: dict) -> List[dict]:
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


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    result: dict,
    rows: Sequence[dict],
    elapsed_sec: float,
    startup_available_memory_mib: Optional[float],
    resources: dict,
) -> None:
    csv_path = output_dir / "random_wthor_blend_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "requested_positions": args.positions,
        "evaluated_positions": rows[0]["positions"] if rows else 0,
        "data_split": args.data_split,
        "split_positions": result["split_positions"],
        "split_seed": args.split_seed,
        "sample_seed": args.sample_seed,
        "blend_params": list(BLEND_PARAMS),
        "egaroucid_level": args.egaroucid_level,
        "jobs": args.jobs,
        "egaroucid_threads_per_job": args.egaroucid_threads,
        "progress_interval_sec": args.progress_interval_sec,
        "elapsed_sec": elapsed_sec,
        "startup_available_memory_mib": startup_available_memory_mib,
        "resources": resources,
        "hint_cache_stats": result["hint_cache_stats"],
        "invalid_policy_samples": result["invalid_policy_samples"],
        "illegal_label_samples": result["illegal_label_samples"],
        "results": list(rows),
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
    print("同時評価処理数", args.jobs)
    print("1処理当たりのEgaroucidスレッド数", args.egaroucid_threads)
    print("進捗表示間隔", f"{args.progress_interval_sec:g}秒")
    if args.hint_cache_db is not None:
        print("共有hintキャッシュ", args.hint_cache_db)
    print("WTHOR全局面数", f"{available_positions:,}")
    if startup_available_memory_mib is not None:
        print("実行開始前の空き物理メモリ", f"{startup_available_memory_mib:.0f} MiB")
    print("出力先", output_dir)


def print_results(rows: Sequence[dict], elapsed_sec: float, resources: dict, output_dir: Path) -> None:
    print()
    print("結果")
    print("alpha  1位一致率 (95%信頼区間)       上位3手一致率")
    for row in rows:
        print(
            f"{row['alpha']:>4.1f}   "
            f"{row['top1_accuracy'] * 100.0:>7.3f}% "
            f"({row['top1_ci95_lower'] * 100.0:.3f}% - {row['top1_ci95_upper'] * 100.0:.3f}%)   "
            f"{row['top3_accuracy'] * 100.0:>7.3f}%"
        )
    print("総実行時間", f"{elapsed_sec:.3f}秒")
    if resources.get("available"):
        print("実行中の最小空き物理メモリ", f"{resources['minimum_available_memory_mib']:.0f} MiB")
        print("関連プロセスの最大使用メモリ", f"{resources['maximum_process_tree_memory_mib']:.0f} MiB")
        print("実行中の平均CPU使用率", f"{resources['average_cpu_percent']:.1f}%")
    print("集計CSV", output_dir / "random_wthor_blend_summary.csv")
    print("集計JSON", output_dir / "random_wthor_blend_summary.json")


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WTHORからN局面をseed付きで無作為抽出し、"
            "alpha=0.0から1.0まで0.2刻みの人間着手一致率をまとめて測定します。"
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
        help="既存のhintキャッシュを共有する場合のSQLiteファイル",
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
    evaluator_args = make_evaluator_args(args, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor = ResourceMonitor()
    monitor.start()
    progress = ProgressReporter(args.positions, args.progress_interval_sec)
    start_time = time.perf_counter()
    try:
        result = evaluator.evaluate(
            evaluator_args,
            progress_callback=progress.update,
            progress_interval_sec=min(10.0, args.progress_interval_sec),
        )
    finally:
        elapsed_sec = time.perf_counter() - start_time
        resources = monitor.stop()

    rows = make_summary_rows(result)
    progress.finish(rows, elapsed_sec)
    write_summary(
        output_dir,
        args,
        result,
        rows,
        elapsed_sec,
        startup_available_memory_mib,
        resources,
    )
    print_results(rows, elapsed_sec, resources, output_dir)


if __name__ == "__main__":
    main()
