#!/usr/bin/env python3
"""WTHOR人間着手一致率実験のCLI、実行制御、成果物出力を担う。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import evaluate_wthor_blend_human_match as wthor
from wthor_hint_pipeline import (
    check_memory,
    collect_hints,
    format_duration,
)
from wthor_human_match_evaluation import (
    ALL_LEGAL_HINT_COUNT,
    BLEND_PARAMS,
    CONSOLE_LEVELS,
    CONSOLE_ONLY_HINT_COUNT,
    CONSOLE_REFERENCE_LEVEL,
    IncrementalAgreementMetrics,
    evaluate_agreement,
    load_position_groups,
    make_blend_summary_rows,
    make_console_level_summary_row,
    make_level21_reuse_validation,
    metrics_to_result,
    predict_policy_logits,
    split_position_count,
    validate_aggregate_counts,
    wilson_interval,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_VERSION = 5
SUMMARY_FIELDS = (
    "alpha",
    "positions",
    "top1_hits",
    "top1_accuracy",
    "top1_ci95_lower",
    "top1_ci95_upper",
    "top3_hits",
    "top3_accuracy",
    "top3_ci95_lower",
    "top3_ci95_upper",
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
    "top3_ci95_lower",
    "top3_ci95_upper",
    "elapsed_sec",
    "engine_seconds",
)


def configure_output_streams() -> None:
    """Make progress visible immediately under IDEs, pipes, and loggers."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True, write_through=True)
        except (OSError, TypeError, ValueError):
            pass


def report_stage(message: str) -> None:
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][段階] {message}",
        file=sys.stderr,
        flush=True,
    )


def format_live_metric(metric: dict) -> str:
    positions = int(metric["positions"])
    if positions == 0:
        return "未算出 (n=0)"
    top1_hits = int(metric["hits"][1])
    top3_hits = int(metric["hits"][3])
    top1 = top1_hits / positions
    top3 = top3_hits / positions
    top1_lower, top1_upper = wilson_interval(top1_hits, positions)
    top3_lower, top3_upper = wilson_interval(top3_hits, positions)
    return (
        f"top-1 {top1:.3%} [{top1_lower:.3%}, {top1_upper:.3%}] "
        f"| top-3 {top3:.3%} [{top3_lower:.3%}, {top3_upper:.3%}] "
        f"| n={positions:,}"
    )


def format_console_summary_line(row: dict) -> str:
    return (
        f"level {row['level']:2d}: "
        f"top-1 {row['top1_accuracy']:.3%} "
        f"[{row['top1_ci95_lower']:.3%}, "
        f"{row['top1_ci95_upper']:.3%}], "
        f"top-3 {row['top3_accuracy']:.3%} "
        f"[{row['top3_ci95_lower']:.3%}, "
        f"{row['top3_ci95_upper']:.3%}]"
    )


class AgreementProgressReporter:
    """Add partial hint rows and print the corresponding agreement values."""

    def __init__(self, metrics: IncrementalAgreementMetrics):
        self.metrics = metrics

    def accept_rows(self, level: int, rows: Sequence[tuple]) -> None:
        for state_key, hint in rows:
            self.metrics.add_hint(level, tuple(state_key), hint)

    def report(self, progress: dict) -> None:
        blend_metrics, console_metrics, _, _ = self.metrics.snapshot()
        reported_by_level = progress.get("reported_hints_by_level", {})
        target_by_level = progress.get("target_hints_by_level", {})

        def hint_progress(level: int) -> str:
            reported = reported_by_level.get(
                level,
                reported_by_level.get(str(level)),
            )
            target = target_by_level.get(
                level,
                target_by_level.get(str(level)),
            )
            if reported is None or target is None:
                return ""
            return f" | hint {int(reported):,}/{int(target):,}"

        lines = []
        for alpha in BLEND_PARAMS:
            metric = blend_metrics[alpha]
            if int(metric["positions"]) == 0:
                continue
            lines.append(
                f"  [blend alpha={alpha:.1f}] "
                f"{format_live_metric(metric)}"
                f"{hint_progress(CONSOLE_REFERENCE_LEVEL)}"
            )
        for level in CONSOLE_LEVELS:
            metric = console_metrics[level]
            if int(metric["positions"]) == 0:
                continue
            lines.append(
                f"  [Console level={level:2d}] "
                f"{format_live_metric(metric)}"
                f"{hint_progress(level)}"
            )
        if not lines:
            lines.append("  一致率: 初回hint結果待ち")
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("0より大きい値を指定してください")
    return value


def write_csv(
    path: Path,
    rows: Sequence[dict],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_dir(
    args: argparse.Namespace,
    workers: int,
) -> Path:
    return (
        SCRIPT_DIR
        / "output"
        / (
            f"random_wthor_{args.data_split}_n{args.positions}"
            f"_seed{args.sample_seed}_workers{workers}"
            f"_threads{args.egaroucid_threads}"
            f"_blendhint{ALL_LEGAL_HINT_COUNT}"
            f"_consolehint{CONSOLE_ONLY_HINT_COUNT}"
            f"_v{SUMMARY_VERSION}"
        )
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_experiment_identity(
    args: argparse.Namespace,
    workers: int,
    sample_hash: str,
    sample_records_hash: str,
) -> dict:
    return {
        "identity_version": 5,
        "summary_version": SUMMARY_VERSION,
        "positions": args.positions,
        "data_split": args.data_split,
        "split_seed": args.split_seed,
        "sample_seed": args.sample_seed,
        "sample_global_indices_sha256": sample_hash,
        "sample_records_sha256": sample_records_hash,
        "console_levels": list(CONSOLE_LEVELS),
        "blend_hint_count": ALL_LEGAL_HINT_COUNT,
        "console_hint_count_by_level": {
            str(level): (
                ALL_LEGAL_HINT_COUNT
                if level == CONSOLE_REFERENCE_LEVEL
                else CONSOLE_ONLY_HINT_COUNT
            )
            for level in CONSOLE_LEVELS
        },
        "nobook": True,
        "clear_cache_between_positions": False,
        "egaroucid_exe": str(args.egaroucid_exe.resolve()),
        "egaroucid_exe_sha256": file_sha256(args.egaroucid_exe),
        "egaroucid_threads": args.egaroucid_threads,
        "egaroucid_retries": args.egaroucid_retries,
        "workers": workers,
        "partition": (
            "level_fixed_persistent_actors_with_atomic_per_level_"
            "cursor_elastic_after_level21_v1"
        ),
        "weights": str(args.weights.resolve()),
        "weights_sha256": file_sha256(args.weights),
        "score_temperature": args.score_temperature,
        "blend_params": list(BLEND_PARAMS),
    }


def ensure_experiment_identity(
    output_dir: Path,
    identity: dict,
) -> None:
    path = output_dir / "experiment_identity.json"
    if path.exists():
        with path.open(encoding="utf-8") as identity_file:
            existing = json.load(identity_file)
        if existing != identity:
            differing = sorted(
                key
                for key in set(existing) | set(identity)
                if existing.get(key) != identity.get(key)
            )
            raise ValueError(
                "出力先の実験条件が現在の指定と一致しません。"
                f"別の--output-dirを指定してください: differing={differing}"
            )
        return
    if any(output_dir.glob("hint_score_cache_level*.sqlite3")):
        raise ValueError(
            "実験条件メタデータのないhintキャッシュが出力先にあります。"
            "別の--output-dirを指定してください"
        )
    with path.open("w", encoding="utf-8") as identity_file:
        json.dump(identity, identity_file, ensure_ascii=False, indent=2)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WTHORからN局面を無作為抽出し、ブレンド方策と"
            "Egaroucid各levelの人間着手一致率を測定します。"
        )
    )
    parser.add_argument(
        "positions",
        metavar="N",
        type=positive_int,
        help="評価する標本局面数",
    )
    parser.add_argument(
        "--data-split",
        choices=("all", "train", "val", "test"),
        default="test",
    )
    parser.add_argument("--split-seed", type=int, default=613)
    parser.add_argument("--sample-seed", type=int, default=613)
    parser.add_argument(
        "--board-data-dir",
        type=Path,
        default=None,
        help=(
            "盤面データのディレクトリ。省略時は"
            "%%EGAROUCID_DATA%%/train_data/board_data/records1"
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=wthor.default_weights_file(),
    )
    parser.add_argument(
        "--egaroucid-exe",
        type=Path,
        default=wthor.default_egaroucid_exe(),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help=(
            "同時に動かすConsoleプロセス数。省略時は"
            "論理CPU数 // --egaroucid-threads。全levelを同時に"
            "開始するため11以上が必要"
        ),
    )
    parser.add_argument(
        "--egaroucid-threads",
        type=positive_int,
        default=2,
        help=(
            "1つのConsoleプロセスが使う探索スレッド数"
            "（既定値: 2）"
        ),
    )
    parser.add_argument(
        "--egaroucid-timeout-sec",
        type=positive_float,
        default=1800.0,
    )
    parser.add_argument(
        "--egaroucid-retries",
        type=int,
        choices=range(0, 6),
        default=2,
        help="timeout・異常終了時にConsoleを再起動する回数",
    )
    parser.add_argument(
        "--score-temperature",
        type=positive_float,
        default=1.0,
    )
    parser.add_argument(
        "--policy-batch-size",
        type=positive_int,
        default=1024,
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=positive_float,
        default=30.0,
        help=(
            "標準エラーへ探索進捗と暫定一致率を表示する間隔"
            "（既定値: 30秒）"
        ),
    )
    parser.add_argument(
        "--minimum-remaining-memory-mib",
        type=positive_float,
        default=16384.0,
    )
    parser.add_argument(
        "--estimated-engine-memory-mib",
        type=positive_float,
        default=1450.0,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="標本と実行構成だけを表示し、探索しない",
    )
    return parser


def require_all_levels_to_start(workers: int) -> int:
    minimum_workers = len(CONSOLE_LEVELS)
    if workers < minimum_workers:
        raise ValueError(
            "level 1から21の専用Consoleを同時に起動するため、"
            f"--workersには{minimum_workers}以上を指定してください: "
            f"workers={workers}"
        )
    return workers


def main() -> None:
    configure_output_streams()
    parser = make_argparser()
    args = parser.parse_args()
    if args.board_data_dir is None:
        try:
            args.board_data_dir = wthor.default_board_data_dir()
        except KeyError:
            parser.error(
                "--board-data-dirを指定するか、"
                "EGAROUCID_DATA環境変数を設定してください"
            )
    started_at_unix = time.time()
    total_start = time.perf_counter()
    logical_cpus = os.cpu_count() or 1
    workers = args.workers or max(
        1,
        logical_cpus // args.egaroucid_threads,
    )
    workers = require_all_levels_to_start(workers)
    search_thread_budget = workers * args.egaroucid_threads
    if search_thread_budget > logical_cpus:
        print(
            "注意: Console探索スレッド総数が論理CPU数を超えます: "
            f"{search_thread_budget} > {logical_cpus}",
            file=sys.stderr,
            flush=True,
        )

    report_stage("WTHOR標本の確認と読込を開始")
    dat_files = wthor.discover_dat_files(args.board_data_dir)
    available_positions = wthor.count_position_samples(dat_files, None)
    selected_split_positions = split_position_count(
        available_positions,
        args.data_split,
    )
    if args.positions > selected_split_positions:
        raise ValueError(
            f"N={args.positions} exceeds the {args.data_split} split size "
            f"of {selected_split_positions}"
        )

    load_start = time.perf_counter()
    (
        groups,
        split_positions,
        sample_hash,
        sample_records_hash,
    ) = load_position_groups(
        dat_files,
        available_positions,
        args.positions,
        args.data_split,
        args.split_seed,
        args.sample_seed,
    )
    load_elapsed = time.perf_counter() - load_start
    maximum_useful_processes = len(CONSOLE_LEVELS) * len(groups)
    workers = min(workers, maximum_useful_processes)
    search_thread_budget = workers * args.egaroucid_threads
    initial_level21_processes = min(
        len(groups),
        max(1, workers - (len(CONSOLE_LEVELS) - 1)),
    )
    output_dir = args.output_dir or default_output_dir(args, workers)

    report_stage(
        "WTHOR標本の読込を完了 "
        f"({format_duration(load_elapsed)})"
    )
    print("評価標本", f"{args.positions:,}局面", flush=True)
    print("一意な盤面・手番", f"{len(groups):,}状態", flush=True)
    print(
        "重複再利用",
        f"{args.positions - len(groups):,}局面",
        flush=True,
    )
    print("Console worker", workers, flush=True)
    print(
        "1 worker当たりの探索スレッド",
        args.egaroucid_threads,
        flush=True,
    )
    print("探索スレッド総数", search_thread_budget, flush=True)
    print("論理CPUスレッド", logical_cpus, flush=True)
    print("level 21", "ブレンドとConsole単体で共有", flush=True)
    print(
        "level 1-19のConsole",
        "cold cache開始時は各level 1プロセスを常駐",
        flush=True,
    )
    print(
        "level 21のConsole",
        f"cold cache開始時は{initial_level21_processes}プロセス、"
        "低level完了後の空き枠も使用。level 21完了後は残りlevelへ割当",
        flush=True,
    )
    print(
        "hint候補数",
        f"level 21={ALL_LEGAL_HINT_COUNT}（全合法手）、"
        f"level 1-19={CONSOLE_ONLY_HINT_COUNT}（top-3測定分）",
        flush=True,
    )
    print("出力先", output_dir, flush=True)
    if args.dry_run:
        report_stage("dry-runを完了（探索は未実行）")
        return

    startup_available_memory_mib = check_memory(
        workers,
        args.estimated_engine_memory_mib,
        args.minimum_remaining_memory_mib,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_identity = make_experiment_identity(
        args,
        workers,
        sample_hash,
        sample_records_hash,
    )
    ensure_experiment_identity(output_dir, experiment_identity)

    report_stage("Policy Networkの推論を開始")
    policy_start = time.perf_counter()
    policy_logits = predict_policy_logits(
        groups,
        args.weights,
        args.policy_batch_size,
    )
    policy_elapsed = time.perf_counter() - policy_start
    report_stage(
        "Policy Networkの推論を完了 "
        f"({format_duration(policy_elapsed)})"
    )
    live_metrics = IncrementalAgreementMetrics(
        groups,
        policy_logits,
        args.score_temperature,
    )
    progress_reporter = AgreementProgressReporter(live_metrics)

    report_stage("Egaroucidによるhint探索を開始")
    hint_start = time.perf_counter()
    hints, cache_stats, level_timing, cpu_stats = collect_hints(
        groups,
        CONSOLE_LEVELS,
        output_dir,
        workers,
        args.egaroucid_exe,
        args.egaroucid_threads,
        args.egaroucid_timeout_sec,
        args.egaroucid_retries,
        args.progress_interval_sec,
        on_rows=progress_reporter.accept_rows,
        progress_callback=progress_reporter.report,
    )
    hint_elapsed = time.perf_counter() - hint_start
    report_stage(
        "Egaroucidによるhint探索を完了 "
        f"({format_duration(hint_elapsed)})"
    )

    report_stage("最終一致率の検算と集計を開始")
    aggregate_start = time.perf_counter()
    (
        blend_metrics,
        console_metrics,
        invalid_policy_samples,
        illegal_label_samples,
    ) = evaluate_agreement(
        groups,
        hints,
        policy_logits,
        args.score_temperature,
    )
    if live_metrics.result() != (
        blend_metrics,
        console_metrics,
        invalid_policy_samples,
        illegal_label_samples,
    ):
        raise RuntimeError(
            "途中進捗の集計値と全hintから再計算した最終値が一致しません"
        )
    blend_rows = make_blend_summary_rows(
        metrics_to_result(blend_metrics)
    )
    console_rows = []
    for level in CONSOLE_LEVELS:
        timing = level_timing[level]
        console_rows.append(
            make_console_level_summary_row(
                level,
                metrics_to_result({0.0: console_metrics[level]}),
                timing["elapsed_sec"],
                timing["engine_seconds"],
            )
        )
    aggregate_elapsed = time.perf_counter() - aggregate_start
    report_stage(
        "最終一致率の検算と集計を完了 "
        f"({format_duration(aggregate_elapsed)})"
    )
    level21_validation = make_level21_reuse_validation(
        blend_rows,
        console_rows,
    )
    if not level21_validation["aggregate_counts_equal"]:
        raise RuntimeError(
            "level 21 and blend alpha=0.0 aggregate counts differ"
        )
    validate_aggregate_counts(
        args.positions,
        blend_rows,
        console_rows,
        invalid_policy_samples,
        illegal_label_samples,
    )

    elapsed_sec = time.perf_counter() - total_start
    total_cache_misses = sum(
        int(stats["misses"])
        for stats in cache_stats.values()
    )
    total_hint_computations = sum(
        int(stats["scheduled_computations"])
        for stats in cache_stats.values()
    )
    summary = {
        "summary_version": SUMMARY_VERSION,
        "issue": 613,
        "started_at_unix": started_at_unix,
        "finished_at_unix": time.time(),
        "requested_positions": args.positions,
        "evaluated_positions": blend_rows[0]["positions"],
        "unique_states": len(groups),
        "duplicate_samples_reused": args.positions - len(groups),
        "sample_global_indices_sha256": sample_hash,
        "sample_records_sha256": sample_records_hash,
        "experiment_identity": experiment_identity,
        "data_split": args.data_split,
        "split_positions": split_positions,
        "split_seed": args.split_seed,
        "sample_seed": args.sample_seed,
        "blend_params": list(BLEND_PARAMS),
        "console_levels": list(CONSOLE_LEVELS),
        "blend_console_hint_count": ALL_LEGAL_HINT_COUNT,
        "console_hint_count_by_level": {
            str(level): (
                ALL_LEGAL_HINT_COUNT
                if level == CONSOLE_REFERENCE_LEVEL
                else CONSOLE_ONLY_HINT_COUNT
            )
            for level in CONSOLE_LEVELS
        },
        "egaroucid_policy_support": {
            "blend_level_21": "all_legal_moves",
            "standalone_levels_1_to_19": "top_3_moves",
            "standalone_level_21": (
                "reused_from_all_legal_blend_evaluation"
            ),
        },
        "level21_reuse_validation": level21_validation,
        "execution": {
            "architecture": (
                "single bounded spawn process pool; level-fixed persistent "
                "Console actors atomically claim states from one cursor per "
                "level; level 21 consumes released slots until complete, "
                "then remaining levels scale out into every available slot"
            ),
            "workers": workers,
            "egaroucid_threads_per_worker": args.egaroucid_threads,
            "egaroucid_retries": args.egaroucid_retries,
            "search_thread_budget": search_thread_budget,
            "logical_cpu_threads": logical_cpus,
            "hint_command_stagger_sec": 0.0,
            "manager_proxy_hot_path": False,
            "nested_process_pools": False,
            "level21_duplicate_evaluation": False,
            "initial_level21_processes_cold_cache": (
                initial_level21_processes
            ),
            "elastic_lower_scaling_after_level21": True,
            "atomic_state_claim": True,
            "actual_actor_assignment_recorded_at": (
                "level_timing.<level>.actors[*].state_indices"
            ),
            "state_partition": (
                "level_fixed_persistent_actors_with_atomic_per_level_"
                "cursor_elastic_after_level21_v1"
            ),
            "cache_reset_between_positions": False,
            "transposition_table_scope": (
                "persistent within each level-fixed actor across every "
                "state claimed by that actor; never shared across levels "
                "or processes; additional actors use independent tables; "
                "a retriable Console error restarts only that task's "
                "Console before retrying the failed position"
            ),
            "reproducibility_note": (
                "worker count, engine thread count, sample, and allocation "
                "algorithm are part of experiment_identity; exact same-level "
                "actor assignment can depend on process completion timing "
                "and is recorded in level_timing"
            ),
        },
        "hint_computations": {
            "actual_cache_misses": total_cache_misses,
            "scheduled_computations": total_hint_computations,
            "without_state_grouping": (
                args.positions * len(CONSOLE_LEVELS)
            ),
            "saved_by_grouping_or_cache": (
                args.positions * len(CONSOLE_LEVELS)
                - total_hint_computations
            ),
        },
        "cache_stats_by_level": {
            str(level): stats
            for level, stats in cache_stats.items()
        },
        "level_timing": {
            str(level): timing
            for level, timing in level_timing.items()
        },
        "timing": {
            "elapsed_sec": elapsed_sec,
            "sample_loading_sec": load_elapsed,
            "hint_evaluation_sec": hint_elapsed,
            "policy_inference_sec": policy_elapsed,
            "aggregation_sec": aggregate_elapsed,
        },
        "resources": {
            "startup_available_memory_mib": (
                startup_available_memory_mib
            ),
            **cpu_stats,
        },
        "agreement_definition": {
            "metric": "board_symmetry_aware",
            "description": (
                "手番側と相手側の石配置をそれぞれ不変に保つ、"
                "正方形盤面の回転・鏡映による8通りの変換で、"
                "人間の実着手から移る合法手を同値手とする。"
            ),
        },
        "confidence_interval": {
            "method": "Wilson score interval",
            "confidence_level": 0.95,
            "finite_population_correction": False,
            "assumption": (
                "各標本をBernoulli試行として扱う周辺信頼区間。"
                "同一対局内および重複局面間の相関は考慮しない。"
            ),
            "comparison_note": (
                "alpha間・level間の差の検定には、同じ標本を使う"
                "対応あり解析が別途必要。Wilson区間の重なりだけでは"
                "有意差を判定しない。"
            ),
        },
        "ranking": {
            "blend": (
                "重み付き幾何平均で合成した方策と同じ着手順位を、"
                "確率を直接乗算せず、alpha * Policy Networkの"
                "変換前出力 + (1-alpha) * "
                "(Egaroucid評価値 / temperature) で計算する"
            ),
            "standalone_levels_1_to_19": (
                "Consoleへ直接要求したhint 3の出力順"
            ),
            "standalone_level_21": (
                "ブレンド用hint 64の先頭3手"
            ),
            "alpha_zero_tie_break": "Consoleのhint出力順",
            "other_tie_break": "合法手の固定順",
        },
        "invalid_policy_samples": invalid_policy_samples,
        "illegal_label_samples": illegal_label_samples,
        "results": blend_rows,
        "console_level_results": console_rows,
    }

    report_stage("CSV・JSON成果物の保存を開始")
    write_csv(
        output_dir / "random_wthor_blend_summary.csv",
        blend_rows,
        SUMMARY_FIELDS,
    )
    write_csv(
        output_dir / "random_wthor_console_level_summary.csv",
        console_rows,
        CONSOLE_SUMMARY_FIELDS,
    )
    with (
        output_dir / "random_wthor_blend_summary.json"
    ).open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=2)
    report_stage("CSV・JSON成果物の保存を完了")

    print()
    print("ブレンド方策")
    for row in blend_rows:
        print(
            f"alpha={row['alpha']:.1f}: "
            f"top-1 {row['top1_accuracy']:.3%} "
            f"[{row['top1_ci95_lower']:.3%}, "
            f"{row['top1_ci95_upper']:.3%}], "
            f"top-3 {row['top3_accuracy']:.3%} "
            f"[{row['top3_ci95_lower']:.3%}, "
            f"{row['top3_ci95_upper']:.3%}]"
        )
    print()
    print("Console単体")
    for row in console_rows:
        print(format_console_summary_line(row))
    print()
    print("hint評価時間", format_duration(hint_elapsed))
    print("総実行時間", format_duration(elapsed_sec))
    if cpu_stats["average_cpu_percent"] is not None:
        print(
            "hint評価中の平均CPU使用率",
            f"{cpu_stats['average_cpu_percent']:.1f}%",
        )
    print(
        "集計JSON",
        output_dir / "random_wthor_blend_summary.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
