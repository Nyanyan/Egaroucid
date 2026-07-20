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
    evaluate_agreement,
    load_position_groups,
    make_blend_summary_rows,
    make_console_level_summary_row,
    make_level21_reuse_validation,
    metrics_to_result,
    predict_policy_logits,
    split_position_count,
    validate_aggregate_counts,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_VERSION = 3
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
        "identity_version": 3,
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
        "partition": "fixed_strided_shards_v1",
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
            "%EGAROUCID_DATA%/train_data/board_data/records1"
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
            "論理CPU数 // --egaroucid-threads"
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


def main() -> None:
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
    search_thread_budget = workers * args.egaroucid_threads
    if search_thread_budget > logical_cpus:
        print(
            "注意: Console探索スレッド総数が論理CPU数を超えます: "
            f"{search_thread_budget} > {logical_cpus}",
            file=sys.stderr,
        )

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
    workers = min(workers, len(groups))
    search_thread_budget = workers * args.egaroucid_threads
    output_dir = args.output_dir or default_output_dir(args, workers)

    print("評価標本", f"{args.positions:,}局面")
    print("一意な盤面・手番", f"{len(groups):,}状態")
    print("重複再利用", f"{args.positions - len(groups):,}局面")
    print("Console worker", workers)
    print("1 worker当たりの探索スレッド", args.egaroucid_threads)
    print("探索スレッド総数", search_thread_budget)
    print("論理CPUスレッド", logical_cpus)
    print("level 21", "ブレンドとConsole単体で共有")
    print(
        "hint候補数",
        f"level 21={ALL_LEGAL_HINT_COUNT}（全合法手）、"
        f"level 1-19={CONSOLE_ONLY_HINT_COUNT}（top-3測定分）",
    )
    print("出力先", output_dir)
    if args.dry_run:
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
    )
    hint_elapsed = time.perf_counter() - hint_start

    policy_start = time.perf_counter()
    policy_logits = predict_policy_logits(
        groups,
        args.weights,
        args.policy_batch_size,
    )
    policy_elapsed = time.perf_counter() - policy_start

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
                "single spawn process pool; one deterministic state shard "
                "and one persistent Console child per task"
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
            "state_partition": "fixed_strided_shards_v1",
            "cache_reset_between_positions": False,
            "transposition_table_scope": (
                "persistent within each fixed state shard"
            ),
            "reproducibility_note": (
                "worker count, engine thread count, sample, and partition "
                "are part of experiment_identity"
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
                "合法手上で alpha * policy_logit + "
                "(1-alpha) * (Egaroucid score / temperature) "
                "を直接順位付けするlog空間計算"
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
        print(
            f"level {row['level']:2d}: "
            f"top-1 {row['top1_accuracy']:.3%}, "
            f"top-3 {row['top3_accuracy']:.3%}"
        )
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
    )


if __name__ == "__main__":
    main()
