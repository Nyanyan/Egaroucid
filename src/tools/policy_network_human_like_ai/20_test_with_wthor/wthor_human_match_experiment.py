#!/usr/bin/env python3
"""WTHOR人間着手一致率実験のCLI、実行制御、成果物出力を担う。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import evaluate_wthor_blend_human_match as wthor
from wthor_data_source import add_board_data_arguments, resolve_board_data_args
from wthor_edax import DEFAULT_EDAX_EXE, EDAX_HASH_BITS, EDAX_LEVELS
from wthor_engine_comparison import collect_comparison_hints
from wthor_hint_pipeline import (
    check_memory,
    format_duration,
)
from wthor_human_match_evaluation import (
    ALL_LEGAL_HINT_COUNT,
    BLEND_PARAMS,
    CONSOLE_LEVELS,
    CONSOLE_REFERENCE_LEVEL,
    IncrementalAgreementMetrics,
    IncrementalEngineAgreementMetrics,
    agreement_results_equal,
    evaluate_agreement,
    evaluate_engine_agreement,
    load_position_groups,
    make_blend_summary_rows,
    make_console_level_summary_row,
    make_level21_reuse_validation,
    metrics_to_result,
    predict_policy_logits,
    prepare_position_groups,
    split_position_count,
    validate_aggregate_counts,
    wilson_interval,
)
from wthor_likelihood import LIKELIHOOD_DEFINITION, LIKELIHOOD_FIELDS


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_VERSION = 8
CONSOLE_RESULT_LEVELS = tuple(
    level for level in CONSOLE_LEVELS if level != CONSOLE_REFERENCE_LEVEL
)
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
    *LIKELIHOOD_FIELDS,
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
    *LIKELIHOOD_FIELDS,
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


def report_preparation_progress(completed: int, total: int, elapsed: float) -> None:
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = format_duration((total - completed) / rate) if rate > 0 else "未算出"
    percent = 100.0 * completed / total if total else 100.0
    report_stage(
        f"一致率の集計準備 {completed:,}/{total:,}状態 ({percent:.1f}%) "
        f"経過={format_duration(elapsed)} 速度={rate:,.0f}状態/秒 残り={remaining}"
    )


def evaluation_top_n(args) -> tuple:
    return (1, 3) if getattr(args, "top3", False) else (1,)


def standalone_hint_count(args) -> int:
    return ALL_LEGAL_HINT_COUNT if getattr(args, "engine_likelihood", False) else max(evaluation_top_n(args))


def format_live_metric(metric: dict) -> str:
    positions = int(metric["positions"])
    if positions == 0:
        return "未算出 (n=0)"
    parts = []
    for top_n, hits in sorted(metric["hits"].items()):
        lower, upper = wilson_interval(int(hits), positions)
        parts.append(f"top-{top_n} {hits / positions:.3%} [{lower:.3%}, {upper:.3%}]")
    parts.append(f"n={positions:,}")
    if "likelihood_sum" in metric:
        parts.append(f"likelihood {metric['likelihood_sum'] / positions:.3%}")
    return " | ".join(parts)


def format_summary_accuracy(row: dict) -> str:
    return ", ".join(
        f"top-{top_n} {row[f'top{top_n}_accuracy']:.3%} "
        f"[{row[f'top{top_n}_ci95_lower']:.3%}, {row[f'top{top_n}_ci95_upper']:.3%}]"
        for top_n in (1, 3) if f"top{top_n}_accuracy" in row
    )


def format_console_summary_line(row: dict) -> str:
    text = f"level {row['level']:2d}: {format_summary_accuracy(row)}"
    if "mean_likelihood_percent" in row:
        text += f", likelihood {row['mean_likelihood_percent']:.3f}% (NLL {row['negative_log_likelihood']:.6f})"
    return text


def write_metric_snapshot(path, named_metrics, requested_positions):
    """Persist small aggregates on the owning collector thread, never hint rows."""
    if path is None:
        return
    rows = []
    for name, metric in named_metrics:
        row = make_console_level_summary_row(0, metrics_to_result({0.0: metric}), 0.0)
        for key in ("level", "elapsed_sec", "engine_seconds"):
            row.pop(key, None)
        positions = row["positions"]
        if not positions:
            row["top1_accuracy"] = None
        row.update(name=name, status=("pending" if not positions else
                   "complete" if positions == requested_positions else "partial"))
        rows.append(row)
    payload = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "requested_positions": requested_positions, "rows": rows}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class AgreementProgressReporter:
    """Add partial hint rows and print the corresponding agreement values."""

    def __init__(self, metrics: IncrementalAgreementMetrics, progress_path=None, requested_positions=None):
        self.metrics = metrics
        self.progress_path = progress_path
        self.requested_positions = requested_positions

    def accept_rows(self, level: int, rows: Sequence[tuple]) -> None:
        for state_key, hint in rows:
            self.metrics.add_hint(level, tuple(state_key), hint)

    def format_lines(self, progress: dict) -> list:
        blend_metrics, console_metrics, _, _ = self.metrics.snapshot()
        write_metric_snapshot(self.progress_path,
                              [(f"alpha_{alpha:.1f}", blend_metrics[alpha]) for alpha in BLEND_PARAMS]
                              + [(f"egaroucid_l{level}", console_metrics[level]) for level in CONSOLE_RESULT_LEVELS],
                              self.requested_positions)
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
            lines.append(
                f"  [blend alpha={alpha:.1f}] "
                f"{format_live_metric(metric)}"
                f"{hint_progress(CONSOLE_REFERENCE_LEVEL)}"
            )
        for level in CONSOLE_RESULT_LEVELS:
            metric = console_metrics[level]
            lines.append(
                f"  [Egaroucid for Console level={level:2d}] "
                f"{format_live_metric(metric)}"
                f"{hint_progress(level)}"
            )
        return lines

    def report(self, progress: dict) -> None:
        sys.stderr.write("\n".join(self.format_lines(progress)) + "\n")
        sys.stderr.flush()


class EdaxProgressReporter:
    def __init__(self, metrics, progress_path=None, requested_positions=None):
        self.metrics = metrics
        self.progress_path = progress_path
        self.requested_positions = requested_positions

    def accept_rows(self, level, rows):
        for state_key, hint in rows:
            self.metrics.add_hint(level, tuple(state_key), hint)

    def format_lines(self, progress):
        metrics = self.metrics.snapshot()
        write_metric_snapshot(self.progress_path,
                              [(f"edax_l{level}", metrics[level]) for level in EDAX_LEVELS],
                              self.requested_positions)
        reported = progress.get("reported_hints_by_level", {})
        targets = progress.get("target_hints_by_level", {})
        lines = []
        for level in EDAX_LEVELS:
            suffix = (
                f" | hint {reported.get(level, 0):,}/{targets[level]:,}"
                if level in targets else ""
            )
            lines.append(f"  [Edax level={level:2d}] {format_live_metric(metrics[level])}{suffix}")
        return lines

    def report(self, progress):
        sys.stderr.write("\n".join(self.format_lines(progress)) + "\n")
        sys.stderr.flush()


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("0より大きい値を指定してください")
    return value


def write_csv(
    path: Path,
    rows: Sequence[dict],
    fieldnames: Sequence[str],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
            f"_consolehint{standalone_hint_count(args)}"
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
    model = getattr(args, "model", None)
    identity = {
        "identity_version": 7,
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
                else standalone_hint_count(args)
            )
            for level in CONSOLE_LEVELS
        },
        "nobook": True,
        "edax": make_edax_identity(args, workers),
        "clear_cache_between_positions": False,
        "egaroucid_exe": str(args.egaroucid_exe.resolve()),
        "egaroucid_exe_sha256": file_sha256(args.egaroucid_exe),
        "egaroucid_threads": args.egaroucid_threads,
        "egaroucid_retries": args.egaroucid_retries,
        "workers": workers,
        "worker_budget": "shared_between_egaroucid_and_edax",
        "partition": (
            "level_fixed_persistent_actors_with_atomic_per_level_"
            "cursor_elastic_after_level21_v1"
        ),
        "weights": str(args.weights.resolve()) if model is None else None,
        "weights_sha256": file_sha256(args.weights) if model is None else None,
        "score_temperature": args.score_temperature,
        "blend_params": list(BLEND_PARAMS),
    }
    if model is not None:
        identity["model"] = str(model.resolve())
        identity["model_sha256"] = file_sha256(model)
    if getattr(args, "engine_likelihood", False):
        identity["engine_likelihood"] = True
        identity["top_n_values"] = list(evaluation_top_n(args))
    split_manifest = getattr(args, "split_manifest", None)
    if split_manifest is not None:
        identity["split_manifest_sha256"] = file_sha256(split_manifest)
    return identity


def hint_experiment_identity(identity: dict) -> dict:
    """Only search inputs affect shared hints, never neural-network weights."""
    excluded = {"weights", "weights_sha256", "model", "model_sha256",
                "split_manifest_sha256", "score_temperature", "blend_params"}
    return {key: value for key, value in identity.items() if key not in excluded}


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
    if any(output_dir.glob("*hint_score_cache_level*.sqlite3")):
        raise ValueError(
            "実験条件メタデータのないhintキャッシュが出力先にあります。"
            "別の--output-dirを指定してください"
        )
    with path.open("w", encoding="utf-8") as identity_file:
        json.dump(identity, identity_file, ensure_ascii=False, indent=2)


def make_edax_identity(args, workers):
    exe = getattr(args, "edax_exe", None)
    if exe is None:
        return None
    eval_file = args.edax_eval_file or exe.resolve().parent / "data" / "eval.dat"
    return {
        "levels": list(EDAX_LEVELS),
        "exe": str(exe.resolve()),
        "exe_sha256": file_sha256(exe),
        "eval_file": str(eval_file.resolve()),
        "eval_sha256": file_sha256(eval_file),
        "threads": args.edax_threads,
        "workers": min(args.edax_workers or workers, workers, len(EDAX_LEVELS) * args.positions),
        "hash_bits": EDAX_HASH_BITS,
        "hint_count": standalone_hint_count(args),
        "book": False,
        "ponder": False,
    }


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WTHORからN局面を無作為抽出し、ブレンド方策と"
            "Egaroucid・Edax各levelの人間着手一致率を測定します。"
        )
    )
    parser.add_argument(
        "positions",
        metavar="N",
        nargs="?",
        default=100000,
        type=positive_int,
        help="標本抽出する局面数。省略時は対称変換前のtestから10万件を重複なしで無作為抽出",
    )
    parser.add_argument(
        "--data-split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="入力ディレクトリ内の追加分割。分割済みtest全体を使う既定値はall",
    )
    parser.add_argument("--top3", action="store_true",
                        help="Include top-3 agreement (default: top-1 only)")
    parser.add_argument("--engine-likelihood", action="store_true",
                        help="Search all legal moves at every engine level and report score-softmax likelihood")
    parser.add_argument("--split-seed", type=int, default=613)
    parser.add_argument("--sample-seed", type=int, default=613)
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Reuse the exact game split saved during training")
    parser.add_argument("--model", type=Path, default=None,
                        help="Keras H5 model; takes precedence over --weights")
    parser.add_argument("--hint-cache-dir", type=Path, default=None,
                        help="Share identical engine hints between model evaluations; run sequentially")
    add_board_data_arguments(parser)
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
            "EgaroucidとEdaxを合わせた探索プロセス数の上限。省略時は"
            "論理CPU数 // max(両エンジンのスレッド数)。全12設定を"
            "同時に開始するため12以上が必要"
        ),
    )
    parser.add_argument("--edax-exe", type=Path, default=DEFAULT_EDAX_EXE)
    parser.add_argument("--edax-eval-file", type=Path, default=None,
                        help="Defaults to data/eval.dat beside the Edax executable")
    parser.add_argument("--edax-workers", type=positive_int, default=None,
                        help="Edax cap within the shared --workers budget; defaults to --workers")
    parser.add_argument("--edax-threads", type=positive_int, default=1)
    parser.add_argument("--edax-timeout-sec", type=positive_float, default=1800.0)
    parser.add_argument("--edax-retries", type=int, choices=range(0, 6), default=2)
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
            "標準エラーへ集計準備・探索の進捗と暫定一致率を表示する間隔"
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
    minimum_workers = len(CONSOLE_LEVELS) + len(EDAX_LEVELS)
    if workers < minimum_workers:
        raise ValueError(
            "EgaroucidとEdaxの全12設定を同時に起動するため、"
            f"--workersには{minimum_workers}以上を指定してください: "
            f"workers={workers}"
        )
    return workers


def main() -> None:
    configure_output_streams()
    parser = make_argparser()
    args = parser.parse_args()
    resolve_board_data_args(args, parser)
    started_at_unix = time.time()
    total_start = time.perf_counter()
    logical_cpus = os.cpu_count() or 1
    workers = args.workers or max(
        1,
        logical_cpus // max(args.egaroucid_threads, args.edax_threads),
    )
    try:
        workers = require_all_levels_to_start(workers)
    except ValueError as error:
        parser.error(str(error))
    search_thread_budget = workers * max(args.egaroucid_threads, args.edax_threads)
    if search_thread_budget > logical_cpus:
        print(
            "注意: 探索スレッド数の上限が論理CPU数を超えます: "
            f"{search_thread_budget} > {logical_cpus}",
            file=sys.stderr,
            flush=True,
        )

    report_stage("WTHOR標本の確認と読込を開始")
    print("人間着手データ", args.board_data_dir, flush=True)
    dat_files = wthor.discover_dat_files(args.board_data_dir)
    available_positions = wthor.count_position_samples(dat_files, None)
    selected_split_positions = split_position_count(
        available_positions,
        args.data_split,
    )
    if args.split_manifest is None and args.positions is not None and args.positions > selected_split_positions:
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
        args.split_manifest,
    )
    if args.positions is None:
        args.positions = split_positions
    load_elapsed = time.perf_counter() - load_start
    maximum_useful_processes = (len(CONSOLE_LEVELS) + len(EDAX_LEVELS)) * len(groups)
    workers = min(workers, maximum_useful_processes)
    search_thread_budget = workers * max(args.egaroucid_threads, args.edax_threads)
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
    print("両エンジン合計のworker上限", workers, flush=True)
    print(
        "Egaroucid 1 worker当たりの探索スレッド",
        args.egaroucid_threads,
        flush=True,
    )
    print("探索スレッド数の上限", search_thread_budget, flush=True)
    print("論理CPUスレッド", logical_cpus, flush=True)
    print("Egaroucid level 21", "ブレンド用（単体結果はalpha=0.0）", flush=True)
    print(
        "level 1,5,9,13,17のConsole",
        "cold cache開始時は各level 1プロセスを常駐",
        flush=True,
    )
    print(
        "level 21のConsole",
        "各エンジン・levelの起動枠を確保し、残りの共有枠を使用。"
        "完了した探索の枠は両エンジンで再利用",
        flush=True,
    )
    print(
        "hint候補数",
        f"level 21={ALL_LEGAL_HINT_COUNT}（全合法手）、"
        f"level 1,5,9,13,17={standalone_hint_count(args)}",
        flush=True,
    )
    print("出力先", output_dir, flush=True)
    experiment_identity = make_experiment_identity(args, workers, sample_hash, sample_records_hash)
    edax_config = experiment_identity["edax"]
    print("Edax levels", ",".join(map(str, EDAX_LEVELS)), flush=True)
    print("Edax worker上限（合計worker枠の内数）", edax_config["workers"], flush=True)
    print("Edax 1 worker当たりの探索スレッド", args.edax_threads, flush=True)
    if args.dry_run:
        report_stage("dry-runを完了（探索は未実行）")
        return

    startup_available_memory_mib = check_memory(
        workers,
        max(args.estimated_engine_memory_mib, 192.0),
        args.minimum_remaining_memory_mib,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_experiment_identity(output_dir, experiment_identity)
    hint_dir = args.hint_cache_dir or output_dir
    if args.hint_cache_dir is not None:
        if hint_dir.resolve() == output_dir.resolve():
            raise ValueError("--hint-cache-dir must differ from --output-dir")
        hint_dir.mkdir(parents=True, exist_ok=True)
        ensure_experiment_identity(hint_dir, hint_experiment_identity(experiment_identity))

    report_stage("Policy Networkの推論を開始")
    policy_start = time.perf_counter()
    policy_logits = predict_policy_logits(
        groups,
        args.weights,
        args.policy_batch_size,
        args.model,
    )
    policy_elapsed = time.perf_counter() - policy_start
    report_stage(
        "Policy Networkの推論を完了 "
        f"({format_duration(policy_elapsed)})"
    )
    report_stage("一致率の集計準備を開始（合法手・対称性を両エンジン共通で一括計算）")
    preparation_start = time.perf_counter()
    prepared_groups = prepare_position_groups(
        groups,
        progress_interval_sec=args.progress_interval_sec,
        on_progress=report_preparation_progress,
    )
    preparation_elapsed = time.perf_counter() - preparation_start
    report_stage(f"一致率の集計準備を完了 ({format_duration(preparation_elapsed)})")
    live_metrics = IncrementalAgreementMetrics(
        groups,
        policy_logits,
        args.score_temperature,
        top_n_values=evaluation_top_n(args),
        engine_likelihood=args.engine_likelihood,
        prepared_groups=prepared_groups,
    )
    progress_reporter = AgreementProgressReporter(live_metrics, output_dir / "progress_egaroucid.json", args.positions)
    edax_live_metrics = IncrementalEngineAgreementMetrics(
        groups, EDAX_LEVELS, evaluation_top_n(args), args.engine_likelihood,
        args.score_temperature, prepared_groups=prepared_groups,
    )
    edax_reporter = EdaxProgressReporter(edax_live_metrics, output_dir / "progress_edax.json", args.positions)
    # Persist initial snapshots; all blend rows fill as level-21 hints arrive.
    progress_reporter.format_lines({})
    edax_reporter.format_lines({})
    edax_dir = (getattr(args, "hint_cache_dir", None) or output_dir) / "edax"
    edax_dir.mkdir(parents=True, exist_ok=True)
    ensure_experiment_identity(edax_dir, {
        "edax": edax_config,
        "sample_global_indices_sha256": sample_hash,
        "sample_records_sha256": sample_records_hash,
        "partition": experiment_identity["partition"],
    })
    report_stage(f"Egaroucid・Edaxのhint探索を並行開始（合計{workers} worker以内）")
    hint_start = time.perf_counter()
    engine_searches = collect_comparison_hints(groups, workers, {
        "egaroucid": {
            "levels": CONSOLE_LEVELS, "output_dir": hint_dir, "workers": workers,
            "egaroucid_exe": args.egaroucid_exe,
            "egaroucid_threads": args.egaroucid_threads,
            "timeout_sec": args.egaroucid_timeout_sec,
            "max_retries": args.egaroucid_retries,
            "progress_interval_sec": args.progress_interval_sec,
            "standalone_hint_count": standalone_hint_count(args),
            "on_rows": progress_reporter.accept_rows,
        },
        "edax": {
            "levels": EDAX_LEVELS, "output_dir": edax_dir,
            "workers": edax_config["workers"],
            "egaroucid_exe": args.edax_exe,
            "egaroucid_threads": args.edax_threads,
            "timeout_sec": args.edax_timeout_sec,
            "max_retries": args.edax_retries,
            "progress_interval_sec": args.progress_interval_sec,
            "standalone_hint_count": standalone_hint_count(args),
            "edax_eval_file": Path(edax_config["eval_file"]),
            "on_rows": edax_reporter.accept_rows,
        },
    }, metric_formatters={
        "egaroucid": progress_reporter.format_lines,
        "edax": edax_reporter.format_lines,
    })
    hint_elapsed = time.perf_counter() - hint_start
    (hints, cache_stats, level_timing, cpu_stats), egaroucid_elapsed = engine_searches["egaroucid"]
    (edax_hints, edax_cache_stats, edax_timing, edax_cpu_stats), edax_elapsed = engine_searches["edax"]
    edax_metrics = evaluate_engine_agreement(
        groups, edax_hints, evaluation_top_n(args), args.engine_likelihood,
        args.score_temperature, prepared_groups=prepared_groups,
    )
    if not agreement_results_equal((edax_live_metrics.snapshot(), {}, 0, 0), (edax_metrics, {}, 0, 0)):
        raise RuntimeError("Edaxの途中集計値と最終一致率が一致しません")
    edax_rows = [
        make_console_level_summary_row(
            level, metrics_to_result({0.0: edax_metrics[level]}),
            edax_timing[level]["elapsed_sec"], edax_timing[level]["engine_seconds"],
        )
        for level in EDAX_LEVELS
    ]
    report_stage(f"Egaroucid・Edaxのhint探索を完了 ({format_duration(hint_elapsed)})")

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
        top_n_values=evaluation_top_n(args),
        engine_likelihood=args.engine_likelihood,
        prepared_groups=prepared_groups,
    )
    if not agreement_results_equal(live_metrics.result(), (
        blend_metrics,
        console_metrics,
        invalid_policy_samples,
        illegal_label_samples,
    )):
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
    # Keep the reuse check, but report the reference engine only as alpha=0.0.
    console_rows = [row for row in console_rows if row["level"] in CONSOLE_RESULT_LEVELS]
    validate_aggregate_counts(
        args.positions,
        blend_rows,
        [*console_rows, *edax_rows],
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
        "board_data_dir": str(args.board_data_dir),
        "evaluated_positions": blend_rows[0]["positions"],
        "top_n_values": list(evaluation_top_n(args)),
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
        "console_levels": list(CONSOLE_RESULT_LEVELS),
        "egaroucid_hint_levels": list(CONSOLE_LEVELS),
        "edax_levels": list(EDAX_LEVELS),
        "edax_level_results": edax_rows,
        "edax_cache_stats_by_level": edax_cache_stats,
        "edax_level_timing": edax_timing,
        "edax_resources": edax_cpu_stats,
        "blend_console_hint_count": ALL_LEGAL_HINT_COUNT,
        "console_hint_count_by_level": {
            str(level): (
                ALL_LEGAL_HINT_COUNT
                if level == CONSOLE_REFERENCE_LEVEL
                else standalone_hint_count(args)
            )
            for level in CONSOLE_LEVELS
        },
        "egaroucid_policy_support": {
            "blend_level_21": "all_legal_moves",
            "standalone_levels_1_5_9_13_17": "all_legal_moves" if args.engine_likelihood else f"top_{max(evaluation_top_n(args))}_moves",
        },
        "level21_reuse_validation": level21_validation,
        "execution": {
            "architecture": (
                "concurrent Egaroucid and Edax spawn process pools sharing "
                "one global engine-slot limit; level-fixed persistent "
                "Console actors atomically claim states from one cursor per "
                "level; level 21 consumes released slots until complete, "
                "then remaining levels scale out into every available slot"
            ),
            "workers": workers,
            "worker_budget": "shared_between_egaroucid_and_edax",
            "egaroucid_threads_per_worker": args.egaroucid_threads,
            "egaroucid_retries": args.egaroucid_retries,
            "search_thread_budget": search_thread_budget,
            "logical_cpu_threads": logical_cpus,
            "hint_command_stagger_sec": 0.0,
            "manager_proxy_hot_path": False,
            "nested_process_pools": False,
            "level21_duplicate_evaluation": False,
            "startup_slots_reserved_by_engine": {
                "egaroucid": len(CONSOLE_LEVELS),
                "edax": min(len(EDAX_LEVELS), edax_config["workers"]),
            },
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
            "egaroucid_hint_evaluation_sec": egaroucid_elapsed,
            "edax_hint_evaluation_sec": edax_elapsed,
            "policy_inference_sec": policy_elapsed,
            "agreement_preparation_sec": preparation_elapsed,
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
        "likelihood_definition": {
            **LIKELIHOOD_DEFINITION,
            "blend_policy": "softmax(alpha * network_logits + (1-alpha) * engine_scores / temperature) over legal moves",
            "reported_for": "all participants" if args.engine_likelihood else "blend alphas including 0 (engine score softmax) and 1 (policy network)",
            "engine_policy": "softmax(native engine scores / temperature) over all legal moves; descriptive score-based probabilities, not calibrated human probabilities",
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
            "standalone_levels_1_5_9_13_17": (
                f"Consoleへ直接要求したhint {standalone_hint_count(args)}の出力順"
            ),
            "edax": f"Edax native hint {standalone_hint_count(args)} output order; book and pondering disabled",
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
        [field for field in SUMMARY_FIELDS if field in blend_rows[0]],
    )
    write_csv(
        output_dir / "random_wthor_console_level_summary.csv",
        console_rows,
        [field for field in CONSOLE_SUMMARY_FIELDS if field in console_rows[0]],
    )
    write_csv(
        output_dir / "random_wthor_edax_level_summary.csv",
        edax_rows, [field for field in CONSOLE_SUMMARY_FIELDS if field in edax_rows[0]],
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
            f"{format_summary_accuracy(row)}, "
            f"likelihood {row['mean_likelihood_percent']:.3f}% "
            f"(NLL {row['negative_log_likelihood']:.6f})"
        )
    print()
    print("Egaroucid for Console単体")
    for row in console_rows:
        print(format_console_summary_line(row))
    print()
    print("Edax単体")
    for row in edax_rows:
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
