#!/usr/bin/env python3
"""Evaluate a new model on a previous sample using read-only level-21 hints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import time

import evaluate_wthor_blend_human_match as wthor
from wthor_hint_pipeline import HintCache
from wthor_human_match_evaluation import (
    ALL_LEGAL_HINT_COUNT, CONSOLE_REFERENCE_LEVEL, IncrementalAgreementMetrics,
    load_position_groups, make_blend_summary_rows, metrics_to_result,
    predict_policy_logits, prepare_position_groups,
)
from wthor_human_match_experiment import (
    file_sha256, positive_int, report_preparation_progress, report_stage, write_csv,
)


class ReadOnlyHintCache(HintCache):
    """Reuse hint validation without creating files, changing PRAGMAs or writing."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)


def load_reference(reference_dir):
    path = reference_dir / "random_wthor_blend_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    identity = summary["experiment_identity"]
    saved = json.loads((reference_dir / "experiment_identity.json").read_text(encoding="utf-8"))
    if identity != saved:
        raise ValueError("Reference summary and experiment_identity.json differ")
    if identity.get("console_hint_count_by_level", {}).get("21") != ALL_LEGAL_HINT_COUNT:
        raise ValueError("Reference must contain all legal moves at Egaroucid level 21")
    temperature = float(identity["score_temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid reference score_temperature")
    alphas = tuple(float(a) for a in identity["blend_params"] if 0 < float(a) <= 1)
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("Reference must contain distinct positive blend parameters")
    return summary, identity, alphas


def load_reference_sample(summary, identity, board_data_dir):
    if identity.get("split_manifest_sha256"):
        raise ValueError("References using --split-manifest are not supported by this launcher")
    files = wthor.discover_dat_files(board_data_dir)
    available = wthor.count_position_samples(files, None)
    groups, split_positions, indices_hash, records_hash = load_position_groups(
        files, available, identity["positions"], identity["data_split"],
        identity["split_seed"], identity["sample_seed"],
    )
    actual = {
        "sample_global_indices_sha256": indices_hash,
        "sample_records_sha256": records_hash,
    }
    for key, value in actual.items():
        if value != identity[key] or value != summary[key]:
            raise ValueError(f"Reference sample mismatch: {key}")
    if len(groups) != summary["unique_states"] or split_positions != summary["split_positions"]:
        raise ValueError("Reference sample counts differ")
    return groups


def load_reference_hints(reference_dir, groups):
    path = reference_dir / "hint_score_cache_level21_hint64.sqlite3"
    cache = ReadOnlyHintCache(path)
    try:
        hints = cache.load(CONSOLE_REFERENCE_LEVEL, ALL_LEGAL_HINT_COUNT, [g.key for g in groups])
    finally:
        cache.close()
    if len(hints) != len(groups):
        raise ValueError(
            f"Incomplete level-21 cache: {len(hints):,}/{len(groups):,} states. "
            "No engine search will be started."
        )
    return hints


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Completed previous experiment with summary and level-21 SQLite cache")
    parser.add_argument("--model", type=Path, required=True, help="New selected_model.h5")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, empty result directory")
    parser.add_argument("--board-data-dir", type=Path,
                        help="Override reference data location; sample hashes must still match")
    parser.add_argument("--policy-batch-size", type=positive_int, default=512)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the full sample and cache without model inference or output")
    return parser


def run(args):
    started = time.perf_counter()
    reference_dir = args.reference_dir.resolve()
    summary, identity, alphas = load_reference(reference_dir)
    board_data_dir = args.board_data_dir or Path(summary["board_data_dir"])
    top_n_values = tuple(summary["top_n_values"])
    if top_n_values not in ((1,), (1, 3)):
        raise ValueError(f"Unsupported reference top-N settings: {top_n_values}")
    if not args.dry_run:
        if not args.model.is_file():
            raise FileNotFoundError(f"Training must finish first: {args.model}")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("--output-dir must be empty; choose a new result directory")

    report_stage("前回と同じ標本を読み込み、SHA-256を照合")
    groups = load_reference_sample(summary, identity, board_data_dir)
    report_stage("保存済みlevel 21の全合法手キャッシュを読み取り・検証")
    hints = load_reference_hints(reference_dir, groups)
    report_stage(
        f"標本一致: {identity['positions']:,}局面 / キャッシュ: {len(hints):,}状態 / "
        f"alpha={','.join(map(str, alphas))} / temperature={identity['score_temperature']}"
    )
    if args.dry_run:
        report_stage("dry-run完了（モデル推論・エンジン探索・結果書き込みなし）")
        return

    model_hash = file_sha256(args.model)
    report_stage("新モデルのPolicy Network推論を開始")
    logits = predict_policy_logits(groups, None, args.policy_batch_size, args.model)
    prepared = prepare_position_groups(groups, on_progress=report_preparation_progress)
    metrics = IncrementalAgreementMetrics(
        groups, logits, identity["score_temperature"], top_n_values,
        prepared_groups=prepared, blend_params=alphas,
    )
    report_stage("新モデルを含む方策の一致率・尤度を集計")
    last_report = time.perf_counter()
    for i, group in enumerate(groups, 1):
        metrics.add_hint(CONSOLE_REFERENCE_LEVEL, group.key, hints[group.key])
        if time.perf_counter() - last_report >= 30:
            report_stage(f"集計 {i:,}/{len(groups):,}状態")
            last_report = time.perf_counter()
    blend, _, invalid, illegal = metrics.result()
    if invalid or illegal or any(m["positions"] != summary["evaluated_positions"] for m in blend.values()):
        raise ValueError("Evaluated position count differs from the completed reference")
    rows = make_blend_summary_rows(metrics_to_result(blend), alphas)
    if file_sha256(args.model) != model_hash:
        raise ValueError("Model changed during inference; wait for training to finish")
    result = {
        "mode": "new_model_from_read_only_level21_cache",
        "reference_dir": str(reference_dir),
        "reference_experiment_identity": identity,
        "model": str(args.model.resolve()), "model_sha256": model_hash,
        "board_data_dir": str(board_data_dir.resolve()),
        "evaluated_positions": rows[0]["positions"], "unique_states": len(groups),
        "sample_global_indices_sha256": identity["sample_global_indices_sha256"],
        "sample_records_sha256": identity["sample_records_sha256"],
        "blend_params": alphas, "score_temperature": identity["score_temperature"],
        "top_n_values": top_n_values, "policy_batch_size": args.policy_batch_size,
        "cache_hits": len(hints), "engine_searches": 0,
        "agreement_definition": summary.get("agreement_definition"),
        "likelihood_definition": {
            **(summary.get("likelihood_definition") or {}),
            "reported_for": "positive blend alphas using the new model",
        },
        "confidence_interval": summary.get("confidence_interval"),
        "elapsed_sec": time.perf_counter() - started, "results": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "cached_wthor_blend_summary.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(args.output_dir / "cached_wthor_blend_summary.csv", rows, tuple(rows[0]))
    for row in rows:
        print(f"alpha={row['alpha']:.1f}: Top-1={row['top1_accuracy']:.3%}, "
              f"likelihood={row['mean_likelihood_percent']:.3f}%, "
              f"NLL={row['negative_log_likelihood']:.6f}", flush=True)
    report_stage(f"結果を保存: {args.output_dir.resolve()}")
    return result


def main():
    parser = make_argparser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
