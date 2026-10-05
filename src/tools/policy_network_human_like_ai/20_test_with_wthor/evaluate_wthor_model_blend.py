#!/usr/bin/env python3
"""Evaluate geometric mixtures of two policy models on a previous WTHOR sample."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from evaluate_wthor_blend_from_cache import load_reference_sample
from evaluate_wthor_blend_human_match import rank_for_distribution
from wthor_human_match_evaluation import (
    predict_policy_logits, prepare_position_groups, wilson_interval,
)
from wthor_human_match_experiment import (
    file_sha256, positive_int, report_preparation_progress, report_stage, write_csv,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "40_test_strength"))
from policy_opponent_gtp import geometric_model_distribution  # noqa: E402


def parse_betas(value):
    try:
        betas = tuple(float(token.strip()) for token in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("betas must be comma-separated numbers") from error
    if not betas or any(not math.isfinite(beta) or not 0 <= beta <= 1 for beta in betas):
        raise argparse.ArgumentTypeError("each beta must be finite and in [0, 1]")
    if len(set(betas)) != len(betas):
        raise argparse.ArgumentTypeError("betas must be distinct")
    return betas


def load_sample(reference_dir, board_data_dir=None):
    """Reproduce and hash-check the sample without opening any engine cache."""
    summary = json.loads(
        (reference_dir / "random_wthor_blend_summary.json").read_text(encoding="utf-8")
    )
    identity = json.loads(
        (reference_dir / "experiment_identity.json").read_text(encoding="utf-8")
    )
    if identity != summary["experiment_identity"]:
        raise ValueError("Reference summary and experiment_identity.json differ")
    data_dir = (board_data_dir or Path(summary["board_data_dir"])).resolve()
    groups = load_reference_sample(summary, identity, data_dir)
    return groups, summary, identity, data_dir


def evaluate_predictions(prepared, normal, weighted, betas, top_n_values=(1,)):
    """Use tournament probabilities/tie order and WTHOR symmetry-aware targets."""
    totals = {beta: {"positions": 0, "hits": {n: 0 for n in top_n_values}} for beta in betas}
    started = last_report = time.perf_counter()
    total = len(prepared.groups_by_key)
    for index, (key, items) in enumerate(prepared.groups_by_key.items(), 1):
        for item in items:
            for beta in betas:
                probabilities = geometric_model_distribution(
                    normal[key], weighted[key], item.legal, beta, logits=True,
                )
                metric = totals[beta]
                for _, targets, count in item.valid_targets:
                    rank = rank_for_distribution(probabilities, item.legal, targets)
                    metric["positions"] += count
                    for top_n in top_n_values:
                        if rank <= top_n:
                            metric["hits"][top_n] += count
        now = time.perf_counter()
        if index == total or now - last_report >= 10:
            elapsed = now - started
            eta = elapsed * (total - index) / index
            report_stage(f"集計 {index:,}/{total:,}状態 / elapsed={elapsed:.1f}s / ETA={eta:.1f}s")
            last_report = now

    rows = []
    for beta, metric in totals.items():
        positions = metric["positions"]
        row = {"model": f"model_blend_beta_{beta:g}", "beta": beta, "positions": positions}
        for top_n, hits in metric["hits"].items():
            lower, upper = wilson_interval(hits, positions)
            row.update({
                f"top{top_n}_hits": hits,
                f"top{top_n}_accuracy": hits / positions if positions else None,
                f"top{top_n}_ci95_lower": lower if positions else None,
                f"top{top_n}_ci95_upper": upper if positions else None,
            })
        rows.append(row)
    return rows


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Previous run_random_wthor_blend_experiment output; only its sample is reused")
    parser.add_argument("--normal-model", type=Path, required=True,
                        help="Normal model (beta=0 endpoint)")
    parser.add_argument("--weighted-model", type=Path, required=True,
                        help="Loss-weighted model (beta=1 endpoint)")
    parser.add_argument("--betas", type=parse_betas, default=parse_betas("0.2,0.4,0.6,0.8"))
    parser.add_argument("--board-data-dir", type=Path,
                        help="Override reference data location; sample hashes must match")
    parser.add_argument("--policy-batch-size", type=positive_int, default=512)
    parser.add_argument("--top3", action="store_true", help="Also evaluate Top-3; default is Top-1 only")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "output" / "wthor_model_blend",
                        help="New, empty result directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate models and the full sample without inference or result writes")
    return parser


def run(args):
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("--output-dir must be empty; choose a new result directory")
    models = {}
    for name in ("normal", "weighted"):
        path = getattr(args, f"{name}_model").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        models[name] = {"path": str(path), "sha256": file_sha256(path)}

    reference_dir = args.reference_dir.resolve()
    report_stage("前回と同じ標本を読み込み、SHA-256を照合")
    groups, summary, identity, data_dir = load_sample(reference_dir, args.board_data_dir)
    prepared = prepare_position_groups(groups, on_progress=report_preparation_progress)
    if prepared.invalid_policy_samples or prepared.illegal_label_samples:
        raise ValueError("Reference sample contains invalid or illegal human moves")
    positions = sum(count for items in prepared.groups_by_key.values()
                    for item in items for _, _, count in item.valid_targets)
    if positions <= 0 or positions != summary["evaluated_positions"]:
        raise ValueError("Evaluated position count differs from the completed reference")
    report_stage(f"標本一致: {positions:,}局面 / {len(groups):,}状態 / beta={args.betas}")
    if args.dry_run:
        report_stage("dry-run完了（モデル推論・エンジン探索・結果書き込みなし）")
        return

    predictions = {}
    for name, model in models.items():
        report_stage(f"{name}モデルの推論を開始: {model['path']}")
        predictions[name] = predict_policy_logits(
            groups, None, args.policy_batch_size, Path(model["path"]),
        )
        report_stage(f"{name}モデルの推論完了: {len(predictions[name]):,}状態")
    top_n_values = (1, 3) if args.top3 else (1,)
    report_stage("モデル間の幾何平均方策の着手一致率を集計")
    rows = evaluate_predictions(
        prepared, predictions["normal"], predictions["weighted"], args.betas, top_n_values,
    )
    for model in models.values():
        if file_sha256(Path(model["path"])) != model["sha256"]:
            raise ValueError("Model changed during inference; wait for training to finish")
    result = {
        "mode": "geometric_policy_model_human_match",
        "reference_dir": str(reference_dir), "reference_experiment_identity": identity,
        "models": models, "betas": args.betas,
        "formula": "normalized P_normal^(1-beta) * P_weighted^beta over legal moves",
        "board_data_dir": str(data_dir), "evaluated_positions": positions,
        "unique_states": len(groups),
        "sample_global_indices_sha256": identity["sample_global_indices_sha256"],
        "sample_records_sha256": identity["sample_records_sha256"],
        "top_n_values": top_n_values, "policy_batch_size": args.policy_batch_size,
        "agreement_definition": summary.get("agreement_definition"),
        "confidence_interval": "95% Wilson interval over position samples",
        "engine_searches": 0, "elapsed_sec": time.perf_counter() - started,
        "results": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wthor_model_blend_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    write_csv(output_dir / "wthor_model_blend_summary.csv", rows, tuple(rows[0]))
    for row in rows:
        rates = ", ".join(f"Top-{n}={row[f'top{n}_accuracy']:.3%}" for n in top_n_values)
        print(f"{row['model']}: {rates} ({row['top1_hits']:,}/{row['positions']:,})", flush=True)
    report_stage(f"結果を保存: {output_dir}")
    return result


def main():
    parser = make_argparser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
