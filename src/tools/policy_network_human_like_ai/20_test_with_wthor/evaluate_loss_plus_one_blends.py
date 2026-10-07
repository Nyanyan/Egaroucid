#!/usr/bin/env python3
"""Compare the loss+1 network with level 21 and both previous policy networks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from evaluate_mistake_test_models import atomic_json, first_move, verify_subsets
from evaluate_wthor_blend_from_cache import load_reference, load_reference_hints, load_reference_sample
from evaluate_wthor_human_match import calc_legal_batch, equivalent_policy_mask
from policy_opponent_gtp import geometric_model_distribution
from prepare_mistake_test_data import read_test_records
from wthor_dataset import BOARD_DTYPE
from wthor_human_match_evaluation import (
    egaroucid_log_scores, load_position_groups, predict_policy_logits,
    prepare_position_groups, wilson_interval,
)
from wthor_human_match_experiment import file_sha256, report_preparation_progress, report_stage, write_csv

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
TRAIN = HERE.parent / "10_train_policy_network" / "trained"
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
THRESHOLDS = (1, 2, 4, 6, 10, 20)
PARTNERS = ("egaroucid_l21", "normal", "exponential")
CONFIGS = tuple(dict(model=f"loss_plus_one_with_{partner}_alpha_{alpha:.1f}",
                     partner=partner, alpha=alpha) for partner in PARTNERS for alpha in ALPHAS)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_key(row):
    player, opponent, side = (int(row[field]) for field in ("player", "opponent", "color"))
    return (player, opponent, side) if side == 0 else (opponent, player, side)


def records_from_groups(groups):
    """Expand group counts without changing any sample multiplicity or human label."""
    records = np.empty(sum(group.sample_count for group in groups), dtype=BOARD_DTYPE)
    offset = 0
    for group in groups:
        player, opponent = ((group.black, group.white) if group.side == 0 else (group.white, group.black))
        for label, count in group.policy_counts:
            records[offset:offset + count] = (player, opponent, group.side, label, 0)
            offset += count
    return records


def load_inputs(args):
    report_stage("前回と同じWTHOR testの10万局面を復元・照合")
    summary, identity, _ = load_reference(args.reference_dir)
    if identity["positions"] != 100000 or identity["data_split"] != "all" or identity["sample_seed"] != 613:
        raise ValueError("reference must be the previous 100,000-position test sample, seed 613")
    data_dir = (args.board_data_dir or Path(summary["board_data_dir"])).resolve()
    groups = load_reference_sample(summary, identity, data_dir)
    if summary["evaluated_positions"] != 100000 or sum(g.sample_count for g in groups) != 100000:
        raise ValueError("reference must contain exactly 100,000 valid records")
    sample = dict(name="wthor_test_100000", groups=groups, records=records_from_groups(groups),
                  losses=None, source_indices=None,
                  hints=load_reference_hints(args.reference_dir, groups))

    report_stage("石損別データと完了済みレベル21キャッシュを検証")
    datasets, evaluation = args.mistake_run_dir / "datasets", args.mistake_run_dir / "evaluation"
    counts, records, losses, indices, union_dir, manifests = verify_subsets(datasets)
    previous = read_json(evaluation / "mistake_test_summary.json")
    verified = read_json(evaluation / "verification.json")
    old_identity = read_json(evaluation / "experiment_identity.json")
    if (read_json(args.mistake_run_dir / "run_status.json")["state"] != "complete"
            or verified["status"] != "passed"
            or verified["summary_sha256"] != file_sha256(evaluation / "mistake_test_summary.json")
            or verified["predictions_sha256"] != file_sha256(evaluation / "predictions.npz")
            or previous["identity"] != old_identity):
        raise ValueError("mistake reference is not a verified completed evaluation")
    if (old_identity["dataset_manifests"] != manifests
            or old_identity["dataset_counts_sha256"] != file_sha256(datasets / "dataset_counts.json")
            or counts["analysis"]["identity"]["level"] != 10):
        raise ValueError("mistake subset identity or level-10 loss definition differs")
    if data_dir.name != "test" or file_sha256(data_dir.parent / "manifest.json") != counts["source"]["manifest_sha256"]:
        raise ValueError("sample and mistake subsets must come from the same unaugmented test split")
    source_records, source_ids = read_test_records(data_dir)
    _, subset_ids = read_test_records(union_dir)
    if (len(source_records) != counts["source_positions"]
            or not np.array_equal(source_records[indices], records)
            or not np.array_equal(source_ids[indices], subset_ids)):
        raise ValueError("mistake records or game IDs differ from the original test")
    for field in ("egaroucid_exe_sha256", "egaroucid_threads", "nobook", "clear_cache_between_positions",
                  "console_hint_count_by_level", "score_temperature"):
        if old_identity.get(field) != identity.get(field):
            raise ValueError(f"reference engine configuration differs: {field}")
    mistake_groups, _, index_hash, record_hash = load_position_groups(
        sorted(union_dir.glob("*.dat")), len(records), len(records), "all", 613, 613)
    if (index_hash != old_identity["sample_global_indices_sha256"]
            or record_hash != old_identity["sample_records_sha256"]):
        raise ValueError("mistake reference sample differs")
    mistakes = dict(name="mistakes", groups=mistake_groups, records=records, losses=losses,
                    source_indices=indices, hints=load_reference_hints(evaluation, mistake_groups))
    source_losses = np.fromfile(datasets / "source_test.losses", dtype="u1")
    dataset_counts = [{"dataset": sample["name"], "minimum_loss": None, "positions": len(sample["records"])}]
    for threshold in THRESHOLDS:
        if not np.array_equal(indices[losses >= threshold], np.flatnonzero(source_losses >= threshold)):
            raise ValueError(f"incorrect membership for loss >= {threshold}")
        dataset_counts.append(dict(dataset=f"loss_ge_{threshold}", minimum_loss=threshold,
                                   positions=int((losses >= threshold).sum())))
    provenance = dict(sample_reference=str(args.reference_dir.resolve()),
                      sample_identity=identity, mistake_reference=str(args.mistake_run_dir.resolve()),
                      mistake_identity=old_identity, dataset_counts=dataset_counts,
                      source_test_manifest_sha256=counts["source"]["manifest_sha256"],
                      source_loss_sha256=counts["source_loss_sha256"],
                      caches=[dict(path=str((directory / "hint_score_cache_level21_hint64.sqlite3").resolve()),
                                   sha256=file_sha256(directory / "hint_score_cache_level21_hint64.sqlite3"))
                              for directory in (args.reference_dir, evaluation)])
    return [sample, mistakes], provenance


def predict_blends(groups, prepared, logits, hints, temperature, progress=None):
    moves = np.empty((len(groups), len(CONFIGS)), dtype=np.int8)
    last_report = time.perf_counter()
    for i, group in enumerate(groups):
        key = group.key
        legal = prepared.groups_by_key[key][0].legal
        new = logits["new"][key]
        engine = egaroucid_log_scores(hints[key], legal, temperature, require_all_legal=True)
        for j, config in enumerate(CONFIGS):
            alpha, partner = config["alpha"], config["partner"]
            if alpha == 1:
                move = first_move(new, legal)
            elif partner == "egaroucid_l21":
                if alpha == 0:
                    # Previous evaluations use native engine order only for this endpoint.
                    move = max(hints[key].move_order, key=lambda p: engine[p])
                else:
                    scores = np.full(64, -np.inf)
                    selected = list(legal)
                    scores[selected] = alpha * new[selected] + (1 - alpha) * engine[selected]
                    move = first_move(scores, legal)
            else:
                move = first_move(geometric_model_distribution(logits[partner][key], new, legal, alpha), legal)
            moves[i, j] = move
        if progress is not None and (i + 1 == len(groups) or time.perf_counter() - last_report >= 15):
            progress(i + 1, len(groups))
            last_report = time.perf_counter()
    return moves


def summarize_predictions(dataset, prepared, state_moves):
    records, groups = dataset["records"], dataset["groups"]
    by_key = {g.key: i for i, g in enumerate(groups)}
    target_maps = {key: {p: sum(1 << move for move in targets) for p, targets, _ in items[0].valid_targets}
                   for key, items in prepared.groups_by_key.items()}
    keys = [record_key(row) for row in records]
    states = np.array([by_key[key] for key in keys], dtype=np.int64)
    labels = records["policy"]
    targets = np.array([target_maps[key][int(label)] for key, label in zip(keys, labels)], dtype=np.uint64)
    moves = state_moves[states]
    matched = (np.left_shift(np.uint64(1), moves.astype(np.uint64)) & targets[:, None]) != 0
    exact = moves == labels[:, None]
    conditions = ((None,) if dataset["losses"] is None else THRESHOLDS)
    rows = []
    for threshold in conditions:
        selected = np.ones(len(records), dtype=bool) if threshold is None else dataset["losses"] >= threshold
        count = int(selected.sum())
        for col, config in enumerate(CONFIGS):
            hits, strict = int(matched[selected, col].sum()), int(exact[selected, col].sum())
            low, high = wilson_interval(hits, count) if count else (None, None)
            rows.append(dict(**config, dataset=dataset["name"] if threshold is None else f"loss_ge_{threshold}",
                             minimum_loss=threshold, positions=count, top1_hits=hits,
                             top1_accuracy=hits / count if count else None,
                             top1_accuracy_percent=100 * hits / count if count else None,
                             top1_ci95_lower=low, top1_ci95_upper=high,
                             exact_top1_hits=strict, exact_top1_accuracy=strict / count if count else None))
    return rows, moves, targets


def verify_predictions(records, losses, moves, targets, rows):
    """Independently recompute legality/symmetry and recount with scalar bit tests."""
    if moves.shape != (len(records), len(CONFIGS)) or np.any((moves < 0) | (moves >= 64)):
        raise ValueError("invalid prediction shape or missing move")
    legal = calc_legal_batch(records["player"], records["opponent"])
    bits = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
    target_bool = equivalent_policy_mask(records["player"], records["opponent"], records["policy"].astype(np.int64))
    independent = (target_bool.astype(np.uint64) * bits).sum(axis=1, dtype=np.uint64)
    if not np.array_equal(independent, targets):
        raise ValueError("symmetry target verification failed")
    if np.any((np.left_shift(np.uint64(1), moves.astype(np.uint64)) & legal[:, None]) == 0):
        raise ValueError("illegal prediction")
    columns = {config["model"]: i for i, config in enumerate(CONFIGS)}
    for row in rows:
        selected = np.arange(len(records)) if row["minimum_loss"] is None else np.flatnonzero(losses >= row["minimum_loss"])
        col = columns[row["model"]]
        hits = sum(bool(int(targets[i]) & (1 << int(moves[i, col]))) for i in selected)
        exact = sum(int(records["policy"][i]) == int(moves[i, col]) for i in selected)
        if (len(selected), hits, exact) != (row["positions"], row["top1_hits"], row["exact_top1_hits"]):
            raise ValueError("independent accuracy recount failed")
    # Each family's alpha=1 endpoint is the identical new network.
    if not np.array_equal(moves[:, 5], moves[:, 11]) or not np.array_equal(moves[:, 5], moves[:, 17]):
        raise ValueError("alpha=1 endpoints disagree")


def limit_smoke_states(dataset, count):
    result = dict(dataset)
    result["groups"] = dataset["groups"][:count]
    keys = {group.key for group in result["groups"]}
    selected = np.array([record_key(row) in keys for row in dataset["records"]])
    result["records"] = dataset["records"][selected]
    for field in ("losses", "source_indices"):
        if dataset[field] is not None:
            result[field] = dataset[field][selected]
    return result


def run(args):
    started = time.perf_counter()
    output = (args.output_dir or HERE / "output" / (
        "loss_plus_one_blends_20261006" + (f"_smoke_{args.smoke_states}" if args.smoke_states else ""))).resolve()
    if not args.dry_run and output.exists() and any(output.iterdir()):
        raise ValueError("output is not empty; choose a new --output-dir")
    models = {name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
              for name, path in (("new", args.new_model), ("normal", args.normal_model), ("exponential", args.exponential_model))}
    datasets, provenance = load_inputs(args)
    if args.smoke_states:
        datasets = [limit_smoke_states(dataset, args.smoke_states) for dataset in datasets]
    report_stage("混合相手: レベル21・通常NN・指数重みNN / alpha=0,0.2,0.4,0.6,0.8,1")
    for row in provenance["dataset_counts"]:
        print(f"  {row['dataset']}: {row['positions']:,}局面", flush=True)
    if args.dry_run:
        report_stage("dry-run完了: モデル3個・全評価データ・レベル21キャッシュを検証。推論と結果保存は未実行")
        return {"status": "dry-run", "models": models, "provenance": provenance}

    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "run_status.json"
    def status(state, stage, **extra):
        atomic_json(status_path, dict(state=state, stage=stage, updated_at_unix=time.time(), **extra))
    try:
        status("running", "inference")
        # Inference can be shared; search hints stay separate for each reference run.
        unique = {g.key: g for dataset in datasets for g in dataset["groups"]}
        logits = {}
        for name, model in models.items():
            report_stage(f"{name}モデル推論: {len(unique):,}状態")
            logits[name] = predict_policy_logits(list(unique.values()), None, args.policy_batch_size, Path(model["path"]))
        rows, prediction_files = [], {}
        for dataset in datasets:
            name = dataset["name"]
            status("running", f"aggregate_{name}")
            prepared = prepare_position_groups(dataset["groups"], on_progress=report_preparation_progress)
            if prepared.invalid_policy_samples or prepared.illegal_label_samples:
                raise ValueError("invalid or illegal human labels")
            def progress(done, total):
                report_stage(f"{name}: 混合手の計算 {done:,}/{total:,}状態")
                status("running", f"aggregate_{name}", completed_states=done, total_states=total)
            state_moves = predict_blends(dataset["groups"], prepared, logits, dataset["hints"],
                                         provenance["sample_identity"]["score_temperature"], progress)
            subset_rows, moves, targets = summarize_predictions(dataset, prepared, state_moves)
            prediction_path = output / f"{name}_predictions.npz"
            np.savez_compressed(prediction_path, model_names=np.array([c["model"] for c in CONFIGS]),
                                records=dataset["records"], predicted_policies=moves, symmetry_target_masks=targets,
                                losses=dataset["losses"] if dataset["losses"] is not None else np.empty(0, dtype="u1"),
                                source_indices=dataset["source_indices"] if dataset["source_indices"] is not None else np.empty(0, dtype="<u8"))
            report_stage(f"{name}: 保存した予測の合法性・対称性・一致数を独立検算")
            with np.load(prediction_path, allow_pickle=False) as saved:
                verify_predictions(saved["records"], saved["losses"], saved["predicted_policies"],
                                   saved["symmetry_target_masks"], subset_rows)
            prediction_files[prediction_path.name] = file_sha256(prediction_path)
            rows.extend(subset_rows)
            write_csv(output / "progress.csv", rows, tuple(rows[0]))
        for model in models.values():
            if file_sha256(Path(model["path"])) != model["sha256"]:
                raise ValueError("model changed during inference")
        if len(rows) != 126:
            raise ValueError("expected 3 partners x 6 alphas x 7 datasets = 126 results")
        result = dict(version=1, mode="smoke" if args.smoke_states else "full", models=models,
                      alphas=ALPHAS, thresholds=THRESHOLDS, configurations=CONFIGS, provenance=provenance,
                      smoke_states=args.smoke_states, engine_searches=0, policy_batch_size=args.policy_batch_size,
                      formula="softmax(alpha * new_logits + (1-alpha) * partner_logits) over legal moves; level21 partner_logits = score / reference_temperature",
                      alpha_definition="0 = partner alone; 1 = new loss+1 network alone",
                      agreement_definition="Top-1 accepting board-preserving symmetry, all test records equally weighted",
                      tie_break="ascending policy index, except level21 alpha=0 uses native engine order among best scores",
                      sample_prediction_order="grouped by state and human label, preserving original multiplicities",
                      confidence_interval="95% Wilson over positions, without within-game correlation correction",
                      elapsed_sec=time.perf_counter() - started, prediction_files=prediction_files, results=rows)
        atomic_json(output / "summary.json", result)
        write_csv(output / "summary.csv", rows, tuple(rows[0]))
        wide = []
        for config in CONFIGS:
            entry = dict(config)
            for row in rows:
                if row["model"] == config["model"]:
                    entry[row["dataset"] + "_accuracy_percent"] = row["top1_accuracy_percent"]
            wide.append(entry)
        write_csv(output / "accuracy_matrix.csv", wide, tuple(wide[0]))
        atomic_json(output / "verification.json", dict(status="passed", result_rows=len(rows),
                    summary_sha256=file_sha256(output / "summary.json"), prediction_files=prediction_files,
                    checks=["identical reference samples and complete read-only hints", "model hashes unchanged",
                            "independently recomputed legal moves and symmetry masks", "scalar recount of all 126 rows",
                            "identical alpha=1 endpoints"]))
        status("complete", "verified", mode=result["mode"], result_rows=len(rows))
        report_stage(f"126条件の評価と検算が完了: {output}")
        return result
    except BaseException as error:
        status("failed", "error", error=repr(error))
        raise


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-model", type=Path, default=TRAIN / "conv8_bn_loss_plus_one_e10_20261006/train/selected_model.h5")
    parser.add_argument("--normal-model", type=Path, default=WORKSPACE / "experiment_bat/20261002/results/train/selected_model.h5")
    parser.add_argument("--exponential-model", type=Path, default=TRAIN / "prepared_conv8_bn_e10_20261003_221904_955/selected_model.h5")
    parser.add_argument("--reference-dir", type=Path, default=WORKSPACE / "experiment_bat/20261002/results/human")
    parser.add_argument("--mistake-run-dir", type=Path, default=HERE / "output/mistake_test_20261005")
    parser.add_argument("--board-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy-batch-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="Validate all real inputs/caches without inference or writes")
    parser.add_argument("--smoke-states", type=int, default=0, help="Smoke test only: limit states after validating all real inputs")
    return parser


if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()
    if args.policy_batch_size <= 0 or args.smoke_states < 0:
        parser.error("policy-batch-size must be positive and smoke-states must be nonnegative")
    run(args)
