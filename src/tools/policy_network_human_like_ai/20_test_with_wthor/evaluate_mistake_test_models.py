#!/usr/bin/env python3
"""Evaluate all 27 participants on every record in four nested mistake test sets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import threading
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "00_data_processing"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "40_test_strength"))
from pipeline_common import load_manifest
from prepare_mistake_test_data import THRESHOLDS, read_test_records
from policy_opponent_gtp import geometric_model_distribution
from wthor_engine_comparison import collect_comparison_hints
from wthor_hint_pipeline import check_memory
from wthor_human_match_evaluation import (
    BLEND_PARAMS, CONSOLE_LEVELS, egaroucid_log_scores, load_position_groups,
    predict_policy_logits, prepare_position_groups, wilson_interval,
)
from wthor_human_match_experiment import (
    ensure_experiment_identity, file_sha256, make_argparser as base_parser,
    make_experiment_identity, report_preparation_progress, report_stage, write_csv,
)

BETAS = (0.2, 0.4, 0.6, 0.8)
MODEL_NAMES = (tuple(f"alpha_{a:.1f}" for a in BLEND_PARAMS)
               + tuple(f"egaroucid_l{level}" for level in CONSOLE_LEVELS if level != 21)
               + tuple(f"edax_l{level}" for level in CONSOLE_LEVELS)
               + tuple(f"loss_weighted_alpha_{a:.1f}" for a in BLEND_PARAMS if a > 0)
               + tuple(f"model_blend_beta_{b:g}" for b in BETAS) + ("random",))


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def verify_subsets(root):
    counts = json.loads((root / "dataset_counts.json").read_text(encoding="utf-8"))
    if tuple(counts["thresholds"]) != THRESHOLDS:
        raise ValueError("expected thresholds 1,2,4,6")
    source_losses = np.fromfile(root / "source_test.losses", dtype="u1")
    if (len(source_losses) != counts["source_positions"]
            or file_sha256(root / "source_test.losses") != counts["source_loss_sha256"]):
        raise ValueError("source loss data differs from manifest")
    union = union_ids = union_indices = union_losses = union_dir = None
    hashes = {}
    for threshold in THRESHOLDS:
        name = f"{counts['source_dataset']}_MISS_{threshold}"
        directory = root / name
        manifest = load_manifest(directory, "board_split")
        if (manifest["selection"]["minimum_loss"] != threshold
                or manifest["source"] != counts["source"]
                or manifest["analysis"] != counts["analysis"]
                or manifest["source_split"] != "test" or manifest["augmentation"]):
            raise ValueError("subset provenance differs")
        records, ids = read_test_records(directory / "test")
        stem = directory / "test" / "0000000"
        indices = np.fromfile(stem.with_suffix(".source_indices"), dtype="<u8")
        losses = np.fromfile(stem.with_suffix(".losses"), dtype="u1")
        expected = np.flatnonzero(source_losses >= threshold)
        if not np.array_equal(indices, expected) or not np.array_equal(losses, source_losses[expected]):
            raise ValueError("subset does not contain exactly the requested source indices")
        if len(records) != len(indices) or len(records) != manifest["splits"]["test"]["positions"]:
            raise ValueError("subset count differs")
        if threshold == 1:
            union, union_ids, union_indices, union_losses, union_dir = records, ids, indices, losses, directory / "test"
        else:
            mask = union_losses >= threshold
            if not np.array_equal(records, union[mask]) or not np.array_equal(ids, union_ids[mask]):
                raise ValueError("nested subset records or IDs differ")
        reported = next(row for row in counts["datasets"] if row["minimum_loss"] == threshold)
        if len(records) != reported["positions"] or len(np.unique(ids)) != reported["games"]:
            raise ValueError("dataset_counts and actual records differ")
        hashes[name] = file_sha256(directory / "manifest.json")
    return counts, union, union_losses, union_indices, union_dir, hashes


def import_reference_caches(reference, output, identity):
    """Copy immutable SQLite snapshots; never open previous results for writing."""
    previous = json.loads((reference / "experiment_identity.json").read_text(encoding="utf-8"))
    for field in ("egaroucid_exe_sha256", "egaroucid_threads", "nobook", "clear_cache_between_positions",
                  "console_levels", "console_hint_count_by_level"):
        if previous.get(field) != identity.get(field):
            raise ValueError(f"reference engine configuration differs: {field}")
    for field in ("exe_sha256", "eval_sha256", "threads", "hash_bits", "hint_count", "book", "ponder", "levels"):
        if previous["edax"].get(field) != identity["edax"].get(field):
            raise ValueError(f"reference Edax configuration differs: {field}")
    sources = []
    for engine in ("egaroucid", "edax"):
        for level in CONSOLE_LEVELS:
            hint_count = 64 if engine == "egaroucid" and level == 21 else 1
            relative = Path("edax") if engine == "edax" else Path()
            relative /= f"{'edax_' if engine == 'edax' else ''}hint_score_cache_level{level}_hint{hint_count}.sqlite3"
            src, dst = reference / relative, output / relative
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                temp = dst.with_suffix(".import.sqlite3")
                source = sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)
                target = sqlite3.connect(temp)
                try:
                    source.backup(target)
                finally:
                    target.close()
                    source.close()
                temp.replace(dst)
            sources.append(str(src.resolve()))
    return {"reference_identity_sha256": file_sha256(reference / "experiment_identity.json"),
            "source_caches": sources, "method": "read-only SQLite backup; retain existing destination on resume"}


def first_move(scores, legal):
    values = np.asarray(scores)[list(legal)]
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite score on a legal move")
    return int(legal[int(np.argmax(values))])


class MistakeAgreement:
    """One prediction per state; all source occurrences retain their loss/label."""

    def __init__(self, groups, prepared, records, losses, normal, weighted, temperature, seed):
        self.groups, self.prepared = groups, prepared
        self.losses = losses
        self.normal, self.weighted, self.temperature = normal, weighted, temperature
        self.names = {name: i for i, name in enumerate(MODEL_NAMES)}
        self.key_index = {g.key: i for i, g in enumerate(groups)}
        self.predictions = np.full((len(groups), len(MODEL_NAMES) - 1), -1, dtype=np.int8)
        self.record_states = np.empty(len(records), dtype=np.int64)
        self.targets = np.empty(len(records), dtype=np.uint64)
        self.labels = records["policy"].astype(np.int8)
        self.random_moves = np.empty(len(records), dtype=np.int8)
        self.expected_random = np.empty(len(records), dtype=np.float64)
        self.lock = threading.RLock()
        targets = {}
        for key, items in prepared.groups_by_key.items():
            if len(items) != 1:
                raise ValueError("duplicate prepared state")
            targets[key] = {p: tuple(t) for p, t, _ in items[0].valid_targets}
        rng = random.Random(seed)
        for i, row in enumerate(records):
            player, opponent, side, label = (int(row[f]) for f in ("player", "opponent", "color", "policy"))
            key = (player, opponent, side) if side == 0 else (opponent, player, side)
            self.record_states[i] = self.key_index[key]
            target = targets[key][label]
            self.targets[i] = sum(1 << move for move in target)
            legal = prepared.groups_by_key[key][0].legal
            self.random_moves[i] = rng.choice(legal)
            self.expected_random[i] = len(target) / len(legal)
        for i, group in enumerate(groups):
            key = group.key
            legal = prepared.groups_by_key[key][0].legal
            self.predictions[i, self.names["alpha_1.0"]] = first_move(normal[key], legal)
            self.predictions[i, self.names["loss_weighted_alpha_1.0"]] = first_move(weighted[key], legal)
            for beta in BETAS:
                distribution = geometric_model_distribution(normal[key], weighted[key], legal, beta)
                self.predictions[i, self.names[f"model_blend_beta_{beta:g}"]] = first_move(distribution, legal)

    def accept(self, engine, level, rows):
        with self.lock:
            for key, hint in rows:
                key = tuple(key)
                index = self.key_index[key]
                legal = self.prepared.groups_by_key[key][0].legal
                if engine == "edax" or level != 21:
                    self.predictions[index, self.names[f"{engine}_l{level}"]] = hint.move_order[0]
                    continue
                engine_scores = egaroucid_log_scores(hint, legal, self.temperature, require_all_legal=True)
                # alpha=0 uses the engine's original order for tied scores.
                self.predictions[index, self.names["alpha_0.0"]] = max(
                    hint.move_order, key=lambda move: engine_scores[move])
                for prefix, logits in (("", self.normal[key]), ("loss_weighted_", self.weighted[key])):
                    for alpha in BLEND_PARAMS[1:-1]:
                        scores = np.full(64, -np.inf)
                        indices = list(legal)
                        scores[indices] = alpha * logits[indices] + (1 - alpha) * engine_scores[indices]
                        self.predictions[index, self.names[f"{prefix}alpha_{alpha:.1f}"]] = first_move(scores, legal)

    def rows(self):
        with self.lock:
            moves = np.column_stack((self.predictions[self.record_states], self.random_moves))
        rows = []
        for col, name in enumerate(MODEL_NAMES):
            valid = moves[:, col] >= 0
            move_bits = np.left_shift(np.uint64(1), np.maximum(moves[:, col], 0).astype(np.uint64))
            matched = valid & ((move_bits & self.targets) != 0)
            exact = valid & (moves[:, col] == self.labels)
            for threshold in THRESHOLDS:
                selected = self.losses >= threshold
                requested = int(selected.sum())
                positions = int((selected & valid).sum())
                hits = int((selected & matched).sum())
                strict = int((selected & exact).sum())
                lo, hi = wilson_interval(hits, positions) if positions else (None, None)
                rows.append(dict(model=name, minimum_loss=threshold, requested_positions=requested,
                                 positions=positions, top1_hits=hits,
                                 top1_accuracy=hits / positions if positions else None,
                                 top1_accuracy_percent=100 * hits / positions if positions else None,
                                 top1_ci95_lower=lo, top1_ci95_upper=hi,
                                 exact_top1_hits=strict, exact_top1_accuracy=strict / positions if positions else None,
                                 expected_random_accuracy=float(self.expected_random[selected].mean())
                                 if name == "random" and requested else None,
                                 status="complete" if positions == requested else "partial" if positions else "pending"))
        return rows

    def save_progress(self, output):
        with self.lock:
            rows = self.rows()
            atomic_json(output / "progress.json", {"updated_at_unix": time.time(), "results": rows})
            write_csv(output / "progress.csv", rows, tuple(rows[0]))
        complete = sum(row["status"] == "complete" for row in rows)
        return [f"  [mistake subsets] complete={complete}/108 model/threshold combinations; progress.json updated"]


def run(args):
    started = time.perf_counter()
    if args.workers < 12:
        raise ValueError("--workers must be at least 12")
    if args.policy_batch_size <= 0 or not math.isfinite(args.progress_interval_sec) or args.progress_interval_sec <= 0:
        raise ValueError("batch size and progress interval must be positive and finite")
    counts, records, losses, indices, union_dir, manifests = verify_subsets(args.datasets_dir)
    report_stage(f"Verified four nested datasets; union={len(records):,} records")
    groups, _, index_hash, record_hash = load_position_groups(
        sorted(union_dir.glob("*.dat")), len(records), len(records), "all", 613, 613)
    base = base_parser().parse_args([
        str(max(1, len(records))), "--model", str(args.normal_model), "--board-data-dir", str(union_dir),
        "--workers", str(args.workers), "--egaroucid-exe", str(args.egaroucid_exe),
        "--egaroucid-threads", "1", "--edax-threads", "1",
        "--edax-exe", str(args.edax_exe), "--progress-interval-sec", str(args.progress_interval_sec)])
    identity = make_experiment_identity(base, args.workers, index_hash, record_hash)
    identity.update(mode="full_mistake_test_27_models_v1", dataset_manifests=manifests,
                    dataset_counts_sha256=file_sha256(args.datasets_dir / "dataset_counts.json"),
                    model_names=list(MODEL_NAMES), thresholds=list(THRESHOLDS),
                    weighted_model=str(args.weighted_model.resolve()), weighted_model_sha256=file_sha256(args.weighted_model),
                    reference_identity_sha256=file_sha256(args.reference_dir / "experiment_identity.json"),
                    random_seed=args.random_seed, cache_policy="reuse completed state hints; search missing states only",
                    policy_batch_size=args.policy_batch_size)
    # Track evaluation resources as well as the executable for resume validation.
    resources = args.egaroucid_exe.parent / "resources"
    identity["egaroucid_evaluation_sha256"] = {p.name: file_sha256(p) for p in
                                              (resources / "eval.egev2", resources / "eval_move_ordering_end.egev")}
    if args.dry_run:
        print(json.dumps({"counts": counts["datasets"], "unique_states": len(groups), "identity": identity}, indent=2))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_experiment_identity(output, identity)
    check_memory(args.workers, 1450, 16384)
    imported = import_reference_caches(args.reference_dir, output, identity)
    atomic_json(output / "cache_import.json", imported)
    prepared = prepare_position_groups(groups, on_progress=report_preparation_progress)
    if prepared.invalid_policy_samples or prepared.illegal_label_samples:
        raise ValueError("invalid human moves in test")
    report_stage("Normal model inference")
    normal = predict_policy_logits(groups, None, args.policy_batch_size, args.normal_model)
    report_stage("Loss-weighted model inference")
    weighted = predict_policy_logits(groups, None, args.policy_batch_size, args.weighted_model)
    metrics = MistakeAgreement(groups, prepared, records, losses, normal, weighted,
                               identity["score_temperature"], args.random_seed)
    metrics.save_progress(output)
    options = {}
    for engine in ("egaroucid", "edax"):
        options[engine] = dict(
            levels=CONSOLE_LEVELS, output_dir=output / "edax" if engine == "edax" else output,
            workers=args.workers, egaroucid_exe=args.edax_exe if engine == "edax" else args.egaroucid_exe,
            egaroucid_threads=1, timeout_sec=1800, max_retries=2,
            progress_interval_sec=args.progress_interval_sec, standalone_hint_count=1, reuse_partial_cache=True,
            on_rows=lambda level, rows, engine=engine: metrics.accept(engine, level, rows))
        if engine == "edax":
            options[engine]["edax_eval_file"] = Path(identity["edax"]["eval_file"])
    report_stage(f"Search missing states only; shared budget={args.workers} engine processes")
    searches = collect_comparison_hints(groups, args.workers, options, metric_formatters={
        engine: lambda progress: metrics.save_progress(output) for engine in options}) if groups else {}
    rows = metrics.rows()
    if len(rows) != 108 or any(row["status"] != "complete" for row in rows):
        raise ValueError("not all 108 evaluations completed")
    if (file_sha256(args.normal_model) != identity["model_sha256"]
            or file_sha256(args.weighted_model) != identity["weighted_model_sha256"]):
        raise ValueError("model file changed during evaluation")
    # Persist individual predictions so all thresholds can be independently recounted.
    np.savez_compressed(output / "predictions.npz", model_names=np.array(MODEL_NAMES),
                        source_indices=indices, losses=losses, human_policies=metrics.labels,
                        symmetry_target_masks=metrics.targets,
                        predicted_policies=np.column_stack((metrics.predictions[metrics.record_states], metrics.random_moves)))
    result = dict(version=1, identity=identity, dataset_counts=counts["datasets"], unique_states=len(groups),
                  evaluated_union_positions=len(records), elapsed_seconds=time.perf_counter() - started,
                  agreement_definition="Top-1 human move agreement, accepting symmetries that preserve both players' discs",
                  exact_agreement_definition="Top-1 equals the original human move, without symmetry equivalence",
                  random_definition="Python random.Random; one uniform legal draw per union record in original test order; seed recorded in identity; same draw reused across thresholds",
                  confidence_interval="95% Wilson over positions; no within-game correlation correction",
                  cache_import=imported, engine_cache_statistics={e: value[0][1] for e, value in searches.items()},
                  engine_elapsed_seconds={e: value[1] for e, value in searches.items()}, results=rows,
                  predictions_sha256=file_sha256(output / "predictions.npz"))
    write_csv(output / "mistake_test_summary.csv", rows, tuple(rows[0]))
    wide = []
    for name in MODEL_NAMES:
        row = {"model": name}
        for entry in rows:
            if entry["model"] == name:
                row[f"loss_ge_{entry['minimum_loss']}_accuracy_percent"] = entry["top1_accuracy_percent"]
                row[f"loss_ge_{entry['minimum_loss']}_hits"] = entry["top1_hits"]
                row[f"loss_ge_{entry['minimum_loss']}_positions"] = entry["positions"]
        wide.append(row)
    write_csv(output / "mistake_test_accuracy_matrix.csv", wide, tuple(wide[0]))
    atomic_json(output / "mistake_test_summary.json", result)
    metrics.save_progress(output)
    report_stage(f"All 108 evaluations complete: {output}")
    return result


def make_argparser():
    defaults = base_parser().parse_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True, help="Completed previous 27-model experiment's engine cache directory")
    parser.add_argument("--normal-model", type=Path, required=True)
    parser.add_argument("--weighted-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--egaroucid-exe", type=Path, default=defaults.egaroucid_exe)
    parser.add_argument("--edax-exe", type=Path, default=defaults.edax_exe)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--policy-batch-size", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=57)
    parser.add_argument("--progress-interval-sec", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(make_argparser().parse_args())
