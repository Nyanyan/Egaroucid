#!/usr/bin/env python3
"""Evaluate the completed GPW 30-player roster on the precedent's seven samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from evaluate_loss_plus_one_blends import (
    HERE, WORKSPACE, THRESHOLDS, calc_legal_batch, equivalent_policy_mask,
    first_move, limit_smoke_states, load_inputs, record_key,
)
from evaluate_mistake_test_models import atomic_json
from evaluate_wthor_blend_from_cache import ReadOnlyHintCache
from policy_opponent_gtp import geometric_model_distribution
from wthor_engine_comparison import collect_comparison_hints
from wthor_hint_pipeline import check_memory
from wthor_human_match_evaluation import (
    egaroucid_log_scores, predict_policy_logits, prepare_position_groups, wilson_interval,
)
from wthor_human_match_experiment import file_sha256, positive_int, report_stage, write_csv
from battle_blend_strength import OutputRunLock

REPO = HERE.parents[3]
LEVELS = (1, 3, 5, 9, 13, 17, 21)
ALPHAS = (.2, .4, .6, .8, 1.0)
FAMILIES = (("normal", "egaroucid_l21"), ("loss_a", "normal"), ("loss_b", "loss_a"))
MODEL_NAMES = (tuple(f"egaroucid_l{level}" for level in LEVELS)
               + tuple(f"normal_egaroucid_alpha_{a:.1f}" for a in ALPHAS)
               + tuple(f"edax_l{level}" for level in LEVELS)
               + tuple(f"loss_a_normal_alpha_{a:.1f}" for a in ALPHAS)
               + tuple(f"loss_b_loss_a_alpha_{a:.1f}" for a in ALPHAS) + ("random",))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def input_file(identity):
    path = Path(identity["path"])
    path = path if path.is_absolute() else REPO / path
    if file_sha256(path) != identity["sha256"]:
        raise ValueError(f"File differs from the tournament: {path}")
    return path.resolve()


def load_tournament(directory):
    manifest = read_json(directory / "strength_manifest.json")
    result = read_json(directory / "strength_results.json")
    config = manifest["configuration"]
    participants, engine = config["participants"], config["engine"]
    families = [{"pi1": a, "pi2": b, "alphas": list(ALPHAS)} for a, b in FAMILIES]
    if (result["experiment_id"] != manifest["experiment_id"]
            or result["completed_match_sets"] != result["total_match_sets"]
            or result["completed_match_sets"] <= 0
            or tuple(p["name"] for p in result["players"]) != MODEL_NAMES
            or participants["paper_roster"]["total_players"] != 30
            or participants["paper_roster"]["policy_families"] != families
            or tuple(participants["baseline_levels"]) != LEVELS
            or tuple(participants["edax_levels"]) != LEVELS
            or not engine["logit_blend"] or engine["score_temperature"] != 1
            or engine["blend_egaroucid_level"] != 21):
        raise ValueError("Expected the completed GPW 30-player tournament with the recorded three blend families")
    inputs = config["inputs"]
    models = {name: inputs[key] for name, key in (
        ("normal", "policy_model"), ("loss_a", "loss_a_model"), ("loss_b", "loss_b_model"))}
    files = {name: input_file(value) for name, value in models.items()}
    for key in ("egaroucid_executable", "edax_executable", "edax_evaluation_data"):
        files[key] = input_file(inputs[key])
    return manifest, models, files


def cache_path(directory, engine, level):
    count = 64 if engine == "egaroucid" and level == 21 else 1
    prefix = "edax_" if engine == "edax" else ""
    return directory / ("edax" if engine == "edax" else "") / f"{prefix}hint_score_cache_level{level}_hint{count}.sqlite3"


def reference_hints(dataset, directory):
    """Read all prior engine predictions without modifying any reference cache."""
    hints, sources = {}, {}
    keys = [g.key for g in dataset["groups"]]
    for engine in ("egaroucid", "edax"):
        for level in LEVELS:
            if level == 3:
                continue
            path = cache_path(directory, engine, level)
            cache = ReadOnlyHintCache(path)
            try:
                rows = cache.load(level, 64 if engine == "egaroucid" and level == 21 else 1, keys)
            finally:
                cache.close()
            if len(rows) != len(keys):
                raise ValueError(f"Incomplete reference cache: {path} ({len(rows)}/{len(keys)})")
            hints[engine, level] = rows
            sources[str(path.resolve())] = file_sha256(path)
    return hints, sources


def validate_engines(manifest, provenance):
    identity = provenance["sample_identity"]
    mistake = provenance["mistake_identity"]
    for key in ("exe_sha256", "eval_sha256", "threads", "hash_bits", "hint_count", "book", "ponder", "levels"):
        if identity["edax"][key] != mistake["edax"][key]:
            raise ValueError(f"Reference Edax settings differ: {key}")
    inputs = manifest["configuration"]["inputs"]
    if (inputs["egaroucid_executable"]["sha256"] != identity["egaroucid_exe_sha256"]
            or inputs["edax_executable"]["sha256"] != identity["edax"]["exe_sha256"]
            or inputs["edax_evaluation_data"]["sha256"] != identity["edax"]["eval_sha256"]
            or identity["egaroucid_threads"] != 1 or not identity["nobook"]
            or identity["clear_cache_between_positions"]
            or identity["edax"]["threads"] != 1 or identity["edax"]["hash_bits"] != 22
            or identity["edax"]["hint_count"] != 1 or identity["edax"]["book"] or identity["edax"]["ponder"]):
        raise ValueError("Tournament executables or precedent evaluation settings differ")


def predict_states(groups, prepared, logits, hints):
    """Use the tournament's mixture direction and the precedent's tie breaking."""
    moves = np.empty((len(groups), 29), dtype=np.int8)
    columns = {name: i for i, name in enumerate(MODEL_NAMES)}
    for i, group in enumerate(groups):
        key = group.key
        legal = prepared.groups_by_key[key][0].legal
        scores = egaroucid_log_scores(hints["egaroucid", 21][key], legal, 1., require_all_legal=True)
        for engine in ("egaroucid", "edax"):
            for level in LEVELS:
                hint = hints[engine, level][key]
                move = (max(hint.move_order, key=lambda p: scores[p])
                        if engine == "egaroucid" and level == 21 else hint.move_order[0])
                moves[i, columns[f"{engine}_l{level}"]] = move
        for alpha in ALPHAS:
            mixed = (logits["normal"][key] if alpha == 1 else
                     alpha * logits["normal"][key] + (1 - alpha) * scores)
            moves[i, columns[f"normal_egaroucid_alpha_{alpha:.1f}"]] = first_move(mixed, legal)
            for pi1, pi2 in FAMILIES[1:]:
                distribution = geometric_model_distribution(logits[pi2][key], logits[pi1][key], legal, alpha)
                moves[i, columns[f"{pi1}_{pi2}_alpha_{alpha:.1f}"]] = first_move(distribution, legal)
        if (i + 1) % 10000 == 0:
            report_stage(f"Blend predictions: {i + 1:,}/{len(groups):,} states")
    return moves


def expand_predictions(dataset, prepared, state_moves, seed):
    records = dataset["records"]
    by_key = {g.key: i for i, g in enumerate(dataset["groups"])}
    keys = [record_key(row) for row in records]
    target_maps = {key: {label: sum(1 << m for m in targets) for label, targets, _ in items[0].valid_targets}
                   for key, items in prepared.groups_by_key.items()}
    targets = np.array([target_maps[key][int(row["policy"])] for key, row in zip(keys, records)], dtype=np.uint64)
    moves = np.empty((len(records), 30), dtype=np.int8)
    moves[:, :29] = state_moves[[by_key[key] for key in keys]]
    # Match both earlier random evaluations, including repeated state occurrences.
    order = (sorted(range(len(records)), key=lambda i: (keys[i], int(records[i]["policy"])))
             if dataset["losses"] is None else range(len(records)))
    rng = random.Random(seed)
    for i in order:
        moves[i, -1] = rng.choice(tuple(sorted(prepared.groups_by_key[keys[i]][0].legal)))
    return moves, targets


def summarize_and_verify(dataset, moves, targets):
    records, losses = dataset["records"], dataset["losses"]
    if moves.shape != (len(records), 30) or np.any((moves < 0) | (moves >= 64)):
        raise ValueError("Incomplete or invalid predictions")
    legal = calc_legal_batch(records["player"], records["opponent"])
    bits = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
    mask = equivalent_policy_mask(records["player"], records["opponent"], records["policy"].astype(np.int64))
    independent = (mask.astype(np.uint64) * bits).sum(axis=1, dtype=np.uint64)
    move_bits = np.left_shift(np.uint64(1), moves.astype(np.uint64))
    if not np.array_equal(targets, independent) or np.any((move_bits & legal[:, None]) == 0):
        raise ValueError("Independent legality/symmetry verification failed")
    hits, exact = (move_bits & targets[:, None]) != 0, moves == records["policy"][:, None]
    expected = np.array([bin(int(t)).count("1") / bin(int(l)).count("1") for t, l in zip(targets, legal)])
    rows = []
    for threshold in ([None] if losses is None else THRESHOLDS):
        selected = np.arange(len(records)) if threshold is None else np.flatnonzero(losses >= threshold)
        n = len(selected)
        for col, name in enumerate(MODEL_NAMES):
            count, strict = int(hits[selected, col].sum()), int(exact[selected, col].sum())
            # Recount independently from saved moves, not the vectorized hit array.
            if count != sum(bool(int(targets[i]) & (1 << int(moves[i, col]))) for i in selected):
                raise ValueError("Independent hit recount failed")
            low, high = wilson_interval(count, n) if n else (None, None)
            rows.append(dict(model=name, dataset=dataset["name"] if threshold is None else f"loss_ge_{threshold}",
                             minimum_loss=threshold, positions=n, top1_hits=count,
                             top1_accuracy=count / n if n else None,
                             top1_accuracy_percent=100 * count / n if n else None,
                             top1_ci95_lower=low, top1_ci95_upper=high,
                             exact_top1_hits=strict, exact_top1_accuracy=strict / n if n else None,
                             expected_random_accuracy=float(expected[selected].mean()) if name == "random" and n else None))
    return rows


def level3_hints(dataset, output, files, args):
    options = {}
    for engine in ("egaroucid", "edax"):
        destination = output / dataset["name"] / engine
        destination.mkdir(parents=True, exist_ok=True)
        options[engine] = dict(levels=(3,), output_dir=destination, workers=args.workers,
                               egaroucid_exe=files[f"{engine}_executable"], egaroucid_threads=1,
                               timeout_sec=1800, max_retries=2, progress_interval_sec=30,
                               standalone_hint_count=1, reuse_partial_cache=True)
        if engine == "edax":
            options[engine]["edax_eval_file"] = files["edax_evaluation_data"]
    check_memory(args.workers, 1450, 16384)
    collected = collect_comparison_hints(dataset["groups"], args.workers, options)
    return ({(engine, 3): result[0][0][3] for engine, result in collected.items()},
            {engine: result[0][1] for engine, result in collected.items()})


def run(args):
    started = time.perf_counter()
    manifest, models, files = load_tournament(args.tournament_dir)
    datasets, provenance = load_inputs(args)
    validate_engines(manifest, provenance)
    if args.smoke_states:
        datasets = [limit_smoke_states(ds, args.smoke_states) for ds in datasets]
    hints_by_dataset, references = {}, {}
    for dataset, directory in zip(datasets, (args.reference_dir, args.mistake_run_dir / "evaluation")):
        report_stage(f"Validate all cached reference engines: {dataset['name']}")
        hints_by_dataset[dataset["name"]], sources = reference_hints(dataset, directory)
        references.update(sources)
    for path in (args.tournament_dir / "strength_manifest.json", args.tournament_dir / "strength_results.json",
                 args.reference_dir / "experiment_identity.json",
                 args.mistake_run_dir / "evaluation/experiment_identity.json"):
        references[str(path.resolve())] = file_sha256(path)
    resources = files["egaroucid_executable"].parent / "resources"
    resource_hashes = {p.name: file_sha256(p) for p in (resources / "eval.egev2", resources / "eval_move_ordering_end.egev")}
    if resource_hashes != provenance["mistake_identity"]["egaroucid_evaluation_sha256"]:
        raise ValueError("Egaroucid evaluation resources differ from the reference")
    identity = dict(version=1, tournament_id=manifest["experiment_id"], models=models,
                    model_names=list(MODEL_NAMES), provenance=provenance, reference_files=references,
                    egaroucid_resources=resource_hashes, policy_batch_size=args.policy_batch_size,
                    random_seed=manifest["configuration"]["participants"]["paper_roster"]["random"]["seed"],
                    smoke_states=args.smoke_states, implementation_sha256=file_sha256(Path(__file__)))
    for row in provenance["dataset_counts"]:
        report_stage(f"{row['dataset']}: {row['positions']:,} positions (full sample)")
    if args.dry_run:
        report_stage("Dry-run passed: 30 players, 3 models, 7 samples and all prior caches verified; only L3 needs new searches")
        return identity

    output = args.output_dir or HERE / "output" / ("gpw30_human_match_20261008" +
                                                  (f"_smoke_{args.smoke_states}" if args.smoke_states else ""))
    output = output.resolve()
    if not (output / "experiment_identity.json").exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Output has unrelated files; choose another --output-dir")
    with OutputRunLock(output):
        identity_path = output / "experiment_identity.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Output experiment conditions differ; choose another --output-dir")
        atomic_json(identity_path, identity)
        def status(state, stage, **extra):
            atomic_json(output / "run_status.json", dict(state=state, stage=stage, updated_at_unix=time.time(), **extra))
            report_stage(stage)
        try:
            pending = [ds for ds in datasets if not (output / f"{ds['name']}_checkpoint.json").exists()]
            unique = {g.key: g for ds in pending for g in ds["groups"]}
            logits = {}
            for name in ("normal", "loss_a", "loss_b") if pending else ():
                status("running", f"Infer {name}: {len(unique):,} unique states")
                logits[name] = predict_policy_logits(list(unique.values()), None, args.policy_batch_size, files[name])
            rows, prediction_files, statistics = [], {}, {}
            for dataset in datasets:
                name = dataset["name"]
                path = output / f"{name}_predictions.npz"
                checkpoint = output / f"{name}_checkpoint.json"
                if checkpoint.exists():
                    saved = read_json(checkpoint)
                    if file_sha256(path) != saved["sha256"]:
                        raise ValueError(f"Saved predictions changed: {path}")
                    statistics[name] = saved["level3_cache_statistics"]
                else:
                    status("running", f"Search Egaroucid/Edax L3: {name}")
                    extra, statistics[name] = level3_hints(dataset, output, files, args)
                    hints = {**hints_by_dataset[name], **extra}
                    prepared = prepare_position_groups(dataset["groups"])
                    if prepared.invalid_policy_samples or prepared.illegal_label_samples:
                        raise ValueError("Invalid human labels")
                    status("running", f"Aggregate 30 players: {name}")
                    state_moves = predict_states(dataset["groups"], prepared, logits, hints)
                    moves, targets = expand_predictions(dataset, prepared, state_moves, identity["random_seed"])
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, records=dataset["records"], model_names=np.asarray(MODEL_NAMES),
                                        predicted_policies=moves, symmetry_target_masks=targets,
                                        losses=dataset["losses"] if dataset["losses"] is not None else np.empty(0, dtype="u1"))
                    temporary.replace(path)
                status("running", f"Verify saved predictions: {name}")
                with np.load(path, allow_pickle=False) as saved:
                    if (not np.array_equal(saved["records"], dataset["records"])
                            or tuple(saved["model_names"]) != MODEL_NAMES
                            or not np.array_equal(saved["losses"], dataset["losses"] if dataset["losses"] is not None else [])):
                        raise ValueError("Saved prediction sample differs")
                    rows.extend(summarize_and_verify(dataset, saved["predicted_policies"], saved["symmetry_target_masks"]))
                prediction_files[path.name] = file_sha256(path)
                atomic_json(checkpoint, dict(sha256=prediction_files[path.name], level3_cache_statistics=statistics[name]))
                write_csv(output / "progress.csv", rows, tuple(rows[0]))
            for name, model in models.items():
                if file_sha256(files[name]) != model["sha256"]:
                    raise ValueError(f"Model changed during evaluation: {name}")
            if len(rows) != 210:
                raise ValueError("Expected exactly 30 players x 7 conditions")
            result = dict(status="passed", mode="smoke" if args.smoke_states else "full", identity=identity,
                          results=rows, prediction_files=prediction_files, level3_cache_statistics=statistics,
                          elapsed_sec=time.perf_counter() - started,
                          agreement_definition="Top-1, accepting board-preserving symmetry; exact Top-1 also reported",
                          blend_formula="normalized pi1**alpha * pi2**(1-alpha), legal argmax, temperature 1",
                          random_definition="seed 57; sample: sorted state/label order; mistakes: source test order; independent draw per occurrence",
                          confidence_interval="95% Wilson over positions; no within-game correlation correction")
            atomic_json(output / "summary.json", result)
            write_csv(output / "summary.csv", rows, tuple(rows[0]))
            wide = []
            for name in MODEL_NAMES:
                entry = dict(model=name)
                for row in rows:
                    if row["model"] == name:
                        entry[row["dataset"] + "_accuracy_percent"] = row["top1_accuracy_percent"]
                wide.append(entry)
            write_csv(output / "accuracy_matrix.csv", wide, tuple(wide[0]))
            atomic_json(output / "verification.json", dict(status="passed", result_rows=len(rows),
                        summary_sha256=file_sha256(output / "summary.json"), prediction_files=prediction_files,
                        checks=["tournament roster and model hashes", "identical reference samples and complete caches",
                                "saved predictions: independent legal moves, symmetry targets, scalar hit recount"]))
            status("complete", f"Verified 210 results: {output}", mode=result["mode"], result_rows=210)
            return result
        except BaseException as error:
            status("failed", "Evaluation interrupted; repeat the same command to resume", error=repr(error))
            raise


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-dir", type=Path, default=HERE.parent / "40_test_strength/output/gpw30_independent_xot_50sets")
    parser.add_argument("--reference-dir", type=Path, default=WORKSPACE / "experiment_bat/20261002/results/human")
    parser.add_argument("--mistake-run-dir", type=Path, default=HERE / "output/mistake_test_20261005")
    parser.add_argument("--board-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=positive_int, default=32, help="Shared process budget for the two L3 engines (minimum 2)")
    parser.add_argument("--policy-batch-size", type=positive_int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="Verify real inputs and caches; no inference, searches or output")
    parser.add_argument("--smoke-states", type=positive_int, default=0, help="Limit states after full sample validation; separate output by default")
    return parser


def main():
    parser = make_argparser()
    args = parser.parse_args()
    if args.workers < 2:
        parser.error("--workers must be at least 2")
    run(args)


if __name__ == "__main__":
    main()
