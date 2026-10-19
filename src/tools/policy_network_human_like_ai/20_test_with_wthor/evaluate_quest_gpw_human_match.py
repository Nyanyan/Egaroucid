#!/usr/bin/env python3
"""Evaluate the GPW 30-player roster separately on each Quest board-data folder."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np

import evaluate_gpw_human_match as gpw
from evaluate_loss_plus_one_blends import records_from_groups
from evaluate_wthor_blend_human_match import BOARD_SAMPLE_SIZE, discover_dat_files
from wthor_human_match_evaluation import load_position_groups
from wthor_human_match_experiment import configure_output_streams, report_preparation_progress


def inspect_datasets(root, names, positions):
    directories = sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("*.dat")))
    if names:
        missing = set(names) - {p.name for p in directories}
        if missing:
            raise ValueError(f"Unknown dataset folders: {sorted(missing)}")
        directories = [p for p in directories if p.name in names]
    if not directories:
        raise ValueError(f"No dataset folders with .dat files under {root}")
    datasets = []
    for directory in directories:
        files, available = [], 0
        for path in discover_dat_files(directory):
            size = path.stat().st_size
            if size % BOARD_SAMPLE_SIZE:
                raise ValueError(f"Truncated {BOARD_SAMPLE_SIZE}-byte records: {path}")
            available += size // BOARD_SAMPLE_SIZE
            files.append(dict(name=path.name, size=size, sha256=gpw.file_sha256(path)))
        selected = available if positions is None else positions
        if not 0 < selected <= available:
            raise ValueError(f"{directory.name}: requested {selected} positions, available {available}")
        datasets.append(dict(name=directory.name, directory=str(directory.resolve()), files=files,
                             available_positions=available, positions=selected))
    return datasets


def load_dataset(info, seed):
    directory = Path(info["directory"])
    files = [directory / item["name"] for item in info["files"]]
    groups, _, index_hash, record_hash = load_position_groups(
        files, info["available_positions"], info["positions"], "all", seed, seed)
    dataset = dict(name=info["name"], groups=groups, records=records_from_groups(groups), losses=None)
    if len(dataset["records"]) != info["positions"]:
        raise ValueError("Loaded sample count differs from the requested count")
    return dataset, dict(sample_global_indices_sha256=index_hash, sample_records_sha256=record_hash,
                         unique_states=len(groups))


def collect_dataset_hints(groups, directory, files, workers):
    # With small budgets, start fewer levels and let the existing scheduler rotate them.
    caps = ({"egaroucid": workers, "edax": workers} if workers >= 2 * len(gpw.LEVELS)
            else {"egaroucid": workers // 2, "edax": workers - workers // 2})
    options = {}
    for engine in ("egaroucid", "edax"):
        destination = directory / engine
        destination.mkdir(parents=True, exist_ok=True)
        options[engine] = dict(levels=gpw.LEVELS, output_dir=destination, workers=caps[engine],
                               egaroucid_exe=files[f"{engine}_executable"], egaroucid_threads=1,
                               timeout_sec=1800, max_retries=2, progress_interval_sec=30,
                               standalone_hint_count=1, reuse_partial_cache=True)
        if engine == "edax":
            options[engine]["edax_eval_file"] = files["edax_evaluation_data"]
    gpw.check_memory(workers, 1450, 16384)
    collected = gpw.collect_comparison_hints(groups, workers, options)
    hints = {(engine, level): result[0][0][level]
             for engine, result in collected.items() for level in gpw.LEVELS}
    statistics = {engine: result[0][1] for engine, result in collected.items()}
    return hints, statistics


def evaluate_dataset(info, directory, files, args, random_seed, status):
    result_path, prediction_path = directory / "summary.json", directory / "predictions.npz"
    if result_path.exists():
        saved = gpw.read_json(result_path)
        if (saved["status"] != "passed" or saved["dataset"] != info
                or len(saved["results"]) != len(gpw.MODEL_NAMES)
                or gpw.file_sha256(prediction_path) != saved["predictions_sha256"]):
            raise ValueError(f"Completed dataset files differ: {directory}")
        status("running", f"Reuse verified dataset: {info['name']}")
        return saved["results"]
    directory.mkdir(parents=True, exist_ok=True)
    status("running", f"Load {info['name']}: {info['positions']:,} positions")
    dataset, sample = load_dataset(info, args.sample_seed)
    prepared = gpw.prepare_position_groups(dataset["groups"], on_progress=report_preparation_progress)
    if prepared.invalid_policy_samples or prepared.illegal_label_samples:
        raise ValueError(f"Invalid human move labels: {info['name']}")
    status("running", f"Search Egaroucid/Edax L1,3,5,9,13,17,21: {info['name']}")
    hints, statistics = collect_dataset_hints(dataset["groups"], directory, files, args.workers)
    logits = {}
    for name in ("normal", "loss_a", "loss_b"):
        status("running", f"Infer {name}: {info['name']} ({len(dataset['groups']):,} states)")
        logits[name] = gpw.predict_policy_logits(dataset["groups"], None, args.policy_batch_size, files[name])
    status("running", f"Aggregate and verify 30 players: {info['name']}")
    state_moves = gpw.predict_states(dataset["groups"], prepared, logits, hints)
    moves, targets = gpw.expand_predictions(dataset, prepared, state_moves, random_seed)
    del logits, hints, prepared, state_moves
    gc.collect()
    temporary = prediction_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, records=dataset["records"], model_names=np.asarray(gpw.MODEL_NAMES),
                        predicted_policies=moves, symmetry_target_masks=targets)
    temporary.replace(prediction_path)
    del moves, targets
    with np.load(prediction_path, allow_pickle=False) as saved:
        if (not np.array_equal(saved["records"], dataset["records"])
                or tuple(saved["model_names"]) != gpw.MODEL_NAMES):
            raise ValueError("Saved sample or roster differs")
        rows = gpw.summarize_and_verify(dataset, saved["predicted_policies"], saved["symmetry_target_masks"])
    if len(rows) != 30 or any(row["positions"] != info["positions"] for row in rows):
        raise ValueError("Expected 30 results using every selected position")
    gpw.write_csv(directory / "summary.csv", rows, tuple(rows[0]))
    gpw.atomic_json(result_path, dict(status="passed", dataset=info, sample=sample, results=rows,
                    predictions_sha256=gpw.file_sha256(prediction_path), engine_cache_statistics=statistics,
                    verification="Saved moves: independent legality, symmetry targets and scalar hit recount"))
    return rows


def write_results(output, rows):
    gpw.write_csv(output / "summary.csv", rows, tuple(rows[0]))
    wide = []
    for name in gpw.MODEL_NAMES:
        entry = dict(model=name)
        for row in rows:
            if row["model"] == name:
                entry[row["dataset"] + "_accuracy_percent"] = row["top1_accuracy_percent"]
        wide.append(entry)
    gpw.write_csv(output / "accuracy_matrix.csv", wide, tuple(wide[0]))


def run(args):
    started = time.perf_counter()
    manifest, models, files = gpw.load_tournament(args.tournament_dir)
    datasets = inspect_datasets(args.board_data_root, args.dataset, args.positions)
    resources = files["egaroucid_executable"].parent / "resources"
    resource_hashes = {p.name: gpw.file_sha256(p) for p in (
        resources / "eval.egev2", resources / "eval_move_ordering_end.egev")}
    random_seed = manifest["configuration"]["participants"]["paper_roster"]["random"]["seed"]
    identity = dict(version=1, tournament_id=manifest["experiment_id"], models=models,
                    model_names=list(gpw.MODEL_NAMES), datasets=datasets,
                    input_hashes={name: gpw.file_sha256(path) for name, path in files.items()},
                    egaroucid_resources=resource_hashes, sample_seed=args.sample_seed, random_seed=random_seed,
                    policy_batch_size=args.policy_batch_size,
                    engines=dict(levels=list(gpw.LEVELS), threads=1, book=False, ponder=False,
                                 clear_cache_between_positions=False, edax_hash_bits=22,
                                 egaroucid_l21_hint_count=64, other_hint_count=1, score_temperature=1),
                    implementation_sha256={p.name: gpw.file_sha256(p) for p in (
                        Path(__file__), Path(gpw.__file__), gpw.HERE / "evaluate_loss_plus_one_blends.py",
                        gpw.HERE / "wthor_human_match_evaluation.py", gpw.HERE / "wthor_hint_pipeline.py")})
    suffix = f"n{args.positions}_seed{args.sample_seed}"
    output = (args.output_dir or gpw.HERE / "output" / f"quest_gpw30_{suffix}").resolve()
    for info in datasets:
        gpw.report_stage(f"{info['name']}: {info['positions']:,} / {info['available_positions']:,} positions")
    gpw.report_stage(f"Output: {output}")
    if args.dry_run:
        gpw.report_stage("Dry-run passed: roster, models, engines, data file sizes/hashes and counts checked; no searches")
        return identity
    if not (output / "experiment_identity.json").exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Output has unrelated files; choose another --output-dir")
    with gpw.OutputRunLock(output):
        identity_path = output / "experiment_identity.json"
        if identity_path.exists() and gpw.read_json(identity_path) != identity:
            raise ValueError("Output experiment conditions differ; choose another --output-dir")
        gpw.atomic_json(identity_path, identity)

        def status(state, stage, **extra):
            gpw.atomic_json(output / "run_status.json", dict(state=state, stage=stage,
                            updated_at_unix=time.time(), **extra))
            gpw.report_stage(stage)

        try:
            rows = []
            for info in datasets:
                rows.extend(evaluate_dataset(info, output / info["name"], files, args, random_seed, status))
                write_results(output, rows)
                gc.collect()
            # Check that the models/engines and source data stayed fixed during a long run.
            if ({name: gpw.file_sha256(path) for name, path in files.items()} != identity["input_hashes"]
                    or inspect_datasets(args.board_data_root, args.dataset, args.positions) != datasets
                    or {name: gpw.file_sha256(resources / name) for name in resource_hashes} != resource_hashes):
                raise ValueError("Inputs changed during evaluation")
            gpw.atomic_json(output / "summary.json", dict(status="passed", identity=identity, results=rows,
                elapsed_sec=time.perf_counter() - started,
                agreement_definition="Top-1 with board-preserving symmetry; exact Top-1 also reported",
                random_definition="Uniform legal move per occurrence, sorted state/label order; seed reset per folder",
                confidence_interval="95% Wilson over positions; no within-game correlation correction"))
            status("complete", f"Verified {len(rows)} results: {output}", result_rows=len(rows))
            return rows
        except BaseException as error:
            status("failed", "Evaluation interrupted; repeat the same command to resume", error=repr(error))
            raise


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-data-root", type=Path, default=gpw.WORKSPACE / "othllo_quest_data/board_data")
    parser.add_argument("--dataset", action="append", help="Evaluate only this folder name; repeat to select several")
    parser.add_argument("--tournament-dir", type=Path,
                        default=gpw.HERE.parent / "40_test_strength/output/gpw30_independent_xot_50sets")
    parser.add_argument("--positions", type=gpw.positive_int, default=100000,
                        help="Randomly sample N positions within EACH folder without replacement (default: 100000)")
    parser.add_argument("--sample-seed", type=int, default=613)
    parser.add_argument("--workers", type=gpw.positive_int, default=32, help="Shared engine process budget (minimum 2)")
    parser.add_argument("--policy-batch-size", type=gpw.positive_int, default=512)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate input files/counts; no inference, searches or writes")
    return parser


def main():
    configure_output_streams()
    parser = make_argparser()
    args = parser.parse_args()
    if args.workers < 2:
        parser.error("--workers must be at least 2")
    run(args)


if __name__ == "__main__":
    main()
