"""Evaluate all 30 reference players on the fine-tuning run's held-out Quest test."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import time

from experiment_common import (BASE, DEFAULT_RUN, DEFAULT_RESULTS, TOURNAMENT, read, write,
                               validate_model, file_sha256)
from evaluate_finetuned import load_quest
import evaluate_quest_gpw_human_match as quest
import evaluate_gpw_human_match as gpw

REFERENCE = BASE / "20_test_with_wthor/output/gpw30_under1400_test_n10000_seed613"
ENGINE_SETTINGS = dict(levels=list(gpw.LEVELS), threads=1, book=False, ponder=False,
    clear_cache_between_positions=False, edax_hash_bits=22, egaroucid_l21_hint_count=64,
    other_hint_count=1, score_temperature=1)


def seed_caches(output, files, manifest):
    """Copy a completed compatible cache read-only, then compute only missing states."""
    record_path = output / "cache_seed.json"
    previous = read(REFERENCE / "summary.json")
    identity = read(REFERENCE / "experiment_identity.json")
    if (previous["status"] != "passed" or previous["identity"] != identity
            or identity["tournament_id"] != manifest["experiment_id"]
            or identity["engines"] != ENGINE_SETTINGS
            or identity["input_hashes"] != {name: file_sha256(path) for name, path in files.items()}):
        raise ValueError("Cache reference engine/model settings differ")
    resources = files["egaroucid_executable"].parent / "resources"
    if identity["egaroucid_resources"] != {name: file_sha256(resources / name) for name in identity["egaroucid_resources"]}:
        raise ValueError("Egaroucid resource files differ from cache reference")
    for name, expected in identity["implementation_sha256"].items():
        if file_sha256(BASE / "20_test_with_wthor" / name) != expected:
            raise ValueError(f"Cache reference implementation changed: {name}")
    origin = REFERENCE / "under_1400_test"
    sources = {str(path.relative_to(origin)): dict(path=str(path), sha256=file_sha256(path))
               for path in origin.rglob("*.sqlite3")}
    record = dict(reference=str(REFERENCE), reference_identity_sha256=file_sha256(REFERENCE / "experiment_identity.json"),
                  sources=sources, reuse="read-only SQLite backup; only missing states are searched",
                  note="Engine transposition tables are reused within a worker; prior search history may differ")
    if record_path.exists():
        if read(record_path) != record:
            raise ValueError("Cache source changed since this evaluation began")
        return record
    if len(sources) != 14:
        raise ValueError("Expected Egaroucid and Edax caches at all seven levels")
    for relative, evidence in sources.items():
        destination = output / "test" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("Unrecorded cache exists; choose a clean output directory")
        with closing(sqlite3.connect(Path(evidence["path"]).as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(destination)) as target:
                source.backup(target)
    write(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(sample_seed=613)
    parser.add_argument("--model", type=Path, default=DEFAULT_RUN / "selected_model.h5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "quest_baselines")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policy-batch-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 2:
        parser.error("--workers must be at least two")
    validate_model(args.model)
    dataset, provenance = load_quest(args.model)
    manifest, models, files = gpw.load_tournament(TOURNAMENT)
    identity = dict(dataset=provenance, tournament=manifest["experiment_id"], models=models,
                    engines=ENGINE_SETTINGS, seed=57, implementation_sha256=file_sha256(Path(__file__)))
    if args.dry_run:
        print(f"Verified {len(dataset['records'])} Quest test positions and 30-player roster", flush=True)
        return
    output = args.output_dir
    with gpw.OutputRunLock(output):
        identity_path = output / "experiment_identity.json"
        if identity_path.exists() and read(identity_path) != identity:
            raise ValueError("Existing evaluation identity differs")
        write(identity_path, identity)
        seed_caches(output, files, manifest)
        def status(state, stage, **extra):
            write(output / "run_status.json", dict(state=state, stage=stage, updated_at_unix=time.time(), **extra))
            print(stage, flush=True)
        try:
            rows = quest.evaluate_dataset(provenance["info"], output / "test", files, args, 57, status)
            if len(rows) != 30 or any(row["positions"] != 6692 for row in rows):
                raise ValueError("Incomplete 30-player test evaluation")
            write(output / "summary.json", dict(status="passed", identity=identity, results=rows,
                  predictions_sha256=file_sha256(output / "test/predictions.npz")))
            status("complete", "All 30 players verified on the 6692 Quest test positions")
        except BaseException as error:
            status("failed", "Evaluation interrupted; repeat the same command to resume", error=repr(error))
            raise


if __name__ == "__main__":
    main()
