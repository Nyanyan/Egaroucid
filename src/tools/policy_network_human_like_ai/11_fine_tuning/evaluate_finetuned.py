"""Verify and evaluate the fixed model on Quest test and the saved WTHOR sample."""
import argparse
from pathlib import Path
import time

from experiment_common import (DEFAULT_RUN, DEFAULT_RESULTS, MODEL_ID, WTHOR_REFERENCE,
                               TOURNAMENT, read, write, validate_model, file_sha256)
import numpy as np
import evaluate_gpw_human_match as gpw
import evaluate_quest_gpw_human_match as quest
from evaluate_loss_plus_one_blends import records_from_groups, record_key
from evaluate_wthor_blend_from_cache import load_reference, load_reference_sample
from pipeline_common import load_manifest
from policy_accuracy import symmetry_aware_policy_ranks


def load_quest(model):
    identity = read(model.parent / "training_identity.json")
    directory = Path(identity["source_directory"])
    manifest = load_manifest(directory, "board_split")
    if manifest["rating_range"] != dict(min_inclusive=None, max_exclusive=800):
        raise ValueError("Incorrect Quest rating range")
    info = quest.inspect_datasets(directory, ["test"], None)[0]
    dataset, sample = quest.load_dataset(info, 613)
    dataset["name"] = "quest_under800_test"
    if len(dataset["records"]) != 6692:
        raise ValueError("Expected all 6692 held-out Quest positions")
    return dataset, dict(info=info, sample=sample, manifest_sha256=file_sha256(directory / "manifest.json"))


def load_wthor():
    reference = read(WTHOR_REFERENCE / "summary.json")
    identity = read(WTHOR_REFERENCE / "experiment_identity.json")
    tournament, _, _ = gpw.load_tournament(TOURNAMENT)
    if (reference["status"] != "passed" or reference["mode"] != "full" or reference["identity"] != identity
            or identity["tournament_id"] != tournament["experiment_id"]):
        raise ValueError("WTHOR reference differs from the verified GPW30 tournament")
    provenance = identity["provenance"]
    original = Path(provenance["sample_reference"])
    summary, sample_identity, _ = load_reference(original)
    if sample_identity != provenance["sample_identity"] or sample_identity["positions"] != 100000:
        raise ValueError("WTHOR sample identity differs")
    directory = Path(summary["board_data_dir"])
    if directory.name != "test" or file_sha256(directory.parent / "manifest.json") != provenance["source_test_manifest_sha256"]:
        raise ValueError("WTHOR test manifest differs")
    groups = load_reference_sample(summary, sample_identity, directory)
    records = records_from_groups(groups)
    dataset = dict(name="wthor_test_100000", records=records, groups=groups, losses=None)
    path = WTHOR_REFERENCE / "wthor_test_100000_predictions.npz"
    if file_sha256(path) != reference["prediction_files"][path.name]:
        raise ValueError("WTHOR saved predictions changed")
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(records, saved["records"]) or tuple(saved["model_names"]) != gpw.MODEL_NAMES:
            raise ValueError("WTHOR sample/roster mismatch")
        old_rows = gpw.summarize_and_verify(dataset, saved["predicted_policies"], saved["symmetry_target_masks"])
    if old_rows != [r for r in reference["results"] if r["dataset"] == dataset["name"]]:
        raise ValueError("Recounted WTHOR reference results differ")
    return dataset, dict(sample=sample_identity, reference_predictions_sha256=file_sha256(path),
                         reference_summary_sha256=file_sha256(WTHOR_REFERENCE / "summary.json")), old_rows


def evaluate(dataset, provenance, model, output, old_rows=()):
    output.mkdir(parents=True, exist_ok=True)
    evidence = validate_model(model)
    identity = dict(model=evidence, dataset=provenance, implementation_sha256=file_sha256(Path(__file__)))
    summary_path = output / "summary.json"
    if summary_path.exists():
        saved = read(summary_path)
        if saved["identity"] != identity or file_sha256(output / "predictions.npz") != saved["predictions_sha256"]:
            raise ValueError("Existing evaluation identity differs")
        print(f"Reuse verified {dataset['name']}: {saved['result']['top1_accuracy_percent']:.4f}%", flush=True)
        return saved
    groups, records = dataset["groups"], dataset["records"]
    print(f"Infer {dataset['name']}: {len(groups):,} states / {len(records):,} positions", flush=True)
    prepared = gpw.prepare_position_groups(groups)
    if prepared.invalid_policy_samples or prepared.illegal_label_samples:
        raise ValueError("Invalid human move labels")
    logits = gpw.predict_policy_logits(groups, None, 512, model)
    dense = np.stack([logits[record_key(row)] for row in records])
    bits = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
    legal_bits = gpw.calc_legal_batch(records["player"], records["opponent"])
    legal = (legal_bits[:, None] & bits) != 0
    equivalent = gpw.equivalent_policy_mask(records["player"], records["opponent"], records["policy"].astype(np.int64))
    targets = (equivalent.astype(np.uint64) * bits).sum(axis=1)
    if not np.isfinite(dense[legal]).all():
        raise ValueError("Nonfinite legal move scores")
    moves = np.argmax(np.where(legal, dense, -np.inf), axis=1).astype(np.int8)
    ranks = symmetry_aware_policy_ranks(dense, legal, equivalent)
    hits = (bits[moves] & targets) != 0
    recounted = sum(bool(int(targets[i]) & (1 << int(move))) for i, move in enumerate(moves))
    if not np.all(bits[moves] & legal_bits) or not np.array_equal(ranks == 1, hits) or int(hits.sum()) != recounted:
        raise ValueError("Independent legality/rank/hit verification failed")
    n, count = len(records), int(hits.sum())
    low, high = gpw.wilson_interval(count, n)
    row = dict(model=MODEL_ID, dataset=dataset["name"], minimum_loss=None, positions=n,
               top1_hits=count, top1_accuracy=count/n, top1_accuracy_percent=100*count/n,
               top1_ci95_lower=low, top1_ci95_upper=high, exact_top1_hits=int((moves == records["policy"]).sum()),
               exact_top1_accuracy=float((moves == records["policy"]).mean()), expected_random_accuracy=None)
    np.savez_compressed(output / "predictions.npz", records=records, predicted_policies=moves,
                        symmetry_target_masks=targets, ranks=ranks, model_name=MODEL_ID)
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["records"], records) or not np.array_equal(saved["predicted_policies"], moves):
            raise ValueError("Saved predictions differ")
    if validate_model(model) != evidence:
        raise ValueError("Model changed during evaluation")
    result = dict(status="passed", identity=identity, result=row,
                  comparison=list(old_rows) + [row], predictions_sha256=file_sha256(output / "predictions.npz"),
                  agreement_definition="Legal Top-1, board-preserving symmetries; ascending policy index resolves ties",
                  confidence_interval="95% Wilson over positions; within-game correlation not corrected")
    write(summary_path, result)
    gpw.write_csv(output / "summary.csv", result["comparison"], tuple(row))
    print(f"{dataset['name']}: {count}/{n} = {100*count/n:.4f}%", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_RUN / "selected_model.h5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    validate_model(args.model)
    with gpw.OutputRunLock(args.output_dir / "new_model_evaluation"):
        dataset, provenance = load_quest(args.model)
        evaluate(dataset, provenance, args.model, args.output_dir / "quest_new")
        dataset, provenance, old_rows = load_wthor()
        evaluate(dataset, provenance, args.model, args.output_dir / "wthor", old_rows)


if __name__ == "__main__":
    main()
