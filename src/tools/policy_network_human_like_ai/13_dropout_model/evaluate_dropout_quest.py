"""Shared full-test Quest evaluator for inference-time dropout sweeps."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from sweep_common import BASE, BANDS, data_root, read, write, sha256, write_csv

sys.path.insert(0, str(BASE / "10_train_policy_network"))
from prepared_policy_data import PreparedSplit
from pipeline_common import load_manifest
from policy_data import (position_samples_to_split_arrays, legal_bitboard_to_mask,
                         valid_policy_mask)
sys.path.insert(0, str(BASE))
from policy_accuracy import equivalent_policy_mask, symmetry_aware_policy_ranks

BITS = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))


def inspect_band(root, band):
    directory = (root / "board_split" / band).resolve()
    manifest = load_manifest(directory, "board_split")
    low, high = map(int, band.split("-"))
    if (manifest.get("dataset") != band or manifest.get("augmentation_factor", 1) != 1
            or manifest.get("split_unit") != "identical_transcript_group_across_all_rating_bands"
            or manifest.get("rating_range") != dict(min_inclusive=low, max_exclusive=high)
            or manifest.get("sample_weights")):
        raise ValueError(f"Expected unaugmented, game-disjoint Quest data for {band}")
    paths = sorted((directory / "test").glob("*.dat"))
    expected = {entry["path"] for entry in manifest["files"] if entry["path"].startswith("test/")}
    actual = {path.relative_to(directory).as_posix() for path in (directory / "test").iterdir() if path.is_file()}
    if not paths or actual != expected:
        raise ValueError(f"Quest test files differ from the manifest: {band}")
    count = sum(path.stat().st_size // 19 for path in paths)
    if count != manifest["splits"]["test"]["positions"] or count <= 0:
        raise ValueError(f"Quest test position count differs: {band}")
    return paths, dict(rating_band=band, rating_range=manifest["rating_range"], positions=count,
        games=manifest["splits"]["test"]["games"], manifest_sha256=sha256(directory / "manifest.json"),
        test_files=[entry for entry in manifest["files"] if entry["path"].startswith("test/")])


def score_records(records, probabilities):
    arrays = position_samples_to_split_arrays(records)
    valid, invalid, illegal = valid_policy_mask(arrays.policy, arrays.legal)
    if invalid or illegal or not valid.all():
        raise ValueError("Invalid or illegal human move label in Quest test")
    if probabilities.shape != (len(records), 64) or not np.isfinite(probabilities).all():
        raise ValueError("Policy output must be finite and have shape (positions, 64)")
    legal = legal_bitboard_to_mask(arrays.legal)
    targets = equivalent_policy_mask(arrays.player, arrays.opponent, arrays.policy)
    masks = (targets.astype(np.uint64) * BITS).sum(axis=1)
    moves = np.argmax(np.where(legal, probabilities, -np.inf), axis=1).astype(np.int8)
    hits = (BITS[moves] & masks) != 0
    ranks = symmetry_aware_policy_ranks(probabilities, legal, targets)
    if not np.all(BITS[moves] & arrays.legal) or not np.array_equal(hits, ranks == 1):
        raise ValueError("Legality/ranking verification failed")
    return moves, masks, hits


def evaluate_band(model, paths, info, output, batch_size, interval, identity, reference_model=None):
    output.mkdir(parents=True, exist_ok=True)
    result_path, predictions_path = output / "summary.json", output / "predictions.npz"
    expected = dict(model=identity, dataset=info)
    if result_path.exists():
        saved = read(result_path)
        if (saved["identity"] != expected or saved["status"] != "complete"
                or sha256(predictions_path) != saved["predictions_sha256"]):
            raise ValueError(f"Saved evaluation changed: {output}")
        print(f"Reuse {info['rating_band']}: {saved['result']['top1_accuracy_percent']:.4f}%", flush=True)
        return saved["result"]
    split = PreparedSplit(paths)
    started, last_report, count, hits = time.monotonic(), 0.0, 0, 0
    moves_parts, target_parts, games_parts = [], [], []
    records_digest = hashlib.sha256()
    try:
        for records in split.batches(max(8192, batch_size)):
            features = position_samples_to_split_arrays(records).x
            probabilities = model.predict(features, batch_size=batch_size, verbose=0)
            moves, targets, matched = score_records(records, probabilities)
            if reference_model is not None:
                reference = reference_model.predict(features, batch_size=batch_size, verbose=0)
                expected_moves = score_records(records, reference)[0]
                if not np.array_equal(moves, expected_moves):
                    raise ValueError("Dropout zero must select the same legal moves as the original model")
            moves_parts.append(moves)
            target_parts.append(targets)
            records_digest.update(records.tobytes())
            count += len(records)
            hits += int(matched.sum())
            now = time.monotonic()
            if now - last_report >= interval or count == info["positions"]:
                elapsed = now - started
                eta = elapsed / count * (info["positions"] - count)
                write(output.parent / "evaluation_progress.json", dict(rating_band=info["rating_band"],
                    positions=count, total_positions=info["positions"], elapsed_sec=elapsed,
                    eta_sec=eta, updated_at_unix=time.time()))
                print(f"[Quest {info['rating_band']}] {count:,}/{info['positions']:,} | "
                      f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s", flush=True)
                last_report = now
        for path in paths:
            games_parts.append(np.fromfile(path.with_suffix(".game_ids"), dtype="<u8"))
        game_ids = np.concatenate(games_parts)
        moves, targets = np.concatenate(moves_parts), np.concatenate(target_parts)
        if count != info["positions"] or len(game_ids) != count or len(np.unique(game_ids)) != info["games"]:
            raise ValueError("Quest test count or game ID count differs from manifest")
        recounted = sum(bool(int(target) & (1 << int(move))) for target, move in zip(targets, moves))
        if recounted != hits:
            raise ValueError("Independent scalar hit recount differs")
        temporary = predictions_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, predicted_policies=moves, symmetry_target_masks=targets,
                            game_ids=game_ids)
        temporary.replace(predictions_path)
        with np.load(predictions_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["predicted_policies"], moves) or not np.array_equal(saved["symmetry_target_masks"], targets):
                raise ValueError("Saved predictions differ")
        row = dict(dropout_rate=identity["dropout_rate"], rating_band=info["rating_band"],
            positions=count, games=info["games"], top1_hits=hits, top1_accuracy=hits/count,
            top1_accuracy_percent=100*hits/count, adopted_epoch=identity["adopted_epoch"],
            test_records_sha256=records_digest.hexdigest(), model_sha256=identity["model_sha256"],
            inference_seed=identity.get("inference_seed", 613))
        write(result_path, dict(status="complete", identity=expected, result=row,
            predictions_sha256=sha256(predictions_path), elapsed_sec=time.monotonic()-started,
            agreement_definition="legal top-1; board-preserving symmetries; ascending policy index breaks ties"))
        print(f"[result] dropout={identity['dropout_rate']:g}, {info['rating_band']}: "
              f"{hits:,}/{count:,} = {row['top1_accuracy_percent']:.4f}%", flush=True)
        return row
    finally:
        split.close()


if __name__ == "__main__":
    from run_dropout_sweep import main
    main()
