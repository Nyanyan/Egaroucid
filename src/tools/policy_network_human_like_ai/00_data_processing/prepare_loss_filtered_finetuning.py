"""Select uniformly weighted WTHOR mistakes; augment only the training split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analysis_pipeline import Progress
from board_symmetry import TRANSFORM_NAMES, augment_positions
from pipeline_common import (BOARD_DTYPE, BoardWriter, SPLIT_NAMES, WEIGHT_DTYPE,
                             WEIGHT_FORMAT, load_manifest, output_directory,
                             source_info, write_manifest)
from wthor_dataset import file_sha256


def decode_losses(weights):
    """Invert the recorded, unclipped exp2 weights without using them in training."""
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights < 1):
        raise ValueError("Expected finite, unnormalized exp2 loss weights")
    losses = np.rint(2 * np.log2(weights))
    if (np.any(losses < 0) or np.any(losses > 128)
            or not np.allclose(weights, np.exp2(losses / 2), rtol=1e-12, atol=0)):
        raise ValueError("Weights do not encode integer disc losses in [0, 128]")
    return losses.astype(np.uint8)


def run(args):
    if not 1 <= args.min_loss <= 128 or args.positions_per_batch < 1:
        raise ValueError("Invalid loss threshold or batch size")
    root = args.data_root.resolve()
    ordinary = root / "board_split" / args.dataset
    weighted = root / "board_split" / (args.dataset + "_LOSS_WEIGHTED")
    stats_path = root / "cache" / (args.dataset + "_LOSS_WEIGHTED") / "statistics/loss_distribution.json"
    print("Validating source board manifests/checksums and loss statistics...", flush=True)
    original = load_manifest(ordinary, "board_split")
    source = load_manifest(weighted, "board_split")
    statistics = json.loads(stats_path.read_text(encoding="utf-8"))
    spec = source.get("sample_weights", {})
    if (spec.get("format") != WEIGHT_FORMAT or spec.get("formula") != "2 ** (disc_loss / 2)"
            or spec.get("normalization") != "none in files" or spec.get("clipping") is not None):
        raise ValueError("Only original, unclipped exp2 weights can identify disc losses")
    if (original["dataset"] != args.dataset or source["source"] != original["source"]
            or original["source"] != statistics["source"] or statistics["dataset"] != args.dataset
            or source["analysis"]["configuration"] != statistics["configuration"]
            or source["analysis"]["level"] != 10
            or source["split_unit"] != "identical_transcript_group"):
        raise ValueError("Board, transcript, analysis or split provenance differs")
    # Weighted and ordinary records must have identical ordering and game IDs.
    weighted_files = {entry["path"]: entry for entry in source["files"]}
    for entry in original["files"]:
        if entry["path"].endswith((".dat", ".game_ids")) and weighted_files.get(entry["path"]) != entry:
            raise ValueError("Weighted records/IDs differ from ordinary WTHOR: " + entry["path"])
    counts, previous_games = {}, set()
    total = sum(source["splits"][name]["positions"] for name in SPLIT_NAMES)
    progress, done = Progress("filter + augment", total, args.progress_seconds), 0
    progress.show(0, "train x8; validation/test x1; uniform sample weights", force=True)
    with output_directory(args.output_dir) as output:
        for name in SPLIT_NAMES:
            writer = BoardWriter(output / name, 1_000_000)
            histogram = np.zeros(129, dtype=np.int64)
            source_games, selected_games = set(), set()
            source_count = selected_count = 0
            factor = 8 if name == "train" else 1
            try:
                for path in sorted((weighted / name).glob("*.dat")):
                    boards = np.memmap(path, mode="r", dtype=BOARD_DTYPE)
                    ids = np.memmap(path.with_suffix(".game_ids"), mode="r", dtype="<u8")
                    weights = np.memmap(path.with_suffix(".weights"), mode="r", dtype=WEIGHT_DTYPE)
                    if len(boards) != len(ids) or len(ids) != len(weights):
                        raise ValueError("Record/ID/weight alignment differs")
                    for start in range(0, len(boards), args.positions_per_batch):
                        end = min(start + args.positions_per_batch, len(boards))
                        losses = decode_losses(weights[start:end])
                        histogram += np.bincount(losses, minlength=129)
                        keep = losses >= args.min_loss
                        game_ids = np.asarray(ids[start:end])
                        source_games.update(np.unique(game_ids).tolist())
                        selected_games.update(np.unique(game_ids[keep]).tolist())
                        samples, kept_ids = np.asarray(boards[start:end])[keep], game_ids[keep]
                        selected_count += len(samples)
                        if factor == 8:
                            samples, kept_ids = augment_positions(samples, kept_ids)
                        writer.write(samples, kept_ids)
                        source_count += end - start
                        done += end - start
                        progress.show(done, f"{name}: {selected_count:,} selected -> {writer.positions:,} records")
                    del boards, ids, weights
            finally:
                writer.close()
            expected = source["splits"][name]
            if source_count != expected["positions"] or len(source_games) != expected["games"]:
                raise ValueError("Source counts differ from manifest")
            if source_games & previous_games:
                raise ValueError("Game IDs overlap between splits")
            previous_games.update(source_games)
            if histogram.tolist() != [entry["count"] for entry in statistics["distribution"][name]["bins"]]:
                raise ValueError("Decoded losses differ from saved analysis histogram")
            if not selected_count or writer.positions != factor * selected_count:
                raise ValueError("Empty split or incorrect augmentation count")
            counts[name] = dict(games=len(selected_games), source_positions=source_count,
                                selected_source_positions=selected_count, positions=writer.positions,
                                augmentation_factor=factor)
            print(f"{name}: {json.dumps(counts[name])}", flush=True)
        print("Saving output checksums and provenance...", flush=True)
        result = write_manifest(output, "board_split_increased", f"{args.dataset}_MISS_{args.min_loss}_FT",
            source_dataset=args.dataset, source=source_info(ordinary),
            loss_source=source_info(weighted), loss_statistics_sha256=file_sha256(stats_path),
            analysis=source["analysis"], seed=original["seed"], ratios=original["ratios"],
            split_unit=original["split_unit"], record_format=original["record_format"], record_bytes=19,
            game_ids_format=original["game_ids_format"], score_convention=original["score_convention"],
            selection=dict(minimum_loss=args.min_loss, comparison=">=", policy="original human move",
                           loss=statistics["loss_formula"],
                           method="invert validated exp2(disc_loss/2) sidecars; match all split histograms"),
            sample_weighting="uniform; source weights used only to select records",
            augmentation_factor=dict(train=8, val=1, test=1), transforms=TRANSFORM_NAMES,
            ordering="original position, then transform; symmetric duplicates are retained", splits=counts,
            preprocessing_sha256=file_sha256(Path(__file__)))
    progress.show(done, f"saved {args.output_dir}", force=True)
    return result


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = os.environ.get("POLICY_DATA_ROOT")
    if not data_root and os.environ.get("EGAROUCID_DATA"):
        data_root = Path(os.environ["EGAROUCID_DATA"]) / "20261002_policy_network_human_like_data"
    parser.add_argument("--data-root", type=Path, default=data_root, required=not bool(data_root))
    parser.add_argument("--dataset", default=os.environ.get("POLICY_DATASET", "WTHOR_1977_2024"))
    parser.add_argument("--min-loss", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positions-per-batch", type=int, default=100_000)
    parser.add_argument("--progress-seconds", type=float, default=5)
    return parser


if __name__ == "__main__":
    run(make_argparser().parse_args())
