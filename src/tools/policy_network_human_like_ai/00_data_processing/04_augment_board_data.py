#!/usr/bin/env python3
"""Increase every existing board split eightfold using rotations/reflections."""

import argparse
from contextlib import nullcontext

import numpy as np

from board_symmetry import TRANSFORM_NAMES, augment_positions
from pipeline_common import (BOARD_DTYPE, BoardWriter, SPLIT_NAMES, add_dataset_arguments,
    dataset_directory, load_manifest, output_directory, source_info, write_manifest,
    WEIGHT_DTYPE, has_sample_weights, validate_weight_file)


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--positions-per-batch", type=int, default=100000)
    parser.add_argument("--positions-per-file", type=int, default=1000000)
    return parser


def run(args):
    if args.positions_per_batch <= 0:
        raise ValueError("positions_per_batch must be positive")
    source = dataset_directory(args.data_root, "board_split", args.dataset)
    destination = dataset_directory(args.data_root, "board_split_increased", args.dataset)
    original = load_manifest(source, "board_split")
    weighted = has_sample_weights(original)
    counts, previous_games = {}, set()
    with output_directory(destination, args.replace) as output:
        for name in SPLIT_NAMES:
            writer = BoardWriter(output / name, args.positions_per_file, weighted=weighted)
            source_count, split_games = 0, set()
            try:
                for path in sorted((source / name).glob("*.dat")):
                    size = path.stat().st_size
                    ids_path = path.with_suffix(".game_ids")
                    if size % BOARD_DTYPE.itemsize or ids_path.stat().st_size != size // BOARD_DTYPE.itemsize * 8:
                        raise ValueError(f"incomplete board data or game IDs in {path}")
                    weights_path = path.with_suffix(".weights")
                    if weighted:
                        validate_weight_file(weights_path, size // BOARD_DTYPE.itemsize)
                    elif weights_path.exists():
                        raise ValueError("sample weights present without manifest metadata")
                    with (path.open("rb") as boards, ids_path.open("rb") as ids_file,
                          weights_path.open("rb") if weighted else nullcontext() as weights_file):
                        while True:
                            samples = np.fromfile(boards, dtype=BOARD_DTYPE, count=args.positions_per_batch)
                            if not len(samples):
                                break
                            game_ids = np.fromfile(ids_file, dtype="<u8", count=len(samples))
                            split_games.update(np.unique(game_ids).tolist())
                            augmented, expanded_ids = augment_positions(samples, game_ids)
                            weights = np.fromfile(weights_file, dtype=WEIGHT_DTYPE, count=len(samples)) if weighted else None
                            writer.write(augmented, expanded_ids, None if weights is None else np.repeat(weights, 8))
                            source_count += len(samples)
                    print(f"{name}: {source_count:,} source positions -> {writer.positions:,}", flush=True)
            finally:
                writer.close()
            if split_games & previous_games:
                raise ValueError("the same original game occurs in multiple splits")
            previous_games.update(split_games)
            expected = original["splits"][name]
            if source_count != expected["positions"] or len(split_games) != expected["games"]:
                raise ValueError(f"board input counts do not match manifest for {name}")
            if writer.positions != source_count * 8:
                raise ValueError("augmentation must produce exactly eight copies per position")
            counts[name] = {"games": len(split_games), "source_positions": source_count, "positions": writer.positions}
        result = write_manifest(output, "board_split_increased", args.dataset, source=source_info(source),
            seed=original["seed"], ratios=original["ratios"], split_unit=original["split_unit"],
            record_format=original["record_format"], record_bytes=19,
            game_ids_format=original["game_ids_format"], score_convention=original["score_convention"],
            augmentation_factor=8, transforms=TRANSFORM_NAMES,
            ordering="original position, then transform; symmetric duplicates are retained", splits=counts,
            **({"sample_weights": original["sample_weights"]} if weighted else {}))
    return result


if __name__ == "__main__":
    result = run(make_argparser().parse_args())
    print("board_split_increased:", result["splits"])
