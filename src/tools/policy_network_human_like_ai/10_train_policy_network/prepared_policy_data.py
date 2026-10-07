"""Bounded-memory readers for the existing train/val/test board partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_data_processing"))
from pipeline_common import (SPLIT_NAMES, dataset_directory, load_manifest, WEIGHT_DTYPE,
                             NONNEGATIVE_WEIGHT_FORMAT, has_sample_weights, validate_weight_file)
from policy_data import BOARD_DTYPE, BOARD_SAMPLE_SIZE, position_samples_to_targets
from wthor_dataset import file_sha256


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    default_root = os.environ.get("POLICY_DATA_ROOT")
    if not default_root and os.environ.get("EGAROUCID_DATA"):
        default_root = Path(os.environ["EGAROUCID_DATA"]) / "20261002_policy_network_human_like_data"
    parser.add_argument("--data-root", type=Path, default=default_root,
                        help="Pipeline root; defaults to POLICY_DATA_ROOT.")
    parser.add_argument("--dataset", default=os.environ.get("POLICY_DATASET", "WTHOR_1977_2024"),
                        help="Directory under board_split_increased; defaults to POLICY_DATASET or WTHOR_1977_2024.")
    parser.add_argument("--board-split-dir", type=Path,
                        help="Override with an existing stage 03/04 dataset directory.")
    for name in SPLIT_NAMES:
        parser.add_argument(f"--max-{name}-samples", type=int,
                            help=f"Optional reproducible sample cap within {name}; default: all positions.")


def resolve_board_split_dir(args: argparse.Namespace) -> Path:
    if args.board_split_dir is not None:
        return args.board_split_dir
    return dataset_directory(args.data_root, "board_split_increased", args.dataset)


class PreparedSplit:
    """Memory map compact records; expand only one batch into float features."""

    def __init__(self, paths: list[Path], cap: int | None = None, seed: int = 613, *, weighted=False):
        self.records = [np.memmap(path, dtype=BOARD_DTYPE, mode="r") for path in paths]
        self.weights = [np.memmap(path.with_suffix(".weights"), dtype=WEIGHT_DTYPE, mode="r") for path in paths] if weighted else []
        self.weight_scale = 1.0
        self.ends = np.cumsum([len(records) for records in self.records])
        self.total_positions = int(self.ends[-1])
        if cap is not None and cap <= 0:
            raise ValueError("sample caps must be positive")
        self.indices = None
        if cap is not None and cap < self.total_positions:
            self.indices = np.sort(np.random.default_rng(seed).choice(self.total_positions, cap, replace=False))
        self.positions = self.total_positions if self.indices is None else len(self.indices)

    def close(self) -> None:
        # tf.data may retain the generator closure; release mapped files explicitly.
        self.records.clear()
        self.weights.clear()

    def read(self, start: int, end: int) -> np.ndarray:
        return self._read(self.records, start, end)

    def read_weights(self, start: int, end: int) -> np.ndarray:
        return self._read(self.weights, start, end)

    def _read(self, arrays, start, end):
        if self.indices is None:
            first = int(np.searchsorted(self.ends, start, side="right"))
            parts = []
            for i in range(first, len(arrays)):
                offset = int(self.ends[i - 1]) if i else 0
                parts.append(arrays[i][max(0, start - offset):min(end, int(self.ends[i])) - offset])
                if end <= self.ends[i]:
                    break
            return np.concatenate(parts)
        selected = self.indices[start:end]
        result = np.empty(len(selected), dtype=arrays[0].dtype)
        shards = np.searchsorted(self.ends, selected, side="right")
        for i in np.unique(shards):
            mask = shards == i
            offset = int(self.ends[i - 1]) if i else 0
            result[mask] = arrays[i][selected[mask] - offset]
        return result

    def batches(self, batch_size: int, *, shuffle: bool = False, seed: int = 613, include_weights=False):
        # A block is a multiple of the batch size, so only the final block has
        # a partial batch. Shuffle both block order and positions in each block.
        block_size = max(1, 65536 // batch_size) * batch_size
        starts = np.arange(0, self.positions, block_size)
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(starts)
        for start in starts:
            end = min(int(start) + block_size, self.positions)
            samples = self.read(int(start), end)
            weights = self.read_weights(int(start), end) if include_weights and self.weights else None
            if shuffle:
                order = rng.permutation(len(samples))
                samples = samples[order]
                if weights is not None:
                    weights = weights[order]
            for offset in range(0, len(samples), batch_size):
                batch = samples[offset:offset + batch_size]
                if include_weights:
                    yield batch, None if weights is None else weights[offset:offset + batch_size]
                else:
                    yield batch

    def as_dataset(self, batch_size: int, *, shuffle: bool = False, seed: int = 613, initial_epoch: int = 0):
        import tensorflow as tf

        epoch = initial_epoch

        def generate():
            nonlocal epoch
            iteration_seed = seed + epoch
            epoch += 1
            for samples, weights in self.batches(batch_size, shuffle=shuffle, seed=iteration_seed, include_weights=True):
                targets = position_samples_to_targets(samples)
                yield targets if weights is None else (*targets, (weights / self.weight_scale).astype(np.float32))

        signature = (tf.TensorSpec((None, 128), tf.float32), tf.TensorSpec((None,), tf.int64))
        if self.weights:
            signature += (tf.TensorSpec((None,), tf.float32),)
        data = tf.data.Dataset.from_generator(generate, output_signature=signature)
        count = (self.positions + batch_size - 1) // batch_size
        return data.apply(tf.data.experimental.assert_cardinality(count)).prefetch(2)


def load_prepared_splits(directory: Path, caps: dict, seed: int, output_dir: Path):
    stage = json.loads((directory / "manifest.json").read_text(encoding="utf-8")).get("stage")
    if stage not in ("board_split", "board_split_increased"):
        raise ValueError("dataset must be stage 03/04 output")
    print("Checking prepared dataset checksums and game partitions...", flush=True)
    manifest = load_manifest(directory, stage)
    weighted = has_sample_weights(manifest)
    allow_zero = weighted and manifest["sample_weights"]["format"] == NONNEGATIVE_WEIGHT_FORMAT
    listed = {entry["path"] for entry in manifest["files"]}
    splits, previous_games, used = {}, set(), {}
    try:
        for index, name in enumerate(SPLIT_NAMES):
            paths = sorted((directory / name).glob("*.dat"))
            actual = {p.relative_to(directory).as_posix() for p in (directory / name).iterdir() if p.is_file()}
            expected_files = {p for p in listed if p.startswith(name + "/")}
            if not paths or actual != expected_files:
                raise ValueError(f"prepared files do not match manifest for {name}")
            games, positions, weight_sum = set(), 0, 0.0
            for path in paths:
                size = path.stat().st_size
                ids_path = path.with_suffix(".game_ids")
                if not size or size % BOARD_SAMPLE_SIZE or ids_path.stat().st_size != size // BOARD_SAMPLE_SIZE * 8:
                    raise ValueError(f"incomplete board data or game IDs in {path.name}")
                positions += size // BOARD_SAMPLE_SIZE
                weights_path = path.with_suffix(".weights")
                if weighted:
                    weight_sum += validate_weight_file(weights_path, size // BOARD_SAMPLE_SIZE, allow_zero=allow_zero)
                elif weights_path.exists():
                    raise ValueError("sample weights present without manifest metadata")
                ids = np.memmap(ids_path, dtype="<u8", mode="r")
                for start in range(0, len(ids), 65536):
                    games.update(np.unique(ids[start:start + 65536]).tolist())
                del ids
            expected = manifest["splits"][name]
            if positions != expected["positions"] or len(games) != expected["games"]:
                raise ValueError(f"prepared split counts do not match for {name}")
            if games & previous_games:
                raise ValueError("the same original game occurs in multiple prepared splits")
            previous_games.update(games)
            split = PreparedSplit(paths, caps.get(name), seed + index, weighted=weighted)
            splits[name] = split
            used[name] = {"available_positions": positions, "used_positions": split.positions,
                          "games": len(games), "sample_seed": seed + index,
                          "sample_indices_sha256": None if split.indices is None else
                          hashlib.sha256(split.indices.astype("<i8").tobytes()).hexdigest()}
            if weighted:
                used[name]["raw_weight_sum"] = weight_sum
            print(f"{name}: {split.positions:,} / {positions:,} positions, {len(games):,} games", flush=True)
        if weighted:
            # Keep every ratio intact while avoiding an enormous loss scale.
            # Only train determines the scale; val/test do not affect training.
            scale = used["train"]["raw_weight_sum"] / used["train"]["available_positions"]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("training sample weights must have a finite positive mean")
            for split in splits.values():
                split.weight_scale = scale
    except BaseException:
        for split in splits.values():
            split.close()
        raise
    # Store portable provenance, without copying machine-specific source paths.
    metadata = {"dataset": manifest["dataset"], "stage": stage,
                "manifest_sha256": file_sha256(directory / "manifest.json"),
                "augmentation_factor": manifest.get("augmentation_factor", 1),
                "seed": seed, "splits": used, "files": manifest["files"]}
    if weighted:
        metadata["sample_weights"] = {**manifest["sample_weights"],
                                      "training_normalization": "divide all splits by full train mean",
                                      "training_divisor": scale}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prepared_dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return splits
