"""Board loading and policy features, shared by preprocessing and training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wthor_dataset import (
    BOARD_DTYPE, load_game_dataset, load_manifest_splits, save_split_manifest,
    split_counts, split_game_positions,
)


BOARD_SAMPLE_SIZE = 19
INPUT_SIZE = 128
POLICY_SIZE = 64

assert BOARD_DTYPE.itemsize == BOARD_SAMPLE_SIZE

BIT_MASKS = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)).reshape(1, 64)
POLICY_BIT_MASKS = (np.uint64(1) << np.arange(0, 64, dtype=np.uint64)).reshape(1, 64)
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
NOT_FILE_A = np.uint64(0x7F7F7F7F7F7F7F7F)
NOT_FILE_H = np.uint64(0xFEFEFEFEFEFEFEFE)


@dataclass(frozen=True)
class BoardFile:
    path: Path
    n_position_samples: int


@dataclass(frozen=True)
class SplitArrays:
    x: np.ndarray
    policy: np.ndarray
    legal: np.ndarray
    player: np.ndarray
    opponent: np.ndarray
    n_discs: np.ndarray


def discover_board_files(data_root: Path, board_data_index_start: int, board_data_index_end: int) -> List[BoardFile]:
    files: List[BoardFile] = []
    for board_data_index in range(board_data_index_start, board_data_index_end + 1):
        board_data_dir = data_root / f"records{board_data_index}"
        if not board_data_dir.exists():
            raise FileNotFoundError(f"missing board data directory: {board_data_dir}")
        dat_files = sorted(board_data_dir.glob("*.dat"), key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
        if not dat_files:
            raise FileNotFoundError(f"no .dat files in {board_data_dir}")
        for path in dat_files:
            size = path.stat().st_size
            if size % BOARD_SAMPLE_SIZE != 0:
                raise ValueError(f"incomplete board sample in {path}")
            if size == 0:
                continue
            files.append(BoardFile(path=path, n_position_samples=size // BOARD_SAMPLE_SIZE))
    if not files:
        raise FileNotFoundError(f"no board data found under {data_root}")
    return files


def shift_east(x: np.ndarray) -> np.ndarray:
    return (x & NOT_FILE_H) >> np.uint64(1)


def shift_west(x: np.ndarray) -> np.ndarray:
    return (x & NOT_FILE_A) << np.uint64(1)


def shift_north(x: np.ndarray) -> np.ndarray:
    return x >> np.uint64(8)


def shift_south(x: np.ndarray) -> np.ndarray:
    return (x << np.uint64(8)) & FULL


def shift_ne(x: np.ndarray) -> np.ndarray:
    return (x & NOT_FILE_H) >> np.uint64(9)


def shift_nw(x: np.ndarray) -> np.ndarray:
    return (x & NOT_FILE_A) >> np.uint64(7)


def shift_se(x: np.ndarray) -> np.ndarray:
    return ((x & NOT_FILE_H) << np.uint64(7)) & FULL


def shift_sw(x: np.ndarray) -> np.ndarray:
    return ((x & NOT_FILE_A) << np.uint64(9)) & FULL


SHIFT_FUNCS = (shift_east, shift_west, shift_north, shift_south, shift_ne, shift_nw, shift_se, shift_sw)


def calc_legal_batch(player: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    empty = ~(player | opponent)
    legal = np.zeros_like(player, dtype=np.uint64)
    for shift in SHIFT_FUNCS:
        x = shift(player) & opponent
        for _ in range(5):
            x |= shift(x) & opponent
        legal |= shift(x) & empty
    return legal


def legal_bitboard_to_mask(legal: np.ndarray) -> np.ndarray:
    return (legal.reshape(-1, 1) & POLICY_BIT_MASKS) != 0


def valid_policy_mask(policies: np.ndarray, legal: np.ndarray) -> Tuple[np.ndarray, int, int]:
    valid_policy = (0 <= policies) & (policies < POLICY_SIZE)
    label_bits = np.zeros_like(legal, dtype=np.uint64)
    label_bits[valid_policy] = np.left_shift(np.uint64(1), policies[valid_policy].astype(np.uint64))
    legal_label = valid_policy & ((legal & label_bits) != 0)
    return legal_label, int(np.count_nonzero(~valid_policy)), int(np.count_nonzero(valid_policy & ~legal_label))


def position_samples_to_targets(position_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    player_bits = position_samples["player"].astype(np.uint64, copy=False)
    opponent_bits = position_samples["opponent"].astype(np.uint64, copy=False)
    n = len(position_samples)
    x = np.empty((n, INPUT_SIZE), dtype=np.float32)
    x[:, :64] = ((player_bits.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    x[:, 64:] = ((opponent_bits.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    y_policy = position_samples["policy"].astype(np.int64, copy=False)
    return x, y_policy


def position_samples_to_split_arrays(position_samples: np.ndarray) -> SplitArrays:
    player_bits = position_samples["player"].astype(np.uint64, copy=False)
    opponent_bits = position_samples["opponent"].astype(np.uint64, copy=False)
    x, y_policy = position_samples_to_targets(position_samples)
    legal = calc_legal_batch(player_bits, opponent_bits)
    n_discs = np.count_nonzero(x, axis=1).astype(np.int16, copy=False)
    return SplitArrays(
        x=x,
        policy=y_policy,
        legal=legal,
        player=player_bits,
        opponent=opponent_bits,
        n_discs=n_discs,
    )


def load_game_three_way_split(
    files: Sequence[BoardFile],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_position_samples: Optional[int],
    sample_caps: Optional[Dict[str, int]] = None,
    output_dir: Optional[Path] = None,
    split_manifest: Optional[Path] = None,
) -> Dict[str, SplitArrays]:
    paths = [board_file.path for board_file in files]
    if split_manifest is None:
        position_samples, game_ids = load_game_dataset(paths)
        indices, summary = split_game_positions(
            position_samples, game_ids, train_ratio, val_ratio, test_ratio,
            seed, max_position_samples, sample_caps,
        )
    else:
        if max_position_samples is not None or sample_caps:
            raise ValueError("prepared splits cannot be combined with position caps; set caps during preprocessing")
        indices, summary = load_manifest_splits(split_manifest, paths)
        position_samples = np.concatenate([np.fromfile(path, dtype=BOARD_DTYPE) for path in paths])
        # The prepared manifest determines partition membership and shuffle seed.
        seed = summary["seed"]
    print("dataset_split", json.dumps(summary))
    if output_dir is not None:
        save_split_manifest(output_dir, paths, indices, summary)
    # Shuffle only within each already assigned partition.
    rng = np.random.default_rng(seed)
    return {
        name: position_samples_to_split_arrays(position_samples[rng.permutation(selected)])
        for name, selected in indices.items()
    }


# Keep callers of the old entry point on the same game-level split.
load_shuffled_three_way_split = load_game_three_way_split


def load_prepared_board_splits(directory: Path, seed: int, output_dir: Optional[Path] = None) -> Dict[str, SplitArrays]:
    """Load stage 03/04 membership as-is; never split augmented positions again."""
    from pipeline_common import SPLIT_NAMES, load_manifest, source_info

    stage = json.loads((directory / "manifest.json").read_text(encoding="utf-8")).get("stage")
    if stage not in ("board_split", "board_split_increased"):
        raise ValueError("--board-split-dir must point to stage 03 or stage 04 output")
    manifest = load_manifest(directory, stage)
    results, previous_games = {}, set()
    rng = np.random.default_rng(seed)
    for name in SPLIT_NAMES:
        chunks, game_ids = [], set()
        for path in sorted((directory / name).glob("*.dat")):
            size = path.stat().st_size
            ids_path = path.with_suffix(".game_ids")
            if size % BOARD_SAMPLE_SIZE or ids_path.stat().st_size != size // BOARD_SAMPLE_SIZE * 8:
                raise ValueError(f"incomplete board data or game IDs in {path}")
            chunks.append(np.fromfile(path, dtype=BOARD_DTYPE))
            game_ids.update(np.unique(np.fromfile(ids_path, dtype="<u8")).tolist())
        if not chunks:
            raise ValueError(f"no board data in prepared split {name}")
        samples = np.concatenate(chunks)
        expected = manifest["splits"][name]
        if len(samples) != expected["positions"] or len(game_ids) != expected["games"]:
            raise ValueError(f"prepared split counts do not match for {name}")
        if game_ids & previous_games:
            raise ValueError("the same original game occurs in multiple prepared splits")
        previous_games.update(game_ids)
        results[name] = position_samples_to_split_arrays(samples[rng.permutation(len(samples))])
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prepared_dataset.json").write_text(
            json.dumps({"source": source_info(directory), "dataset": manifest}, indent=2), encoding="utf-8")
    return results
