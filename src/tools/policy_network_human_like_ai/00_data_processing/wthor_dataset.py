"""Game-level splits for ordered 19-byte policy board data (no TensorFlow)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


BOARD_DTYPE = np.dtype([
    ("player", "<u8"), ("opponent", "<u8"), ("color", "i1"),
    ("policy", "i1"), ("score", "i1"),
])
SPLIT_NAMES = ("train", "val", "test")


def game_ids_path(board_path: Path) -> Path:
    return board_path.with_suffix(".game_ids")


def infer_ordered_game_ids(samples: np.ndarray) -> np.ndarray:
    """Recover WTHOR games before filtering, across .dat file boundaries.

    Only full-opening, chronological data is supported without explicit IDs.
    Passes have no sample of their own in expand_transcript.cpp.
    """
    if not len(samples):
        raise ValueError("no position samples loaded")
    initial_player = np.uint64((1 << 28) | (1 << 35))
    initial_opponent = np.uint64((1 << 27) | (1 << 36))
    starts = (
        (samples["player"] == initial_player)
        & (samples["opponent"] == initial_opponent)
        & (samples["color"] == 0)
    )
    if not starts[0]:
        raise ValueError(
            "board data must start at the initial position to recover game IDs; "
            "use WTHOR board data including each game's opening, or provide .game_ids files"
        )
    occupied = samples["player"] | samples["opponent"]
    policy = samples["policy"][:-1]
    valid = (policy >= 0) & (policy < 64)
    move_bits = np.left_shift(np.uint64(1), np.clip(policy, 0, 63).astype(np.uint64))
    consecutive = (
        valid
        & ((occupied[:-1] & move_bits) == 0)
        & (occupied[1:] == (occupied[:-1] | move_bits))
    )
    broken = np.flatnonzero(~starts[1:] & ~consecutive)
    if len(broken):
        raise ValueError(
            f"cannot recover game boundary at position {int(broken[0]) + 1}; "
            "use chronological games starting at the initial position, or .game_ids files"
        )
    return np.cumsum(starts, dtype=np.int64) - 1


def load_game_dataset(paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load game IDs scoped to each records directory; never split at file edges."""
    groups: Dict[Path, list] = {}
    for path in paths:
        groups.setdefault(path.parent, []).append(path)
    # Keep global position indices identical to the input file order.
    if list(paths) != [path for group in groups.values() for path in group]:
        raise ValueError("board files from each records directory must be consecutive")
    all_samples = []
    all_ids = []
    next_game_id = 0
    for group in groups.values():
        chunks = []
        id_chunks = []
        have_ids = [game_ids_path(path).exists() for path in group]
        if any(have_ids) and not all(have_ids):
            raise ValueError(f"missing .game_ids files in {group[0].parent}; regenerate the dataset")
        for path in group:
            size = path.stat().st_size
            if size % BOARD_DTYPE.itemsize:
                raise ValueError(f"incomplete board sample in {path}")
            samples = np.fromfile(path, dtype=BOARD_DTYPE)
            chunks.append(samples)
            if all(have_ids):
                ids_path = game_ids_path(path)
                if ids_path.stat().st_size != len(samples) * 8:
                    raise ValueError(f"game ID count does not match board samples: {ids_path}")
                id_chunks.append(np.fromfile(ids_path, dtype="<u8"))
        samples = np.concatenate(chunks)
        if not len(samples):
            continue
        if all(have_ids):
            _, ids = np.unique(np.concatenate(id_chunks), return_inverse=True)
        else:
            ids = infer_ordered_game_ids(samples)
        all_samples.append(samples)
        all_ids.append(ids + next_game_id)
        next_game_id += int(ids.max()) + 1
    if not all_samples:
        raise ValueError("no position samples loaded")
    return np.concatenate(all_samples), np.concatenate(all_ids)


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("all split ratios must be finite and positive")
    ratios /= ratios.sum()
    # Hold out floor(5%) each by default; all rounding remainder goes to train.
    n_val = int(np.floor(total * ratios[1]))
    n_test = int(np.floor(total * ratios[2]))
    n_train = total - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"too few games: split produced an empty part: train={n_train} val={n_val} test={n_test}"
        )
    return n_train, n_val, n_test


def split_game_positions(
    samples: np.ndarray,
    game_ids: np.ndarray,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 613,
    max_position_samples: Optional[int] = None,
    sample_caps: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, np.ndarray], dict]:
    if len(samples) != len(game_ids):
        raise ValueError("game ID count does not match board samples")
    valid = (samples["policy"] >= 0) & (samples["policy"] < 64)
    # Identify games before filtering invalid labels, so filtering cannot move a boundary.
    unique_ids, first, inverse, counts = np.unique(
        game_ids, return_index=True, return_inverse=True, return_counts=True,
    )
    eligible = np.zeros(len(unique_ids), dtype=bool)
    eligible[inverse[valid]] = True
    if max_position_samples is not None:
        if max_position_samples <= 0:
            raise ValueError("max_position_samples must be positive")
        # Keep whole games in source order, including games spanning multiple files.
        ordered = np.argsort(first)
        ordered = ordered[eligible[ordered]]
        eligible[:] = False
        eligible[ordered[np.cumsum(counts[ordered]) <= max_position_samples]] = True
    games = np.flatnonzero(eligible)
    n_train, n_val, _ = split_counts(len(games), train_ratio, val_ratio, test_ratio)
    order = np.random.default_rng(seed).permutation(games)
    assigned = np.full(len(unique_ids), -1, dtype=np.int8)
    assigned[order[:n_train]] = 0
    assigned[order[n_train:n_train + n_val]] = 1
    assigned[order[n_train + n_val:]] = 2
    position_split = assigned[inverse]
    indices = {}
    summary = {
        "unit": "game",
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "rounding": "floor_validation_and_test_remainder_to_train",
        "max_position_samples": max_position_samples,
        "available_games": len(unique_ids),
        "games": len(games),
        "available_position_samples": len(samples),
        "invalid_policy_samples": int(np.count_nonzero(~valid)),
        "splits": {},
    }
    for split_id, name in enumerate(SPLIT_NAMES):
        selected = np.flatnonzero(valid & (position_split == split_id))
        available = len(selected)
        cap = (sample_caps or {}).get(name)
        if cap is not None:
            if cap <= 0:
                raise ValueError(f"{name} sample cap must be positive")
            if cap < len(selected):
                selected = np.sort(np.random.default_rng(seed + split_id + 1).choice(selected, cap, replace=False))
        indices[name] = selected
        summary["splits"][name] = {
            "games": int(np.count_nonzero(assigned == split_id)),
            "available_position_samples": available,
            "position_samples": len(selected),
        }
    return indices, summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_split_manifest(output_dir: Path, paths: Sequence[Path], indices: Dict[str, np.ndarray], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "dataset_split.npz"
    np.savez_compressed(index_path, **indices)
    manifest = dict(summary)
    manifest["indices_file"] = index_path.name
    manifest["indices_sha256"] = file_sha256(index_path)
    manifest["files"] = [
        {"path": str(path.resolve()), "sha256": file_sha256(path), "size": path.stat().st_size}
        for path in paths
    ]
    (output_dir / "dataset_split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest_positions(manifest_path: Path, paths: Sequence[Path], split_name: str) -> Tuple[np.ndarray, dict]:
    """Reuse training membership and reject a different/reordered dataset."""
    if split_name not in SPLIT_NAMES:
        raise ValueError("--split-manifest requires --data-split train, val, or test")
    indices, manifest = load_manifest_splits(manifest_path, paths)
    return indices[split_name], manifest


def load_manifest_splits(manifest_path: Path, paths: Sequence[Path]) -> Tuple[Dict[str, np.ndarray], dict]:
    """Load all prepared partitions after checking the input and index files."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("unit") != "game":
        raise ValueError("expected a game-level split manifest")
    if len(paths) != len(manifest["files"]):
        raise ValueError("split manifest board-file count mismatch")
    for path, source in zip(paths, manifest["files"]):
        if path.stat().st_size != source["size"] or file_sha256(path) != source["sha256"]:
            raise ValueError(f"split manifest board data mismatch: {path}")
    index_path = manifest_path.parent / manifest["indices_file"]
    if file_sha256(index_path) != manifest["indices_sha256"]:
        raise ValueError("split manifest indices checksum mismatch")
    with np.load(index_path, allow_pickle=False) as archive:
        indices = {name: archive[name] for name in SPLIT_NAMES}
    return indices, manifest
