"""Shared board readers and ranking helpers for the current WTHOR evaluator.

The former blend-evaluation CLI is archived in legacy. Keep this module name
for the current experiment and other launchers that import its data helpers.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "30_blend_with_egaroucid"))

from blend_policy_with_egaroucid import (  # noqa: E402
    ALL_LEGAL_HINT_COUNT, BLACK, WHITE, POLICY_SIZE, BoardState,
    default_egaroucid_exe, default_weights_file,
)
from evaluate_wthor_human_match import (  # noqa: E402
    BOARD_DTYPE, BOARD_SAMPLE_SIZE, equivalent_policy_mask,
)


def discover_dat_files(board_data_dir: Path) -> List[Path]:
    files = sorted(board_data_dir.glob("*.dat"), key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
    if not files:
        raise FileNotFoundError(f"no .dat files found in {board_data_dir}")
    return files


def count_position_samples(dat_files: Sequence[Path], max_positions: Optional[int]) -> int:
    total = 0
    for path in dat_files:
        n = path.stat().st_size // BOARD_SAMPLE_SIZE
        if max_positions is not None:
            n = min(n, max_positions - total)
        if n <= 0:
            break
        total += n
    return total


def choose_global_positions(total_positions: int, sample_positions: Optional[int], seed: int) -> Optional[np.ndarray]:
    if sample_positions is None:
        return None
    if total_positions <= 0:
        return np.empty(0, dtype=np.int64)
    n = min(sample_positions, total_positions)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_positions, size=n, replace=False).astype(np.int64, copy=False))


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("total split size must be positive")
    if ratio_sum <= 0.0:
        raise ValueError("train/validation/test ratios must have a positive sum")
    n_train = int(np.floor(total * (train_ratio / ratio_sum)))
    n_val = int(np.floor(total * (val_ratio / ratio_sum)))
    n_test = total - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"split produced an empty part: train={n_train} val={n_val} test={n_test}")
    return n_train, n_val, n_test


def choose_data_split_positions(
    total_positions: int,
    data_split: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    sample_positions: Optional[int],
    sample_seed: int,
) -> Tuple[Optional[np.ndarray], int]:
    if data_split == "all":
        return choose_global_positions(total_positions, sample_positions, sample_seed), total_positions

    n_train, n_val, n_test = split_counts(total_positions, train_ratio, val_ratio, test_ratio)
    order = np.random.default_rng(split_seed).permutation(total_positions)
    split_ranges = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }
    start, end = split_ranges[data_split]
    selected = order[start:end]
    split_positions = len(selected)
    if sample_positions is not None and sample_positions < split_positions:
        sample_rng = np.random.default_rng(sample_seed)
        selected = selected[sample_rng.choice(split_positions, size=sample_positions, replace=False)]
    return np.sort(selected.astype(np.int64, copy=False)), split_positions


class PositionSampleReader:
    def __init__(self, dat_files: Sequence[Path], available_positions: int):
        self.entries: List[Tuple[int, int, Path, int]] = []
        self.global_ends: List[int] = []
        self.mmaps: Dict[Path, np.memmap] = {}
        global_start = 0
        for path in dat_files:
            total_file_positions = path.stat().st_size // BOARD_SAMPLE_SIZE
            file_positions = min(
                total_file_positions,
                max(0, available_positions - global_start),
            )
            if file_positions <= 0:
                break
            global_end = global_start + file_positions
            self.entries.append(
                (global_start, global_end, path, total_file_positions)
            )
            self.global_ends.append(global_end)
            global_start = global_end

    def get(self, global_position: int):
        entry_index = int(
            np.searchsorted(self.global_ends, global_position, side="right")
        )
        if entry_index >= len(self.entries):
            raise IndexError(f"global position is out of range: {global_position}")
        global_start, global_end, path, total_file_positions = self.entries[
            entry_index
        ]
        if global_position < global_start or global_position >= global_end:
            raise IndexError(f"global position is out of range: {global_position}")
        mmap = self.mmaps.get(path)
        if mmap is None:
            mmap = np.memmap(
                path,
                dtype=BOARD_DTYPE,
                mode="r",
                shape=(total_file_positions,),
            )
            self.mmaps[path] = mmap
        return mmap[global_position - global_start]

    def close(self) -> None:
        self.mmaps.clear()


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def hint_cache_key(level: int, state: BoardState, side: int, hint_count: int) -> str:
    return (
        f"level={int(level)}:hint={int(hint_count)}:"
        f"{state.black:016x}:{state.white:016x}:{int(side)}"
    )


def position_sample_to_state(position_sample) -> Tuple[BoardState, int, int, int, int, int]:
    player = int(position_sample["player"])
    opponent = int(position_sample["opponent"])
    side = int(position_sample["color"])
    policy = int(position_sample["policy"])
    if side == BLACK:
        black, white = player, opponent
    elif side == WHITE:
        black, white = opponent, player
    else:
        raise ValueError(f"invalid side color: {side}")
    move_number = popcount(player | opponent) - 3
    return BoardState(black, white, side), side, policy, player, opponent, move_number


def rank_for_distribution(
    distribution: np.ndarray,
    legal_policies: Sequence[int],
    target_policies: Sequence[int],
    tie_break_order: Optional[Sequence[int]] = None,
) -> int:
    if not legal_policies:
        return POLICY_SIZE + 1
    legal_set = set(legal_policies)
    targets = set(policy for policy in target_policies if policy in legal_set)
    if not targets:
        return POLICY_SIZE + 1

    ordered_ties: List[int] = []
    seen = set()
    for policy in tie_break_order or ():
        if policy in legal_set and policy not in seen:
            ordered_ties.append(policy)
            seen.add(policy)
    for policy in legal_policies:
        if policy not in seen:
            ordered_ties.append(policy)
            seen.add(policy)
    tie_rank = {policy: rank for rank, policy in enumerate(ordered_ties)}
    ranked = sorted(
        legal_policies,
        key=lambda policy: (-float(distribution[policy]), tie_rank[policy]),
    )
    return min(rank for rank, policy in enumerate(ranked, start=1) if policy in targets)


def equivalent_targets(player: int, opponent: int, policy: int) -> List[int]:
    mask = equivalent_policy_mask(
        np.array([player], dtype=np.uint64),
        np.array([opponent], dtype=np.uint64),
        np.array([policy], dtype=np.int64),
    )[0]
    return [int(i) for i in np.nonzero(mask)[0]]
