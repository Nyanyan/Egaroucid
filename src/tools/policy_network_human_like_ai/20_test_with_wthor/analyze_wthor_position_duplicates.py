#!/usr/bin/env python3
"""
Count duplicate WTHOR position samples for Egaroucid hint-cache planning.

The key is the exact Egaroucid hint input: black bitboard, white bitboard, and
side to move. This is different from the side-relative policy-network input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BLEND_DIR = SCRIPT_DIR.parents[0] / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import BLACK, WHITE  # noqa: E402
from evaluate_wthor_human_match import BOARD_DTYPE, BOARD_SAMPLE_SIZE  # noqa: E402
from evaluate_wthor_blend_human_match import default_board_data_dir, discover_dat_files  # noqa: E402


HINT_KEY_DTYPE = np.dtype(
    [
        ("black", "<u8"),
        ("white", "<u8"),
        ("side", "i1"),
    ],
    align=False,
)


def rss_mib() -> Optional[float]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def update_peak(peak: Optional[float]) -> Optional[float]:
    current = rss_mib()
    if current is None:
        return peak
    return current if peak is None else max(peak, current)


def position_samples_to_hint_keys(position_samples: np.ndarray) -> tuple[np.ndarray, int]:
    player = position_samples["player"].astype(np.uint64, copy=False)
    opponent = position_samples["opponent"].astype(np.uint64, copy=False)
    side = position_samples["color"].astype(np.int8, copy=False)
    valid_side = (side == BLACK) | (side == WHITE)

    keys = np.empty(len(position_samples), dtype=HINT_KEY_DTYPE)
    black_to_move = side == BLACK
    keys["black"] = np.where(black_to_move, player, opponent)
    keys["white"] = np.where(black_to_move, opponent, player)
    keys["side"] = side
    return keys, int(np.count_nonzero(~valid_side))


def collect_hint_keys(dat_files: Sequence[Path], batch_size: int, max_positions: Optional[int]) -> tuple[np.ndarray, int, list[dict], Optional[float]]:
    chunks: List[np.ndarray] = []
    per_file = []
    total_position_samples = 0
    invalid_side_samples = 0
    peak = update_peak(None)
    for dat_file in dat_files:
        file_position_samples = dat_file.stat().st_size // BOARD_SAMPLE_SIZE
        if max_positions is not None:
            file_position_samples = min(file_position_samples, max(0, max_positions - total_position_samples))
        if file_position_samples <= 0:
            break
        mmap = np.memmap(dat_file, dtype=BOARD_DTYPE, mode="r", shape=(dat_file.stat().st_size // BOARD_SAMPLE_SIZE,))
        file_invalid_side_samples = 0
        for start in range(0, file_position_samples, batch_size):
            end = min(file_position_samples, start + batch_size)
            keys, bad_side = position_samples_to_hint_keys(np.asarray(mmap[start:end]))
            chunks.append(keys.copy())
            file_invalid_side_samples += bad_side
            peak = update_peak(peak)
        total_position_samples += file_position_samples
        invalid_side_samples += file_invalid_side_samples
        per_file.append(
            {
                "path": str(dat_file),
                "position_samples": int(file_position_samples),
                "invalid_side_samples": int(file_invalid_side_samples),
            }
        )
    if chunks:
        keys = np.concatenate(chunks)
    else:
        keys = np.empty(0, dtype=HINT_KEY_DTYPE)
    peak = update_peak(peak)
    return keys, invalid_side_samples, per_file, peak


def analyze(args: argparse.Namespace) -> dict:
    start_time = time.time()
    dat_files = discover_dat_files(args.board_data_dir)
    keys, invalid_side_samples, per_file, peak = collect_hint_keys(dat_files, args.batch_size, args.max_positions)
    position_samples = int(len(keys))
    peak = update_peak(peak)
    unique_keys = np.unique(keys)
    peak = update_peak(peak)
    unique_hint_positions = int(len(unique_keys))
    duplicate_hint_positions = position_samples - unique_hint_positions
    result = {
        "board_data_dir": str(args.board_data_dir),
        "position_samples": position_samples,
        "unique_hint_positions": unique_hint_positions,
        "duplicate_hint_positions": duplicate_hint_positions,
        "duplicate_hint_fraction": duplicate_hint_positions / position_samples if position_samples else 0.0,
        "invalid_side_samples": invalid_side_samples,
        "batch_size": args.batch_size,
        "elapsed_sec": time.time() - start_time,
        "peak_rss_mib": peak,
        "per_file": per_file,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wthor_position_duplicates.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count duplicate WTHOR position samples for hint-cache planning.")
    parser.add_argument("--board-data-dir", type=Path, default=default_board_data_dir())
    parser.add_argument("--batch-size", type=int, default=1000000)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output" / "wthor_position_duplicates")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    result = analyze(args)
    print("board_data_dir", result["board_data_dir"])
    print("position_samples", result["position_samples"])
    print("unique_hint_positions", result["unique_hint_positions"])
    print("duplicate_hint_positions", result["duplicate_hint_positions"])
    print("duplicate_hint_fraction", f"{result['duplicate_hint_fraction']:.6f}")
    print("invalid_side_samples", result["invalid_side_samples"])
    print("elapsed_sec", f"{result['elapsed_sec']:.3f}")
    if result["peak_rss_mib"] is not None:
        print("peak_rss_mib", f"{result['peak_rss_mib']:.3f}")
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
