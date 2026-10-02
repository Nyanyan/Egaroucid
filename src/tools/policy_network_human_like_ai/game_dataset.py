"""Compatibility imports for the dataset implementation in 00_data_processing."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "00_data_processing"))

from wthor_dataset import (  # noqa: E402, F401
    BOARD_DTYPE, SPLIT_NAMES, file_sha256, game_ids_path,
    infer_ordered_game_ids, load_game_dataset, load_manifest_positions,
    load_manifest_splits, save_split_manifest, split_counts, split_game_positions,
)
