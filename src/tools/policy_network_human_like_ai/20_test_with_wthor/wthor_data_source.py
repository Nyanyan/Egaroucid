"""Locate the prepared human-move test split without machine-specific paths."""

import argparse
import os
from pathlib import Path


DEFAULT_DATASET = "WTHOR_1977_2024"
DEFAULT_DATA_SUBDIRECTORY = "20261002_policy_network_human_like_data"


def dataset_name(value: str) -> str:
    if (not value or value in (".", "..") or value.endswith((".", " "))
            or any(char in value for char in '<>:"/\\|?*')
            or any(ord(char) < 32 for char in value)):
        raise argparse.ArgumentTypeError("--dataset must be a single directory name")
    return value


def default_board_data_dir(dataset: str = DEFAULT_DATASET) -> Path:
    dataset = dataset_name(dataset)
    root = os.environ.get("POLICY_DATA_ROOT")
    if root:
        data_root = Path(root)
    elif os.environ.get("EGAROUCID_DATA"):
        data_root = Path(os.environ["EGAROUCID_DATA"]) / DEFAULT_DATA_SUBDIRECTORY
    else:
        raise ValueError(
            "set POLICY_DATA_ROOT or EGAROUCID_DATA, or pass --board-data-dir"
        )
    return data_root / "board_split_increased" / dataset / "test"


def add_board_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", type=dataset_name, default=DEFAULT_DATASET,
        help=f"Dataset under $POLICY_DATA_ROOT/board_split_increased (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--board-data-dir", type=Path, default=None,
        help="Override $POLICY_DATA_ROOT/board_split_increased/<dataset>/test. "
             "POLICY_DATA_ROOT defaults to $EGAROUCID_DATA/" + DEFAULT_DATA_SUBDIRECTORY + ".",
    )


def resolve_board_data_args(args, parser: argparse.ArgumentParser) -> None:
    if args.board_data_dir is None:
        try:
            args.board_data_dir = default_board_data_dir(args.dataset)
        except ValueError as error:
            parser.error(str(error))
