#!/usr/bin/env python3
"""Prepare reproducible game-level splits of the existing WTHOR board data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wthor_dataset import load_game_dataset, save_split_manifest, split_game_positions


def default_board_data_dir() -> Path:
    data_root = os.environ.get("EGAROUCID_DATA")
    if not data_root:
        raise ValueError("set EGAROUCID_DATA or pass --board-data-dir pointing to WTHOR records1")
    return Path(data_root) / "train_data" / "board_data" / "records1"


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-data-dir", type=Path, default=None,
                        help="Default: $EGAROUCID_DATA/train_data/board_data/records1")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "output" / "wthor")
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--max-position-samples", type=int, default=None,
                        help="Optional cap: include only whole games fitting within this limit.")
    return parser


def prepare(args: argparse.Namespace) -> dict:
    board_data_dir = args.board_data_dir if args.board_data_dir is not None else default_board_data_dir()
    paths = sorted(board_data_dir.glob("*.dat"),
                   key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
    if not paths:
        raise FileNotFoundError(f"no WTHOR board data found in {board_data_dir}")
    samples, game_ids = load_game_dataset(paths)
    indices, summary = split_game_positions(
        samples, game_ids, args.train_ratio, args.val_ratio, args.test_ratio,
        args.seed, args.max_position_samples,
    )
    summary["source"] = "WTHOR"
    save_split_manifest(args.output_dir, paths, indices, summary)
    return summary


def main() -> None:
    args = make_argparser().parse_args()
    summary = prepare(args)
    print(json.dumps(summary, indent=2))
    print("split_manifest", args.output_dir / "dataset_split.json")


if __name__ == "__main__":
    main()
