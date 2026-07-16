#!/usr/bin/env python3
"""Convert yearly transcript CSV files to numbered, completed-game text files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


BOARD_SIZE = 8
EMPTY = 0
BLACK = 1
WHITE = -1
DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)
TRANSCRIPT_RE = re.compile(r"(?:[a-hA-H][1-8])+")


def new_board() -> list[list[int]]:
    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    board[3][3] = WHITE
    board[3][4] = BLACK
    board[4][3] = BLACK
    board[4][4] = WHITE
    return board


def flips_for_move(
    board: list[list[int]], player: int, row: int, column: int
) -> list[tuple[int, int]]:
    if board[row][column] != EMPTY:
        return []

    flips: list[tuple[int, int]] = []
    for delta_row, delta_column in DIRECTIONS:
        line: list[tuple[int, int]] = []
        r, c = row + delta_row, column + delta_column
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == -player:
            line.append((r, c))
            r += delta_row
            c += delta_column
        if line and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            flips.extend(line)
    return flips


def has_legal_move(board: list[list[int]], player: int) -> bool:
    return any(
        flips_for_move(board, player, row, column)
        for row in range(BOARD_SIZE)
        for column in range(BOARD_SIZE)
    )


def is_completed_transcript(transcript: str) -> bool:
    """Return True only for a legal transcript ending at a terminal position."""
    if not TRANSCRIPT_RE.fullmatch(transcript):
        return False

    board = new_board()
    player = BLACK
    for move_index in range(0, len(transcript), 2):
        if not has_legal_move(board, player):
            player = -player  # Passes are implicit in WTHOR-style transcripts.
            if not has_legal_move(board, player):
                return False  # The transcript contains moves after game end.

        column = ord(transcript[move_index].lower()) - ord("a")
        row = ord(transcript[move_index + 1]) - ord("1")
        flips = flips_for_move(board, player, row, column)
        if not flips:
            return False
        board[row][column] = player
        for r, c in flips:
            board[r][c] = player
        player = -player

    return not has_legal_move(board, player) and not has_legal_move(board, -player)


def convert_csv(input_path: Path, output_path: Path) -> tuple[int, int]:
    total = 0
    completed = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, output_path.open(
        "w", encoding="ascii", newline="\n"
    ) as destination:
        reader = csv.DictReader(source)
        if reader.fieldnames is not None and "transcript" in reader.fieldnames:
            transcripts = ((row.get("transcript") or "") for row in reader)
        else:
            # Older yearly files are named .csv but contain only one transcript per line.
            source.seek(0)
            transcripts = source
        for raw_transcript in transcripts:
            transcript = raw_transcript.strip().lower()
            if not transcript:
                continue
            total += 1
            if is_completed_transcript(transcript):
                destination.write(transcript + "\n")
                completed += 1
    return total, completed


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Pick completed games from yearly CSV files and write one transcript per line."
    )
    parser.add_argument("--start-year", type=int, required=True, help="first year to convert")
    parser.add_argument(
        "--output-start-number", type=int, required=True,
        help="number of the output file corresponding to the start year",
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=repository_root / "train_data/transcript/records1_raw",
        help="directory containing YYYY.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=repository_root / "train_data/transcript/records1",
        help="directory for seven-digit numbered .txt files",
    )
    parser.add_argument(
        "--end-year", type=int,
        help="last year to convert (default: latest available year)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_start_number < 0:
        print("error: --output-start-number must be non-negative", file=sys.stderr)
        return 2
    if args.end_year is not None and args.end_year < args.start_year:
        print("error: --end-year must be at least --start-year", file=sys.stderr)
        return 2

    csv_files: list[tuple[int, Path]] = []
    for path in args.input_dir.glob("*.csv"):
        if path.stem.isdigit():
            year = int(path.stem)
            if year >= args.start_year and (args.end_year is None or year <= args.end_year):
                csv_files.append((year, path))
    csv_files.sort()
    if not csv_files:
        print("error: no matching yearly CSV files found", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_games = total_completed = 0
    for offset, (year, input_path) in enumerate(csv_files):
        output_number = args.output_start_number + offset
        output_path = args.output_dir / f"{output_number:07d}.txt"
        total, completed = convert_csv(input_path, output_path)
        total_games += total
        total_completed += completed
        print(f"{year}: {completed}/{total} -> {output_path}")

    print(f"completed: {total_completed}/{total_games} games in {len(csv_files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
