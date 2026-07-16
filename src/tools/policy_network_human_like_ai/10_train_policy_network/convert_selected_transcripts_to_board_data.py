#!/usr/bin/env python3
"""
Convert selected transcript_release/0002 games to policy-network board data.

Input:
  train_data/transcript/Egaroucid_Train_Data_v2_selected/<random_depth>/*.txt

Output:
  train_data/board_data/Egaroucid_Train_Data_v2_selected/records0/*.dat

Records before each file's random_depth are played but not written, following
the dataset recommendation to exclude the random opening segment from training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import struct
from typing import Iterable, List, Optional, Sequence, Tuple


HW = 8
HW2 = 64
HW2_M1 = 63
BLACK = 0
WHITE = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_input_root() -> Path:
    return repo_root() / "train_data" / "transcript" / "Egaroucid_Train_Data_v2_selected"


def default_output_root() -> Path:
    return repo_root() / "train_data" / "board_data" / "Egaroucid_Train_Data_v2_selected" / "records0"


def policy_from_coord(file_char: str, rank_char: str) -> int:
    file_char = file_char.lower()
    if not ("a" <= file_char <= "h") or not ("1" <= rank_char <= "8"):
        return -1
    x = ord(file_char) - ord("a")
    y = ord(rank_char) - ord("1")
    return HW2_M1 - (y * HW + x)


def bit_from_policy(policy: int) -> int:
    return 1 << policy


def initial_board() -> Tuple[int, int, int]:
    black = bit_from_policy(policy_from_coord("e", "4")) | bit_from_policy(policy_from_coord("d", "5"))
    white = bit_from_policy(policy_from_coord("d", "4")) | bit_from_policy(policy_from_coord("e", "5"))
    return black, white, BLACK


def calc_flips(player: int, opponent: int, policy: int) -> int:
    if policy < 0 or policy >= HW2:
        return 0
    move_bit = bit_from_policy(policy)
    if (player | opponent) & move_bit:
        return 0
    pos = HW2_M1 - policy
    x0 = pos % HW
    y0 = pos // HW
    flips = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        x = x0 + dx
        y = y0 + dy
        line = 0
        while 0 <= x < HW and 0 <= y < HW:
            p = HW2_M1 - (y * HW + x)
            bit = bit_from_policy(p)
            if opponent & bit:
                line |= bit
            else:
                if (player & bit) and line:
                    flips |= line
                break
            x += dx
            y += dy
    return flips


def legal_moves(player: int, opponent: int) -> int:
    occupied = player | opponent
    legal = 0
    for policy in range(HW2):
        bit = bit_from_policy(policy)
        if (occupied & bit) == 0 and calc_flips(player, opponent, policy):
            legal |= bit
    return legal


def apply_move(black: int, white: int, side: int, policy: int) -> Optional[Tuple[int, int, int]]:
    player = black if side == BLACK else white
    opponent = white if side == BLACK else black
    flips = calc_flips(player, opponent, policy)
    if not flips:
        return None
    player ^= flips | bit_from_policy(policy)
    opponent ^= flips
    if side == BLACK:
        black, white = player, opponent
    else:
        white, black = player, opponent
    return black, white, side ^ 1


def pass_if_needed(black: int, white: int, side: int) -> Tuple[int, bool]:
    player = black if side == BLACK else white
    opponent = white if side == BLACK else black
    if legal_moves(player, opponent):
        return side, True
    side ^= 1
    player = black if side == BLACK else white
    opponent = white if side == BLACK else black
    return side, bool(legal_moves(player, opponent))


def popcount(x: int) -> int:
    return bin(x).count("1")


def score_for_side(black: int, white: int, side: int) -> int:
    black_count = popcount(black)
    white_count = popcount(white)
    score_black = black_count - white_count
    return score_black if side == BLACK else -score_black


def convert_transcript(transcript: str, random_depth: int) -> Tuple[List[Tuple[int, int, int, int, int]], Optional[str]]:
    transcript = transcript.strip()
    if len(transcript) % 2 != 0:
        return [], "odd transcript length"
    black, white, side = initial_board()
    pending: List[Tuple[int, int, int, int]] = []
    move_count = len(transcript) // 2
    for move_idx in range(move_count):
        policy = policy_from_coord(transcript[move_idx * 2], transcript[move_idx * 2 + 1])
        if policy < 0:
            return [], f"invalid coordinate at move {move_idx}"
        moved = apply_move(black, white, side, policy)
        if moved is None:
            side ^= 1
            moved = apply_move(black, white, side, policy)
            if moved is None:
                return [], f"illegal move at move {move_idx}"
        if move_idx >= random_depth:
            player = black if side == BLACK else white
            opponent = white if side == BLACK else black
            pending.append((player, opponent, side, policy))
        black, white, side = moved

    side, can_move = pass_if_needed(black, white, side)
    if can_move:
        return [], "transcript ended before both sides had no legal move"

    records = []
    for player, opponent, record_side, policy in pending:
        score = score_for_side(black, white, record_side)
        records.append((player, opponent, record_side, policy, score))
    return records, None


def iter_transcript_files(input_root: Path) -> Iterable[Tuple[int, Path]]:
    for path in sorted(input_root.rglob("*.txt")):
        if path.parent.name.isdigit():
            yield int(path.parent.name), path


def pack_record(record: Tuple[int, int, int, int, int]) -> bytes:
    player, opponent, side, policy, score = record
    return struct.pack("<QQbbb", player, opponent, side, policy, score)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert selected transcripts to policy board data.")
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--records-per-file", type=int, default=5_000_000)
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    out_idx = 0
    records_in_file = 0
    out = (args.output_root / f"{out_idx}.dat").open("wb")
    total_games = 0
    valid_games = 0
    total_records = 0
    invalid_rows: List[dict] = []
    depth_counts = {}

    try:
        for random_depth, path in iter_transcript_files(args.input_root):
            with path.open("r", encoding="utf-8") as f:
                for line_idx, transcript in enumerate(f):
                    total_games += 1
                    records, error = convert_transcript(transcript, random_depth)
                    if error is not None:
                        invalid_rows.append(
                            {
                                "file": str(path.relative_to(repo_root())),
                                "line": line_idx,
                                "random_depth": random_depth,
                                "error": error,
                            }
                        )
                        continue
                    valid_games += 1
                    depth_counts[str(random_depth)] = depth_counts.get(str(random_depth), 0) + 1
                    for record in records:
                        if records_in_file >= args.records_per_file:
                            out.close()
                            out_idx += 1
                            records_in_file = 0
                            out = (args.output_root / f"{out_idx}.dat").open("wb")
                        out.write(pack_record(record))
                        records_in_file += 1
                        total_records += 1
    finally:
        out.close()

    write_csv(args.output_root / "invalid_games.csv", invalid_rows)
    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "total_games": total_games,
        "valid_games": valid_games,
        "invalid_games": len(invalid_rows),
        "total_records": total_records,
        "records_per_file": args.records_per_file,
        "output_files": out_idx + 1 if total_records else 0,
        "random_depth_valid_game_counts": dict(sorted(depth_counts.items(), key=lambda kv: int(kv[0]))),
    }
    with (args.output_root / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
