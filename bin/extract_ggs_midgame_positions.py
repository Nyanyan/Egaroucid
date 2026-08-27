#!/usr/bin/env python3
"""Extract reproducible midgame positions from Egaroucid GGS game logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def other(side: str) -> str:
    return "O" if side == "X" else "X"


def flips_for(cells: list[str], side: str, move: int) -> list[int]:
    if cells[move] != "-":
        return []
    opponent = other(side)
    row, column = divmod(move, 8)
    result: list[int] = []
    for dr, dc in DIRECTIONS:
        r, c = row + dr, column + dc
        line: list[int] = []
        while 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == opponent:
            line.append(r * 8 + c)
            r += dr
            c += dc
        if line and 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == side:
            result.extend(line)
    return result


def has_legal(cells: list[str], side: str) -> bool:
    return any(cell == "-" and flips_for(cells, side, move) for move, cell in enumerate(cells))


def move_index(coordinate: str) -> int:
    if len(coordinate) != 2 or coordinate[0] not in "abcdefgh" or coordinate[1] not in "12345678":
        raise ValueError(f"invalid coordinate {coordinate!r}")
    return (ord(coordinate[1]) - ord("1")) * 8 + ord(coordinate[0]) - ord("a")


def parse_game(path: Path) -> tuple[list[str], str, list[str], int | None]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key.strip()] = value.strip()
    board = fields.get("initial board", "")
    transcript = fields.get("transcript", "")
    if len(board) < 66 or len(transcript) % 2:
        raise ValueError("missing or malformed game fields")
    cells = list(board[:64])
    side = board[65]
    if side not in "XO" or any(cell not in "XO-" for cell in cells):
        raise ValueError("malformed initial board")
    moves = [transcript[i:i + 2] for i in range(0, len(transcript), 2)]
    score_text = fields.get("black's score")
    return cells, side, moves, int(score_text) if score_text is not None else None


def replay(path: Path, targets: set[int]) -> list[dict[str, object]]:
    cells, side, moves, expected_score = parse_game(path)
    selected: list[dict[str, object]] = []
    seen_targets: set[int] = set()

    def record(ply: int) -> None:
        empties = cells.count("-")
        if empties in targets and empties not in seen_targets:
            selected.append({
                "board": "".join(cells) + " " + side,
                "empties": empties,
                "game": path.name,
                "ply": ply,
            })
            seen_targets.add(empties)

    record(0)
    for ply, coordinate in enumerate(moves, 1):
        if not has_legal(cells, side):
            side = other(side)
        move = move_index(coordinate)
        flips = flips_for(cells, side, move)
        if not flips:
            raise ValueError(f"illegal {coordinate} at ply {ply} for {side}")
        cells[move] = side
        for square in flips:
            cells[square] = side
        side = other(side)
        if not has_legal(cells, side) and has_legal(cells, other(side)):
            side = other(side)
        record(ply)

    if expected_score is not None and "-" not in cells:
        actual_score = cells.count("X") - cells.count("O")
        if actual_score != expected_score:
            raise ValueError(f"score mismatch: expected {expected_score}, got {actual_score}")
    return selected


def parse_targets(text: str) -> set[int]:
    result = {int(value) for value in text.split(",") if value.strip()}
    if not result or min(result) < 1 or max(result) > 60:
        raise argparse.ArgumentTypeError("targets must be comma-separated empties in 1..60")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-dir", type=Path, default=Path("ggs/log/game"))
    parser.add_argument("--targets", type=parse_targets, default=parse_targets("50,46,42,38,34"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in 0..shard-count-1")

    files = sorted(args.game_dir.glob("*.txt"))
    files = [
        path for path in files
        if int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "big") % args.shard_count
        == args.shard_index
    ]
    files.sort(key=lambda path: hashlib.sha256(f"{args.seed}|{path.name}".encode()).digest())
    if args.max_games is not None:
        files = files[:args.max_games]

    rows: list[dict[str, object]] = []
    rejected = 0
    seen: set[str] = set()
    for path in files:
        try:
            game_rows = replay(path, args.targets)
        except (OSError, ValueError):
            rejected += 1
            continue
        for row in game_rows:
            board = str(row["board"])
            if board not in seen:
                seen.add(board)
                rows.append(row)

    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}|{row['game']}|{row['empties']}|{row['board']}".encode()
        ).digest()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{row['board']}\n" for row in rows), encoding="utf-8", newline="\n")
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
    print(f"games={len(files)} rejected={rejected} positions={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
