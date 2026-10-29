"""GUIが保存したブックを、エンジンとは独立に構造検証する。"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from common import require
from book_fixture import representative
from reference_board import Board


def _read_book(path, count=None):
    data = path.read_bytes()
    positions = []
    root_lines = None
    leaves = 0
    edax_leaves = []
    if data.startswith(b"DICUORAGE\x03"):
        require(len(data) >= 14, "egbk3 header is truncated")
        declared = struct.unpack_from("<I", data, 10)[0]
        require(len(data) == 14 + 25 * declared, "egbk3 record count/size mismatch")
        for offset in range(14, len(data), 25):
            player, opponent, value, level, lines, leaf_value, leaf_move, leaf_level = struct.unpack_from("<QQbbIbbb", data, offset)
            require(-64 <= value <= 64, "invalid position value")
            # 64=pass, 65=no remaining move, 125=undefined (engine/search.hpp).
            require(leaf_move in range(66) or leaf_move == 125, "invalid leaf move")
            positions.append((player, opponent))
            if bin(player | opponent).count("1") == 4:
                root_lines = lines
            leaves += -64 <= leaf_value <= 64 and 0 <= leaf_move < 64
        kind = "egbk3"
    elif data.startswith(b"XADEKOOB\x04\x04"):
        require(len(data) >= 42, "Edax header is truncated")
        declared = struct.unpack_from("<I", data, 38)[0]
        offset = 42
        for _ in range(declared):
            require(offset + 40 <= len(data), "Edax record is truncated")
            player, opponent = struct.unpack_from("<QQ", data, offset)
            links = data[offset + 38]
            require(links <= 64, "invalid link count")
            end = offset + 40 + 2 * (links + 1)
            require(end <= len(data), "Edax links are truncated")
            leaf_value, leaf_move = struct.unpack_from("<bb", data, end - 2)
            edax_leaves.append((player, opponent, leaf_value, leaf_move))
            leaves += -64 <= leaf_value <= 64 and 0 <= leaf_move < 64
            if bin(player | opponent).count("1") == 4:
                root_lines = struct.unpack_from("<I", data, offset + 28)[0]
            offset = end
            positions.append((player, opponent))
        require(offset == len(data), "Edax record count/size mismatch")
        kind = "edax"
    else:
        raise ValueError("Unknown book header")
    require(len(set(positions)) == declared, "duplicate positions")
    require(all(not (player & opponent) and bin(player | opponent).count("1") >= 4
                for player, opponent in positions), "invalid board")
    if count is not None:
        require(declared == count, f"Expected {count} positions, found {declared}")
    result = {"passed": True, "format": kind, "positions": declared, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "root_lines": root_lines,
            "leaves": leaves, "max_discs": max((bin(p | o).count("1") for p, o in positions), default=0)}
    return result, positions, edax_leaves


def inspect_book(path, count=None):
    return _read_book(path, count)[0]


def array_board(player, opponent):
    board = Board()
    board.cells = ["B" if (player >> (63 - i)) & 1 else "W" if (opponent >> (63 - i)) & 1 else "-" for i in range(64)]
    return board


def canonical_array_board(board):
    player = sum(1 << (63 - i) for i, disc in enumerate(board.cells) if disc == board.turn)
    opponent = sum(1 << (63 - i) for i, disc in enumerate(board.cells) if disc not in (board.turn, "-"))
    return representative(player, opponent)


def transcript_positions(transcript):
    require(len(transcript) % 2 == 0, "Odd transcript length")
    board = Board()
    positions = {canonical_array_board(board)}
    for i in range(0, len(transcript), 2):
        move = transcript[i:i + 2]
        if move != "ps" and not board.legal():
            board.pass_turn()
        board.play(move)
        positions.add(canonical_array_board(board))
    return positions


def imported_positions(path):
    metadata, positions, edax_leaves = _read_book(path)
    if metadata["format"] == "egbk3":
        return {representative(p, o) for p, o in positions}
    # An Edax leaf is an extra registered child when imported by Egaroucid.
    # Apply its move with the independent array-based rules, including a pass.
    result = {representative(p, o) for p, o in positions if array_board(p, o).legal()}
    for player, opponent, value, move in edax_leaves:
        if not (-64 <= value <= 64 and 0 <= move < 64):
            continue
        board = array_board(player, opponent)
        cell = 63 - move
        board.play(chr(97 + cell % 8) + str(cell // 8 + 1))
        if not board.legal() and not board.ended():
            board.pass_turn()
        result.add(canonical_array_board(board))
    return result


def compare_union(actual, inputs):
    metadata, positions, _ = _read_book(actual)
    require(metadata["format"] == "egbk3", "Compare the merged egbk3 with its input books")
    expected = set().union(*(imported_positions(path) for path in inputs))
    observed = {representative(p, o) for p, o in positions}
    missing, unexpected = len(expected - observed), len(observed - expected)
    require(not missing and not unexpected, f"Book union differs: {missing} missing, {unexpected} unexpected")
    metadata.update(expected_positions=len(expected), missing_positions=missing, unexpected_positions=unexpected)
    return metadata


def compare_contains(actual, required, min_added=0, transcripts=()):
    metadata, positions, _ = _read_book(actual)
    require(metadata["format"] == "egbk3", "Inspect the learned egbk3")
    expected = set().union(*(imported_positions(path) for path in required))
    original = expected.copy()
    expected.update(set().union(*(transcript_positions(text) for text in transcripts)))
    observed = {representative(p, o) for p, o in positions}
    missing, added = len(expected - observed), len(observed - original)
    require(not missing, f"Learned book lost {missing} original positions")
    require(added >= min_added, f"Expected at least {min_added} added positions, found {added}")
    metadata.update(required_positions=len(expected), missing_positions=missing, added_positions=added, minimum_added=min_added)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--expected-union", nargs="+", type=Path)
    parser.add_argument("--contains", nargs="+", type=Path)
    parser.add_argument("--min-added", type=int, default=0)
    parser.add_argument("--transcript", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.expected_union:
            result = compare_union(args.book, args.expected_union)
        elif args.contains or args.transcript:
            result = compare_contains(args.book, args.contains or [], args.min_added, args.transcript)
        else:
            result = inspect_book(args.book, args.count)
    except Exception as error:
        result = {"passed": False, "error": str(error), "file": str(args.book)}
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
