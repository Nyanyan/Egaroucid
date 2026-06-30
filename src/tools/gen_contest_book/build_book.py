from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    BOOK_RECORDS_DIR,
    DEFAULT_BOOK_MAX_LOSS,
    GAME_RECORDS_DIR,
    book_path_for_start,
    record_dir_for_start,
)
from othello import (
    PASS_MOVE,
    Board,
    coord_to_index,
    index_to_coord,
    normalize_board_text,
)


@dataclass
class Record:
    initial_board: str
    transcript: str
    leaf_value: int | None = None


@dataclass
class Node:
    leaf_values: list[int] = field(default_factory=list)
    children: dict[int, "Node"] = field(default_factory=dict)
    value: int | None = None


def parse_block_record(block: dict[str, str]) -> Record | None:
    initial_board = block.get("initial board")
    transcript = block.get("transcript")
    if not initial_board or transcript is None:
        return None
    leaf_value = None
    if "leaf value" in block:
        try:
            leaf_value = int(block["leaf value"])
        except ValueError:
            leaf_value = None
    return Record(normalize_board_text(initial_board), transcript.strip(), leaf_value)


def parse_record_file(path: Path) -> list[Record]:
    records: list[Record] = []
    block: dict[str, str] = {}

    def flush_block() -> None:
        nonlocal block
        if block:
            record = parse_block_record(block)
            if record is not None:
                records.append(record)
            block = {}

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                flush_block()
                continue
            if ": " in line:
                key, value = line.split(": ", 1)
                block[key] = value
                continue
            parts = line.split()
            if len(parts) >= 3 and len(parts[0]) == 64 and len(parts[1]) == 1:
                flush_block()
                records.append(Record(normalize_board_text(parts[0] + " " + parts[1]), parts[2].strip(), None))
        flush_block()
    return records


def add_pass_nodes(node: Node, board: Board) -> Node:
    while not board.legal_moves() and not board.is_end():
        node = node.children.setdefault(PASS_MOVE, Node())
        board.pass_turn()
    return node


def add_record(root: Node, record: Record) -> bool:
    board = Board.from_text(record.initial_board)
    node = add_pass_nodes(root, board)
    transcript = record.transcript
    if len(transcript) % 2 != 0:
        return False
    for pos in range(0, len(transcript), 2):
        node = add_pass_nodes(node, board)
        coord = transcript[pos:pos + 2]
        try:
            move = coord_to_index(coord)
        except ValueError:
            return False
        if move not in board.legal_moves():
            return False
        node = node.children.setdefault(move, Node())
        board.play(move)
    if record.leaf_value is not None:
        node.leaf_values.append(record.leaf_value)
    elif board.is_end():
        node.leaf_values.append(board.score_player())
    return True


def solve_node(node: Node) -> int | None:
    move_scores: list[int] = []
    for move, child in node.children.items():
        child_value = solve_node(child)
        if child_value is None:
            continue
        if move == PASS_MOVE:
            move_scores.append(-child_value)
        else:
            move_scores.append(-child_value)
    if move_scores:
        node.value = max(move_scores)
    elif node.leaf_values:
        node.value = round(sum(node.leaf_values) / len(node.leaf_values))
    else:
        node.value = None
    return node.value


def collect_book_lines(node: Node, board: Board, loss_sum: int, max_loss: int, lines: list[str]) -> None:
    if node.value is None:
        return
    legal_moves = set(board.legal_moves())
    move_items: list[tuple[int, int, Node, int]] = []
    for move, child in node.children.items():
        if move == PASS_MOVE:
            passed = board.copy()
            passed.pass_turn()
            collect_book_lines(child, passed, loss_sum, max_loss, lines)
            continue
        if move not in legal_moves or child.value is None:
            continue
        score = -child.value
        move_loss = node.value - score
        if loss_sum + move_loss <= max_loss:
            move_items.append((move, score, child, move_loss))
    if move_items:
        move_items.sort(key=lambda item: (-item[1], item[0]))
        move_text = " ".join(f"{index_to_coord(move)}:{score}" for move, score, _, _ in move_items)
        lines.append(f"{board.key()} {node.value} {move_text}")
    for move, _, child, move_loss in move_items:
        child_board = board.copy()
        child_board.play(move)
        collect_book_lines(child, child_board, loss_sum + move_loss, max_loss, lines)


def load_records(initial_board: str, records_dir: Path, include_game_records: bool) -> tuple[Node, int, int]:
    root = Node()
    n_seen = 0
    n_used = 0
    paths = sorted(records_dir.glob("*.txt")) if records_dir.exists() else []
    if include_game_records and GAME_RECORDS_DIR.exists():
        paths.extend(sorted(GAME_RECORDS_DIR.glob("*.txt")))
    for path in paths:
        for record in parse_record_file(path):
            n_seen += 1
            if record.initial_board != initial_board:
                continue
            if add_record(root, record):
                n_used += 1
    return root, n_seen, n_used


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_board")
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-book-loss", type=int, default=DEFAULT_BOOK_MAX_LOSS)
    parser.add_argument("--no-game-records", action="store_true")
    args = parser.parse_args()

    initial_board = normalize_board_text(args.initial_board)
    records_dir = args.records_dir or record_dir_for_start(initial_board)
    output = args.output or book_path_for_start(initial_board)
    output.parent.mkdir(parents=True, exist_ok=True)

    root, n_seen, n_used = load_records(initial_board, records_dir, not args.no_game_records)
    solve_node(root)

    board = Board.from_text(initial_board)
    lines: list[str] = []
    collect_book_lines(root, board, 0, args.max_book_loss, lines)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# contest_book_v1\n")
        f.write(f"# initial {initial_board}\n")
        f.write(f"# records_seen {n_seen}\n")
        f.write(f"# records_used {n_used}\n")
        for line in lines:
            f.write(line + "\n")
    print(f"wrote {len(lines)} boards from {n_used}/{n_seen} records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
