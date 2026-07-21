from __future__ import annotations

import argparse
import functools
import heapq
import itertools
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    BOOK_RECORDS_DIR,
    DEFAULT_BOOK_MAX_LOSS,
    DEFAULT_CUT_EMPTY,
    GAME_RECORDS_DIR,
    book_path_for_start,
    record_dir_for_start,
)
from othello import (
    N_CELLS,
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
    leaf_empty: int | None = None


@dataclass
class Node:
    board_key: str = ""
    leaf_values: list[int] = field(default_factory=list)
    children: dict[int, "Node"] = field(default_factory=dict)
    value: int | None = None
    _nodes: dict[str, "Node"] | None = field(default=None, repr=False, compare=False)


N_SYMMETRIES = 8

# Keep this order and these coordinate conversions in sync with
# util.hpp::representative_board() and
# convert_coord_{to,from}_representative_board().  C++ compares the unsigned
# (player, opponent) bitboard pair, not the printable board string.
CPP_REPRESENTATIVE_ORDER = (0, 2, 1, 3, 6, 4, 7, 5)


def _cpp_coord_to_representative(cell: int, symmetry: int) -> int:
    y = cell // 8
    x = cell % 8
    if symmetry == 0:
        return cell
    if symmetry == 1:
        return (7 - y) * 8 + x
    if symmetry == 2:
        return (7 - x) * 8 + (7 - y)
    if symmetry == 3:
        return x * 8 + (7 - y)
    if symmetry == 4:
        return (7 - x) * 8 + y
    if symmetry == 5:
        return x * 8 + y
    if symmetry == 6:
        return y * 8 + (7 - x)
    if symmetry == 7:
        return (7 - y) * 8 + (7 - x)
    raise ValueError(f"invalid C++ representative symmetry {symmetry}")


def _cpp_coord_from_representative(cell: int, symmetry: int) -> int:
    y = cell // 8
    x = cell % 8
    if symmetry == 0:
        return cell
    if symmetry == 1:
        return (7 - y) * 8 + x
    if symmetry == 2:
        return (7 - x) * 8 + (7 - y)
    if symmetry == 3:
        return (7 - x) * 8 + y
    if symmetry == 4:
        return x * 8 + (7 - y)
    if symmetry == 5:
        return x * 8 + y
    if symmetry == 6:
        return y * 8 + (7 - x)
    if symmetry == 7:
        return (7 - y) * 8 + (7 - x)
    raise ValueError(f"invalid C++ representative symmetry {symmetry}")


def move_to_representative(index: int, symmetry: int) -> int:
    """Map an a1=0 Python move into the C++ representative orientation."""
    if not (0 <= index < N_CELLS):
        raise ValueError(f"invalid board index {index}")
    cpp_cell = N_CELLS - 1 - index
    return N_CELLS - 1 - _cpp_coord_to_representative(cpp_cell, symmetry)


def move_from_representative(index: int, symmetry: int) -> int:
    """Map an a1=0 representative move back into the source orientation."""
    if not (0 <= index < N_CELLS):
        raise ValueError(f"invalid board index {index}")
    cpp_cell = N_CELLS - 1 - index
    return N_CELLS - 1 - _cpp_coord_from_representative(cpp_cell, symmetry)


def _transform_bitboard(bits: int, symmetry: int) -> int:
    transformed = 0
    while bits:
        low_bit = bits & -bits
        cell = low_bit.bit_length() - 1
        transformed |= 1 << _cpp_coord_to_representative(cell, symmetry)
        bits ^= low_bit
    return transformed


def _bitboards_to_key(player: int, opponent: int) -> str:
    cells: list[str] = []
    for index in range(N_CELLS):
        bit = 1 << (N_CELLS - 1 - index)
        if player & bit:
            cells.append("X")
        elif opponent & bit:
            cells.append("O")
        else:
            cells.append("-")
    return "".join(cells) + " X"


@functools.lru_cache(maxsize=262144)
def canonicalize_board_key(board_key: str) -> tuple[str, int]:
    """Return the exact C++ representative key and source-to-key symmetry."""
    relative_key = Board.from_text(board_key).key()
    player = 0
    opponent = 0
    for index, cell in enumerate(relative_key[:N_CELLS]):
        bit = 1 << (N_CELLS - 1 - index)
        if cell == "X":
            player |= bit
        elif cell == "O":
            opponent |= bit

    best_pair = (player, opponent)
    best_symmetry = 0
    for symmetry in CPP_REPRESENTATIVE_ORDER[1:]:
        candidate = (
            _transform_bitboard(player, symmetry),
            _transform_bitboard(opponent, symmetry),
        )
        if candidate < best_pair:
            best_pair = candidate
            best_symmetry = symmetry
    return _bitboards_to_key(*best_pair), best_symmetry


def canonicalize_board(board: Board) -> tuple[str, int]:
    return canonicalize_board_key(board.key())


def transform_index(index: int, symmetry: int) -> int:
    """Transform a square with one of the eight D4 board symmetries."""
    if not (0 <= index < N_CELLS):
        raise ValueError(f"invalid board index {index}")
    if not (0 <= symmetry < N_SYMMETRIES):
        raise ValueError(f"invalid symmetry {symmetry}")
    x = index % 8
    y = index // 8
    if symmetry >= 4:
        x = 7 - x
    for _ in range(symmetry % 4):
        x, y = 7 - y, x
    return y * 8 + x


def transform_board_text(board_text: str, symmetry: int) -> str:
    normalized = normalize_board_text(board_text)
    transformed = ["-"] * N_CELLS
    for index, cell in enumerate(normalized[:N_CELLS]):
        transformed[transform_index(index, symmetry)] = cell
    return "".join(transformed) + normalized[N_CELLS:]


def transform_transcript(transcript: str, symmetry: int) -> str:
    transcript = transcript.strip().lower()
    if len(transcript) % 2 != 0:
        raise ValueError("transcript length must be even")
    moves: list[str] = []
    for pos in range(0, len(transcript), 2):
        move = coord_to_index(transcript[pos:pos + 2])
        moves.append(index_to_coord(transform_index(move, symmetry)))
    return "".join(moves)


def align_record_to_initial(record: Record, initial_board: str) -> Record | None:
    """Rotate/reflect a record into the requested start-position orientation."""
    initial_board = normalize_board_text(initial_board)
    candidates: list[str] = []
    for symmetry in range(N_SYMMETRIES):
        if transform_board_text(record.initial_board, symmetry) != initial_board:
            continue
        try:
            candidates.append(transform_transcript(record.transcript, symmetry))
        except ValueError:
            return None
    if not candidates:
        return None
    # A symmetric start can admit several equivalent orientations.  Choosing
    # one canonical transcript makes ingestion independent of input order.
    transcript = min(set(candidates))
    return Record(initial_board, transcript, record.leaf_value, record.leaf_empty)


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
    leaf_empty = None
    if "leaf empty" in block:
        try:
            leaf_empty = int(block["leaf empty"])
        except ValueError:
            leaf_empty = None
    return Record(normalize_board_text(initial_board), transcript.strip(), leaf_value, leaf_empty)


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
                records.append(Record(normalize_board_text(parts[0] + " " + parts[1]), parts[2].strip(), None, None))
        flush_block()
    return records


def node_registry(root: Node) -> dict[str, Node]:
    if root._nodes is not None:
        return root._nodes
    nodes: dict[str, Node] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.board_key:
            previous = nodes.get(node.board_key)
            if previous is not None and previous is not node:
                raise ValueError(f"duplicate Node objects for {node.board_key}")
            nodes[node.board_key] = node
        stack.extend(node.children.values())
    root._nodes = nodes
    return nodes


def get_or_create_node(nodes: dict[str, Node], board: Board) -> Node:
    key, _ = canonicalize_board(board)
    node = nodes.get(key)
    if node is None:
        node = Node(board_key=key)
        nodes[key] = node
    return node


def add_pass_nodes(node: Node, board: Board, nodes: dict[str, Node] | None = None) -> Node:
    if nodes is None:
        nodes = node_registry(node)
    while not board.legal_moves() and not board.is_end():
        parent_key, _ = canonicalize_board(board)
        if node.board_key and node.board_key != parent_key:
            raise ValueError(f"pass node does not match board {parent_key}")
        board.pass_turn()
        child = get_or_create_node(nodes, board)
        existing = node.children.get(PASS_MOVE)
        if existing is not None and existing is not child:
            raise ValueError(f"conflicting pass edge from {node.board_key}")
        node.children[PASS_MOVE] = child
        node = child
    return node


def add_record(root: Node, record: Record) -> bool:
    board = Board.from_text(record.initial_board)
    root_key, _ = canonicalize_board(board)
    if not root.board_key:
        root.board_key = root_key
    if root.board_key != root_key:
        return False
    nodes = node_registry(root)
    nodes[root.board_key] = root

    # Validate the entire record before mutating the DAG.  A malformed record
    # must not leave a valid-looking prefix behind.
    edges: list[tuple[str, int, str]] = []

    def append_pass_edges() -> None:
        while not board.legal_moves() and not board.is_end():
            parent_key, _ = canonicalize_board(board)
            board.pass_turn()
            child_key, _ = canonicalize_board(board)
            edges.append((parent_key, PASS_MOVE, child_key))

    append_pass_edges()
    transcript = record.transcript
    if len(transcript) % 2 != 0:
        return False
    for pos in range(0, len(transcript), 2):
        append_pass_edges()
        coord = transcript[pos:pos + 2]
        try:
            move = coord_to_index(coord)
        except ValueError:
            return False
        if move not in board.legal_moves():
            return False
        parent_key, parent_symmetry = canonicalize_board(board)
        representative_move = move_to_representative(move, parent_symmetry)
        board.play(move)
        child_key, _ = canonicalize_board(board)
        edges.append((parent_key, representative_move, child_key))
    if record.leaf_empty is not None and N_CELLS - board.n_discs() != record.leaf_empty:
        return False

    for parent_key, move, child_key in edges:
        parent = nodes.get(parent_key)
        if parent is None:
            parent = Node(board_key=parent_key)
            nodes[parent_key] = parent
        child = nodes.get(child_key)
        if child is None:
            child = Node(board_key=child_key)
            nodes[child_key] = child
        existing = parent.children.get(move)
        if existing is not None and existing is not child:
            raise ValueError(f"conflicting edge {parent_key} {move}")
        parent.children[move] = child

    leaf_key, _ = canonicalize_board(board)
    node = nodes[leaf_key]
    if record.leaf_value is not None:
        node.leaf_values.append(record.leaf_value)
    elif board.is_end():
        node.leaf_values.append(board.score_player())
    return True


def expected_moves(node: Node) -> set[int]:
    if not node.board_key:
        return set(node.children)
    board = Board.from_text(node.board_key)
    legal_moves = set(board.legal_moves())
    if legal_moves:
        return legal_moves
    if not board.is_end():
        return {PASS_MOVE}
    return set()


def solve_node(
    node: Node,
    memo: dict[str, int | None] | None = None,
    visiting: set[str] | None = None,
) -> int | None:
    if memo is None:
        memo = {}
    if visiting is None:
        visiting = set()
    memo_key = node.board_key or f"node:{id(node)}"
    if memo_key in memo:
        node.value = memo[memo_key]
        return node.value
    if memo_key in visiting:
        raise ValueError(f"cycle in contest-book DAG at {memo_key}")
    visiting.add(memo_key)

    move_scores: list[int] = []
    covered_moves: set[int] = set()
    for move, child in node.children.items():
        child_value = solve_node(child, memo, visiting)
        if child_value is None:
            continue
        move_scores.append(-child_value)
        covered_moves.add(move)

    leaf_value = None
    if node.leaf_values:
        leaf_value = round(sum(node.leaf_values) / len(node.leaf_values))
    required_moves = expected_moves(node)
    fully_expanded = bool(required_moves) and required_moves.issubset(covered_moves)

    if move_scores:
        candidates = move_scores
        if leaf_value is not None and not fully_expanded:
            candidates = [*move_scores, leaf_value]
        node.value = max(candidates)
    elif leaf_value is not None:
        node.value = leaf_value
    else:
        node.value = None
    visiting.remove(memo_key)
    memo[memo_key] = node.value
    return node.value


def collect_book_lines(node: Node, board: Board, loss_sum: int, max_loss: int, cut_empty: int, lines: list[str]) -> None:
    # A transposition may be reachable with different accumulated losses.  A
    # shortest-path traversal makes the result history-independent and emits
    # every board key at most once using its least-loss path.
    counter = itertools.count()
    queue: list[tuple[int, str, int, Node, Board]] = []
    root_key, _ = canonicalize_board(board)
    if node.board_key and node.board_key != root_key:
        raise ValueError(f"root node {node.board_key} does not match board {root_key}")
    canonical_root = Board.from_text(root_key)
    heapq.heappush(queue, (loss_sum, root_key, next(counter), node, canonical_root))
    best_loss: dict[str, int] = {root_key: loss_sum}
    emitted: set[str] = set()
    processed: set[str] = set()

    while queue:
        current_loss, board_key, _, current, current_board = heapq.heappop(queue)
        if current_loss != best_loss.get(board_key) or board_key in processed:
            continue
        processed.add(board_key)
        if current.value is None or N_CELLS - current_board.n_discs() <= cut_empty:
            continue

        legal_moves = set(current_board.legal_moves())
        if not legal_moves and not current_board.is_end():
            child = current.children.get(PASS_MOVE)
            if child is not None and child.value is not None:
                passed = current_board.copy()
                passed.pass_turn()
                child_key, _ = canonicalize_board(passed)
                if child.board_key != child_key:
                    raise ValueError(f"pass edge points to {child.board_key}, expected {child_key}")
                if current_loss < best_loss.get(child_key, max_loss + 1):
                    best_loss[child_key] = current_loss
                    heapq.heappush(
                        queue,
                        (current_loss, child_key, next(counter), child, Board.from_text(child_key)),
                    )
            continue

        move_items: list[tuple[int, int, Node, int]] = []
        for move in sorted(legal_moves):
            child = current.children.get(move)
            if child is None or child.value is None:
                continue
            score = -child.value
            move_loss = current.value - score
            if current_loss + move_loss <= max_loss:
                move_items.append((move, score, child, move_loss))
        move_items.sort(key=lambda item: (-item[1], item[0]))

        if move_items and board_key not in emitted:
            move_text = " ".join(f"{index_to_coord(move)}:{score}" for move, score, _, _ in move_items)
            lines.append(f"{board_key} {current.value} {move_text}")
            emitted.add(board_key)

        for move, _, child, move_loss in move_items:
            child_board = current_board.copy()
            child_board.play(move)
            child_key, _ = canonicalize_board(child_board)
            if child.board_key != child_key:
                raise ValueError(f"move edge points to {child.board_key}, expected {child_key}")
            child_loss = current_loss + move_loss
            if child_loss < best_loss.get(child_key, max_loss + 1):
                best_loss[child_key] = child_loss
                heapq.heappush(
                    queue,
                    (child_loss, child_key, next(counter), child, Board.from_text(child_key)),
                )


def load_records(initial_board: str, records_dir: Path, include_game_records: bool) -> tuple[Node, int, int, list[int]]:
    initial_board = normalize_board_text(initial_board)
    root_board = Board.from_text(initial_board)
    root_key, _ = canonicalize_board(root_board)
    root = Node(board_key=root_key)
    root._nodes = {root.board_key: root}
    n_seen = 0
    n_used = 0
    explicit_cut_empties: list[int] = []
    paths = sorted(records_dir.glob("*.txt")) if records_dir.exists() else []
    if include_game_records and GAME_RECORDS_DIR.exists():
        paths.extend(sorted(GAME_RECORDS_DIR.glob("*.txt")))
    for path in paths:
        for record in parse_record_file(path):
            n_seen += 1
            aligned_record = align_record_to_initial(record, initial_board)
            if aligned_record is None:
                continue
            if add_record(root, aligned_record):
                n_used += 1
                if aligned_record.leaf_empty is not None:
                    explicit_cut_empties.append(aligned_record.leaf_empty)
    return root, n_seen, n_used, explicit_cut_empties


def resolve_book_cut_empty(requested_cut_empty: int | None, explicit_cut_empties: list[int]) -> int:
    if requested_cut_empty is not None:
        return requested_cut_empty
    if explicit_cut_empties:
        return min(explicit_cut_empties)
    return DEFAULT_CUT_EMPTY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_board")
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-book-loss", type=int, default=DEFAULT_BOOK_MAX_LOSS)
    parser.add_argument("--cut-empty", type=int)
    parser.add_argument("--no-game-records", action="store_true")
    args = parser.parse_args()

    initial_board = normalize_board_text(args.initial_board)
    if args.cut_empty is not None and not (0 <= args.cut_empty < N_CELLS):
        raise ValueError("--cut-empty must be in [0, 63]")
    records_dir = args.records_dir or record_dir_for_start(initial_board)
    output = args.output or book_path_for_start(initial_board)
    output.parent.mkdir(parents=True, exist_ok=True)

    root, n_seen, n_used, explicit_cut_empties = load_records(initial_board, records_dir, not args.no_game_records)
    solve_node(root)
    book_cut_empty = resolve_book_cut_empty(args.cut_empty, explicit_cut_empties)

    board = Board.from_text(initial_board)
    lines: list[str] = []
    collect_book_lines(root, board, 0, args.max_book_loss, book_cut_empty, lines)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# contest_book_v1\n")
        f.write(f"# initial {initial_board}\n")
        f.write(f"# records_seen {n_seen}\n")
        f.write(f"# records_used {n_used}\n")
        f.write(f"# cut_empty {book_cut_empty}\n")
        for line in lines:
            f.write(line + "\n")
    print(f"wrote {len(lines)} boards cut_empty={book_cut_empty} from {n_used}/{n_seen} records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
