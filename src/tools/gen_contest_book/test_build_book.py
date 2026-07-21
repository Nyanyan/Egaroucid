from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_book
from build_book import Node, Record
from othello import Board, N_CELLS, coord_to_index


def standard_board_text() -> str:
    cells = ["-"] * N_CELLS
    cells[coord_to_index("d4")] = "O"
    cells[coord_to_index("e4")] = "X"
    cells[coord_to_index("d5")] = "X"
    cells[coord_to_index("e5")] = "O"
    return "".join(cells) + " X"


def replay(initial_board: str, transcript: str) -> Board:
    board = Board.from_text(initial_board)
    for pos in range(0, len(transcript), 2):
        while not board.legal_moves() and not board.is_end():
            board.pass_turn()
        move = coord_to_index(transcript[pos:pos + 2])
        if move not in board.legal_moves():
            raise AssertionError(f"illegal test move {transcript[pos:pos + 2]}")
        board.play(move)
    return board


def new_root(initial_board: str) -> Node:
    root = Node(board_key=Board.from_text(initial_board).key())
    root._nodes = {root.board_key: root}
    return root


def find_node(root: Node, initial_board: str, transcript: str) -> Node:
    board = Board.from_text(initial_board)
    node = root
    for pos in range(0, len(transcript), 2):
        while not board.legal_moves() and not board.is_end():
            node = node.children[build_book.PASS_MOVE]
            board.pass_turn()
        move = coord_to_index(transcript[pos:pos + 2])
        node = node.children[move]
        board.play(move)
    return node


class BuildBookDagTest(unittest.TestCase):
    def test_partial_expansion_keeps_leaf_estimate(self) -> None:
        initial = standard_board_text()
        root = new_root(initial)
        board = Board.from_text(initial)
        move = board.legal_moves()[0]
        coord = build_book.index_to_coord(move)

        self.assertTrue(build_book.add_record(root, Record(initial, "", 10, 60)))
        self.assertTrue(build_book.add_record(root, Record(initial, coord, 0, 59)))

        self.assertEqual(build_book.solve_node(root), 10)

    def test_full_expansion_uses_children_instead_of_stale_leaf(self) -> None:
        initial = standard_board_text()
        root = new_root(initial)
        board = Board.from_text(initial)
        self.assertTrue(build_book.add_record(root, Record(initial, "", 10, 60)))
        for move in board.legal_moves():
            coord = build_book.index_to_coord(move)
            self.assertTrue(build_book.add_record(root, Record(initial, coord, 0, 59)))

        self.assertEqual(build_book.solve_node(root), 0)

    def test_transpositions_merge_and_emit_each_board_once(self) -> None:
        initial = standard_board_text()
        records = [
            Record(initial, "d3c3c4", 2, 57),
            Record(initial, "c4c3d3", 4, 57),
            Record(initial, "d3c3c4e3", 0, 56),
            Record(initial, "c4c3d3c5", 0, 56),
        ]

        def build(records_to_add: list[Record]) -> tuple[Node, list[str]]:
            root = new_root(initial)
            for record in records_to_add:
                self.assertTrue(build_book.add_record(root, record))
            build_book.solve_node(root)
            lines: list[str] = []
            build_book.collect_book_lines(root, Board.from_text(initial), 0, 64, 0, lines)
            return root, lines

        root, lines = build(records)
        target_a = find_node(root, initial, "d3c3c4")
        target_b = find_node(root, initial, "c4c3d3")
        self.assertIs(target_a, target_b)
        self.assertEqual(len(target_a.children), 2)
        self.assertEqual(target_a.value, 0)

        keys = [" ".join(line.split()[:2]) for line in lines]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys.count(target_a.board_key), 1)

        _, reversed_lines = build(list(reversed(records)))
        self.assertEqual(lines, reversed_lines)

    def test_all_symmetric_game_records_align_to_one_canonical_path(self) -> None:
        standard = standard_board_text()
        initial = replay(standard, "d3c3c4e3").key()
        initial_board = Board.from_text(initial)
        first = initial_board.legal_moves()[0]
        after_first = initial_board.copy()
        after_first.play(first)
        second = after_first.legal_moves()[0]
        transcript = build_book.index_to_coord(first) + build_book.index_to_coord(second)
        leaf_empty = N_CELLS - replay(initial, transcript).n_discs()

        records: list[Record] = []
        for symmetry in range(build_book.N_SYMMETRIES):
            transformed_initial = build_book.transform_board_text(initial, symmetry)
            transformed_transcript = build_book.transform_transcript(transcript, symmetry)
            # The transformed record must remain legal in its own orientation.
            replay(transformed_initial, transformed_transcript)
            records.append(Record(transformed_initial, transformed_transcript, 2, leaf_empty))

        aligned = [build_book.align_record_to_initial(record, initial) for record in records]
        self.assertTrue(all(record is not None for record in aligned))
        self.assertEqual(len({record.transcript for record in aligned if record is not None}), 1)

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            game_records_dir = temp_dir / "game_records"
            records_dir = temp_dir / "book_records"
            game_records_dir.mkdir()
            records_dir.mkdir()
            record_file = game_records_dir / "symmetry.txt"
            with record_file.open("w", encoding="utf-8", newline="\n") as f:
                for record in records:
                    f.write(f"initial board: {record.initial_board}\n")
                    f.write(f"transcript: {record.transcript}\n")
                    f.write(f"leaf value: {record.leaf_value}\n")
                    f.write(f"leaf empty: {record.leaf_empty}\n\n")

            with mock.patch.object(build_book, "GAME_RECORDS_DIR", game_records_dir):
                root, n_seen, n_used, _ = build_book.load_records(initial, records_dir, True)

        self.assertEqual((n_seen, n_used), (8, 8))
        canonical_transcript = next(record.transcript for record in aligned if record is not None)
        leaf = find_node(root, initial, canonical_transcript)
        self.assertEqual(leaf.leaf_values, [2] * 8)
        self.assertLess(len(root._nodes or {}), 1 + sum(len(record.transcript) // 2 for record in records))


if __name__ == "__main__":
    unittest.main()
