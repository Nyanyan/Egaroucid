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
    root_key, _ = build_book.canonicalize_board(Board.from_text(initial_board))
    root = Node(board_key=root_key)
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
        _, symmetry = build_book.canonicalize_board(board)
        representative_move = build_book.move_to_representative(move, symmetry)
        node = node.children[representative_move]
        board.play(move)
    return node


def assert_canonical_legal_lines(test: unittest.TestCase, lines: list[str]) -> None:
    keys: list[str] = []
    for line in lines:
        parts = line.split()
        test.assertGreaterEqual(len(parts), 4)
        board_key = " ".join(parts[:2])
        keys.append(board_key)
        canonical_key, symmetry = build_book.canonicalize_board_key(board_key)
        test.assertEqual(canonical_key, board_key)
        test.assertEqual(symmetry, 0)

        board = Board.from_text(board_key)
        legal_moves = set(board.legal_moves())
        policies: list[int] = []
        for move_token in parts[3:]:
            coord, _ = move_token.split(":", 1)
            policy = coord_to_index(coord)
            test.assertIn(policy, legal_moves, f"illegal output move in {line}")
            policies.append(policy)
        test.assertEqual(len(policies), len(set(policies)))
    test.assertEqual(len(keys), len(set(keys)))


def assert_canonical_graph(test: unittest.TestCase, root: Node) -> None:
    nodes = root._nodes or {}
    test.assertGreater(len(nodes), 0)
    for board_key, node in nodes.items():
        test.assertIs(node, nodes[board_key])
        test.assertEqual(node.board_key, board_key)
        test.assertEqual(build_book.canonicalize_board_key(board_key), (board_key, 0))
        board = Board.from_text(board_key)
        legal_moves = set(board.legal_moves())
        for move, child in node.children.items():
            child_board = board.copy()
            if move == build_book.PASS_MOVE:
                test.assertFalse(legal_moves)
                test.assertFalse(board.is_end())
                child_board.pass_turn()
            else:
                test.assertIn(move, legal_moves)
                child_board.play(move)
            child_key, _ = build_book.canonicalize_board(child_board)
            test.assertEqual(child.board_key, child_key)
            test.assertIs(nodes[child_key], child)


class BuildBookDagTest(unittest.TestCase):
    def test_representative_matches_cpp_and_reorients_each_edge(self) -> None:
        initial = "------------------XXXO----OOXO-----OOX----O-OO------------------ X"
        expected_root = "------------------O-OO-----OOX----OOXO----XXXO------------------ X"
        board = Board.from_text(initial)
        root_key, root_symmetry = build_book.canonicalize_board(board)
        self.assertEqual(root_key, expected_root)
        self.assertEqual(root_symmetry, 1)
        white_to_move_initial = "".join(
            "O" if cell == "X" else "X" if cell == "O" else "-"
            for cell in initial[:N_CELLS]
        ) + " O"
        self.assertEqual(
            build_book.canonicalize_board(Board.from_text(white_to_move_initial)),
            (root_key, root_symmetry),
        )

        selected: tuple[int, str, int] | None = None
        for move in board.legal_moves():
            child = board.copy()
            child.play(move)
            _, child_symmetry = build_book.canonicalize_board(child)
            if child_symmetry != root_symmetry:
                selected = move, build_book.index_to_coord(move), child_symmetry
                break
        self.assertIsNotNone(selected)
        move, coord, child_symmetry = selected or (0, "", 0)
        self.assertNotEqual(root_symmetry, child_symmetry)

        root = new_root(initial)
        self.assertTrue(build_book.add_record(root, Record(initial, coord, 4, 49)))
        representative_move = build_book.move_to_representative(move, root_symmetry)
        self.assertEqual(
            build_book.move_from_representative(representative_move, root_symmetry),
            move,
        )
        for symmetry in range(build_book.N_SYMMETRIES):
            for index in range(N_CELLS):
                self.assertEqual(
                    build_book.move_from_representative(
                        build_book.move_to_representative(index, symmetry),
                        symmetry,
                    ),
                    index,
                )
            transformed_initial = build_book.transform_board_text(initial, symmetry)
            transformed_move = build_book.transform_index(move, symmetry)
            transformed_key, transformed_to_representative = build_book.canonicalize_board(
                Board.from_text(transformed_initial)
            )
            self.assertEqual(transformed_key, expected_root)
            self.assertEqual(
                build_book.move_to_representative(
                    transformed_move,
                    transformed_to_representative,
                ),
                representative_move,
            )
        self.assertIn(representative_move, Board.from_text(root.board_key).legal_moves())
        child_node = root.children[representative_move]
        child_board = board.copy()
        child_board.play(move)
        self.assertEqual(child_node.board_key, build_book.canonicalize_board(child_board)[0])

        build_book.solve_node(root)
        lines: list[str] = []
        build_book.collect_book_lines(root, board, 0, 64, 0, lines)
        assert_canonical_graph(self, root)
        assert_canonical_legal_lines(self, lines)

    def test_symmetric_records_merge_and_output_is_input_order_independent(self) -> None:
        initial = "------------------XXXO----OOXO-----OOX----O-OO------------------ X"
        board = Board.from_text(initial)
        first = board.legal_moves()[0]
        board.play(first)
        second = board.legal_moves()[0]
        transcript = build_book.index_to_coord(first) + build_book.index_to_coord(second)
        leaf_empty = N_CELLS - replay(initial, transcript).n_discs()
        records = [
            Record(
                build_book.transform_board_text(initial, symmetry),
                build_book.transform_transcript(transcript, symmetry),
                6,
                leaf_empty,
            )
            for symmetry in range(build_book.N_SYMMETRIES)
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
        leaf_nodes = {
            find_node(root, record.initial_board, record.transcript).board_key
            for record in records
        }
        self.assertEqual(len(leaf_nodes), 1)
        leaf = find_node(root, records[0].initial_board, records[0].transcript)
        self.assertEqual(leaf.leaf_values, [6] * build_book.N_SYMMETRIES)
        assert_canonical_graph(self, root)
        assert_canonical_legal_lines(self, lines)

        _, reversed_lines = build(list(reversed(records)))
        self.assertEqual(lines, reversed_lines)

    def test_forced_pass_edge_uses_canonical_child(self) -> None:
        initial = standard_board_text()
        board = Board.from_text(initial)
        transcript_parts: list[str] = []
        found_pass = False
        for _ in range(N_CELLS):
            legal_moves = board.legal_moves()
            if not legal_moves:
                if board.is_end():
                    break
                found_pass = True
                board.pass_turn()
                legal_moves = board.legal_moves()
            move = legal_moves[0]
            transcript_parts.append(build_book.index_to_coord(move))
            board.play(move)
            if found_pass:
                break
        self.assertTrue(found_pass, "deterministic test line did not reach a forced pass")

        transcript = "".join(transcript_parts)
        leaf_empty = N_CELLS - board.n_discs()
        root = new_root(initial)
        self.assertTrue(build_book.add_record(root, Record(initial, transcript, 0, leaf_empty)))

        replay_board = Board.from_text(initial)
        node = root
        saw_pass_edge = False
        for pos in range(0, len(transcript), 2):
            if not replay_board.legal_moves() and not replay_board.is_end():
                passed = replay_board.copy()
                passed.pass_turn()
                child = node.children[build_book.PASS_MOVE]
                self.assertEqual(child.board_key, build_book.canonicalize_board(passed)[0])
                node = child
                replay_board.pass_turn()
                saw_pass_edge = True
            move = coord_to_index(transcript[pos:pos + 2])
            _, symmetry = build_book.canonicalize_board(replay_board)
            node = node.children[build_book.move_to_representative(move, symmetry)]
            replay_board.play(move)
        self.assertTrue(saw_pass_edge)
        assert_canonical_graph(self, root)

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

    def test_real_collision_books_rebuild_to_unique_legal_representatives(self) -> None:
        for book_index in (57, 465, 556):
            with self.subTest(book_index=book_index):
                trained_dir = THIS_DIR / "trained"
                matches = list(trained_dir.glob(f"{book_index:07d}_*.egcb"))
                self.assertEqual(len(matches), 1)
                initial = next(
                    line[len("# initial "):]
                    for line in matches[0].read_text(encoding="utf-8").splitlines()
                    if line.startswith("# initial ")
                )

                root, _, n_used, _ = build_book.load_records(
                    initial,
                    build_book.record_dir_for_start(initial),
                    True,
                )
                self.assertGreater(n_used, 0)
                build_book.solve_node(root)
                lines: list[str] = []
                build_book.collect_book_lines(
                    root,
                    Board.from_text(initial),
                    0,
                    build_book.DEFAULT_BOOK_MAX_LOSS,
                    30,
                    lines,
                )
                self.assertGreater(len(lines), 0)
                assert_canonical_graph(self, root)
                assert_canonical_legal_lines(self, lines)


if __name__ == "__main__":
    unittest.main()
