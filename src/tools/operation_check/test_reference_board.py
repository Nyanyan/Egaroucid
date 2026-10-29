"""Known positions and rejection paths for the independent rules oracle."""

import unittest

from common import CheckFailure
from reference_board import Board


class ReferenceBoardTests(unittest.TestCase):
    def test_known_initial_and_reply(self):
        board = Board()
        self.assertEqual(len(board.cells), 64)
        self.assertEqual(board.legal(), {"c4", "d3", "e6", "f5"})
        board.play("f5")
        self.assertEqual(board.turn, "W")
        self.assertEqual(board.cells.count("B"), 4)
        self.assertEqual(board.cells.count("W"), 1)
        self.assertEqual(board.legal(), {"d6", "f4", "f6"})

    def test_illegal_move_and_pass(self):
        for move in ("a1", "d4", "ps"):
            with self.subTest(move=move), self.assertRaises(CheckFailure):
                Board().play(move)

    def test_terminal_position(self):
        board = Board()
        board.cells = list("-W" + "B" * 62)
        self.assertEqual(board.legal(), {"a1"})
        board.play("a1")
        self.assertTrue(board.ended())
        self.assertEqual(board.cells.count("B"), 64)


if __name__ == "__main__":
    unittest.main()
