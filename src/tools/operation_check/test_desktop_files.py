from pathlib import Path
import struct
import tempfile
import unittest

from common import CheckFailure
from book_fixture import create_book, representative
from book_probe import compare_union, imported_positions, inspect_book
from reference_board import Board
from image_probe import inspect_image
from PIL import Image


class SavedFilesTests(unittest.TestCase):
    def test_book_fixture_is_deterministic_and_truncation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.egbk3"
            self.assertEqual(create_book(path), 90)
            first = path.read_bytes()
            self.assertEqual(inspect_book(path, 90)["positions"], 90)
            create_book(path)
            self.assertEqual(path.read_bytes(), first)
            path.write_bytes(first[:-1])
            with self.assertRaises(CheckFailure):
                inspect_book(path)

    def test_edax_record_count_is_not_just_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.dat"
            header = b"XADEKOOB\x04\x04" + bytes(28) + struct.pack("<I", 1)
            record = struct.pack("<QQ", 0b0011, 0b1100) + bytes(24) + b"\x7f\x40"
            path.write_bytes(header + record)
            self.assertEqual(inspect_book(path, 1)["format"], "edax")
            path.write_bytes(header + record + b"unexpected")
            with self.assertRaises(CheckFailure):
                inspect_book(path)

    def test_edax_leaf_is_promoted_and_a_missing_child_fails_union_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            board = Board()
            player = sum(1 << (63 - i) for i, disc in enumerate(board.cells) if disc == "B")
            opponent = sum(1 << (63 - i) for i, disc in enumerate(board.cells) if disc == "W")
            edax = path / "input.dat"
            edax.write_bytes(b"XADEKOOB\x04\x04" + bytes(28) + struct.pack("<I", 1) +
                             struct.pack("<QQ", player, opponent) + bytes(24) + bytes((0, 26))) # F5 leaf
            expected = imported_positions(edax)
            self.assertEqual(len(expected), 2)
            board.play("f5")
            child_player = sum(1 << (63 - i) for i, disc in enumerate(board.cells) if disc == "W")
            child_opponent = sum(1 << (63 - i) for i, disc in enumerate(board.cells) if disc == "B")
            self.assertIn(representative(child_player, child_opponent), expected)
            merged = path / "merged.egbk3"
            def write_positions(positions):
                merged.write_bytes(b"DICUORAGE\x03" + struct.pack("<I", len(positions)) + b"".join(
                    struct.pack("<QQbbIbbb", p, o, 0, 1, 1, 127, 125, -1) for p, o in positions))
            write_positions(sorted(expected))
            self.assertEqual(compare_union(merged, [edax])["expected_positions"], 2)
            write_positions([representative(player, opponent)])
            with self.assertRaises(CheckFailure):
                compare_union(merged, [edax])

    def test_pixel_alpha_and_clipboard_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            with path.open("wb") as stream:
                Image.new("RGBA", (2, 2), (36, 153, 114, 0)).save(stream, format="PNG")
            result = inspect_image(path, [[1, 1, 36, 153, 114]], [2, 2], [[0, 0, 0]])
            self.assertTrue(result["passed"])
            self.assertFalse(inspect_image(path, [[1, 1, 0, 0, 0]])["passed"])
            self.assertFalse(inspect_image(path, [], alphas=[[0, 0, 255]])["passed"])
            self.assertFalse(inspect_image(path, [], clipboard={"png": {"sha256": "stale"}})["passed"])


if __name__ == "__main__":
    unittest.main()
