"""Verify exponential inversion, preserved partitions, and train-only scaling."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import train_loss_plus_one as launcher
import prepared_policy_data as data
from board_test_support import make_game
from pipeline_common import BoardWriter, SPLIT_NAMES, WEIGHT_FORMAT, load_manifest, write_manifest


def make_fixture(root):
    counts, expected = {}, {}
    for index, name in enumerate(SPLIT_NAMES):
        boards, _ = make_game(index + 10)
        # Include odd disc losses and retain all eight symmetry rows.
        losses = (np.arange(len(boards)) + index * 3) % 15
        expected[name] = np.repeat(losses + 1, 8)
        writer = BoardWriter(root / name, 113, weighted=True)
        try:
            writer.write(np.repeat(boards, 8), np.full(len(boards) * 8, index, dtype="<u8"),
                         np.repeat(np.exp2(losses / 2), 8))
        finally:
            writer.close()
        counts[name] = {"positions": len(boards) * 8, "source_positions": len(boards), "games": 1}
    write_manifest(root, "board_split_increased", "fixture_LOSS_WEIGHTED", splits=counts,
                   seed=613, augmentation_factor=8,
                   sample_weights={"format": WEIGHT_FORMAT, "suffix": ".weights",
                                   "formula": launcher.SOURCE_FORMULA, "normalization": "none in files",
                                   "clipping": None})
    return expected


class LossPlusOneTest(unittest.TestCase):
    def test_all_integer_losses_including_odd_and_extremes(self):
        losses = np.arange(129)
        np.testing.assert_array_equal(losses + 1, launcher.linear_weights(np.exp2(losses / 2)))
        np.testing.assert_array_equal([1, 2, 3, 5, 7], launcher.linear_weights(np.exp2(np.array([0, 1, 2, 4, 6]) / 2)))
        for bad in ([0], [-1], [.5], [float("nan")], [float("inf")], [2.0 ** 65], [1.1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                launcher.linear_weights(bad)

    def test_conversion_preserves_boards_ids_splits_and_training_ratios(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, target = root / "source", root / "target"
            expected = make_fixture(source)
            original = load_manifest(source, "board_split_increased")
            manifest = launcher.prepare_dataset(source, target)
            self.assertEqual(manifest, launcher.prepare_dataset(source, target))
            self.assertEqual(original, load_manifest(source, "board_split_increased"))
            self.assertEqual(launcher.TARGET_FORMULA, manifest["sample_weights"]["formula"])
            self.assertEqual(8, manifest["augmentation_factor"])
            before = {entry["path"]: entry for entry in original["files"]}
            for entry in manifest["files"]:
                if not entry["path"].endswith(".weights"):
                    self.assertEqual(before[entry["path"]], entry)
            splits = data.load_prepared_splits(target, {}, 613, root / "training")
            try:
                scale = expected["train"].mean()
                for name, split in splits.items():
                    weights = split.read_weights(0, split.positions)
                    np.testing.assert_array_equal(expected[name], weights)
                    self.assertEqual(scale, split.weight_scale)
                    self.assertEqual(int(expected[name].sum()), manifest["splits"][name]["weight_sum"])
                    np.testing.assert_array_equal(expected[name] / scale, weights / split.weight_scale)
            finally:
                for split in splits.values():
                    split.close()

    def test_rejects_wrong_formula_corrupt_source_and_inplace_conversion(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, target = root / "source", root / "target"
            make_fixture(source)
            with self.assertRaisesRegex(ValueError, "separate"):
                launcher.prepare_dataset(source, source)
            path = source / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["sample_weights"]["formula"] = launcher.TARGET_FORMULA
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unnormalized"):
                launcher.prepare_dataset(source, target)
            manifest["sample_weights"]["formula"] = launcher.SOURCE_FORMULA
            path.write_text(json.dumps(manifest))
            next((source / "train").glob("*.weights")).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                launcher.prepare_dataset(source, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
