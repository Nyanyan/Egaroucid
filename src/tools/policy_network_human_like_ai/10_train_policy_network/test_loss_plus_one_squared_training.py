"""Verify squared loss weights, unchanged partitions and normalized ratios."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import prepared_policy_data as data
from pipeline_common import WEIGHT_FORMAT
from test_loss_plus_one_training import make_fixture
from train_loss_plus_one import loss_weights, prepare_dataset
from train_loss_plus_one_squared import WEIGHTING


class LossPlusOneSquaredTest(unittest.TestCase):
    def test_all_losses_and_explicit_examples(self):
        losses = np.arange(129)
        np.testing.assert_array_equal((losses + 1) ** 2, loss_weights(np.exp2(losses / 2), WEIGHTING))
        np.testing.assert_array_equal([1, 4, 9, 25, 49],
                                     loss_weights(np.exp2(np.array([0, 1, 2, 4, 6]) / 2), WEIGHTING))

    def test_partitions_unchanged_and_squared_ratios_survive_normalization(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, target = root / "source", root / "target"
            previous = make_fixture(source)
            original = json.loads((source / "manifest.json").read_text())
            manifest = prepare_dataset(source, target, WEIGHTING)
            self.assertEqual(manifest, prepare_dataset(source, target, WEIGHTING))
            self.assertEqual(WEIGHT_FORMAT, manifest["sample_weights"]["format"])
            self.assertEqual("(disc_loss + 1) ** 2", manifest["sample_weights"]["formula"])
            before = {entry["path"]: entry for entry in original["files"]}
            for entry in manifest["files"]:
                if not entry["path"].endswith(".weights"):
                    self.assertEqual(before[entry["path"]], entry)
            splits = data.load_prepared_splits(target, {}, 613, root / "train")
            try:
                scale = (previous["train"] ** 2).mean()
                for name, split in splits.items():
                    actual = split.read_weights(0, split.positions)
                    np.testing.assert_array_equal(previous[name] ** 2, actual)
                    self.assertEqual(scale, split.weight_scale)
                    normalized = actual / split.weight_scale
                    np.testing.assert_array_equal(previous[name] ** 2 / scale, normalized)
                    self.assertFalse(np.allclose(previous[name] / previous["train"].mean(), normalized))
                    base = normalized[previous[name] == 1][0]
                    self.assertAlmostEqual(9, normalized[previous[name] == 3][0] / base)
                    self.assertEqual(1, actual.min())
                    self.assertEqual(actual.sum(), manifest["splits"][name]["weight_sum"])
            finally:
                for split in splits.values():
                    split.close()


if __name__ == "__main__":
    unittest.main()
