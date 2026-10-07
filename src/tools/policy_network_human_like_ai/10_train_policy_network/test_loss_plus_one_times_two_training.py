"""Verify the corrected formula and its equivalence after normalization."""
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
from train_loss_plus_one import linear_weights, prepare_dataset
from train_loss_plus_one_times_two import WEIGHTING


class LossPlusOneTimesTwoTest(unittest.TestCase):
    def test_all_losses_and_explicit_examples(self):
        losses = np.arange(129)
        np.testing.assert_array_equal((losses + 1) * 2, linear_weights(np.exp2(losses / 2), WEIGHTING))
        np.testing.assert_array_equal([2, 4, 6, 10, 14],
                                     linear_weights(np.exp2(np.array([0, 1, 2, 4, 6]) / 2), WEIGHTING))

    def test_preserved_data_and_identical_normalized_weights_to_loss_plus_one(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, target = root / "source", root / "target"
            previous = make_fixture(source)
            original = json.loads((source / "manifest.json").read_text())
            manifest = prepare_dataset(source, target, WEIGHTING)
            self.assertEqual(manifest, prepare_dataset(source, target, WEIGHTING))
            self.assertEqual(WEIGHT_FORMAT, manifest["sample_weights"]["format"])
            self.assertEqual("(disc_loss + 1) * 2", manifest["sample_weights"]["formula"])
            before = {entry["path"]: entry for entry in original["files"]}
            for entry in manifest["files"]:
                if not entry["path"].endswith(".weights"):
                    self.assertEqual(before[entry["path"]], entry)
            splits = data.load_prepared_splits(target, {}, 613, root / "train")
            try:
                for name, split in splits.items():
                    actual = split.read_weights(0, split.positions)
                    np.testing.assert_array_equal(2 * previous[name], actual)
                    self.assertEqual(2 * previous["train"].mean(), split.weight_scale)
                    np.testing.assert_array_equal(previous[name] / previous["train"].mean(),
                                                  actual / split.weight_scale)
                    self.assertEqual(2, actual.min())
                    self.assertEqual(actual.sum(), manifest["splits"][name]["weight_sum"])
            finally:
                for split in splits.values():
                    split.close()


if __name__ == "__main__":
    unittest.main()
