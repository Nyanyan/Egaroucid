"""Verify exact twice-loss conversion and zero-weight training semantics."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import prepared_policy_data as data
from pipeline_common import NONNEGATIVE_WEIGHT_FORMAT, WEIGHT_DTYPE, WEIGHT_FORMAT, validate_weight_file, write_manifest
from test_loss_plus_one_training import make_fixture
from train_loss_plus_one import linear_weights, prepare_dataset
from train_loss_times_two import WEIGHTING


class TwiceLossTrainingTest(unittest.TestCase):
    def test_exact_integer_losses_and_zero(self):
        losses = np.arange(129)
        np.testing.assert_array_equal(2 * losses, linear_weights(np.exp2(losses / 2), WEIGHTING))

    def test_partition_and_weight_alignment_train_only_normalization(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, target = root / "source", root / "target"
            previous = make_fixture(source)
            manifest = prepare_dataset(source, target, WEIGHTING)
            self.assertEqual(manifest, prepare_dataset(source, target, WEIGHTING))
            original = json.loads((source / "manifest.json").read_text())
            originals = {entry["path"]: entry for entry in original["files"]}
            for entry in manifest["files"]:
                if not entry["path"].endswith(".weights"):
                    self.assertEqual(originals[entry["path"]], entry)
            self.assertEqual(NONNEGATIVE_WEIGHT_FORMAT, manifest["sample_weights"]["format"])
            self.assertEqual("2 * disc_loss", manifest["sample_weights"]["formula"])
            splits = data.load_prepared_splits(target, {}, 613, root / "train")
            try:
                scale = (2 * (previous["train"] - 1)).mean()
                for name, split in splits.items():
                    expected = 2 * (previous[name] - 1)
                    np.testing.assert_array_equal(expected, split.read_weights(0, split.positions))
                    self.assertEqual(scale, split.weight_scale)
                    self.assertTrue(np.any(expected == 0))
                    self.assertEqual(expected.sum(), manifest["splits"][name]["weight_sum"])
                # Real Keras batches preserve zeros and apply the weighted loss.
                import tensorflow as tf
                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)
                inputs = tf.keras.Input(shape=(128,))
                outputs = tf.keras.layers.Dense(64, activation="softmax", kernel_initializer="zeros",
                                                bias_initializer="zeros")(inputs)
                model = tf.keras.Model(inputs, outputs)
                model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
                val_mean = (2 * (previous["val"] - 1)).mean()
                actual = model.evaluate(splits["val"].as_dataset(32), verbose=0)
                self.assertAlmostEqual(np.log(64) * val_mean / scale, actual, places=5)
            finally:
                for split in splits.values():
                    split.close()

    def test_zero_needs_explicit_format_and_negative_is_always_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            np.array([0, 2, 4], dtype=WEIGHT_DTYPE).tofile(path)
            self.assertEqual(6, validate_weight_file(path, 3, allow_zero=True))
            with self.assertRaisesRegex(ValueError, "positive"):
                validate_weight_file(path, 3)
            np.array([-1, 2, 4], dtype=WEIGHT_DTYPE).tofile(path)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                validate_weight_file(path, 3, allow_zero=True)

    def test_all_zero_training_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, target = root / "source", root / "target"
            make_fixture(source)
            manifest = prepare_dataset(source, target, WEIGHTING)
            for path in (target / "train").glob("*.weights"):
                np.zeros(path.stat().st_size // 8, dtype=WEIGHT_DTYPE).tofile(path)
            metadata = {k: v for k, v in manifest.items() if k not in ("version", "stage", "dataset", "files")}
            write_manifest(target, manifest["stage"], manifest["dataset"], **metadata)
            with self.assertRaisesRegex(ValueError, "positive mean"):
                data.load_prepared_splits(target, {}, 613, root / "train")


if __name__ == "__main__":
    unittest.main()
