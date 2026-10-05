"""Sample weights stay aligned through sharding, caps, shuffle and Keras loss."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

import prepared_policy_data as data
from board_test_support import make_game
from pipeline_common import BoardWriter, WEIGHT_FORMAT, write_manifest


class WeightedTrainingTest(unittest.TestCase):
    def test_weights_follow_shuffled_capped_records_across_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boards = np.zeros(131079, dtype=data.BOARD_DTYPE)
            boards["player"] = np.arange(len(boards))
            weights = np.arange(len(boards), dtype=np.float64) + 1
            writer = BoardWriter(root, 40000, weighted=True)
            writer.write(boards, np.arange(len(boards), dtype="<u8"), weights)
            writer.close()
            for cap in (None, 70001):
                split = data.PreparedSplit(sorted(root.glob("*.dat")), cap, weighted=True)
                try:
                    seen = []
                    for samples, actual in split.batches(1000, shuffle=True, seed=17, include_weights=True):
                        np.testing.assert_array_equal(samples["player"] + 1, actual)
                        seen.extend(samples["player"].tolist())
                    expected = np.arange(len(boards)) if cap is None else split.indices
                    np.testing.assert_array_equal(sorted(seen), expected)
                finally:
                    split.close()

    def test_loader_normalizes_using_train_only_and_keras_applies_weights(self):
        import tensorflow as tf
        for device in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts, expected = {}, {}
            for index, name in enumerate(("train", "val", "test")):
                boards, _ = make_game(index + 10)
                weights = np.exp2(np.arange(len(boards)) % 9 / 2) * (index + 1)
                expected[name] = weights
                writer = BoardWriter(root / name, 17, weighted=True)
                writer.write(boards, np.full(len(boards), index, dtype="<u8"), weights)
                writer.close()
                counts[name] = {"positions": len(boards), "games": 1}
            write_manifest(root, "board_split", "fixture", splits=counts,
                           sample_weights={"format": WEIGHT_FORMAT, "suffix": ".weights", "formula": "2 ** (disc_loss / 2)"})
            with contextlib.redirect_stdout(io.StringIO()):
                splits = data.load_prepared_splits(root, {}, 613, root / "training")
            try:
                scale = expected["train"].mean()
                for name, split in splits.items():
                    self.assertAlmostEqual(scale, split.weight_scale)
                    actual = np.concatenate([batch[2] for batch in split.as_dataset(11).as_numpy_iterator()])
                    np.testing.assert_allclose(expected[name] / scale, actual)
                inputs = tf.keras.Input(shape=(128,))
                outputs = tf.keras.layers.Dense(64, activation="softmax", kernel_initializer="zeros", bias_initializer="zeros")(inputs)
                model = tf.keras.Model(inputs, outputs)
                model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
                # Uniform predictions have CE=log(64); non-unit val weights prove
                # that the third dataset component actually affects the loss.
                actual_loss = model.evaluate(splits["val"].as_dataset(11), verbose=0)
                self.assertAlmostEqual(np.log(64) * expected["val"].mean() / scale, actual_loss, places=5)
                history = model.fit(splits["train"].as_dataset(11, shuffle=True), epochs=1, verbose=0)
                self.assertTrue(np.isfinite(history.history["loss"][-1]))
            finally:
                for split in splits.values():
                    split.close()


if __name__ == "__main__":
    unittest.main()
