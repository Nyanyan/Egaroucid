"""Small, CPU-only checks for the rate grid and full-test Quest evaluation."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import evaluate_dropout_quest as quest
import run_dropout_sweep as sweep
from sweep_common import BANDS, RATES, read
from board_test_support import make_game, legal_moves
from pipeline_common import BoardWriter, write_manifest


class SweepTest(unittest.TestCase):
    def fixture(self, root, band="800-1000"):
        directory = root / "board_split" / band
        counts = {}
        low, high = map(int, band.split("-"))
        for index, name in enumerate(("train", "val", "test")):
            records, _ = make_game(index + low, max_moves=8)
            writer = BoardWriter(directory / name, 100)
            writer.write(records, np.full(len(records), index, dtype="<u8"))
            writer.close()
            counts[name] = dict(positions=len(records), games=1)
        write_manifest(directory, "board_split", band, splits=counts, augmentation_factor=1,
            rating_range=dict(min_inclusive=low, max_exclusive=high),
            split_unit="identical_transcript_group_across_all_rating_bands")
        return records

    def test_grid_and_inference_defaults(self):
        self.assertEqual((0, .2, .4, .6, .8), RATES)
        self.assertEqual(("800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-2200"), BANDS)
        args = sweep.make_argparser().parse_args([])
        self.assertEqual((512, 613), (args.batch_size, args.seed))
        self.assertFalse(hasattr(args, 'epochs'))

    def test_legal_top1_and_initial_board_equivalence(self):
        records, _ = make_game(613, max_moves=40)
        probabilities = np.zeros((len(records), 64), dtype=np.float32)
        probabilities[np.arange(len(records)), records["policy"]] = 1
        for index, record in enumerate(records):
            legal = legal_moves(int(record["player"]), int(record["opponent"]))
            illegal = next(move for move in range(64) if not (legal & (1 << move)))
            probabilities[index, illegal] = 1000
        legal = legal_moves(int(records[0]["player"]), int(records[0]["opponent"]))
        alternative = next(move for move in range(64) if legal & (1 << move) and move != records[0]["policy"])
        probabilities[0, alternative] = 2
        moves, masks, hits = quest.score_records(records, probabilities)
        self.assertTrue(hits.all())
        self.assertNotEqual(records[0]["policy"], moves[0])
        self.assertEqual(legal, int(masks[0]))
        for record, move in zip(records, moves):
            self.assertTrue(legal_moves(int(record["player"]), int(record["opponent"])) & (1 << int(move)))
        changed = False
        for index in range(1, len(records)):
            legal = legal_moves(int(records[index]["player"]), int(records[index]["opponent"]))
            alternatives = [move for move in range(64) if legal & (1 << move) and not int(masks[index]) & (1 << move)]
            if alternatives:
                probabilities[index, alternatives[0]] = 3
                changed = True
                break
        self.assertTrue(changed)
        self.assertEqual(len(records)-1, int(quest.score_records(records, probabilities)[2].sum()))

    def test_full_band_evidence_cache_and_corruption_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self.fixture(root)
            paths, info = quest.inspect_band(root, "800-1000")
            probabilities = np.zeros((len(records), 64), dtype=np.float32)
            probabilities[np.arange(len(records)), records["policy"]] = 1
            model = mock.Mock()
            model.predict.return_value = probabilities
            evidence = dict(dropout_rate=.2, adopted_epoch=3, model_sha256="model")
            output = root / "results/800-1000"
            result = quest.evaluate_band(model, paths, info, output, 4, 1, evidence)
            self.assertEqual((8, 8, 100), (result["positions"], result["top1_hits"], result["top1_accuracy_percent"]))
            self.assertEqual("complete", read(output / "summary.json")["status"])
            model.predict.side_effect = AssertionError("Cache should avoid repeat prediction")
            self.assertEqual(result, quest.evaluate_band(model, paths, info, output, 4, 1, evidence))
            (output / "predictions.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Saved evaluation changed"):
                quest.evaluate_band(model, paths, info, output, 4, 1, evidence)

    def test_preflight_and_cross_model_sample_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for band in BANDS:
                self.fixture(root / "othello_quest_split", band)
            args = sweep.make_argparser().parse_args(["--quest-root", str(root / 'othello_quest_split'),
                                                      "--output-dir", str(root / "output")])
            identity = sweep.build_identity(args)
            self.assertEqual(7, len(identity["datasets"]))
            self.assertEqual(56, sum(item["positions"] for item in identity["datasets"]))
            self.assertEqual('inference_only_no_training', identity['operation'])
            args.output_dir.mkdir()
            row = dict(dropout_rate=.2, rating_band=BANDS[0], positions=8, test_records_sha256="same",
                       top1_accuracy_percent=50)
            sweep.write_results(args.output_dir, [row], identity)
            with self.assertRaisesRegex(ValueError, "different test records"):
                sweep.write_results(args.output_dir, [row, dict(row, dropout_rate=.4, test_records_sha256="other")], identity)


class InferenceDropoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

    def test_zero_reproduces_the_pretrained_model(self):
        import tensorflow as tf
        from dropout_model import build_inference_dropout_model
        base = tf.keras.models.load_model(sweep.DEFAULT_MODEL, compile=False)
        model = build_inference_dropout_model(base, 0)
        reference = build_inference_dropout_model(base, 0)
        features = np.random.default_rng(613).integers(0, 2, (4, 128)).astype(np.float32)
        np.testing.assert_array_equal(model(features, training=False).numpy(), reference(features, training=False).numpy())
        for expected, actual in zip(base.get_weights(), model.get_weights()):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual(base.count_params(), model.count_params())

    def test_only_dropout_is_active_and_h5_preserves_it(self):
        import tensorflow as tf
        from dropout_model import build_inference_dropout_model
        print('[verification] load original WTHOR model', flush=True)
        base = tf.keras.models.load_model(sweep.DEFAULT_MODEL, compile=False)
        print('[verification] clone fixed weights and attach active Dropout', flush=True)
        before = base.get_weights()
        model = build_inference_dropout_model(base, .4, 613)
        print('[verification] verify stochastic inference and frozen weights', flush=True)
        features = np.random.default_rng(613).integers(0, 2, (4, 128)).astype(np.float32)
        predictor = sweep.LogitPredictor(model)
        first = predictor.predict(features, batch_size=4)
        second = predictor.predict(features, batch_size=4)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(0, len(model.trainable_weights))
        for expected, actual in zip(before, model.get_weights()):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual('float32', model.get_layer('conv_0').dtype_policy.name)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'dropout.h5'
            model.save(path, include_optimizer=False)
            restored = tf.keras.models.load_model(path, compile=False)
            predictor = sweep.LogitPredictor(restored)
            self.assertFalse(np.array_equal(predictor.predict(features, batch_size=4), predictor.predict(features, batch_size=4)))
            for expected, actual in zip(before, restored.get_weights()):
                np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    import faulthandler
    faulthandler.dump_traceback_later(30, repeat=True)
    unittest.main(verbosity=2)
