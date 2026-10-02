import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import train_policy_network as train
from paper_policy_models import CENTER_POLICIES, PAPER_POLICIES, expand_paper_policy


class PaperPolicyModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train.enable_memory_growth()

    def tearDown(self):
        tf.keras.backend.clear_session()

    def specs(self, configs):
        args = train.make_argparser().parse_args(["--configs", configs])
        return train.parse_model_specs(args.configs, args)

    def test_paper_presets_and_mlp_can_be_selected_together(self):
        specs = self.specs("96x3,Conv4,conv6,conv8,Conv8+BN")
        self.assertEqual(["mlp", "conv4", "conv6", "conv8", "conv8_bn"], [s.architecture for s in specs])
        self.assertEqual([3, 4, 6, 8, 8], [s.depth for s in specs])
        self.assertEqual([96, 128, 128, 128, 128], [s.width for s in specs])
        with self.assertRaisesRegex(ValueError, "distinct output names"):
            self.specs("conv8_bn,Conv8+BN")

    def test_convolution_shapes_match_paper_table_one(self):
        expected = {
            "conv4": [64, 64, 128, 128],
            "conv6": [64, 64, 128, 128, 256, 256],
            "conv8": [64, 64, 128, 128, 256, 256, 256, 256],
            "conv8_bn": [64, 64, 128, 128, 256, 256, 256, 256],
        }
        for name, channels in expected.items():
            with self.subTest(name=name):
                model = train.build_model(self.specs(name)[0])
                convs = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
                self.assertEqual(channels, [layer.filters for layer in convs])
                for layer in convs:
                    self.assertEqual((3, 3), layer.kernel_size)
                    self.assertEqual((1, 1), layer.strides)
                    self.assertEqual((8, 8), layer.output_shape[1:3])
                    self.assertEqual("same", layer.padding)
                    self.assertIsInstance(layer.kernel_initializer, tf.keras.initializers.HeNormal)
                self.assertEqual([128, 60], [layer.units for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)])
                self.assertEqual((None, 128), model.input_shape)
                self.assertEqual((None, 64), model.output_shape)
                norms = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
                self.assertEqual(9 if name == "conv8_bn" else 0, len(norms))
                for layer in model.layers:
                    self.assertNotIsInstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D))
                if name == "conv8_bn":
                    self.assertIsInstance(model.get_layer("conv_relu_0").input._keras_history.layer, tf.keras.layers.BatchNormalization)
                    self.assertIsInstance(model.get_layer("fc128_relu").input._keras_history.layer, tf.keras.layers.BatchNormalization)
                tf.keras.backend.clear_session()

    def test_input_planes_preserve_square_and_side_order(self):
        model = train.build_model(self.specs("conv4")[0])
        encoder = tf.keras.Model(model.input, model.get_layer("pieces_channels_last").output)
        features = np.arange(128, dtype=np.float32).reshape(1, 128)
        encoded = encoder(features).numpy()
        np.testing.assert_array_equal(encoded[0, :, :, 0], features[0, :64].reshape(8, 8))
        np.testing.assert_array_equal(encoded[0, :, :, 1], features[0, 64:].reshape(8, 8))

    def test_sixty_way_output_maps_all_moves_without_extra_parameters(self):
        inputs = tf.keras.Input((60,))
        mapper = tf.keras.Model(inputs, expand_paper_policy(inputs))
        scores = np.arange(1, 121, dtype=np.float32).reshape(2, 60)
        probabilities = scores / scores.sum(axis=1, keepdims=True)
        actual = mapper(probabilities).numpy()
        self.assertEqual(0, mapper.count_params())
        np.testing.assert_array_equal(actual[:, PAPER_POLICIES], probabilities)
        np.testing.assert_array_equal(actual[:, CENTER_POLICIES], 0.0)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0)
        labels60 = np.array([27, 33])  # Immediately after each omitted center pair.
        labels64 = np.array(PAPER_POLICIES)[labels60]
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        np.testing.assert_allclose(loss(labels60, probabilities).numpy(), loss(labels64, actual).numpy(), rtol=1e-6)

    def test_bn_trains_and_h5_reloads_without_custom_objects(self):
        model = train.build_model(self.specs("conv8_bn")[0])
        features = np.random.default_rng(613).integers(0, 2, size=(2, 128)).astype(np.float32)
        before = model.get_layer("policy_logits60").get_weights()[0].copy()
        loss = model.train_on_batch(features, np.array([19, 44]))
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(before, model.get_layer("policy_logits60").get_weights()[0]))
        self.assertTrue(np.any(model.get_layer("conv_bn_0").moving_mean.numpy() != 0))
        expected = model(features, training=False).numpy()
        np.testing.assert_allclose(expected.sum(axis=1), 1.0, rtol=1e-6)
        np.testing.assert_array_equal(expected[:, CENTER_POLICIES], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.h5"
            model.save(path)
            reloaded = tf.keras.models.load_model(path)
            np.testing.assert_allclose(reloaded(features, training=False).numpy(), expected, rtol=1e-5, atol=1e-7)

    def test_binary_export_rejects_cnn_without_writing_a_file(self):
        spec = self.specs("conv4")[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            with self.assertRaisesRegex(ValueError, "only MLP"):
                train.export_binary_weights(None, spec, path)
            self.assertFalse(path.exists())

    def test_selected_cnn_clears_old_binary_and_mlp_still_exports(self):
        spec = self.specs("8x1")[0]
        model = train.build_model(spec)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            model.save(run / "model.h5")
            weights = run / "policy_network_weights.bin"
            train.export_binary_weights(model, spec, weights)
            result = {"output_dir": str(run), "binary_weights": str(weights)}
            train.save_selected_model(result, root)
            self.assertEqual(weights.read_bytes(), (root / "selected_policy_network_weights.bin").read_bytes())
            result["binary_weights"] = None
            train.save_selected_model(result, root)
            self.assertFalse((root / "selected_policy_network_weights.bin").exists())
            self.assertTrue((root / "selected_model.h5").exists())
            self.assertIsNone(json.loads((root / "selected_summary.json").read_text())["binary_weights"])

    def test_evaluator_accepts_h5_without_binary_weights(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20_test_with_wthor"))
        from evaluate_wthor_human_match import parse_args

        args = parse_args(["--board-data-dir", "data", "--model", "conv8.h5"])
        self.assertEqual(Path("conv8.h5"), args.model)
        self.assertIsNone(args.weights)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--board-data-dir", "data"])


if __name__ == "__main__":
    unittest.main()
