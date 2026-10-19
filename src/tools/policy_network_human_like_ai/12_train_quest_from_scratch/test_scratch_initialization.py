"""Verify fresh weights, seed reproducibility and no checkpoint loading at initialization."""
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
from train_quest_scratch import fresh_model, parser


class ScratchInitializationTests(unittest.TestCase):
    def test_random_initialization_is_repeatable_without_loading_a_model(self):
        import tensorflow as tf
        from train_policy_network import enable_memory_growth
        enable_memory_growth()
        args = parser().parse_args([])
        digests = []
        with patch.object(tf.keras.models, 'load_model', side_effect=AssertionError('Pretrained weights loaded')):
            for seed in (613, 613, 614):
                tf.keras.backend.clear_session()
                args.seed = seed
                model = fresh_model(args)
                self.assertEqual(model.count_params(), 4434620)
                self.assertEqual(int(model.optimizer.iterations.numpy()), 0)
                bn = model.get_layer('conv_bn_0')
                np.testing.assert_array_equal(bn.moving_mean.numpy(), 0)
                np.testing.assert_array_equal(bn.moving_variance.numpy(), 1)
                kernel = model.get_layer('conv_0').kernel.numpy()
                self.assertGreater(float(kernel.std()), 0)
                digests.append(hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest())
        self.assertEqual(digests[0], digests[1])
        self.assertNotEqual(digests[0], digests[2])


if __name__ == '__main__':
    unittest.main()
