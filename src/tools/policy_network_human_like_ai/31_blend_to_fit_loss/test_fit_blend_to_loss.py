"""Verify geometric Top-1, endpoint direction, legality and nonmonotonic fit."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import fit_blend_to_loss as fit


class BlendTests(unittest.TestCase):
    def test_compact_sweep_matches_full_legal_policy_and_losses(self):
        from fit_validation_model_pairs import compact_metric
        rng = np.random.default_rng(3159)
        strong = rng.normal(size=(40, 64)).astype('f4')
        weak = rng.normal(size=(40, 64)).astype('f4')
        legal = np.full(40, 1023, dtype=np.uint64)
        losses = rng.integers(0, 15, size=(40, 64)).astype('f4')
        possible = fit.possible_top1(strong, weak, legal)
        calculate, moves = compact_metric((strong, weak), possible, losses)
        for alpha in np.linspace(0, 1, 101):
            np.testing.assert_array_equal(moves(alpha)[0], fit.select_moves((strong, weak), legal, alpha))
            self.assertEqual(calculate(alpha), fit.metric((strong, weak), legal, losses, alpha))

    def test_possible_top1_covers_every_alpha_and_excludes_never_selected(self):
        from selected_loss import possible_top1
        rng = np.random.default_rng(315)
        strong = rng.normal(size=(30, 64)).astype('f4')
        weak = rng.normal(size=(30, 64)).astype('f4')
        legal = np.full(30, 63, dtype=np.uint64)
        possible = possible_top1(strong, weak, legal)
        for alpha in np.linspace(0, 1, 501):
            chosen = fit.select_moves((strong, weak), legal, alpha)
            self.assertTrue(np.all(possible[np.arange(30), chosen]))
        self.assertFalse(np.any(possible[:, 6:]))
        strong[0, :6] = [3, 2, 0, -100, -100, -100]
        weak[0, :6] = [0, 2, 3, -100, -100, -100]
        possible = possible_top1(strong[:1], weak[:1], legal[:1])
        np.testing.assert_array_equal(np.flatnonzero(possible[0]), [0, 1, 2])

    def test_endpoints_and_geometric_reference(self):
        from policy_opponent_gtp import geometric_model_distribution
        strong = np.zeros((2, 64), dtype=np.float32)
        weak = strong.copy()
        strong[:, [0, 1, 2]] = [3, 1, 100]
        weak[:, [0, 1, 2]] = [0, 4, 100]
        legal = np.array([3, 3], dtype=np.uint64)
        logits = (strong, weak)
        np.testing.assert_array_equal(fit.select_moves(logits, legal, 0), [0, 0])
        np.testing.assert_array_equal(fit.select_moves(logits, legal, 1), [1, 1])
        for alpha in [0, .2, .5, .9, 1]:
            distribution = geometric_model_distribution(strong[0], weak[0], [0, 1], alpha)
            self.assertEqual(fit.select_moves(logits, legal, alpha)[0], distribution.argmax())

    def test_fit_can_select_interior_when_endpoints_have_same_loss(self):
        strong = np.full((1, 64), -100, dtype=np.float32)
        weak = strong.copy()
        strong[0, :3] = [3, 2, 0]
        weak[0, :3] = [0, 2, 3]
        legal = np.array([7], dtype=np.uint64)
        losses = np.full((1, 64), np.nan)
        losses[0, :3] = [5, 1, 5]
        with tempfile.TemporaryDirectory() as directory:
            selected = fit.fit_alpha((strong, weak), legal, losses, 1, 10, Path(directory))
        self.assertEqual(selected['mean_disc_loss'], 1)
        self.assertGreater(selected['alpha'], 0)
        self.assertLess(selected['alpha'], 1)


if __name__ == '__main__':
    unittest.main()
