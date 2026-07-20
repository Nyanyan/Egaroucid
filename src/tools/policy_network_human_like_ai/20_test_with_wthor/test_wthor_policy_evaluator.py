#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_wthor_human_match import (
    POLICY_SIZE,
    choose_data_split_positions,
    equivalent_policy_mask,
    split_counts,
    symmetry_aware_policy_ranks,
)
from policy_accuracy import xy_to_policy


class PolicyEvaluatorTest(unittest.TestCase):
    def test_split_matches_training_counts(self) -> None:
        self.assertEqual(
            (6_428_225, 803_528, 803_529),
            split_counts(8_035_282, 0.8, 0.1, 0.1),
        )

    def test_split_is_disjoint_complete_and_deterministic(self) -> None:
        parts = {}
        for name in ("train", "val", "test"):
            selected, count = choose_data_split_positions(
                100,
                name,
                0.8,
                0.1,
                0.1,
                613,
            )
            self.assertEqual(count, len(selected))
            parts[name] = set(int(value) for value in selected)

        self.assertFalse(parts["train"] & parts["val"])
        self.assertFalse(parts["train"] & parts["test"])
        self.assertFalse(parts["val"] & parts["test"])
        self.assertEqual(set(range(100)), set.union(*parts.values()))

        repeated, _ = choose_data_split_positions(
            100,
            "test",
            0.8,
            0.1,
            0.1,
            613,
        )
        self.assertEqual(sorted(parts["test"]), repeated.tolist())

    def test_equal_probabilities_have_unique_fixed_ranks(self) -> None:
        probabilities = np.zeros((1, POLICY_SIZE), dtype=np.float32)
        probabilities[0, 4] = 0.75
        probabilities[0, 9] = 0.75
        legal_mask = np.zeros((1, POLICY_SIZE), dtype=bool)
        legal_mask[0, [4, 9]] = True
        policies = np.array([9], dtype=np.int64)
        equivalent_mask = np.zeros((1, POLICY_SIZE), dtype=bool)
        equivalent_mask[0, 9] = True

        rank = symmetry_aware_policy_ranks(
            probabilities,
            legal_mask,
            equivalent_mask,
        )

        self.assertEqual(2, int(rank[0]))

    def test_symmetric_rank_uses_best_equivalent_move(self) -> None:
        probabilities = np.zeros((1, POLICY_SIZE), dtype=np.float32)
        probabilities[0, 4] = 0.75
        probabilities[0, 9] = 0.25
        legal_mask = np.zeros((1, POLICY_SIZE), dtype=bool)
        legal_mask[0, [4, 9]] = True
        policies = np.array([9], dtype=np.int64)
        equivalent_mask = np.zeros((1, POLICY_SIZE), dtype=bool)
        equivalent_mask[0, [4, 9]] = True

        rank = symmetry_aware_policy_ranks(
            probabilities,
            legal_mask,
            equivalent_mask,
        )

        self.assertEqual(1, int(rank[0]))

    def test_initial_board_reflection_makes_moves_equivalent(self) -> None:
        def bit(x: int, y: int) -> np.uint64:
            return np.uint64(1) << np.uint64(xy_to_policy(x, y))

        player = np.array(
            [bit(3, 4) | bit(4, 3)],
            dtype=np.uint64,
        )
        opponent = np.array(
            [bit(3, 3) | bit(4, 4)],
            dtype=np.uint64,
        )
        human_policy = xy_to_policy(3, 2)
        reflected_policy = xy_to_policy(2, 3)

        equivalent = equivalent_policy_mask(
            player,
            opponent,
            np.array([human_policy], dtype=np.int64),
        )

        self.assertTrue(bool(equivalent[0, human_policy]))
        self.assertTrue(bool(equivalent[0, reflected_policy]))


if __name__ == "__main__":
    unittest.main()
