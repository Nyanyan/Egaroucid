#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_wthor_blend_human_match import rank_for_distribution


class RankForDistributionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.distribution = np.zeros(64, dtype=np.float32)
        self.distribution[[10, 20, 30]] = [0.5, 0.5, 0.25]
        self.legal = [10, 20, 30]

    def test_equal_probability_moves_have_unique_ranks(self) -> None:
        self.assertEqual(1, rank_for_distribution(self.distribution, self.legal, [10]))
        self.assertEqual(2, rank_for_distribution(self.distribution, self.legal, [20]))
        self.assertEqual(3, rank_for_distribution(self.distribution, self.legal, [30]))

    def test_hint_order_breaks_alpha_zero_ties(self) -> None:
        hint_order = [20, 10, 30]
        self.assertEqual(
            1,
            rank_for_distribution(
                self.distribution,
                self.legal,
                [20],
                hint_order,
            ),
        )
        self.assertEqual(
            2,
            rank_for_distribution(
                self.distribution,
                self.legal,
                [10],
                hint_order,
            ),
        )


if __name__ == "__main__":
    unittest.main()
