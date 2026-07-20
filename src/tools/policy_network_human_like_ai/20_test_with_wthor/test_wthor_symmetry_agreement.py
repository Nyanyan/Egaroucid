#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_wthor_blend_human_match import (
    equivalent_targets,
    make_worker_progress_event,
    rank_for_distribution,
)
from run_random_wthor_blend_experiment import (
    BLEND_PARAMS,
    make_blend_summary_rows,
    make_console_level_summary_row,
)


def bitboard(policies: list[int]) -> int:
    result = 0
    for policy in policies:
        result |= 1 << policy
    return result


def result_row(
    alpha: float,
    top_n: int,
    *,
    exact_hits: int,
    symmetric_hits: int,
    positions: int,
) -> dict:
    return {
        "blend_param": alpha,
        "top_n": top_n,
        "exact_hits": exact_hits,
        "symmetric_hits": symmetric_hits,
        "positions": positions,
        "exact_accuracy": exact_hits / positions,
        "symmetric_accuracy": symmetric_hits / positions,
    }


class SymmetryAgreementTest(unittest.TestCase):
    def test_reflected_human_move_is_an_equivalent_target(self) -> None:
        # This position is invariant only under the a-file/h-file reflection.
        player = bitboard([10, 13])
        opponent = bitboard([19, 20])

        self.assertEqual({0, 7}, set(equivalent_targets(player, opponent, 0)))

        distribution = np.zeros(64, dtype=np.float32)
        distribution[[0, 7, 8]] = [0.1, 0.8, 0.4]
        legal = [0, 7, 8]
        self.assertEqual(3, rank_for_distribution(distribution, legal, [0]))
        self.assertEqual(
            1,
            rank_for_distribution(
                distribution,
                legal,
                equivalent_targets(player, opponent, 0),
            ),
        )

    def test_blend_summary_uses_symmetry_aware_counts(self) -> None:
        topn = []
        for alpha in BLEND_PARAMS:
            topn.append(
                result_row(
                    alpha,
                    1,
                    exact_hits=0,
                    symmetric_hits=2,
                    positions=2,
                )
            )
            topn.append(
                result_row(
                    alpha,
                    3,
                    exact_hits=1,
                    symmetric_hits=2,
                    positions=2,
                )
            )

        rows = make_blend_summary_rows({"topn": topn})

        self.assertEqual(len(BLEND_PARAMS), len(rows))
        for row in rows:
            self.assertEqual(2, row["top1_hits"])
            self.assertEqual(1.0, row["top1_accuracy"])
            self.assertEqual(2, row["top3_hits"])
            self.assertEqual(1.0, row["top3_accuracy"])

    def test_console_summary_uses_symmetry_aware_counts(self) -> None:
        result = {
            "topn": [
                result_row(
                    0.0,
                    1,
                    exact_hits=0,
                    symmetric_hits=2,
                    positions=2,
                ),
                result_row(
                    0.0,
                    3,
                    exact_hits=1,
                    symmetric_hits=2,
                    positions=2,
                ),
            ]
        }

        row = make_console_level_summary_row(21, result, 1.0)

        self.assertEqual(2, row["top1_hits"])
        self.assertEqual(1.0, row["top1_accuracy"])
        self.assertEqual(2, row["top3_hits"])
        self.assertEqual(1.0, row["top3_accuracy"])

    def test_progress_uses_symmetry_aware_counts(self) -> None:
        event = make_worker_progress_event(
            worker_id=0,
            attempted_positions=3,
            metrics={
                0.5: {
                    "positions": 3,
                    "exact_hits": {1: 0},
                    "symmetric_hits": {1: 2},
                }
            },
            blend_params=[0.5],
            n_values=[1],
            start_time=0.0,
        )

        self.assertEqual(2, event["results"][0]["top1_hits"])


if __name__ == "__main__":
    unittest.main()
