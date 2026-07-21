#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate_wthor_blend_human_match import (
    ALL_LEGAL_HINT_COUNT,
    BLACK,
    BOARD_DTYPE,
    BlendedPolicy,
    BoardState,
    POLICY_SIZE,
    equivalent_targets,
    hint_cache_key,
    hint_scores_with_cache,
    make_metrics,
    make_worker_progress_event,
    rank_for_distribution,
    require_complete_egaroucid_scores,
    update_metrics_for_position_sample,
)
from merge_wthor_blend_results import merge_results
from run_random_wthor_blend_experiment import (
    BLEND_PARAMS,
    make_blend_summary_rows,
    make_console_level_summary_row,
)
from run_wthor_blend_shards import shard_done
from wthor_human_match_evaluation import wilson_interval
from blend_policy_with_egaroucid import (
    hint_cache_key as blended_policy_hint_cache_key,
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
    hits: int,
    positions: int,
) -> dict:
    return {
        "blend_param": alpha,
        "top_n": top_n,
        "hits": hits,
        "positions": positions,
        "accuracy": hits / positions,
    }


class SymmetryAgreementTest(unittest.TestCase):
    def test_hint_request_covers_every_possible_legal_move(self) -> None:
        state = type("State", (), {"black": 1, "white": 2})()

        self.assertEqual(POLICY_SIZE, ALL_LEGAL_HINT_COUNT)
        self.assertNotEqual(
            hint_cache_key(21, state, 0, 3),
            hint_cache_key(21, state, 0, ALL_LEGAL_HINT_COUNT),
        )
        self.assertNotEqual(
            blended_policy_hint_cache_key(21, 1, 2, 0, 3),
            blended_policy_hint_cache_key(
                21,
                1,
                2,
                0,
                ALL_LEGAL_HINT_COUNT,
            ),
        )

    def test_evaluator_requests_all_legal_moves_independent_of_top_n(self) -> None:
        state = BoardState.initial()
        player, opponent = state.player_opponent_bits(BLACK)
        policy = state.legal_policies(BLACK)[0]
        sample = np.zeros((), dtype=BOARD_DTYPE)
        sample["player"] = player
        sample["opponent"] = opponent
        sample["color"] = BLACK
        sample["policy"] = policy

        class FakeBlender:
            def policy_distribution(self, board, side, legal):
                result = np.zeros(POLICY_SIZE, dtype=np.float32)
                result[legal] = 1.0 / len(legal)
                return result

            def egaroucid_distribution(self, scores, legal):
                result = np.zeros(POLICY_SIZE, dtype=np.float32)
                result[legal] = 1.0 / len(legal)
                return result

        metrics = make_metrics([0.5], [1, 3], ["01_10"])
        with patch(
            "evaluate_wthor_blend_human_match.hint_scores_with_cache",
            return_value=({}, ""),
        ) as hint_scores:
            update_metrics_for_position_sample(
                sample,
                FakeBlender(),
                [0.5],
                [1, 3],
                metrics,
                [],
                0,
                0,
                None,
            )

        self.assertEqual(
            ALL_LEGAL_HINT_COUNT,
            hint_scores.call_args.args[3],
        )

    def test_partial_egaroucid_policy_is_rejected(self) -> None:
        legal = BoardState.initial().legal_policies(BLACK)
        complete = {policy: float(policy) for policy in legal}

        require_complete_egaroucid_scores(complete, legal)
        with self.assertRaisesRegex(ValueError, "missing="):
            require_complete_egaroucid_scores(
                {policy: complete[policy] for policy in legal[:-1]},
                legal,
            )
        with self.assertRaisesRegex(ValueError, "unexpected="):
            require_complete_egaroucid_scores(
                {**complete, 0: 0.0},
                legal,
            )

    def test_evaluator_rejects_partial_cached_hint_scores(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)

        class FakeHintRunner:
            level = 21

        class FakeBlender:
            hint_runner = FakeHintRunner()

        class FakeHintCache:
            def get(self, key):
                return {legal[0]: 0.0}, "partial"

        with self.assertRaisesRegex(ValueError, "does not cover exactly"):
            hint_scores_with_cache(
                FakeBlender(),
                state,
                BLACK,
                ALL_LEGAL_HINT_COUNT,
                FakeHintCache(),
            )

    def test_evaluator_does_not_cache_fresh_partial_hint_scores(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)

        class FakeHintRunner:
            level = 21

            def hint_scores(self, board, side, hint_count):
                return {legal[0]: 0.0}, "partial"

        class FakeBlender:
            hint_runner = FakeHintRunner()

        class FakeHintCache:
            put_called = False

            def get(self, key):
                return None

            def put(self, key, scores, raw_hint):
                self.put_called = True

        hint_cache = FakeHintCache()
        with self.assertRaisesRegex(ValueError, "does not cover exactly"):
            hint_scores_with_cache(
                FakeBlender(),
                state,
                BLACK,
                ALL_LEGAL_HINT_COUNT,
                hint_cache,
            )
        self.assertFalse(hint_cache.put_called)

    def test_evaluator_rejects_partial_hint_request(self) -> None:
        state = BoardState.initial()

        class FakeHintRunner:
            level = 21

            def hint_scores(self, board, side, hint_count):
                raise AssertionError("partial hint request must fail first")

        class FakeBlender:
            hint_runner = FakeHintRunner()

        with self.assertRaisesRegex(ValueError, "expected hint 64"):
            hint_scores_with_cache(
                FakeBlender(),
                state,
                BLACK,
                3,
                None,
            )

    def test_blended_policy_requests_all_legal_hint_scores(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)

        class FakeHintRunner:
            level = 21
            timeout_sec = 1.0
            requested_hint_count = None

            def hint_scores(self, board, side, hint_count):
                self.requested_hint_count = hint_count
                return {policy: float(policy) for policy in legal}, "complete"

        blender = object.__new__(BlendedPolicy)
        blender.cache_egaroucid = False
        blender.egaroucid_cache = {}
        blender.shared_hint_cache = None
        blender.hint_runner = FakeHintRunner()

        scores, _ = blender.cached_hint_scores(state, BLACK)

        self.assertEqual(set(legal), set(scores))
        self.assertEqual(
            ALL_LEGAL_HINT_COUNT,
            blender.hint_runner.requested_hint_count,
        )

    def test_blended_policy_rejects_partial_memory_cache(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)
        memory_key = (state.black, state.white, BLACK)

        blender = object.__new__(BlendedPolicy)
        blender.cache_egaroucid = True
        blender.egaroucid_cache = {
            memory_key: ({legal[0]: 0.0}, "partial")
        }
        blender.shared_hint_cache = None

        with self.assertRaisesRegex(ValueError, "does not cover exactly"):
            blender.cached_hint_scores(state, BLACK)

    def test_merge_rejects_legacy_partial_hint_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "all-legal-move"):
            merge_results([{"console_hint_count": 3}])

    def test_merge_requires_explicit_all_legal_policy_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicitly declare"):
            merge_results(
                [
                    {
                        "console_hint_count": ALL_LEGAL_HINT_COUNT,
                    }
                ]
            )

    def test_shard_done_rejects_legacy_partial_hint_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result_path = output_dir / "wthor_blend_human_match.json"
            result_path.write_text(
                json.dumps(
                    {
                        "topn": [{"top_n": 1}],
                        "console_hint_count": 3,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(shard_done(output_dir))

            result_path.write_text(
                json.dumps(
                    {
                        "topn": [{"top_n": 1}],
                        "console_hint_count": ALL_LEGAL_HINT_COUNT,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(shard_done(output_dir))

            result_path.write_text(
                json.dumps(
                    {
                        "topn": [{"top_n": 1}],
                        "console_hint_count": ALL_LEGAL_HINT_COUNT,
                        "egaroucid_policy_support": "all_legal_moves",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(shard_done(output_dir))

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
                    hits=2,
                    positions=2,
                )
            )
            topn.append(
                result_row(
                    alpha,
                    3,
                    hits=2,
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
                    hits=2,
                    positions=2,
                ),
                result_row(
                    0.0,
                    3,
                    hits=2,
                    positions=2,
                ),
            ]
        }

        row = make_console_level_summary_row(21, result, 1.0)

        self.assertEqual(2, row["top1_hits"])
        self.assertEqual(1.0, row["top1_accuracy"])
        self.assertEqual(2, row["top3_hits"])
        self.assertEqual(1.0, row["top3_accuracy"])
        expected_lower, expected_upper = wilson_interval(2, 2)
        self.assertEqual(expected_lower, row["top1_ci95_lower"])
        self.assertEqual(expected_upper, row["top1_ci95_upper"])
        self.assertEqual(expected_lower, row["top3_ci95_lower"])
        self.assertEqual(expected_upper, row["top3_ci95_upper"])

    def test_progress_uses_symmetry_aware_counts(self) -> None:
        event = make_worker_progress_event(
            worker_id=0,
            attempted_positions=3,
            metrics={
                0.5: {
                    "positions": 3,
                    "hits": {1: 2},
                }
            },
            blend_params=[0.5],
            n_values=[1],
            start_time=0.0,
        )

        self.assertEqual(2, event["results"][0]["top1_hits"])


if __name__ == "__main__":
    unittest.main()
