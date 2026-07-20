#!/usr/bin/env python3

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from evaluate_wthor_blend_human_match import (
    BLACK,
    BOARD_DTYPE,
    BoardState,
)
from blend_policy_with_egaroucid import policy_to_coord
from wthor_hint_pipeline import (
    HintTask,
    make_hint_tasks,
    run_hint_task,
    validate_hint_data,
)
from wthor_human_match_evaluation import (
    ALL_LEGAL_HINT_COUNT,
    CONSOLE_ONLY_HINT_COUNT,
    HintData,
    egaroucid_log_scores,
    load_position_groups,
    make_level21_reuse_validation,
    validate_aggregate_counts,
)
from wthor_human_match_experiment import ensure_experiment_identity


class HumanMatchExperimentTest(unittest.TestCase):
    def test_duplicate_states_are_grouped_without_losing_labels(self) -> None:
        state = BoardState.initial()
        player, opponent = state.player_opponent_bits(BLACK)
        legal = state.legal_policies(BLACK)
        records = np.zeros(2, dtype=BOARD_DTYPE)
        records["player"] = player
        records["opponent"] = opponent
        records["color"] = BLACK
        records["policy"] = legal[:2]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "0.dat"
            records.tofile(path)
            groups, split_positions, _, sample_records_hash = (
                load_position_groups(
                    [path],
                    available_positions=2,
                    positions=2,
                    data_split="all",
                    split_seed=613,
                    sample_seed=613,
                )
            )

        self.assertEqual(2, split_positions)
        self.assertEqual(1, len(groups))
        self.assertEqual(2, groups[0].sample_count)
        self.assertEqual(
            {(legal[0], 1), (legal[1], 1)},
            set(groups[0].policy_counts),
        )
        self.assertEqual(64, len(sample_records_hash))

    def test_hint_tasks_cover_each_state_exactly_once(self) -> None:
        states = [(index, index + 1, BLACK) for index in range(10)]
        tasks = make_hint_tasks(
            levels=[21],
            states_by_level={21: states},
            workers=4,
            egaroucid_exe=Path("egaroucid.exe"),
            egaroucid_threads=1,
            timeout_sec=10.0,
            max_retries=2,
        )

        flattened = [
            state
            for task in tasks
            for state in task.states
        ]
        self.assertEqual(4, len(tasks))
        self.assertEqual(states, sorted(flattened))
        self.assertEqual(len(states), len(set(flattened)))

    def test_resume_skips_only_complete_fixed_shards(self) -> None:
        states = [(index, index + 1, BLACK) for index in range(8)]
        cached = {
            state: HintData({}, ())
            for state in states[0::2]
        }
        tasks = make_hint_tasks(
            levels=[21],
            states_by_level={21: states},
            workers=2,
            egaroucid_exe=Path("egaroucid.exe"),
            egaroucid_threads=1,
            timeout_sec=10.0,
            max_retries=2,
            cached_by_level={21: cached},
        )

        self.assertEqual(1, len(tasks))
        self.assertEqual(tuple(states[1::2]), tasks[0].states)

    def test_level21_console_counts_reuse_alpha_zero(self) -> None:
        validation = make_level21_reuse_validation(
            [
                {
                    "alpha": 0.0,
                    "positions": 10,
                    "top1_hits": 6,
                    "top3_hits": 9,
                }
            ],
            [
                {
                    "level": 21,
                    "positions": 10,
                    "top1_hits": 6,
                    "top3_hits": 9,
                }
            ],
        )

        self.assertTrue(validation["duplicate_hint_evaluation_avoided"])
        self.assertTrue(validation["aggregate_counts_equal"])

    def test_aggregate_count_validation_rejects_top3_below_top1(self) -> None:
        bad_row = {
            "positions": 10,
            "top1_hits": 8,
            "top3_hits": 7,
        }
        with self.assertRaises(RuntimeError):
            validate_aggregate_counts(
                10,
                [bad_row],
                [bad_row],
                0,
                0,
            )

    def test_experiment_identity_rejects_condition_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            ensure_experiment_identity(output_dir, {"workers": 16})
            ensure_experiment_identity(output_dir, {"workers": 16})
            with self.assertRaises(ValueError):
                ensure_experiment_identity(
                    output_dir,
                    {"workers": 32},
                )

    def test_top3_hint_is_valid_for_standalone_console_metrics(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)
        top3 = legal[:CONSOLE_ONLY_HINT_COUNT]
        scores = {
            policy: float(100 - rank)
            for rank, policy in enumerate(top3)
        }
        raw_hint = "\n".join(
            f"0 | 0 | 0 | {policy_to_coord(policy)} | {scores[policy]}"
            for policy in top3
        )

        move_order = validate_hint_data(
            state,
            BLACK,
            scores,
            raw_hint,
            CONSOLE_ONLY_HINT_COUNT,
        )
        self.assertEqual(tuple(top3), move_order)

        log_scores = egaroucid_log_scores(
            HintData(scores, move_order),
            legal,
            1.0,
            require_all_legal=False,
        )
        self.assertTrue(np.isfinite(log_scores[top3]).all())
        self.assertTrue(
            np.isneginf(
                log_scores[
                    [policy for policy in legal if policy not in top3]
                ]
            ).all()
        )

    def test_level21_rejects_partial_hint_scores(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)
        top3 = legal[:CONSOLE_ONLY_HINT_COUNT]
        scores = {
            policy: float(100 - rank)
            for rank, policy in enumerate(top3)
        }
        raw_hint = "\n".join(
            f"0 | 0 | 0 | {policy_to_coord(policy)} | {scores[policy]}"
            for policy in top3
        )

        with self.assertRaises(ValueError):
            validate_hint_data(
                state,
                BLACK,
                scores,
                raw_hint,
                ALL_LEGAL_HINT_COUNT,
            )

    def test_level21_rejects_hint_order_inconsistent_with_scores(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)
        scores = {
            policy: float(100 - rank)
            for rank, policy in enumerate(legal)
        }
        raw_hint = "\n".join(
            f"0 | 0 | 0 | {policy_to_coord(policy)} | {scores[policy]}"
            for policy in legal[:-1]
        )
        raw_hint += "\n0 | 0 | 0 | a1 | 0"

        with self.assertRaises(ValueError):
            validate_hint_data(
                state,
                BLACK,
                scores,
                raw_hint,
                ALL_LEGAL_HINT_COUNT,
            )

    def test_worker_restarts_after_incomplete_hint_output(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)
        top3 = legal[:CONSOLE_ONLY_HINT_COUNT]

        class FakeRunner:
            attempts = 0

            def __init__(self, *args, **kwargs):
                pass

            def hint_scores(self, *args, **kwargs):
                FakeRunner.attempts += 1
                moves = top3[:2] if FakeRunner.attempts == 1 else top3
                scores = {
                    policy: float(100 - rank)
                    for rank, policy in enumerate(moves)
                }
                raw_hint = "\n".join(
                    "0 | 0 | 0 | "
                    f"{policy_to_coord(policy)} | {scores[policy]}"
                    for policy in moves
                )
                return scores, raw_hint

            def close(self):
                pass

        task = HintTask(
            level=1,
            hint_count=CONSOLE_ONLY_HINT_COUNT,
            states=((state.black, state.white, BLACK),),
            egaroucid_exe="unused.exe",
            egaroucid_threads=1,
            timeout_sec=1.0,
            max_retries=1,
        )
        with patch(
            "wthor_hint_pipeline.EgaroucidHintRunner",
            FakeRunner,
        ):
            result = run_hint_task(task)

        self.assertEqual(2, FakeRunner.attempts)
        self.assertEqual(1, result["states"])


if __name__ == "__main__":
    unittest.main()
