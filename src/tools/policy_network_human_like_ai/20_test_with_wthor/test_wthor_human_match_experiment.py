#!/usr/bin/env python3

import argparse
import io
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

import wthor_hint_pipeline as hint_pipeline
from evaluate_wthor_blend_human_match import (
    BLACK,
    BOARD_DTYPE,
    BoardState,
)
from blend_policy_with_egaroucid import policy_to_coord
from wthor_hint_pipeline import (
    HintTask,
    collect_hints,
    make_hint_tasks,
    run_hint_task,
    validate_hint_data,
)
from wthor_human_match_evaluation import (
    ALL_LEGAL_HINT_COUNT,
    CONSOLE_ONLY_HINT_COUNT,
    HintData,
    PositionGroup,
    egaroucid_log_scores,
    load_position_groups,
    make_level21_reuse_validation,
    validate_aggregate_counts,
)
from wthor_human_match_experiment import (
    SUMMARY_VERSION,
    default_output_dir,
    ensure_experiment_identity,
    make_experiment_identity,
    require_all_levels_to_start,
)


class HumanMatchExperimentTest(unittest.TestCase):
    def test_at_least_one_worker_per_level_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "11以上"):
            require_all_levels_to_start(10)
        self.assertEqual(11, require_all_levels_to_start(11))

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

    def test_lower_levels_keep_one_full_stream_and_level21_uses_shards(
        self,
    ) -> None:
        states = [(index, index + 1, BLACK) for index in range(8)]
        tasks = make_hint_tasks(
            levels=[1, 3, 21],
            states_by_level={level: states for level in (1, 3, 21)},
            workers=4,
            egaroucid_exe=Path("egaroucid.exe"),
            egaroucid_threads=1,
            timeout_sec=10.0,
            max_retries=2,
        )

        self.assertEqual([21, 3, 1, 21, 21, 21], [t.level for t in tasks])
        for level in (1, 3):
            level_tasks = [task for task in tasks if task.level == level]
            self.assertEqual(1, len(level_tasks))
            self.assertEqual(tuple(states), level_tasks[0].states)
        level21_tasks = [task for task in tasks if task.level == 21]
        self.assertEqual(4, len(level21_tasks))
        for shard, task in enumerate(level21_tasks):
            self.assertEqual(tuple(states[shard::4]), task.states)
        self.assertEqual(
            {1, 3, 21},
            {task.level for task in tasks[:4]},
        )

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

    def test_partial_lower_level_cache_replays_the_full_stream(self) -> None:
        states = [(index, index + 1, BLACK) for index in range(8)]
        cached = {
            state: HintData({}, ())
            for state in states[:4]
        }
        tasks = make_hint_tasks(
            levels=[1],
            states_by_level={1: states},
            workers=16,
            egaroucid_exe=Path("egaroucid.exe"),
            egaroucid_threads=2,
            timeout_sec=10.0,
            max_retries=2,
            cached_by_level={1: cached},
        )

        self.assertEqual(1, len(tasks))
        self.assertEqual(tuple(states), tasks[0].states)

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

    def test_v4_identity_records_level_specific_console_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            executable = temp_path / "egaroucid.exe"
            weights = temp_path / "weights.bin"
            executable.write_bytes(b"console")
            weights.write_bytes(b"weights")
            args = argparse.Namespace(
                positions=10000,
                data_split="test",
                split_seed=613,
                sample_seed=613,
                egaroucid_exe=executable,
                egaroucid_threads=2,
                egaroucid_retries=2,
                weights=weights,
                score_temperature=1.0,
            )
            identity = make_experiment_identity(
                args,
                workers=16,
                sample_hash="sample",
                sample_records_hash="records",
            )

        self.assertEqual(4, SUMMARY_VERSION)
        self.assertEqual(4, identity["identity_version"])
        self.assertEqual(
            "one_persistent_stream_per_level_1_to_19_"
            "plus_level21_fixed_strided_shards_v2",
            identity["partition"],
        )
        self.assertTrue(default_output_dir(args, 16).name.endswith("_v4"))

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
        events = []

        class EventQueue:
            def put(self, event):
                events.append(event)

        with patch(
            "wthor_hint_pipeline.EgaroucidHintRunner",
            FakeRunner,
        ), patch.object(
            hint_pipeline,
            "_HINT_PROGRESS_QUEUE",
            EventQueue(),
        ):
            result = run_hint_task(task)

        self.assertEqual(2, FakeRunner.attempts)
        self.assertEqual(1, result["states"])
        self.assertEqual(
            ["task_started", "retry", "rows", "task_finished"],
            [event["kind"] for event in events],
        )
        self.assertEqual(1, len(events[2]["rows"]))

    def test_different_levels_create_and_close_different_consoles(self) -> None:
        state = BoardState.initial()

        class FakeRunner:
            instances = []

            def __init__(self, *args, level, **kwargs):
                self.level = level
                self.close_count = 0
                FakeRunner.instances.append(self)

            def hint_scores(self, current_state, side, hint_count):
                moves = current_state.legal_policies(side)[:hint_count]
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
                self.close_count += 1

        def task(level, repetitions=1):
            return HintTask(
                level=level,
                hint_count=CONSOLE_ONLY_HINT_COUNT,
                states=(
                    (state.black, state.white, BLACK),
                ) * repetitions,
                egaroucid_exe="unused.exe",
                egaroucid_threads=1,
                timeout_sec=1.0,
                max_retries=0,
            )

        with patch(
            "wthor_hint_pipeline.EgaroucidHintRunner",
            FakeRunner,
        ), patch.object(
            hint_pipeline,
            "_HINT_PROGRESS_QUEUE",
            None,
        ):
            level1_result = run_hint_task(task(1, repetitions=2))
            run_hint_task(task(3))

        self.assertEqual(2, level1_result["states"])
        self.assertEqual([1, 3], [runner.level for runner in FakeRunner.instances])
        self.assertEqual(
            [1, 1],
            [runner.close_count for runner in FakeRunner.instances],
        )

    def test_hint_progress_uses_a_timer_before_a_future_finishes(self) -> None:
        state = BoardState.initial()
        legal = state.legal_policies(BLACK)
        top3 = legal[:CONSOLE_ONLY_HINT_COUNT]
        scores = {
            policy: float(100 - rank)
            for rank, policy in enumerate(top3)
        }
        raw_hint = "\n".join(
            "0 | 0 | 0 | "
            f"{policy_to_coord(policy)} | {scores[policy]}"
            for policy in top3
        )
        state_key = (state.black, state.white, BLACK)
        task_result = {
            "level": 1,
            "rows": [(state_key, scores, raw_hint)],
            "states": 1,
            "started_at_unix": 1.0,
            "finished_at_unix": 2.0,
            "elapsed_sec": 1.0,
        }

        class FakeFuture:
            def result(self):
                return task_result

            def cancel(self):
                return False

            def cancelled(self):
                return False

            def done(self):
                return True

        future = FakeFuture()

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def submit(self, function, task):
                return future

            def shutdown(self, *args, **kwargs):
                pass

        class FakeQueue:
            def get_nowait(self):
                raise hint_pipeline.queue.Empty

            def close(self):
                pass

            def join_thread(self):
                pass

        class FakeContext:
            def Event(self):
                return hint_pipeline.threading.Event()

            def Queue(self):
                return FakeQueue()

        wait_calls = 0

        def fake_wait(pending, timeout, return_when):
            nonlocal wait_calls
            wait_calls += 1
            time.sleep(0.004)
            if wait_calls == 1:
                return set(), set(pending)
            return set(pending), set()

        progress = []
        published_rows = []
        group = PositionGroup(
            state.black,
            state.white,
            BLACK,
            ((top3[0], 1),),
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            hint_pipeline,
            "ProcessPoolExecutor",
            FakeExecutor,
        ), patch.object(
            hint_pipeline.multiprocessing,
            "get_context",
            return_value=FakeContext(),
        ), patch.object(
            hint_pipeline,
            "wait",
            side_effect=fake_wait,
        ), patch.object(
            hint_pipeline.sys,
            "stderr",
            io.StringIO(),
        ):
            collect_hints(
                [group],
                [1],
                Path(temp_dir),
                workers=1,
                egaroucid_exe=Path("unused.exe"),
                egaroucid_threads=1,
                timeout_sec=1.0,
                max_retries=0,
                progress_interval_sec=0.001,
                on_rows=lambda level, rows: published_rows.extend(rows),
                progress_callback=progress.append,
            )

        self.assertGreaterEqual(wait_calls, 2)
        self.assertGreaterEqual(len(progress), 3)
        self.assertEqual(0, progress[0]["completed_states"])
        self.assertFalse(progress[1]["final"])
        self.assertTrue(progress[-1]["final"])
        self.assertEqual(1, progress[-1]["completed_states"])
        self.assertEqual(1, len(published_rows))

    def test_finished_lower_level_slot_starts_waiting_level21(self) -> None:
        groups = []
        state = BoardState.initial()
        for _ in range(3):
            side = state.side
            legal = state.legal_policies(side)
            groups.append(
                PositionGroup(
                    state.black,
                    state.white,
                    side,
                    ((legal[0], 1),),
                )
            )
            state.apply_move(side, legal[0])

        def task_result(task):
            rows = []
            for state_key in task.states:
                current_state = BoardState(*state_key)
                legal = current_state.legal_policies(state_key[2])
                moves = (
                    legal
                    if task.hint_count == ALL_LEGAL_HINT_COUNT
                    else legal[:task.hint_count]
                )
                scores = {
                    policy: float(100 - rank)
                    for rank, policy in enumerate(moves)
                }
                raw_hint = "\n".join(
                    "0 | 0 | 0 | "
                    f"{policy_to_coord(policy)} | {scores[policy]}"
                    for policy in moves
                )
                rows.append((state_key, scores, raw_hint))
            return {
                "level": task.level,
                "rows": rows,
                "states": len(rows),
                "started_at_unix": 1.0,
                "finished_at_unix": 2.0,
                "elapsed_sec": 1.0,
            }

        class FakeFuture:
            def __init__(self, task, index):
                self.task = task
                self.index = index

            def result(self):
                return task_result(self.task)

            def cancel(self):
                return False

            def cancelled(self):
                return False

            def done(self):
                return True

        class FakeExecutor:
            instance = None

            def __init__(self, *args, **kwargs):
                self.submitted = []
                FakeExecutor.instance = self

            def submit(self, function, task):
                future = FakeFuture(task, len(self.submitted))
                self.submitted.append(future)
                return future

            def shutdown(self, *args, **kwargs):
                pass

        class FakeQueue:
            def get_nowait(self):
                raise hint_pipeline.queue.Empty

            def close(self):
                pass

            def join_thread(self):
                pass

        class FakeContext:
            def Event(self):
                return hint_pipeline.threading.Event()

            def Queue(self):
                return FakeQueue()

        submitted_counts_at_wait = []
        first_wait = True

        def fake_wait(pending, timeout, return_when):
            nonlocal first_wait
            submitted_counts_at_wait.append(
                len(FakeExecutor.instance.submitted)
            )
            if first_wait:
                first_wait = False
                finished = next(
                    future
                    for future in pending
                    if future.task.level == 1
                )
            else:
                finished = min(pending, key=lambda future: future.index)
            return {finished}, set(pending) - {finished}

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            hint_pipeline,
            "ProcessPoolExecutor",
            FakeExecutor,
        ), patch.object(
            hint_pipeline.multiprocessing,
            "get_context",
            return_value=FakeContext(),
        ), patch.object(
            hint_pipeline,
            "wait",
            side_effect=fake_wait,
        ), patch.object(
            hint_pipeline.sys,
            "stderr",
            io.StringIO(),
        ):
            collect_hints(
                groups,
                [1, 21],
                Path(temp_dir),
                workers=2,
                egaroucid_exe=Path("unused.exe"),
                egaroucid_threads=1,
                timeout_sec=1.0,
                max_retries=0,
                progress_interval_sec=60.0,
            )

        submitted_levels = [
            future.task.level
            for future in FakeExecutor.instance.submitted
        ]
        self.assertEqual([21, 1, 21], submitted_levels)
        self.assertEqual(2, submitted_counts_at_wait[0])
        self.assertEqual(3, submitted_counts_at_wait[1])


if __name__ == "__main__":
    unittest.main()
