#!/usr/bin/env python3

import argparse
from collections import Counter
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
    actor_claim_capacity,
    claim_next_hint_state,
    collect_hints,
    make_initial_actor_levels,
    run_hint_task,
    select_lower_level_for_actor,
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


def make_actor_test_row(state_key, hint_count):
    state = BoardState(*state_key)
    legal = state.legal_policies(state_key[2])
    moves = legal[:hint_count]
    scores = {
        policy: float(100 - rank)
        for rank, policy in enumerate(moves)
    }
    raw_hint = "\n".join(
        "0 | 0 | 0 | "
        f"{policy_to_coord(policy)} | {scores[policy]}"
        for policy in moves
    )
    return state_key, scores, raw_hint


class FakeActorValue:
    def __init__(self, value):
        self.value = value


class FakeActorQueue:
    def get_nowait(self):
        raise hint_pipeline.queue.Empty

    def close(self):
        pass

    def join_thread(self):
        pass


class FakeActorContext:
    def Event(self):
        return hint_pipeline.threading.Event()

    def Queue(self):
        return FakeActorQueue()

    def Value(self, _typecode, value):
        return FakeActorValue(value)

    def Lock(self):
        return hint_pipeline.threading.Lock()


class FakeActorFuture:
    def __init__(self, executor, task, index):
        self.executor = executor
        self.task = task
        self.index = index
        self.cached_result = None

    def result(self):
        if self.cached_result is None:
            self.cached_result = self.executor.run_actor(self.task)
        return self.cached_result

    def cancel(self):
        return False

    def cancelled(self):
        return False

    def done(self):
        return True


class FakeActorExecutor:
    instance = None

    def __init__(self, *args, initargs=(), **kwargs):
        self.work_states_by_level = initargs[2]
        self.work_cursors_by_level = initargs[3]
        self.work_locks_by_level = initargs[4]
        self.submitted = []
        self.claimed = []
        FakeActorExecutor.instance = self

    def submit(self, _function, task):
        future = FakeActorFuture(self, task, len(self.submitted))
        self.submitted.append(future)
        return future

    def run_actor(self, task):
        states = self.work_states_by_level[task.level]
        cursor = self.work_cursors_by_level[task.level]
        lock = self.work_locks_by_level[task.level]
        with lock:
            start = int(cursor.value)
            cursor.value = len(states)
        claimed = (task.initial_state, *states[start:])
        self.claimed.extend((task.level, state) for state in claimed)
        rows = [
            make_actor_test_row(state, task.hint_count)
            for state in claimed
        ]
        return {
            "level": task.level,
            "rows": rows,
            "states": len(rows),
            "started_at_unix": 1.0,
            "finished_at_unix": 2.0,
            "elapsed_sec": 1.0,
        }

    def shutdown(self, *args, **kwargs):
        pass


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

    def test_actor_claim_capacity_counts_unclaimed_states(self) -> None:
        self.assertEqual(7, actor_claim_capacity(10, 3))
        self.assertEqual(0, actor_claim_capacity(3, 3))
        self.assertEqual(2, actor_claim_capacity(2, 0))

    def test_initial_actor_levels_prioritize_level21_until_it_is_cached(
        self,
    ) -> None:
        all_levels = list(range(1, 22, 2))
        cold_start = make_initial_actor_levels(
            all_levels,
            {level: 100 for level in all_levels},
            workers=16,
        )
        self.assertEqual(6, cold_start.count(21))
        self.assertEqual(
            {level: 1 for level in range(1, 20, 2)},
            {
                level: cold_start.count(level)
                for level in range(1, 20, 2)
            },
        )
        self.assertEqual(
            [21, 19, 17, 21, 21],
            make_initial_actor_levels(
                [17, 19, 21],
                {17: 10, 19: 10, 21: 10},
                workers=5,
            ),
        )
        self.assertEqual(
            [19, 17, 19, 17],
            make_initial_actor_levels(
                [17, 19, 21],
                {17: 10, 19: 10, 21: 0},
                workers=4,
            ),
        )

    def test_lower_actor_selection_uses_remaining_work_per_actor(self) -> None:
        self.assertEqual(
            1,
            select_lower_level_for_actor(
                [1, 3, 21],
                {1: 100, 3: 20, 21: 1000},
                {1: 20, 3: 0, 21: 0},
                {1: 2, 3: 1, 21: 0},
            ),
        )
        self.assertEqual(
            1,
            select_lower_level_for_actor(
                [1, 3, 21],
                {1: 2, 3: 1, 21: 1000},
                {1: 1, 3: 1, 21: 0},
                {1: 1, 3: 0, 21: 0},
            )
        )
        self.assertIsNone(
            select_lower_level_for_actor(
                [1, 3, 21],
                {1: 2, 3: 1, 21: 1000},
                {1: 2, 3: 1, 21: 0},
                {1: 1, 3: 0, 21: 0},
            )
        )

    def test_atomic_cursor_claims_each_state_once_across_actors(self) -> None:
        states = tuple(
            (index, index + 1, BLACK)
            for index in range(100)
        )
        cursor = FakeActorValue(0)
        lock = hint_pipeline.threading.Lock()
        claimed = []
        claimed_lock = hint_pipeline.threading.Lock()

        def claim_until_empty():
            while True:
                state = claim_next_hint_state(1)
                if state is None:
                    return
                with claimed_lock:
                    claimed.append(state)

        with patch.object(
            hint_pipeline,
            "_HINT_WORK_STATES_BY_LEVEL",
            {1: states},
        ), patch.object(
            hint_pipeline,
            "_HINT_WORK_CURSORS_BY_LEVEL",
            {1: cursor},
        ), patch.object(
            hint_pipeline,
            "_HINT_WORK_LOCKS_BY_LEVEL",
            {1: lock},
        ):
            actors = [
                hint_pipeline.threading.Thread(target=claim_until_empty)
                for _ in range(8)
            ]
            for actor in actors:
                actor.start()
            for actor in actors:
                actor.join()
            self.assertIsNone(claim_next_hint_state(1))

        self.assertEqual(Counter(states), Counter(claimed))

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

    def test_v5_identity_records_elastic_level_actor_partition(self) -> None:
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

        self.assertEqual(5, SUMMARY_VERSION)
        self.assertEqual(5, identity["identity_version"])
        self.assertEqual(
            "level_fixed_persistent_actors_with_atomic_per_level_"
            "cursor_elastic_after_level21_v1",
            identity["partition"],
        )
        self.assertTrue(default_output_dir(args, 16).name.endswith("_v5"))

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
            FakeActorExecutor,
        ), patch.object(
            hint_pipeline.multiprocessing,
            "get_context",
            return_value=FakeActorContext(),
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

    def test_level21_completion_adds_lower_actor_without_duplicate_claims(
        self,
    ) -> None:
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

        submitted_counts_at_wait = []
        first_wait = True

        def fake_wait(pending, timeout, return_when):
            nonlocal first_wait
            submitted_counts_at_wait.append(
                len(FakeActorExecutor.instance.submitted)
            )
            if first_wait:
                first_wait = False
                finished = next(
                    future
                    for future in pending
                    if future.task.level == 21
                )
            else:
                finished = min(pending, key=lambda future: future.index)
            return {finished}, set(pending) - {finished}

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            hint_pipeline,
            "ProcessPoolExecutor",
            FakeActorExecutor,
        ), patch.object(
            hint_pipeline.multiprocessing,
            "get_context",
            return_value=FakeActorContext(),
        ), patch.object(
            hint_pipeline,
            "wait",
            side_effect=fake_wait,
        ), patch.object(
            hint_pipeline.sys,
            "stderr",
            io.StringIO(),
        ):
            _, _, level_timing, _ = collect_hints(
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
            for future in FakeActorExecutor.instance.submitted
        ]
        self.assertEqual([21, 1, 1], submitted_levels)
        self.assertTrue(
            all(
                future.task.initial_state is not None
                for future in FakeActorExecutor.instance.submitted
            )
        )
        self.assertEqual(2, submitted_counts_at_wait[0])
        self.assertEqual(3, submitted_counts_at_wait[1])
        claims = Counter(FakeActorExecutor.instance.claimed)
        expected = {
            (level, group.key)
            for level in (1, 21)
            for group in groups
        }
        self.assertEqual(expected, set(claims))
        self.assertTrue(all(count == 1 for count in claims.values()))
        for level in (1, 21):
            recorded_indices = [
                index
                for actor in level_timing[level]["actors"]
                for index in actor["state_indices"]
            ]
            self.assertEqual(Counter(range(3)), Counter(recorded_indices))

    def test_cached_level21_scales_lower_level_from_the_start(self) -> None:
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

        def fake_cache_load(_cache, level, _hint_count, state_keys):
            if level != 21:
                return {}
            result = {}
            for state_key in state_keys:
                _, scores, _ = make_actor_test_row(
                    state_key,
                    ALL_LEGAL_HINT_COUNT,
                )
                move_order = tuple(
                    BoardState(*state_key).legal_policies(state_key[2])
                )
                result[state_key] = HintData(scores, move_order)
            return result

        def fake_wait(pending, timeout, return_when):
            finished = min(pending, key=lambda future: future.index)
            return {finished}, set(pending) - {finished}

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            hint_pipeline.HintCache,
            "load",
            new=fake_cache_load,
        ), patch.object(
            hint_pipeline,
            "ProcessPoolExecutor",
            FakeActorExecutor,
        ), patch.object(
            hint_pipeline.multiprocessing,
            "get_context",
            return_value=FakeActorContext(),
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
                workers=3,
                egaroucid_exe=Path("unused.exe"),
                egaroucid_threads=1,
                timeout_sec=1.0,
                max_retries=0,
                progress_interval_sec=60.0,
            )

        submitted_levels = [
            future.task.level
            for future in FakeActorExecutor.instance.submitted
        ]
        self.assertEqual([1, 1, 1], submitted_levels)
        claims = Counter(FakeActorExecutor.instance.claimed)
        self.assertEqual(
            {(1, group.key) for group in groups},
            set(claims),
        )
        self.assertTrue(all(count == 1 for count in claims.values()))

    def test_partial_level_cache_replays_every_state_once(self) -> None:
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

        def fake_cache_load(_cache, level, hint_count, state_keys):
            self.assertEqual(1, level)
            state_key = state_keys[0]
            _, scores, _ = make_actor_test_row(state_key, hint_count)
            move_order = tuple(
                BoardState(*state_key).legal_policies(state_key[2])[
                    :hint_count
                ]
            )
            return {state_key: HintData(scores, move_order)}

        def fake_wait(pending, timeout, return_when):
            finished = min(pending, key=lambda future: future.index)
            return {finished}, set(pending) - {finished}

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            hint_pipeline.HintCache,
            "load",
            new=fake_cache_load,
        ), patch.object(
            hint_pipeline,
            "ProcessPoolExecutor",
            FakeActorExecutor,
        ), patch.object(
            hint_pipeline.multiprocessing,
            "get_context",
            return_value=FakeActorContext(),
        ), patch.object(
            hint_pipeline,
            "wait",
            side_effect=fake_wait,
        ), patch.object(
            hint_pipeline.sys,
            "stderr",
            io.StringIO(),
        ):
            _, cache_stats, _, _ = collect_hints(
                groups,
                [1],
                Path(temp_dir),
                workers=2,
                egaroucid_exe=Path("unused.exe"),
                egaroucid_threads=1,
                timeout_sec=1.0,
                max_retries=0,
                progress_interval_sec=60.0,
            )

        claims = Counter(FakeActorExecutor.instance.claimed)
        self.assertEqual(
            {(1, group.key) for group in groups},
            set(claims),
        )
        self.assertTrue(all(count == 1 for count in claims.values()))
        self.assertEqual(3, cache_stats[1]["scheduled_computations"])


if __name__ == "__main__":
    unittest.main()
