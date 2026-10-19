"""Global tail transition, single-game scheduling, and TT reset/reuse checks."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import battle_blend_strength as battle
from blend_gtp_engine import BlendGtpEngine
from blend_policy_with_egaroucid import (
    BoardState, EgaroucidHintRunner, SharedHintCache, default_egaroucid_exe,
)
from strength_engine import GameRunner, ProcessManager
from strength_tournament import MatchSetTask, PlayerSpec
from test_strength_tournament_defaults import OPENING, OPENINGS, game_result


class TailSchedulingTest(unittest.TestCase):
    def run_schedule(self, count, threshold, *, abandoned_at_start=False):
        calls, cancellations, runners = [], [], []
        blocked_started = threading.Event()
        lock = threading.Lock()
        old_waiters = 0
        live = 0
        maximum = 0
        cached_scores = ({19: 4.0, 26: -2.0}, "completed search")

        def leave_claim(color):
            cache = SharedHintCache(root / "results/egaroucid_hint_cache_v2.sqlite3")
            try:
                # Use a fresh claim with the production 12-hour expiry. An
                # expired dummy claim would hide the original long wait.
                if not cache.try_claim(f"cancelled-{color}", 43200.0):
                    raise AssertionError("Test claim already held")
                cache.put("completed-position", *cached_scores)
            finally:
                cache.close()

        def verify_tail_cache():
            cache = SharedHintCache(root / "results/egaroucid_hint_cache_v2.sqlite3")
            try:
                if cache.get("completed-position") != cached_scores:
                    raise AssertionError("Cleanup discarded a completed search")
                for color in (True, False):
                    if cache.try_claim(f"cancelled-{color}", 43200.0):
                        raise AssertionError("An unused legacy cache was modified")
            finally:
                cache.close()

        class FakeRunner:
            def __init__(self, specs, manager, timeout, **kwargs):
                self.manager = manager
                self.tail = kwargs.get("sequential_fresh_tt", False)
                self.pools = [Mock(created_processes=0) for _ in specs]
                runners.append(self)

            def play_single_game(self, task, black, search_threads=None):
                nonlocal old_waiters, live, maximum
                if not self.tail:
                    if task.set_index == 1:
                        with lock:
                            # Simulate a killed search that leaves its claim
                            # behind. Tail games must not wait on this worker.
                            leave_claim(black)
                            old_waiters += 1
                            if old_waiters == 2:
                                blocked_started.set()
                        if not self.manager.stop_event.wait(10):
                            raise TimeoutError("transition did not cancel running games")
                        cancellations.append((task.task_id, black))
                        raise InterruptedError("intentional tail cancellation")
                    if not blocked_started.wait(10):
                        raise TimeoutError("both colors of second set were not started")
                if self.tail and (old_waiters or abandoned_at_start):
                    verify_tail_cache()
                with lock:
                    live += 1
                    maximum = max(maximum, live) if self.tail else maximum
                    calls.append((task.task_id, black, self.tail, search_threads))
                result = replace(game_result(black, 2), transcript=task.opening)
                with lock:
                    live -= 1
                return result

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine, openings = root / "engine.exe", root / "openings.txt"
            engine.write_bytes(b"fake")
            openings.write_text("\n".join(OPENINGS) + "\n")
            if abandoned_at_start:
                for color in (True, False):
                    leave_claim(color)
            args = battle.make_argparser().parse_args([
                "--egaroucid-exe", str(engine), "--openings", str(openings),
                "--output-dir", str(root / "results"), "--baseline-levels", "1,2",
                "--alphas=", "--edax-levels=", "--match-sets-per-pair", str(count),
                "--parallel-match-sets", "2", "--parallel-games", "4",
                "--baseline-processes-per-player", "4", "--no-performance-monitor",
                "--sequential-tail-match-sets", str(threshold),
            ])
            with patch.object(battle, "GameRunner", FakeRunner), \
                 patch.object(battle, "available_memory_mib", return_value=1e9), \
                 patch.object(battle, "validate_scheduled_openings"), redirect_stdout(io.StringIO()):
                battle.run_tournament(args)
            rows = (args.output_dir / "strength_games.jsonl").read_text().splitlines()
            self.assertEqual(count, len(rows))
            status = json.loads((args.output_dir / "engine_allocation_live.json").read_text())
            self.assertEqual(0, status["remaining_match_sets"])
            self.assertEqual(0, status["allocated_player_processes"])
            failure = args.output_dir / "strength_failed_tasks.jsonl"
            self.assertFalse(failure.exists() and failure.read_text().strip())
        self.assertEqual(1, maximum)
        tail_calls = [c for c in calls if c[2]]
        for first, second in zip(tail_calls[::2], tail_calls[1::2]):
            self.assertEqual((first[0], True), first[:2])
            self.assertEqual((first[0], False), second[:2])
            self.assertEqual({0: 32, 1: 32}, first[3])
            self.assertEqual(first[3], second[3])
        return calls, cancellations

    def test_global_36_to_35_cancels_uncommitted_pair_without_failure_or_replay(self):
        calls, cancellations = self.run_schedule(36, 35)
        self.assertEqual(2, len(cancellations))
        self.assertEqual(72, len(calls))
        self.assertEqual(2, sum(not call[2] for call in calls))

    def test_resume_below_threshold_starts_directly_in_single_game_mode(self):
        calls, cancellations = self.run_schedule(2, 35, abandoned_at_start=True)
        self.assertFalse(cancellations)
        self.assertTrue(all(call[2] for call in calls))


class TailTTTest(unittest.TestCase):
    def test_color_swap_reuses_same_pools_and_clears_tt_after_each_game(self):
        specs = [PlayerSpec("blend_a", ("fake",), 2, True, .6),
                 PlayerSpec("blend_b", ("fake",), 2, True, .8)]
        manager = ProcessManager(1)
        runner = GameRunner(specs, manager, 5, game_boundary_threads=True, sequential_fresh_tt=True)
        engines = [Mock(), Mock()]
        pools = [Mock(), Mock()]
        for pool, engine in zip(pools, engines):
            pool.acquire.return_value = engine
            engine.request.return_value = "="
        runner.pools = pools
        # A completed board avoids exercising the unrelated move generator.
        state = BoardState.from_board_string("X" * 64, 0)
        with patch("strength_engine.BoardState.initial", return_value=state):
            for color in (True, False):
                runner.play_single_game(MatchSetTask(0, 0, 1, 0, ""), color, {0: 32, 1: 32})
        for pool, engine in zip(pools, engines):
            self.assertEqual(2, pool.acquire.call_count)
            for call in pool.acquire.call_args_list:
                self.assertEqual((0,), call.args)
                self.assertEqual({"search_threads": 32}, call.kwargs)
            self.assertEqual([("clear_cache",), ("clear_cache",)],
                             [call.args for call in engine.request.call_args_list])
            pool.close_idle.assert_not_called()

    def test_native_blend_tt_clear_keeps_pid_and_search_usable(self):
        exe = default_egaroucid_exe()
        if not Path(exe).is_file():
            self.skipTest("native Egaroucid unavailable")
        runner = EgaroucidHintRunner(exe, level=1, threads=32, timeout_sec=30)
        wrapper = object.__new__(BlendGtpEngine)
        wrapper.blender = Mock(hint_runner=runner)
        try:
            state = BoardState.initial()
            runner.raw_hint(state, state.side)
            pid = runner.proc.pid
            for _ in range(2):
                self.assertEqual((True, ""), wrapper.dispatch("clear_cache", []))
                self.assertEqual(pid, runner.proc.pid)
                self.assertIsNone(runner.proc.poll())
                self.assertTrue(runner.raw_hint(state, state.side))
        finally:
            proc = runner.proc
            runner.close()
            if proc is not None:
                for stream in (proc.stdin, proc.stdout):
                    if stream is not None:
                        stream.close()


if __name__ == "__main__":
    unittest.main()
