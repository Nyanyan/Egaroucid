import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import battle_blend_strength as battle
from strength_engine import GameRunner, PlayerProcessPool
from strength_resources import AdaptiveEnginePools, GameThreadBudget, finish_priorities
from strength_tournament import GameResult, MatchSetTask, PendingTaskQueue, PlayerSpec


class FakePool:
    def __init__(self):
        self.created_processes = 2
        self.retire_calls = 0
        self.growth = []
        self.search_threads = 1
        self.reserved_search_threads = 1
        self.pending_thread_change = False
        self.pending_thread_processes = 0
        self.pending_extra_search_threads = 0

    def search_thread_status(self):
        return {"actual_search_threads": {str(self.search_threads): self.created_processes},
                "idle_extra_search_threads": 0, "active_search_threads": 0,
                "pending_thread_processes": self.pending_thread_processes,
                "pending_extra_search_threads": self.pending_extra_search_threads}

    def grow(self, capacity):
        self.growth.append(capacity)

    def process_reservation(self, games_by_color):
        return max(self.created_processes, sum(games_by_color)), max(0, sum(games_by_color) - self.created_processes)

    def close_idle(self):
        closed = self.created_processes
        self.created_processes = 0
        return closed

    def retire(self):
        self.retire_calls += 1
        closed = self.created_processes
        self.created_processes = 0
        return closed

    def set_search_threads(self, threads):
        self.search_threads = self.reserved_search_threads = threads
        return True


class EngineResourcesTest(unittest.TestCase):
    def specs(self):
        return [PlayerSpec(name, (name,), 2, False)
                for name in ("egaroucid_l1", "egaroucid_l13", "edax_l21")]

    def controller(self, extra=600, startup=100000, minimum=1000):
        pools = [FakePool() for _ in range(3)]
        controller = AdaptiveEnginePools(
            self.specs(), pools, [300, 200, 200], [9, 0, 0],
            extra_memory_mib=extra, startup_available_mib=startup,
            minimum_available_mib=minimum, fixed_memory_mib=0,
            max_match_sets=6, output_path=None,
        )
        return controller, pools

    def test_low_levels_take_priority_without_blocking_other_pairs(self):
        low = MatchSetTask(0, 0, 1, 9, "later")
        heavy = MatchSetTask(1, 2, 3, 0, "older")
        queue = PendingTaskQueue([heavy, low], 4)
        chosen = queue.pop_schedulable([0]*4, [1]*4, player_priorities=[9, 0, 0, 0])
        self.assertEqual(low, chosen)
        self.assertEqual(heavy, queue.pop_schedulable([1, 1, 0, 0], [1]*4, player_priorities=[9, 0, 0, 0]))

    def test_finished_player_retires_and_heavy_players_receive_memory(self):
        controller, pools = self.controller(extra=0)
        pending = PendingTaskQueue([MatchSetTask(i, 1, 2, i, "x") for i in range(10)], 3)
        with patch("strength_resources.available_memory_mib", return_value=100000), redirect_stdout(io.StringIO()):
            controller.rebalance(pending, [0, 1, 1], [1, 20, 20])
            reserved = controller.reserved_memory_mib
            controller.rebalance(pending, [0, 1, 1], [1, 20, 20])
        self.assertTrue(controller.retired[0])
        self.assertEqual(1, pools[0].retire_calls)
        self.assertEqual(0, controller.capacities[0])
        self.assertGreater(sum(controller.capacities[1:]), 2)
        self.assertEqual(reserved, controller.reserved_memory_mib)
        self.assertLessEqual(reserved, controller.budget_mib)

    def test_inflight_last_set_is_not_retired_and_memory_is_not_reused_early(self):
        controller, pools = self.controller(extra=0)
        pending = PendingTaskQueue([MatchSetTask(i, 1, 2, i, "x") for i in range(4)], 3)
        with patch("strength_resources.available_memory_mib", return_value=100000), redirect_stdout(io.StringIO()):
            controller.rebalance(pending, [1, 1, 0], [1, 1, 1])
        self.assertEqual(0, pools[0].retire_calls)
        self.assertEqual([1, 1, 1], controller.capacities)

    def test_budget_reserves_lazy_processes_and_obeys_physical_and_live_limits(self):
        controller, pools = self.controller(extra=20000, startup=6000, minimum=1000)
        self.assertEqual(6000 - 1000 - 2048, controller.budget_mib)
        pending = PendingTaskQueue([MatchSetTask(i, i % 2, 2, i, "x") for i in range(100)], 3)
        for pool in pools:
            pool.created_processes = 0
        with patch("strength_resources.available_memory_mib", return_value=2500), redirect_stdout(io.StringIO()):
            controller.rebalance(pending, [0, 0, 0], [1, 1, 1])
        self.assertEqual([1, 1, 1], controller.capacities)
        with patch("strength_resources.available_memory_mib", return_value=100000), redirect_stdout(io.StringIO()):
            for _ in range(3):
                controller.rebalance(pending, [0, 0, 0], [1, 1, 1])
        self.assertGreater(sum(controller.capacities), 3)
        self.assertLessEqual(controller.reserved_memory_mib, controller.budget_mib)
        self.assertLessEqual(controller.budget_mib, controller.initial_memory_mib + 20000)

    def test_pool_cannot_retire_a_leased_engine_and_closes_all_idle_engines(self):
        manager = Mock(stop_event=threading.Event())
        pool = PlayerProcessPool(self.specs()[0], manager, 10)
        engines = [Mock(usable=True), Mock(usable=True)]
        with patch("strength_engine.GtpProcess", side_effect=engines):
            black, white = pool.acquire(0), pool.acquire(1)
        with self.assertRaisesRegex(RuntimeError, "leased"):
            pool.retire()
        pool.release(0, black)
        pool.release(1, white)
        self.assertEqual(2, pool.retire())
        self.assertEqual(0, pool.retire())
        for engine in engines:
            engine.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "retired"):
            pool.acquire(0)
        with self.assertRaises(ValueError):
            pool.grow(2)

    def test_pool_growth_keeps_existing_engines(self):
        pool = PlayerProcessPool(self.specs()[0], Mock(stop_event=threading.Event()), 10)
        with patch("strength_engine.GtpProcess", side_effect=[Mock(usable=True), Mock(usable=True)]):
            first = pool.acquire(0)
            pool.grow(2)
            second = pool.acquire(0)
        self.assertIsNot(first, second)
        self.assertEqual(2, pool.created_processes)
        first.close.assert_not_called()

    def test_idle_reclamation_keeps_active_game_and_reserves_unstarted_colors(self):
        pool = PlayerProcessPool(self.specs()[0], Mock(stop_event=threading.Event()), 10)
        first, second = Mock(usable=True), Mock(usable=True)
        with patch("strength_engine.GtpProcess", side_effect=[first, second]):
            pool.acquire(0)
            pool.acquire(1)
        pool.release(0, first)
        self.assertEqual((6, 4), pool.process_reservation((3, 3)))
        self.assertEqual(1, pool.close_idle())
        first.close.assert_called_once()
        second.close.assert_not_called()
        self.assertEqual((6, 5), pool.process_reservation((3, 3)))
        pool.release(1, second)
        self.assertEqual(1, pool.close_idle())

    def demand_controller(self, startup=120000):
        specs = [PlayerSpec("heavy", ("engine", "-t", "1"), 6, False),
                 PlayerSpec("opponent", ("engine", "-t", "1"), 2, False),
                 PlayerSpec("policy", ("wrapper",), 6, True, 1.0)]
        pools = [FakePool() for _ in specs]
        for pool in pools:
            pool.created_processes = 0
        budget = GameThreadBudget(specs)
        resources = AdaptiveEnginePools(specs, pools, [1300, 1260, 40], [0] * 3,
            extra_memory_mib=0, startup_available_mib=startup, minimum_available_mib=8192,
            fixed_memory_mib=4000, max_match_sets=35, output_path=None,
            adaptive_threads=True, thread_budget=budget)
        return resources, budget

    def admit_demand(self, resources, budget, tasks, available=120000):
        pending = PendingTaskQueue(tasks, 3)
        active = [0] * 3
        admitted = []
        with patch("strength_resources.available_memory_mib", return_value=available), redirect_stdout(io.StringIO()):
            resources.rebalance(pending, active, [1] * 3)
            while pending:
                task = pending.pop_schedulable(active, resources.capacities)
                if task is None or not budget.reserve_set(task):
                    break
                if not resources.can_admit():
                    for color in (True, False):
                        budget.release_game(task, color)
                    break
                for index in (task.p0_idx, task.p1_idx):
                    active[index] += 1
                admitted.append(task)
            budget.distribute(admitted)
        return admitted

    def test_concentrated_remainder_reserves_an_opponent_for_every_search_process(self):
        resources, budget = self.demand_controller()
        tasks = [MatchSetTask(i, 0, 2, i, "x") for i in range(250)]
        admitted = self.admit_demand(resources, budget, tasks)
        self.assertEqual(17, len(admitted))
        self.assertEqual(70, budget.allocated)
        self.assertEqual(34, len(budget.player_threads(0)))
        self.assertEqual(34, budget.allocated_policy_processes)
        self.assertEqual(68, budget.allocated_player_processes)
        self.assertEqual([1] * 32 + [2] * 2, sorted(budget.player_threads(0)))
        self.assertLessEqual(resources.reserved_memory_mib, resources.budget_mib)
        # Potential opponent slots do not consume reservations.
        self.assertEqual(4000 + 70 * 64 + 34 * (1300 + 40), resources.reserved_memory_mib)

    def test_completed_players_get_no_slots_or_new_processes(self):
        resources, budget = self.demand_controller()
        # Participant 1 has already finished; it still owns two idle engines.
        completed = resources.pools[1]
        completed.created_processes = 2
        tasks = [MatchSetTask(i, 0, 2, i, "x") for i in range(50)]
        admitted = self.admit_demand(resources, budget, tasks)
        self.assertEqual(1, completed.retire_calls)
        self.assertEqual(0, completed.created_processes)
        self.assertEqual([], completed.growth)
        self.assertEqual(0, resources.capacities[1])
        self.assertEqual([0, 0], budget.process_counts()[1])
        self.assertEqual([], budget.player_threads(1))
        # After all other admitted games finish, every reservation returns to 0.
        for task in admitted:
            for color in (True, False):
                budget.release_game(task, color)
        with patch("strength_resources.available_memory_mib", return_value=120000), redirect_stdout(io.StringIO()):
            resources.rebalance(PendingTaskQueue([], 3), [0, 0, 0], [1] * 3)
        self.assertEqual([0, 0, 0], resources.capacities)
        self.assertEqual(0, budget.allocated)
        self.assertEqual([1, 1, 1], [pool.retire_calls for pool in resources.pools])

    def test_concentrated_model_limit_follows_complete_pairs_and_budget(self):
        for total in (12, 40, 70, 96, 128):
            for opponent_alpha in (None, 1.0):
                with self.subTest(total=total, opponent_alpha=opponent_alpha):
                    specs = [PlayerSpec("heavy", (), 200, False),
                             PlayerSpec("opponent", (), 200, False, opponent_alpha)]
                    budget = GameThreadBudget(specs, total=total)
                    admitted = []
                    for i in range(100):
                        task = MatchSetTask(i, 0, 1, i, "x")
                        if not budget.reserve_set(task):
                            break
                        admitted.append(task)
                    budget.distribute(admitted)
                    counts = [sum(colors) for colors in budget.process_counts()]
                    self.assertEqual([2 * (total // 4)] * 2, counts)
                    self.assertLessEqual(counts[0], total // 2)
                    self.assertEqual(total, budget.allocated)

    def test_live_edax_returns_slots_only_after_ack_and_grows_after_games_finish(self):
        specs = [PlayerSpec("edax_l13", (), 20, False), PlayerSpec("policy", (), 20, True, 1.0)]
        budget = GameThreadBudget(specs, total=12)
        first, second = [MatchSetTask(i, 0, 1, i, "x") for i in range(2)]
        budget.reserve_set(first)
        budget.distribute([first])
        self.assertEqual([5, 5], budget.player_threads(0))
        budget.begin_admission()
        self.assertFalse(budget.reserve_set(second))
        budget.distribute([])
        changed = []
        def acknowledge(threads):
            self.assertEqual(12, budget.allocated)  # Slots not released before ACK.
            changed.append(threads)
        budget.resize_at_move_boundary(first, True, 0, acknowledge)
        self.assertEqual([1], changed)
        self.assertEqual(8, budget.allocated)
        budget.begin_admission()
        self.assertTrue(budget.reserve_set(second))
        budget.distribute([second])
        for color in (True, False):
            budget.release_game(second, color)
        budget.begin_admission()
        budget.distribute([])
        change = Mock()
        budget.resize_at_move_boundary(first, True, 0, change)
        change.assert_called_once_with(5)
        self.assertEqual(12, budget.allocated)

    def test_failed_live_resize_keeps_reservation_and_does_not_oversubscribe(self):
        specs = [PlayerSpec("edax_l13", (), 20, False), PlayerSpec("policy", (), 20, True, 1.0)]
        budget = GameThreadBudget(specs, total=12)
        first, second = [MatchSetTask(i, 0, 1, i, "x") for i in range(2)]
        budget.reserve_set(first)
        budget.distribute([first])
        budget.begin_admission()
        self.assertFalse(budget.reserve_set(second))
        budget.distribute([])
        with self.assertRaises(TimeoutError):
            budget.resize_at_move_boundary(first, True, 0, Mock(side_effect=TimeoutError))
        self.assertEqual(12, budget.allocated)
        self.assertFalse(budget.reserve_set(second))
        self.assertEqual(0, budget.live_thread_changes)

    def test_concurrent_admission_waits_for_live_resize_acknowledgement(self):
        specs = [PlayerSpec("edax_l13", (), 20, False), PlayerSpec("policy", (), 20, True, 1.0)]
        budget = GameThreadBudget(specs, total=12)
        first, second = [MatchSetTask(i, 0, 1, i, "x") for i in range(2)]
        budget.reserve_set(first)
        budget.distribute([first])
        budget.begin_admission()
        budget.reserve_set(second)
        budget.distribute([])
        changing, acknowledge, reserved = threading.Event(), threading.Event(), threading.Event()
        result = []
        def change(threads):
            changing.set()
            if not acknowledge.wait(3):
                raise TimeoutError("test resize was not acknowledged")
        def admit():
            result.append(budget.reserve_set(second))
            reserved.set()
        resize_worker = threading.Thread(target=budget.resize_at_move_boundary,
                                         args=(first, True, 0, change))
        admit_worker = threading.Thread(target=admit)
        resize_worker.start()
        try:
            self.assertTrue(changing.wait(3))
            admit_worker.start()
            self.assertFalse(reserved.wait(.05))
        finally:
            acknowledge.set()
            resize_worker.join(3)
            if admit_worker.ident is not None:
                admit_worker.join(3)
        self.assertEqual([True], result)
        self.assertEqual(12, budget.allocated)

    def test_does_not_reclaim_threads_when_they_cannot_fit_another_pair(self):
        specs = [PlayerSpec("edax_l13", (), 40, False), PlayerSpec("policy", (), 40, True, 1.0)]
        budget = GameThreadBudget(specs)
        tasks = [MatchSetTask(i, 0, 1, i, "x") for i in range(18)]
        for task in tasks[:17]:
            self.assertTrue(budget.reserve_set(task))
        budget.distribute(tasks[:17])
        budget.begin_admission()
        self.assertFalse(budget.reserve_set(tasks[-1]))
        budget.distribute([])
        change = Mock()
        for task in tasks[:17]:
            for color in (True, False):
                budget.resize_at_move_boundary(task, color, 0, change)
        change.assert_not_called()
        self.assertEqual(70, budget.allocated)

    def test_memory_limit_reduces_processes_then_assigns_remaining_threads(self):
        resources, budget = self.demand_controller(startup=30000)
        tasks = [MatchSetTask(i, 0, 1, i, "x") for i in range(250)]
        admitted = self.admit_demand(resources, budget, tasks, available=30000)
        self.assertEqual(2, len(admitted))
        self.assertEqual(70, budget.allocated)
        self.assertEqual([8, 8, 9, 9, 9, 9, 9, 9], sorted(
            t for plan in budget.allocations.values() for t in plan.values()))
        self.assertLessEqual(resources.reserved_memory_mib, resources.budget_mib)

    def test_short_remainder_assigns_threads_without_modifying_running_games(self):
        resources, budget = self.demand_controller()
        tasks = [MatchSetTask(i, 0, 1, i, "x") for i in range(2)]
        admitted = self.admit_demand(resources, budget, tasks)
        previous = {key: dict(plan) for key, plan in budget.allocations.items()}
        self.assertEqual(2, len(admitted))
        self.assertEqual(70, budget.allocated)
        budget.release_game(tasks[0], True)
        with patch("strength_resources.available_memory_mib", return_value=120000):
            resources.rebalance(PendingTaskQueue([], 3), resources.active_counts, [1] * 3)
        self.assertTrue(all(plan == previous[key] for key, plan in budget.allocations.items()))

    def test_blocked_expensive_pair_does_not_hide_an_affordable_pair(self):
        queue = PendingTaskQueue([MatchSetTask(0, 0, 1, 0, "x"), MatchSetTask(1, 0, 2, 1, "x")], 3)
        self.assertEqual(1, queue.pop_schedulable([0] * 3, [35] * 3,
                                                excluded_pairs={(0, 1)}).task_id)
        self.assertEqual(0, queue.pop_schedulable([0] * 3, [35] * 3).task_id)

    def test_scheduler_reserves_both_players_within_70_slots_and_completes_both_colors(self):
        reached = threading.Event()
        lock = threading.Lock()
        plans = []
        engine_commands = []
        runner = None

        def make_runner(specs, manager, timeout, **kwargs):
            nonlocal runner
            runner = GameRunner(specs, manager, timeout, **kwargs)
            runner.play_single_game = play
            return runner

        def make_engine(manager, command, timeout):
            with lock:
                engine_commands.append(command)
            return Mock(usable=True)

        def play(task, color, search_threads):
            leases = []
            try:
                for index, color_pool in ((task.p0_idx, 0 if color else 1),
                                          (task.p1_idx, 1 if color else 0)):
                    pool = runner.pools[index]
                    engine = pool.acquire(color_pool, search_threads.get(index))
                    leases.append((pool, color_pool, engine))
                with lock:
                    plans.append(search_threads)
                    if len(plans) == 34:
                        reached.set()
                if not reached.wait(5):
                    raise AssertionError("34 games did not start together")
                black, white = (task.p0_idx, task.p1_idx) if color else (task.p1_idx, task.p0_idx)
                return GameResult(task.p0_idx, task.p1_idx, color, black, white, 0, 32, 32, task.opening)
            finally:
                for pool, color_pool, engine in leases:
                    pool.release(color_pool, engine)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "engine.exe"
            engine.write_bytes(b"mock engine")
            model = root / "model.h5"
            model.write_bytes(b"mock model")
            openings = root / "openings.txt"
            openings.write_text("f5d6c3d3c4f4c5b3\n")
            args = battle.make_argparser().parse_args([
                "--egaroucid-exe", str(engine), "--policy-model", str(model),
                "--openings", str(openings), "--output-dir", str(root / "results"),
                "--baseline-levels", "13", "--alphas", "1.0", "--edax-levels=",
                "--match-sets-per-pair", "50", "--parallel-games", "70",
                "--parallel-match-sets", "35", "--adaptive-engine-threads", "--no-performance-monitor",
            ])
            with patch.object(battle, "GameRunner", side_effect=make_runner), \
                 patch("strength_engine.GtpProcess", side_effect=make_engine), \
                 patch.object(battle, "available_memory_mib", return_value=120000), \
                 patch("strength_resources.available_memory_mib", return_value=120000), \
                 patch.object(battle, "start_policy_server", return_value=battle.PolicyServerInfo(100, "tensorflow", "CPU", "test")), \
                 patch.object(battle, "validate_input_files"), \
                 patch.object(battle, "validate_scheduled_openings"), redirect_stdout(io.StringIO()):
                battle.run_tournament(args)
            rows = [json.loads(line) for line in (args.output_dir / "strength_games.jsonl").read_text().splitlines()]
            self.assertEqual(50, len(rows))
            self.assertEqual(100, len(plans))
            self.assertEqual(36, sum(plan[0] for plan in plans[:34]))
            self.assertEqual([1] * 32 + [2] * 2, sorted(plan[0] for plan in plans[:34]))
            native = [command for command in engine_commands if "-t" in command]
            self.assertGreaterEqual(len(native), 34)
            self.assertEqual([1] * 32 + [2] * 2,
                             sorted(int(command[command.index("-t") + 1]) for command in native[:34]))

    def test_per_game_allocations_reuse_matching_engines_and_restart_only_after_release(self):
        spec = PlayerSpec("egaroucid_l13", ("engine", "-t", "1"), 2, False)
        pool = PlayerProcessPool(spec, Mock(stop_event=threading.Event()), 10)
        first, second, third = [Mock(usable=True) for _ in range(3)]
        with patch("strength_engine.GtpProcess", side_effect=[first, second, third]) as factory:
            pool.acquire(0, search_threads=2)
            pool.acquire(1, search_threads=3)
            self.assertEqual({"2": 1, "3": 1}, pool.search_thread_status()["actual_search_threads"])
            self.assertFalse(pool.pending_thread_change)
            pool.release(0, first)
            self.assertEqual(1, pool.search_thread_status()["idle_extra_search_threads"])
            self.assertIs(first, pool.acquire(0, search_threads=2))
            first.close.assert_not_called()
            pool.release(0, first)
            self.assertIs(third, pool.acquire(0, search_threads=4))
            first.close.assert_called_once()
            second.close.assert_not_called()
            self.assertEqual("4", factory.call_args.args[1][-1])

    def test_budget_reserves_extra_threads_once_and_keeps_idle_memory_accounted(self):
        specs = self.specs()
        pools = [FakePool() for _ in specs]
        budget = GameThreadBudget(specs, total=64)
        controller = AdaptiveEnginePools(specs, pools, [300, 200, 200], [9, 0, 0],
            extra_memory_mib=600, startup_available_mib=100000, minimum_available_mib=1000,
            fixed_memory_mib=0, max_match_sets=6, output_path=None,
            adaptive_threads=True, thread_budget=budget)
        self.assertEqual(1400 + 64 * 64, controller.reserved_memory_mib)
        status = pools[0].search_thread_status()
        status["idle_extra_search_threads"] = 5
        pools[0].search_thread_status = lambda: status
        self.assertEqual(1400 + 69 * 64, controller.reserved_memory_mib)

    def test_pool_thread_change_preserves_active_lease_and_uses_correct_flags(self):
        for name, alpha, flag in (("egaroucid_l13", None, "-t"),
                                  ("edax_l21", None, "-n"), ("alpha_0.6", .6, "--egaroucid-threads")):
            with self.subTest(name=name):
                spec = PlayerSpec(name, ("engine", flag, "1"), 2, alpha is not None, alpha)
                pool = PlayerProcessPool(spec, Mock(stop_event=threading.Event()), 10)
                first, second = Mock(usable=True), Mock(usable=True)
                with patch("strength_engine.GtpProcess", side_effect=[first, second]) as factory:
                    self.assertIs(first, pool.acquire(0))
                    self.assertTrue(pool.set_search_threads(2))
                    first.close.assert_not_called()
                    self.assertTrue(pool.pending_thread_change)
                    pool.release(0, first)
                    first.close.assert_called_once()
                    self.assertIs(second, pool.acquire(0))
                    self.assertEqual(("engine", flag, "2"), factory.call_args.args[1])
                    self.assertFalse(pool.pending_thread_change)
                    pool.set_search_threads(1)
                    self.assertEqual(2, pool.reserved_search_threads)
                    pool.release(0, second)
                    self.assertEqual(1, pool.reserved_search_threads)
        policy = PlayerSpec("alpha_1.0", ("python", "--egaroucid-threads", "1"), 2, True, 1.0)
        self.assertFalse(PlayerProcessPool(policy, Mock(), 10).set_search_threads(2))

    def test_thread_reservations_count_only_engines_still_awaiting_rollover(self):
        spec = PlayerSpec("egaroucid_l13", ("engine", "-t", "1"), 2, False)
        pool = PlayerProcessPool(spec, Mock(stop_event=threading.Event()), 10)
        engines = [Mock(usable=True) for _ in range(3)]
        with patch("strength_engine.GtpProcess", side_effect=engines):
            old_black, old_white = pool.acquire(0), pool.acquire(1)
            pool.set_search_threads(2)
            self.assertEqual(2, pool.search_thread_status()["pending_extra_search_threads"])
            pool.release(0, old_black)
            pool.acquire(0)
        self.assertEqual({"actual_search_threads": {"1": 1, "2": 1},
                          "pending_thread_processes": 1, "pending_extra_search_threads": 1,
                          "active_search_threads": 3, "idle_extra_search_threads": 0},
                         pool.search_thread_status())
        old_white.close.assert_not_called()
        pool.set_search_threads(1)
        self.assertEqual(0, pool.search_thread_status()["pending_extra_search_threads"])

    def test_thread_change_does_not_restart_native_or_blend_engine_midgame(self):
        from blend_policy_with_egaroucid import BoardState, coord_to_policy, parse_side, policy_to_coord
        specs = [PlayerSpec("egaroucid_l1", ("native", "-t", "1"), 2, False),
                 PlayerSpec("alpha_0.6", ("blend", "--egaroucid-threads", "1"), 2, True, .6)]
        runner = GameRunner(specs, Mock(stop_event=threading.Event()), 10, game_boundary_threads=True)
        created = []
        class Engine:
            def __init__(self, manager, command, timeout):
                self.command, self.usable = command, True
                self.state = BoardState.initial()
                self.moves = 0
                created.append(self)
            def close(self):
                self.usable = False
            def request(self, command):
                self_test.assertTrue(self.usable)
                parts = command.split()
                if parts[0] == "clear_board":
                    self.state = BoardState.initial()
                elif parts[0] == "setboard":
                    self.state = BoardState.from_board_string(parts[1], parse_side(parts[2]))
                elif parts[0] == "play":
                    side = parse_side(parts[1])
                    if parts[2] == "pass":
                        self.state.side = side ^ 1
                    else:
                        self.state.apply_move(side, coord_to_policy(parts[2]))
                elif parts[0] == "genmove":
                    side = parse_side(parts[1])
                    move = self.state.legal_policies(side)[0]
                    self.state.apply_move(side, move)
                    self.moves += 1
                    # Simulate a CPU feedback update while the first game is running.
                    if len(created) == 2:
                        for pool in runner.pools:
                            pool.set_search_threads(2)
                        self_test.assertTrue(all(engine.usable for engine in created))
                        self_test.assertTrue(all(engine.command[-1] == "1" for engine in created))
                    return "= " + policy_to_coord(move)
                return "="
        self_test = self
        task = MatchSetTask(0, 0, 1, 0, "f5d6c3d3c4f4c5b3")
        with patch("strength_engine.GtpProcess", Engine):
            first = runner.play_single_game(task, True)
            self.assertEqual(2, len(created))
            self.assertTrue(all(engine.moves > 10 and not engine.usable for engine in created))
            second = runner.play_single_game(task, True)
        self.assertEqual(4, len(created))
        self.assertTrue(all(engine.command[-1] == "2" for engine in created[2:]))
        self.assertEqual(first.transcript, second.transcript)

    def test_priority_order_is_level_1_then_5_then_9_and_excludes_blends(self):
        specs = [PlayerSpec(name, (), 2, False, alpha) for name, alpha in
                 [("edax_l5", None), ("egaroucid_l1", None), ("edax_l9", None),
                  ("egaroucid_l13", None), ("alpha_1.0", 1.0)]]
        self.assertEqual([5, 9, 1, 0, 0], finish_priorities(specs, True))

    def test_completed_tournament_retires_every_pool_and_saves_both_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "engine.exe"
            engine.write_bytes(b"mock engine")
            openings = root / "openings.txt"
            openings.write_text("f5d6c3d3c4f4c5b3\n", encoding="utf-8")
            args = battle.make_argparser().parse_args([
                "--egaroucid-exe", str(engine), "--openings", str(openings),
                "--output-dir", str(root / "results"), "--baseline-levels", "1,13,17",
                "--alphas=", "--edax-levels=", "--match-sets-per-pair", "3",
                "--parallel-match-sets", "3", "--no-performance-monitor",
                "--finish-low-levels-first", "--extra-engine-memory-mib", "4000",
                "--adaptive-engine-threads",
            ])
            pools = [FakePool() for _ in range(3)]
            game_plans, distributions = [], []
            distribute = GameThreadBudget.distribute
            def record_distribution(budget, tasks):
                distribute(budget, tasks)
                if tasks:
                    distributions.append(budget.allocated)
            def play(task, color, search_threads):
                game_plans.append(search_threads)
                black = task.p0_idx if color else task.p1_idx
                white = task.p1_idx if color else task.p0_idx
                return GameResult(task.p0_idx, task.p1_idx, color, black, white, 0, 32, 32, task.opening)
            with patch.object(battle, "GameRunner") as game_runner, \
                 patch.object(battle, "available_memory_mib", return_value=1e9), \
                 patch("strength_resources.available_memory_mib", return_value=1e9), \
                 patch.object(GameThreadBudget, "distribute", record_distribution), \
                 patch.object(battle, "validate_scheduled_openings"), redirect_stdout(io.StringIO()):
                game_runner.return_value.pools = pools
                game_runner.return_value.play_single_game.side_effect = play
                battle.run_tournament(args)
            rows = [json.loads(line) for line in (args.output_dir / "strength_games.jsonl").read_text().splitlines()]
            self.assertEqual(9, len(rows))
            self.assertEqual(18, sum(len(row["result"]["color_games"]) for row in rows))
            self.assertEqual(18, len(game_plans))
            self.assertTrue(all(len(plan) == 2 and min(plan.values()) >= 1 for plan in game_plans))
            self.assertTrue(distributions)
            self.assertEqual({70}, set(distributions))
            self.assertEqual([1, 1, 1], [pool.retire_calls for pool in pools])
            state = json.loads((args.output_dir / "engine_allocation_live.json").read_text())
            self.assertTrue(all(player["retired"] for player in state["players"]))
            self.assertEqual(70, state["thread_budget"])
            self.assertEqual(0, state["allocated_search_threads"])
            self.assertEqual("least_projected_completion_first", state["admission_priority"])
            self.assertTrue(all(player["completed_sets"] == player["target_sets"] == 6
                                for player in state["players"]))


class ProgressSchedulingTest(unittest.TestCase):
    def choose(self, tasks, completed, active=None, targets=None, capacities=None, **kwargs):
        n = len(completed)
        return PendingTaskQueue(tasks, n).pop_schedulable(
            active or [0] * n, capacities or [35] * n,
            completed_counts=completed, target_counts=targets or [100] * n, **kwargs)

    def test_lagging_model_precedes_older_opening_and_fast_low_level_priority(self):
        fast = MatchSetTask(0, 1, 2, 0, "older")
        lagging = MatchSetTask(1, 0, 3, 40, "later")
        self.assertEqual(lagging, self.choose([fast, lagging], [10, 90, 90, 90],
                                             player_priorities=[0, 9, 9, 0]))

    def test_two_lagging_models_are_advanced_together(self):
        mixed = MatchSetTask(0, 0, 2, 0, "older")
        both = MatchSetTask(1, 0, 1, 10, "later")
        self.assertEqual(both, self.choose([mixed, both], [10, 10, 90]))

    def test_catchup_allocates_more_processes_and_stops_at_the_projected_gap(self):
        tasks = ([MatchSetTask(i, 0, 3, i, "x") for i in range(30)]
                 + [MatchSetTask(i + 30, 1, 2, i, "x") for i in range(30)])
        pending = PendingTaskQueue(tasks, 4)
        active = [0] * 4
        picked = []
        for _ in range(18):
            task = pending.pop_schedulable(active, [35] * 4,
                                           completed_counts=[80, 90, 90, 90], target_counts=[100] * 4)
            picked.append(task)
            active[task.p0_idx] += 1
            active[task.p1_idx] += 1
        self.assertTrue(all(task.p0_idx == 0 for task in picked[:10]))
        self.assertEqual((1, 2), (picked[10].p0_idx, picked[10].p1_idx))
        self.assertGreater(active[0], active[1])
        # Two color games per set mean more processes for the lagging model.
        self.assertGreater(2 * active[0], 2 * active[2])

    def test_busy_or_memory_blocked_lagging_model_does_not_idle_others(self):
        lagging = MatchSetTask(0, 0, 1, 0, "x")
        other = MatchSetTask(1, 2, 3, 10, "y")
        self.assertEqual(other, self.choose([lagging, other], [10, 90, 90, 90],
                                            active=[1, 0, 0, 0], capacities=[1, 35, 35, 35]))
        self.assertEqual(other, self.choose([lagging, other], [10, 90, 90, 90],
                                            excluded_pairs={(0, 1)}))

    def test_different_targets_use_completion_fraction(self):
        small = MatchSetTask(0, 0, 2, 0, "x")
        large = MatchSetTask(1, 1, 2, 10, "y")
        self.assertEqual(large, self.choose([small, large], [8, 40, 95], targets=[10, 100, 100]))

    def test_failing_and_requeued_work_does_not_count_as_completed(self):
        lagging = MatchSetTask(0, 0, 1, 10, "x")
        other = MatchSetTask(1, 2, 3, 0, "y")
        pending = PendingTaskQueue([other, lagging], 4)
        kwargs = dict(completed_counts=[10, 90, 90, 90], target_counts=[100] * 4)
        failed = pending.pop_schedulable([0] * 4, [35] * 4, **kwargs)
        self.assertEqual(lagging, failed)
        pending.push_front(failed)
        self.assertEqual(lagging, pending.pop_schedulable([0] * 4, [35] * 4, **kwargs))
        self.assertEqual(other, pending.pop_schedulable([0] * 4, [35] * 4, **kwargs))
        self.assertFalse(pending)

    def test_catchup_preserves_each_pair_openings_and_exact_task_set(self):
        tasks = [MatchSetTask(3 * j + i, i, 3, j, str(j)) for j in range(4) for i in range(3)]
        pending = PendingTaskQueue(tasks, 4)
        completed, picked = [5, 10, 10, 25], []
        while pending:
            task = pending.pop_schedulable([0] * 4, [35] * 4,
                                           completed_counts=completed, target_counts=[14, 14, 14, 42])
            picked.append(task)
            completed[task.p0_idx] += 1
            completed[task.p1_idx] += 1
        self.assertEqual({task.task_id for task in tasks}, {task.task_id for task in picked})
        self.assertEqual(len(tasks), len(picked))
        for i in range(3):
            self.assertEqual([0, 1, 2, 3], [task.set_index for task in picked if task.p0_idx == i])


class GameThreadBudgetTest(unittest.TestCase):
    def specs(self):
        return [PlayerSpec("native", ("engine", "-t", "1"), 64, False),
                PlayerSpec("blend", ("wrapper", "--egaroucid-threads", "1"), 64, True, .6),
                PlayerSpec("policy", ("wrapper",), 64, True, 1.0)]

    def test_even_and_remainder_allocations_total_64(self):
        for count, players in ((8, (0, 1)), (4, (0, 1)), (10, (0, 1)), (16, (0, 2))):
            with self.subTest(count=count, players=players):
                budget = GameThreadBudget(self.specs(), total=64)
                tasks = [MatchSetTask(i, *players, i, "x") for i in range(count)]
                for task in tasks:
                    self.assertTrue(budget.reserve_set(task))
                budget.distribute(tasks)
                allocations = [threads for plan in budget.allocations.values() for threads in plan.values()]
                self.assertEqual(64, sum(allocations) + budget.allocated_policy_processes)
                self.assertLessEqual(max(allocations) - min(allocations), 1)
                self.assertFalse(budget.player_threads(2))

    def test_running_games_keep_their_budget_as_completed_slots_are_reassigned(self):
        budget = GameThreadBudget(self.specs(), total=64)
        tasks = [MatchSetTask(i, 0, 1, i, "x") for i in range(8)]
        for task in tasks:
            budget.reserve_set(task)
        budget.distribute(tasks)
        ongoing = budget.for_game(tasks[0], False)
        self.assertFalse(budget.reserve_set(MatchSetTask(10, 0, 1, 10, "x")))
        budget.release_game(tasks[0], True)
        next_task = MatchSetTask(10, 0, 1, 10, "x")
        self.assertTrue(budget.reserve_set(next_task))
        budget.distribute([next_task])
        self.assertEqual(ongoing, budget.for_game(tasks[0], False))
        self.assertEqual(64, budget.allocated)
        self.assertEqual({0: 1, 1: 1}, budget.for_game(next_task, True))

    def test_cap_and_policy_only_games_leave_unused_threads_without_overbooking(self):
        specs = self.specs() + [PlayerSpec("policy2", ("wrapper",), 64, True, 1.0)]
        budget = GameThreadBudget(specs, max_threads=4)
        task = MatchSetTask(0, 0, 1, 0, "x")
        budget.reserve_set(task)
        budget.distribute([task])
        self.assertEqual(16, budget.allocated)
        policy = MatchSetTask(1, 2, 3, 0, "x")
        self.assertTrue(budget.reserve_set(policy))
        budget.distribute([policy])
        self.assertEqual(20, budget.allocated)
        self.assertEqual({}, budget.for_game(policy, True))
        for color in (True, False):
            budget.release_game(task, color)
            budget.release_game(policy, color)
        self.assertEqual(0, budget.allocated)

    def test_admission_cannot_overbook_64_even_before_engines_have_started(self):
        budget = GameThreadBudget(self.specs(), total=64)
        tasks = [MatchSetTask(i, 0, 1, i, "x") for i in range(17)]
        for task in tasks[:16]:
            self.assertTrue(budget.reserve_set(task))
        self.assertFalse(budget.reserve_set(tasks[-1]))
        self.assertEqual(64, budget.allocated)
        self.assertEqual(32, len(budget.allocations))


if __name__ == "__main__":
    unittest.main()
