#!/usr/bin/env python3

from dataclasses import replace
import csv
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import battle_blend_strength
import run_strength_full
import strength_reporting
from strength_engine import GtpProcess, ProcessManager
from strength_reporting import (
    PerformanceMonitor,
    build_pair_rows,
    build_summary_rows,
    build_text_report,
    estimate_elos,
    write_outputs,
)
from strength_tournament import (
    GameResult,
    PendingTaskQueue,
    PlayerSpec,
    ResultStore,
    TournamentStats,
    combine_color_games,
    conservative_score_half_width,
    make_match_set_tasks,
    normalized_duration_weights,
    target_match_sets_by_pair,
    score_confidence_interval,
    student_t_critical_975,
)


OPENING = "f5d6c3d3c4f4c5b3"


def game_result(p0_is_black: bool, disc_diff: int) -> GameResult:
    return GameResult(
        p0_idx=0,
        p1_idx=1,
        p0_is_black=p0_is_black,
        black_idx=0 if p0_is_black else 1,
        white_idx=1 if p0_is_black else 0,
        p0_disc_diff=disc_diff,
        black_stones=32 + disc_diff // 2,
        white_stones=32 - disc_diff // 2,
        transcript=OPENING,
    )


class StrengthTournamentDefaultsTest(unittest.TestCase):
    @staticmethod
    def two_player_specs():
        return [
            PlayerSpec("player_0", ("player_0",), 2, False),
            PlayerSpec("player_1", ("player_1",), 2, False),
        ]

    def test_defaults_define_the_500_set_tournament(self) -> None:
        args = battle_blend_strength.make_argparser().parse_args([])
        levels, alphas = battle_blend_strength.validate_args(args)
        player_count = len(levels) + len(alphas)
        tasks = make_match_set_tasks(
            player_count,
            [OPENING],
            args.match_sets_per_pair,
            args.seed,
        )

        self.assertEqual(500, args.match_sets_per_pair)
        self.assertEqual(16, player_count)
        self.assertEqual(120, player_count * (player_count - 1) // 2)
        self.assertEqual(60_000, len(tasks))
        self.assertEqual(120_000, len(tasks) * 2)
        self.assertEqual(20, args.parallel_match_sets)
        self.assertEqual(2, args.baseline_processes_per_player)
        self.assertEqual(10, args.blend_processes_per_player)
        self.assertEqual(1, args.engine_threads)
        help_text = battle_blend_strength.make_argparser().format_help().lower()
        self.assertNotIn("random", help_text)
        self.assertNotIn("external", help_text)

    def test_wrapper_uses_the_only_parser(self) -> None:
        self.assertIs(
            battle_blend_strength.make_argparser,
            run_strength_full.make_argparser,
        )
        args = run_strength_full.make_argparser().parse_args([])
        self.assertEqual(500, args.match_sets_per_pair)
        self.assertEqual("xot_500sets_16players", args.output_dir.name)

    def test_alpha_is_not_silently_rounded(self) -> None:
        args = battle_blend_strength.make_argparser().parse_args(
            ["--alphas", "0.0,0.25,1.0"]
        )
        levels, alphas = battle_blend_strength.validate_args(args)
        specs = battle_blend_strength.build_player_specs(
            args,
            levels,
            alphas,
            policy_server_port=12345,
        )
        alpha_quarter = next(spec for spec in specs if spec.alpha == 0.25)

        self.assertEqual("alpha_0.25", alpha_quarter.name)
        alpha_index = alpha_quarter.command.index("--alpha")
        self.assertEqual("0.25", alpha_quarter.command[alpha_index + 1])

        close_values = [0.12345678901231, 0.12345678901232]
        self.assertNotEqual(
            battle_blend_strength.format_alpha(close_values[0]),
            battle_blend_strength.format_alpha(close_values[1]),
        )

    def test_policy_server_ready_enforces_the_requested_runtime(self) -> None:
        info = battle_blend_strength.parse_policy_server_ready(
            "READY 12345 tensorflow GPU 2.10.0",
            requested_backend="tensorflow",
            allow_policy_cpu=False,
        )
        self.assertEqual("2.10.0", info.backend_version)
        self.assertEqual("tensorflow/2.10.0/GPU", info.runtime)
        with self.assertRaisesRegex(ValueError, "without a GPU"):
            battle_blend_strength.parse_policy_server_ready(
                "READY 12345 tensorflow CPU 2.10.0",
                requested_backend="tensorflow",
                allow_policy_cpu=False,
            )
        cpu_info = battle_blend_strength.parse_policy_server_ready(
            "READY 12345 tensorflow CPU 2.10.0",
            requested_backend="tensorflow",
            allow_policy_cpu=True,
        )
        self.assertEqual("tensorflow/2.10.0/CPU", cpu_info.runtime)
        with self.assertRaisesRegex(ValueError, "unexpected backend"):
            battle_blend_strength.parse_policy_server_ready(
                "READY 12345 numpy CPU 1.24.3",
                requested_backend="tensorflow",
                allow_policy_cpu=True,
            )
        with self.assertRaisesRegex(ValueError, "invalid policy server"):
            battle_blend_strength.parse_policy_server_ready(
                "READY 12345 tensorflow GPU",
                requested_backend="tensorflow",
                allow_policy_cpu=False,
            )

    def test_manifest_records_tensorflow_distribution_versions(self) -> None:
        versions = {
            "numpy": "1.24.3",
            "psutil": "5.9.8",
            "tensorflow-gpu": "2.10.0",
            "tensorflow-intel": "2.13.0",
        }

        def distribution_version(name: str) -> str:
            if name not in versions:
                raise PackageNotFoundError(name)
            return versions[name]

        with patch(
            "battle_blend_strength.importlib.metadata.version",
            side_effect=distribution_version,
        ):
            runtime = battle_blend_strength.package_runtime_versions(
                "tensorflow"
            )

        self.assertEqual("2.10.0", runtime["packages"]["tensorflow-gpu"])
        self.assertEqual("2.13.0", runtime["packages"]["tensorflow-intel"])
        self.assertEqual(
            "not-installed",
            runtime["packages"]["tensorflow"],
        )
        self.assertEqual(
            "not-installed",
            runtime["packages"]["tf-nightly"],
        )

    def test_hint_cache_option_controls_the_shared_cache_only(self) -> None:
        args = battle_blend_strength.make_argparser().parse_args(
            ["--baseline-levels", "", "--alphas", "0.2,0.4"]
        )
        levels, alphas = battle_blend_strength.validate_args(args)
        cached = battle_blend_strength.build_player_specs(
            args,
            levels,
            alphas,
            policy_server_port=12345,
        )
        self.assertTrue(
            all("--hint-cache-db" in spec.command for spec in cached)
        )
        self.assertTrue(
            all("--cache-egaroucid" not in spec.command for spec in cached)
        )

        args.no_hint_cache = True
        uncached = battle_blend_strength.build_player_specs(
            args,
            levels,
            alphas,
            policy_server_port=12345,
        )
        self.assertTrue(
            all("--hint-cache-db" not in spec.command for spec in uncached)
        )

    def test_every_pair_gets_the_same_opening_for_each_set_index(self) -> None:
        tasks = make_match_set_tasks(
            player_count=4,
            openings=["first", "second", "third"],
            match_sets_per_pair=3,
            seed=57,
        )
        by_set = {}
        for task in tasks:
            by_set.setdefault(task.set_index, set()).add(task.opening)

        self.assertEqual(
            {0: {"first"}, 1: {"second"}, 2: {"third"}},
            by_set,
        )
        battle_blend_strength.validate_scheduled_openings(
            ["f5", "f5d6", "f5d6c3"],
            3,
        )
        with self.assertRaisesRegex(ValueError, "distinct XOT"):
            battle_blend_strength.validate_scheduled_openings(
                ["only-one"],
                2,
            )
        with self.assertRaisesRegex(ValueError, "invalid XOT opening"):
            battle_blend_strength.validate_scheduled_openings(["f"], 1)
        with self.assertRaisesRegex(ValueError, "invalid XOT opening"):
            battle_blend_strength.validate_scheduled_openings(["a1"], 1)

    def test_limited_schedule_has_exact_pair_targets(self) -> None:
        tasks = make_match_set_tasks(
            player_count=16,
            openings=[OPENING],
            match_sets_per_pair=1,
            seed=57,
        )[:64]
        matrix = target_match_sets_by_pair(tasks, 16)
        self.assertEqual(64, sum(
            matrix[player_idx][opponent_idx]
            for player_idx in range(16)
            for opponent_idx in range(player_idx + 1, 16)
        ))
        self.assertEqual(
            128,
            sum(sum(row) for row in matrix),
        )

    def test_pair_queue_respects_capacity_and_requeues_atomically(self) -> None:
        tasks = make_match_set_tasks(4, [OPENING], 2, 57)
        pending = PendingTaskQueue(tasks, player_count=4)
        active = [0, 0, 0, 0]
        capacities = [1, 1, 1, 1]
        first = pending.pop_schedulable(active, capacities)
        self.assertIsNotNone(first)
        assert first is not None
        active[first.p0_idx] += 1
        active[first.p1_idx] += 1
        second = pending.pop_schedulable(active, capacities)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertTrue(
            {first.p0_idx, first.p1_idx}.isdisjoint(
                {second.p0_idx, second.p1_idx}
            )
        )
        remaining_before_retry = len(pending)
        pending.push_front(first)
        self.assertEqual(remaining_before_retry + 1, len(pending))

    def test_pair_queue_finishes_each_opening_wave_before_the_next(self) -> None:
        tasks = make_match_set_tasks(
            player_count=4,
            openings=["first", "second", "third"],
            match_sets_per_pair=3,
            seed=57,
        )
        pending = PendingTaskQueue(tasks, player_count=4)
        popped = []
        while pending:
            task = pending.pop_schedulable(
                active_counts=[0, 0, 0, 0],
                capacities=[3, 3, 3, 3],
                duration_weights=[1000.0, 1.0, 1.0, 1.0],
            )
            self.assertIsNotNone(task)
            assert task is not None
            popped.append(task)

        pair_count = 4 * 3 // 2
        self.assertEqual(
            [0] * pair_count + [1] * pair_count + [2] * pair_count,
            [task.set_index for task in popped],
        )
        for set_index in range(3):
            wave = popped[
                set_index * pair_count:(set_index + 1) * pair_count
            ]
            self.assertEqual(
                pair_count,
                len({(task.p0_idx, task.p1_idx) for task in wave}),
            )

    def test_pair_queue_can_use_the_next_wave_when_older_work_is_blocked(
        self,
    ) -> None:
        tasks = make_match_set_tasks(
            player_count=4,
            openings=["first", "second", "third"],
            match_sets_per_pair=3,
            seed=57,
        )
        pending = PendingTaskQueue(tasks, player_count=4)
        first = pending.pop_schedulable(
            active_counts=[0, 0, 0, 0],
            capacities=[1, 1, 1, 1],
        )
        self.assertIsNotNone(first)
        assert first is not None
        active = [1, 1, 1, 1]
        active[first.p0_idx] = 0
        active[first.p1_idx] = 0

        next_task = pending.pop_schedulable(
            active_counts=active,
            capacities=[1, 1, 1, 1],
        )
        self.assertIsNotNone(next_task)
        assert next_task is not None
        self.assertEqual(
            (first.p0_idx, first.p1_idx, 1),
            (next_task.p0_idx, next_task.p1_idx, next_task.set_index),
        )
        self.assertIsNone(
            pending.pop_schedulable(
                active_counts=active,
                capacities=[1, 1, 1, 1],
            )
        )

    def test_duration_weights_do_not_compare_warmup_with_one_second(self) -> None:
        self.assertEqual(
            [1.0, 1.0, 1.0],
            normalized_duration_weights(
                [99.0, 1.0, 1.0],
                [0, 0, 0],
            ),
        )
        self.assertEqual(
            [10.0, 12.0, 24.0, 12.0],
            normalized_duration_weights(
                [10.0, 12.0, 1000.0, float("nan")],
                [1, 1, 1, 0],
            ),
        )
        with self.assertRaisesRegex(ValueError, "equal length"):
            normalized_duration_weights([1.0], [0, 0])

    def test_color_games_are_combined_by_color_not_completion_order(self) -> None:
        task = make_match_set_tasks(2, [OPENING], 1, 57)[0]
        result = combine_color_games(
            task,
            [
                game_result(False, -3),
                game_result(True, 5),
            ],
        )

        self.assertEqual(1.0, result.p0_disc_diff)
        self.assertEqual(5, result.p0_black_disc_diff)
        self.assertEqual(-3, result.p0_white_disc_diff)
        self.assertEqual((True, False), tuple(
            game.p0_is_black for game in result.color_games
        ))

    def test_paired_and_actual_game_records_are_distinct(self) -> None:
        task = make_match_set_tasks(2, [OPENING], 1, 57)[0]
        result = combine_color_games(
            task,
            [game_result(True, 5), game_result(False, -3)],
        )
        stats = TournamentStats(2)
        stats.record(result)

        self.assertEqual([1, 0, 0], stats.results[0][1])
        self.assertEqual([1, 0, 1], stats.actual_results[0][1])
        self.assertEqual(1, stats.n_played[0][1])
        self.assertEqual(2, stats.actual_n_played[0][1])

    def test_confidence_interval_uses_match_sets_as_observations(self) -> None:
        self.assertAlmostEqual(2.009575, student_t_critical_975(49), places=5)
        self.assertAlmostEqual(
            0.04393,
            conservative_score_half_width(500),
            places=5,
        )
        interval = score_confidence_interval(25, 0, 25)
        self.assertLess(interval.low, 0.5)
        self.assertGreater(interval.high, 0.5)
        self.assertAlmostEqual(0.14354, interval.half_width, places=4)
        all_wins = score_confidence_interval(50, 0, 0)
        self.assertLess(all_wins.low, 1.0)
        self.assertEqual(1.0, all_wins.high)

    def test_participant_summary_is_descriptive_and_elo_is_finite(self) -> None:
        specs = self.two_player_specs()
        stats = TournamentStats(2)
        task = make_match_set_tasks(2, [OPENING], 1, 57)[0]
        stats.record(
            combine_color_games(
                task,
                [game_result(True, 4), game_result(False, 2)],
            )
        )

        ratings = estimate_elos(specs, stats)
        self.assertTrue(all(math.isfinite(value) for value in ratings.values()))
        self.assertGreater(ratings["player_0"], ratings["player_1"])

        target_matrix = [[0, 1], [1, 0]]
        summary = build_summary_rows(
            specs,
            stats,
            ratings,
            target_matrix,
        )
        self.assertIn("paired_set_score_descriptive", summary[0])
        self.assertIn("paired_set_elo_descriptive", summary[0])
        self.assertNotIn("paired_set_score", summary[0])
        self.assertFalse(
            any("ci95" in field for field in summary[0])
        )

        pair_rows = build_pair_rows(specs, stats, target_matrix)
        self.assertIn("paired_set_score_ci95_low", pair_rows[0])
        self.assertIn("paired_set_score_ci95_high", pair_rows[0])
        report = build_text_report(specs, stats, target_matrix, ratings)
        self.assertIn("Participant-wide descriptive summary", report)
        self.assertIn("score(desc)", report)
        self.assertNotIn("\tCI95\t", report)

    def test_interim_elo_uses_a_connected_sparse_pair_graph(self) -> None:
        specs = [
            PlayerSpec(f"player_{idx}", (f"player_{idx}",), 2, False)
            for idx in range(4)
        ]
        stats = TournamentStats(4)
        for player_idx, opponent_idx in ((0, 1), (1, 2), (2, 3)):
            stats.results[player_idx][opponent_idx] = [1, 0, 0]
            stats.results[opponent_idx][player_idx] = [0, 0, 1]

        with patch(
            "strength_reporting.fit_elo_from_winrates",
            wraps=strength_reporting.fit_elo_from_winrates,
        ) as fitter:
            ratings = estimate_elos(specs, stats)

        self.assertEqual(
            {spec.name for spec in specs},
            set(ratings),
        )
        self.assertTrue(all(math.isfinite(value) for value in ratings.values()))
        self.assertGreater(ratings["player_0"], ratings["player_1"])
        self.assertGreater(ratings["player_1"], ratings["player_2"])
        self.assertGreater(ratings["player_2"], ratings["player_3"])
        games = fitter.call_args.kwargs["games"]
        self.assertEqual(2.0, games[0, 1])
        self.assertEqual(0.0, games[0, 2])
        self.assertEqual(0.0, games[0, 3])
        self.assertEqual(0.0, games[1, 3])

    def test_interim_elo_is_withheld_for_a_disconnected_pair_graph(self) -> None:
        specs = [
            PlayerSpec(f"player_{idx}", (f"player_{idx}",), 2, False)
            for idx in range(4)
        ]
        stats = TournamentStats(4)
        for player_idx, opponent_idx in ((0, 1), (2, 3)):
            stats.results[player_idx][opponent_idx] = [1, 0, 0]
            stats.results[opponent_idx][player_idx] = [0, 0, 1]

        ratings = estimate_elos(specs, stats)

        self.assertEqual({}, ratings)
        report = build_text_report(
            specs,
            stats,
            [
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
            ],
            ratings,
        )
        self.assertIn(
            "played-pair graph is not connected (2/6 pairs played)",
            report,
        )

    def test_report_files_have_unambiguous_fields_and_no_double_cr(self) -> None:
        specs = self.two_player_specs()
        stats = TournamentStats(2)
        task = make_match_set_tasks(2, [OPENING], 1, 57)[0]
        stats.record(
            combine_color_games(
                task,
                [game_result(True, 4), game_result(False, -2)],
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            legacy_paths = [
                output_dir / "strength_win_rate_matrix.tsv",
                output_dir / "strength_disc_diff_matrix.tsv",
            ]
            for path in legacy_paths:
                path.write_text("stale\n", encoding="utf-8")
            write_outputs(
                specs,
                stats,
                output_dir,
                completed_match_sets=1,
                total_match_sets=1,
                target_match_sets_by_pair=[[0, 1], [1, 0]],
                experiment_id="test",
            )

            delimited_paths = [
                output_dir / "strength_summary.csv",
                output_dir / "strength_pair_results.csv",
                output_dir / "strength_paired_set_score_matrix.tsv",
                output_dir / "strength_paired_disc_diff_matrix.tsv",
                output_dir / "strength_progress_matrix.tsv",
            ]
            for path in delimited_paths:
                raw = path.read_bytes()
                self.assertNotIn(b"\r\r\n", raw, path.name)
            self.assertTrue(all(not path.exists() for path in legacy_paths))

            result = json.loads(
                (output_dir / "strength_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(3, result["schema_version"])
            self.assertIn(
                "paired_set_score_descriptive",
                result["summary"][0],
            )
            self.assertFalse(
                any("ci95" in field for field in result["summary"][0])
            )
            self.assertIn(
                "paired_set_score_ci95_half_width",
                result["pair_results"][0],
            )
            self.assertIn(
                "shared_opening_schedule",
                result["scoring"],
            )

    def test_unplayed_pairs_are_missing_and_use_planned_pair_targets(self) -> None:
        specs = [
            PlayerSpec(f"player_{idx}", (f"player_{idx}",), 2, False)
            for idx in range(3)
        ]
        stats = TournamentStats(3)
        task = make_match_set_tasks(2, [OPENING], 1, 57)[0]
        stats.record(
            combine_color_games(
                task,
                [game_result(True, 4), game_result(False, -2)],
            )
        )
        target_matrix = [
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ]
        ratings = estimate_elos(specs, stats)

        unplayed_pair = next(
            row
            for row in build_pair_rows(specs, stats, target_matrix)
            if row["player"] == "player_1"
            and row["opponent"] == "player_2"
        )
        self.assertEqual(2, unplayed_pair["planned_match_sets"])
        self.assertIsNone(unplayed_pair["paired_set_score"])
        self.assertIsNone(unplayed_pair["paired_set_score_ci95_low"])
        self.assertIsNone(
            unplayed_pair["avg_paired_disc_diff_descriptive"]
        )

        summary = build_summary_rows(
            specs,
            stats,
            ratings,
            target_matrix,
        )
        self.assertEqual(2, summary[2]["planned_match_sets"])
        self.assertIsNone(summary[2]["paired_set_score_descriptive"])
        self.assertIsNone(
            summary[2]["avg_paired_disc_diff_descriptive"]
        )
        report = build_text_report(
            specs,
            stats,
            target_matrix,
            ratings,
        )
        self.assertIn("0/2", report)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_outputs(
                specs,
                stats,
                output_dir,
                completed_match_sets=1,
                total_match_sets=3,
                target_match_sets_by_pair=target_matrix,
                experiment_id="uneven-plan",
            )
            result = json.loads(
                (output_dir / "strength_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                target_matrix,
                result["planned_match_sets_by_pair"],
            )
            result_pair = next(
                row
                for row in result["pair_results"]
                if row["player"] == "player_1"
                and row["opponent"] == "player_2"
            )
            self.assertIsNone(result_pair["paired_set_score"])
            score_matrix = (
                output_dir / "strength_paired_set_score_matrix.tsv"
            ).read_text(encoding="utf-8")
            self.assertIn("player_2\t-\t-\t-\t-\t", score_matrix)

    def test_result_store_round_trip_and_strict_task_identity(self) -> None:
        task = make_match_set_tasks(2, [OPENING], 1, 57)[0]
        result = combine_color_games(
            task,
            [game_result(True, 4), game_result(False, -4)],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResultStore(Path(temp_dir))
            store.open()
            store.append_result(task, result)
            store.close()

            completed, loaded = store.load({task.task_id: task})
            self.assertEqual({task.task_id}, completed)
            self.assertEqual(result, loaded[0])

            changed = replace(task, opening="different")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                store.load({task.task_id: changed})

            with store.results_path.open("ab") as output:
                output.write(b'{"partial":')
            completed, loaded = store.load({task.task_id: task})
            self.assertEqual({task.task_id}, completed)
            self.assertEqual([result], loaded)
            self.assertTrue(
                store.results_path.read_bytes().endswith(b"\n")
            )

            data = store.results_path.read_bytes()
            store.results_path.write_bytes(data[:-1])
            completed, loaded = store.load({task.task_id: task})
            self.assertEqual({task.task_id}, completed)
            self.assertEqual([result], loaded)
            self.assertTrue(
                store.results_path.read_bytes().endswith(b"\n")
            )

    def test_gtp_command_timeout_kills_the_failed_process(self) -> None:
        manager = ProcessManager(minimum_available_memory_mib=1.0)
        process = GtpProcess(
            manager,
            [
                sys.executable,
                "-u",
                "-c",
                (
                    "import sys,time\n"
                    "for line in sys.stdin:\n"
                    "    time.sleep(10)\n"
                ),
            ],
            timeout_sec=0.1,
        )
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                process.request("genmove black")
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertFalse(process.usable)
        finally:
            process.close()
            manager.close_all()

    def test_process_manager_closes_descendant_process_tree(self) -> None:
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil is unavailable")
        manager = ProcessManager(minimum_available_memory_mib=1.0)
        parent = manager.spawn(
            [
                sys.executable,
                "-u",
                "-c",
                (
                    "import subprocess,sys,time\n"
                    "time.sleep(0.25)\n"
                    "child=subprocess.Popen("
                    "[sys.executable,'-c','import time;time.sleep(60)'])\n"
                    "print(child.pid,flush=True)\n"
                ),
            ]
        )
        try:
            self.assertIsNotNone(parent.stdout)
            child_pid = int(parent.stdout.readline().strip())
            parent.wait(timeout=5.0)
            manager.terminate(parent, graceful=False)
            deadline = time.monotonic() + 5.0
            while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(psutil.pid_exists(child_pid))
        finally:
            manager.close_all()

    def test_output_lock_rejects_a_second_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with battle_blend_strength.OutputRunLock(output_dir):
                with self.assertRaisesRegex(RuntimeError, "another tournament"):
                    with battle_blend_strength.OutputRunLock(output_dir):
                        pass

    def test_performance_samples_accumulate_across_resume_runs(self) -> None:
        def sample(run_id: str, cpu: float) -> dict:
            return {
                "run_id": run_id,
                "sampled_at_unix_sec": 1.0,
                "elapsed_sec": 1.0,
                "cpu_percent": cpu,
                "system_memory_used_mib": 100.0,
                "available_memory_mib": 200.0,
                "system_memory_percent": 33.0,
                "gpu_percent": 4.0,
                "gpu_memory_used_mib": 5.0,
                "child_processes": 6,
                "configured_minimum_available_memory_mib": 10.0,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            manager = ProcessManager(minimum_available_memory_mib=10.0)
            for run_id, cpu in (("run_a", 20.0), ("run_b", 40.0)):
                monitor = PerformanceMonitor(
                    output_dir,
                    interval_sec=1.0,
                    manager=manager,
                    run_id=run_id,
                )
                monitor.thread = Mock()
                monitor.samples = [sample(run_id, cpu)]
                monitor.stop()
            with (
                output_dir / "performance_samples.csv"
            ).open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(["run_a", "run_b"], [
                row["run_id"] for row in rows
            ])
            summary = json.loads(
                (output_dir / "performance_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(2, summary["run_count"])
            self.assertEqual(30.0, summary["average_cpu_percent"])

    def test_completed_resume_regenerates_final_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            openings_path = root / "openings.txt"
            openings_path.write_text(OPENING + "\n", encoding="utf-8")
            output_dir = root / "output"
            common = [
                "--baseline-levels",
                "1,3",
                "--alphas",
                "",
                "--match-sets-per-pair",
                "1",
                "--openings",
                str(openings_path),
                "--output-dir",
                str(output_dir),
            ]
            dry_args = battle_blend_strength.make_argparser().parse_args(
                common + ["--dry-run"]
            )
            with redirect_stdout(io.StringIO()):
                battle_blend_strength.run_tournament(dry_args)

            task = make_match_set_tasks(2, [OPENING], 1, 57)[0]
            result = combine_color_games(
                task,
                [game_result(True, 4), game_result(False, -4)],
            )
            store = ResultStore(output_dir)
            store.open()
            store.append_result(task, result)
            store.close()

            resume_args = battle_blend_strength.make_argparser().parse_args(
                common + ["--resume"]
            )
            with redirect_stdout(io.StringIO()):
                battle_blend_strength.run_tournament(resume_args)
            progress = json.loads(
                (output_dir / "strength_progress.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("already_complete", progress["stop_reason"])
            self.assertEqual(0, progress["remaining_match_sets"])
            result_json = json.loads(
                (output_dir / "strength_results.json").read_text(
                    encoding="utf-8"
                )
            )
            blend_commands = [
                player["command"]
                for player in result_json["players"]
                if "--policy-server-port" in player["command"]
            ]
            self.assertFalse(blend_commands)


if __name__ == "__main__":
    unittest.main()
