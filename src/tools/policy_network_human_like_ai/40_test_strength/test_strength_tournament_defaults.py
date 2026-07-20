#!/usr/bin/env python3

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import battle_blend_strength
import run_strength_full


class StrengthTournamentDefaultsTest(unittest.TestCase):
    def test_battle_defaults_define_requested_tournament(self) -> None:
        args = battle_blend_strength.make_argparser().parse_args([])
        baseline_levels = battle_blend_strength.parse_int_list(
            args.baseline_levels
        )
        blend_params = battle_blend_strength.parse_float_list(args.blend_params)

        self.assertTrue(args.no_random_player)
        self.assertEqual(100, args.match_sets_per_pair)
        self.assertFalse(args.measure_move_stone_loss)
        self.assertEqual(10, len(baseline_levels))
        self.assertEqual(6, len(blend_params))

        participant_count = len(baseline_levels) + len(blend_params)
        pair_count = participant_count * (participant_count - 1) // 2
        tasks = battle_blend_strength.make_tasks(
            list(range(participant_count)),
            ["f5d6c3d3c4f4c5b3"],
            args.match_sets_per_pair,
            args.seed,
            args.same_openings_for_all_pairs,
        )

        self.assertEqual(16, participant_count)
        self.assertEqual(120, pair_count)
        self.assertEqual(12_000, len(tasks))
        self.assertEqual(24_000, sum(task.actual_games for task in tasks))
        self.assertTrue(all(task.actual_games == 2 for task in tasks))
        battle_blend_strength.validate_input_files(args)

        args.baseline_processes_per_player = args.processes_per_player
        args.blend_processes_per_player = args.processes_per_player
        players = battle_blend_strength.build_players(args)
        self.assertEqual(16, len(players))
        self.assertNotIn("random_legal", {player.name for player in players})
        self.assertTrue(
            all(
                not player.measures_move_stone_loss
                for player in players
                if player.alpha is not None
            )
        )
        self.assertTrue(
            all(
                "--measure-move-stone-loss" not in player.command
                for player in players
            )
        )

    def test_legacy_match_count_alias_is_preserved(self) -> None:
        args = battle_blend_strength.make_argparser().parse_args(
            ["--games-per-pair", "7"]
        )

        self.assertEqual(7, args.match_sets_per_pair)

    def test_random_player_and_stone_loss_are_explicit_opt_ins(self) -> None:
        args = battle_blend_strength.make_argparser().parse_args(
            ["--include-random-player", "--measure-move-stone-loss"]
        )

        self.assertFalse(args.no_random_player)
        self.assertTrue(args.measure_move_stone_loss)
        args.baseline_processes_per_player = args.processes_per_player
        args.blend_processes_per_player = args.processes_per_player
        players = battle_blend_strength.build_players(args)
        self.assertEqual(17, len(players))
        self.assertIn("random_legal", {player.name for player in players})
        alpha_players = [
            player for player in players if player.alpha is not None
        ]
        self.assertTrue(
            all(player.measures_move_stone_loss for player in alpha_players)
        )
        for player in alpha_players:
            expected_count = 0 if player.name == "alpha_0.0" else 1
            self.assertEqual(
                expected_count,
                player.command.count("--measure-move-stone-loss"),
            )

    def test_wrapper_forwards_production_defaults(self) -> None:
        args = run_strength_full.make_argparser().parse_args([])
        args.baseline_processes_per_player = args.processes_per_player
        args.blend_processes_per_player = args.processes_per_player
        command = run_strength_full.make_command(args)

        self.assertTrue(args.no_random_player)
        self.assertEqual(100, args.match_sets_per_pair)
        self.assertEqual("xot_100sets_16players", args.output_dir.name)
        self.assertEqual("tensorflow", args.policy_backend)
        self.assertIn("--no-random-player", command)
        self.assertNotIn("--include-random-player", command)
        self.assertNotIn("--measure-move-stone-loss", command)
        match_count_index = command.index("--match-sets-per-pair")
        self.assertEqual("100", command[match_count_index + 1])

    def test_wrapper_displays_progress_and_writes_the_same_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "run.log"
            captured = io.StringIO()
            with redirect_stdout(captured):
                returncode = run_strength_full.run_command(
                    [sys.executable, "-c", "print('progress line')"],
                    log_path,
                )
            log_text = log_path.read_text(encoding="utf-8")

        self.assertEqual(0, returncode)
        self.assertIn("progress line", captured.getvalue())
        self.assertIn("progress line", log_text)


if __name__ == "__main__":
    unittest.main()
