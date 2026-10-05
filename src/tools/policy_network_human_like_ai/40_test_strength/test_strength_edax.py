#!/usr/bin/env python3
"""Edax configuration, persistence, and GTP interoperability checks."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import battle_blend_strength as battle
from blend_policy_with_egaroucid import (
    BoardState, coord_to_policy, policy_to_coord, side_to_gtp_color,
)
from strength_engine import EdaxXBoardProcess, GameRunner, ProcessManager
from strength_tournament import (
    MatchSetTask, PlayerSpec, ensure_manifest, make_manifest,
)


OPENING = "f5d6c3d3c4f4c5b3"


class EdaxConfigurationTest(unittest.TestCase):
    def test_edax_defaults_use_versions_directory_and_can_be_disabled(self):
        args = battle.make_argparser().parse_args([])
        levels, alphas = battle.validate_args(args)
        specs = battle.build_player_specs(args, levels, alphas, 0)
        self.assertEqual(17, len(specs))
        self.assertEqual(
            ["edax_l1", "edax_l5", "edax_l9", "edax_l13", "edax_l17", "edax_l21"],
            [spec.name for spec in specs if spec.name.startswith("edax_")],
        )
        self.assertEqual(
            battle.BIN_DIR / "versions" / "edax_4_6" / "wEdax-x86-64.exe",
            args.edax_exe,
        )
        self.assertEqual(args.edax_exe.parent / "data" / "eval.dat",
                         battle.edax_eval_path(args))
        args.edax_levels = ""
        self.assertEqual(11, len(battle.build_player_specs(args, levels, alphas, 0)))

    def test_edax_participates_with_native_and_blended_players(self):
        args = battle.make_argparser().parse_args([
            "--baseline-levels", "1", "--alphas", "0.0,0.2",
            "--edax-levels", "1,3", "--engine-threads", "2",
            "--edax-processes-per-player", "4",
            "--edax-exe", "some directory/edax.exe",
            "--edax-eval-file", "other directory/eval.dat",
        ])
        levels, alphas = battle.validate_args(args)
        specs = battle.build_player_specs(args, levels, alphas, 1234)
        self.assertEqual(
            ["egaroucid_l1", "alpha_0.0", "alpha_0.2", "edax_l1", "edax_l3"],
            [spec.name for spec in specs],
        )
        edax = specs[-1]
        self.assertFalse(edax.setboard_before_genmove)
        self.assertEqual(2, edax.concurrent_match_set_capacity)
        self.assertEqual(str(args.edax_exe.resolve()), edax.command[0])
        for flag, value in (("-l", "3"), ("-n", "2"), ("-h", "22"),
                            ("-book-usage", "off"), ("-ponder", "off"),
                            ("-eval-file", str(args.edax_eval_file.resolve()))):
            self.assertEqual(value, edax.command[edax.command.index(flag) + 1])
        self.assertNotIn("-t", edax.command)

    def test_invalid_edax_settings_are_rejected(self):
        for option, value in (
            ("--edax-levels", "-1"), ("--edax-levels", "61"),
            ("--edax-levels", "1,1"), ("--edax-levels", "bad"),
            ("--edax-processes-per-player", "1"),
            ("--edax-processes-per-player", "3"),
            ("--estimated-edax-memory-mib", "0"),
        ):
            with self.subTest(option=option, value=value):
                args = battle.make_argparser().parse_args([option, value])
                with self.assertRaises(ValueError):
                    battle.validate_args(args)

    def test_edax_counts_towards_minimum_participants(self):
        args = battle.make_argparser().parse_args([
            "--baseline-levels", "", "--alphas", "", "--edax-levels", "0,60",
        ])
        self.assertEqual(([], []), battle.validate_args(args))
        args.edax_levels = "1"
        with self.assertRaisesRegex(ValueError, "two total participants"):
            battle.validate_args(args)

    @staticmethod
    def fixture_args(root):
        exe = root / "edax.exe"
        exe.write_bytes(b"Edax executable fixture")
        (root / "data").mkdir()
        (root / "data" / "eval.dat").write_bytes(b"Edax evaluation fixture")
        openings = root / "openings.txt"
        openings.write_text(OPENING + "\n", encoding="utf-8")
        return battle.make_argparser().parse_args([
            "--baseline-levels", "", "--alphas", "", "--edax-levels", "1,3,5",
            "--edax-exe", str(exe), "--egaroucid-exe", str(root / "missing.exe"),
            "--openings", str(openings), "--match-sets-per-pair", "1",
            "--output-dir", str(root / "output"), "--dry-run",
        ])

    def test_edax_files_required_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture_args(Path(directory))
            battle.validate_input_files(args, [])
            battle.edax_eval_path(args).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Edax evaluation data"):
                battle.validate_input_files(args, [])
            args.edax_levels = ""
            args.baseline_levels = "1,3"
            args.egaroucid_exe = args.edax_exe
            battle.validate_input_files(args, [])
            args.edax_levels = "1"
            args.edax_exe = Path(directory) / "missing_edax.exe"
            with self.assertRaisesRegex(FileNotFoundError, "Edax executable"):
                battle.validate_input_files(args, [])

    def test_edax_dry_run_includes_schedule_memory_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture_args(Path(directory))
            with patch.object(battle, "available_memory_mib", return_value=100_000), \
                    patch.object(ProcessManager, "spawn") as spawn, \
                    redirect_stdout(io.StringIO()):
                battle.run_tournament(args)
            spawn.assert_not_called()
            dry = json.loads((args.output_dir / "strength_dry_run.json").read_text())
            self.assertEqual(3, dry["total_match_sets"])
            self.assertEqual(6, dry["total_actual_games"])
            self.assertEqual(6, dry["estimated_max_player_processes"])
            self.assertEqual(6, dry["estimated_edax_processes"])
            self.assertEqual(0, dry["estimated_heavy_egaroucid_processes"])
            self.assertEqual(6 * 256, dry["estimated_max_tournament_memory_mib"])
            manifest_path = args.output_dir / "strength_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            config = manifest["configuration"]
            self.assertEqual([1, 3, 5], config["participants"]["edax_levels"])
            self.assertNotIn("egaroucid_executable", config["inputs"])
            self.assertEqual(battle.file_identity(args.edax_exe),
                             config["inputs"]["edax_executable"])
            self.assertEqual(battle.file_identity(battle.edax_eval_path(args)),
                             config["inputs"]["edax_evaluation_data"])
            for path in (args.edax_exe, battle.edax_eval_path(args)):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                changed = battle.build_manifest_configuration(args, [], [], [OPENING], [])
                # Keep the original schedule to isolate the changed input.
                changed["schedule"] = config["schedule"]
                with self.assertRaisesRegex(ValueError, "different experiment"):
                    ensure_manifest(manifest_path, make_manifest(changed), True, True)
                path.write_bytes(original)


class StrictPassEngine:
    """Small legal-move engine that rejects terminal or out-of-turn passes."""

    def __init__(self):
        self.state = BoardState.initial()
        self.pass_count = 0

    def request(self, command):
        if command == "clear_board":
            self.state = BoardState.initial()
            return "="
        action, color, *moves = command.split()
        assert color == side_to_gtp_color(self.state.side), command
        legal = self.state.legal_policies(self.state.side)
        if moves == ["pass"]:
            assert not legal, "pass with legal moves"
            assert self.state.legal_policies(self.state.side ^ 1), "terminal pass"
            self.state.side ^= 1
            self.pass_count += 1
            return "="
        policy = min(legal) if action == "genmove" else coord_to_policy(moves[0])
        assert policy in legal, command
        self.state.apply_move(self.state.side, policy)
        return "= " + policy_to_coord(policy) if action == "genmove" else "="


class EdaxGameTest(unittest.TestCase):
    def test_xboard_adapter_changes_only_cores_and_restores_level_after_new(self):
        engine = object.__new__(EdaxXBoardProcess)
        engine.level = 13
        engine._exchange = Mock(return_value=[])
        engine.set_search_threads(4)
        engine._exchange.assert_called_once_with(["cores 4"])
        engine.request("clear_board")
        self.assertEqual(["new", "force", "easy", "nopost", "sd 13"], engine._exchange.call_args.args[0])
        engine._exchange.side_effect = [["move P@c4", "1-0 {White wins 34-30}"], []]
        self.assertEqual("= c4", engine.request("genmove b"))
        self.assertEqual(["force"], engine._exchange.call_args.args[0])

    @unittest.skipUnless(
        os.name == "nt" and battle.DEFAULT_EDAX_EXE.is_file()
        and battle.default_egaroucid_exe().is_file(), "local Windows engines required")
    def test_installed_edax_resizes_during_games_without_restart_or_board_reset(self):
        import psutil
        args = battle.make_argparser().parse_args([
            "--baseline-levels", "1", "--alphas", "", "--edax-levels", "1"])
        specs = battle.build_player_specs(args, [1], [], 0)
        manager = ProcessManager(8192)
        runner = GameRunner(specs, manager, 15, game_boundary_threads=True)
        changes, calls = [], [0]
        def at_boundary(task, color, player, change):
            calls[0] += 1
            if calls[0] not in (1, 8, 16):
                return
            engine = next(iter(runner.pools[player]._leased_engines))
            pid = engine.process.pid
            board_before = engine._exchange(["debug"])
            threads = 4 if calls[0] == 8 else 1
            change(threads)
            self.assertEqual(board_before, engine._exchange(["debug"]))
            self.assertEqual(pid, engine.process.pid)
            self.assertEqual({str(threads): 1}, runner.pools[player].search_thread_status()["actual_search_threads"])
            # Windows process snapshots can still include exiting workers
            # immediately after the protocol ACK. Require convergence, with
            # a bounded wait; a leaked or unchanged worker fails this check.
            process = psutil.Process(pid)
            deadline = time.monotonic() + 2
            expected = 3 if threads == 1 else 6
            observed = process.num_threads()
            while observed != expected and time.monotonic() < deadline:
                time.sleep(.01)
                observed = process.num_threads()
            self.assertEqual(expected, observed)
            changes.append((threads, observed))
        runner.thread_budget = Mock(resize_at_move_boundary=at_boundary)
        try:
            for color in (True, False):
                calls[0] = 0
                # The other color's prior idle pool remains, so use active status below.
                if not color:
                    runner.pools[1].close_idle()
                result = runner.play_single_game(MatchSetTask(0, 0, 1, 0, OPENING), color,
                                                 search_threads={0: 1, 1: 1})
                final = BoardState.from_transcript(result.transcript)
                self.assertFalse(final.legal_policies(0))
                self.assertFalse(final.legal_policies(1))
                self.assertTrue(result.transcript.startswith(OPENING))
            self.assertEqual([1, 4, 1] * 2, [threads for threads, _ in changes])
            for start in (0, 3):
                self.assertEqual(3, changes[start + 1][1] - changes[start][1])
                self.assertEqual(changes[start][1], changes[start + 2][1])
        finally:
            manager.close_all()

    def test_required_passes_are_sent_and_terminal_passes_are_not(self):
        specs = [PlayerSpec(f"p{i}", (), 2, False) for i in range(2)]
        engines = [StrictPassEngine(), StrictPassEngine()]
        runner = GameRunner(specs, Mock(), 1)
        runner.pools = [Mock(acquire=Mock(return_value=engine)) for engine in engines]
        for swapped in (True, False):
            result = runner.play_single_game(MatchSetTask(0, 0, 1, 0, OPENING), swapped)
            final = BoardState.from_transcript(result.transcript)
            self.assertFalse(final.legal_policies(0))
            self.assertFalse(final.legal_policies(1))
            for engine in engines:
                self.assertEqual(final.black, engine.state.black)
                self.assertEqual(final.white, engine.state.white)
        self.assertGreater(engines[0].pass_count, 0)

    @unittest.skipUnless(
        os.name == "nt" and battle.DEFAULT_EDAX_EXE.is_file()
        and battle.default_egaroucid_exe().is_file(),
        "local Windows Edax and Egaroucid executables are required",
    )
    def test_installed_edax_plays_both_colors_and_reuses_processes(self):
        args = battle.make_argparser().parse_args([
            "--baseline-levels", "1", "--alphas", "", "--edax-levels", "1",
        ])
        levels, alphas = battle.validate_args(args)
        specs = battle.build_player_specs(args, levels, alphas, 0)
        manager = ProcessManager(1)
        runner = GameRunner(specs, manager, 15)
        try:
            engine = runner.pools[1].acquire(0)
            try:
                self.assertEqual("= Edax", engine.request("name"))
                self.assertEqual("= 4.6", engine.request("version"))
            finally:
                runner.pools[1].release(0, engine)
            for swapped in (True, False, True):
                result = runner.play_single_game(
                    MatchSetTask(0, 0, 1, 0, OPENING), swapped,
                )
                final = BoardState.from_transcript(result.transcript)
                self.assertTrue(result.transcript.startswith(OPENING))
                self.assertFalse(final.legal_policies(0))
                self.assertFalse(final.legal_policies(1))
        finally:
            manager.close_all()


if __name__ == "__main__":
    unittest.main()
