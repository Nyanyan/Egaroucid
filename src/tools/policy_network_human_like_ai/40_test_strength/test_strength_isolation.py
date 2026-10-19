"""Independent openings, per-game TT lifetime, and uncached hint regression tests."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

import battle_blend_strength as battle
from blend_gtp_engine import make_argparser as wrapper_parser
from blend_policy_with_egaroucid import BlendedPolicy, BoardState, coord_to_policy, parse_side, policy_to_coord
from extend_strength_tournament import ModelExtension
from strength_engine import GameRunner
from strength_elo_confidence import (analyze_clustered_elos, bootstrap_independent_pair_counts,
                                     load_clustered_elo_data, write_clustered_elo_outputs)
from strength_tournament import MatchSetTask, PlayerSpec, make_manifest, make_match_set_tasks
from test_strength_elo_confidence import StrengthEloConfidenceTest


class OpeningSamplingTest(unittest.TestCase):
    def test_independent_unique_reproducible_and_prefix_stable(self):
        population = [f"opening_{i}" for i in range(1000)]
        def schedules(players, count, seed=57):
            result = {}
            for task in make_match_set_tasks(players, population, count, seed):
                result.setdefault((task.p0_idx, task.p1_idx), []).append(task.opening)
            return result
        original = schedules(6, 50)
        self.assertEqual(original, schedules(6, 50))
        self.assertNotEqual(original, schedules(6, 50, 58))
        self.assertNotEqual(original[0, 1], original[2, 3])
        self.assertGreater(len(set(sum(original.values(), []))), 50)
        for pair, sequence in original.items():
            self.assertEqual(50, len(set(sequence)))
            self.assertEqual(sequence, schedules(8, 50)[pair])
            self.assertEqual(sequence[:10], schedules(6, 10)[pair])

    def test_duplicates_do_not_increase_available_population(self):
        with self.assertRaisesRegex(ValueError, "distinct XOT"):
            make_match_set_tasks(2, ["one", "one"], 2, 57)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "openings.txt"
            path.write_text("f5\nF5\nf5d6\n")
            self.assertEqual({"f5", "f5d6"}, set(battle.read_openings(path, 57)))

    def test_old_reference_is_rejected_before_any_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {"schedule": {"same_opening_sequence_for_every_pair": True}, "engine": {"hint_cache": True}}
            (path / "strength_manifest.json").write_text(json.dumps(make_manifest(config)))
            with self.assertRaisesRegex(ValueError, "old tournament protocol"):
                ModelExtension(path, path / "unused.h5", "new_model")


class FakeEngine:
    def __init__(self, manager, command, timeout):
        self.usable = True
        self.state = BoardState.initial()
        self.tt = []
        self.events = []
        self.reject_clear = False

    def close(self):
        self.events.append("close")
        self.tt.clear()
        self.usable = False

    def request(self, command):
        self.events.append(command.split()[0])
        action, *args = command.split()
        if action == "clear_cache":
            if self.reject_clear:
                raise RuntimeError("reset failed")
            self.tt.clear()
        elif action == "clear_board":
            self.state = BoardState.initial()
        elif action == "setboard":
            self.state = BoardState.from_board_string(args[0], parse_side(args[1]))
        elif action == "play":
            side = parse_side(args[0])
            if args[1] == "pass":
                self.state.side = side ^ 1
            else:
                self.state.apply_move(side, coord_to_policy(args[1]))
        elif action == "genmove":
            side = parse_side(args[0])
            move = self.state.legal_policies(side)[0]
            self.tt.append(move)
            self.state.apply_move(side, move)
            return "= " + policy_to_coord(move)
        return "="


class GameIsolationTest(unittest.TestCase):
    def run_games(self, specs, *, sequential=False, reject_clear=False, invalid=False):
        runner = GameRunner(specs, Mock(stop_event=threading.Event()), 1,
                            sequential_fresh_tt=sequential)
        created = []
        def create(*args):
            engine = FakeEngine(*args)
            engine.reject_clear = reject_clear
            created.append(engine)
            return engine
        for pool in runner.pools:
            original = pool.release
            def release(color, engine, original=original):
                self.assertFalse(engine.tt, "TT must be empty before returning an engine to its pool")
                original(color, engine)
            pool.release = release
        with patch("strength_engine.GtpProcess", side_effect=create):
            if invalid:
                with self.assertRaisesRegex(ValueError, "illegal XOT"):
                    runner.play_single_game(MatchSetTask(0, 0, 1, 0, "a1"), True)
            else:
                for color in (True, False, True):
                    result = runner.play_single_game(MatchSetTask(0, 0, 1, 0, ""), color)
                    self.assertEqual(60, len(result.transcript) // 2)
        return created

    def test_parallel_and_tail_modes_reset_only_after_each_whole_game(self):
        specs = [PlayerSpec(f"blend_{i}", (), 2, True, .6) for i in range(2)]
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                engines = self.run_games(specs, sequential=sequential)
                self.assertEqual(6, sum(e.events.count("clear_cache") for e in engines))
                for engine in engines:
                    self.assertEqual("clear_cache", engine.events[-1])
                    self.assertGreater(engine.events.index("clear_cache"), engine.events.index("genmove"))
                    self.assertNotIn("close", engine.events)

    def test_native_engines_are_discarded_after_every_game(self):
        engines = self.run_games([PlayerSpec(f"native_{i}", (), 2, False) for i in range(2)])
        self.assertEqual(6, len(engines))
        self.assertTrue(all(not e.usable for e in engines))

    def test_failed_resets_retire_engines_and_failed_games_clear_both_sides(self):
        specs = [PlayerSpec(f"blend_{i}", (), 2, True, .6) for i in range(2)]
        engines = self.run_games(specs, reject_clear=True)
        self.assertEqual(6, len(engines))
        self.assertTrue(all(not e.usable for e in engines))
        engines = self.run_games(specs, invalid=True)
        self.assertTrue(all(e.events[-1] == "clear_cache" for e in engines))

    def test_tournament_wrappers_recompute_identical_hints(self):
        args = battle.make_argparser().parse_args(["--baseline-levels=", "--alphas=.2,.8", "--edax-levels="])
        levels, alphas = battle.validate_args(args)
        for spec in battle.build_player_specs(args, levels, alphas, 12345):
            options = wrapper_parser().parse_args(list(spec.command)[2:])
            self.assertFalse(options.cache_egaroucid)
            self.assertIsNone(options.hint_cache_db)
        blender = object.__new__(BlendedPolicy)
        blender.cache_egaroucid = False
        blender.shared_hint_cache = None
        state = BoardState.initial()
        scores = {move: 0.0 for move in state.legal_policies(state.side)}
        blender.hint_runner = Mock(level=21, hint_scores=Mock(return_value=(scores, "hint")))
        for _ in range(2):
            blender.cached_hint_scores(state, state.side)
        self.assertEqual(2, blender.hint_runner.hint_scores.call_count)


class IndependentEloTest(unittest.TestCase):
    def test_bootstrap_independence_reproducibility_and_counts(self):
        scores = np.tile(np.asarray([0, .5, 1, 1])[:, None], (1, 3))
        counts = bootstrap_independent_pair_counts(scores, 1000, 57)
        np.testing.assert_array_equal(counts, bootstrap_independent_pair_counts(scores, 1000, 57))
        np.testing.assert_array_equal(counts.sum(axis=2), np.full((1000, 3), 4))
        self.assertFalse(np.array_equal(counts[:, 0], counts[:, 1]))
        np.testing.assert_allclose(counts.mean(axis=0), np.tile([1, 1, 2], (3, 1)), atol=.15)

    def test_independent_log_analysis_output_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            StrengthEloConfidenceTest()._write_synthetic_output(root)
            manifest_path = root / "strength_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["configuration"]["schedule"].update(
                same_opening_sequence_for_every_pair=False,
                opening_sampling="independent_per_pair_without_replacement_v1")
            manifest_path.write_text(json.dumps(manifest))
            raw = root / "strength_games.jsonl"
            records = [json.loads(line) for line in raw.read_text().splitlines()]
            for row in records:
                t = row["task"]
                opening = f"pair_{t['p0_idx']}_{t['p1_idx']}_set_{t['set_index']}"
                t["opening"] = row["result"]["opening"] = opening
            raw.write_text("".join(json.dumps(row) + "\n" for row in records))
            data = load_clustered_elo_data(root)
            self.assertTrue(data.independent_pairs)
            analysis = analyze_clustered_elos(data, bootstrap_replicates=80)
            np.testing.assert_allclose(analysis.bootstrap_elos.mean(axis=1), 1500)
            write_clustered_elo_outputs(root, data, analysis, 80, 613, .95)
            metadata = json.loads((root / "strength_elo_ci95.json").read_text())
            self.assertEqual("pair_stratified_percentile_bootstrap", metadata["method"])
            self.assertIsNone(metadata["bca"])
            with np.load(root / metadata["bootstrap"]["counts_file"]) as archive:
                np.testing.assert_array_equal(archive["counts"], analysis.bootstrap_pair_outcome_counts)
            records[3]["task"]["opening"] = records[3]["result"]["opening"] = records[0]["task"]["opening"]
            raw.write_text("".join(json.dumps(row) + "\n" for row in records))
            with self.assertRaisesRegex(ValueError, "unique within each"):
                load_clustered_elo_data(root)


if __name__ == "__main__":
    unittest.main()
