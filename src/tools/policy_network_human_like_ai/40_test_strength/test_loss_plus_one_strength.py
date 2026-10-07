"""Keep only the standalone new model's games, retaining prior opponents' results."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import battle_blend_strength as battle
from extend_strength_tournament import ModelExtension, make_argparser as extension_parser
import run_loss_plus_one_strength as launcher
from strength_tournament import ResultStore, combine_color_games, make_manifest, make_match_set_tasks
import test_strength_extension as fixtures


class LossPlusOneStrengthTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StrengthExtensionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.source = self.root / "with_exponential"
        self.source.mkdir()
        self.exponential = self.root / "exponential.h5"
        self.exponential.write_bytes(b"exponential")
        previous = ModelExtension(self.fixture.reference, self.exponential, "loss_weighted")
        options = extension_parser().parse_args([
            "--output-dir", str(self.source), "--include-random", "--model-blend-betas", "0.4",
            "--sequential-tail-match-sets", "0"])
        args = previous.make_args(options)
        previous.validate_reference(args)
        tasks = make_match_set_tasks(previous.player_count, previous.openings, 3, args.seed)
        levels, alphas = battle.validate_args(args, previous.resource_alphas)
        configuration = battle.build_manifest_configuration(args, levels, alphas, previous.openings, tasks)
        previous.update_configuration(configuration)
        manifest = make_manifest(configuration)
        (self.source / "strength_manifest.json").write_text(json.dumps(manifest))
        runtime = make_manifest({"experiment_id": manifest["experiment_id"],
                                "policy_server": {"backend": "tensorflow", "backend_version": "test", "device": "CPU"},
                                "gpu_model_and_driver": []})
        (self.source / "strength_runtime_manifest.json").write_text(json.dumps(runtime))
        specs = previous.add_specs(args, battle.build_player_specs(args, levels, alphas, 0), 0)
        aggregate = {"experiment_id": manifest["experiment_id"], "completed_match_sets": len(tasks),
                     "players": [{"name": s.name} for s in specs]}
        (self.source / "strength_results.json").write_text(json.dumps(aggregate))
        store = ResultStore(self.source)
        store.open()
        for task in tasks:
            store.append_result(task, combine_color_games(task, [fixtures.color_result(task, True), fixtures.color_result(task, False)]))
        store.close()
        self.options = launcher.make_argparser().parse_args([
            "--reference-dir", str(self.source), "--model", str(self.fixture.new_model),
            "--match-sets-per-pair", "3", "--sequential-tail-match-sets", "0",
            "--output-dir", str(self.root / "new_only")])

    def test_schedule_excludes_every_exponential_player_and_preserves_openings(self):
        self.assertEqual(50, launcher.make_argparser().parse_args([]).match_sets_per_pair)
        with redirect_stdout(io.StringIO()):
            args, extension = launcher.prepare(self.options)
        self.assertFalse(args.output_dir.exists())
        self.assertEqual((1.0,), extension.alphas)
        self.assertEqual(["loss_weighted_alpha_0.2", "model_blend_beta_0.4"], extension.excluded_names)
        tasks = make_match_set_tasks(4, extension.openings, 3, args.seed)
        imports = extension.missing_imports(tasks, [])
        self.assertEqual(9, len(imports))
        self.assertEqual(6, extension.random_import_count)
        pending = [t for t in tasks if t.task_id not in {r.task_id for r in imports}]
        self.assertEqual(9, len(pending))
        self.assertTrue(all(2 in (t.p0_idx, t.p1_idx) for t in pending))
        self.assertTrue(all(t.opening == self.fixture.openings[t.set_index] for t in tasks))
        for result in imports:
            for game in result.color_games:
                self.assertEqual((result.p0_idx, result.p1_idx), (game.p0_idx, game.p1_idx))
                self.assertEqual(result.p0_idx if game.p0_is_black else result.p1_idx, game.black_idx)
                self.assertEqual(result.p1_idx if game.p0_is_black else result.p0_idx, game.white_idx)
                self.assertEqual(result.opening, game.transcript)

    def test_runs_only_new_model_games_and_resume_never_replays(self):
        with redirect_stdout(io.StringIO()):
            args, extension = launcher.prepare(self.options)
        args.no_performance_monitor = True
        before = {(directory.name, p.name): p.read_bytes() for directory in (self.source, self.fixture.reference)
                  for p in directory.iterdir() if p.is_file()}
        played, models = [], []
        def start(settings, manager, run_id):
            models.append(Path(settings.policy_model))
            return battle.PolicyServerInfo(10000 + len(models), "tensorflow", "CPU", "test")
        def play(task, black):
            self.assertIn(2, (task.p0_idx, task.p1_idx))
            played.append((task.task_id, black))
            return fixtures.color_result(task, black)
        with patch.object(battle, "GameRunner") as runner, \
             patch.object(battle, "start_policy_server", side_effect=start), \
             patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), redirect_stdout(io.StringIO()):
            runner.return_value.play_single_game.side_effect = play
            battle.run_tournament(args, extension=extension)
            specs = runner.call_args.args[0]
            self.assertEqual(["egaroucid_l1", "alpha_0.2", "loss_plus_one_alpha_1.0", "random"], [s.name for s in specs])
            command = specs[2].command
            self.assertEqual("10002", command[command.index("--policy-server-port") + 1])
            resumed_args, resumed_extension = launcher.prepare(self.options)
            resumed_args.no_performance_monitor = True
            battle.run_tournament(resumed_args, extension=resumed_extension)
        self.assertEqual(18, len(played))
        self.assertEqual([self.fixture.old_model, self.fixture.new_model], models)
        after = {(directory.name, p.name): p.read_bytes() for directory in (self.source, self.fixture.reference)
                 for p in directory.iterdir() if p.is_file()}
        self.assertEqual(before, after)
        tasks = make_match_set_tasks(4, extension.openings, 3, args.seed)
        completed, _ = ResultStore(args.output_dir).load({t.task_id: t for t in tasks}, read_only=True)
        self.assertEqual(18, len(completed))

    def test_truncated_reference_is_rejected_without_repair_or_output(self):
        path = self.source / "strength_games.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"broken":')
        before = path.read_bytes()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "incomplete"):
            launcher.prepare(self.options)
        self.assertEqual(before, path.read_bytes())
        self.assertFalse(self.options.output_dir.exists())

    def test_reference_output_and_wrong_opening_prefix_are_rejected(self):
        self.options.output_dir = self.source
        with self.assertRaisesRegex(ValueError, "separate"):
            launcher.prepare(self.options)
        self.options.output_dir = self.root / "new_only"
        self.options.match_sets_per_pair = 4
        with self.assertRaisesRegex(ValueError, "opening sequence"):
            launcher.prepare(self.options)


if __name__ == "__main__":
    unittest.main()
