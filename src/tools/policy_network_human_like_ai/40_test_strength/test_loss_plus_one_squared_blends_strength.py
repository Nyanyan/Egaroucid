"""Verify squared-model blends preserve old games and run only appended pairs."""
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import battle_blend_strength as battle
from extend_strength_tournament import ModelExtension, make_argparser as extension_parser
import run_loss_plus_one_squared_blends_strength as launcher
from strength_tournament import ResultStore, combine_color_games, make_manifest, make_match_set_tasks
import test_strength_extension as fixtures


class SquaredBlendsStrengthTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StrengthExtensionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.model = self.root / "squared/train/selected_model.h5"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"squared model")
        (self.model.parent.parent / "run.json").write_text(json.dumps({
            "status": "completed", "returncode": 0, "sample_weight_formula": "(disc_loss + 1) ** 2",
            "model": str(self.model)}))
        (self.model.parent / "prepared_dataset.json").write_text(json.dumps({
            "sample_weights": {"formula": "(disc_loss + 1) ** 2"}}))
        self.source = self.root / "squared_tournament"
        self.source.mkdir()
        previous = ModelExtension(self.fixture.reference, self.model, launcher.LABEL)
        previous.alphas = (1.0,)
        options = extension_parser().parse_args([
            "--output-dir", str(self.source), "--include-random", "--sequential-tail-match-sets", "0"])
        args = previous.make_args(options)
        previous.validate_reference(args)
        tasks = make_match_set_tasks(previous.player_count, previous.openings, 3, args.seed)
        levels, alphas = battle.validate_args(args, previous.resource_alphas)
        config = battle.build_manifest_configuration(args, levels, alphas, previous.openings, tasks)
        previous.update_configuration(config)
        manifest = make_manifest(config)
        (self.source / "strength_manifest.json").write_text(json.dumps(manifest))
        server = {"backend": "tensorflow", "backend_version": "test", "device": "CPU"}
        runtime = make_manifest({"experiment_id": manifest["experiment_id"], "policy_server": server,
                                 "additional_policy_server": server, "gpu_model_and_driver": []})
        (self.source / "strength_runtime_manifest.json").write_text(json.dumps(runtime))
        specs = previous.add_specs(args, battle.build_player_specs(args, levels, alphas, 0), 0)
        (self.source / "strength_results.json").write_text(json.dumps({
            "experiment_id": manifest["experiment_id"], "completed_match_sets": len(tasks),
            "players": [{"name": s.name} for s in specs]}))
        store = ResultStore(self.source)
        store.open()
        for task in tasks:
            store.append_result(task, combine_color_games(task, [fixtures.color_result(task, True),
                                                                 fixtures.color_result(task, False)]))
        store.close()
        _, self.source_results = ResultStore(self.source).load({t.task_id: t for t in tasks}, read_only=True)
        self.options = launcher.make_argparser().parse_args([
            "--reference-dir", str(self.source), "--model", str(self.model),
            "--match-sets-per-pair", "3", "--output-dir", str(self.root / "blends"),
            "--sequential-tail-match-sets", "0"])

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            return launcher.prepare(self.options)

    def test_preserves_old_indices_transcripts_and_openings_and_appends_four_blends(self):
        defaults = launcher.make_argparser().parse_args([])
        self.assertEqual(50, defaults.match_sets_per_pair)
        self.assertEqual("0.2,0.4,0.6,0.8", defaults.alphas)
        args, extension = self.prepare()
        self.assertFalse(args.output_dir.exists())
        self.assertEqual(8, extension.player_count)
        tasks = make_match_set_tasks(8, extension.openings, 3, args.seed)
        imports = extension.missing_imports(tasks, [])
        self.assertEqual(18, len(imports))
        for old, imported in zip(self.source_results, imports):
            self.assertEqual(old, replace(imported, task_id=old.task_id))
        imported_ids = {r.task_id for r in imports}
        pending = [t for t in tasks if t.task_id not in imported_ids]
        self.assertEqual(Counter({(i, j): 3 for i in range(8) for j in range(i + 1, 8) if j >= 4}),
                         Counter((t.p0_idx, t.p1_idx) for t in pending))
        self.assertEqual(66, len(pending))
        self.assertTrue(all(t.opening == self.fixture.openings[t.set_index] for t in tasks))

    def test_runs_only_new_pairs_connects_both_models_and_resumes_without_replay(self):
        args, extension = self.prepare()
        args.no_performance_monitor = True
        before = {p: p.read_bytes() for folder in (self.source, self.fixture.reference)
                  for p in folder.iterdir() if p.is_file()}
        played, models = [], []

        def start(settings, manager, run_id):
            models.append(Path(settings.policy_model))
            return battle.PolicyServerInfo(10000 + len(models), "tensorflow", "CPU", "test")

        def play(task, black):
            self.assertGreaterEqual(task.p1_idx, 4)
            played.append((task.task_id, black))
            return fixtures.color_result(task, black)

        with patch.object(battle, "GameRunner") as runner, \
             patch.object(battle, "start_policy_server", side_effect=start), \
             patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), redirect_stdout(io.StringIO()):
            runner.return_value.play_single_game.side_effect = play
            battle.run_tournament(args, extension=extension)
            specs = runner.call_args.args[0]
            self.assertEqual(["egaroucid_l1", "alpha_0.2", "loss_plus_one_squared_alpha_1.0", "random"]
                             + [launcher.blend_name(a) for a in (.2, .4, .6, .8)], [s.name for s in specs])
            for spec, alpha in zip(specs[4:], (.2, .4, .6, .8)):
                command = spec.command
                self.assertEqual(str(alpha), command[command.index("--beta") + 1])
                self.assertEqual("10001", command[command.index("--normal-port") + 1])
                self.assertEqual("10002", command[command.index("--weighted-port") + 1])
            resumed, resumed_extension = self.prepare()
            resumed.no_performance_monitor = True
            battle.run_tournament(resumed, extension=resumed_extension)
        self.assertEqual(132, len(played))
        self.assertEqual([self.fixture.old_model, self.model], models)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        config = json.loads((args.output_dir / "strength_manifest.json").read_text())["configuration"]
        self.assertEqual([s.name for s in specs[3:]], [p["name"] for p in config["participants"]["policy_opponents"]])
        tasks = make_match_set_tasks(8, extension.openings, 3, args.seed)
        _, results = ResultStore(args.output_dir).load({t.task_id: t for t in tasks}, read_only=True)
        self.assertEqual(84, len(results))
        self.assertEqual([], extension.missing_imports(tasks, results))

    def test_changed_model_and_unfinished_training_are_rejected(self):
        self.model.write_bytes(b"different squared model")
        with self.assertRaisesRegex(ValueError, "Squared model differs"):
            self.prepare()
        self.model.write_bytes(b"squared model")
        path = self.model.parent.parent / "run.json"
        record = json.loads(path.read_text())
        record["status"] = "running"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "finished successfully"):
            self.prepare()
        self.assertFalse(self.options.output_dir.exists())

    def test_truncated_source_stops_without_repair_or_output(self):
        path = self.source / "strength_games.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"broken":')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()
        self.assertEqual(before, path.read_bytes())
        self.assertFalse(self.options.output_dir.exists())

    def test_invalid_weights_and_reference_output_are_rejected(self):
        for weights in ("", "0,0.2", "1", "nan", "0.2,0.2"):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.options.alphas = weights
                self.prepare()
        self.options.alphas = "0.2,0.4,0.6,0.8"
        self.options.output_dir = self.source
        with self.assertRaisesRegex(ValueError, "separate"):
            self.prepare()
        self.options.output_dir = self.root / "blends"
        self.options.match_sets_per_pair = 4
        with self.assertRaisesRegex(ValueError, "same complete opening prefix|opening sequence"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
