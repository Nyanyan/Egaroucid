"""Check L21 blend wiring, full result reuse, and restart without duplicate games."""
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import battle_blend_strength as battle
import run_loss_plus_one_squared_egaroucid_strength as launcher
import test_loss_plus_one_squared_blends_strength as blend_fixtures
import test_strength_extension as fixtures
from strength_elo_confidence import load_clustered_elo_data
from strength_tournament import ResultStore, combine_color_games, make_manifest, make_match_set_tasks


class SquaredEgaroucidStrengthTest(unittest.TestCase):
    def setUp(self):
        self.fixture = blend_fixtures.SquaredBlendsStrengthTest()
        original_parser = battle.make_argparser

        def logit_parser():
            parser = original_parser()
            parser.set_defaults(logit_blend=True)
            return parser

        with patch.object(battle, "make_argparser", side_effect=logit_parser):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.model = self.fixture.root, self.fixture.model
        args, previous = self.fixture.prepare()
        self.source = args.output_dir
        self.source.mkdir()
        tasks = make_match_set_tasks(previous.player_count, previous.openings, 3, args.seed)
        levels, alphas = battle.validate_args(args, previous.resource_alphas)
        config = battle.build_manifest_configuration(args, levels, alphas, previous.openings, tasks)
        previous.update_configuration(config)
        manifest = make_manifest(config)
        (self.source / "strength_manifest.json").write_text(json.dumps(manifest))
        runtime_config = json.loads((self.fixture.source / "strength_runtime_manifest.json").read_text())["configuration"]
        runtime_config["experiment_id"] = manifest["experiment_id"]
        (self.source / "strength_runtime_manifest.json").write_text(json.dumps(make_manifest(runtime_config)))
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
        self.retained_names = [s.name for s in specs]
        self.options = launcher.make_argparser().parse_args([
            "--reference-dir", str(self.source), "--model", str(self.model),
            "--match-sets-per-pair", "3", "--sequential-tail-match-sets", "0",
            "--output-dir", str(self.root / "with_l21")])

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            return launcher.prepare(self.options)

    def test_imports_every_old_pair_and_only_schedules_new_blends(self):
        args, extension = self.prepare()
        self.assertFalse(args.output_dir.exists())
        self.assertEqual(12, extension.player_count)
        self.assertEqual((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, .2, .4, .6, .8), extension.resource_alphas)
        tasks = make_match_set_tasks(extension.player_count, extension.openings, 3, args.seed)
        imports = extension.missing_imports(tasks, [])
        self.assertEqual(84, len(imports))
        for old, imported in zip(self.source_results, imports):
            self.assertEqual(old, replace(imported, task_id=old.task_id))
        pending = [t for t in tasks if t.task_id not in {r.task_id for r in imports}]
        self.assertEqual(114, len(pending))
        self.assertEqual(Counter({(i, j): 3 for i in range(12) for j in range(i + 1, 12) if j >= 8}),
                         Counter((t.p0_idx, t.p1_idx) for t in pending))
        self.assertTrue(all(len({t.opening for t in tasks if (t.p0_idx, t.p1_idx) == pair}) == 3
                            for pair in {(t.p0_idx, t.p1_idx) for t in tasks}))

    def test_runs_l21_blends_on_squared_server_and_resumes_without_replaying(self):
        args, extension = self.prepare()
        args.no_performance_monitor = True
        before = {p: p.read_bytes() for folder in (self.source, self.fixture.fixture.reference)
                  for p in folder.iterdir() if p.is_file()}
        played, models = [], []

        def start(settings, manager, run_id):
            models.append(Path(settings.policy_model))
            return battle.PolicyServerInfo(10000 + len(models), "tensorflow", "CPU", "test")

        def play(task, black):
            self.assertGreaterEqual(task.p1_idx, 8)
            played.append((task.task_id, black))
            return fixtures.color_result(task, black)

        with patch.object(battle, "GameRunner") as runner, \
             patch.object(battle, "start_policy_server", side_effect=start), \
             patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), redirect_stdout(io.StringIO()):
            runner.return_value.play_single_game.side_effect = play
            battle.run_tournament(args, extension=extension)
            specs = runner.call_args.args[0]
            self.assertEqual(self.retained_names, [s.name for s in specs[:8]])
            self.assertEqual(1, sum(s.name == "loss_plus_one_squared_alpha_1.0" for s in specs))
            for spec, alpha in zip(specs[8:], (.2, .4, .6, .8)):
                self.assertEqual(launcher.blend_name(alpha), spec.name)
                self.assertEqual(alpha, spec.alpha)
                command = spec.command
                for flag, value in (("--alpha", str(alpha)), ("--egaroucid-level", "21"),
                                    ("--policy-server-port", "10002"), ("--score-temperature", "1.0")):
                    self.assertEqual(value, command[command.index(flag) + 1])
                self.assertIn("--logit-blend", command)
                self.assertNotIn("--beta", command)
            # Simulate an interrupted run by retaining a completed subset of the log.
            log = args.output_dir / "strength_games.jsonl"
            lines = log.read_bytes().splitlines(keepends=True)
            log.write_bytes(b"".join(lines[:-2]))  # Two complete paired sets.
            resumed, resumed_extension = self.prepare()
            resumed.no_performance_monitor = True
            battle.run_tournament(resumed, extension=resumed_extension)
            self.assertEqual(232, len(played))
            resumed, resumed_extension = self.prepare()
            resumed.no_performance_monitor = True
            battle.run_tournament(resumed, extension=resumed_extension)
        self.assertEqual(232, len(played))
        self.assertEqual([self.fixture.fixture.old_model, self.model] * 2, models)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        config = json.loads((args.output_dir / "strength_manifest.json").read_text())["configuration"]
        self.assertEqual([s.name for s in specs[3:]], [p["name"] for p in config["participants"]["policy_opponents"]])
        tasks = make_match_set_tasks(12, extension.openings, 3, args.seed)
        _, results = ResultStore(args.output_dir).load({t.task_id: t for t in tasks}, read_only=True)
        self.assertEqual(198, len(results))
        self.assertEqual([], extension.missing_imports(tasks, results))
        self.assertEqual(tuple(s.name for s in specs), load_clustered_elo_data(args.output_dir).names)

    def test_endpoints_nonfinite_duplicates_and_source_output_are_rejected(self):
        for alphas in ("", "0,0.2", "1.0", "nan", "inf", "0.2,0.2"):
            with self.subTest(alphas=alphas), self.assertRaises(ValueError):
                self.options.alphas = alphas
                self.prepare()
        self.options.alphas = "0.2,0.4,0.6,0.8"
        self.options.output_dir = self.source
        with self.assertRaisesRegex(ValueError, "separate"):
            self.prepare()

    def test_changed_model_and_incomplete_source_cannot_be_imported(self):
        self.model.write_bytes(b"changed model")
        with self.assertRaisesRegex(ValueError, "Squared model differs"):
            self.prepare()
        self.model.write_bytes(b"squared model")
        path = self.source / "strength_games.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"broken":')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()
        self.assertEqual(before, path.read_bytes())
        self.assertFalse(self.options.output_dir.exists())

    def test_changed_retained_blend_formula_is_rejected(self):
        path = self.source / "strength_manifest.json"
        config = json.loads(path.read_text())["configuration"]
        config["participants"]["policy_opponents"][1]["formula"] = "arithmetic mixture"
        path.write_text(json.dumps(make_manifest(config)))
        with self.assertRaisesRegex(ValueError, "roster or blend formula"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
