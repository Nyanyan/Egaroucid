from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import battle_blend_strength as battle
from extend_strength_tournament import ModelExtension, make_argparser, main
from strength_elo_confidence import load_clustered_elo_data
from strength_recovery import prepare_extension_output, validate_prefix_change
from strength_tournament import (
    GameResult, ResultStore, combine_color_games, make_manifest, make_match_set_tasks,
)


OPENING = "f5d6c3d3c4f4c5b3"


def color_result(task, black):
    return GameResult(task.p0_idx, task.p1_idx, black,
                      task.p0_idx if black else task.p1_idx,
                      task.p1_idx if black else task.p0_idx,
                      2, 33 if black else 31, 31 if black else 33, task.opening)


class StrengthExtensionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.old_model, self.new_model = self.root / "old.h5", self.root / "new.h5"
        self.old_model.write_bytes(b"old model")
        self.new_model.write_bytes(b"new model")
        engine = self.root / "engine.exe"
        engine.write_bytes(b"test engine")
        openings = self.root / "openings.txt"
        rotated = "".join(chr(ord('h') - (ord(OPENING[i]) - ord('a'))) + str(9 - int(OPENING[i + 1]))
                          for i in range(0, len(OPENING), 2))
        transposed = "".join(chr(ord('a') + int(OPENING[i + 1]) - 1) + str(ord(OPENING[i]) - ord('a') + 1)
                             for i in range(0, len(OPENING), 2))
        openings.write_text("\n".join((OPENING, rotated, transposed)) + "\n")
        args = battle.make_argparser().parse_args([
            "--baseline-levels", "1", "--alphas", "0.2", "--edax-levels=",
            "--openings", str(openings), "--egaroucid-exe", str(engine),
            "--policy-model", str(self.old_model), "--policy-backend", "tensorflow",
            "--allow-policy-cpu", "--match-sets-per-pair", "3",
            "--parallel-match-sets", "1", "--parallel-games", "2",
            "--blend-processes-per-player", "2", "--output-dir", str(self.reference),
        ])
        self.openings = battle.read_openings(openings, args.seed)
        self.source_tasks = make_match_set_tasks(2, self.openings, 3, args.seed)
        configuration = battle.build_manifest_configuration(args, [1], [0.2], self.openings, self.source_tasks)
        manifest = make_manifest(configuration)
        (self.reference / "strength_manifest.json").write_text(json.dumps(manifest))
        runtime = make_manifest({
            "experiment_id": manifest["experiment_id"],
            "policy_server": {"backend": "tensorflow", "backend_version": "test", "device": "CPU"},
            "gpu_model_and_driver": [],
        })
        (self.reference / "strength_runtime_manifest.json").write_text(json.dumps(runtime))
        specs = battle.build_player_specs(args, [1], [0.2], 0)
        aggregate = {"experiment_id": manifest["experiment_id"], "completed_match_sets": 3,
                     "players": [{"name": s.name} for s in specs]}
        (self.reference / "strength_results.json").write_text(json.dumps(aggregate))
        store = ResultStore(self.reference)
        store.open()
        for task in self.source_tasks:
            store.append_result(task, combine_color_games(task, [color_result(task, True), color_result(task, False)]))
        store.close()
        self.extension = ModelExtension(self.reference, self.new_model, "weighted")
        options = make_argparser().parse_args([
            "--reference-dir", str(self.reference), "--model", str(self.new_model),
            "--output-dir", str(self.root / "extended"),
            "--sequential-tail-match-sets", "0",
        ])
        self.args = self.extension.make_args(options)
        self.args.no_performance_monitor = True
        self.extension.validate_reference(self.args)

    def test_runs_only_new_pairs_binds_two_models_and_resumes_without_replay(self):
        before = {p.name: p.read_bytes() for p in self.reference.iterdir()}
        played = []
        servers = []
        def start(args, manager, run_id):
            servers.append(Path(args.policy_model))
            return battle.PolicyServerInfo(10000 + len(servers), "tensorflow", "CPU", "test")
        def play(task, black):
            played.append((task.p0_idx, task.p1_idx, black))
            self.assertGreaterEqual(task.p1_idx, 2)
            return color_result(task, black)
        with patch.object(battle, "GameRunner") as runner, patch.object(battle, "start_policy_server", side_effect=start), \
             patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), redirect_stdout(io.StringIO()):
            runner.return_value.play_single_game.side_effect = play
            battle.run_tournament(self.args, extension=self.extension)
            specs = runner.call_args.args[0]
            old_command, new_command = specs[1].command, specs[2].command
            self.assertEqual("10001", old_command[old_command.index("--policy-server-port") + 1])
            self.assertEqual("10002", new_command[new_command.index("--policy-server-port") + 1])
            battle.run_tournament(self.args, extension=self.extension)
        self.assertEqual([self.old_model, self.new_model], servers)
        self.assertEqual(12, len(played))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.reference.iterdir()})
        tasks = make_match_set_tasks(3, self.openings, 3, self.args.seed)
        completed, results = ResultStore(self.args.output_dir).load({t.task_id: t for t in tasks})
        self.assertEqual(9, len(completed))
        self.assertEqual(3, sum(r.p1_idx < 2 for r in results))
        # The combined table retains a complete per-XOT round robin for Elo CI.
        data = load_clustered_elo_data(self.args.output_dir)
        self.assertEqual(("egaroucid_l1", "alpha_0.2", "weighted_alpha_0.2"), data.names)

    def test_remaps_task_ids_and_recovers_partial_import_without_duplicates(self):
        tasks = make_match_set_tasks(3, self.openings, 3, self.args.seed)
        imported = self.extension.missing_imports(tasks, [])
        self.assertEqual(3, len(imported))
        self.assertEqual(imported[1:], self.extension.missing_imports(tasks, imported[:1]))
        self.assertEqual([], self.extension.missing_imports(tasks, imported))
        task = next(t for t in tasks if t.p0_idx == 0 and t.p1_idx == 1)
        self.assertEqual(task.task_id, imported[0].task_id)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.extension.missing_imports(tasks, [replace(imported[0], p0_disc_diff=-2)])

    def test_changed_source_model_stops(self):
        self.old_model.write_bytes(b"other model")
        with self.assertRaisesRegex(ValueError, "Reference input changed"):
            self.extension.validate_reference(self.args)

    def test_read_only_source_never_repairs_truncated_log(self):
        path = self.reference / "strength_games.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"broken":')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ResultStore(self.reference).load({t.task_id: t for t in self.source_tasks}, read_only=True)
        self.assertEqual(before, path.read_bytes())

    def test_reference_directory_cannot_be_output(self):
        options = make_argparser().parse_args([
            "--reference-dir", str(self.reference), "--model", str(self.new_model),
            "--output-dir", str(self.reference),
        ])
        with self.assertRaisesRegex(ValueError, "separate"):
            self.extension.make_args(options)

    def test_prefix_recovery_preserves_completed_results_and_backups(self):
        before_reference = {p.name: p.read_bytes() for p in self.reference.iterdir()}
        def start(args, manager, run_id):
            return battle.PolicyServerInfo(10001, "tensorflow", "CPU", "test")
        with patch.object(battle, "GameRunner") as runner, \
             patch.object(battle, "start_policy_server", side_effect=start), \
             patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), redirect_stdout(io.StringIO()):
            runner.return_value.play_single_game.side_effect = color_result
            battle.run_tournament(self.args, extension=self.extension)
            before = {p.name: p.read_bytes() for p in self.args.output_dir.iterdir()}
            self.args.match_sets_per_pair = 2
            self.extension.validate_reference(self.args)
            self.assertEqual(2, len(self.extension.results))
            prepare_extension_output(self.args, self.extension, dry_run=True)
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.args.output_dir.iterdir()})
            prepare_extension_output(self.args, self.extension)
            played = runner.return_value.play_single_game.call_count
            battle.run_tournament(self.args, extension=self.extension)
            self.assertEqual(played, runner.return_value.play_single_game.call_count)
        backups = list(self.args.output_dir.glob("recovery_*/before"))
        self.assertEqual(1, len(backups))
        self.assertEqual(before['strength_games.jsonl'], (backups[0] / 'strength_games.jsonl').read_bytes())
        tasks = make_match_set_tasks(3, self.openings, 2, self.args.seed)
        _, results = ResultStore(self.args.output_dir).load({t.task_id: t for t in tasks})
        self.assertEqual(6, len(results))
        self.assertEqual({0, 1}, {r.set_index for r in results})
        self.assertEqual(before_reference, {p.name: p.read_bytes() for p in self.reference.iterdir()})
        aggregate = json.loads((self.args.output_dir / "strength_results.json").read_text())
        self.assertEqual(6, aggregate["completed_match_sets"])
        self.assertEqual(6, aggregate["total_match_sets"])

    def test_prefix_recovery_rejects_changed_search_conditions(self):
        import copy
        config = copy.deepcopy(self.extension.configuration)
        config['imported_reference'] = {'imported_match_sets': 3}
        changed = copy.deepcopy(config)
        changed['engine']['blend_egaroucid_level'] -= 1
        with self.assertRaisesRegex(ValueError, "settings differ"):
            validate_prefix_change(config, changed)

    def test_hint_timeout_override_extends_outer_wait_and_rejects_invalid_values(self):
        options = make_argparser().parse_args([
            '--output-dir', str(self.root / 'new-output'), '--egaroucid-hint-timeout-sec', '5400'])
        args = self.extension.make_args(options)
        self.assertEqual((5400, 5500), (args.egaroucid_hint_timeout_sec, args.gtp_command_timeout_sec))
        options.gtp_command_timeout_sec = 5700
        args = self.extension.make_args(options)
        self.assertEqual((5400, 5700), (args.egaroucid_hint_timeout_sec, args.gtp_command_timeout_sec))
        options.gtp_command_timeout_sec = 5400
        with self.assertRaisesRegex(ValueError, 'must exceed'):
            self.extension.make_args(options)
        options.gtp_command_timeout_sec = None
        for invalid in (0, -1, float('nan'), float('inf')):
            options.egaroucid_hint_timeout_sec = invalid
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                self.extension.make_args(options)

    def test_pending_timeouts_preserve_results_and_persist_across_resume(self):
        def start(args, manager, run_id):
            return battle.PolicyServerInfo(10001, 'tensorflow', 'CPU', 'test')
        with patch.object(battle, 'GameRunner') as runner, \
             patch.object(battle, 'start_policy_server', side_effect=start), \
             patch.object(battle, 'available_memory_mib', return_value=1e9), \
             patch.object(battle, 'validate_input_files'), redirect_stdout(io.StringIO()):
            runner.return_value.play_single_game.side_effect = color_result
            battle.run_tournament(self.args, extension=self.extension)
            raw = self.args.output_dir / 'strength_games.jsonl'
            before = raw.read_bytes()
            pending = self.args.output_dir / 'strength_pending_opponents.json'
            pending.write_text(json.dumps({'egaroucid_hint_timeout_sec': 5400, 'gtp_command_timeout_sec': 5700}))
            main(['--output-dir', str(self.args.output_dir), '--dry-run'])
            self.assertTrue(pending.exists())
            self.assertEqual(before, raw.read_bytes())
            main(['--output-dir', str(self.args.output_dir), '--prepare-only'])
            self.assertFalse(pending.exists())
            self.assertEqual(before, raw.read_bytes())
            saved = json.loads((self.args.output_dir / 'strength_manifest.json').read_text())['configuration']
            self.assertEqual(5400, saved['engine']['egaroucid_hint_timeout_sec'])
            self.assertEqual(5700, saved['engine']['gtp_command_timeout_sec'])
            receipt = json.loads(next(self.args.output_dir.glob('recovery_*/recovery.json')).read_text())
            self.assertEqual(1800, receipt['previous_timeouts']['egaroucid_hint_timeout_sec'])
            self.assertEqual(5400, receipt['timeouts_for_remaining_games']['egaroucid_hint_timeout_sec'])
            with patch('strength_recovery.prepare_extension_output') as prepare:
                main(['--output-dir', str(self.args.output_dir), '--dry-run'])
                resumed = prepare.call_args.args[0]
                self.assertEqual((5400, 5700), (resumed.egaroucid_hint_timeout_sec, resumed.gtp_command_timeout_sec))
                main(['--output-dir', str(self.args.output_dir), '--egaroucid-hint-timeout-sec', '7200', '--dry-run'])
                resumed = prepare.call_args.args[0]
                self.assertEqual((7200, 7300), (resumed.egaroucid_hint_timeout_sec, resumed.gtp_command_timeout_sec))

    def test_default_uses_first_50_of_reference_100_without_changing_seed(self):
        self.extension.configuration['schedule']['match_sets_per_pair'] = 100
        options = make_argparser().parse_args([
            '--reference-dir', str(self.reference), '--model', str(self.new_model),
            '--output-dir', str(self.root / 'new-output'),
        ])
        args = self.extension.make_args(options)
        self.assertEqual(50, args.match_sets_per_pair)
        self.assertEqual(self.extension.configuration['schedule']['xot_seed'], args.seed)

    def test_new_output_requires_reference_and_model(self):
        with self.assertRaisesRegex(ValueError, "requires --reference-dir and --model"):
            main(['--output-dir', str(self.root / 'new-output'), '--dry-run'])

    def test_legacy_reference_enables_threads_and_manifest_records_game_boundary(self):
        import copy
        self.extension.configuration['engine'].pop('adaptive_search_threads')
        options = make_argparser().parse_args([
            '--reference-dir', str(self.reference), '--model', str(self.new_model),
            '--output-dir', str(self.root / 'new-output'),
        ])
        args = self.extension.make_args(options)
        self.assertTrue(args.adaptive_engine_threads)
        self.assertEqual(64, args.max_engine_threads)
        self.assertEqual(70, args.engine_thread_budget)
        current = battle.build_manifest_configuration(args, [1], [.2], self.openings, self.source_tasks)
        self.extension.update_configuration(current)
        self.assertEqual('next_game', current['engine']['adaptive_search_threads']['apply_at'])
        self.assertEqual('fixed_total_between_games', current['engine']['adaptive_search_threads']['allocation'])
        self.assertEqual(70, current['engine']['adaptive_search_threads']['total_threads'])
        previous = copy.deepcopy(current)
        previous['engine'].pop('adaptive_search_threads')
        validate_prefix_change(previous, current)
        options.adaptive_engine_threads = False
        self.assertFalse(self.extension.make_args(options).adaptive_engine_threads)

    def test_old_cpu_feedback_cap_is_replaced_by_total_thread_budget(self):
        self.extension.configuration['engine']['adaptive_search_threads'] = {
            'enabled': True, 'maximum_per_process': 4, 'cpu_target_percent': 90}
        options = make_argparser().parse_args([
            '--reference-dir', str(self.reference), '--model', str(self.new_model),
            '--output-dir', str(self.root / 'new-output')])
        args = self.extension.make_args(options)
        self.assertEqual((70, 64), (args.engine_thread_budget, args.max_engine_threads))
        options.max_engine_threads = 8
        self.assertEqual(8, self.extension.make_args(options).max_engine_threads)

    def test_pending_opponents_preserve_existing_games_remap_ids_and_resume_only_missing_pairs(self):
        def start(args, manager, run_id):
            return battle.PolicyServerInfo(101 if Path(args.policy_model) == self.old_model else 102,
                                           "tensorflow", "CPU", "test")
        with patch.object(battle, "GameRunner") as runner, \
             patch.object(battle, "start_policy_server", side_effect=start), \
             patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), redirect_stdout(io.StringIO()):
            runner.return_value.play_single_game.side_effect = color_result
            battle.run_tournament(self.args, extension=self.extension)
            old_tasks = make_match_set_tasks(3, self.openings, 3, self.args.seed)
            raw_path = self.args.output_dir / 'strength_games.jsonl'
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            missing = next(row for row in rows if row['task']['p1_idx'] == 2)
            rows.remove(missing)
            raw_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            old_raw = raw_path.read_bytes()
            _, before = ResultStore(self.args.output_dir).load({t.task_id: t for t in old_tasks})
            store = ResultStore(self.args.output_dir)
            store.append_failure(old_tasks[missing['task']['task_id']], 1, RuntimeError('test'), '')
            store.close()
            request = self.args.output_dir / 'strength_pending_opponents.json'
            request.write_text(json.dumps({'model_blend_betas': '0.2,0.4,0.6,0.8',
                                          'include_random': True, 'engine_thread_budget': 70}))
            main(['--output-dir', str(self.args.output_dir), '--dry-run'])
            self.assertTrue(request.exists())
            self.assertEqual(old_raw, raw_path.read_bytes())
            main(['--output-dir', str(self.args.output_dir), '--prepare-only'])
            self.assertFalse(request.exists())
            manifest = json.loads((self.args.output_dir / 'strength_manifest.json').read_text())
            config = manifest['configuration']
            self.assertEqual(70, config['engine']['adaptive_search_threads']['total_threads'])
            self.assertEqual(5, len(config['participants']['policy_opponents']))
            new_tasks = make_match_set_tasks(8, self.openings, 3, self.args.seed)
            _, after = ResultStore(self.args.output_dir).load({t.task_id: t for t in new_tasks})
            by_key = lambda results: {(r.p0_idx, r.p1_idx, r.set_index): r for r in results}
            old_by_key, new_by_key = by_key(before), by_key(after)
            self.assertEqual(old_by_key.keys(), new_by_key.keys())
            self.assertTrue(any(old.task_id != new_by_key[key].task_id for key, old in old_by_key.items()))
            for key, old in old_by_key.items():
                self.assertEqual(replace(old, task_id=new_by_key[key].task_id), new_by_key[key])
            backup = next(self.args.output_dir.glob('recovery_*/before'))
            self.assertEqual(old_raw, (backup / 'strength_games.jsonl').read_bytes())
            failure = json.loads((self.args.output_dir / 'strength_failed_tasks.jsonl').read_text())
            self.assertEqual(missing['task'], failure['original_task'])
            self.assertEqual(failure['task'], new_tasks[failure['task']['task_id']].identity())
            played = []
            def play(task, color):
                key = (task.p0_idx, task.p1_idx, task.set_index)
                self.assertNotIn(key, old_by_key)
                played.append(key)
                return color_result(task, color)
            runner.return_value.play_single_game.side_effect = play
            main(['--output-dir', str(self.args.output_dir)])
            self.assertEqual((84 - len(before)) * 2, len(played))
            specs = runner.call_args.args[0]
            self.assertEqual(['model_blend_beta_0.2', 'model_blend_beta_0.4',
                              'model_blend_beta_0.6', 'model_blend_beta_0.8', 'random'],
                             [s.name for s in specs[-5:]])
            for spec in specs[-5:-1]:
                self.assertEqual('101', spec.command[spec.command.index('--normal-port') + 1])
                self.assertEqual('102', spec.command[spec.command.index('--weighted-port') + 1])
                self.assertEqual(1.0, spec.alpha)
            self.assertEqual(self.args.seed, specs[-1].random_seed)
            played_count = len(played)
            main(['--output-dir', str(self.args.output_dir)])
            self.assertEqual(played_count, len(played))
        data = load_clustered_elo_data(self.args.output_dir)
        self.assertEqual(8, len(data.names))

    def test_pending_parallelism_change_preserves_results_and_applies_on_next_start(self):
        def start(args, manager, run_id):
            return battle.PolicyServerInfo(101, "tensorflow", "CPU", "test")
        with patch.object(battle, "GameRunner") as runner, \
             patch.object(battle, "start_policy_server", side_effect=start), \
             patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), redirect_stdout(io.StringIO()):
            runner.return_value.play_single_game.side_effect = color_result
            battle.run_tournament(self.args, extension=self.extension)
            raw = self.args.output_dir / "strength_games.jsonl"
            before = raw.read_bytes()
            pending = self.args.output_dir / "strength_pending_opponents.json"
            pending.write_text(json.dumps({"parallel_games": 70, "adaptive_engine_threads": True,
                                           "engine_threads": 1, "engine_thread_budget": 70,
                                           "max_engine_threads": 64}))
            main(["--output-dir", str(self.args.output_dir), "--dry-run"])
            self.assertTrue(pending.exists())
            self.assertEqual(before, raw.read_bytes())
            main(["--output-dir", str(self.args.output_dir), "--prepare-only"])
            self.assertFalse(pending.exists())
            self.assertEqual(before, raw.read_bytes())
            config = json.loads((self.args.output_dir / "strength_manifest.json").read_text())["configuration"]
            self.assertEqual((70, 35), (config["engine"]["parallel_games"], config["engine"]["parallel_match_sets"]))
            self.assertEqual(1, config["engine"]["engine_threads"])
            self.assertEqual("admitted_games_first", config["engine"]["adaptive_search_threads"]["process_allocation"])
            self.assertEqual(64, config["engine"]["adaptive_search_threads"]["maximum_per_process"])

    def test_invalid_betas_and_changes_to_existing_opponents_are_rejected(self):
        import copy
        for betas in ('0.2,0.2', '0', '1', 'nan', '-0.2', '1.1'):
            options = make_argparser().parse_args(['--output-dir', str(self.root / 'new'),
                                                   '--model-blend-betas', betas])
            with self.assertRaises(ValueError):
                self.extension.make_args(options)
        original = copy.deepcopy(self.extension.configuration)
        original['participants']['policy_opponents'] = [{'name': 'existing', 'beta': .2}]
        changed = copy.deepcopy(original)
        changed['participants']['policy_opponents'][0]['beta'] = .4
        with self.assertRaisesRegex(ValueError, 'cannot be'):
            validate_prefix_change(original, changed)


if __name__ == "__main__":
    unittest.main()
