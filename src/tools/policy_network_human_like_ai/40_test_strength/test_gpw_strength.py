"""Check paper roster identity, model routing, and complete round-robin records."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import battle_blend_strength as battle
import run_gpw_strength as launcher
from strength_elo_confidence import load_clustered_elo_data
from strength_tournament import GameResult, make_match_set_tasks


class GpwStrengthTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = launcher.make_argparser().parse_args([
            "--output-dir", str(self.root / "results"), "--no-performance-monitor",
            "--sequential-tail-match-sets", "0",
        ])
        for name, argument in (("normal", "policy_model"), ("loss_a", "loss_a_model"),
                               ("loss_b", "loss_b_model")):
            folder = self.root / name
            folder.mkdir()
            model = folder / "selected_model.h5"
            model.write_bytes(name.encode())
            setattr(self.args, argument, model)
            records = {
                "selected_summary.json": {"name": "conv8_bn", "spec": {"architecture": "conv8_bn"},
                                          "epochs_ran": 10, "adopted_epoch": 10},
                "training_identity.json": {"settings": {"epochs": 10, "batch_size": 512}},
                "prepared_dataset.json": {
                    "sample_weights": {"formula": launcher.MODEL_FORMULAS[name]} if name != "normal" else None,
                    "augmentation_factor": 8,
                    "splits": {key: {"games": size} for key, size in
                               (("train", 120768), ("val", 6707), ("test", 6706))},
                },
            }
            for filename, record in records.items():
                (folder / filename).write_text(json.dumps(record), encoding="utf-8")
        engine = self.root / "engine.exe"
        engine.write_bytes(b"fake engine and evaluation data")
        self.args.egaroucid_exe = self.args.edax_exe = self.args.edax_eval_file = engine
        self.args.openings = self.root / "openings.txt"
        corpus = battle.BIN_DIR / "problem/xot/openingslarge.txt"
        with corpus.open() as stream:
            lines = [next(stream).strip() for _ in range(64)]
        self.args.openings.write_text("\n".join(lines), encoding="utf-8")
        self.openings = battle.read_openings(self.args.openings, self.args.seed)

    def specs(self, roster, normal_port=0, a_port=0):
        levels, alphas = battle.validate_args(self.args, roster.resource_alphas)
        original = battle.build_player_specs(self.args, levels, alphas, normal_port)
        return roster.add_specs(self.args, original, a_port, normal_port)

    def test_roster_includes_both_level_3_players_and_21750_paired_sets(self):
        roster = launcher.prepare(self.args)
        specs = self.specs(roster)
        expected = ([f"egaroucid_l{level}" for level in launcher.LEVELS]
                    + [f"normal_egaroucid_alpha_{a:.1f}" for a in launcher.ALPHAS]
                    + [f"edax_l{level}" for level in launcher.LEVELS]
                    + [f"loss_a_normal_alpha_{a:.1f}" for a in launcher.ALPHAS]
                    + [f"loss_b_loss_a_alpha_{a:.1f}" for a in launcher.ALPHAS] + ["random"])
        self.assertEqual(expected, [spec.name for spec in specs])
        self.assertEqual(30, len(set(expected)))
        for name in ("egaroucid_l3", "edax_l3"):
            command = next(spec.command for spec in specs if spec.name == name)
            self.assertEqual("3", command[command.index("-l") + 1])
        tasks = make_match_set_tasks(len(specs), self.openings, self.args.match_sets_per_pair, self.args.seed)
        self.assertEqual(21750, len(tasks))
        self.assertEqual(Counter({(i, j): 50 for i in range(30) for j in range(i + 1, 30)}),
                         Counter((t.p0_idx, t.p1_idx) for t in tasks))
        self.assertFalse(self.args.output_dir.exists())
        self.assertTrue(self.args.no_hint_cache)

    def test_all_pairs_record_both_colors_three_model_routing_and_resume(self):
        self.args.match_sets_per_pair = 3
        self.args.resume = True
        roster = launcher.prepare(self.args)
        models, played = [], []

        def start(settings, manager, run_id):
            models.append(Path(settings.policy_model))
            return battle.PolicyServerInfo(10000 + len(models), "tensorflow", "CPU", "test")

        def play(task, black, search_threads=None):
            played.append((task.p0_idx, task.p1_idx, black))
            return GameResult(task.p0_idx, task.p1_idx, black,
                              task.p0_idx if black else task.p1_idx, task.p1_idx if black else task.p0_idx,
                              2, 33 if black else 31, 31 if black else 33, task.opening)

        real_runner = battle.GameRunner

        def make_runner(*args, **kwargs):
            instance = real_runner(*args, **kwargs)
            instance.play_single_game = play
            return instance

        with patch.object(battle, "GameRunner", side_effect=make_runner) as runner, \
             patch.object(battle, "start_policy_server", side_effect=start), \
             patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), redirect_stdout(io.StringIO()):
            battle.run_tournament(self.args, extension=roster)
            specs = runner.call_args.args[0]
            raw = (self.args.output_dir / "strength_games.jsonl").read_bytes()
            battle.run_tournament(self.args, extension=launcher.prepare(self.args))
        self.assertEqual([self.args.policy_model, self.args.loss_a_model, self.args.loss_b_model], models)
        self.assertEqual(Counter({(i, j, color): 3 for i in range(30) for j in range(i + 1, 30)
                                  for color in (True, False)}), Counter(played))
        self.assertEqual(raw, (self.args.output_dir / "strength_games.jsonl").read_bytes())
        for spec in specs:
            command = spec.command
            if spec.name.startswith("normal_egaroucid_"):
                self.assertEqual("10001", command[command.index("--policy-server-port") + 1])
                self.assertIn("--logit-blend", command)
            elif spec.name.startswith(("loss_a_normal_", "loss_b_loss_a_")):
                ports = ("10001", "10002") if spec.name.startswith("loss_a_normal_") else ("10002", "10003")
                self.assertEqual(ports, (command[command.index("--normal-port") + 1],
                                         command[command.index("--weighted-port") + 1]))
                self.assertEqual(spec.name.rsplit("_", 1)[1], command[command.index("--beta") + 1])
                self.assertEqual("logits", command[command.index("--output") + 1])
        config = json.loads((self.args.output_dir / "strength_manifest.json").read_text())["configuration"]
        self.assertEqual(3, config["policy_inference"]["server_count"])
        self.assertEqual(30, config["participants"]["paper_roster"]["total_players"])
        self.assertFalse(config["schedule"]["same_opening_sequence_for_every_pair"])
        self.assertEqual("after_every_game_clear_or_discard_process", config["engine"]["tt_reset"])
        self.assertEqual("2 ** (disc_loss / 2)",
                         config["participants"]["paper_roster"]["training"]["loss_b"]["sample_weight_formula"])
        runtime = json.loads((self.args.output_dir / "strength_runtime_manifest.json").read_text())["configuration"]
        self.assertEqual(runtime["policy_server"], runtime["additional_policy_server"])
        self.assertEqual(runtime["policy_server"], runtime["extra_policy_servers"]["loss_b"])
        data = load_clustered_elo_data(self.args.output_dir)
        self.assertEqual(tuple(spec.name for spec in specs), data.names)
        self.assertTrue(data.independent_pairs)

    def test_dry_run_reserves_memory_for_all_three_policy_servers(self):
        self.args.dry_run = True
        with patch.object(battle, "available_memory_mib", return_value=1e9), \
             patch.object(battle, "validate_input_files"), \
             patch.object(battle, "start_policy_server") as start, redirect_stdout(io.StringIO()):
            battle.run_tournament(self.args, extension=launcher.prepare(self.args))
        start.assert_not_called()
        report = json.loads((self.args.output_dir / "strength_dry_run.json").read_text())
        expected = (3 * self.args.estimated_policy_server_memory_mib
                    + 4 * max(self.args.estimated_engine_memory_mib + self.args.estimated_wrapper_memory_mib,
                              self.args.estimated_edax_memory_mib)
                    + self.args.engine_thread_budget * self.args.estimated_search_thread_memory_mib)
        self.assertEqual(expected, report["estimated_max_tournament_memory_mib"])
        self.assertEqual(43500, report["total_actual_games"])

    def test_swapped_weighted_models_stop_before_output(self):
        self.args.loss_a_model, self.args.loss_b_model = self.args.loss_b_model, self.args.loss_a_model
        with self.assertRaisesRegex(ValueError, "Wrong training weight formula"):
            launcher.prepare(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_changed_training_conditions_stop_before_output(self):
        path = self.args.loss_b_model.parent / "selected_summary.json"
        summary = json.loads(path.read_text())
        summary["adopted_epoch"] = 7
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "10 epochs"):
            launcher.prepare(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_incompatible_levels_or_blend_formula_stop_before_output(self):
        for key, value in (("baseline_levels", "1,5,9,13,17,21"), ("edax_levels", "1,5,9,13,17,21"),
                           ("alphas", "0,0.2,0.4,0.6,0.8,1"), ("score_temperature", 2),
                           ("logit_blend", False), ("policy_backend", "numpy")):
            original = getattr(self.args, key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                setattr(self.args, key, value)
                launcher.prepare(self.args)
            setattr(self.args, key, original)
        self.assertFalse(self.args.output_dir.exists())

    def test_loss_b_runtime_must_match_normal_and_loss_a(self):
        roster = launcher.prepare(self.args)
        normal = battle.PolicyServerInfo(1, "tensorflow", "GPU", "test")
        a = battle.PolicyServerInfo(2, "tensorflow", "GPU", "test")
        roster.extra_policy_servers["loss_b"] = battle.PolicyServerInfo(3, "tensorflow", "CPU", "test")
        with self.assertRaisesRegex(ValueError, "loss_b"):
            roster.validate_runtime(normal, a)


if __name__ == "__main__":
    unittest.main()
