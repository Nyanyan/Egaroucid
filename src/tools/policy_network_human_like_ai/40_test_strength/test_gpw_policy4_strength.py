"""Verify four-policy model routing, paired games, and reproducible resume."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import battle_blend_strength as battle
import run_gpw_policy4_strength as launcher
from strength_tournament import GameResult, make_match_set_tasks


class FourPolicyTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = launcher.make_argparser().parse_args([
            "--output-dir", str(self.root / "results"), "--no-performance-monitor",
            "--sequential-tail-match-sets", "0",
        ])
        for argument in ("policy_model", "loss_a_model", "loss_b_model", "egaroucid_exe"):
            path = self.root / argument
            path.write_bytes(argument.encode())
            setattr(self.args, argument, path)
        self.args.openings = self.root / "openings.txt"
        with (battle.BIN_DIR / "problem/xot/openingslarge.txt").open() as stream:
            lines = [next(stream).strip() for _ in range(64)]
        self.args.openings.write_text("\n".join(lines), encoding="utf-8")
        # Training metadata validation is shared with, and tested by, the GPW runner.
        evidence = patch.object(launcher.gpw, "model_evidence", return_value={"test": True})
        evidence.start()
        self.addCleanup(evidence.stop)

    def test_full_round_robin_routes_three_models_and_resumes_without_replay(self):
        self.args.resume = True
        models, played = [], []

        def start(settings, manager, run_id):
            models.append(settings.policy_model)
            return battle.PolicyServerInfo(10000 + len(models), "tensorflow", "CPU", "test")

        def play(task, black, search_threads=None):
            played.append((task.p0_idx, task.p1_idx, black))
            return GameResult(task.p0_idx, task.p1_idx, black,
                              task.p0_idx if black else task.p1_idx,
                              task.p1_idx if black else task.p0_idx,
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
            battle.run_tournament(self.args, extension=launcher.prepare(self.args))
            specs = runner.call_args.args[0]
            raw = (self.args.output_dir / "strength_games.jsonl").read_bytes()
            battle.run_tournament(self.args, extension=launcher.prepare(self.args))

        self.assertEqual([self.args.policy_model, self.args.loss_a_model, self.args.loss_b_model], models)
        self.assertEqual(["egaroucid_l21", "normal", "loss_a", "loss_b"], [s.name for s in specs])
        for spec, option, port in ((specs[1], "--policy-server-port", "10001"),
                                   (specs[2], "--weighted-port", "10002"),
                                   (specs[3], "--weighted-port", "10003")):
            self.assertEqual(port, spec.command[spec.command.index(option) + 1])
            self.assertEqual(1.0, spec.alpha)
        self.assertEqual(Counter({(i, j, color): 50 for i in range(4) for j in range(i + 1, 4)
                                  for color in (True, False)}), Counter(played))
        self.assertEqual(raw, (self.args.output_dir / "strength_games.jsonl").read_bytes())
        config = json.loads((self.args.output_dir / "strength_manifest.json").read_text())["configuration"]
        self.assertEqual(4, config["participants"]["paper_roster"]["total_players"])
        self.assertEqual(3, config["policy_inference"]["server_count"])
        self.assertEqual(58, config["schedule"]["xot_seed"])
        self.assertFalse(config["schedule"]["same_opening_sequence_for_every_pair"])

    def test_new_seed_resamples_fifty_unique_openings_per_pair(self):
        def schedule(count, seed):
            return make_match_set_tasks(count, battle.read_openings(self.args.openings, seed), 50, seed)

        tasks = schedule(4, self.args.seed)
        self.assertEqual(tasks, schedule(4, self.args.seed))
        old = schedule(30, launcher.gpw.make_argparser().parse_args([]).seed)
        for i in range(4):
            for j in range(i + 1, 4):
                selected = [t.opening for t in tasks if (t.p0_idx, t.p1_idx) == (i, j)]
                self.assertEqual(50, len(set(selected)))
                self.assertNotEqual(selected, [t.opening for t in old if (t.p0_idx, t.p1_idx) == (i, j)])

    def test_extra_participants_rejected(self):
        self.args.edax_levels = "21"
        with self.assertRaisesRegex(ValueError, "Four-policy roster"):
            launcher.prepare(self.args)


if __name__ == "__main__":
    unittest.main()
