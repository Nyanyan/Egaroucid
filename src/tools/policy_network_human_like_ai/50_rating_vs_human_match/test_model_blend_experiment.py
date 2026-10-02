import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run_model_blend_experiment as experiment
from report_model_blend_experiment import anchored_elos, BASELINES, write_report
from game_dataset import BOARD_DTYPE, save_split_manifest, file_sha256
from wthor_human_match_evaluation import load_position_groups
from wthor_human_match_experiment import hint_experiment_identity
from blend_policy_with_egaroucid import BlendedPolicy, BoardState
from battle_blend_strength import make_argparser, validate_input_files


class ModelBlendExperimentTest(unittest.TestCase):
    def test_saved_test_membership_is_used_instead_of_legacy_random_split(self):
        state = BoardState.initial()
        legal = state.legal_policies(0)
        records = np.zeros(2, dtype=BOARD_DTYPE)
        records["player"] = state.black
        records["opponent"] = state.white
        records["policy"] = legal[:2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "0.dat"
            records.tofile(data)
            save_split_manifest(root, [data], {"train": np.array([0]), "val": np.array([], dtype=int),
                                             "test": np.array([1])}, {"unit": "game"})
            groups, count, _, _ = load_position_groups([data], 2, 1, "test", 999, 613, root / "dataset_split.json")
            self.assertEqual(1, count)
            self.assertEqual(((legal[1], 1),), groups[0].policy_counts)
            with self.assertRaisesRegex(ValueError, "exceeds saved"):
                load_position_groups([data], 2, 2, "test", 999, 613, root / "dataset_split.json")

    def test_shared_hints_ignore_model_but_preserve_sample_identity(self):
        first = {"model": "dense", "model_sha256": "a", "weights": None,
                 "split_manifest_sha256": "b", "sample_records_sha256": "sample", "workers": 16}
        second = dict(first, model="conv", model_sha256="c", split_manifest_sha256="d")
        self.assertEqual(hint_experiment_identity(first), hint_experiment_identity(second))
        self.assertNotEqual(hint_experiment_identity(first), hint_experiment_identity(dict(second, workers=12)))

    def test_logit_blending_preserves_a_winning_move_with_underflowed_probability(self):
        state = BoardState.initial()
        legal = state.legal_policies(0)
        logits = np.full((1, 64), -3000.0, dtype=np.float32)
        logits[0, legal[0]] = -1000
        logits[0, legal[1]] = 0
        blender = object.__new__(BlendedPolicy)
        blender.logit_blend = True
        blender.policy_client = Mock()
        blender.policy_client.predict.return_value = logits
        blender.score_temperature = 1.0
        scores = {p: -64.0 for p in legal}
        scores[legal[0]] = 64.0
        blender.cached_hint_scores = Mock(return_value=(scores, ""))
        mixed, _, _, _, _ = blender.blended_distribution(state, 0, 0.1)
        self.assertEqual(legal[0], int(np.argmax(mixed)))
        np.testing.assert_allclose(mixed.sum(), 1.0)
        blender.cached_hint_scores.reset_mock()
        pure, _, _, _, _ = blender.blended_distribution(state, 0, 1.0)
        self.assertEqual(legal[1], int(np.argmax(pure)))
        blender.cached_hint_scores.assert_not_called()

    def test_tensorflow_strength_does_not_require_mlp_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("engine.exe", "openings.txt", "model.h5"):
                (root / name).touch()
            args = make_argparser().parse_args(["--egaroucid-exe", str(root / "engine.exe"),
                "--openings", str(root / "openings.txt"), "--policy-model", str(root / "model.h5"),
                "--weights", str(root / "missing.bin"), "--policy-backend", "tensorflow"])
            with patch("battle_blend_strength.importlib.util.find_spec", return_value=object()):
                validate_input_files(args, [0.5])
            args.policy_backend = "numpy"
            with self.assertRaisesRegex(FileNotFoundError, "policy weights"):
                validate_input_files(args, [0.5])

    def test_anchor_is_applied_inside_each_bootstrap_replicate(self):
        points = {name: 1400.0 + 20 * i for i, name in enumerate(BASELINES)}
        points["alpha_1.0"] = 1800.0
        metadata = {"results": [{"name": name, "paired_set_elo": value} for name, value in points.items()],
                    "bootstrap": {"replicates": 3}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replicates.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(points))
                writer.writeheader()
                for offset in (-100, 0, 100):
                    writer.writerow({name: value + offset for name, value in points.items()})
            ratings = anchored_elos(metadata, path)
            np.testing.assert_allclose(ratings["alpha_1.0"], [1810, 1810, 1810])
            self.assertAlmostEqual(1500, np.mean([ratings[name][0] for name in BASELINES]))

    def test_changed_inputs_cannot_resume_and_incomplete_strength_is_not_marked_done(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = root / "launch_manifest.json"
            experiment.ensure_launcher_identity(identity, {"model": "a"})
            with self.assertRaisesRegex(ValueError, "changed"):
                experiment.ensure_launcher_identity(identity, {"model": "b"})
            (root / "strength").mkdir()
            (root / "strength" / "strength_progress.json").write_text(json.dumps({"remaining_match_sets": 1}))
            with patch.object(experiment, "run_logged"):
                with self.assertRaisesRegex(RuntimeError, "before completion"):
                    experiment.run_stage(["command"], [], root, "strength")
            self.assertFalse((root / "strength_completed.json").exists())

    def test_report_joins_both_models_including_alpha_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = list(BASELINES) + [f"alpha_{float(a)}" for a in (0, .2, .4, .6, .8, 1)]
            for model in ("dense512", "conv8_bn"):
                human = root / model / "human"
                strength = root / model / "strength"
                human.mkdir(parents=True)
                strength.mkdir()
                base = {"positions": 100, "top1_accuracy": .5, "top1_ci95_lower": .4, "top1_ci95_upper": .6,
                        "top3_accuracy": .8, "top3_ci95_lower": .7, "top3_ci95_upper": .9}
                measurement = {"sample_global_indices_sha256": "indices", "sample_records_sha256": "records",
                               "results": [dict(base, alpha=a) for a in (0, .2, .4, .6, .8, 1)],
                               "console_level_results": [dict(base, level=level) for level in range(1, 22, 2)]}
                (human / "random_wthor_blend_summary.json").write_text(json.dumps(measurement))
                sources = {}
                for filename in ("strength_games.jsonl", "strength_results.json", "strength_manifest.json"):
                    (strength / filename).write_text("test fixture")
                    sources[filename] = file_sha256(strength / filename)
                draws = strength / "strength_elo_bootstrap_replicates.csv"
                points = {name: float(1000 + 50 * i) for i, name in enumerate(names)}
                with draws.open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=names)
                    writer.writeheader()
                    for offset in (-10, 0, 10):
                        writer.writerow({name: value + offset for name, value in points.items()})
                metadata = {"results": [{"name": name, "paired_set_elo": value} for name, value in points.items()],
                            "bootstrap": {"replicates": 3}, "source_sha256": sources,
                            "generated_sha256": {draws.name: file_sha256(draws)}}
                (strength / "strength_elo_ci95.json").write_text(json.dumps(metadata))
            write_report(root, ["dense512", "conv8_bn"])
            stem = root / "rating_vs_human_match_dense512_conv8_bn"
            with stem.with_suffix(".csv").open(newline="") as output:
                rows = list(csv.DictReader(output))
            self.assertEqual(32, len(rows))
            self.assertEqual(2, sum(row["participant"] == "alpha_0.0" for row in rows))
            self.assertTrue(stem.with_suffix(".png").stat().st_size > 1000)
            self.assertTrue(stem.with_suffix(".svg").stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main()
