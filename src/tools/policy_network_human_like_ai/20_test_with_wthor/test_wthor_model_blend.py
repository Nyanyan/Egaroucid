import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_wthor_model_blend as evaluation
from evaluate_wthor_blend_human_match import BOARD_DTYPE, equivalent_targets
from wthor_human_match_evaluation import BoardState, load_position_groups, prepare_position_groups


class ModelBlendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.reference = self.root / "reference"
        self.reference.mkdir()
        state = BoardState.initial()
        records = []
        # Initial symmetry, multiple labels, duplicates, black and white to move.
        for step in range(9):
            legal = state.legal_policies(state.side)
            for policy in (legal[0], legal[-1], legal[-1]):
                record = np.zeros(1, dtype=BOARD_DTYPE)
                record["player"], record["opponent"] = state.player_opponent_bits(state.side)
                record["color"], record["policy"] = state.side, policy
                records.append(record)
            state.apply_move(state.side, legal[step % len(legal)])
        self.records = np.concatenate(records)
        self.dat = self.data / "0.dat"
        self.records.tofile(self.dat)
        self.groups, _, indices_hash, records_hash = load_position_groups(
            [self.dat], len(self.records), len(self.records), "all", 613, 613,
        )
        self.identity = {
            "positions": len(self.records), "data_split": "all", "split_seed": 613, "sample_seed": 613,
            "sample_global_indices_sha256": indices_hash, "sample_records_sha256": records_hash,
        }
        self.summary = {
            "experiment_identity": self.identity, "board_data_dir": str(self.data),
            "evaluated_positions": len(self.records), "unique_states": len(self.groups),
            "split_positions": len(self.records),
            "sample_global_indices_sha256": indices_hash, "sample_records_sha256": records_hash,
        }
        self.save_reference()
        self.normal_model, self.weighted_model = self.root / "normal.h5", self.root / "weighted.h5"
        self.normal_model.write_bytes(b"normal")
        self.weighted_model.write_bytes(b"weighted")
        self.args = evaluation.make_argparser().parse_args([
            "--reference-dir", str(self.reference),
            "--normal-model", str(self.normal_model), "--weighted-model", str(self.weighted_model),
            "--output-dir", str(self.root / "output"),
        ])
        rng = np.random.default_rng(72)
        self.normal = {g.key: rng.normal(size=64) for g in self.groups}
        self.weighted = {g.key: rng.normal(size=64) for g in self.groups}

    def save_reference(self):
        (self.reference / "experiment_identity.json").write_text(json.dumps(self.identity))
        (self.reference / "random_wthor_blend_summary.json").write_text(json.dumps(self.summary))

    def run_evaluation(self, prediction=None):
        with patch.object(evaluation, "predict_policy_logits", side_effect=prediction or [self.normal, self.weighted]), patch(
            "subprocess.Popen", side_effect=AssertionError("No engines may start"),
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return evaluation.run(self.args)

    def test_counts_match_independent_probability_product_with_duplicates_and_symmetry(self):
        # Deliberately dominant illegal entries must never win.
        self.normal[self.groups[0].key][0] = 1000
        self.weighted[self.groups[0].key][0] = 1000
        self.args.betas = (0, .2, .4, .6, .8, 1)
        self.args.top3 = True
        result = self.run_evaluation()
        for row in result["results"]:
            hits = {1: 0, 3: 0}
            for group in self.groups:
                state = BoardState(*group.key)
                legal = state.legal_policies(group.side)
                # Compute the geometric mean from probabilities, independently of the evaluator.
                p = np.exp(self.normal[group.key][legal])
                q = np.exp(self.weighted[group.key][legal])
                p, q = p / p.sum(), q / q.sum()
                mixed = p ** (1 - row["beta"]) * q ** row["beta"]
                ranked = [legal[i] for i in np.argsort(-mixed, kind="stable")]
                player, opponent = state.player_opponent_bits(group.side)
                for label, count in group.policy_counts:
                    targets = set(equivalent_targets(player, opponent, label))
                    for n in hits:
                        hits[n] += count * bool(set(ranked[:n]) & targets)
            self.assertEqual(len(self.records), row["positions"])
            for n in hits:
                self.assertEqual(hits[n], row[f"top{n}_hits"])
                self.assertEqual(hits[n] / len(self.records), row[f"top{n}_accuracy"])
        # No SQLite cache was created or required.
        self.assertFalse(list(self.reference.glob("*.sqlite3*")))
        self.assertEqual(0, result["engine_searches"])
        self.assertEqual(self.identity["sample_records_sha256"], result["sample_records_sha256"])

    def test_top1_only_outputs_and_tied_symmetry_match(self):
        tied = {g.key: np.zeros(64) for g in self.groups}
        result = self.run_evaluation([tied, tied])
        self.assertEqual([.2, .4, .6, .8], list(result["betas"]))
        self.assertEqual([f"model_blend_beta_{b:g}" for b in result["betas"]],
                         [row["model"] for row in result["results"]])
        csv = (self.args.output_dir / "wthor_model_blend_summary.csv").read_text()
        self.assertIn("top1_accuracy", csv)
        self.assertNotIn("top3", csv)
        saved = json.loads((self.args.output_dir / "wthor_model_blend_summary.json").read_text())
        self.assertEqual(result["results"], saved["results"])
        first = prepare_position_groups(self.groups[:1])
        row = evaluation.evaluate_predictions(first, tied, tied, (.2,))[0]
        self.assertEqual(3, row["top1_hits"])  # All initial moves are symmetry-equivalent.

    def test_dry_run_checks_sample_and_writes_nothing(self):
        self.args.dry_run = True
        with patch.object(evaluation, "predict_policy_logits") as predict:
            evaluation.run(self.args)
        predict.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_data_changes_fail_before_inference(self):
        self.records[0]["policy"] = 0
        self.records.tofile(self.dat)
        with patch.object(evaluation, "predict_policy_logits") as predict:
            with self.assertRaisesRegex(ValueError, "sample_records_sha256"):
                evaluation.run(self.args)
        predict.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_reference_identity_and_count_mismatches_are_rejected(self):
        self.summary["experiment_identity"] = {**self.identity, "sample_seed": 42}
        self.save_reference()
        with self.assertRaisesRegex(ValueError, "identity.json differ"):
            evaluation.run(self.args)
        self.summary["experiment_identity"] = self.identity
        self.summary["evaluated_positions"] -= 1
        self.save_reference()
        with self.assertRaisesRegex(ValueError, "position count differs"):
            evaluation.run(self.args)

    def test_existing_output_is_preserved(self):
        self.args.output_dir.mkdir()
        saved = self.args.output_dir / "previous.csv"
        saved.write_text("keep")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            evaluation.run(self.args)
        self.assertEqual("keep", saved.read_text())

    def test_models_must_exist_and_remain_unchanged(self):
        self.normal_model.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Model not found"):
            evaluation.run(self.args)
        self.normal_model.write_bytes(b"normal")

        def changed(groups, weights, batch_size, model):
            self.normal_model.write_bytes(b"changed")
            return self.normal

        with self.assertRaisesRegex(ValueError, "Model changed during inference"):
            self.run_evaluation(changed)
        self.assertFalse(self.args.output_dir.exists())

    def test_invalid_betas_and_nonfinite_model_outputs(self):
        for value in ("", "nan", "inf", "-0.2", "1.1", "0.2,0.2", "0.2,", "bad"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                evaluation.parse_betas(value)
        self.assertEqual((0, .5, 1), evaluation.parse_betas("0, 0.5,1"))
        legal = BoardState(*self.groups[0].key).legal_policies(self.groups[0].side)
        self.normal[self.groups[0].key][legal[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.run_evaluation()
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
