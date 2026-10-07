"""Regression checks against previous blend evaluators, including all endpoints."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_loss_plus_one_blends as evaluation
import test_mistake_test_models as fixtures
from evaluate_wthor_model_blend import evaluate_predictions
from wthor_human_match_evaluation import HintData, evaluate_agreement, prepare_position_groups


class LossPlusOneBlendTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MistakeTest()
        self.fixture.setUp()
        self.logits = {"new": self.fixture.weighted, "normal": self.fixture.normal,
                       "exponential": {key: -values for key, values in self.fixture.normal.items()}}
        self.dataset = dict(name="mistakes", groups=self.fixture.groups, records=self.fixture.records,
                            losses=self.fixture.losses, source_indices=np.arange(len(self.fixture.records)),
                            hints=self.fixture.hints[21])

    def compute(self, dataset=None):
        dataset = dataset or self.dataset
        prepared = prepare_position_groups(dataset["groups"])
        moves = evaluation.predict_blends(dataset["groups"], prepared, self.logits, dataset["hints"], 1.0)
        return evaluation.summarize_predictions(dataset, prepared, moves)

    def test_all_alphas_thresholds_match_previous_evaluators(self):
        rows, moves, targets = self.compute()
        self.assertEqual(108, len(rows))
        evaluation.verify_predictions(self.fixture.records, self.fixture.losses, moves, targets, rows)
        actual = {(r["partner"], r["alpha"], r["minimum_loss"]): r for r in rows}
        for threshold in evaluation.THRESHOLDS:
            groups = self.fixture.selected_groups(threshold)
            expected, _, invalid, illegal = evaluate_agreement(groups, self.fixture.hints, self.logits["new"], 1.0, (1,))
            self.assertEqual((0, 0), (invalid, illegal))
            for alpha, metric in expected.items():
                self.assertEqual(metric["hits"][1], actual["egaroucid_l21", alpha, threshold]["top1_hits"])
            prepared = prepare_position_groups(groups)
            for partner in ("normal", "exponential"):
                with contextlib.redirect_stdout(io.StringIO()):
                    expected_rows = evaluate_predictions(prepared, self.logits[partner], self.logits["new"], evaluation.ALPHAS)
                for row in expected_rows:
                    self.assertEqual(row["top1_hits"], actual[partner, row["beta"], threshold]["top1_hits"])
        self.assertTrue(any(r["top1_hits"] > r["exact_top1_hits"] for r in rows))

    def test_sample_expansion_keeps_duplicate_labels_and_both_sides(self):
        records = evaluation.records_from_groups(self.fixture.groups)
        self.assertEqual(len(self.fixture.records), len(records))
        np.testing.assert_array_equal(self.fixture.records, records)
        sample = dict(self.dataset, name="wthor_test_100000", records=records, losses=None, source_indices=None)
        rows, moves, targets = self.compute(sample)
        self.assertEqual(18, len(rows))
        self.assertTrue(all(row["positions"] == len(records) for row in rows))
        evaluation.verify_predictions(records, np.empty(0), moves, targets, rows)

    def test_native_engine_tie_order_and_geometric_probability_formula(self):
        prepared = self.fixture.prepared
        moves = evaluation.predict_blends(self.fixture.groups, prepared, self.logits, self.dataset["hints"], 1.0)
        for i, group in enumerate(self.fixture.groups):
            legal = prepared.groups_by_key[group.key][0].legal
            self.assertEqual(self.dataset["hints"][group.key].move_order[0], moves[i, 0])
            for j, config in enumerate(evaluation.CONFIGS):
                if config["partner"] == "egaroucid_l21":
                    continue
                alpha = config["alpha"]
                p = np.exp(self.logits["new"][group.key][list(legal)])
                q = np.exp(self.logits[config["partner"]][group.key][list(legal)])
                probability = (p / p.sum()) ** alpha * (q / q.sum()) ** (1 - alpha)
                self.assertEqual(legal[int(np.argmax(probability))], moves[i, j])

    def test_missing_hints_and_nonfinite_logits_are_rejected(self):
        key = self.fixture.groups[0].key
        hint = self.dataset["hints"][key]
        self.dataset["hints"][key] = HintData({}, ())
        with self.assertRaises(ValueError):
            self.compute()
        self.dataset["hints"][key] = hint
        self.logits["new"][key][list(hint.scores)] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.compute()

    def test_smoke_preserves_occurrences_and_empty_subsets_are_explicit(self):
        subset = evaluation.limit_smoke_states(self.dataset, 2)
        self.assertEqual(6, len(subset["records"]))
        np.testing.assert_array_equal(np.arange(6), subset["source_indices"])
        subset["losses"] = np.ones(6, dtype="u1")
        rows, moves, targets = self.compute(subset)
        for row in rows:
            if row["minimum_loss"] > 1:
                self.assertEqual(0, row["positions"])
                self.assertIsNone(row["top1_accuracy"])
        evaluation.verify_predictions(subset["records"], subset["losses"], moves, targets, rows)
        moves[0, 0] = 64
        with self.assertRaisesRegex(ValueError, "invalid prediction"):
            evaluation.verify_predictions(subset["records"], subset["losses"], moves, targets, rows)

    def test_runner_dry_run_and_verified_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in self.logits:
                (root / f"{name}.h5").write_bytes(name.encode())
            args = evaluation.make_argparser().parse_args([
                "--new-model", str(root / "new.h5"), "--normal-model", str(root / "normal.h5"),
                "--exponential-model", str(root / "exponential.h5"), "--output-dir", str(root / "output")])
            sample = dict(self.dataset, name="wthor_test_100000", losses=None, source_indices=None)
            provenance = {"sample_identity": {"score_temperature": 1.0}, "dataset_counts": []}
            with patch.object(evaluation, "load_inputs", return_value=([sample, self.dataset], provenance)), \
                 patch.object(evaluation, "predict_policy_logits", side_effect=lambda groups, weights, batch, model: self.logits[model.stem]) as infer, \
                 contextlib.redirect_stdout(io.StringIO()):
                args.dry_run = True
                evaluation.run(args)
                infer.assert_not_called()
                self.assertFalse(args.output_dir.exists())
                args.dry_run = False
                result = evaluation.run(args)
                self.assertEqual(3, infer.call_count)
                self.assertEqual(126, len(result["results"]))
                self.assertEqual(0, result["engine_searches"])
                self.assertEqual("complete", json.loads((args.output_dir / "run_status.json").read_text())["state"])
                self.assertEqual("passed", json.loads((args.output_dir / "verification.json").read_text())["status"])
                self.assertEqual(19, len((args.output_dir / "accuracy_matrix.csv").read_text().splitlines()))
                with self.assertRaisesRegex(ValueError, "not empty"):
                    evaluation.run(args)


if __name__ == "__main__":
    unittest.main()
