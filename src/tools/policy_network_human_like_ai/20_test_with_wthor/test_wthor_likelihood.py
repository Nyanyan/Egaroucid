import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_wthor_human_match as policy_evaluator
from wthor_human_match_evaluation import (
    BLEND_PARAMS, BoardState, HintData, IncrementalAgreementMetrics,
    PositionGroup, agreement_results_equal, make_blend_summary_rows, metrics_to_result,
)
from wthor_likelihood import legal_log_probabilities, likelihood_summary


class LikelihoodTest(unittest.TestCase):
    def test_exact_move_likelihood_is_not_symmetry_agreement(self):
        state = BoardState.initial()
        legal = state.legal_policies(state.side)
        # Three observations of the 10% move, one of the 40% move, and
        # invalid labels which must not enter either denominator.
        group = PositionGroup(state.black, state.white, state.side,
                              ((legal[0], 3), (legal[-1], 1), (-1, 2), (0, 5)))
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        logits = np.full(64, 9999.0)  # Illegal logits must be ignored.
        logits[legal] = np.log(probabilities)
        hint = HintData({p: 0.0 for p in legal}, tuple(legal))
        metrics = IncrementalAgreementMetrics([group], {group.key: logits}, 1.0)
        metrics.add_hint(21, group.key, hint)
        self.assertFalse(metrics.add_hint(21, group.key, hint))
        blends, _, invalid, illegal = metrics.result()
        self.assertEqual((2, 5), (invalid, illegal))
        rows = make_blend_summary_rows(metrics_to_result(blends))
        for row, alpha in zip(rows, BLEND_PARAMS):
            expected = probabilities ** alpha
            expected /= expected.sum()
            mean = (3 * expected[0] + expected[-1]) / 4
            log_mean = (3 * math.log(expected[0]) + math.log(expected[-1])) / 4
            self.assertEqual(4, row["positions"])
            self.assertEqual(1.0, row["top1_accuracy"])
            self.assertAlmostEqual(mean, row["mean_likelihood"])
            self.assertAlmostEqual(100 * mean, row["mean_likelihood_percent"])
            self.assertAlmostEqual(log_mean, row["mean_log_likelihood"])
            self.assertAlmostEqual(-log_mean, row["negative_log_likelihood"])
        self.assertAlmostEqual(17.5, rows[-1]["mean_likelihood_percent"])

    def test_score_temperature_and_log_space_avoid_underflow(self):
        state = BoardState.initial()
        legal = state.legal_policies(state.side)
        group = PositionGroup(state.black, state.white, state.side, ((legal[-1], 1),))
        hint = HintData(dict(zip(legal, (0.0, -2.0, -4.0, -20000.0))), tuple(legal))
        metric = IncrementalAgreementMetrics([group], {group.key: np.zeros(64)}, 2.0)
        metric.add_hint(21, group.key, hint)
        row = make_blend_summary_rows(metrics_to_result(metric.result()[0]))[0]
        expected_nll = 10000 + math.log(1 + math.exp(-1) + math.exp(-2))
        self.assertEqual(0.0, row["mean_likelihood"])
        self.assertAlmostEqual(expected_nll, row["negative_log_likelihood"])
        self.assertEqual(0.0, legal_log_probabilities(np.zeros(64), [7])[7])
        self.assertIsNone(likelihood_summary(0.0, 0.0, 0)["mean_likelihood"])
        for temperature in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                IncrementalAgreementMetrics([], {}, temperature)

    def test_live_comparison_tolerates_only_floating_roundoff(self):
        first = ({1.0: {"positions": 3, "hits": {1: 2, 3: 3},
                        "likelihood_sum": 1.1, "log_likelihood_sum": -4.0}}, {}, 0, 0)
        second = copy.deepcopy(first)
        second[0][1.0]["likelihood_sum"] += 1e-14
        self.assertTrue(agreement_results_equal(first, second))
        second[0][1.0]["hits"][1] += 1
        self.assertFalse(agreement_results_equal(first, second))

    def test_policy_only_csv_json_normalize_legal_moves(self):
        state = BoardState.initial()
        legal = state.legal_policies(state.side)
        player, opponent = state.player_opponent_bits(state.side)
        records = np.zeros(2, dtype=policy_evaluator.BOARD_DTYPE)
        records["player"], records["opponent"] = player, opponent
        records["policy"] = [legal[0], legal[-1]]
        probabilities = np.full((2, 64), 0.5)
        probabilities[:, legal] = [0.01, 0.02, 0.03, 0.04]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records.tofile(root / "0.dat")
            args = policy_evaluator.parse_args([
                "--board-data-dir", str(root), "--weights", "unused.bin",
                "--output-dir", str(root / "results"),
            ])
            with patch.object(policy_evaluator, "BinaryPolicyNetwork"), patch.object(
                policy_evaluator, "predict_batch", return_value=probabilities,
            ):
                result = policy_evaluator.evaluate(args)
            self.assertAlmostEqual(25.0, result["likelihood"]["mean_likelihood_percent"])
            self.assertAlmostEqual(-math.log(0.1 * 0.4) / 2, result["likelihood"]["negative_log_likelihood"])
            saved = json.loads((root / "results/wthor_human_match.json").read_text())
            self.assertEqual(result["likelihood"], saved["likelihood"])
            for name in ("wthor_human_match_topn.csv", "wthor_human_match_by_move10.csv"):
                with (root / "results" / name).open() as f:
                    row = next(csv.DictReader(f))
                self.assertAlmostEqual(25.0, float(row["mean_likelihood_percent"]))


if __name__ == "__main__":
    unittest.main()
