"""Check GPW blend direction, reference agreement rules and random sampling."""
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_gpw_human_match as evaluation
from test_mistake_test_models import MistakeTest as Fixture
from wthor_human_match_evaluation import evaluate_agreement


class GPWHumanMatchTest(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.fixture.setUp()
        self.logits = dict(normal=self.fixture.normal, loss_a=self.fixture.weighted,
                           loss_b={key: -value for key, value in self.fixture.normal.items()})
        self.hints = {(engine, level): self.fixture.hints[21]
                      for engine in ("egaroucid", "edax") for level in evaluation.LEVELS}
        self.dataset = dict(name="mistakes", groups=self.fixture.groups, records=self.fixture.records,
                            losses=self.fixture.losses, source_indices=np.arange(len(self.fixture.records)))

    def compute(self, dataset=None):
        ds = dataset or self.dataset
        prepared = evaluation.prepare_position_groups(ds["groups"])
        state_moves = evaluation.predict_states(ds["groups"], prepared, self.logits, self.hints)
        moves, targets = evaluation.expand_predictions(ds, prepared, state_moves, 57)
        return evaluation.summarize_and_verify(ds, moves, targets), moves, targets, state_moves

    def test_three_families_match_probability_formula_and_engine_tie_order(self):
        _, _, _, moves = self.compute()
        self.assertEqual(30, len(set(evaluation.MODEL_NAMES)))
        for i, group in enumerate(self.dataset["groups"]):
            legal = self.fixture.prepared.groups_by_key[group.key][0].legal
            for pi1, pi2 in evaluation.FAMILIES:
                one = self.logits[pi1][group.key][list(legal)]
                two = (np.array([self.hints["egaroucid", 21][group.key].scores[m] for m in legal])
                       if pi2 == "egaroucid_l21" else self.logits[pi2][group.key][list(legal)])
                p, q = np.exp(one - max(one)), np.exp(two - max(two))
                p, q = p / p.sum(), q / q.sum()
                prefix = "normal_egaroucid" if pi1 == "normal" else f"{pi1}_{pi2}"
                for alpha in evaluation.ALPHAS:
                    col = evaluation.MODEL_NAMES.index(f"{prefix}_alpha_{alpha:.1f}")
                    self.assertEqual(legal[np.argmax(p ** alpha * q ** (1 - alpha))], moves[i, col])
            for engine in ("egaroucid", "edax"):
                for level in evaluation.LEVELS:
                    col = evaluation.MODEL_NAMES.index(f"{engine}_l{level}")
                    self.assertEqual(self.hints[engine, level][group.key].move_order[0], moves[i, col])

    def test_threshold_counts_and_normal_blends_match_precedent(self):
        rows, _, _, _ = self.compute()
        self.assertEqual(180, len(rows))
        actual = {(r["model"], r["minimum_loss"]): r for r in rows}
        for threshold in evaluation.THRESHOLDS:
            groups = self.fixture.selected_groups(threshold)
            expected, _, invalid, illegal = evaluate_agreement(groups, self.fixture.hints, self.logits["normal"], 1., (1,))
            self.assertEqual((0, 0), (invalid, illegal))
            for alpha in evaluation.ALPHAS:
                row = actual[f"normal_egaroucid_alpha_{alpha:.1f}", threshold]
                self.assertEqual(expected[alpha]["hits"][1], row["top1_hits"])
                self.assertEqual(int((self.fixture.losses >= threshold).sum()), row["positions"])
        self.assertTrue(any(r["top1_hits"] > r["exact_top1_hits"] for r in rows))

    def test_random_preserves_precedent_order_and_duplicate_occurrences(self):
        for sample in (False, True):
            ds = dict(self.dataset, losses=None, name="wthor_test_100000") if sample else self.dataset
            rows, moves, _, _ = self.compute(ds)
            self.assertEqual(30 if sample else 180, len(rows))
            keys = [evaluation.record_key(row) for row in ds["records"]]
            order = (sorted(range(len(keys)), key=lambda i: (keys[i], int(ds["records"][i]["policy"])))
                     if sample else range(len(keys)))
            rng = random.Random(57)
            for i in order:
                legal = sorted(self.fixture.prepared.groups_by_key[keys[i]][0].legal)
                self.assertEqual(rng.choice(legal), moves[i, -1])

    def test_illegal_moves_and_incorrect_symmetry_targets_are_rejected(self):
        _, moves, targets, _ = self.compute()
        invalid = moves.copy()
        invalid[0, 0] = 64
        with self.assertRaisesRegex(ValueError, "invalid predictions"):
            evaluation.summarize_and_verify(self.dataset, invalid, targets)
        with self.assertRaisesRegex(ValueError, "symmetry"):
            evaluation.summarize_and_verify(self.dataset, moves, np.zeros_like(targets))

    def test_incomplete_reference_cache_is_rejected_without_search(self):
        with patch.object(evaluation, "ReadOnlyHintCache") as cache:
            cache.return_value.load.return_value = {}
            with self.assertRaisesRegex(ValueError, "Incomplete reference cache"):
                evaluation.reference_hints(self.dataset, Path("unused"))
            cache.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
