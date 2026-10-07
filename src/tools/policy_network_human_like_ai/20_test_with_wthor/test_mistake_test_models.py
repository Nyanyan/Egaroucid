"""Independent threshold, duplicate, symmetry, tie, and reference-cache checks."""
from collections import Counter
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_mistake_test_models as evaluation
from prepare_mistake_test_data import export_subsets
from wthor_dataset import BOARD_DTYPE
from wthor_human_match_evaluation import (
    BoardState, HintData, PositionGroup, evaluate_agreement, evaluate_engine_agreement,
    prepare_position_groups,
)


class MistakeTest(unittest.TestCase):
    def setUp(self):
        state = BoardState.initial()
        records = []
        groups = []
        self.hints = {level: {} for level in evaluation.CONSOLE_LEVELS}
        self.normal, self.weighted = {}, {}
        rng = np.random.default_rng(317)
        for step in range(12):
            legal = state.legal_policies(state.side)
            labels = (legal[0], legal[-1], legal[-1])
            key = (state.black, state.white, state.side)
            groups.append(PositionGroup(*key, tuple(Counter(labels).items())))
            for label in labels:
                player, opponent = state.player_opponent_bits(state.side)
                records.append((player, opponent, state.side, label, 0))
            self.normal[key] = rng.normal(size=64) if step else np.zeros(64)
            self.weighted[key] = rng.normal(size=64) if step else np.zeros(64)
            # Equal engine scores but reverse legal tie order.
            for level in self.hints:
                self.hints[level][key] = HintData({move: 0.0 for move in legal}, tuple(reversed(legal)))
            state.apply_move(state.side, legal[step % len(legal)])
        self.records = np.array(records, dtype=BOARD_DTYPE)
        self.losses = np.resize(np.array([1, 2, 3, 4, 5, 6, 7, 128], dtype="u1"), len(records))
        self.groups = groups
        self.prepared = prepare_position_groups(groups)

    def metrics(self):
        return evaluation.MistakeAgreement(self.groups, self.prepared, self.records, self.losses,
                                           self.normal, self.weighted, 1.0, 57)

    def selected_groups(self, threshold):
        counts = {}
        for row in self.records[self.losses >= threshold]:
            player, opponent, side, label = [int(row[f]) for f in ("player", "opponent", "color", "policy")]
            key = (player, opponent, side) if side == 0 else (opponent, player, side)
            counts.setdefault(key, Counter())[label] += 1
        return [PositionGroup(*key, tuple(c.items())) for key, c in counts.items()]

    def test_all_thresholds_match_existing_evaluators_with_ties_duplicates_and_both_colors(self):
        metrics = self.metrics()
        before = metrics.rows()
        self.assertEqual(28, sum(row["status"] == "complete" for row in before))
        for engine in ("egaroucid", "edax"):
            for level, hints in self.hints.items():
                metrics.accept(engine, level, list(hints.items()))
                metrics.accept(engine, level, list(hints.items()))  # idempotent replay
        actual = {(r["model"], r["minimum_loss"]): r for r in metrics.rows()}
        self.assertEqual(108, len(actual))
        for threshold in evaluation.THRESHOLDS:
            groups = self.selected_groups(threshold)
            count = int((self.losses >= threshold).sum())
            normal, console, invalid, illegal = evaluate_agreement(groups, self.hints, self.normal, 1.0, (1,))
            weighted, _, _, _ = evaluate_agreement(groups, self.hints, self.weighted, 1.0, (1,))
            edax = evaluate_engine_agreement(groups, self.hints, (1,))
            self.assertEqual((0, 0), (invalid, illegal))
            expected = {f"alpha_{a:.1f}": m for a, m in normal.items()}
            expected.update({f"loss_weighted_alpha_{a:.1f}": m for a, m in weighted.items() if a > 0})
            expected.update({f"egaroucid_l{level}": m for level, m in console.items() if level != 21})
            expected.update({f"edax_l{level}": m for level, m in edax.items()})
            for name, expected_row in expected.items():
                row = actual[name, threshold]
                self.assertEqual(count, row["positions"])
                self.assertEqual(expected_row["hits"][1], row["top1_hits"], (name, threshold))
            # Independently compute beta mixtures as products of probabilities.
            for beta in evaluation.BETAS:
                hits = 0
                for group in groups:
                    item = self.prepared.groups_by_key[group.key][0]
                    legal = item.legal
                    p = np.exp(self.normal[group.key][list(legal)])
                    q = np.exp(self.weighted[group.key][list(legal)])
                    values = (p / p.sum()) ** (1 - beta) * (q / q.sum()) ** beta
                    move = legal[int(np.argmax(values))]
                    targets = {p: t for p, t, _ in item.valid_targets}
                    hits += sum(count for label, count in group.policy_counts if move in targets[label])
                self.assertEqual(hits, actual[f"model_blend_beta_{beta:g}", threshold]["top1_hits"])
        self.assertTrue(all(r["status"] == "complete" for r in actual.values()))
        self.assertTrue(any(r["top1_hits"] > r["exact_top1_hits"] for r in actual.values()))

    def test_random_draws_are_reproducible_and_shared_across_nested_subsets(self):
        first, second = self.metrics(), self.metrics()
        np.testing.assert_array_equal(first.random_moves, second.random_moves)
        for row in first.rows():
            if row["model"] != "random":
                continue
            selected = self.losses >= row["minimum_loss"]
            hits = sum(bool(int(mask) & (1 << int(move))) for mask, move in
                       zip(first.targets[selected], first.random_moves[selected]))
            self.assertEqual(hits, row["top1_hits"])

    def test_alpha_zero_uses_best_score_then_engine_tie_order(self):
        metrics = self.metrics()
        group = self.groups[1]
        legal = self.prepared.groups_by_key[group.key][0].legal
        scores = {move: 1.0 for move in legal}
        scores[legal[0]] = 0.0
        metrics.accept("egaroucid", 21, [(group.key, HintData(scores, legal))])
        self.assertEqual(legal[1], metrics.predictions[1, metrics.names["alpha_0.0"]])

    def test_partial_cache_reuses_old_rows_and_searches_only_missing_states(self):
        import wthor_hint_pipeline as pipeline
        from test_wthor_human_match_experiment import FakeActorContext, FakeActorExecutor, make_actor_test_row
        groups = self.groups[:3]
        _, scores, raw = make_actor_test_row(groups[0].key, 1)
        seen = []
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            cache = pipeline.HintCache(directory / "hint_score_cache_level1_hint1.sqlite3")
            cache.put_many(1, 1, [(groups[0].key, scores, raw)])
            cache.close()
            with patch.object(pipeline, "ProcessPoolExecutor", FakeActorExecutor), patch.object(
                pipeline.multiprocessing, "get_context", return_value=FakeActorContext()), patch.object(
                pipeline, "wait", side_effect=lambda pending, **kw: (set(pending), set())), patch.object(
                pipeline.sys, "stderr", io.StringIO()):
                hints, stats, _, _ = pipeline.collect_hints(
                    groups, [1], directory, 2, Path("unused.exe"), 1, 1.0, 0, 60.0,
                    standalone_hint_count=1, reuse_partial_cache=True,
                    on_rows=lambda level, rows: seen.extend(key for key, _ in rows))
            self.assertEqual(2, stats[1]["scheduled_computations"])
            self.assertEqual({(1, g.key) for g in groups[1:]}, set(FakeActorExecutor.instance.claimed))
            self.assertEqual(3, len(seen))
            self.assertEqual({g.key for g in groups}, set(seen))
            self.assertEqual(scores, hints[1][groups[0].key].scores)

    def test_exports_include_boundaries_duplicates_and_zero_length_subsets(self):
        for losses in (self.losses, np.ones(len(self.records), dtype="u1")):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                ids = np.arange(len(self.records), dtype="<u8") // 3
                counts = export_subsets(root, "fixture", self.records, ids, losses, {})
                for count in counts:
                    stem = root / count["dataset"] / "test" / "0000000"
                    selected = np.flatnonzero(losses >= count["minimum_loss"])
                    np.testing.assert_array_equal(selected, np.fromfile(stem.with_suffix(".source_indices"), dtype="<u8"))
                    np.testing.assert_array_equal(self.records[selected], np.fromfile(stem.with_suffix(".dat"), dtype=BOARD_DTYPE))
                    np.testing.assert_array_equal(ids[selected], np.fromfile(stem.with_suffix(".game_ids"), dtype="<u8"))
                    self.assertEqual(len(selected), count["positions"])

    def test_reference_engine_mismatch_is_rejected_before_any_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, output = root / "reference", root / "output"
            reference.mkdir()
            (reference / "experiment_identity.json").write_text(json.dumps({"egaroucid_exe_sha256": "old"}))
            with self.assertRaisesRegex(ValueError, "egaroucid_exe_sha256"):
                evaluation.import_reference_caches(reference, output, {"egaroucid_exe_sha256": "new"})
            self.assertFalse(output.exists())

    def test_pipeline_failure_is_recorded_and_never_marked_complete(self):
        import subprocess
        import run_mistake_test_experiment as runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = runner.make_argparser().parse_args([
                "--output-dir", str(root), "--reference-dir", str(root / "reference"),
                "--normal-model", str(root / "normal.h5"), "--weighted-model", str(root / "weighted.h5")])
            with patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "fixture")):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run(args)
            status = json.loads((root / "run_status.json").read_text())
            self.assertEqual("failed", status["state"])
            self.assertIn("CalledProcessError", status["error"])


if __name__ == "__main__":
    unittest.main()
