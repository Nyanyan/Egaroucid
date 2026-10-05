import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_wthor_blend_from_cache as cached
from blend_policy_with_egaroucid import policy_to_coord
from wthor_hint_pipeline import HintCache
from wthor_human_match_evaluation import (
    BLEND_PARAMS, BoardState, IncrementalAgreementMetrics, load_position_groups,
    make_blend_summary_rows, metrics_to_result,
)


class CachedBlendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.data = self.root / "data"
        self.data.mkdir()
        state = BoardState.initial()
        records = np.zeros(3, dtype=cached.wthor.BOARD_DTYPE)
        for i in range(3):
            records[i]["player"], records[i]["opponent"] = state.player_opponent_bits(state.side)
            legal = state.legal_policies(state.side)
            records[i]["color"] = state.side
            records[i]["policy"] = legal[-1]
            if i == 1:
                state.apply_move(state.side, legal[0])
        self.dat = self.data / "0.dat"
        records.tofile(self.dat)
        self.groups, _, indices_hash, records_hash = load_position_groups([self.dat], 3, 3, "all", 613, 613)
        self.identity = {
            "positions": 3, "data_split": "all", "split_seed": 613, "sample_seed": 613,
            "sample_global_indices_sha256": indices_hash, "sample_records_sha256": records_hash,
            "console_hint_count_by_level": {"21": 64}, "score_temperature": 1.0,
            "blend_params": list(BLEND_PARAMS),
        }
        self.summary = {
            "experiment_identity": self.identity, "board_data_dir": str(self.data),
            "evaluated_positions": 3, "unique_states": len(self.groups), "split_positions": 3,
            "sample_global_indices_sha256": indices_hash, "sample_records_sha256": records_hash,
            "top_n_values": [1],
        }
        (self.reference / "experiment_identity.json").write_text(json.dumps(self.identity))
        (self.reference / "random_wthor_blend_summary.json").write_text(json.dumps(self.summary))
        self.cache_path = self.reference / "hint_score_cache_level21_hint64.sqlite3"
        cache = HintCache(self.cache_path)
        try:
            rows = []
            for group in self.groups:
                legal = BoardState(*group.key).legal_policies(group.side)
                scores = {p: float(-i) for i, p in enumerate(legal)}
                raw = "\n".join(f"0 | 0 | 0 | {policy_to_coord(p)} | {scores[p]}" for p in legal)
                rows.append((group.key, scores, raw))
            cache.put_many(21, 64, rows)
        finally:
            cache.close()
        self.model = self.root / "model.h5"
        self.model.write_bytes(b"mock model")
        self.args = cached.make_argparser().parse_args([
            "--reference-dir", str(self.reference), "--model", str(self.model),
            "--output-dir", str(self.root / "output"),
        ])

    def test_new_model_only_matches_previous_blend_math_without_engine_processes(self):
        hints = cached.load_reference_hints(self.reference, self.groups)
        logits = {g.key: np.linspace(-2, 3, 64) for g in self.groups}
        previous = IncrementalAgreementMetrics(self.groups, logits, 1.0, (1,))
        previous.add_hints(21, hints)
        expected = make_blend_summary_rows(metrics_to_result(previous.result()[0]))[1:]
        before = self.cache_path.read_bytes()
        with patch.object(cached, "predict_policy_logits", return_value=logits), patch(
            "subprocess.Popen", side_effect=AssertionError("Engine must not start"),
        ), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            result = cached.run(self.args)
        self.assertEqual(expected, result["results"])
        self.assertEqual([0.2, 0.4, 0.6, 0.8, 1.0], list(result["blend_params"]))
        self.assertEqual(0, result["engine_searches"])
        self.assertEqual(before, self.cache_path.read_bytes())
        self.assertTrue((self.args.output_dir / "cached_wthor_blend_summary.csv").is_file())
        self.assertFalse((self.args.output_dir / "random_wthor_edax_level_summary.csv").exists())

    def test_missing_cache_fails_before_inference_without_creating_database(self):
        self.cache_path.unlink()
        with patch.object(cached, "predict_policy_logits") as infer, self.assertRaises(sqlite3.Error):
            cached.run(self.args)
        infer.assert_not_called()
        self.assertFalse(self.cache_path.exists())
        self.assertFalse(self.args.output_dir.exists())

    def test_incomplete_cache_stops_without_search(self):
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("DELETE FROM hint_scores WHERE key=(SELECT key FROM hint_scores LIMIT 1)")
        with self.assertRaisesRegex(ValueError, "Incomplete level-21 cache"):
            cached.load_reference_hints(self.reference, self.groups)

    def test_different_labels_fail_sample_hash_validation(self):
        records = np.fromfile(self.dat, dtype=cached.wthor.BOARD_DTYPE)
        records[0]["policy"] = 0
        records.tofile(self.dat)
        with self.assertRaisesRegex(ValueError, "sample_records_sha256"):
            cached.load_reference_sample(self.summary, self.identity, self.data)

    def test_dry_run_allows_unfinished_model_and_writes_nothing(self):
        self.model.unlink()
        self.args.dry_run = True
        with patch.object(cached, "predict_policy_logits") as infer:
            cached.run(self.args)
        infer.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_read_only_connection_rejects_writes(self):
        cache = cached.ReadOnlyHintCache(self.cache_path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                cache.conn.execute("DELETE FROM hint_scores")
        finally:
            cache.close()


if __name__ == "__main__":
    unittest.main()
