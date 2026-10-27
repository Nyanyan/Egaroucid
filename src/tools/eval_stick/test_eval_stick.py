"""Meaningful contract tests for source transforms, scoring and cache identity."""
import copy
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from build_eval_stick import probability, reference_key, processing_order
from common import canonical, legal, play, transform, transform_move
from engine import OwnedJob, deadline_watchdog, parse_solve
from score_eval_stick import judge


class EvalStickTests(unittest.TestCase):
    def setUp(self):
        self.board = "---------------------------OX------XO---------------------------"

    def test_all_symmetries_preserve_legal_moves_and_children(self):
        original_legal = legal(self.board, "X")
        self.assertEqual(set(original_legal), {"d3", "c4", "f5", "e6"})
        for symmetry in range(8):
            target = ["-"] * 64
            for i, ch in enumerate(self.board):
                target[transform(i, symmetry)] = ch
            board = "".join(target)
            self.assertEqual(set(legal(board, "X")), {transform_move(m, symmetry) for m in original_legal})
            self.assertEqual(canonical(board, "X")[:2], canonical(self.board, "X")[:2])
            child, turn = play(self.board, "X", "d3")
            transformed_child, transformed_turn = play(board, "X", transform_move("d3", symmetry))
            self.assertEqual(turn, transformed_turn)
            self.assertEqual(canonical(child, turn)[:2], canonical(transformed_child, turn)[:2])

    def test_cache_identity_changes_with_search_conditions(self):
        pos = {"board": self.board, "turn": "X", "empties": 50}
        setting = {"binary_sha256": "a", "hash": 28, "threads": 28}
        baseline = reference_key(pos, "d3", setting)
        for field, value in (("binary_sha256", "b"), ("hash", 27), ("threads", 8)):
            other = dict(setting, **{field: value})
            self.assertNotEqual(baseline, reference_key(pos, "d3", other))
        self.assertNotEqual(baseline, reference_key(pos, "d3", setting, True))
        self.assertNotEqual(baseline, reference_key(pos, "c4", setting))
        self.assertEqual([probability(e) for e in (50, 48, 47, 44, 43, 41)], [88, 88, 93, 93, 98, 98])

    def test_processing_order_is_reproducible_balanced_and_separate_from_rank(self):
        original = [{"id": str(i), "selection_order": i, "empties": 41 + i % 10} for i in range(200)]
        first = processing_order(copy.deepcopy(original))
        second = processing_order(copy.deepcopy(original))
        self.assertEqual(first, second)
        self.assertEqual({p["id"] for p in first}, {p["id"] for p in original})
        for offset in range(0, 200, 10):
            self.assertEqual(sorted(p["empties"] for p in first[offset:offset+10]), list(range(41, 51)))
        self.assertNotEqual({p["selection_rank"] for p in first[:10]}, set(range(10)))

    def test_tied_best_and_unknown_move_are_not_zero_filled(self):
        pos = {"id": "test", "board": self.board, "turn": "X", "empties": 60, "game": "g",
               "candidates": [{"move": "d3", "value": 2}, {"move": "c4", "value": 2}, {"move": "f5", "value": -2}]}
        self.assertTrue(judge(pos, {"move": "c4"})["best_match"])
        self.assertEqual(judge(pos, {"move": "f5"})["loss"], 4)
        result = judge(pos, {"move": "e6"})
        self.assertIsNone(result["loss"])
        self.assertTrue(result["unscored_move"])
        with self.assertRaises(ValueError):
            judge(pos, {"move": "a1"})

    def test_console_parser_preserves_time_depth_and_nodes(self):
        raw = "| 60 | 28@74% | g4 | -3 | 000:00:01.766 | 117519584 | 66545630 |\n"
        result = parse_solve(raw)[0]
        self.assertEqual(result["depth"], 28)
        self.assertEqual(result["probability"], 74)
        self.assertAlmostEqual(result["time_seconds"], 1.766)
        self.assertEqual(result["nodes"], 117519584)

    @unittest.skipUnless(os.name == "nt", "Windows ownership contract")
    def test_owned_child_stops_when_job_handle_closes(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        try:
            job = OwnedJob(child)
            job.close()
            child.wait(timeout=3)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill(); child.wait()

    def test_deadline_also_stops_a_child_before_any_protocol_result(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        timer = None
        try:
            with patch.dict(os.environ, {"EGAROUCID_EVAL_STICK_DEADLINE": str(time.time() + .05)}):
                timer = deadline_watchdog(child)
            child.wait(timeout=3)
            self.assertIsNotNone(child.poll())
        finally:
            if timer is not None:
                timer.cancel()
            if child.poll() is None:
                child.kill(); child.wait()


if __name__ == "__main__":
    unittest.main()
