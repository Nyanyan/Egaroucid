import io
import os
import queue
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import wthor_hint_pipeline as pipeline
from wthor_edax import DEFAULT_EDAX_EXE, EDAX_LEVELS, EdaxHintRunner, parse_edax_hint_output
from wthor_human_match_evaluation import BoardState, HintData, PositionGroup, evaluate_engine_agreement
from wthor_human_match_experiment import make_argparser, make_edax_identity
from test_wthor_human_match_experiment import FakeActorContext, FakeActorExecutor


class EdaxTest(unittest.TestCase):
    def test_hint_parser_accepts_score_bounds_without_dropping_moves(self):
        raw = (
            "13@73%  -21 0:00.000 66966 e8 C2\n"
            "13@73%  -51 0:00.000 2973 h7 H8\n"
            "13@73% <-63 0:00.000 8792 g2 H1\n"
            "13@73% >+63 0:00.001 8792 8792000 a1\n"
            "13@73% ?+02 0:00.000 2 c4\n>"
        )
        scores = parse_edax_hint_output(raw)
        self.assertEqual([-21., -51., -63., 63., 2.], list(scores.values()))
        self.assertEqual(-63., scores[49])
        with self.assertRaises(ValueError):
            parse_edax_hint_output(raw + "\n13@73% <-63 0:00.000 1 g2\n")

    def test_positive_score_bound_is_not_a_prompt(self):
        runner = EdaxHintRunner(DEFAULT_EDAX_EXE, timeout_sec=1)
        runner.stdout_queue = queue.Queue()
        raw = "\n13@73% >+63 0:00.000 123 e8\n13@73% -51 0:00.000 23 h7\n>"
        for char in raw + "next command output":
            runner.stdout_queue.put(char)
        self.assertEqual(raw, runner._read_until_prompt())
        self.assertEqual("n", runner.stdout_queue.get_nowait())

    def test_bounded_hint_is_complete_and_retains_its_qualifier_in_cache_text(self):
        state = BoardState(0x001414343c101604, 0x3e090b0b030f0800, 0)
        runner = EdaxHintRunner(DEFAULT_EDAX_EXE)
        raw = ("13@73% -21 0:00.000 66966 e8 C2\n"
               "13@73% -51 0:00.000 2973 h7 H8\n"
               "13@73% <-63 0:00.000 8792 g2 H1\n>")
        with patch.object(runner, "raw_hint", return_value=raw):
            scores, normalized = runner.hint_scores(state, state.side, 64)
        self.assertEqual((3, 8, 49), pipeline.validate_hint_data(state, state.side, scores, normalized, 64))
        self.assertIn("g2 | -63.0 | Edax bound <", normalized)
        with tempfile.TemporaryDirectory() as directory:
            cache = pipeline.HintCache(Path(directory) / "edax.sqlite3")
            try:
                key = (state.black, state.white, state.side)
                cache.put_many(13, 64, [(key, scores, normalized)])
                loaded = cache.load(13, 64, [key])[key]
                self.assertEqual(scores, loaded.scores)
                self.assertEqual((3, 8, 49), loaded.move_order)
            finally:
                cache.close()

    def test_quiet_hint_parser_uses_pv_first_move_and_keeps_ties(self):
        raw = "\n    1   -04  0:00.000  14  d3 c3\n   21@95% +02  0:01.250  999  456 c4\n>"
        scores = parse_edax_hint_output(raw)
        self.assertEqual([44, 37], list(scores))
        self.assertEqual([-4.0, 2.0], list(scores.values()))
        with self.assertRaises(ValueError):
            parse_edax_hint_output("no results\n>")
        with self.assertRaises(ValueError):
            parse_edax_hint_output(raw + "\n1 +04 0:00.000 1 d3\n")

    def test_defaults_and_identity_cover_edax_inputs(self):
        args = make_argparser().parse_args(["3"])
        self.assertEqual((1, 5, 9, 13, 17, 21), EDAX_LEVELS)
        self.assertEqual(DEFAULT_EDAX_EXE, args.edax_exe)
        with tempfile.TemporaryDirectory() as folder:
            args.edax_exe = Path(folder) / "edax.exe"
            args.edax_eval_file = Path(folder) / "eval.dat"
            args.edax_exe.write_bytes(b"exe")
            args.edax_eval_file.write_bytes(b"first eval")
            first = make_edax_identity(args, 6)
            args.edax_eval_file.write_bytes(b"second eval")
            second = make_edax_identity(args, 6)
            self.assertNotEqual(first["eval_sha256"], second["eval_sha256"])
            runner = EdaxHintRunner(args.edax_exe, eval_file=args.edax_eval_file, level=21, threads=2)
            command = runner.command()
            self.assertEqual("2", command[command.index("-n") + 1])
            self.assertEqual("21", command[command.index("-l") + 1])
            self.assertNotIn("-t", command)

    def test_native_hint_order_wins_over_researched_scores(self):
        state = BoardState.initial()
        for _ in range(3):
            state.apply_move(state.side, state.legal_policies(state.side)[0])
        legal = state.legal_policies(state.side)
        group = PositionGroup(state.black, state.white, state.side, ((legal[0], 1),))
        order = tuple(legal[:3])
        hint = HintData({move: float(i) for i, move in enumerate(order)}, order)
        metrics = evaluate_engine_agreement([group], {21: {group.key: hint}})
        self.assertEqual(1, metrics[21]["hits"][1])

    def test_edax_level21_uses_three_hints_and_a_separate_cache(self):
        state = BoardState.initial()
        group = PositionGroup(state.black, state.white, state.side, ((44, 2),))
        with tempfile.TemporaryDirectory() as folder, patch.object(
            pipeline, "ProcessPoolExecutor", FakeActorExecutor,
        ), patch.object(pipeline.multiprocessing, "get_context", return_value=FakeActorContext()), patch.object(
            pipeline, "wait", side_effect=lambda pending, **kw: (set(pending), set()),
        ), patch.object(pipeline.sys, "stderr", io.StringIO()):
            hints, stats, _, _ = pipeline.collect_hints(
                [group], [21], Path(folder), 1, Path("unused.exe"), 1, 1.0, 0, 60.0,
                engine="edax", edax_eval_file=Path("eval.dat"),
            )
            task = FakeActorExecutor.instance.submitted[0].task
            self.assertEqual("edax", task.engine)
            self.assertEqual(3, task.hint_count)
            self.assertTrue(Path(stats[21]["path"]).name.startswith("edax_"))
            metrics = evaluate_engine_agreement([group], hints)
            self.assertEqual({"positions": 2, "hits": {1: 2, 3: 2}}, metrics[21])
            _, warm_stats, _, _ = pipeline.collect_hints(
                [group], [21], Path(folder), 1, Path("unused.exe"), 1, 1.0, 0, 60.0,
                engine="edax", edax_eval_file=Path("eval.dat"),
            )
            self.assertEqual(1, warm_stats[21]["hits"])
            self.assertEqual(0, warm_stats[21]["scheduled_computations"])

    @unittest.skipUnless(os.environ.get("WTHOR_EDAX_INTEGRATION") == "1", "opt-in real Edax smoke test")
    def test_real_edax_regression_full_legal_hints_after_bounded_score(self):
        state = BoardState(0x001414343c101604, 0x3e090b0b030f0800, 0)
        runner = EdaxHintRunner(DEFAULT_EDAX_EXE, level=13, threads=1, timeout_sec=30)
        try:
            for current in (state, BoardState.initial(), state):
                scores, raw = runner.hint_scores(current, current.side, 64)
                order = pipeline.validate_hint_data(current, current.side, scores, raw, 64)
                self.assertEqual(set(current.legal_policies(current.side)), set(order))
        finally:
            runner.close()

    @unittest.skipUnless(os.environ.get("WTHOR_EDAX_INTEGRATION") == "1", "opt-in real Edax smoke test")
    def test_real_edax_all_levels_on_both_colors(self):
        state = BoardState.initial()
        states = []
        while bin(state.black | state.white).count("1") < 61:
            legal = state.legal_policies(state.side)
            if not legal:
                state.side = 1 - state.side
                legal = state.legal_policies(state.side)
                if not legal:
                    break
            if bin(state.black | state.white).count("1") >= 58:
                states.append(BoardState(state.black, state.white, state.side))
            state.apply_move(state.side, legal[0])
        self.assertEqual({0, 1}, {s.side for s in states})
        for level in EDAX_LEVELS:
            runner = EdaxHintRunner(DEFAULT_EDAX_EXE, level=level, timeout_sec=15)
            try:
                for state in states:
                    scores, raw = runner.hint_scores(state, state.side)
                    order = pipeline.validate_hint_data(state, state.side, scores, raw, 3)
                    self.assertEqual(min(3, len(state.legal_policies(state.side))), len(order))
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
