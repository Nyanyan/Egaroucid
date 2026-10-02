import io
import os
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
