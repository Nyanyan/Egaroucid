from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import wthor_hint_pipeline as pipeline
from test_wthor_human_match_experiment import FakeActorContext, FakeActorExecutor
from wthor_human_match_evaluation import (
    BoardState, CONSOLE_LEVELS, HintData, PositionGroup, evaluate_agreement,
    make_blend_summary_rows, metrics_to_result,
)
from wthor_human_match_experiment import evaluation_top_n, format_live_metric, make_argparser
from evaluate_wthor_human_match import make_argparser as policy_parser


class OptionalTop3Test(unittest.TestCase):
    def test_top3_is_optional_and_likelihood_still_uses_all_legal_moves(self):
        state = BoardState.initial()
        legal = state.legal_policies(state.side)
        group = PositionGroup(state.black, state.white, state.side, ((legal[0], 2),))
        self.assertEqual([1], policy_parser().parse_args(["--weights", "unused.bin"]).top_n)
        for flags, expected in [([], (1,)), (["--top3"], (1, 3))]:
            with self.subTest(flags=flags):
                top_n = evaluation_top_n(make_argparser().parse_args(flags))
                self.assertEqual(expected, top_n)
                hints = {
                    level: {group.key: HintData(
                        {move: 0.0 for move in (legal if level == 21 else legal[:max(top_n)])},
                        tuple(legal if level == 21 else legal[:max(top_n)]),
                    )}
                    for level in CONSOLE_LEVELS
                }
                blend, console, _, _ = evaluate_agreement(
                    [group], hints, {group.key: np.zeros(64)}, 1.0, top_n_values=top_n,
                )
                for metric in [*blend.values(), *console.values()]:
                    self.assertEqual(set(expected), set(metric["hits"]))
                    self.assertEqual(bool(flags), "top-3" in format_live_metric(metric))
                for row in make_blend_summary_rows(metrics_to_result(blend)):
                    self.assertEqual(bool(flags), "top3_hits" in row)
                    self.assertAlmostEqual(0.25, row["mean_likelihood"])

    def test_top1_requests_one_engine_move_but_all_blend_moves(self):
        state = BoardState.initial()
        legal = state.legal_policies(state.side)
        group = PositionGroup(state.black, state.white, state.side, ((legal[0], 1),))
        for engine in ("egaroucid", "edax"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as directory, patch.object(
                pipeline, "ProcessPoolExecutor", FakeActorExecutor,
            ), patch.object(pipeline.multiprocessing, "get_context", return_value=FakeActorContext()), patch.object(
                pipeline, "wait", side_effect=lambda pending, **kw: (set(pending), set()),
            ), redirect_stderr(io.StringIO()):
                hints, stats, _, _ = pipeline.collect_hints(
                    [group], [1, 21], Path(directory), 2, Path("unused"), 1, 1, 0, 30,
                    engine=engine, standalone_hint_count=1,
                )
                self.assertEqual(1, len(hints[1][group.key].move_order))
                self.assertIn("_hint1.sqlite3", stats[1]["path"])
                self.assertEqual(len(legal) if engine == "egaroucid" else 1,
                                 len(hints[21][group.key].move_order))


if __name__ == "__main__":
    unittest.main()
