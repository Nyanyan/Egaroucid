"""Progress percentages use inclusive bins, cached results and original game counts."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analysis_pipeline as pipeline
from miss_analysis import AnalysisCache
from test_miss_board_data import analysis_text, step
from test_weighted_board_data import analyze


class LossLoggingTest(unittest.TestCase):
    def test_boundaries_zero_loss_and_maximum_loss(self):
        counts = pipeline.loss_histogram([{"loss": loss} for loss in (0, 1, 2, 3, 4, 5, 6, 7, 128)])
        self.assertEqual("loss[positions=9]: 1-2=22.22%, 3-4=22.22%, 5-6=22.22%, 7+=22.22%",
                         pipeline.format_loss_percentages(counts))
        self.assertEqual("loss[positions=0]: 1-2=--, 3-4=--, 5-6=--, 7+=--",
                         pipeline.format_loss_percentages(pipeline.loss_histogram([])))

    def test_resumed_analysis_and_final_report_use_the_same_denominator(self):
        unique = {"f5": bytes([0]), "f5d6": bytes([0, 1])}
        raw = {"f5": analysis_text("f5", unique["f5"], [2]),
               "f5d6": analysis_text("f5d6", unique["f5d6"], [0, 7])}
        accepted = {"train": [(1, "f5"), (2, "f5")], "val": [(3, "f5d6")], "test": []}
        expected = "loss[positions=4]: 1-2=50.00%, 3-4=0.00%, 5-6=0.00%, 7+=25.00%"
        args = step.make_argparser().parse_args(["--dataset", "test", "--min-loss", "3", "--workers", "1"])

        with tempfile.TemporaryDirectory() as directory:
            with contextlib.closing(AnalysisCache(Path(directory), {})) as cache:
                cache.put("f5", raw["f5"], 0)
                with (mock.patch.object(pipeline, "AnalyzeEngine") as engine,
                      contextlib.redirect_stdout(io.StringIO()) as output):
                    engine.return_value.analyze.side_effect = raw.__getitem__
                    stats = pipeline.analyze_missing(unique, cache, args,
                                                     game_counts=pipeline.transcript_multiplicities(accepted))
                self.assertEqual({"cache_hits": 1, "analyzed_games": 1}, stats)
                engine.return_value.analyze.assert_called_once_with("f5d6")
                logs = output.getvalue().splitlines()
                self.assertIn("loss[positions=2]: 1-2=100.00%", "\n".join(logs))
                self.assertIn(expected, logs[-1])
                self.assertIn("ETA", logs[-1])

                with (mock.patch.object(pipeline, "AnalyzeEngine", side_effect=AssertionError("no new search")),
                      contextlib.redirect_stdout(io.StringIO()) as output):
                    stats = pipeline.analyze_missing(unique, cache, args,
                                                     game_counts=pipeline.transcript_multiplicities(accepted))
                self.assertEqual({"cache_hits": 2, "analyzed_games": 0}, stats)
                self.assertIn(expected, output.getvalue().splitlines()[-1])

                with contextlib.redirect_stdout(io.StringIO()) as output:
                    report = analyze.loss_distribution(accepted, unique, cache, 5)
                self.assertEqual(4, report["all"]["positions"])
                self.assertEqual(50, report["all"]["bins"][2]["percentage"])
                self.assertEqual(25, report["all"]["bins"][7]["percentage"])
                self.assertIn(expected, output.getvalue())
                self.assertIn("train: loss[positions=2]: 1-2=100.00%", output.getvalue())
                self.assertIn("test: loss[positions=0]: 1-2=--", output.getvalue())


if __name__ == "__main__":
    unittest.main()
