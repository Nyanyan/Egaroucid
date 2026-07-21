#!/usr/bin/env python3

from contextlib import redirect_stderr
import io
import unittest

from wthor_human_match_evaluation import BLEND_PARAMS, CONSOLE_LEVELS
from wthor_human_match_experiment import (
    AgreementProgressReporter,
    format_live_metric,
    make_argparser,
)


def metric(positions=0, top1_hits=0, top3_hits=0):
    return {
        "positions": positions,
        "hits": {1: top1_hits, 3: top3_hits},
    }


class FakeIncrementalMetrics:
    def __init__(self):
        self.added = []
        self.blend = {alpha: metric() for alpha in BLEND_PARAMS}
        self.console = {level: metric() for level in CONSOLE_LEVELS}

    def add_hint(self, level, state_key, hint):
        self.added.append((level, state_key, hint))

    def snapshot(self):
        return self.blend, self.console, 0, 0


class ProgressReportingTest(unittest.TestCase):
    def test_metric_without_positions_is_not_reported_as_zero_percent(self):
        self.assertEqual("未算出 (n=0)", format_live_metric(metric()))
        self.assertEqual(
            "60.000%/90.000% (n=10)",
            format_live_metric(metric(10, 6, 9)),
        )

    def test_report_lists_every_alpha_and_level_with_denominators(self):
        metrics = FakeIncrementalMetrics()
        metrics.blend[0.0] = metric(10, 6, 9)
        metrics.console[21] = metric(10, 6, 9)
        reporter = AgreementProgressReporter(metrics)
        output = io.StringIO()

        with redirect_stderr(output):
            reporter.report({"final": False})

        text = output.getvalue()
        for alpha in BLEND_PARAMS:
            self.assertIn(f"alpha={alpha:.1f}", text)
        for level in CONSOLE_LEVELS:
            self.assertIn(f"level {level:2d}", text)
        self.assertIn("60.000%/90.000% (n=10)", text)
        self.assertIn("未算出 (n=0)", text)
        self.assertIn("暫定値", text)

    def test_final_report_does_not_show_interim_warning(self):
        metrics = FakeIncrementalMetrics()
        reporter = AgreementProgressReporter(metrics)
        output = io.StringIO()

        with redirect_stderr(output):
            reporter.report({"final": True})

        self.assertIn("完了", output.getvalue())
        self.assertNotIn("暫定値", output.getvalue())

    def test_rows_are_forwarded_to_incremental_metrics(self):
        metrics = FakeIncrementalMetrics()
        reporter = AgreementProgressReporter(metrics)
        state_key = (1, 2, 0)
        hint = object()

        reporter.accept_rows(21, [(state_key, hint)])

        self.assertEqual([(21, state_key, hint)], metrics.added)

    def test_progress_interval_cli_default_and_override(self):
        parser = make_argparser()
        self.assertEqual(
            30.0,
            parser.parse_args(["10000"]).progress_interval_sec,
        )
        self.assertEqual(
            5.0,
            parser.parse_args(
                ["10000", "--progress-interval-sec", "5"]
            ).progress_interval_sec,
        )


if __name__ == "__main__":
    unittest.main()
