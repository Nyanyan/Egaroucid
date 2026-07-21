#!/usr/bin/env python3

from contextlib import redirect_stderr
import io
import unittest

from wthor_human_match_evaluation import BLEND_PARAMS, CONSOLE_LEVELS
from wthor_human_match_experiment import (
    AgreementProgressReporter,
    format_console_summary_line,
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
            "top-1 60.000% [31.267%, 83.182%] "
            "| top-3 90.000% [59.585%, 98.212%] | n=10",
            format_live_metric(metric(10, 6, 9)),
        )

    def test_console_stdout_line_includes_both_confidence_intervals(self):
        self.assertEqual(
            "level 19: top-1 60.000% [31.267%, 83.182%], "
            "top-3 90.000% [59.585%, 98.212%]",
            format_console_summary_line(
                {
                    "level": 19,
                    "top1_accuracy": 0.6,
                    "top1_ci95_lower": 0.31267376973365824,
                    "top1_ci95_upper": 0.8318196702937638,
                    "top3_accuracy": 0.9,
                    "top3_ci95_lower": 0.5958499732047615,
                    "top3_ci95_upper": 0.9821237869049271,
                }
            ),
        )

    def test_report_uses_one_line_per_calculated_model(self):
        metrics = FakeIncrementalMetrics()
        for alpha in BLEND_PARAMS:
            metrics.blend[alpha] = metric(10, 6, 9)
        metrics.console[21] = metric(10, 6, 9)
        reporter = AgreementProgressReporter(metrics)
        output = io.StringIO()

        with redirect_stderr(output):
            reporter.report(
                {
                    "final": False,
                    "reported_hints_by_level": {21: 1},
                    "target_hints_by_level": {21: 100},
                }
            )

        text = output.getvalue()
        for alpha in BLEND_PARAMS:
            self.assertEqual(1, text.count(f"[blend alpha={alpha:.1f}]"))
        self.assertEqual(1, text.count("[Console level=21]"))
        self.assertNotIn("[Console level= 1]", text)
        self.assertIn(
            "top-1 60.000% [31.267%, 83.182%] "
            "| top-3 90.000% [59.585%, 98.212%] | n=10",
            text,
        )
        self.assertIn("| hint 1/100", text)
        self.assertNotIn("未算出", text)
        self.assertNotIn("注意", text)
        model_lines = [
            line
            for line in text.splitlines()
            if "[blend alpha=" in line or "[Console level=" in line
        ]
        self.assertEqual(len(BLEND_PARAMS) + 1, len(model_lines))

    def test_report_waits_for_first_calculated_model_without_noise(self):
        metrics = FakeIncrementalMetrics()
        reporter = AgreementProgressReporter(metrics)
        output = io.StringIO()

        with redirect_stderr(output):
            reporter.report({"final": False})

        self.assertEqual("  一致率: 初回hint結果待ち\n", output.getvalue())

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
