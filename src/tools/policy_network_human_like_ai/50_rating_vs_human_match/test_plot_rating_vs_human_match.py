#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

from plot_rating_vs_human_match import (
    AXIS_LABEL_FONT_SIZE,
    BASE_FONT_SIZE,
    DEFAULT_INPUT,
    LABEL_FONT_SIZE,
    SERIES_STYLES,
    TITLE_FONT_SIZE,
    plot_results,
    read_results,
)


class RatingHumanMatchPlotTest(unittest.TestCase):
    def test_input_contains_both_series(self) -> None:
        results = read_results(DEFAULT_INPUT)

        self.assertEqual(16, len(results))
        self.assertEqual(10, sum(result.series == "console" for result in results))
        self.assertEqual(6, sum(result.series == "blend" for result in results))

    def test_plot_has_no_confidence_interval_error_bars(self) -> None:
        results = read_results(DEFAULT_INPUT)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_stem = Path(temp_dir) / "plot"
            with patch.object(
                plt.Axes,
                "errorbar",
                side_effect=AssertionError("error bars must not be drawn"),
            ):
                plot_results(results, output_stem)
            self.assertGreater(
                output_stem.with_suffix(".png").stat().st_size,
                10_000,
            )

    def test_requested_series_colors(self) -> None:
        self.assertEqual("#005AFF", SERIES_STYLES["console"]["color"])
        self.assertEqual("#FF4B00", SERIES_STYLES["blend"]["color"])

    def test_text_is_large_and_bold_enough(self) -> None:
        self.assertGreaterEqual(BASE_FONT_SIZE, 20)
        self.assertGreaterEqual(LABEL_FONT_SIZE, 20)
        self.assertGreaterEqual(AXIS_LABEL_FONT_SIZE, 24)
        self.assertGreaterEqual(TITLE_FONT_SIZE, 32)


if __name__ == "__main__":
    unittest.main()
