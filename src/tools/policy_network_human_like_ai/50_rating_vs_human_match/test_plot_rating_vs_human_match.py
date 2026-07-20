#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from plot_rating_vs_human_match import (
    DEFAULT_INPUT,
    plot_results,
    rating_ci_scale,
    read_results,
)


class RatingConfidenceIntervalTest(unittest.TestCase):
    def test_68_percent_interval_matches_strength_log_recalculation(self) -> None:
        results = read_results(DEFAULT_INPUT)
        level1 = next(result for result in results if result.label == "level 1")
        alpha0 = next(result for result in results if result.label == "alpha=0.0")

        self.assertAlmostEqual(
            389.6789063564,
            level1.rating_ci95_half_width * rating_ci_scale(0.68),
            places=9,
        )
        self.assertAlmostEqual(
            362.1779846845,
            alpha0.rating_ci95_half_width * rating_ci_scale(0.68),
            places=9,
        )

    def test_confidence_must_be_between_zero_and_one(self) -> None:
        for invalid in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    rating_ci_scale(invalid)

    def test_default_and_original_intervals_have_valid_label_layouts(self) -> None:
        results = read_results(DEFAULT_INPUT)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for confidence in (0.68, 0.95):
                with self.subTest(confidence=confidence):
                    confidence_name = str(confidence).replace(".", "_")
                    output_stem = output_dir / f"plot_ci{confidence_name}"
                    plot_results(results, output_stem, confidence)
                    self.assertGreater(
                        output_stem.with_suffix(".png").stat().st_size,
                        10_000,
                    )


if __name__ == "__main__":
    unittest.main()
