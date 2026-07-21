#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer

from plot_rating_vs_human_match import (
    AXIS_LABEL_FONT_SIZE,
    BASE_FONT_SIZE,
    DEFAULT_INPUT,
    DEFAULT_LABEL_POSITIONS,
    LABEL_FONT_SIZE,
    LEADER_LINE_THRESHOLD_POINTS,
    LabelEditor,
    LabelPosition,
    SERIES_STYLES,
    TITLE_FONT_SIZE,
    annotation_position,
    create_plot,
    label_bbox,
    label_candidates,
    load_label_positions,
    read_results,
    save_label_positions,
)


class RatingHumanMatchPlotTest(unittest.TestCase):
    def test_input_contains_both_series(self) -> None:
        results = read_results(DEFAULT_INPUT)

        self.assertEqual(16, len(results))
        self.assertEqual(10, sum(result.series == "console" for result in results))
        self.assertEqual(6, sum(result.series == "blend" for result in results))

    def test_plot_has_horizontal_and_vertical_confidence_intervals(self) -> None:
        results = read_results(DEFAULT_INPUT)
        figure, _ = create_plot(results)
        errorbars = [
            container
            for container in figure.axes[0].containers
            if isinstance(container, ErrorbarContainer)
        ]

        self.assertEqual(2, len(errorbars))
        self.assertTrue(all(errorbar.has_xerr for errorbar in errorbars))
        self.assertTrue(all(errorbar.has_yerr for errorbar in errorbars))
        plt.close(figure)

    def test_ratings_match_50_set_results(self) -> None:
        loaded_results = read_results(DEFAULT_INPUT)
        results = {result.label: result for result in loaded_results}

        self.assertAlmostEqual(
            2238.9648171704375,
            results["alpha=0.0"].rating,
        )
        self.assertAlmostEqual(
            2179.368198724032,
            results["alpha=0.0"].rating_ci_lower,
        )
        self.assertAlmostEqual(
            2294.994042320917,
            results["alpha=0.0"].rating_ci_upper,
        )
        self.assertAlmostEqual(
            386.58207142362244,
            results["alpha=1.0"].rating,
        )
        self.assertAlmostEqual(
            326.05871764385046,
            results["alpha=1.0"].rating_ci_lower,
        )
        self.assertAlmostEqual(
            453.2212892796188,
            results["alpha=1.0"].rating_ci_upper,
        )
        self.assertAlmostEqual(
            1500.0,
            sum(result.rating for result in loaded_results) / len(loaded_results),
        )

    def test_human_match_values_match_30000_position_results(self) -> None:
        results = {
            result.label: result
            for result in read_results(DEFAULT_INPUT)
        }
        expected_top1 = {
            "level 1": 53.27,
            "level 3": 59.45333333333334,
            "level 5": 61.36333333333334,
            "level 7": 62.35,
            "level 9": 62.78333333333334,
            "level 11": 63.31,
            "level 13": 62.96333333333334,
            "level 15": 62.94333333333333,
            "level 17": 62.92,
            "level 19": 62.74,
            "alpha=0.0": 63.39333333333333,
            "alpha=0.2": 64.22,
            "alpha=0.4": 64.56333333333333,
            "alpha=0.6": 65.14666666666666,
            "alpha=0.8": 64.17333333333334,
            "alpha=1.0": 57.14666666666667,
        }

        for label, expected in expected_top1.items():
            self.assertAlmostEqual(expected, results[label].top1)
        self.assertAlmostEqual(
            52.70503574571409,
            results["level 1"].top1_ci_lower,
        )
        self.assertAlmostEqual(
            53.83412692348205,
            results["level 1"].top1_ci_upper,
        )
        self.assertAlmostEqual(
            64.60555118634693,
            results["alpha=0.6"].top1_ci_lower,
        )
        self.assertAlmostEqual(
            65.68390362387456,
            results["alpha=0.6"].top1_ci_upper,
        )

    def test_requested_series_colors(self) -> None:
        self.assertEqual("#005AFF", SERIES_STYLES["console"]["color"])
        self.assertEqual("#FF4B00", SERIES_STYLES["blend"]["color"])

    def test_text_is_large_and_bold_enough(self) -> None:
        self.assertGreaterEqual(BASE_FONT_SIZE, 30)
        self.assertGreaterEqual(LABEL_FONT_SIZE, 30)
        self.assertGreaterEqual(AXIS_LABEL_FONT_SIZE, 38)
        self.assertGreaterEqual(TITLE_FONT_SIZE, 48)

    def test_figure_plot_area_and_labels_have_white_backgrounds(self) -> None:
        results = read_results(DEFAULT_INPUT)
        figure, annotations = create_plot(results)
        axis = figure.axes[0]

        self.assertEqual(to_rgba("#FFFFFF"), figure.get_facecolor())
        self.assertEqual(to_rgba("#FFFFFF"), axis.get_facecolor())
        for annotation in annotations.values():
            self.assertEqual(
                to_rgba("#FFFFFF"),
                annotation.get_bbox_patch().get_facecolor(),
            )
        plt.close(figure)

    def test_label_placement_prefers_nearby_positions(self) -> None:
        first_candidates = label_candidates()[:8]
        self.assertTrue(
            all(max(abs(dx), abs(dy)) <= 15.0 for dx, dy in first_candidates)
        )

    def test_initial_position_file_covers_every_label(self) -> None:
        results = read_results(DEFAULT_INPUT)
        valid_labels = {result.label for result in results}
        positions = load_label_positions(DEFAULT_LABEL_POSITIONS, valid_labels)

        self.assertEqual(valid_labels, set(positions))

    def test_initial_label_positions_do_not_overlap(self) -> None:
        results = read_results(DEFAULT_INPUT)
        valid_labels = {result.label for result in results}
        positions = load_label_positions(DEFAULT_LABEL_POSITIONS, valid_labels)
        figure, annotations = create_plot(results, positions)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = {
            label: label_bbox(annotation, renderer)
            for label, annotation in annotations.items()
        }
        overlaps = [
            (first_label, second_label)
            for index, first_label in enumerate(boxes)
            for second_label in list(boxes)[index + 1 :]
            if boxes[first_label].overlaps(boxes[second_label])
        ]
        plt.close(figure)

        self.assertEqual([], overlaps)

    def test_leader_lines_are_shown_only_for_distant_labels(self) -> None:
        results = read_results(DEFAULT_INPUT)
        valid_labels = {result.label for result in results}
        positions = load_label_positions(DEFAULT_LABEL_POSITIONS, valid_labels)
        positions["level 1"] = LabelPosition(12.0, 0.0, "left", "center")
        positions["level 17"] = LabelPosition(36.0, 0.0, "left", "center")
        figure, annotations = create_plot(results, positions)

        self.assertFalse(annotations["level 1"].arrow_patch.get_visible())
        self.assertTrue(annotations["level 17"].arrow_patch.get_visible())
        self.assertGreater(
            abs(positions["level 17"].dx),
            LEADER_LINE_THRESHOLD_POINTS,
        )
        plt.close(figure)

    def test_position_file_round_trip(self) -> None:
        positions = {
            "level 1": LabelPosition(12.5, -3.25, "left", "top"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "positions.json"
            save_label_positions(path, positions)
            loaded = load_label_positions(path, {"level 1"})

        self.assertEqual(positions, loaded)

    def test_dragged_position_can_be_saved_and_reloaded(self) -> None:
        results = read_results(DEFAULT_INPUT)
        valid_labels = {result.label for result in results}
        positions = load_label_positions(DEFAULT_LABEL_POSITIONS, valid_labels)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_stem = Path(temp_dir) / "plot"
            positions_path = Path(temp_dir) / "positions.json"
            figure, annotations = create_plot(results, positions)
            editor = LabelEditor(
                figure,
                annotations,
                output_stem,
                positions_path,
            )
            annotation = annotations["level 1"]
            start = annotation_position(annotation)
            editor.drag_state = (
                annotation,
                100.0,
                100.0,
                start.dx,
                start.dy,
            )
            editor.on_motion(SimpleNamespace(x=125.0, y=80.0))
            self.assertTrue(annotation.arrow_patch.get_visible())
            editor.on_release(SimpleNamespace())
            editor.save()
            saved = load_label_positions(positions_path, valid_labels)
            plt.close(figure)

        self.assertNotEqual(start.dx, saved["level 1"].dx)
        self.assertNotEqual(start.dy, saved["level 1"].dy)


if __name__ == "__main__":
    unittest.main()
