#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from plot_rating_vs_human_match import (
    AXIS_LABEL_FONT_SIZE,
    BASE_FONT_SIZE,
    DEFAULT_INPUT,
    DEFAULT_LABEL_POSITIONS,
    LABEL_FONT_SIZE,
    LabelEditor,
    LabelPosition,
    SERIES_STYLES,
    TITLE_FONT_SIZE,
    annotation_position,
    create_plot,
    label_candidates,
    load_label_positions,
    plot_results,
    read_results,
    save_label_positions,
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
            label: annotation.get_window_extent(renderer)
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
            editor.on_release(SimpleNamespace())
            editor.save()
            saved = load_label_positions(positions_path, valid_labels)
            plt.close(figure)

        self.assertNotEqual(start.dx, saved["level 1"].dx)
        self.assertNotEqual(start.dy, saved["level 1"].dy)


if __name__ == "__main__":
    unittest.main()
