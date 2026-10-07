import hashlib
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_rating_vs_human_match import (
    FIGURES, ReportPlots, STYLES, configure_plotting, render_report, validate_report,
)
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from label_editor import apply_label_positions, figure_annotations


def example_report():
    datasets = [
        {"dataset": "overall", "label": "全体", "positions": 1000, "minimum_loss": None},
        {"dataset": "loss2", "label": "2石以上", "positions": 700, "minimum_loss": 2},
        {"dataset": "loss4", "label": "4石以上", "positions": 300, "minimum_loss": 4},
    ]
    specs = [
        ("alpha_0.0", "Egaroucid", "E21", 2100, [.60, .30, .20]),
        ("alpha_1.0", "normal", "N1.0", 1000, [.63, .38, .25]),
        ("loss_plus_one_squared_blend_alpha_0.2", "squared_blend", "B0.2", 1200, [.60, .42, .28]),
        ("loss_plus_one_squared_alpha_1.0", "squared_blend", "B1.0", 400, [.45, .40, .35]),
    ]
    models = []
    for model_id, family, short, elo, values in specs:
        models.append({
            "model_id": model_id, "model": short, "short": short, "group": family,
            "elo": elo, "elo_low": elo - 12, "elo_high": elo + 27,
            "results": [dict(dataset, model_id=model_id, top1_accuracy=value,
                             top1_ci95_lower=value - .03, top1_ci95_upper=value + .04)
                        for dataset, value in zip(datasets, values)],
        })
    return {"models": models, "dataset_counts": datasets, "series_paths": {
        "Egaroucid": ["alpha_0.0"],
        "normal": ["alpha_0.0", "alpha_1.0"],
        "squared_blend": ["alpha_1.0", specs[2][0], specs[3][0]],
    }}


class ReportPlotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_plotting()

    def tearDown(self):
        plt.close("all")

    def test_measurements_are_joined_by_dataset_id_without_mutating_input(self):
        data = example_report()
        data["models"][0]["results"].reverse()
        normalized = validate_report(data)
        self.assertEqual("overall", normalized["models"][0]["results"][0]["dataset"])
        self.assertEqual("loss4", data["models"][0]["results"][0]["dataset"])
        self.assertEqual(.60, normalized["models"][0]["results"][0]["top1_accuracy"])

    def test_overall_comparison_selects_two_disc_loss_by_condition_not_position(self):
        data = example_report()
        loss_one = dict(data["dataset_counts"][1], dataset="loss1", label="1石以上", minimum_loss=1)
        data["dataset_counts"].insert(1, loss_one)
        for model in data["models"]:
            model["results"].insert(1, dict(loss_one, model_id=model["model_id"], top1_accuracy=.12,
                                           top1_ci95_lower=.10, top1_ci95_upper=.14))
        plots = ReportPlots(data)
        figure = plots.elo_vs_agreement()
        self.assertTrue(figure.axes[1].get_title().startswith("2石以上"))
        labels = figure_annotations(figure)
        self.assertIn("loss2/alpha_1.0", labels)
        self.assertNotIn("loss1/alpha_1.0", labels)
        first_bar = next(item for item in figure.axes[1].containers if isinstance(item, ErrorbarContainer))
        self.assertAlmostEqual(30, float(first_bar.lines[0].get_ydata()[0]))
        self.assertEqual("#249972", plots.index["alpha_0.0"]["color"])
        self.assertEqual("#249972", labels["overall/alpha_0.0"].get_color())
        for dataset in data["dataset_counts"]:
            if dataset["dataset"] == "loss2":
                dataset["minimum_loss"] = 3
        with self.assertRaisesRegex(ValueError, "minimum_loss=2"):
            ReportPlots(data).elo_vs_agreement()

    def test_invalid_cohorts_intervals_and_series_fail_before_rendering(self):
        for failure in ("positions", "missing", "duplicate", "unknown", "uncovered", "interval", "percent", "nan"):
            with self.subTest(failure=failure):
                data = example_report()
                row = data["models"][0]["results"][0]
                if failure == "positions":
                    row["positions"] += 1
                elif failure == "missing":
                    data["models"][0]["results"].pop()
                elif failure == "duplicate":
                    data["models"][0]["results"].append(dict(row))
                elif failure == "unknown":
                    data["series_paths"]["normal"].append("missing_model")
                elif failure == "uncovered":
                    data["series_paths"].pop("squared_blend")
                elif failure == "interval":
                    row["top1_ci95_upper"] = .1
                elif failure == "percent":
                    row.update(top1_accuracy=60, top1_ci95_lower=57, top1_ci95_upper=64)
                elif failure == "nan":
                    data["models"][0]["elo_low"] = float("nan")
                with self.assertRaises(ValueError):
                    validate_report(data)

    def test_shared_endpoints_have_one_marker_and_orange_path_preserves_alpha_order(self):
        plots = ReportPlots(example_report())
        figure, axis = plt.subplots()
        plots.scatter_panel(axis, 0)
        orange = [line for line in axis.lines if line.get_color() == STYLES["squared_blend"]["color"]
                  and len(line.get_xdata()) == 3][0]
        np.testing.assert_allclose([1000, 1200, 400], orange.get_xdata())
        np.testing.assert_allclose([63, 60, 45], orange.get_ydata())
        bars = [container for container in axis.containers if isinstance(container, ErrorbarContainer)]
        self.assertEqual(4, len(bars))
        self.assertEqual("#D62728", plots.index["alpha_1.0"]["color"])
        self.assertEqual("#D55E00", plots.index["loss_plus_one_squared_alpha_1.0"]["color"])

    def test_every_model_is_labeled_in_every_scatter_panel_including_intermediate_levels(self):
        data = example_report()
        for model_id, group, short, elo in (("egaroucid_l1", "Egaroucid", "E1", 700),
                                          ("edax_l5", "Edax", "D5", 1500),
                                          ("alpha_0.2", "normal", "N0.2", 1800)):
            model = deepcopy(data["models"][0])
            model.update(model_id=model_id, group=group, short=short, model=short,
                         elo=elo, elo_low=elo - 12, elo_high=elo + 27)
            for row in model["results"]:
                row["model_id"] = model_id
            data["models"].append(model)
            data["series_paths"].setdefault(group, []).append(model_id)
        plots = ReportPlots(data)
        for name, indices in (("elo_vs_agreement", (0, 1)), ("mistake_elo_vs_agreement", (1, 2))):
            figure = getattr(plots, name)()
            annotations = apply_label_positions(figure, {})
            expected = {f"{data['dataset_counts'][k]['dataset']}/{model['model_id']}": model["short"]
                        for k in indices for model in data["models"]}
            self.assertEqual(expected, {key: label.get_text() for key, label in annotations.items()})
            self.assertEqual(set(expected), set(figure_annotations(figure)))
            self.assertTrue(all(label.get_visible() for label in annotations.values()))

    def test_scatter_uses_saved_95_percent_bounds_and_ten_percent_opacity(self):
        plots = ReportPlots(example_report())
        figure, axis = plt.subplots()
        plots.scatter_panel(axis, 0)
        bands = [item for item in axis.collections if isinstance(item, PolyCollection)]
        self.assertEqual(2, len(bands))
        orange_band = bands[-1]
        self.assertAlmostEqual(.10, orange_band.get_alpha())
        np.testing.assert_allclose(orange_band.get_facecolor()[0], to_rgba("#D55E00", .10))
        vertices = orange_band.get_paths()[0].vertices
        for x, low, high in ((1000, 60, 67), (1200, 57, 64), (400, 42, 49)):
            bounds = vertices[vertices[:, 0] == x, 1]
            self.assertAlmostEqual(low, bounds.min())
            self.assertAlmostEqual(high, bounds.max())
        first = next(item for item in axis.containers if isinstance(item, ErrorbarContainer))
        np.testing.assert_allclose(first.lines[2][0].get_segments()[0], [[2088, 60], [2127, 60]])
        np.testing.assert_allclose(first.lines[2][1].get_segments()[0], [[2100, 57], [2100, 64]])
        for bar in first.lines[2]:
            np.testing.assert_allclose(bar.get_colors()[:, 3], 1.0)
            self.assertEqual(1.0, bar.get_alpha())
        for cap in first.lines[1]:
            self.assertEqual(1.0, to_rgba(cap.get_color())[3])
            self.assertEqual(1.0, cap.get_alpha())

    def test_every_threshold_line_has_its_own_matching_ci_band(self):
        plots = ReportPlots(example_report(), ci_alpha=.1)
        figure = plots.threshold_trends()
        for axis in figure.axes:
            if not axis.get_visible():
                continue
            bands = [item for item in axis.collections if isinstance(item, PolyCollection)]
            self.assertEqual(len(axis.lines), len(bands))
            for line, band in zip(axis.lines, bands):
                np.testing.assert_allclose(band.get_facecolor()[0], to_rgba(line.get_color(), .1))
        orange_axis = figure.axes[2]
        self.assertEqual(["B0.0", "B0.2", "B1.0"], [line.get_label() for line in orange_axis.lines])
        self.assertTrue(all(line.get_color() == "#D55E00" for line in orange_axis.lines))

    def test_alpha_override_and_invalid_alpha(self):
        plots = ReportPlots(example_report(), ci_alpha=.2)
        figure, axis = plt.subplots()
        plots.scatter_panel(axis, 0)
        self.assertAlmostEqual(.2, next(item for item in axis.collections if isinstance(item, PolyCollection)).get_alpha())
        for errorbar in axis.containers:
            if isinstance(errorbar, ErrorbarContainer):
                for bar in errorbar.lines[2]:
                    np.testing.assert_allclose(bar.get_colors()[:, 3], 1.0)
        for value in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                ReportPlots(example_report(), ci_alpha=value)

    def test_all_six_figures_and_source_manifest_are_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report_data.json"
            source.write_text(json.dumps(example_report()), encoding="utf-8")
            original_bytes = source.read_bytes()
            output = root / "figures"
            generated = render_report(source, output, dpi=45)
            self.assertEqual(12, len(generated))
            for name in FIGURES:
                self.assertTrue((output / (name + ".png")).read_bytes().startswith(b"\x89PNG"))
                self.assertIn("<svg", (output / (name + ".svg")).read_text(encoding="utf-8"))
            manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256(original_bytes).hexdigest(), manifest["input_sha256"])
            self.assertEqual(original_bytes, source.read_bytes())
            self.assertEqual(.90, manifest["ci_transparency"])
            self.assertEqual(.10, manifest["ci_alpha"])
            self.assertEqual(1.0, manifest["ci_errorbar_alpha"])


if __name__ == "__main__":
    unittest.main()
