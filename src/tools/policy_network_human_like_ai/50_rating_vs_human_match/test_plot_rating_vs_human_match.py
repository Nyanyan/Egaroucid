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


def model_points(axis):
    return [line for line in axis.lines if (line.get_gid() or "").startswith("model-point:")]


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


def short_family_report():
    data = example_report()
    specs = [
        ("egaroucid_l21", "E", "E21"),
        ("normal_egaroucid_alpha_1.0", "N", "N1.0"),
        ("loss_a_normal_alpha_0.2", "A", "A0.2"),
        ("loss_a_normal_alpha_1.0", "A", "A1.0"),
        ("loss_b_loss_a_alpha_1.0", "B", "B1.0"),
    ]
    data["models"].append(deepcopy(data["models"][-1]))
    for model, (model_id, group, short) in zip(data["models"], specs):
        model.update(model_id=model_id, group=group, short=short, model=short)
        for row in model["results"]:
            row["model_id"] = model_id
    data["models"][-1].update(elo=300, elo_low=288, elo_high=327)
    data["datasets"] = data.pop("dataset_counts")
    data.pop("series_paths")
    ids = [spec[0] for spec in specs]
    data["series"] = {"E": ids[:1], "N": ids[:2], "A": ids[1:4], "B": ids[3:]}
    data["method"] = {"elo": {"method": "pair_stratified_percentile_bootstrap"}}
    return data


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

    def test_short_family_schema_preserves_measurements_without_mutating_input(self):
        data = short_family_report()
        data["models"][0]["results"].reverse()
        original = deepcopy(data)
        normalized = validate_report(data)
        self.assertEqual(original, data)
        self.assertEqual(["Egaroucid", "normal", "loss_a_normal", "loss_a_normal", "loss_b_loss_a"],
                         [model["group"] for model in normalized["models"]])
        self.assertEqual(data["datasets"], normalized["dataset_counts"])
        self.assertEqual(data["series"]["B"], normalized["series_paths"]["loss_b_loss_a"])
        for model, result in zip(data["models"], normalized["models"]):
            for field in ("model_id", "elo", "elo_low", "elo_high"):
                self.assertEqual(model[field], result[field])
            self.assertEqual({row["dataset"]: row for row in model["results"]},
                             {row["dataset"]: row for row in result["results"]})
        data["series"]["Egaroucid"] = data["series"]["E"]
        with self.assertRaisesRegex(ValueError, "Duplicate series alias"):
            validate_report(data)

    def test_new_blend_families_share_endpoints_and_use_paper_alpha_labels(self):
        plots = ReportPlots(short_family_report())
        figure, axis = plt.subplots()
        plots.scatter_panel(axis, 0)
        self.assertEqual(5, len(model_points(axis)))
        for family, expected_x in (("loss_a_normal", [1000, 1200, 400]),
                                   ("loss_b_loss_a", [400, 300])):
            lines = [line for line in axis.lines if line.get_color() == STYLES[family]["color"]
                     and np.array_equal(line.get_xdata(), expected_x)]
            self.assertEqual(1, len(lines))
            np.testing.assert_allclose(expected_x, lines[0].get_xdata())
        self.assertTrue(all(model["color"] == "#D62728" for model in plots.models))
        self.assertEqual(["α=0.0（Egaroucid L21）", "α=1.0（通常モデル）", "α=1.2",
                          "α=2.0（石損モデルA）", "α=3.0（石損モデルB）"],
                         [label.get_text() for label in figure_annotations(figure).values()])
        figure = plots.threshold_trends()
        for axis, labels in zip(figure.axes[1:], (["α=0.0（Egaroucid L21）", "α=1.0（通常モデル）"],
                                                  ["α=1.0（通常モデル）", "α=1.2", "α=2.0（石損モデルA）"],
                                                  ["α=2.0（石損モデルA）", "α=3.0（石損モデルB）"])):
            self.assertEqual(labels, [line.get_label() for line in axis.lines])
        self.assertIn("対戦組ごと", plots.elo_ci_note)
        self.assertIn("percentile", plots.elo_ci_note)

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
        first_point = model_points(figure.axes[1])[0]
        self.assertAlmostEqual(30, float(first_point.get_ydata()[0]))
        self.assertEqual("#D62728", plots.index["alpha_0.0"]["color"])
        self.assertEqual("#D62728", labels["overall/alpha_0.0"].get_color())
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

    def test_shared_endpoints_have_one_marker_and_blend_path_preserves_alpha_order(self):
        plots = ReportPlots(example_report())
        figure, axis = plt.subplots()
        plots.scatter_panel(axis, 0)
        blend = [line for line in axis.lines if line.get_color() == STYLES["squared_blend"]["color"]
                  and len(line.get_xdata()) == 3][0]
        np.testing.assert_allclose([1000, 1200, 400], blend.get_xdata())
        np.testing.assert_allclose([63, 60, 45], blend.get_ydata())
        self.assertEqual(4, len(model_points(axis)))
        self.assertEqual("#D62728", plots.index["alpha_1.0"]["color"])
        self.assertEqual("#D62728", plots.index["loss_plus_one_squared_alpha_1.0"]["color"])

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
            names = {"alpha_0.0": "α=0.0（Egaroucid L21）", "alpha_1.0": "α=1.0（通常モデル）",
                     "loss_plus_one_squared_blend_alpha_0.2": "α=1.2",
                     "loss_plus_one_squared_alpha_1.0": "α=2.0（石損モデルA）", "alpha_0.2": "α=0.2"}
            expected = {f"{data['dataset_counts'][k]['dataset']}/{model['model_id']}": names.get(model["model_id"], model["short"])
                        for k in indices for model in data["models"]}
            self.assertEqual(expected, {key: label.get_text() for key, label in annotations.items()})
            self.assertEqual(set(expected), set(figure_annotations(figure)))
            self.assertTrue(all(label.get_visible() for label in annotations.values()))

    def test_pointwise_scatter_uses_saved_bounds_fifteen_percent_opacity_and_markers(self):
        plots = ReportPlots(example_report())
        figure, axis = plt.subplots()
        plots.scatter_panel(axis, 0)
        bands = [item for item in axis.collections if isinstance(item, PolyCollection)]
        self.assertEqual(2, len(bands))
        blend_band = bands[-1]
        self.assertAlmostEqual(.15, blend_band.get_alpha())
        np.testing.assert_allclose(blend_band.get_facecolor()[0], to_rgba("#D62728", .15))
        vertices = blend_band.get_paths()[0].vertices
        for x, low, high in ((1000, 60, 67), (1200, 57, 64), (400, 42, 49)):
            bounds = vertices[vertices[:, 0] == x, 1]
            self.assertAlmostEqual(low, bounds.min())
            self.assertAlmostEqual(high, bounds.max())
        self.assertFalse(any(isinstance(item, ErrorbarContainer) for item in axis.containers))
        first = model_points(axis)[0]
        np.testing.assert_allclose(first.get_xdata(), [2100])
        np.testing.assert_allclose(first.get_ydata(), [60])
        self.assertEqual(1.0, first.get_alpha())

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
        blend_axis = figure.axes[2]
        self.assertEqual(["α=1.0（通常モデル）", "α=1.2", "α=2.0（石損モデルA）"],
                         [line.get_label() for line in blend_axis.lines])
        self.assertTrue(all(line.get_color() == "#D62728" for line in blend_axis.lines))

    def test_alpha_override_and_invalid_alpha(self):
        plots = ReportPlots(example_report(), ci_alpha=.2)
        figure, axis = plt.subplots()
        plots.scatter_panel(axis, 0)
        self.assertAlmostEqual(.2, next(item for item in axis.collections if isinstance(item, PolyCollection)).get_alpha())
        self.assertFalse(any(isinstance(item, ErrorbarContainer) for item in axis.containers))
        self.assertTrue(all(point.get_alpha() == 1.0 for point in model_points(axis)))
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
            generated = render_report(source, output, dpi=45, ci_mode="pointwise")
            self.assertEqual(12, len(generated))
            for name in FIGURES:
                self.assertTrue((output / (name + ".png")).read_bytes().startswith(b"\x89PNG"))
                self.assertIn("<svg", (output / (name + ".svg")).read_text(encoding="utf-8"))
            manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256(original_bytes).hexdigest(), manifest["input_sha256"])
            self.assertEqual(original_bytes, source.read_bytes())
            self.assertEqual(.85, manifest["ci_transparency"])
            self.assertEqual(.15, manifest["ci_alpha"])
            self.assertEqual(1.0, manifest["ci_errorbar_alpha"])
            self.assertFalse(manifest["scatter_errorbars"])


if __name__ == "__main__":
    unittest.main()
