"""Guard scientific cohort selection, preserved uncertainty, and paper editing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from label_editor import LabelPosition, apply_label_positions, arrange_unplaced_labels, save_label_positions
from matplotlib.text import Text
from paper_campaign import (
    ALPHA_NAMES, BETA_NAMES, CURVES, DEFAULT_CAMPAIGN, DEFAULT_FONT_SIZE, DEFAULT_OUTPUT,
    OUTPUT_STEMS, PAPER_CURVES, PAPER_FIGURES,
    create_paper_figure, load_paper_data, mark_best_agreement, paper_curve_bounds, paper_y_limits,
    point_label, render_paper, validate_measurements,
)
from plot_rating_vs_human_match import make_argparser
from test_label_editor import drag_label


@unittest.skipUnless((DEFAULT_CAMPAIGN / "quest_all_100k/priority_validation.json").is_file(),
                     "The verified experiment artifacts are not installed")
class PaperCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_paper_data()

    def tearDown(self):
        plt.close("all")

    def test_default_selects_five_latest_cohorts_and_original_24_player_elo(self):
        self.assertIsNone(make_argparser().parse_args([]).input)
        self.assertFalse(hasattr(make_argparser().parse_args([]), "paper_png_dir"))
        self.assertEqual(make_argparser().parse_args([]).font_size, DEFAULT_FONT_SIZE)
        self.assertEqual(list(self.data["datasets"]), list(PAPER_FIGURES))
        self.assertEqual(len(self.data["models"]), 24)
        rows = {(k, r["model"]): r for k, d in self.data["datasets"].items() for r in d["rows"]}
        self.assertEqual(len(rows), 120)
        self.assertEqual(rows["wthor_100k", "alpha_0.8"]["hits"], "66597")
        self.assertEqual(rows["quest_800-1000", "alpha_2.0"]["hits"], "36421")
        self.assertEqual(rows["quest_1400-1600", "alpha_1.0"]["hits"], "51837")
        self.assertEqual(rows["quest_2000-2200", "alpha_1.0"]["hits"], "65989")
        self.assertEqual({d["positions"] for k, d in self.data["datasets"].items() if k.startswith("quest_")}, {100000})
        self.assertNotIn("neural_choice_filter", str(self.data["sources"]))
        self.assertFalse(any(k.startswith("wthor_full_test") for k in self.data["sources"]))

    def test_wrong_sample_partial_cells_and_mismatched_band_are_rejected(self):
        key = "quest_800-1000"
        path = DEFAULT_CAMPAIGN / "quest_all_100k" / (key + "_priority_predictions.npz")
        with np.load(path, allow_pickle=False) as z:
            names = list(map(str, z["model_names"]))
            pred, masks = z["predicted_policies"], z["symmetry_target_masks"]
        original = {r["model"]: r for r in self.data["datasets"][key]["rows"]}
        curves = {g:self.data["datasets"][key]["bands"][g] for g in CURVES}
        hits = ((masks[:, None] >> pred.astype(np.uint64)) & np.uint64(1)) != 0
        signature = hashlib.sha256(hits.tobytes()).hexdigest()
        validate_measurements(key, names, pred, masks, original, curves, signature)
        rows = deepcopy(original); rows[names[0]]["complete"] = "False"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_measurements(key, names, pred, masks, rows, curves, signature)
        wrong = pred.copy(); wrong[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "Missing predictions"):
            validate_measurements(key, names, wrong, masks, original, curves, signature)
        bands = deepcopy(curves); bands["proposed"]["point"][0][1] += .01
        with self.assertRaisesRegex(ValueError, "coordinates mismatch"):
            validate_measurements(key, names, pred, masks, original, bands, signature)
        with self.assertRaisesRegex(ValueError, "Bootstrap cohort mismatch"):
            validate_measurements(key, names, pred, masks, original, curves, "old test-only sample")

    def test_all_models_labeled_once_parameter_order_and_saved_edits_preserved(self):
        key = "wthor_100k"
        fig, labels = create_paper_figure(self.data, key)
        self.assertEqual(len(labels), 24)
        ax = fig.axes[0]
        self.assertEqual(sum((line.get_gid() or "").startswith("model-point:") for line in ax.lines), 24)
        lines = {line.get_gid(): line for line in ax.lines}
        rows = {r["model"]: r for r in self.data["datasets"][key]["rows"]}
        np.testing.assert_allclose(lines["curve:level21_normal"].get_xdata(), [float(rows[n]["elo"]) for n in ALPHA_NAMES[:8]])
        np.testing.assert_allclose(lines["curve:normal_level1"].get_ydata(), [100 * float(rows[n]["accuracy"]) for n in BETA_NAMES])
        np.testing.assert_allclose(lines["curve:normal_loss"].get_xdata(), [float(rows[n]["elo"]) for n in ALPHA_NAMES[7:]])
        eg = lines["curve:egaroucid"]
        names = PAPER_CURVES["egaroucid"]["names"]
        self.assertEqual(names[:2], ["egaroucid_l1", "egaroucid_l2"])
        self.assertEqual(names[-2:], ["egaroucid_l17", "alpha_0.0"])
        np.testing.assert_allclose(eg.get_xdata(), [float(rows[n]["elo"]) for n in names])
        np.testing.assert_allclose(eg.get_ydata(), [100 * float(rows[n]["accuracy"]) for n in names])
        self.assertEqual(lines["model-point:egaroucid_l1"].get_color(), PAPER_CURVES["normal_level1"]["color"])
        self.assertEqual(len(ax.patches), 4)
        self.assertEqual(len(ax.containers), 0)  # No cross-shaped error bars.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary); uid = key + "/alpha_0.8"
            saved = {uid: LabelPosition(-24, 31, "right", "bottom", "編集したα=0.8")}
            save_label_positions(output / "label_positions.json", key, saved)
            with patch("paper_campaign.load_paper_data", return_value=self.data):
                render_paper(output_dir=output, figures=[key], formats=["svg", "png"], dpi=45)
            manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["label_positions"][key][uid]["dx"], -24)
            self.assertIn("編集したα=0.8", (output / (OUTPUT_STEMS[key] + ".svg")).read_text(encoding="utf-8"))
            png = OUTPUT_STEMS[key] + ".png"
            self.assertTrue((output / png).is_file())
            self.assertFalse((output / "paper_img").exists())
            self.assertIs(manifest["automatic_copy"], False)

    def test_paper_names_local_alphas_distinct_styles_and_no_title(self):
        self.assertEqual(DEFAULT_OUTPUT.name, "gpw2026_paper_figures")
        self.assertEqual([PAPER_CURVES[k]["label"] for k in PAPER_CURVES],
                         ["レベル21-通常モデル", "通常モデル-レベル1", "通常モデル-石損モデル", "Egaroucid"])
        self.assertEqual(point_label("alpha_1.2"), "α=0.2")
        self.assertEqual(point_label("beta_l1_0.4"), "α=0.4")
        self.assertEqual(point_label("alpha_1.0"), "α=1.0（通常モデル）")
        self.assertEqual(point_label("alpha_2.0"), "α=1.0（石損モデル）")
        self.assertEqual(point_label("alpha_0.0"), "α=0.0（Lv. 21）")
        self.assertEqual(point_label("egaroucid_l1"), "α=1.0\n（Lv. 1）")
        self.assertEqual(point_label("egaroucid_l17"), "Lv. 17")
        fig, labels = create_paper_figure(self.data, "wthor_100k")
        ax = fig.axes[0]
        self.assertEqual(ax.xaxis.label.get_fontsize(), 24)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 24)
        self.assertEqual(ax.get_xticklabels()[0].get_fontsize(), 24)
        self.assertEqual(ax.get_yticklabels()[0].get_fontsize(), 24)
        self.assertTrue(all(a.get_fontsize() == DEFAULT_FONT_SIZE - 1 for a in labels.values()))
        self.assertTrue(all(t.get_fontsize() == 24 for t in ax.get_legend().get_texts()))
        self.assertEqual(ax.get_title(), "")
        self.assertEqual(len(fig.texts), 0)
        lines = [line for line in ax.lines if (line.get_gid() or "").startswith("curve:")]
        self.assertEqual(len({line.get_color() for line in lines}), 4)
        self.assertEqual(len({line.get_linestyle() for line in lines}), 4)
        for group, spec in PAPER_CURVES.items():
            displayed = paper_curve_bounds(self.data["datasets"]["wthor_100k"], spec)
            original = self.data["datasets"]["wthor_100k"]["bands"][spec["source"]]
            self.assertEqual(displayed["critical_value"], original["critical_value"])
            self.assertEqual(displayed["point"], [original["point"][i] for i in spec["indices"]])

    def test_paper_editor_drag_rename_legend_save_and_reload(self):
        key = "quest_1400-1600"
        uid = key + "/alpha_1.4"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def edit(**kwargs):
                editor = plt.gcf()._label_editor
                label = editor.annotations[uid]
                drag_label(editor.figure, label)
                editor.selected_annotation = label
                with patch("label_editor.request_label_text", return_value="変更したモデル名"):
                    editor.on_key_press(SimpleNamespace(key="e"))
                legend_text = editor.figure.axes[0].get_legend().get_texts()[0]
                target = next(t for t in editor.text_targets.values() if t.artist is legend_text)
                target.set_text("変更した系列名")
                editor.on_key_press(SimpleNamespace(key="ctrl+s"))

            with patch("paper_campaign.load_paper_data", return_value=self.data), patch(
                    "plot_rating_vs_human_match.enable_interactive_backend"), patch(
                    "paper_campaign.plt.show", side_effect=edit):
                render_paper(output_dir=output, figures=[key], formats=["svg"], dpi=45, adjust_labels=True)
            positions = (output / "label_positions.json").read_bytes()
            with patch("paper_campaign.load_paper_data", return_value=self.data):
                render_paper(output_dir=output, figures=[key], formats=["svg"], dpi=45)
            self.assertEqual(positions, (output / "label_positions.json").read_bytes())
            svg = (output / (OUTPUT_STEMS[key] + ".svg")).read_text(encoding="utf-8")
            self.assertIn("変更したモデル名", svg)
            self.assertIn("変更した系列名", svg)
            self.assertNotIn("<title>", svg)

    def test_manual_highest_label_and_position_survive_save_and_reopen(self):
        key = "wthor_100k"
        uid = key + "/alpha_0.8"
        edited_text = "α=0.8（最高）\n66.60%"
        edited_position = (12, 12)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def edit(**kwargs):
                editor = plt.gcf()._label_editor
                label = editor.annotations[uid]
                editor.selected_annotation = label
                with patch("label_editor.request_label_text", return_value=edited_text):
                    editor.on_key_press(SimpleNamespace(key="e"))
                label.set_position(edited_position)
                editor.on_key_press(SimpleNamespace(key="ctrl+s"))
                self.assertEqual(label.get_text(), edited_text)
                self.assertEqual(label.xyann, edited_position)

            with patch("paper_campaign.load_paper_data", return_value=self.data), patch(
                    "plot_rating_vs_human_match.enable_interactive_backend"), patch(
                    "paper_campaign.plt.show", side_effect=edit):
                render_paper(output_dir=output, figures=[key], formats=["svg", "pdf"], dpi=45, adjust_labels=True)
            saved_bytes = (output / "label_positions.json").read_bytes()
            saved = json.loads(saved_bytes)["figures"][key][uid]
            self.assertEqual(saved["text"], edited_text)
            with patch("paper_campaign.load_paper_data", return_value=self.data):
                render_paper(output_dir=output, figures=[key], formats=["svg", "pdf"], dpi=45)
            self.assertEqual((output / "label_positions.json").read_bytes(), saved_bytes)
            manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
            actual = manifest["label_positions"][key][uid]
            self.assertEqual(actual["text"], edited_text)
            self.assertEqual((actual["dx"], actual["dy"]), edited_position)
            svg = (output / (OUTPUT_STEMS[key] + ".svg")).read_text(encoding="utf-8")
            for line in edited_text.splitlines():
                self.assertIn(line, svg)

    def test_highest_value_is_restored_without_overwriting_custom_name_or_position(self):
        key = "wthor_100k"
        uid = key + "/alpha_0.8"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_label_positions(output / "label_positions.json", key,
                {uid:LabelPosition(18.72, 25.92, "right", "bottom", "編集したモデル名")})
            original = (output / "label_positions.json").read_bytes()
            with patch("paper_campaign.load_paper_data", return_value=self.data):
                render_paper(output_dir=output, figures=[key], formats=["svg"], dpi=45)
            manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
            actual = manifest["label_positions"][key][uid]
            self.assertEqual(actual["text"], "編集したモデル名, 66.60%")
            self.assertEqual((actual["dx"], actual["dy"]), (18.72, 25.92))
            self.assertEqual((output / "label_positions.json").read_bytes(), original)

            def edit(**kwargs):
                editor = plt.gcf()._label_editor
                label = editor.annotations[uid]
                label.set_text("再編集したモデル名")
                label.set_position((32, 47))
                editor.on_key_press(SimpleNamespace(key="ctrl+s"))
                self.assertEqual(label.get_text(), "再編集したモデル名, 66.60%")

            with patch("paper_campaign.load_paper_data", return_value=self.data), patch(
                    "plot_rating_vs_human_match.enable_interactive_backend"), patch(
                    "paper_campaign.plt.show", side_effect=edit):
                render_paper(output_dir=output, figures=[key], formats=["svg"], dpi=45, adjust_labels=True)
            saved = json.loads((output / "label_positions.json").read_text(encoding="utf-8"))["figures"][key][uid]
            self.assertEqual(saved["text"], "再編集したモデル名, 66.60%")
            self.assertEqual((saved["dx"], saved["dy"]), (32, 47))

    def test_legends_inside_axes_clear_of_curves_bands_and_labels(self):
        for key in PAPER_FIGURES:
            with self.subTest(dataset=key):
                fig, labels = create_paper_figure(self.data, key)
                arrange_unplaced_labels(fig, labels, set())
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                ax = fig.axes[0]
                lo, hi = paper_y_limits(self.data["datasets"][key])
                self.assertEqual(ax.get_ylim(), (lo, hi))
                self.assertEqual(lo % 10, 0)
                self.assertEqual(hi % 10, 0)
                np.testing.assert_array_equal(ax.get_yticks(), np.arange(lo, hi + 1, 10))
                for bounds in self.data["datasets"][key]["bands"].values():
                    self.assertGreaterEqual(np.min(np.array(bounds["lower"])[:, 1] * 100), lo)
                    self.assertLessEqual(np.max(np.array(bounds["upper"])[:, 1] * 100), hi)
                box = ax.get_legend().get_window_extent(renderer).padded(3)
                bounds = ax.get_window_extent(renderer)
                self.assertTrue(bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1))
                for artist in [*ax.lines, *ax.patches]:
                    path = artist.get_transform().transform_path(artist.get_path())
                    self.assertFalse(path.intersects_bbox(box, filled=artist in ax.patches), artist.get_gid())
                for label in labels.values():
                    self.assertFalse(Text.get_window_extent(label, renderer).overlaps(box), label.get_text())
                    if label.arrow_patch:
                        arrow = label.arrow_patch
                        self.assertFalse(arrow.get_transform().transform_path(arrow.get_path()).intersects_bbox(box, filled=False))
                plt.close(fig)

    def test_lv21_endpoint_is_covered_by_the_joint_eight_vertex_tube(self):
        spec = PAPER_CURVES["egaroucid"]
        for key in PAPER_FIGURES:
            with self.subTest(dataset=key):
                dataset = self.data["datasets"][key]
                bounds = paper_curve_bounds(dataset, spec)
                rows = {r["model"]:r for r in dataset["rows"]}
                expected = [[float(rows[n]["elo"]), float(rows[n]["accuracy"])] for n in spec["names"]]
                np.testing.assert_allclose(bounds["point"], expected)
                self.assertEqual(np.asarray(bounds["lower"]).shape, (8, 2))
                self.assertEqual(np.asarray(bounds["upper"]).shape, (8, 2))
                self.assertGreaterEqual(bounds["bootstrap_max_coverage"], .95)
                fig, _ = create_paper_figure(self.data, key)
                ax = fig.axes[0]
                patch = next(p for p in ax.patches if p.get_gid() == "simultaneous-band:egaroucid")
                xy = ax.transData.transform((expected[-1][0], expected[-1][1] * 100))
                self.assertTrue(patch.get_transform().transform_path(patch.get_path()).contains_point(xy))
                broken = deepcopy(dataset)
                for field in ("point", "lower", "upper", "standard_error"):
                    broken["bands"][spec["source"]][field].pop()
                with self.assertRaisesRegex(ValueError, "missing a displayed endpoint"):
                    paper_curve_bounds(broken, spec)
                plt.close(fig)

    def test_highest_agreement_percentages_and_edited_names(self):
        expected = {
            "wthor_100k": ("alpha_0.8", "66.60%"),
            "wthor_loss_ge_2": ("alpha_2.0", "45.06%"),
            "quest_800-1000": ("alpha_2.0", "36.42%"),
            "quest_1400-1600": ("alpha_1.4", "52.85%"),
            "quest_2000-2200": ("alpha_0.9", "67.08%"),
        }
        for key, (name, percent) in expected.items():
            with self.subTest(dataset=key):
                fig, labels = create_paper_figure(self.data, key)
                label = labels[key + "/" + name]
                suffix = ", " + percent if key == "wthor_100k" else ""
                self.assertEqual(label.get_text(), point_label(name) + suffix)
                label.set_text("編集済みのモデル名, 12.34%")
                for _ in range(2):
                    mark_best_agreement(self.data["datasets"][key], labels)
                    self.assertEqual(label.get_text(), "編集済みのモデル名" + suffix)
                    self.assertEqual(sum("%" in a.get_text() for a in labels.values()), 4 if key == "wthor_100k" else 0)
                plt.close(fig)

    def test_non_wthor_overall_saved_maximum_percent_is_hidden_without_moving_labels(self):
        for key in PAPER_FIGURES:
            if key == "wthor_100k":
                continue
            with self.subTest(dataset=key), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                row = max(self.data["datasets"][key]["rows"], key=lambda r:int(r["hits"]))
                uid = key + "/" + row["model"]
                text = f"編集したモデル名, {100*float(row['accuracy']):.2f}%"
                save_label_positions(output / "label_positions.json", key,
                    {uid:LabelPosition(24, 31, "right", "top", text)})
                original = (output / "label_positions.json").read_bytes()
                with patch("paper_campaign.load_paper_data", return_value=self.data):
                    render_paper(output_dir=output, figures=[key], formats=["svg"], dpi=45)
                m = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
                actual = m["label_positions"][key][uid]
                self.assertEqual(actual["text"], "編集したモデル名")
                self.assertEqual((actual["dx"], actual["dy"]), (24, 31))
                self.assertEqual((output / "label_positions.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
