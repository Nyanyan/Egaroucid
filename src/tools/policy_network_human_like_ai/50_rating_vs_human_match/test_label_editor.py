import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from plot_rating_vs_human_match import ReportPlots, configure_plotting, render_report, save_figure
from label_editor import (
    LabelEditor, LabelPosition, annotation_position, apply_label_positions,
    apply_text_overrides, figure_text_targets, load_label_positions, load_text_overrides, save_label_positions,
)
from test_plot_rating_vs_human_match import example_report
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.text import Text


def drag_label(figure, annotation, dx=140, dy=60):
    figure.canvas.draw()
    box = Text.get_window_extent(annotation, figure.canvas.get_renderer())
    press = MouseEvent("button_press_event", figure.canvas, (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, button=1)
    figure.canvas.callbacks.process("button_press_event", press)
    move = MouseEvent("motion_notify_event", figure.canvas, press.x + dx, press.y + dy, button=1)
    figure.canvas.callbacks.process("motion_notify_event", move)
    release = MouseEvent("button_release_event", figure.canvas, move.x, move.y, button=1)
    figure.canvas.callbacks.process("button_release_event", release)


class LabelEditorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_plotting()

    def tearDown(self):
        plt.close("all")

    def test_drag_save_reload_and_reset_keep_panels_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positions_path = root / "labels.json"
            name = "elo_vs_agreement"
            figure = ReportPlots(example_report()).elo_vs_agreement()
            annotations = apply_label_positions(figure, {})
            key, other_key = "overall/alpha_1.0", "loss2/alpha_1.0"
            initial = annotation_position(annotations[key])
            other_initial = annotation_position(annotations[other_key])
            editor = LabelEditor(figure, annotations, name, positions_path,
                                 lambda: save_figure(figure, root, name, ("png", "svg"), 45))
            drag_label(figure, annotations[key])
            moved = annotation_position(annotations[key])
            self.assertAlmostEqual(initial.dx + 140 * 72 / figure.dpi, moved.dx)
            self.assertAlmostEqual(initial.dy + 60 * 72 / figure.dpi, moved.dy)
            self.assertEqual(other_initial, annotation_position(annotations[other_key]))
            self.assertTrue(annotations[key].arrow_patch.get_visible())
            self.assertIsNone(editor.drag_state)
            editor.on_key_press(SimpleNamespace(key="s"))
            self.assertTrue((root / (name + ".png")).is_file())
            self.assertTrue((root / (name + ".svg")).is_file())
            reloaded = load_label_positions(positions_path)
            self.assertEqual(moved, reloaded[name][key])
            fresh_figure = ReportPlots(example_report()).elo_vs_agreement()
            fresh_annotations = apply_label_positions(fresh_figure, reloaded[name])
            self.assertEqual(moved, annotation_position(fresh_annotations[key]))
            self.assertEqual(other_initial, annotation_position(fresh_annotations[other_key]))
            saved = positions_path.read_bytes()
            editor.on_key_press(SimpleNamespace(key="r"))
            self.assertEqual(initial, annotation_position(annotations[key]))
            self.assertEqual(saved, positions_path.read_bytes())

    def test_each_save_preserves_other_figures_and_rejects_nonfinite_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            position = LabelPosition(20, 30)
            save_label_positions(path, "elo_vs_agreement", {"loss2/alpha_1.0": position})
            save_label_positions(path, "mistake_elo_vs_agreement", {"loss2/alpha_1.0": LabelPosition(-10, -20)})
            document = load_label_positions(path)
            self.assertEqual(position, document["elo_vs_agreement"]["loss2/alpha_1.0"])
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["figures"]["elo_vs_agreement"]["loss2/alpha_1.0"]["dx"] = float("nan")
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                load_label_positions(path)

    def test_batch_render_reuses_saved_positions_and_reset_does_not_overwrite_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report_data.json"
            source.write_text(json.dumps(example_report()), encoding="utf-8")
            positions_path = root / "label_positions.json"
            key = "overall/alpha_1.0"
            position = LabelPosition(60, -20, "right", "top")
            save_label_positions(positions_path, "elo_vs_agreement", {key: position})
            original = positions_path.read_bytes()
            render_report(source, root, figures=["elo_vs_agreement"], dpi=45)
            manifest = json.loads((root / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(60, manifest["label_positions"]["elo_vs_agreement"][key]["dx"])
            self.assertEqual("right", manifest["label_positions"]["elo_vs_agreement"][key]["horizontal_alignment"])
            render_report(source, root, figures=["elo_vs_agreement"], dpi=45, reset_labels=True)
            manifest = json.loads((root / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertNotEqual(60, manifest["label_positions"]["elo_vs_agreement"][key]["dx"])
            self.assertEqual(original, positions_path.read_bytes())

    def test_double_click_renames_and_saves_text_without_changing_model_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "elo_vs_agreement"
            figure = ReportPlots(example_report()).elo_vs_agreement()
            annotations = apply_label_positions(figure, {})
            key, other_key = "overall/alpha_1.0", "loss2/alpha_1.0"
            label = annotations[key]
            original = annotation_position(label)
            editor = LabelEditor(figure, annotations, name, root / "label_positions.json",
                                 lambda: save_figure(figure, root, name, ("svg",), 45))
            figure.canvas.draw()
            box = Text.get_window_extent(label, figure.canvas.get_renderer())
            event = MouseEvent("button_press_event", figure.canvas,
                               (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, button=1, dblclick=True)
            renamed = "通常NN（名前は自由）"
            with patch("label_editor.request_label_text", return_value=renamed) as dialog:
                figure.canvas.callbacks.process("button_press_event", event)
            dialog.assert_called_once_with(figure, "N1.0")
            self.assertEqual(renamed, label.get_text())
            self.assertEqual("N1.0", annotations[other_key].get_text())
            self.assertEqual((original.dx, original.dy), label.xyann)
            editor.on_key_press(SimpleNamespace(key="s"))
            positions = load_label_positions(root / "label_positions.json")
            self.assertEqual(renamed, positions[name][key].text)
            fresh_figure = ReportPlots(example_report()).elo_vs_agreement()
            fresh = apply_label_positions(fresh_figure, positions[name])
            self.assertEqual(renamed, fresh[key].get_text())
            with patch("label_editor.request_label_text", return_value=None):
                editor.on_key_press(SimpleNamespace(key="e"))
            self.assertEqual(renamed, label.get_text())
            editor.on_key_press(SimpleNamespace(key="r"))
            self.assertEqual("N1.0", label.get_text())
            self.assertIn(renamed, (root / (name + ".svg")).read_text(encoding="utf-8"))

    def test_older_positions_without_names_still_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({"version": 2, "figures": {"elo_vs_agreement": {
                "overall/alpha_1.0": {"dx": 40, "dy": 60, "horizontal_alignment": "right",
                                       "vertical_alignment": "top"}}}}), encoding="utf-8")
            figure = ReportPlots(example_report()).elo_vs_agreement()
            annotations = apply_label_positions(figure, load_label_positions(path)["elo_vs_agreement"])
            label = annotations["overall/alpha_1.0"]
            self.assertEqual("N1.0", label.get_text())
            self.assertEqual((40, 60), label.xyann)

    def test_double_click_edits_titles_axes_legends_ticks_and_notes_and_persists_after_redraw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "elo_vs_agreement"
            figure = ReportPlots(example_report()).elo_vs_agreement()
            annotations = apply_label_positions(figure, {})
            targets = apply_text_overrides(figure, {})
            keys = ["figure/title", "axes/dataset:overall/title", "axes/dataset:overall/x/label",
                    "axes/dataset:overall/y/label", "figure/legend/0/label/0", "figure/text/1",
                    "axes/dataset:overall/x/major/0/first"]
            originals = {key: targets[key].artist.get_text() for key in keys}
            editor = LabelEditor(figure, annotations, name, root / "labels.json",
                                 lambda: save_figure(figure, root, name, ("svg",), 45), targets)
            edited = {}
            for i, key in enumerate(keys):
                figure.canvas.draw()
                box = Text.get_window_extent(targets[key].artist, figure.canvas.get_renderer())
                event = MouseEvent("button_press_event", figure.canvas,
                                   (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, button=1, dblclick=True)
                edited[key] = f"変更した文字{i}"
                with patch("label_editor.request_label_text", return_value=edited[key]) as dialog:
                    figure.canvas.callbacks.process("button_press_event", event)
                dialog.assert_called_once_with(figure, originals[key])
                figure.canvas.draw()
                self.assertEqual(edited[key], targets[key].artist.get_text())
            editor.save()
            overrides = load_text_overrides(root / "labels.json")
            fresh = ReportPlots(example_report()).elo_vs_agreement()
            reloaded = apply_text_overrides(fresh, overrides[name])
            fresh.canvas.draw()
            for key, text in edited.items():
                self.assertEqual(text, reloaded[key].artist.get_text())
            editor.reset()
            figure.canvas.draw()
            for key, text in originals.items():
                self.assertEqual(text, targets[key].artist.get_text())
            source = root / "report_data.json"
            source.write_text(json.dumps(example_report()), encoding="utf-8")
            render_report(source, root, figures=[name], formats=["svg"], label_positions_path=root / "labels.json")
            saved_svg = (root / (name + ".svg")).read_text(encoding="utf-8")
            for text in edited.values():
                self.assertIn(text, saved_svg)

    def test_all_six_figures_have_editable_titles_and_table_labels(self):
        from plot_rating_vs_human_match import EDITABLE_FIGURES, FIGURES
        self.assertEqual(set(FIGURES), set(EDITABLE_FIGURES))
        plots = ReportPlots(example_report())
        for name in FIGURES:
            figure = getattr(plots, name)()
            targets = figure_text_targets(figure)
            self.assertIn("figure/title", targets)
            self.assertTrue(any(key.startswith("axes/") for key in targets))
            if name == "agreement_heatmap":
                self.assertTrue(any("/text/" in key for key in targets))
                self.assertTrue(any("/y/major/" in key for key in targets))
            elif name == "threshold_trends":
                self.assertTrue(any("/legend/label/" in key for key in targets))
            plt.close(figure)

    def test_editor_saves_through_render_entrypoint_and_close_alone_saves_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report_data.json"
            source.write_text(json.dumps(example_report()), encoding="utf-8")

            def edit_current_window(**_kwargs):
                self.assertEqual([], plt.rcParams["keymap.save"])
                editor = plt.gcf()._label_editor
                drag_label(editor.figure, editor.annotations["overall/alpha_1.0"])
                editor.on_key_press(SimpleNamespace(key="ctrl+s"))

            with patch("plot_rating_vs_human_match.enable_interactive_backend"), patch(
                "plot_rating_vs_human_match.plt.show", side_effect=edit_current_window
            ):
                generated = render_report(source, root, figures=["elo_vs_agreement"], dpi=45, adjust_labels=True)
            self.assertEqual(2, len(generated))
            self.assertTrue((root / "label_positions.json").is_file())
            manifest = json.loads((root / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertGreater(manifest["label_positions"]["elo_vs_agreement"]["overall/alpha_1.0"]["dx"], 6)
            originals = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
            with patch("plot_rating_vs_human_match.enable_interactive_backend"), patch("plot_rating_vs_human_match.plt.show"):
                generated = render_report(source, root, figures=["elo_vs_agreement"], adjust_labels=True, reset_labels=True)
            self.assertEqual([], generated)
            for filename, contents in originals.items():
                self.assertEqual(contents, (root / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
