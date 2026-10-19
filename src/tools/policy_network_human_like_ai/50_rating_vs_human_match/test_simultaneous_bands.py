"""Statistical, geometry, source-integrity and rendering checks for the tubes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.path import Path as MplPath
import numpy as np

from label_editor import apply_text_overrides
from plot_rating_vs_human_match import ReportPlots, render_report, validate_report
from simultaneous_bands import (bootstrap_agreement, build_simultaneous_bands, calibrate_tube,
                                confidence_paths, draw_tube, tube_polygons)
from test_plot_rating_vs_human_match import example_report


def source_fixture(root):
    data = example_report()
    ids = [model["model_id"] for model in data["models"]]
    # Source columns deliberately differ from report order.
    rng = np.random.default_rng(9)
    estimates = np.array([model["elo"] for model in data["models"]])
    elo = estimates + rng.normal(size=(1600, 4)) * 10
    bounds = np.quantile(elo, [.025, .975], axis=0)
    name = "strength_elo_bootstrap_replicates.csv"
    np.savetxt(root / name, np.column_stack((np.arange(len(elo)), elo[:, ::-1])), delimiter=",",
               header="replicate," + ",".join(ids[::-1]), comments="")
    hits = (np.arange(16)[:, None] & (1 << np.arange(4))) != 0
    moves = np.where(np.tile(hits, (2, 1)), 0, 1).astype(np.int8)
    for filename, losses in (("wthor_test_100000_predictions.npz", np.array([], dtype=np.uint8)),
                             ("mistakes_predictions.npz", np.repeat([2, 4], 16).astype(np.uint8))):
        np.savez(root / filename, model_names=np.array(ids[::-1]), predicted_policies=moves[:, ::-1],
                 symmetry_target_masks=np.ones(32, dtype=np.uint64), losses=losses)
    for k, dataset in enumerate(data["dataset_counts"]):
        dataset["positions"] = 16 if k == 2 else 32
    for i, model in enumerate(data["models"]):
        model["elo_low"], model["elo_high"] = bounds[:, i].tolist()
        for row, dataset in zip(model["results"], data["dataset_counts"]):
            row.update(positions=dataset["positions"], top1_accuracy=.5, top1_ci95_lower=.3, top1_ci95_upper=.7)
    data["sources"] = [dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in root.iterdir() if p.suffix in (".csv", ".npz")]
    report = root / "report_data.json"
    report.write_text(json.dumps(data), encoding="utf-8")
    return validate_report(data), report


class SimultaneousBandsTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_shared_proposed_endpoints_are_one_family_without_sorting_elo(self):
        paths = {"normal": ["e", "n"], "loss_a_normal": ["n", "a0", "a"],
                 "loss_b_loss_a": ["a", "b"], "Egaroucid": ["e1", "e"]}
        curves = confidence_paths(paths)
        self.assertEqual(["e", "n", "a0", "a", "b"], curves["proposed"])
        self.assertEqual(["e1", "e"], curves["Egaroucid"])
        paths["loss_b_loss_a"][0] = "unshared"
        with self.assertRaisesRegex(ValueError, "share adjacent"):
            confidence_paths(paths)

    def test_joint_row_bootstrap_preserves_perfect_model_correlations(self):
        column = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
        hits = np.column_stack((column, column, ~column))
        samples = bootstrap_agreement(hits, 12000, np.random.default_rng(42))
        np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
        np.testing.assert_allclose(samples[:, 0] + samples[:, 2], 1)
        self.assertAlmostEqual(.5, samples[:, 0].mean(), delta=.005)
        self.assertAlmostEqual(np.sqrt(.25 / len(hits)), samples[:, 0].std(), delta=.005)
        np.testing.assert_array_equal(samples[:128], bootstrap_agreement(hits, 128, np.random.default_rng(42)))

    def test_maximum_calibration_covers_both_coordinates_and_all_vertices(self):
        rng = np.random.default_rng(85)
        point = np.tile([1200, .5], (7, 1))
        draws = point + rng.normal(size=(20000, 7, 2)) * [20, .02]
        result = calibrate_tube(point, draws)
        inside = np.all((draws >= result["lower"]) & (draws <= result["upper"]), axis=(1, 2))
        self.assertGreaterEqual(inside.mean(), .95)
        self.assertLess(inside.mean(), .952)
        self.assertGreater(result["critical_value"], 2.8)
        single = calibrate_tube(point[:1], draws[:, :1])
        self.assertGreater(result["critical_value"], single["critical_value"])
        # Duplicating perfectly dependent parameters must not add a multiplicity penalty.
        duplicate = calibrate_tube(np.repeat(point[:1], 7, axis=0), np.repeat(draws[:, :1], 7, axis=1))
        self.assertAlmostEqual(single["critical_value"], duplicate["critical_value"])
        # Unit conversion cannot change calibration.
        scaled = calibrate_tube(point * [1, 100], draws * [1, 100])
        self.assertAlmostEqual(result["critical_value"], scaled["critical_value"])

    def test_swept_rectangles_cover_arbitrary_interpolated_endpoints_and_x_caps(self):
        lower = np.array([[9, .2], [3, .6], [7, .1]])
        upper = lower + [2, .1]
        polygons = tube_polygons(lower, upper)
        self.assertEqual(2, len(polygons))
        rng = np.random.default_rng(7)
        for j, polygon in enumerate(polygons):
            path = MplPath(polygon)
            for t in np.linspace(0, 1, 11):
                a = rng.uniform(lower[j], upper[j], (100, 2))
                b = rng.uniform(lower[j + 1], upper[j + 1], (100, 2))
                self.assertTrue(path.contains_points((1 - t) * a + t * b, radius=1e-9).all())
        self.assertEqual(3, min(p[:, 0].min() for p in polygons))
        self.assertEqual(11, max(p[:, 0].max() for p in polygons))

    def test_repeated_experiments_have_reasonable_nominal_joint_coverage(self):
        # Independent validation against known parameters, not the same draws
        # used for calibration. This checks a regular synthetic setting only.
        rng = np.random.default_rng(214)
        truth = np.array([[1000, .2], [1200, .5], [1400, .7]])
        covered = 0
        for _ in range(300):
            error = rng.normal(size=3) * 8
            point_elo = truth[:, 0] + error - error.mean()
            hits = rng.uniform(size=(400, 1)) < truth[:, 1]
            point = np.column_stack((point_elo, hits.mean(axis=0)))
            elo_errors = rng.normal(size=(1200, 3)) * 8
            elo_draws = point_elo + elo_errors - elo_errors.mean(axis=1, keepdims=True)
            agreement_draws = bootstrap_agreement(hits, 1200, rng)
            bounds = calibrate_tube(point, np.stack((elo_draws, agreement_draws), axis=2))
            covered += np.all((truth >= bounds["lower"]) & (truth <= bounds["upper"]))
        self.assertGreaterEqual(covered / 300, .90)
        self.assertLessEqual(covered / 300, .99)

    def test_overlapping_segments_are_painted_once(self):
        figure = plt.figure(figsize=(4, 2), dpi=100)
        axis = figure.add_axes([0, 0, 1, 1])
        axis.set(xlim=(0, 4), ylim=(0, 100))
        axis.axis("off")
        bounds = dict(lower=[[.5, .25], [1.5, .25], [2.5, .25]],
                      upper=[[1.5, .75], [2.5, .75], [3.5, .75]])
        draw_tube(axis, bounds, color="red", alpha=.1)
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())
        # x=1 is in one segment; x=2 is in two. Both must have alpha .1.
        np.testing.assert_array_equal(pixels[100, 100], pixels[100, 200])
        self.assertTrue(225 <= pixels[100, 200, 1] <= 232)

    def test_sources_cache_and_report_measurements_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, report = source_fixture(root)
            original = deepcopy(data)
            cache = root / "bands.json"
            result = build_simultaneous_bands(data, report, cache)
            self.assertEqual(original, data)
            self.assertEqual(1600, result["replicates"])
            self.assertEqual(4, len(result["signature"]["curves"]["proposed"]))
            with patch("simultaneous_bands.bootstrap_agreement", side_effect=AssertionError("cache missed")):
                self.assertEqual(result, build_simultaneous_bands(data, report, cache))
            # Point estimates, not only source filenames, participate in the cache key.
            changed = deepcopy(data)
            changed["models"][0]["results"][0]["top1_accuracy"] += .01
            with self.assertRaisesRegex(ValueError, "accuracy do not match"):
                build_simultaneous_bands(changed, report, cache)
            source = root / "wthor_test_100000_predictions.npz"
            source.write_bytes(source.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_simultaneous_bands(data, report, cache)

    def test_scatter_defaults_to_simultaneous_with_markers_only_and_preserves_text_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, report = source_fixture(root)
            output = root / "output"
            generated = render_report(report, output, figures=["elo_vs_agreement"], dpi=45)
            self.assertEqual(["elo_vs_agreement.png", "elo_vs_agreement.svg"], generated)
            manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("simultaneous", manifest["ci_mode"])
            self.assertEqual(1600, manifest["simultaneous_bands"]["replicates"])
            bands = json.loads((output / "simultaneous_band_data.json").read_text(encoding="utf-8"))
            plots = ReportPlots(data, simultaneous_bands=bands)
            figure = plots.elo_vs_agreement()
            self.assertEqual(2, len(figure.axes[0].patches))
            self.assertTrue(all(patch.get_alpha() == .15 for patch in figure.axes[0].patches))
            self.assertFalse(any(isinstance(item, ErrorbarContainer) for axis in figure.axes for item in axis.containers))
            for axis in figure.axes:
                self.assertEqual(len(data["models"]), sum((line.get_gid() or "").startswith("model-point:") for line in axis.lines))
            texts = apply_text_overrides(figure, {"figure/text/1": "old pointwise note", "figure/title": "edited",
                                                "figure/text/joint-ci-note-v1": "old cross errorbar note"})
            self.assertIn("figure/text/joint-ci-note-v2", texts)
            self.assertIn("95%同時信頼帯", texts["figure/text/joint-ci-note-v2"].artist.get_text())
            self.assertNotIn("十字", texts["figure/text/joint-ci-note-v2"].artist.get_text())
            self.assertEqual("edited", figure._suptitle.get_text())
            for axis in figure.axes:
                for bounds in bands["datasets"][axis.get_gid().split(":", 1)[1]].values():
                    self.assertLessEqual(axis.get_xlim()[0], min(row[0] for row in bounds["lower"]))
                    self.assertGreaterEqual(axis.get_xlim()[1], max(row[0] for row in bounds["upper"]))

    def test_missing_raw_sources_fail_instead_of_silently_drawing_old_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text(json.dumps(example_report()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "require one hashed source"):
                render_report(report, root, figures=["elo_vs_agreement"])

    def test_invalid_or_degenerate_bootstrap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            calibrate_tube([[1, .5]], np.tile([[[1, .5]]], (100, 1, 1)))
        with self.assertRaises(ValueError):
            calibrate_tube([[1, .5]], np.full((10, 1, 2), np.nan))


if __name__ == "__main__":
    unittest.main()
