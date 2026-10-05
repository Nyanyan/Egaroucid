"""Analysis reports, cache-only exports and augmentation with per-position weights."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

import analysis_pipeline
from board_replay import replay_games
from miss_analysis import AnalysisCache
from pipeline_common import BOARD_DTYPE, SPLIT_NAMES, load_manifest, validate_weight_file
from test_miss_board_data import FakeEngine, make_source


def load_step(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = load_step("03_analyze_transcripts")
weighted = load_step("03_generate_weighted_board_data")
augment = load_step("04_augment_board_data")


class WeightedBoardTest(unittest.TestCase):
    def test_weight_formula_odd_losses_and_no_clipping(self):
        weights = weighted.loss_weights(np.arange(129))
        np.testing.assert_allclose(weights[2:] / weights[:-2], 2)
        self.assertEqual(1, weights[0])
        self.assertAlmostEqual(2 ** .5, weights[1])
        self.assertEqual(2 ** 64, weights[128])
        for bad in ([-1], [129], [float("nan")]):
            with self.assertRaises(ValueError):
                weighted.loss_weights(bad)

    def test_two_stage_reports_export_all_rows_and_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games = make_source(root)
            options = ["--data-root", str(root), "--dataset", "WTHOR_1977_2024"]
            FakeEngine.calls = []
            for _ in range(2):
                with (mock.patch.object(analysis_pipeline, "AnalyzeEngine", FakeEngine),
                      mock.patch.object(analyze, "engine_identity", return_value={"level": 10, "threads": 1}),
                      contextlib.redirect_stdout(io.StringIO())):
                    report = analyze.run(analyze.make_argparser().parse_args(options + ["--workers", "2"]))
            self.assertEqual(3, len(FakeEngine.calls))
            self.assertFalse((root / "board_split").exists())
            statistics = root / "cache/WTHOR_1977_2024_LOSS_WEIGHTED/statistics"
            self.assertGreater((statistics / "loss_distribution.png").stat().st_size, 1000)
            self.assertEqual(1 + 4 * 129, len((statistics / "loss_distribution.csv").read_text().splitlines()))
            total_positions = 0
            expected_by_split = {}
            for name in SPLIT_NAMES:
                boards, ids, _ = replay_games(games[name])
                losses = np.concatenate([np.arange(len(t) // 2) % 5 for _, t in games[name] if t != "f5"])
                expected_by_split[name] = boards, ids, np.exp2(losses / 2)
                expected_counts = np.bincount(losses, minlength=129)
                data = report["distribution"][name]
                self.assertEqual(len(boards), data["positions"])
                np.testing.assert_array_equal(expected_counts, [row["count"] for row in data["bins"]])
                self.assertAlmostEqual(100, sum(row["percentage"] for row in data["bins"]))
                total_positions += len(boards)
            self.assertEqual(total_positions, report["distribution"]["all"]["positions"])
            with (mock.patch.object(analysis_pipeline, "AnalyzeEngine", side_effect=AssertionError("cache only")),
                  contextlib.redirect_stdout(io.StringIO())):
                manifest = weighted.run(weighted.make_argparser().parse_args([
                    "--data-root", str(root), "--cache-name", "WTHOR_1977_2024_LOSS_WEIGHTED",
                    "--positions-per-file", "7"]))
            dataset = root / "board_split/WTHOR_1977_2024_LOSS_WEIGHTED"
            load_manifest(dataset, "board_split")
            self.assertEqual("cache/WTHOR_1977_2024_LOSS_WEIGHTED/analyze.sqlite3", manifest["analysis"]["cache"])
            self.assertEqual(1, manifest["splits"]["train"]["rejected_games"])
            for name, (boards, ids, weights) in expected_by_split.items():
                paths = sorted((dataset / name).glob("*.dat"))
                np.testing.assert_array_equal(boards, np.concatenate([np.fromfile(p, dtype=BOARD_DTYPE) for p in paths]))
                np.testing.assert_array_equal(ids, np.concatenate([np.fromfile(p.with_suffix(".game_ids"), dtype="<u8") for p in paths]))
                np.testing.assert_array_equal(weights, np.concatenate([np.fromfile(p.with_suffix(".weights"), dtype="<f8") for p in paths]))
                self.assertEqual(len(boards), manifest["splits"][name]["positions"])
            with contextlib.redirect_stdout(io.StringIO()):
                augment.run(augment.make_argparser().parse_args([
                    "--data-root", str(root), "--dataset", "WTHOR_1977_2024_LOSS_WEIGHTED",
                    "--positions-per-batch", "5", "--positions-per-file", "13"]))
            increased = root / "board_split_increased/WTHOR_1977_2024_LOSS_WEIGHTED"
            increased_manifest = load_manifest(increased, "board_split_increased")
            self.assertEqual(manifest["sample_weights"], increased_manifest["sample_weights"])
            for name, (_, _, weights) in expected_by_split.items():
                paths = sorted((increased / name).glob("*.weights"))
                np.testing.assert_array_equal(np.repeat(weights, 8), np.concatenate([np.fromfile(p, dtype="<f8") for p in paths]))

    def test_incomplete_cache_stops_without_creating_board_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root)
            with contextlib.closing(AnalysisCache(root / "cache/WTHOR_1977_2024_LOSS_WEIGHTED", {"level": 10, "threads": 1})):
                pass
            args = weighted.make_argparser().parse_args([
                "--data-root", str(root), "--cache-name", "WTHOR_1977_2024_LOSS_WEIGHTED"])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "missing from cache"):
                weighted.run(args)
            self.assertFalse((root / "board_split").exists())

    def test_cache_name_uses_source_from_statistics_and_dataset_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games = make_source(root)
            options = ["--data-root", str(root), "--dataset", "WTHOR_1977_2024"]
            with (mock.patch.object(analysis_pipeline, "AnalyzeEngine", FakeEngine),
                  mock.patch.object(analyze, "engine_identity", return_value={"level": 10, "threads": 1}),
                  contextlib.redirect_stdout(io.StringIO())):
                analyze.run(analyze.make_argparser().parse_args(options + ["--workers", "2"]))
            # Folder name may differ from the original transcript dataset.
            (root / "cache/WTHOR_1977_2024_LOSS_WEIGHTED").rename(root / "cache/renamed_cache")
            args = weighted.make_argparser().parse_args([
                "--data-root", str(root), "--dataset", "renamed_cache"])
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = weighted.run(args)
            self.assertEqual("renamed_cache", manifest["dataset"])
            self.assertEqual("cache/renamed_cache/analyze.sqlite3", manifest["analysis"]["cache"])
            for name in SPLIT_NAMES:
                boards, _, _ = replay_games(games[name])
                self.assertEqual(len(boards), manifest["splits"][name]["positions"])
            self.assertTrue((root / "board_split/renamed_cache/manifest.json").exists())
            self.assertFalse((root / "board_split/renamed_cache_LOSS_WEIGHTED").exists())

    def test_read_only_cache_and_ambiguous_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for level in (10, 11):
                with contextlib.closing(AnalysisCache(root, {"level": level})) as cache:
                    config = cache.configuration
            with self.assertRaisesRegex(ValueError, "configurations"):
                AnalysisCache.open_existing(root)
            with contextlib.closing(AnalysisCache.open_existing(root, config)) as cache:
                with self.assertRaises(sqlite3.OperationalError):
                    cache.put("f5", "raw", 0)

    def test_weight_sidecar_validation_rejects_truncation_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.weights"
            for values in ([1], [1, 0], [1, float("inf")], [1, float("nan")]):
                np.array(values, dtype="<f8").tofile(path)
                with self.assertRaises(ValueError):
                    validate_weight_file(path, 2)


if __name__ == "__main__":
    unittest.main()
