"""Threshold boundaries, split provenance, resumable cache and real console smoke test."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from board_replay import replay_games
import analysis_pipeline
from board_test_support import make_game
from miss_analysis import AnalysisCache, AnalyzeEngine, default_engine, engine_identity, parse_analysis, transcript_key
from pipeline_common import BOARD_DTYPE, SPLIT_NAMES, TranscriptWriter, load_manifest, write_manifest


spec = importlib.util.spec_from_file_location("miss_step", Path(__file__).with_name("03_generate_miss_board_data.py"))
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def analysis_text(transcript, colors, losses=None):
    losses = losses if losses is not None else [i % 5 for i in range(len(colors))]
    rows = []
    for i, (color, loss) in enumerate(zip(colors, losses)):
        rows.append(f"| {i + 1} | {'Black' if color == 0 else 'White'} | {transcript[2*i:2*i+2]} "
                    f"| 10@100% | -5 | a1 | 10@100% | {-5 + loss:+d} | Disagree |")
    return "\n".join(reversed(rows)) + "\n> "


class FakeEngine:
    calls = []
    closed = 0

    def __init__(self, *args):
        pass

    def analyze(self, transcript):
        self.calls.append(transcript)
        boards, _, rejected = replay_games([(0, transcript)])
        assert not rejected
        return analysis_text(transcript, boards["color"].tobytes())

    def close(self):
        type(self).closed += 1


def make_source(root):
    source = root / "transcript_split" / "WTHOR_1977_2024"
    games = {name: [(i + 1, make_game(i + 8)[1])] for i, name in enumerate(SPLIT_NAMES)}
    games["train"].append((100, games["train"][0][1]))  # A duplicate keeps its original ID.
    games["train"].append((101, "f5"))  # Rejected incomplete game.
    for name in SPLIT_NAMES:
        writer = TranscriptWriter(source / name, 2)
        try:
            for game in games[name]:
                writer.write(*game)
        finally:
            writer.close()
    write_manifest(source, "transcript_split", "WTHOR_1977_2024", seed=613,
                   ratios={"train": .9, "val": .05, "test": .05},
                   split_unit="identical_transcript_group",
                   splits={name: {"games": len(games[name])} for name in SPLIT_NAMES})
    return games


class MissBoardTest(unittest.TestCase):
    def test_environment_paths_and_requested_defaults(self):
        with mock.patch.dict(os.environ, {"POLICY_DATA_ROOT": "data", "EGAROUCID_EXE": "engine.exe"}):
            args = step.make_argparser().parse_args(["--dataset", "WTHOR_1977_2024", "--min-loss", "3"])
        self.assertEqual(Path("data"), args.data_root)
        self.assertEqual(Path("engine.exe"), args.egaroucid_exe)
        self.assertEqual(32, args.workers)
        self.assertEqual(25, args.hash_level)
        self.assertEqual(10, step.LEVEL)
        self.assertEqual(1, step.THREADS)

    def test_parser_reverse_order_sign_none_and_worse_alternative(self):
        transcript, colors = "f5d6c3", bytes([0, 1, 1])
        raw = ("| 3 | White | c3 | 1@100% | -8 | None | | | |\n"
               "| 2 | White | d6 | 10@100% | -5 | c4 | 10@100% | -2 | Disagree |\n"
               "| 1 | Black | f5 | 10@100% | +4 | e6 | 10@100% | +1 | |\n> ")
        rows = parse_analysis(raw, transcript, colors)
        self.assertEqual([0, 3, 0], [r["loss"] for r in rows])
        self.assertEqual([4, -2, -8], [r["best_score"] for r in rows])
        self.assertEqual([1, 2, 3], [r["ply"] for r in rows])

    def test_parser_rejects_incomplete_duplicate_wrong_move_color_and_error(self):
        raw = analysis_text("f5d6", bytes([0, 1]))
        for bad in (raw.splitlines()[0], raw + raw, raw.replace("d6", "d7"),
                    raw.replace("White", "Black"), raw + "[ERROR] failed",
                    raw.replace("| -5 |", "| -65 |")):
            with self.subTest(raw=bad), self.assertRaises(ValueError):
                parse_analysis(bad, "f5d6", bytes([0, 1]))

    def test_cache_reopens_preserves_all_output_and_separates_engine_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            raw = analysis_text("f5", bytes([0]), [0])
            with contextlib.closing(AnalysisCache(path, {"level": 10})) as cache:
                cache.put("f5", raw, .1)
            with contextlib.closing(AnalysisCache(path, {"level": 10})) as cache:
                self.assertEqual(raw, cache.get("f5"))
                self.assertEqual({transcript_key("f5")}, cache.keys())
            with contextlib.closing(AnalysisCache(path, {"level": 11})) as cache:
                self.assertEqual(set(), cache.keys())

    def test_warm_engine_reuses_existing_cold_results_without_reanalysis(self):
        transcript = make_game(11)[1]
        colors = replay_games([(0, transcript)])[0]["color"].tobytes()
        raw = analysis_text(transcript, colors)
        identity = {"level": 10, "threads": 1, "hash_level": 25,
                    "files": {"engine": "same-engine"}, "clear_cache_per_game": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with contextlib.closing(AnalysisCache(path, identity)) as cache:
                cache.put(transcript, raw, .1)
                old_configuration = cache.configuration
            with contextlib.closing(AnalysisCache(path, {**identity, "clear_cache_per_game": False})) as cache:
                configuration = cache.configuration
                self.assertNotEqual(old_configuration, configuration)
                self.assertEqual([configuration, old_configuration], cache.compatible_configurations)
                args = step.make_argparser().parse_args(["--dataset", "test", "--min-loss", "3"])
                with (mock.patch.object(analysis_pipeline, "AnalyzeEngine", side_effect=AssertionError("must reuse")),
                      contextlib.redirect_stdout(io.StringIO())):
                    stats = step.analyze_missing({transcript: colors}, cache, args)
                self.assertEqual({"cache_hits": 1, "analyzed_games": 0}, stats)
                self.assertEqual(raw, cache.get(transcript))
                self.assertEqual([(old_configuration,)], cache.db.execute("SELECT configuration FROM analyses").fetchall())
            with contextlib.closing(AnalysisCache.open_existing(path, configuration)) as cache:
                self.assertEqual(raw, cache.get(transcript))
            for field, changed in (("level", 11), ("threads", 2), ("hash_level", 26), ("files", {"engine": "other-engine"})):
                with self.subTest(field=field), contextlib.closing(AnalysisCache(
                        path, {**identity, "clear_cache_per_game": False, field: changed})) as cache:
                    self.assertEqual(set(), cache.keys())
            with contextlib.closing(AnalysisCache(path, {**identity, "clear_cache_per_game": False})) as cache:
                cache.put(transcript, raw + "\nnew run", .1)
                self.assertEqual(raw + "\nnew run", cache.get(transcript))
            with contextlib.closing(AnalysisCache(path, identity)) as cache:
                self.assertEqual(raw, cache.get(transcript))

    def test_end_to_end_threshold_cache_reuse_and_game_split_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games = make_source(root)
            FakeEngine.calls, FakeEngine.closed = [], 0
            options = ["--data-root", str(root), "--dataset", "WTHOR_1977_2024", "--workers", "2",
                       "--games-per-batch", "2", "--positions-per-file", "7"]
            for threshold in (3, 4, 128):
                with (mock.patch.object(analysis_pipeline, "AnalyzeEngine", FakeEngine),
                      mock.patch.object(step, "engine_identity", return_value={"level": 10}),
                      contextlib.redirect_stdout(io.StringIO()) as messages):
                    manifest = step.run(step.make_argparser().parse_args(options + ["--min-loss", str(threshold)]))
                self.assertIn("ETA", messages.getvalue())
                self.assertEqual(3, len(FakeEngine.calls))
                self.assertEqual(1, manifest["splits"]["train"]["rejected_games"])
                self.assertEqual(0 if threshold == 3 else 3, manifest["analysis"]["cache_hits"])
                output = root / "board_split" / f"WTHOR_1977_2024_MISS_{threshold}"
                load_manifest(output, "board_split")
                for name in SPLIT_NAMES:
                    boards, ids, _ = replay_games(games[name])
                    mask = np.concatenate([np.arange(len(transcript) // 2) % 5 >= threshold
                                           for _, transcript in games[name] if transcript != "f5"])
                    paths = sorted((output / name).glob("*.dat"))
                    actual = np.concatenate([np.fromfile(p, dtype=BOARD_DTYPE) for p in paths]) if paths else np.empty(0, dtype=BOARD_DTYPE)
                    actual_ids = np.concatenate([np.fromfile(p.with_suffix(".game_ids"), dtype="<u8") for p in paths]) if paths else np.empty(0, dtype="<u8")
                    np.testing.assert_array_equal(boards[mask], actual)
                    np.testing.assert_array_equal(ids[mask], actual_ids)
                    self.assertEqual(len(actual), manifest["splits"][name]["positions"])
            self.assertGreater(FakeEngine.closed, 0)
            with self.assertRaises(FileExistsError):
                step.run(step.make_argparser().parse_args(options + ["--min-loss", "3"]))

    def test_32_persistent_workers_are_used_and_reused(self):
        lock, barrier = threading.Lock(), threading.Barrier(32, timeout=10)
        instances, calls, closed = [], [], []

        class ConcurrentEngine:
            def __init__(self, *args):
                with lock:
                    instances.append(self)

            def analyze(self, transcript):
                with lock:
                    calls.append(transcript)
                barrier.wait()
                return "raw " + transcript

            def close(self):
                closed.append(self)

        with tempfile.TemporaryDirectory() as directory:
            with contextlib.closing(AnalysisCache(Path(directory), {})) as cache:
                args = step.make_argparser().parse_args(["--dataset", "test", "--min-loss", "3"])
                unique = {f"game{i}": b"" for i in range(96)}
                with (mock.patch.object(analysis_pipeline, "AnalyzeEngine", ConcurrentEngine),
                      mock.patch.object(analysis_pipeline, "parse_analysis"), contextlib.redirect_stdout(io.StringIO())):
                    result = step.analyze_missing(unique, cache, args)
                self.assertEqual(96, result["analyzed_games"])
                self.assertEqual(32, len(instances))
                self.assertEqual(32, len(closed))
                self.assertEqual(set(unique), set(calls))
                self.assertEqual(96, len(cache.keys()))

    def test_failure_preserves_successful_games_and_resumes(self):
        game = make_game(11)[1]
        other = make_game(12)[1]
        unique = {t: replay_games([(0, t)])[0]["color"].tobytes() for t in (game, other)}

        class FailingEngine(FakeEngine):
            def analyze(self, transcript):
                if transcript == other:
                    raise TimeoutError("test timeout")
                return super().analyze(transcript)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            args = step.make_argparser().parse_args(["--dataset", "test", "--min-loss", "3", "--workers", "1"])
            with contextlib.closing(AnalysisCache(path, {})) as cache:
                with (mock.patch.object(analysis_pipeline, "AnalyzeEngine", FailingEngine), contextlib.redirect_stdout(io.StringIO()),
                      self.assertRaisesRegex(RuntimeError, "test timeout")):
                    step.analyze_missing(unique, cache, args)
                self.assertEqual({transcript_key(game)}, cache.keys())
            with contextlib.closing(AnalysisCache(path, {})) as cache:
                with mock.patch.object(analysis_pipeline, "AnalyzeEngine", FakeEngine), contextlib.redirect_stdout(io.StringIO()):
                    result = step.analyze_missing(unique, cache, args)
                self.assertEqual({"cache_hits": 1, "analyzed_games": 1}, result)

    def test_invalid_parameters_fail_before_creating_output(self):
        for option, value in (("--min-loss", "0"), ("--workers", "0"), ("--timeout-sec", "nan"),
                              ("--progress-seconds", "0"), ("--positions-per-file", "0")):
            args = step.make_argparser().parse_args(["--dataset", "test", "--min-loss", "3", option, value])
            with self.subTest(option=option), self.assertRaises(ValueError):
                step.run(args)

    @unittest.skipUnless(os.environ.get("MISS_ANALYZE_INTEGRATION") == "1", "opt-in real Egaroucid test")
    def test_real_console_complete_game_with_pass_and_same_persistent_process(self):
        # Find a terminal game with an implicit pass, using independent scalar replay.
        for seed in range(50):
            reference, transcript = make_game(seed)
            if np.any(reference["color"][1:] == reference["color"][:-1]):
                break
        else:
            self.fail("no forced-pass fixture found")
        colors = reference["color"].tobytes()
        self.assertFalse(engine_identity(default_engine(), 25)["clear_cache_per_game"])
        with contextlib.closing(AnalyzeEngine(default_engine(), 25, 120, threading.Event())) as engine:
            pid = engine.proc.pid
            raw = engine.analyze(transcript)
            rows = parse_analysis(raw, transcript, colors)
            self.assertEqual(len(reference), len(rows))
            self.assertTrue(any(row["loss"] > 0 for row in rows))
            self.assertEqual([int(x) for x in reference["policy"]], [row["played"] for row in rows])
            self.assertEqual(1, len(parse_analysis(engine.analyze("f5"), "f5", bytes([0]))))
            # Another complete game must reset history while keeping the process
            # and its search table available, including when revisiting a game.
            other, other_transcript = make_game(12)
            with mock.patch.object(engine, "command", wraps=engine.command) as commands:
                parse_analysis(engine.analyze(other_transcript), other_transcript, other["color"].tobytes())
                parse_analysis(engine.analyze(transcript), transcript, colors)
            self.assertFalse(any(call.args[0] == "clearcache" for call in commands.call_args_list))
            self.assertEqual(pid, engine.proc.pid)
        self.assertIsNotNone(engine.proc.poll())


if __name__ == "__main__":
    unittest.main()
