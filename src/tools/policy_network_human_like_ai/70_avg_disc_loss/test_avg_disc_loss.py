"""Regression checks for attribution, averaging, input validation, and resume."""
import copy
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import calculate_avg_disc_loss as module


TRANSCRIPT = ("f5f4e3d6e6g5c4d7f6g6g3f3g4e2h5h6e7f7f8e8h4h3d8c7c5c6g8b5b4b6c3"
              "a3f2b3d1c2d3d2c8f1b8g7a5a7h8a6a8h7b7g2h2c1b2h1g1a1b1e1a2a4")


def analysis_output(transcript, colors):
    rows = []
    for index, color in enumerate(colors):
        move = transcript[2 * index:2 * index + 2]
        # Large opening loss must never enter the model's statistics.
        loss = 20 if index < 8 else (2 if color == 0 else 4)
        rows.append(f"| {index + 1} | {'Black' if color == 0 else 'White'} | {move} | 1 | -10 "
                    f"| a1 | 1 | {-10 + loss} | Mistake |")
    return "\n".join(reversed(rows))


def tournament_fixture(directory):
    rows, tasks = [], []
    for task_id, (p0, p1) in enumerate(combinations(range(30), 2)):
        task = module.MatchSetTask(task_id, p0, p1, 0, TRANSCRIPT[:16])
        tasks.append(task)
        games = [{"p0_idx": p0, "p1_idx": p1, "p0_is_black": black,
                  "black_idx": p0 if black else p1, "white_idx": p1 if black else p0,
                  "transcript": TRANSCRIPT} for black in (True, False)]
        rows.append({"schema_version": 2, "task": task.identity(),
                     "result": {**task.identity(), "color_games": games}})
    manifest = {"experiment_id": "fixture", "configuration": {
        "participants": {"paper_roster": {"total_players": 30, "policy_families": [
            {"pi1": pi1, "pi2": pi2, "alphas": list(module.ALPHAS)}
            for _, pi1, pi2 in module.FAMILIES]}},
        "schedule": {"match_sets_per_pair": 1, "task_plan_sha256": module.task_plan_hash(tasks)}}}
    result = {"experiment_id": "fixture", "players": [{"name": p} for p in module.PLAYER_NAMES],
              "completed_match_sets": 435, "total_match_sets": 435,
              "completed_actual_games": 870, "total_actual_games": 870}
    for name, value in (("strength_manifest.json", manifest), ("strength_results.json", result)):
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    write_log(directory, rows)
    return rows


def write_log(directory, rows):
    (directory / "strength_games.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AverageLossTests(unittest.TestCase):
    def test_passes_opening_and_each_players_own_moves(self):
        players = list(module.PROPOSALS)
        game = module.Game(0, 0, players[0], players[6], 8, TRANSCRIPT)
        unique = module.validate_transcripts([game], 999)
        colors = unique[TRANSCRIPT]
        self.assertTrue(any(colors[i] == colors[i - 1] for i in range(1, len(colors))))
        raw = analysis_output(TRANSCRIPT, colors)
        rows = module.parse_analysis(raw, TRANSCRIPT, colors)
        black, white = list(module.game_statistics(game, rows))
        self.assertEqual(52, black["moves"] + white["moves"])
        self.assertEqual(colors[8:].count(0), black["moves"])
        self.assertEqual(colors[8:].count(1), white["moves"])
        self.assertEqual(2, black["avg_disc_loss"])
        self.assertEqual(4, white["avg_disc_loss"])
        baseline = module.Game(0, 0, "edax_l21", players[6], 8, TRANSCRIPT)
        self.assertEqual(1, len(list(module.game_statistics(baseline, rows))))

    def test_average_is_weighted_by_moves_and_includes_zero(self):
        rows = [{"moves": 1, "total_disc_loss": 4, "nonzero_loss_moves": 1},
                {"moves": 3, "total_disc_loss": 0, "nonzero_loss_moves": 0}]
        summary = module.summarize(rows)
        self.assertEqual(1, summary["avg_disc_loss"])
        self.assertEqual(2, summary["avg_total_disc_loss_per_game"])
        self.assertIsNone(module.summarize([])["avg_disc_loss"])

    def test_alternative_clamp_no_alternative_and_white_sign(self):
        raw = ("| 1 | Black | f5 | 1 | 4 | d3 | 1 | 2 | |\n"
               "| 2 | White | f4 | 1 | -4 | a1 | 1 | -1 | |\n"
               "| 3 | Black | e3 | 1 | -9 | none | | | |")
        rows = module.parse_analysis(raw, "f5f4e3", bytes([0, 1, 0]))
        self.assertEqual([0, 3, 0], [row["loss"] for row in rows])

    def test_reject_incomplete_and_illegal_transcripts(self):
        for transcript in (TRANSCRIPT[:-2], "a1" + TRANSCRIPT[2:]):
            with self.subTest(transcript=transcript), self.assertRaises(ValueError):
                module.validate_transcripts([module.Game(0, 0, "a", "b", 8, transcript)], 999)

    def test_tournament_roster_counts_duplicate_missing_and_wrong_color(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = tournament_fixture(directory)
            _, games = module.load_tournament(directory)
            self.assertEqual(660, len(games))
            invalid = [rows[:-1], rows + [rows[0]]]
            wrong_color = copy.deepcopy(rows)
            wrong_color[0]["result"]["color_games"][0]["white_idx"] = 29
            invalid.append(wrong_color)
            wrong_opening = copy.deepcopy(rows)
            wrong_opening[0]["task"]["opening"] = "a1"
            invalid.append(wrong_opening)
            for bad in invalid:
                write_log(directory, bad)
                with self.assertRaises(ValueError):
                    module.load_tournament(directory)

    def test_level21_and_threads_selected_at_launch_and_cache_cleared(self):
        with patch.object(module.subprocess, "Popen") as process, \
                patch.object(module, "_create_windows_kill_job", return_value=1) as job, \
                patch.object(module, "_close_windows_job"), \
                patch.object(module.threading, "Thread"), \
                patch.object(module.AnalyzeEngine, "_read_prompt", return_value=""), \
                patch.object(module.AnalyzeEngine, "command", return_value="") as command:
            engine = module.Level21Engine(Path("engine.exe"), 25, 30, None, threads=32)
            engine.analyze(TRANSCRIPT)
            self.assertEqual(["engine.exe", "-l", "21", "-t", "32"], process.call_args.args[0][:5])
            job.assert_called_once_with(process.return_value)
            self.assertEqual(["clearcache", "init", "play " + TRANSCRIPT, "analyze"],
                             [call.args[0] for call in command.call_args_list])
            engine.close()

    def test_cache_resume_duplicate_occurrences_and_separate_levels(self):
        player = next(iter(module.PROPOSALS))
        games = [module.Game(i, 0, player, "random", 8, TRANSCRIPT) for i in range(2)]
        unique = module.validate_transcripts(games, 999)
        raw = analysis_output(TRANSCRIPT, unique[TRANSCRIPT])
        args = SimpleNamespace(workers=2, egaroucid_exe=None, hash_level=25,
                               timeout_sec=30, progress_seconds=999, tail_games=0, tail_threads=32)
        with TemporaryDirectory() as temporary, patch.object(module, "Level21Engine") as engine:
            output = Path(temporary)
            engine.return_value.analyze.return_value = raw
            with module.closing(module.ThreadedAnalysisCache(output / "cache", {"level": 21}, 32)) as cache:
                self.assertEqual(1, module.analyze_games(unique, cache, args)["analyzed_transcripts"])
                self.assertEqual(1, module.analyze_games(unique, cache, args)["cache_hits"])
                self.assertEqual(1, engine.return_value.analyze.call_count)
                summary = module.aggregate(games, unique, cache, output, 999)
                row = next(row for row in summary if row["player"] == player)
                self.assertEqual(2, row["games"])
                self.assertEqual(2, row["avg_disc_loss"])
            with module.closing(module.AnalysisCache(output / "cache", {"level": 10})) as other:
                self.assertEqual(set(), other.keys())

    def test_parallel_work_hands_unfinished_games_to_one_32_thread_engine(self):
        barrier = threading.Barrier(3, timeout=5)
        live, launched, calls, statuses = set(), [], [], []
        gate = threading.Lock()

        class FakeEngine:
            def __init__(self, exe, hash_level, timeout, cancel, threads):
                self.cancel, self.threads = cancel, threads
                with gate:
                    if threads == 32:
                        # The old processes must be closed before launching the tail.
                        self_overlap = bool(live)
                        if self_overlap:
                            raise AssertionError("Overlapping parallel and tail processes")
                    live.add(self)
                    launched.append(threads)
                    self.proc = SimpleNamespace(pid=len(launched))

            def analyze(self, transcript):
                with gate:
                    calls.append((transcript, self.threads))
                if self.threads == 1 and transcript in ("game0", "game1", "game2"):
                    barrier.wait()
                if self.threads == 1 and transcript == "game0":
                    if not self.cancel.wait(timeout=5):
                        raise AssertionError("Scheduler drained a straggler instead of switching")
                    raise InterruptedError("tail transition")
                return "raw output"

            def close(self):
                with gate:
                    live.remove(self)

        unique = {f"game{i}": b"" for i in range(7)}
        args = SimpleNamespace(workers=3, egaroucid_exe=None, hash_level=25, timeout_sec=30,
                               progress_seconds=999, tail_games=3, tail_threads=32)
        with TemporaryDirectory() as temporary, patch.object(module, "Level21Engine", FakeEngine), \
                patch.object(module, "parse_analysis", return_value=[]):
            with module.closing(module.ThreadedAnalysisCache(Path(temporary), {"level": 21}, 32)) as cache:
                stats = module.analyze_games(unique, cache, args, on_progress=statuses.append)
                self.assertEqual(7, stats["analyzed_transcripts"])
                # Other workers may finish while cancellation propagates; their
                # successful results must be retained instead of recomputed.
                self.assertGreaterEqual(stats["parallel_analyzed_transcripts"], 4)
                self.assertIn(stats["tail_analyzed_transcripts"], (1, 2, 3))
                self.assertEqual(stats["parallel_analyzed_transcripts"], len(cache.caches[1].keys()))
                self.assertEqual(stats["tail_analyzed_transcripts"], len(cache.caches[32].keys()))
                self.assertIn(module.transcript_key("game0"), cache.caches[32].keys())
                self.assertEqual(7, module.analyze_games(unique, cache, args)["cache_hits"])
        self.assertEqual([1, 1, 1, 32], launched)
        self.assertIn(("game0", 1), calls)
        self.assertIn(("game0", 32), calls)
        self.assertFalse(live)
        self.assertTrue(any(s["phase"] == "tail" and s["process_limit"] == 1
                            and s["threads_per_engine"] == 32 for s in statuses))

    def test_defaults_are_32_by_1_then_1_by_32(self):
        args = module.make_argparser().parse_args([])
        self.assertEqual((32, 32, 32), (args.workers, args.tail_games, args.tail_threads))


if __name__ == "__main__":
    unittest.main()
