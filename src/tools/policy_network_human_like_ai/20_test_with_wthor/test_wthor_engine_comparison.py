from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import wthor_engine_comparison as comparison
import wthor_hint_pipeline as pipeline
from test_wthor_human_match_experiment import FakeActorContext, FakeActorExecutor
from wthor_edax import EDAX_LEVELS
from wthor_human_match_evaluation import (
    BoardState, HintData, PositionGroup, IncrementalEngineAgreementMetrics,
    evaluate_engine_agreement,
)
from wthor_human_match_experiment import EdaxProgressReporter


class EngineComparisonTest(unittest.TestCase):
    @staticmethod
    def progress(engine, completed=0, final=False, total=10):
        return {
            "engine": engine, "final": final,
            "completed_states": completed, "total_states": total,
            "elapsed_sec": 30.0, "eta_sec": 0.0 if final else 60.0,
            "active_tasks_by_level": {} if final else {21: 1},
            "completed_tasks": int(final), "submitted_tasks": 1,
            "active_tasks": int(not final), "cache_hits": 0, "retry_count": 0,
            "seconds_since_last_hint": 1.0, "cpu": {"available": False},
        }

    def test_progress_combines_updates_and_retains_finished_engine(self):
        reporter = comparison.ComparisonProgressReporter(
            ("egaroucid", "edax"),
            {engine: lambda row: [f"metric {row['engine']} {row['completed_states']}"]
             for engine in ("egaroucid", "edax")},
        )
        output = io.StringIO()
        with redirect_stderr(output):
            reporter.report(self.progress("edax", 2))
            self.assertEqual("", output.getvalue())
            reporter.report(self.progress("egaroucid", 3))
            first = output.getvalue()
            self.assertIn("5/20", first)
            self.assertIn("metric egaroucid 3\nmetric edax 2", first)
            reporter.report(self.progress("edax", 10, final=True))
            self.assertEqual(first, output.getvalue())
            reporter.report(self.progress("egaroucid", 7))
            reporter.report(self.progress("egaroucid", 10, final=True))
        self.assertEqual(3, output.getvalue().count("][hint"))
        self.assertEqual(1, output.getvalue().count("[hint進捗・完了]"))
        self.assertEqual(2, output.getvalue().count("metric edax 10"))
        self.assertIn("20/20", output.getvalue())

    def test_fully_cached_engines_emit_one_combined_final_block(self):
        reporter = comparison.ComparisonProgressReporter(("egaroucid", "edax"))
        output = io.StringIO()
        with redirect_stderr(output):
            reporter.report(self.progress("egaroucid", final=True, total=0))
            reporter.report(self.progress("edax", final=True, total=0))
        self.assertEqual(1, output.getvalue().count("[hint進捗・完了]"))
        self.assertIn("100.0%", output.getvalue())

    def test_workers_limit_is_global_and_freed_slots_are_reused(self):
        slots = comparison.SharedEngineSlots(
            32, {"egaroucid": 32, "edax": 32}, {"egaroucid": 6, "edax": 6},
        )
        for _ in range(26):
            self.assertTrue(slots.acquire("egaroucid"))
        self.assertFalse(slots.acquire("egaroucid"))
        for _ in range(6):
            self.assertTrue(slots.acquire("edax"))
        self.assertEqual(32, sum(slots.active.values()))
        self.assertFalse(slots.acquire("edax"))
        slots.release("edax")
        self.assertTrue(slots.acquire("egaroucid"))
        self.assertEqual(32, sum(slots.active.values()))

    def test_warm_engine_releases_startup_reservation(self):
        slots = comparison.SharedEngineSlots(
            12, {"egaroucid": 12, "edax": 2}, {"egaroucid": 6, "edax": 2},
        )
        self.assertTrue(slots.acquire("edax"))
        self.assertTrue(slots.acquire("edax"))
        self.assertFalse(slots.acquire("edax"))
        slots.release("edax")
        slots.release("edax")
        slots.finish("edax")
        for _ in range(12):
            self.assertTrue(slots.acquire("egaroucid"))
        self.assertFalse(slots.acquire("egaroucid"))

    def test_both_collectors_start_before_either_finishes(self):
        barrier = threading.Barrier(2)
        shared = []

        def collect(groups, *, engine, shared_slots, progress_callback, print_progress, **kwargs):
            shared.append(shared_slots)
            self.assertFalse(print_progress)
            barrier.wait(timeout=3)
            progress_callback(self.progress(engine, 10, final=True))
            return engine

        options = {engine: {"levels": EDAX_LEVELS, "workers": 32}
                   for engine in ("egaroucid", "edax")}
        output = io.StringIO()
        with patch.object(comparison, "collect_hints", side_effect=collect), redirect_stderr(output):
            results = comparison.collect_comparison_hints([], 32, options)
        self.assertEqual(1, output.getvalue().count("[hint進捗・完了]"))
        self.assertEqual({"egaroucid", "edax"}, set(results))
        self.assertIs(shared[0], shared[1])
        for engine, (result, elapsed) in results.items():
            self.assertEqual(engine, result)
            self.assertGreaterEqual(elapsed, 0)

    def test_one_failed_collector_cancels_the_other(self):
        barrier = threading.Barrier(2)
        cancelled = threading.Event()

        def collect(groups, *, engine, cancellation_event, **kwargs):
            barrier.wait(timeout=3)
            if engine == "edax":
                raise ValueError("Edax failed")
            if cancellation_event.wait(3):
                cancelled.set()
                return None
            raise AssertionError("peer was not cancelled")

        options = {engine: {"levels": EDAX_LEVELS, "workers": 12}
                   for engine in ("egaroucid", "edax")}
        with patch.object(comparison, "collect_hints", side_effect=collect):
            with self.assertRaisesRegex(ValueError, "Edax failed"):
                comparison.collect_comparison_hints([], 12, options)
        self.assertTrue(cancelled.is_set())

    def test_pipeline_returns_all_slots_and_reports_engine(self):
        state = BoardState.initial()
        group = PositionGroup(state.black, state.white, state.side, ((44, 2),))
        slots = comparison.SharedEngineSlots(
            12, {"egaroucid": 12, "edax": 12}, {"egaroucid": 6, "edax": 6},
        )
        for _ in range(6):
            self.assertTrue(slots.acquire("egaroucid"))
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            pipeline, "ProcessPoolExecutor", FakeActorExecutor,
        ), patch.object(pipeline.multiprocessing, "get_context", return_value=FakeActorContext()), patch.object(
            pipeline, "wait", side_effect=lambda pending, **kw: (set(pending), set()),
        ), redirect_stderr(output):
            hints, _, _, _ = pipeline.collect_hints(
                [group], EDAX_LEVELS, Path(directory), 12, Path("unused"), 1, 1, 0, 30,
                engine="edax", shared_slots=slots,
            )
        self.assertEqual(set(EDAX_LEVELS), set(hints))
        self.assertEqual({"egaroucid": 6, "edax": 0}, slots.active)
        self.assertIn("[Edax hint進捗", output.getvalue())
        self.assertIn("Edax完了", output.getvalue())

    def test_edax_live_metrics_match_final_and_show_pending_levels(self):
        state = BoardState.initial()
        legal = state.legal_policies(state.side)
        group = PositionGroup(state.black, state.white, state.side, ((legal[0], 3),))
        hint = HintData({move: 0.0 for move in legal[:3]}, tuple(legal[:3]))
        metrics = IncrementalEngineAgreementMetrics([group], EDAX_LEVELS)
        reporter = EdaxProgressReporter(metrics)
        reporter.accept_rows(1, [(group.key, hint), (group.key, hint)])
        output = io.StringIO()
        with redirect_stderr(output):
            reporter.report({"reported_hints_by_level": {1: 1}, "target_hints_by_level": {1: 1}})
        self.assertIn("[Edax level= 1] top-1 100.000%", output.getvalue())
        self.assertIn("[Edax level=21] 未算出 (n=0)", output.getvalue())
        for level in EDAX_LEVELS[1:]:
            reporter.accept_rows(level, [(group.key, hint)])
        hints = {level: {group.key: hint} for level in EDAX_LEVELS}
        self.assertEqual(evaluate_engine_agreement([group], hints), metrics.snapshot())
        snapshot = metrics.snapshot()
        snapshot[1]["hits"][1] = -1
        self.assertEqual(3, metrics.snapshot()[1]["hits"][1])


if __name__ == "__main__":
    unittest.main()
