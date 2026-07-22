"""Unit tests for the formal root-teacher method comparison.

These tests deliberately do not start Egaroucid for Console.  They exercise
the frozen plan, exclusions, decision conditions, interruption handling, and
state-digest checks with temporary files and mocks only.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


TOOL_DIRECTORY = Path(__file__).resolve().parent
if str(TOOL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOL_DIRECTORY))

import benchmark_root_teacher_methods_formal as formal


def _method(status: str, move: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"status": status}
    if move is not None:
        result["move"] = move
    return result


def _pair(
    board: str,
    main_seed_index: int,
    repetition: int,
    time_status: str = "accepted",
    time_move: str | None = "a1",
    level_status: str = "accepted",
    level_move: str | None = "a1",
) -> dict[str, object]:
    return {
        "board": board,
        "engine_seed_index": main_seed_index,
        "repetition": repetition,
        "methods": {
            formal.TIME_METHOD: _method(time_status, time_move),
            formal.LEVEL_METHOD: _method(level_status, level_move),
        },
    }


def _counts() -> dict[str, int]:
    return {
        "only_time_method_accepted": 0,
        "level_method_worse": 0,
        "unstable_level_33": 0,
        "root_move_outside_candidates": 0,
        "dedicated_deep_seed_disagreement": 0,
        "unresolved_level_33": 0,
    }


def _deep_result(status: str, root: str = "a1", time_score: int = 0, level_score: int = 0) -> dict[str, object]:
    return {
        "status": status,
        "root_best_move": root,
        "queries": {
            "time_move_first": {"result": {"score": time_score}},
            "level_move_first": {"result": {"score": level_score}},
        },
    }


class FormalMethodComparisonTest(unittest.TestCase):
    def test_console_command_and_log_must_confirm_the_seed_and_tournament_build(self) -> None:
        command = formal._build_command(
            Path("console.exe"), "fixed_level_search", 620, level=30
        )
        self.assertIn("-noise", command)
        self.assertEqual("620", command[command.index("-seed") + 1])
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "query.log"
            completed = subprocess.CompletedProcess(command, 0, "", "")
            with mock.patch.object(formal.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(RuntimeError, "did not confirm random seed 620"):
                    formal._run_console(
                        Path("console.exe"),
                        "fixture board",
                        "fixed_level_search",
                        620,
                        log,
                        level=30,
                    )

    def test_seed_derivation_plan_size_and_reversed_repetition_order(self) -> None:
        boards = [f"board-{index:03d}" for index in range(600)]
        main_seeds = formal._derive_engine_seeds("a" * 64)
        deep_seeds = formal._derive_deep_seeds("a" * 64, main_seeds)
        plan = formal._pair_plan(boards, "b" * 64, main_seeds)

        self.assertEqual(4, len(main_seeds))
        self.assertEqual(2, len(deep_seeds))
        self.assertEqual(4_800, len(plan))
        self.assertEqual(4_800, len({item["pair_id"] for item in plan}))
        self.assertFalse({item["seed"] for item in main_seeds} & {item["seed"] for item in deep_seeds})

        grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
        for item in plan:
            grouped.setdefault((str(item["board"]), int(item["engine_seed_index"])), []).append(item)
        self.assertEqual(2_400, len(grouped))
        for rows in grouped.values():
            self.assertEqual(2, len(rows))
            rows.sort(key=lambda item: int(item["repetition"]))
            self.assertEqual(list(reversed(rows[0]["execution_order"])), rows[1]["execution_order"])
            self.assertEqual(rows[0]["execution_order_id"], rows[1]["execution_order_id"])

    def test_previous_fixed_sample_is_excluded_from_new_600_positions(self) -> None:
        boards = [f"board-{index:04d}" for index in range(1_200)]
        previous = boards[:600]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text("{}\n", encoding="utf-8")
            previous_path = root / "previous_fixed_sample.json"
            previous_path.write_text(
                formal._canonical_json(formal._make_fixed_sample_payload(previous)),
                encoding="utf-8",
            )
            fake_environment = {"schema": "test_environment"}
            with (
                mock.patch.object(formal, "load_uncovered_roots", return_value=boards),
                mock.patch.object(formal, "load_excluded_roots", return_value=set()),
                mock.patch.object(
                    formal,
                    "select_teacher_roots",
                    side_effect=lambda population, limit, _seed: list(population)[:limit],
                ),
                mock.patch.object(formal, "_new_execution_environment", return_value=fake_environment),
            ):
                state, selected, _plan, _payload = formal._new_state(
                    coverage,
                    [],
                    [previous_path],
                    root / "unused_console.exe",
                    root / "output",
                )
        self.assertEqual(600, len(selected))
        self.assertFalse(set(selected) & set(previous))
        self.assertEqual(600, state["population"]["excluded_board_count"])

    def test_repetition_disagreement_prevents_selection(self) -> None:
        with mock.patch.object(formal, "POSITION_COUNT", 1), mock.patch.object(formal, "ENGINE_SEED_COUNT", 1):
            disagreement = formal._repetition_consistency(
                [
                    _pair("board", 1, 1, level_move="a1"),
                    _pair("board", 1, 2, level_move="b1"),
                ]
            )
            self.assertEqual(1, disagreement["level_method_disagreements"])
            rule = formal._decision_protocol()
            conditions = formal._selection_conditions(
                rule,
                _counts(),
                disagreement,
                {"upper_95_percent": 0.80},
                2,
            )
        self.assertFalse(conditions["repetitions_consistent_for_both_methods"])
        self.assertFalse(all(conditions.values()))

    def test_deep_seed_disagreement_outside_root_and_lower_score_prevent_selection(self) -> None:
        equal = _deep_result("equal_level_33_score")
        mismatch = _deep_result("equal_level_33_score", time_score=1, level_score=1)
        self.assertEqual(
            "dedicated_seed_disagreement",
            formal._deep_seed_status([equal, mismatch]),
        )
        self.assertEqual(
            "root_move_outside_candidates",
            formal._deep_seed_status(
                [
                    _deep_result("root_move_outside_candidates", root="c1"),
                    _deep_result("root_move_outside_candidates", root="c1"),
                ]
            ),
        )
        self.assertEqual(
            "unstable",
            formal._deep_seed_status(
                [
                    _deep_result("unstable"),
                    _deep_result("unstable"),
                ]
            ),
        )
        self.assertEqual(
            "level_method_worse",
            formal._deep_seed_status(
                [
                    _deep_result("level_method_worse", time_score=2, level_score=1),
                    _deep_result("level_method_worse", time_score=2, level_score=1),
                ]
            ),
        )
        rule = formal._decision_protocol()
        repetitions = {
            "groups_with_any_method_disagreement": 0,
            "groups": formal.POSITION_COUNT * formal.ENGINE_SEED_COUNT,
        }
        for field in (
            "dedicated_deep_seed_disagreement",
            "root_move_outside_candidates",
            "level_method_worse",
            "unstable_level_33",
        ):
            counts = _counts()
            counts[field] = 1
            conditions = formal._selection_conditions(
                rule,
                counts,
                repetitions,
                {"upper_95_percent": 0.80},
                int(rule["required_complete_pairs"]),
            )
            self.assertFalse(all(conditions.values()), field)

    def test_interrupted_pair_is_archived_and_can_be_started_again(self) -> None:
        plan = [{"schedule_index": 1, "pair_id": "a" * 64}]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            pair_directory = formal._pair_directory(output, plan[0])
            pair_directory.mkdir(parents=True)
            (pair_directory / "partial.log").write_text("partial\n", encoding="utf-8")
            formal._reconcile_incomplete_pairs(output, plan, [])
            self.assertFalse(pair_directory.exists())
            archived = list((output / "interrupted_attempts").glob("*_*") )
            self.assertTrue(archived)
            pair_directory.mkdir(parents=True)
            self.assertTrue(pair_directory.is_dir())

    def test_progress_rejects_a_modified_immutable_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state_path = formal._state_path(output)
            formal._atomic_write_text(state_path, formal._canonical_json({"state": 1}))
            progress = {
                "schema": formal.FORMAL_PROGRESS_SCHEMA,
                "experiment_state_sha256": formal.sha256_file(state_path),
                "planned_pairs": 0,
                "completed_pairs": 0,
                "records": [],
            }
            formal._atomic_write_text(formal._progress_path(output), formal._canonical_json(progress))
            self.assertEqual([], formal._load_progress(output, state_path, [], []))
            formal._atomic_write_text(state_path, formal._canonical_json({"state": 2}))
            with self.assertRaises(ValueError):
                formal._load_progress(output, state_path, [], [])


if __name__ == "__main__":
    unittest.main()
