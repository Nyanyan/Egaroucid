"""Focused unit tests for the fixed prepared-input local match runner.

These tests verify input provenance and metadata construction only.  They do
not start a Console process, so they are safe to run without an engine build.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock


TOOL_DIRECTORY = Path(__file__).resolve().parent
if str(TOOL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOL_DIRECTORY))

from build_book import transform_board_text
import run_prepared_root_table_match as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _board(x_indices: tuple[int, ...], o_indices: tuple[int, ...], side: str) -> str:
    cells = ["-"] * 64
    for index in x_indices:
        cells[index] = "X"
    for index in o_indices:
        cells[index] = "O"
    return "".join(cells) + " " + side


class PreparedRunnerTest(unittest.TestCase):
    def _fixed_provenance(self, directory: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Create an isolated clean-worktree source fingerprint for metadata tests."""
        repository_root = directory / "worktree"
        for relative in runner.RUNNER_DEPENDENT_SOURCE_FILES:
            path = repository_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"// {relative}\n", encoding="utf-8")
        return (
            {
                "repository_root": str(repository_root.resolve()),
                "commit": "a" * 40,
                "tracked_worktree_dirty": False,
                "tracked_status_sha256": hashlib.sha256(b"").hexdigest(),
            },
            runner._runner_source_snapshots(repository_root.resolve()),
        )

    def _fixture(self, directory: Path) -> tuple[Path, list[str]]:
        prepared_dir = directory / "prepared"
        environment = prepared_dir / "teacher_execution_environment"
        files = {
            "executable": (Path("Console.exe"), b"fake Console executable\n"),
            "evaluation": (Path("resources/eval.egev2"), b"main evaluation\n"),
            "endgame_move_ordering": (
                Path("resources/eval_move_ordering_end.egev"),
                b"endgame move ordering\n",
            ),
        }
        environment_rows = []
        for role, (relative, content) in files.items():
            path = environment / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            environment_rows.append(
                {
                    "role": role,
                    "relative_path": relative.as_posix(),
                    "path": path.resolve().as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

        openings = [
            _board((0, 1, 2, 3, 8, 9, 16), (63, 62, 61, 60, 55, 54, 47), "X"),
            _board((0, 7, 15, 22, 30, 37, 45), (63, 56, 48, 41, 33, 26, 18), "O"),
        ]
        openings_path = prepared_dir / "openings" / "roots.txt"
        openings_path.parent.mkdir(parents=True, exist_ok=True)
        openings_path.write_text("\n".join(openings) + "\n", encoding="utf-8")
        table_path = prepared_dir / "table" / "contest_root_table.egcb"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text("# fake test table\n", encoding="utf-8")
        payload = {
            "schema": runner.PREPARED_INPUT_SCHEMA,
            "teacher_execution_environment": {
                "path": environment.resolve().as_posix(),
                "files": environment_rows,
            },
            "openings": {
                "path": openings_path.resolve().as_posix(),
                "sha256": _sha256(openings_path),
                "entries": len(openings),
            },
            "table": {
                "path": table_path.resolve().as_posix(),
                "sha256": _sha256(table_path),
                "entries": len(openings),
            },
            "selection": {"level_31_verification_required": True},
        }
        prepared_path = prepared_dir / "prepared_match_input.json"
        prepared_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return prepared_path, openings

    def test_loads_saved_console_environment_and_builds_fixed_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared_path, openings = self._fixture(Path(temporary))
            prepared = runner.load_prepared_match_input(prepared_path)
            self.assertEqual(prepared.openings, tuple(openings))
            self.assertEqual(prepared.executable.parent, prepared.environment_root)
            candidate = runner.engine_command(prepared, table_enabled=True)
            baseline = runner.engine_command(prepared, table_enabled=False)
            self.assertEqual(candidate[:3], [str(prepared.executable), "-quiet", "-noise"])
            self.assertIn("-seed", candidate)
            self.assertEqual(candidate[candidate.index("-seed") + 1], "620")
            self.assertEqual(candidate[candidate.index("-t") + 1], "8")
            self.assertEqual(candidate[candidate.index("-hash") + 1], "29")
            self.assertEqual(candidate[candidate.index("-time") + 1], "60")
            self.assertEqual(candidate[-2:], ["-contestbook", str(prepared.table.parent)])
            self.assertNotIn("-contestbook", baseline)

    def test_metadata_records_evaluation_inputs_and_immutable_board_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, _openings = self._fixture(root)
            prepared = runner.load_prepared_match_input(prepared_path)
            boards = runner.select_starting_boards(prepared.openings)
            with mock.patch.object(
                runner, "_fixed_run_provenance", return_value=self._fixed_provenance(root)
            ):
                spec = runner.build_run_spec(prepared, root / "results.jsonl", boards, 180.0)
            parsed = spec["parsed_args"]
            self.assertEqual(parsed["seed"], 624)
            self.assertEqual(parsed["engine_random_seed"], 620)
            self.assertEqual(parsed["workers"], 1)
            self.assertTrue(parsed["random_symmetry"])
            self.assertTrue(parsed["level_31_verification_required"])
            self.assertTrue(parsed["external_clock_control"])
            self.assertIn("evaluation", spec["artifacts"])
            self.assertIn("endgame_move_ordering", spec["artifacts"])
            self.assertIn("ordered_starting_positions", spec["artifacts"])
            self.assertNotIn("transposition_hash", spec["artifacts"])
            self.assertNotIn("transposition_hash_used", spec["artifacts"])
            self.assertEqual(spec["openings"]["ordered_sha256"], runner._sha256_lines(boards))
            order_path = runner.ordered_starting_positions_path_for(root / "results.jsonl")
            self.assertEqual(order_path.read_text(encoding="utf-8"), "".join(board + "\n" for board in boards))
            self.assertEqual(
                spec["artifacts"]["ordered_starting_positions"], spec["openings"]["ordered_file"]
            )
            protocol = spec["runner_protocol"]
            self.assertEqual(protocol["schema"], runner.MATCH_PROTOCOL_SCHEMA)
            self.assertTrue(protocol["clean_tracked_worktree_required"])
            self.assertEqual(
                [entry["relative_path"] for entry in protocol["source_files"]],
                list(runner.RUNNER_DEPENDENT_SOURCE_FILES),
            )

    def test_requires_level_31_verification_but_not_a_hash_resource(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared_path, _openings = self._fixture(Path(temporary))
            # The fixture intentionally contains no hash29.eghs.  Hash level
            # 29 remains a Console capacity option, not a saved input file.
            runner.load_prepared_match_input(prepared_path)
            payload = json.loads(prepared_path.read_text(encoding="utf-8"))
            payload["selection"]["level_31_verification_required"] = False
            prepared_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "level-31 verification"):
                runner.load_prepared_match_input(prepared_path)

    def test_rejects_pre_v5_prepared_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared_path, _openings = self._fixture(Path(temporary))
            payload = json.loads(prepared_path.read_text(encoding="utf-8"))
            payload["schema"] = "prepared_root_table_match_input_v4"
            prepared_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prepared_root_table_match_input_v5"):
                runner.load_prepared_match_input(prepared_path)

    def test_seed_624_order_and_rotation_match_the_documented_procedure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared_path, openings = self._fixture(Path(temporary))
            prepared = runner.load_prepared_match_input(prepared_path)
            generator = random.Random(624)
            expected_order = generator.sample(openings, len(openings))
            expected = [
                transform_board_text(board, generator.randrange(8)) for board in expected_order
            ]
            self.assertEqual(runner.select_starting_boards(prepared.openings), expected)
            with self.assertRaisesRegex(ValueError, "seed"):
                runner.select_starting_boards(prepared.openings, 625)

    def test_resume_validation_requires_one_x_and_one_o_table_game(self) -> None:
        boards = ["-" * 64 + " X"]
        row = {
            "match": 0,
            "board": boards[0],
            "games": [
                {
                    "game": 0,
                    "candidate_color": "X",
                    "candidate_disc_diff": 3,
                    "record": "",
                    "process_launch_order": runner.process_launch_order(0, 0),
                    "external_clock": {
                        "initial_remaining_msec": {"X": 60000, "O": 60000},
                        "records": [],
                        "final_remaining_msec": {"X": 60000, "O": 60000},
                    },
                },
                {
                    "game": 1,
                    "candidate_color": "O",
                    "candidate_disc_diff": -1,
                    "record": "",
                    "process_launch_order": runner.process_launch_order(0, 1),
                    "external_clock": {
                        "initial_remaining_msec": {"X": 60000, "O": 60000},
                        "records": [],
                        "final_remaining_msec": {"X": 60000, "O": 60000},
                    },
                },
            ],
            "margin": 2,
            "result": "W",
        }
        seen: set[int] = set()
        runner.validate_resume_result(row, boards, seen)
        self.assertEqual(seen, {0})
        row["games"][1]["candidate_color"] = "X"
        with self.assertRaisesRegex(RuntimeError, "color swap"):
            runner.validate_resume_result(row, boards, set())

    def test_immutable_starting_position_sidecar_refuses_a_changed_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "matches.jsonl"
            boards = ["-" * 64 + " X", "-" * 64 + " O"]
            snapshot = runner.ensure_immutable_starting_positions(output, boards)
            self.assertEqual(snapshot["sha256"], runner._sha256_lines(boards))
            path = runner.ordered_starting_positions_path_for(output)
            path.write_text("-" * 64 + " X\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different immutable"):
                runner.ensure_immutable_starting_positions(output, boards)

    def test_process_launch_order_cancels_inside_each_match(self) -> None:
        for match_id in range(4):
            self.assertEqual(
                {
                    tuple(runner.process_launch_order(match_id, 0)),
                    tuple(runner.process_launch_order(match_id, 1)),
                },
                {("candidate", "baseline"), ("baseline", "candidate")},
            )

    def test_formal_runner_refuses_a_tracked_dirty_worktree(self) -> None:
        with mock.patch.object(
            runner,
            "_git_provenance",
            return_value={
                "repository_root": "C:/irrelevant",
                "commit": "a" * 40,
                "tracked_worktree_dirty": True,
                "tracked_status_sha256": "not-used-when-dirty",
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "tracked-dirty"):
                runner._fixed_run_provenance()

    def test_engine_log_accepts_only_the_expected_hash29_initialization_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "engine.log"
            log.write_text(
                "[ERROR] can't open hash29.eghs\n"
                "[ERROR] can't get hash. you can ignore this error\n"
                "random seed = 620\n"
                "ggs tournament build = true\n"
                "> received cmd: settimems X 60000\n"
                "> received cmd: settimems O 60000\n",
                encoding="utf-8",
            )
            audit = runner._audit_engine_log(log, "X", 0, [7], 1)
            self.assertEqual(audit["expected_hash_error_lines"], 2)
            self.assertEqual(audit["unexpected_error_lines"], [])
            self.assertFalse(audit["timeout_suspected"])
            self.assertEqual(audit["random_seed_log_lines"], 1)
            self.assertEqual(audit["ggs_tournament_build_log_lines"], 1)
            self.assertEqual(
                audit["settimems_commands"],
                [
                    {"color": "X", "remaining_msec": 60000},
                    {"color": "O", "remaining_msec": 60000},
                ],
            )
            log.write_text(log.read_text(encoding="utf-8") + "[ERROR] unexpected\n", encoding="utf-8")
            altered = runner._audit_engine_log(log, "X", 0, [7], 1)
            self.assertTrue(altered["timeout_suspected"])
            self.assertEqual(altered["unexpected_error_lines"], ["[ERROR] unexpected"])

    def test_console_source_exposes_the_exact_clock_and_build_log_protocol(self) -> None:
        repository_root = TOOL_DIRECTORY.parents[2]
        command_definition = (repository_root / "src/console/command_definition.hpp").read_text(
            encoding="utf-8"
        )
        command = (repository_root / "src/console/command.hpp").read_text(encoding="utf-8")
        console = (repository_root / "src/Egaroucid_for_Console.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("CMD_ID_SETTIMEMS", command_definition)
        self.assertIn('{"settimems"}', command_definition)
        self.assertIn("void settimems", command)
        self.assertIn("case CMD_ID_SETTIMEMS", command)
        self.assertIn("go(board, options, state, tim());", command)
        self.assertIn('"random seed = "', console)
        self.assertIn('"ggs tournament build = true"', console)


if __name__ == "__main__":
    unittest.main()
