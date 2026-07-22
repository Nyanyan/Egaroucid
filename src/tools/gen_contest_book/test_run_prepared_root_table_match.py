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

    def test_metadata_records_evaluation_inputs_but_not_hash_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared_path, _openings = self._fixture(Path(temporary))
            prepared = runner.load_prepared_match_input(prepared_path)
            boards = runner.select_starting_boards(prepared.openings)
            with mock.patch.object(runner, "_git_provenance", return_value={"commit": "test"}):
                spec = runner.build_run_spec(prepared, Path(temporary) / "results.jsonl", boards, 180.0)
            parsed = spec["parsed_args"]
            self.assertEqual(parsed["seed"], 624)
            self.assertEqual(parsed["engine_random_seed"], 620)
            self.assertEqual(parsed["workers"], 1)
            self.assertTrue(parsed["random_symmetry"])
            self.assertTrue(parsed["level_31_verification_required"])
            self.assertIn("evaluation", spec["artifacts"])
            self.assertIn("endgame_move_ordering", spec["artifacts"])
            self.assertNotIn("transposition_hash", spec["artifacts"])
            self.assertNotIn("transposition_hash_used", spec["artifacts"])
            self.assertEqual(spec["openings"]["ordered_sha256"], runner._sha256_lines(boards))

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
                {"game": 0, "candidate_color": "X", "candidate_disc_diff": 3},
                {"game": 1, "candidate_color": "O", "candidate_disc_diff": -1},
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


if __name__ == "__main__":
    unittest.main()
