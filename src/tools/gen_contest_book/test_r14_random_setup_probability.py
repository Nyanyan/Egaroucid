from __future__ import annotations

import json
import hashlib
import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from build_book import canonicalize_board_key, transform_board_text
import collect_ggs_roots
import generate_ggs_root_teacher
from generate_ggs_root_teacher import (
    ROOT_ORDER_GGS_R14_PROBABILITY,
    TEACHER_SCHEMA,
    select_teacher_roots,
)
import report_ggs_root_teacher_progress
from r14_random_setup_probability import (
    R14_RANDOM_SETUP_PROBABILITY_MODEL,
    audit_r14_random_setup_directory,
    fraction_json,
    local_probability_model_source_fingerprints,
    load_r14_random_setup_priority_manifest,
    order_r14_random_setup_boards,
    ordered_boards_sha256,
    priority_manifest_metadata_path,
    priority_manifest_provenance,
    r14_random_setup_orbit_size,
    r14_random_setup_probability,
    validate_r14_random_setup_board,
    write_r14_random_setup_priority_manifest,
)
from othello import Board, index_to_coord


R14_BOARD = "------------------OOX-----XXOX----OXXX----OXX------------------- X"


class R14RandomSetupProbabilityTest(unittest.TestCase):
    @staticmethod
    def fixture_audit(boards: list[str], tie_seed: int = 620) -> dict[str, object]:
        """Build complete evidence for a small, fixed test input snapshot."""
        ordered = order_r14_random_setup_boards(boards, tie_seed)
        source_text = "".join(f"{board}\n" for board in boards).encode("ascii")
        return {
            "schema": R14_RANDOM_SETUP_PROBABILITY_MODEL["schema"],
            "model": R14_RANDOM_SETUP_PROBABILITY_MODEL,
            "local_probability_model_sources": local_probability_model_source_fingerprints(),
            "start_directory": "fixture",
            "source_files": [
                {
                    "path": "fixture.txt",
                    "bytes": len(source_text),
                    "sha256": hashlib.sha256(source_text).hexdigest(),
                    "rows": len(boards),
                }
            ],
            "rows": len(boards),
            "unique_canonical_rows": len(boards),
            "tie_seed": tie_seed,
            "ordered_roots_sha256": ordered_boards_sha256(ordered),
            "total_probability": fraction_json(
                sum((r14_random_setup_probability(board) for board in ordered), Fraction(0))
            ),
        }

    def test_probability_is_invariant_under_all_rotations_and_reflections(self) -> None:
        expected = r14_random_setup_probability(R14_BOARD)
        self.assertGreater(expected, Fraction(0))
        for symmetry in range(8):
            transformed = transform_board_text(R14_BOARD, symmetry)
            self.assertEqual(expected, r14_random_setup_probability(transformed))
            self.assertEqual(
                r14_random_setup_orbit_size(R14_BOARD),
                r14_random_setup_orbit_size(transformed),
            )

    def test_rejects_a_non_random_setup14_board(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 14 discs"):
            validate_r14_random_setup_board(
                "---------------------------OX------XO--------------------------- X"
            )

    def test_rejects_an_unknown_raw_board_character(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid cell character"):
            validate_r14_random_setup_board("?" + R14_BOARD[1:])

    def test_probability_model_records_the_three_local_source_roles(self) -> None:
        sources = local_probability_model_source_fingerprints()
        self.assertEqual(
            ["enumerated_population", "sampling_steps", "repository_description"],
            [source["id"] for source in sources["files"]],
        )
        for source in sources["files"]:
            self.assertEqual(64, len(source["sha256"]))
            self.assertGreater(source["bytes"], 0)
            self.assertTrue(source["role"])
            self.assertTrue(source["limitation"])

    def test_probability_order_keeps_high_mass_before_low_mass(self) -> None:
        # These examples differ only in the allowed number of O discs.  Five
        # white discs have higher per-position probability than seven.
        five_white = R14_BOARD
        cells = list(R14_BOARD[:64])
        black_indices = [index for index, cell in enumerate(cells) if cell == "X"]
        cells[black_indices[0]] = "O"
        cells[black_indices[1]] = "O"
        seven_white = "".join(cells) + " X"
        self.assertEqual(5, five_white[:64].count("O"))
        self.assertEqual(7, seven_white[:64].count("O"))
        self.assertGreater(
            r14_random_setup_probability(five_white),
            r14_random_setup_probability(seven_white),
        )
        self.assertEqual(
            [five_white, seven_white],
            order_r14_random_setup_boards([seven_white, five_white]),
        )
        self.assertEqual(
            [five_white, seven_white],
            select_teacher_roots(
                [seven_white, five_white],
                2,
                None,
                ROOT_ORDER_GGS_R14_PROBABILITY,
                priority_tie_seed=620,
            ),
        )
        with self.assertRaisesRegex(ValueError, "cohort_seed"):
            select_teacher_roots(
                [five_white], 1, 620, ROOT_ORDER_GGS_R14_PROBABILITY
            )
        frozen = [five_white, seven_white]
        self.assertEqual(
            frozen,
            select_teacher_roots(
                [seven_white, five_white],
                None,
                None,
                ROOT_ORDER_GGS_R14_PROBABILITY,
                priority_boards=frozen,
            ),
        )
        with self.assertRaisesRegex(ValueError, "does not contain every"):
            select_teacher_roots(
                [five_white, seven_white],
                None,
                None,
                ROOT_ORDER_GGS_R14_PROBABILITY,
                priority_boards=[five_white],
            )

    def test_complete_checked_in_corpus_has_total_probability_one(self) -> None:
        start_dir = Path(__file__).resolve().parent / "data" / "records321_14_random_setup"
        if not start_dir.is_dir():
            self.skipTest("the linked worktree does not contain the git-ignored r14 corpus")
        report = audit_r14_random_setup_directory(start_dir, tie_seed=620)
        self.assertEqual(111_534, report["rows"])
        self.assertEqual(111_534, report["unique_canonical_rows"])
        self.assertEqual({"numerator": 1, "denominator": 1}, report["total_probability"])
        self.assertEqual(R14_RANDOM_SETUP_PROBABILITY_MODEL, report["model"])
        self.assertEqual(
            local_probability_model_source_fingerprints(),
            report["local_probability_model_sources"],
        )
        self.assertEqual(
            report["rows"],
            sum(report["counts_by_white_discs"].values()),
        )
        self.assertEqual(
            report["rows"],
            sum(report["counts_by_orbit_size"].values()),
        )

    def test_frozen_priority_manifest_rejects_tampering(self) -> None:
        cells = list(R14_BOARD[:64])
        black_indices = [index for index, cell in enumerate(cells) if cell == "X"]
        cells[black_indices[0]] = "O"
        cells[black_indices[1]] = "O"
        seven_white = "".join(cells) + " X"
        five_white, _ = canonicalize_board_key(R14_BOARD)
        seven_white, _ = canonicalize_board_key(seven_white)
        boards = [five_white, seven_white]
        audit = self.fixture_audit(boards)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "priority.jsonl"
            metadata = write_r14_random_setup_priority_manifest(path, audit, boards, 620)
            loaded, loaded_metadata, loaded_provenance = load_r14_random_setup_priority_manifest(path)
            self.assertEqual(metadata, loaded_metadata)
            self.assertEqual(order_r14_random_setup_boards(boards, 620), loaded)
            provenance = priority_manifest_provenance(path)
            self.assertEqual(loaded_provenance, provenance)
            self.assertEqual(metadata["output"]["sha256"], provenance["sha256"])
            path.write_text("{}\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_r14_random_setup_priority_manifest(path)

    def test_priority_manifest_writer_rejects_stale_probability_model_sources(self) -> None:
        board, _ = canonicalize_board_key(R14_BOARD)
        audit = self.fixture_audit([board])
        sources = audit["local_probability_model_sources"]
        self.assertIsInstance(sources, dict)
        files = sources["files"]
        self.assertIsInstance(files, list)
        files[0]["sha256"] = "0" * 64
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "do not match the current local files"):
                write_r14_random_setup_priority_manifest(
                    Path(temporary) / "priority.jsonl", audit, [board], 620
                )

    def test_frozen_priority_manifest_rejects_a_hash_consistent_wrong_order(self) -> None:
        cells = list(R14_BOARD[:64])
        black_indices = [index for index, cell in enumerate(cells) if cell == "X"]
        cells[black_indices[0]] = "O"
        cells[black_indices[1]] = "O"
        seven_white = "".join(cells) + " X"
        five_white, _ = canonicalize_board_key(R14_BOARD)
        seven_white, _ = canonicalize_board_key(seven_white)
        boards = [five_white, seven_white]
        audit = self.fixture_audit(boards)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "priority.jsonl"
            write_r14_random_setup_priority_manifest(path, audit, boards, 620)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows.reverse()
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank
            payload = "".join(
                json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
            path.write_text(payload, encoding="utf-8", newline="\n")
            metadata_path = priority_manifest_metadata_path(path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["output"]["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            metadata["output"]["bytes"] = len(payload.encode("utf-8"))
            metadata["ordered_roots_sha256"] = ordered_boards_sha256(
                [row["board"] for row in rows]
            )
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(ValueError, "recorded probability order"):
                load_r14_random_setup_priority_manifest(path)

    def test_frozen_priority_manifest_rejects_an_input_audit_for_other_rows(self) -> None:
        cells = list(R14_BOARD[:64])
        black_indices = [index for index, cell in enumerate(cells) if cell == "X"]
        cells[black_indices[0]] = "O"
        cells[black_indices[1]] = "O"
        seven_white = "".join(cells) + " X"
        five_white, _ = canonicalize_board_key(R14_BOARD)
        seven_white, _ = canonicalize_board_key(seven_white)
        boards = [five_white, seven_white]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "priority.jsonl"
            write_r14_random_setup_priority_manifest(
                path, self.fixture_audit(boards), boards, 620
            )
            metadata_path = priority_manifest_metadata_path(path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["input_audit"]["ordered_roots_sha256"] = "0" * 64
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(ValueError, "input audit has an ordered-root SHA-256"):
                load_r14_random_setup_priority_manifest(path)

    def test_progress_reports_exact_probability_sums_for_the_teacher_population(self) -> None:
        cells = list(R14_BOARD[:64])
        black_indices = [index for index, cell in enumerate(cells) if cell == "X"]
        cells[black_indices[0]] = "O"
        cells[black_indices[1]] = "O"
        seven_white = "".join(cells) + " X"
        five_white, _ = canonicalize_board_key(R14_BOARD)
        seven_white, _ = canonicalize_board_key(seven_white)
        expected_accepted = r14_random_setup_probability(five_white)
        expected_rejected = r14_random_setup_probability(seven_white)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "teacher_rows.txt"
            state_path = output.with_suffix(output.suffix + ".state.json")
            priority = Path(temporary) / "priority.jsonl"
            write_r14_random_setup_priority_manifest(
                priority, self.fixture_audit([five_white, seven_white]), [five_white, seven_white], 620
            )
            provenance = priority_manifest_provenance(priority)
            state_path.write_text(
                json.dumps(
                    {
                        "schema": TEACHER_SCHEMA,
                        "roots": [five_white, seven_white],
                        "results": {five_white: {"move": "f5"}},
                        "rejections": {seven_white: {"reason": "fixture"}},
                        "root_order": ROOT_ORDER_GGS_R14_PROBABILITY,
                        "priority_manifest": provenance,
                        "priority_manifest_tie_seed": 620,
                        "r14_local_probability_model_sources": provenance["input_audit"][
                            "local_probability_model_sources"
                        ],
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            counts = report_ggs_root_teacher_progress.progress_counts(state_path)
            sums = counts["probability_sums"]
            self.assertIsNotNone(sums)
            self.assertEqual(fraction_json(expected_accepted), sums["accepted"])
            self.assertEqual(fraction_json(expected_rejected), sums["rejected"])
            self.assertEqual(
                fraction_json(expected_accepted + expected_rejected), sums["processed"]
            )
            report_path = Path(temporary) / "progress.md"
            report_ggs_root_teacher_progress.write_progress_report(state_path, report_path)
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("`records321_14_random_setup` にある `random_setup(14)`", text)
            self.assertIn("not coverage percentages for all `s8r14` starts", text)
            self.assertIn(provenance["sha256"], text)
            self.assertIn("`620`", text)
            self.assertIn("確率順序記録ファイル", text)
            self.assertIn("probability-order record file", text)

    def test_max_new_positions_resumes_the_same_frozen_probability_order(self) -> None:
        cells = list(R14_BOARD[:64])
        black_indices = [index for index, cell in enumerate(cells) if cell == "X"]
        cells[black_indices[0]] = "O"
        cells[black_indices[1]] = "O"
        seven_white = "".join(cells) + " X"
        five_white, _ = canonicalize_board_key(R14_BOARD)
        seven_white, _ = canonicalize_board_key(seven_white)
        boards = [five_white, seven_white]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(
                    {
                        "schema": collect_ggs_roots.REPORT_SCHEMA,
                        "root_discs": 14,
                        "roots": [
                            {
                                "canonical_board": board,
                                "deep_book": False,
                                "root_table": False,
                            }
                            for board in boards
                        ],
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            priority = root / "priority.jsonl"
            audit = self.fixture_audit(boards)
            write_r14_random_setup_priority_manifest(priority, audit, boards, 620)
            exe = root / "teacher.exe"
            exe.write_bytes(b"fixture teacher executable")
            resources = root / "resources"
            resources.mkdir()
            (resources / "eval.egev2").write_bytes(b"fixture evaluation")
            (resources / "eval_move_ordering_end.egev").write_bytes(b"fixture ordering")
            output = root / "teacher_rows.txt"

            def search(_exe: Path, board: str, level: int, *_args: object) -> dict[str, object]:
                return {
                    "move": index_to_coord(Board.from_text(board).legal_moves()[0]),
                    "score": -15,
                    "level": str(level),
                    "depth": f"{level}@74%",
                    "time": "000:00:00.001",
                    "nodes": 1,
                    "nps": 1,
                }

            with mock.patch.object(generate_ggs_root_teacher, "search_root_at_level", side_effect=search):
                first = generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    output,
                    60.0,
                    28,
                    29,
                    method="hint",
                    teacher_level=33,
                    root_order=ROOT_ORDER_GGS_R14_PROBABILITY,
                    priority_manifest_path=priority,
                    max_new_positions=1,
                    checkpoint_every=500,
                )
                self.assertEqual(
                    {"accepted": 1, "rejected": 0, "processed": 1, "requested": 2}, first
                )
                first_state = json.loads(
                    output.with_suffix(output.suffix + ".state.json").read_text(encoding="utf-8")
                )
                self.assertEqual([five_white, seven_white], first_state["roots"])
                self.assertEqual({five_white}, set(first_state["results"]))
                self.assertFalse(output.with_suffix(output.suffix + ".pending.jsonl").exists())
                second = generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    output,
                    60.0,
                    28,
                    29,
                    method="hint",
                    teacher_level=33,
                    root_order=ROOT_ORDER_GGS_R14_PROBABILITY,
                    priority_manifest_path=priority,
                    max_new_positions=1,
                    checkpoint_every=500,
                    resume=True,
                )
            self.assertEqual(
                {"accepted": 2, "rejected": 0, "processed": 2, "requested": 2}, second
            )
            final_state = json.loads(
                output.with_suffix(output.suffix + ".state.json").read_text(encoding="utf-8")
            )
            self.assertEqual({five_white, seven_white}, set(final_state["results"]))
            self.assertEqual(priority_manifest_provenance(priority), final_state["priority_manifest"])
            self.assertEqual(620, final_state["priority_manifest_tie_seed"])
            with self.assertRaisesRegex(ValueError, "resume mismatch for roots"):
                generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    output,
                    60.0,
                    28,
                    29,
                    method="hint",
                    teacher_level=33,
                    root_order=ROOT_ORDER_GGS_R14_PROBABILITY,
                    priority_manifest_path=priority,
                    limit=1,
                    resume=True,
                )

    def test_max_new_positions_reports_a_rejection_as_processed(self) -> None:
        board, _ = canonicalize_board_key(R14_BOARD)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(
                    {
                        "schema": collect_ggs_roots.REPORT_SCHEMA,
                        "root_discs": 14,
                        "roots": [
                            {
                                "canonical_board": board,
                                "deep_book": False,
                                "root_table": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            priority = root / "priority.jsonl"
            write_r14_random_setup_priority_manifest(
                priority, self.fixture_audit([board]), [board], 620
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"fixture teacher executable")
            resources = root / "resources"
            resources.mkdir()
            (resources / "eval.egev2").write_bytes(b"fixture evaluation")
            (resources / "eval_move_ordering_end.egev").write_bytes(b"fixture ordering")
            output = root / "teacher_rows.txt"
            legal_moves = [index_to_coord(move) for move in Board.from_text(board).legal_moves()]
            calls = 0

            def search(_exe: Path, _board: str, level: int, *_args: object) -> dict[str, object]:
                nonlocal calls
                move = legal_moves[calls]
                calls += 1
                return {
                    "move": move,
                    "score": -15,
                    "level": str(level),
                    "depth": f"{level}@74%",
                    "time": "000:00:00.001",
                    "nodes": 1,
                    "nps": 1,
                }

            with mock.patch.object(generate_ggs_root_teacher, "search_root_at_level", side_effect=search):
                result = generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    output,
                    60.0,
                    28,
                    29,
                    min_depth=30,
                    method="hint_then_verify",
                    teacher_level=30,
                    verify_level=31,
                    root_order=ROOT_ORDER_GGS_R14_PROBABILITY,
                    priority_manifest_path=priority,
                    max_new_positions=1,
                    checkpoint_every=500,
                )
            self.assertEqual(
                {"accepted": 0, "rejected": 1, "processed": 1, "requested": 1}, result
            )
            self.assertEqual(3, calls)
            state = json.loads(
                output.with_suffix(output.suffix + ".state.json").read_text(encoding="utf-8")
            )
            self.assertEqual({}, state["results"])
            self.assertEqual({board}, set(state["rejections"]))
            self.assertFalse(output.with_suffix(output.suffix + ".pending.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
