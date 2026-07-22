from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_all_records
import generate_records
import book_artifact
import audit_r14_corpus
import build_root_table
import collect_ggs_roots
import generate_ggs_root_teacher
import report_ggs_root_teacher_progress
from build_book import canonicalize_board_key, transform_board_text
from book_artifact import (
    BookBuildSpec,
    BookValidationError,
    build_book_atomically,
    build_book_if_stale,
    check_book_status,
    file_lock,
    manifest_path_for_book,
    validate_book_file,
)
from othello import Board


INITIAL_BOARD = "---------------------------OX------XO--------------------------- X"
GGS_ROOT = "------------------XXXX----XOOX----OXX-----OXO------------------- X"


def write_records(path: Path, transcripts: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for transcript in transcripts:
            f.write(f"initial board: {INITIAL_BOARD}\n")
            f.write(f"transcript: {transcript}\n\n")


def valid_book_text(*, records_seen: int = 1, records_used: int = 1) -> str:
    return (
        "# contest_book_v1\n"
        f"# initial {INITIAL_BOARD}\n"
        f"# records_seen {records_seen}\n"
        f"# records_used {records_used}\n"
        "# cut_empty 30\n"
        f"{INITIAL_BOARD} 0 d3:0\n"
    )


class GenerateRecordsTests(unittest.TestCase):
    def make_args(self, games: int, batch_size: int = 16) -> SimpleNamespace:
        return SimpleNamespace(
            games=games,
            batch_size=batch_size,
            exe=Path(sys.executable),
            level=19,
            threads=1,
            max_loss_per_move=2,
            max_loss_total=4,
            max_book_loss=4,
            cut_empty=30,
            use_existing_book=False,
        )

    def test_games_is_target_total_not_additional_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            records_dir.mkdir()
            record_file = records_dir / "records.txt"
            write_records(record_file, ["existing-a", "existing-b"])
            output = root / "book.egcb"
            requested_batches: list[int] = []

            def fake_run(_args, _board, _out_dir, n_games, _use_book):
                requested_batches.append(n_games)
                write_records(record_file, [
                    "existing-a",
                    "existing-b",
                    *[f"new-{idx}" for idx in range(n_games)],
                ])

            with (
                mock.patch.object(generate_records, "run_record_batch", side_effect=fake_run),
                mock.patch.object(generate_records, "build_provisional_book") as build,
            ):
                result = generate_records.generate_to_target(
                    self.make_args(games=5, batch_size=4),
                    INITIAL_BOARD,
                    records_dir,
                    output,
                )

            self.assertEqual(5, result)
            self.assertEqual([3], requested_batches)
            build.assert_called_once()
            manifest = json.loads(
                generate_records.generation_manifest_path(records_dir).read_text(encoding="utf-8")
            )
            self.assertEqual("contest_record_generation_manifest_v1", manifest["schema"])
            self.assertEqual(1, len(manifest["batches"]))
            self.assertEqual(3, manifest["batches"][0]["requested_records"])
            self.assertEqual(5, manifest["batches"][0]["after"]["unique_records"])
            self.assertEqual(
                Path(sys.executable).resolve().as_posix(),
                manifest["batches"][0]["profile"]["executable"]["path"],
            )

    def test_tournament_clang_binary_is_default(self) -> None:
        self.assertEqual("Egaroucid_for_Console_clang.exe", generate_records.CONSOLE_EXE.name)

    def test_satisfied_target_does_not_generate_or_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            records_dir.mkdir()
            write_records(records_dir / "records.txt", ["a", "b", "c"])

            with (
                mock.patch.object(generate_records, "run_record_batch") as run,
                mock.patch.object(generate_records, "build_provisional_book") as build,
            ):
                result = generate_records.generate_to_target(
                    self.make_args(games=3),
                    INITIAL_BOARD,
                    records_dir,
                    root / "book.egcb",
                )

            self.assertEqual(3, result)
            run.assert_not_called()
            build.assert_not_called()

    def test_generate_all_resume_skips_completed_start(self) -> None:
        argv = ["generate_all_records.py", "--games", "3", "--resume"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(generate_all_records, "iter_start_boards", return_value=[INITIAL_BOARD]),
            mock.patch.object(generate_all_records, "count_unique_records", return_value=3),
            mock.patch.object(generate_all_records, "ensure_generation_manifest") as ensure_manifest,
            mock.patch.object(generate_all_records.subprocess, "run") as run,
        ):
            self.assertEqual(0, generate_all_records.main())
        run.assert_not_called()
        ensure_manifest.assert_called_once()

    def test_generate_all_forwards_executable_override(self) -> None:
        requested_exe = Path("C:/custom/egaroucid.exe")
        argv = [
            "generate_all_records.py",
            "--games", "1",
            "--limit", "1",
            "--exe", str(requested_exe),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(generate_all_records, "iter_start_boards", return_value=[INITIAL_BOARD]),
            mock.patch.object(generate_all_records.subprocess, "run") as run,
        ):
            self.assertEqual(0, generate_all_records.main())

        command = run.call_args.args[0]
        self.assertEqual(str(requested_exe), command[command.index("--exe") + 1])

    def test_record_directory_lock_prevents_duplicate_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            records_dir.mkdir()
            output = root / "book.egcb"
            args = self.make_args(games=1)
            entered = threading.Event()
            release = threading.Event()
            calls: list[int] = []
            errors: list[BaseException] = []
            results: list[int] = []

            def fake_run(_args, _board, _out_dir, n_games, _use_book):
                calls.append(n_games)
                entered.set()
                self.assertTrue(release.wait(timeout=5))
                write_records(records_dir / "records.txt", ["d3"])

            def worker() -> None:
                try:
                    results.append(generate_records.generate_to_target(
                        args, INITIAL_BOARD, records_dir, output
                    ))
                except BaseException as exc:
                    errors.append(exc)

            with (
                mock.patch.object(generate_records, "run_record_batch", side_effect=fake_run),
                mock.patch.object(generate_records, "build_provisional_book"),
            ):
                first = threading.Thread(target=worker)
                second = threading.Thread(target=worker)
                first.start()
                self.assertTrue(entered.wait(timeout=5))
                second.start()
                time.sleep(0.15)
                self.assertEqual([1], calls)
                release.set()
                first.join(timeout=5)
                second.join(timeout=5)

            self.assertEqual([], errors)
            self.assertEqual([1], calls)
            self.assertEqual([1, 1], sorted(results))

    def test_stale_or_legacy_provisional_book_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            records_dir.mkdir()
            write_records(records_dir / "records.txt", ["d3"])
            output = root / "book.egcb"
            output.write_text(valid_book_text(), encoding="utf-8", newline="\n")
            args = self.make_args(games=2)

            self.assertFalse(generate_records.can_use_provisional_book(
                args, INITIAL_BOARD, records_dir, output
            ))

            spec = generate_records.provisional_book_spec(
                args, INITIAL_BOARD, records_dir, output
            )

            def fake_runner(cmd, *, cwd, check):
                staging = Path(cmd[cmd.index("--output") + 1])
                staging.write_text(valid_book_text(), encoding="utf-8", newline="\n")
                return subprocess.CompletedProcess(cmd, 0)

            build_book_atomically(spec, runner=fake_runner)
            self.assertTrue(generate_records.can_use_provisional_book(
                args, INITIAL_BOARD, records_dir, output
            ))
            write_records(records_dir / "records.txt", ["d3", "c3"])
            self.assertFalse(generate_records.can_use_provisional_book(
                args, INITIAL_BOARD, records_dir, output
            ))


class BookArtifactTests(unittest.TestCase):
    def make_spec(self, root: Path) -> BookBuildSpec:
        records_dir = root / "records"
        records_dir.mkdir()
        write_records(records_dir / "records.txt", ["d3"])
        return BookBuildSpec(
            initial_board=INITIAL_BOARD,
            records_dir=records_dir,
            output=root / "trained" / "book.egcb",
            max_book_loss=4,
            cut_empty=30,
            include_game_records=False,
        )

    def test_validation_rejects_partial_book(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "partial.egcb"
            output.write_text(
                "# contest_book_v1\n" + f"# initial {INITIAL_BOARD}\n",
                encoding="utf-8",
            )
            with self.assertRaises(BookValidationError):
                validate_book_file(output, INITIAL_BOARD)

    def test_persistent_lock_file_is_not_treated_as_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "book.egcb.lock"
            lock_path.write_bytes(b"\0")

            with file_lock(lock_path, timeout_seconds=0.5):
                pass
            self.assertTrue(lock_path.exists())
            with file_lock(lock_path, timeout_seconds=0.5):
                pass

    def test_non_object_manifest_is_reported_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.make_spec(root)
            spec.output.parent.mkdir(parents=True)
            spec.output.write_text(valid_book_text(), encoding="utf-8", newline="\n")
            manifest_path_for_book(spec.output).write_text("[]\n", encoding="utf-8")

            status = check_book_status(spec)

            self.assertFalse(status.current)
            self.assertIn("not an object", status.reason)

    def test_input_fingerprint_io_error_is_reported_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.make_spec(root)

            def fake_runner(cmd, *, cwd, check):
                staging = Path(cmd[cmd.index("--output") + 1])
                staging.write_text(valid_book_text(), encoding="utf-8", newline="\n")
                return subprocess.CompletedProcess(cmd, 0)

            build_book_atomically(spec, runner=fake_runner)
            with mock.patch.object(
                book_artifact, "_build_identity", side_effect=OSError("input disappeared")
            ):
                status = check_book_status(spec)

            self.assertFalse(status.current)
            self.assertIn("cannot fingerprint", status.reason)

    def test_atomic_build_writes_manifest_and_detects_stale_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.make_spec(root)

            def fake_runner(cmd, *, cwd, check):
                self.assertTrue(check)
                self.assertEqual(SCRIPT_DIR, cwd)
                staging = Path(cmd[cmd.index("--output") + 1])
                staging.write_text(valid_book_text(), encoding="utf-8", newline="\n")
                return subprocess.CompletedProcess(cmd, 0)

            metadata = build_book_atomically(spec, runner=fake_runner)
            self.assertEqual(1, metadata.records_used)
            self.assertTrue(spec.output.exists())
            self.assertTrue(manifest_path_for_book(spec.output).exists())
            self.assertTrue(check_book_status(spec).current)
            skip_runner = mock.Mock()
            outcome = build_book_if_stale(spec, runner=skip_runner)
            self.assertFalse(outcome.built)
            skip_runner.assert_not_called()

            write_records(spec.records_dir / "records.txt", ["d3", "c3"])
            status = check_book_status(spec)
            self.assertFalse(status.current)
            self.assertIn("inputs", status.reason)

    def test_manifest_detects_build_option_and_output_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.make_spec(root)

            def fake_runner(cmd, *, cwd, check):
                staging = Path(cmd[cmd.index("--output") + 1])
                staging.write_text(valid_book_text(), encoding="utf-8", newline="\n")
                return subprocess.CompletedProcess(cmd, 0)

            build_book_atomically(spec, runner=fake_runner)
            changed_options = BookBuildSpec(
                initial_board=spec.initial_board,
                records_dir=spec.records_dir,
                output=spec.output,
                max_book_loss=5,
                cut_empty=spec.cut_empty,
                include_game_records=spec.include_game_records,
            )
            self.assertFalse(check_book_status(changed_options).current)

            spec.output.write_text(
                valid_book_text().replace("d3:0", "d3:1"),
                encoding="utf-8",
                newline="\n",
            )
            status = check_book_status(spec)
            self.assertFalse(status.current)
            self.assertIn("content", status.reason)

    def test_invalid_staging_does_not_replace_existing_book(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.make_spec(root)
            spec.output.parent.mkdir(parents=True)
            original = valid_book_text(records_seen=7, records_used=6)
            spec.output.write_text(original, encoding="utf-8", newline="\n")

            def fake_runner(cmd, *, cwd, check):
                staging = Path(cmd[cmd.index("--output") + 1])
                staging.write_text("# contest_book_v1\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0)

            with self.assertRaises(BookValidationError):
                build_book_atomically(spec, runner=fake_runner)
            self.assertEqual(original, spec.output.read_text(encoding="utf-8"))
            self.assertEqual([], list(spec.output.parent.glob(".*.tmp")))

    def test_real_builder_publishes_valid_empty_book(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            records_dir.mkdir()
            spec = BookBuildSpec(
                initial_board=INITIAL_BOARD,
                records_dir=records_dir,
                output=root / "trained" / "book.egcb",
                max_book_loss=4,
                cut_empty=30,
                include_game_records=False,
            )

            metadata = build_book_atomically(spec)

            self.assertEqual(0, metadata.records_seen)
            self.assertEqual(0, metadata.records_used)
            self.assertTrue(check_book_status(spec).current)

    def test_output_lock_serializes_book_and_manifest_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_spec = self.make_spec(root)
            second_spec = BookBuildSpec(
                initial_board=first_spec.initial_board,
                records_dir=first_spec.records_dir,
                output=first_spec.output,
                max_book_loss=5,
                cut_empty=first_spec.cut_empty,
                include_game_records=first_spec.include_game_records,
            )
            first_entered = threading.Event()
            release_first = threading.Event()
            second_entered = threading.Event()
            errors: list[BaseException] = []

            def runner(cmd, *, cwd, check):
                max_loss = int(cmd[cmd.index("--max-book-loss") + 1])
                staging = Path(cmd[cmd.index("--output") + 1])
                staging.write_text(valid_book_text(), encoding="utf-8", newline="\n")
                if max_loss == 4:
                    first_entered.set()
                    self.assertTrue(release_first.wait(timeout=5))
                else:
                    second_entered.set()
                return subprocess.CompletedProcess(cmd, 0)

            def worker(spec: BookBuildSpec) -> None:
                try:
                    build_book_atomically(spec, runner=runner)
                except BaseException as exc:
                    errors.append(exc)

            first = threading.Thread(target=worker, args=(first_spec,))
            second = threading.Thread(target=worker, args=(second_spec,))
            first.start()
            self.assertTrue(first_entered.wait(timeout=5))
            second.start()
            time.sleep(0.15)
            self.assertFalse(second_entered.is_set())
            release_first.set()
            first.join(timeout=5)
            second.join(timeout=5)

            self.assertEqual([], errors)
            self.assertTrue(second_entered.is_set())
            self.assertTrue(check_book_status(second_spec).current)
            self.assertFalse(check_book_status(first_spec).current)


class RootTableTests(unittest.TestCase):
    def write_root_book(self, path: Path, move: str = "d3") -> None:
        path.write_text(
            "# contest_book_v1\n"
            f"# initial {INITIAL_BOARD}\n"
            "# records_seen 1\n"
            "# records_used 1\n"
            "# cut_empty 30\n"
            f"{INITIAL_BOARD} 0 {move}:0\n",
            encoding="utf-8",
            newline="\n",
        )

    def test_builds_canonical_verified_root_table_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "books"
            source_dir.mkdir()
            self.write_root_book(source_dir / "first.egcb")
            output = root / build_root_table.ROOT_TABLE_FILENAME

            result = build_root_table.build_root_table(
                [source_dir],
                output,
                root_discs=4,
                required_starts=[INITIAL_BOARD],
            )

            self.assertEqual(1, result["entries"])
            self.assertEqual(
                {"root_discs": 4, "entries": 1},
                build_root_table.validate_root_table(output, expected_root_discs=4),
            )
            root_discs, entries = build_root_table.load_root_table_entries(output, 4)
            self.assertEqual(4, root_discs)
            self.assertEqual(1, len(entries))
            manifest = json.loads(
                build_root_table.manifest_path_for_root_table(output).read_text(encoding="utf-8")
            )
            self.assertEqual("contest_root_table_manifest_v1", manifest["schema"])
            self.assertEqual(1, manifest["source_count"])
            self.assertEqual(1, manifest["entries"])

    def test_rejects_conflicting_duplicate_canonical_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            self.write_root_book(first_dir / "first.egcb", "d3")
            self.write_root_book(second_dir / "second.egcb", "e6")

            with self.assertRaisesRegex(ValueError, "conflicting verified roots"):
                build_root_table.build_root_table(
                    [first_dir, second_dir],
                    root / build_root_table.ROOT_TABLE_FILENAME,
                    root_discs=4,
                )

    def test_accepts_compact_teacher_root_rows_without_deep_books(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher_rows = root / "teacher_roots.txt"
            teacher_rows.write_text(
                "# level-33 verified roots\n"
                f"{INITIAL_BOARD} 0 d3:0\n",
                encoding="utf-8",
                newline="\n",
            )
            output = root / build_root_table.ROOT_TABLE_FILENAME

            result = build_root_table.build_root_table(
                [],
                output,
                root_discs=4,
                required_starts=[INITIAL_BOARD],
                root_result_files=[teacher_rows],
            )

            self.assertEqual(1, result["entries"])
            manifest = json.loads(
                build_root_table.manifest_path_for_root_table(output).read_text(encoding="utf-8")
            )
            self.assertEqual("root_result", manifest["sources"][0]["kind"])


class GgsRootCollectionTests(unittest.TestCase):
    @staticmethod
    def start_log(match_id: str, board: str, game_id: str | None = None) -> str:
        cells, side = board.split()
        game_id = game_id or f"{match_id}.0"
        return (
            f"GGS RECV> /os: -  {match_id} 2600 egrcd 01:00//00:30 s8r14 R 2600 nyanyan\n"
            "GGS INFO> match start!\n"
            f"GGS INFO> ggs pending search wait {game_id} max 350 {cells} {side}\n"
        )

    def test_deduplicates_symmetric_roots_and_reports_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            books = root / "books"
            books.mkdir()
            RootTableTests().write_root_book(books / "root.egcb")
            table = books / build_root_table.ROOT_TABLE_FILENAME
            build_root_table.build_root_table([books], table, root_discs=4)
            first_log = root / "first.log"
            second_log = root / "second.log"
            first_log.write_text(self.start_log(".41", INITIAL_BOARD), encoding="utf-8")
            symmetric = transform_board_text(INITIAL_BOARD, 1)
            second_log.write_text(self.start_log(".42", symmetric), encoding="utf-8")

            report = collect_ggs_roots.collect_coverage(
                [first_log, second_log], [books], table, root_discs=4
            )

            self.assertEqual(2, report["observed_start_events"])
            self.assertEqual(1, report["unique_canonical_roots"])
            self.assertEqual(
                {"deep_book": 1, "root_table": 1, "either": 1, "uncovered": 0},
                report["coverage"],
            )
            root_row = report["roots"][0]
            self.assertEqual(2, root_row["observed"])
            self.assertTrue(root_row["deep_book"])
            self.assertTrue(root_row["root_table"])
            self.assertEqual([".41", ".42"], [row["match_id"] for row in root_row["occurrences"]])

    def test_keeps_distinct_match_and_game_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.log"
            path.write_text(self.start_log(".70", INITIAL_BOARD, ".59.0"), encoding="utf-8")
            occurrence = collect_ggs_roots.parse_ggs_start_roots(path, root_discs=4)[0]
            self.assertEqual(".70", occurrence.match_id)
            self.assertEqual(".59.0", occurrence.game_id)

    def test_rejects_truncated_match_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "truncated.log"
            path.write_text(
                "GGS RECV> /os: -  .41 2600 egrcd 01:00//00:30 s8r14 R 2600 nyanyan\n"
                "GGS INFO> match start!\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing root board"):
                collect_ggs_roots.parse_ggs_start_roots(path, root_discs=4)


class R14CorpusAuditTests(unittest.TestCase):
    def test_fast_canonicalizer_matches_runtime_book_canonicalizer(self) -> None:
        for board in (INITIAL_BOARD, GGS_ROOT):
            for symmetry in range(8):
                oriented = transform_board_text(board, symmetry)
                self.assertEqual(
                    canonicalize_board_key(oriented)[0],
                    audit_r14_corpus.canonicalize_relative_key(Board.from_text(oriented).key()),
                )

    def test_canonicalizes_d4_aliases_and_keeps_reproducible_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            starts = root / "starts"
            starts.mkdir()
            symmetric = transform_board_text(INITIAL_BOARD, 1)
            (starts / "0000000.txt").write_text(
                f"{INITIAL_BOARD}\n{symmetric}\n{INITIAL_BOARD}\n",
                encoding="utf-8",
                newline="\n",
            )

            report = audit_r14_corpus.audit_corpus(starts, root_discs=4)

            self.assertEqual(audit_r14_corpus.CORPUS_REPORT_SCHEMA, report["schema"])
            self.assertEqual(3, report["raw_start_rows"])
            self.assertEqual(2, report["unique_normalized_starts"])
            self.assertEqual(1, report["unique_canonical_roots"])
            self.assertEqual(1, report["duplicate_normalized_rows"])
            self.assertEqual(2, report["d4_alias_rows"])
            self.assertEqual(3, report["max_rows_for_one_canonical_root"])
            self.assertEqual(64, len(report["canonical_roots_sha256"]))
            self.assertEqual(
                {"deep_book": 0, "root_table": 0, "either": 0, "uncovered": 1},
                report["coverage"],
            )
            self.assertEqual(3, report["roots"][0]["source_rows"])

    def test_rejects_start_with_unexpected_disc_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            starts = Path(temporary) / "starts"
            starts.mkdir()
            (starts / "0000000.txt").write_text("X" * 64 + " X\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "start has 64 discs, expected 4"):
                audit_r14_corpus.audit_corpus(starts, root_discs=4)


class GgsRootTeacherTests(unittest.TestCase):
    @staticmethod
    def coverage_report(board: str) -> dict[str, object]:
        return {
            "schema": collect_ggs_roots.REPORT_SCHEMA,
            "root_discs": 14,
            "roots": [
                {"canonical_board": board, "deep_book": False, "root_table": False},
                {"canonical_board": board, "deep_book": True, "root_table": True},
            ],
        }

    def test_parses_one_legal_search_result(self) -> None:
        output = (
            "|          Level|          Depth|           Move|          Score|           Time|          Nodes|            NPS|\n"
            "|             27|         27@74%|             f5|            -15|  000:00:02.786|      239460993|       85951540|\n"
        )
        self.assertEqual(
            {
                "move": "f5",
                "score": -15,
                "level": "27",
                "depth": "27@74%",
                "time": "000:00:02.786",
                "nodes": 239460993,
                "nps": 85951540,
            },
            generate_ggs_root_teacher.parse_search_result(output, GGS_ROOT),
        )

    def test_atomic_output_write_flushes_before_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "teacher_rows.txt"
            with mock.patch.object(generate_ggs_root_teacher.os, "fsync") as fsync:
                generate_ggs_root_teacher._atomic_write_text(output, "verified\n")
            self.assertEqual("verified\n", output.read_text(encoding="utf-8"))
            fsync.assert_called_once()

    def test_checkpoint_counter_uses_total_completed_roots_after_resume(self) -> None:
        state = {
            "results": {"one": {}, "two": {}, "three": {}},
            "rejections": {"four": {}},
        }
        self.assertEqual(
            4,
            generate_ggs_root_teacher._completed_since_checkpoint(state, 500),
        )
        self.assertEqual(
            0,
            generate_ggs_root_teacher._completed_since_checkpoint(state, 2),
        )

    def test_search_root_accepts_console_result_on_stderr(self) -> None:
        table = (
            "|             27|         27@74%|             f5|            -15|  000:00:02.786|      239460993|       85951540|\n"
        )
        completed = SimpleNamespace(returncode=0, stdout="", stderr=table)
        with mock.patch.object(generate_ggs_root_teacher.subprocess, "run", return_value=completed):
            result = generate_ggs_root_teacher.search_root(
                Path("C:/teacher.exe"), GGS_ROOT, 60.0, 28, 29
            )
        self.assertEqual("f5", result["move"])
        self.assertEqual(-15, result["score"])

    def test_rejects_teacher_below_minimum_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "below 33@74%"):
            generate_ggs_root_teacher.validate_quality(
                {"depth": "32@88%"}, 33, 74
            )

    def test_loads_uncovered_roots_from_full_r14_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            coverage = Path(temporary) / "r14_audit.json"
            coverage.write_text(
                json.dumps(
                    {
                        "schema": audit_r14_corpus.CORPUS_REPORT_SCHEMA,
                        "root_discs": 14,
                        "roots": [
                            {
                                "canonical_board": GGS_ROOT,
                                "source_rows": 1,
                                "deep_book": False,
                                "root_table": False,
                            },
                            {
                                "canonical_board": GGS_ROOT,
                                "source_rows": 1,
                                "deep_book": True,
                                "root_table": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            self.assertEqual(
                [GGS_ROOT],
                generate_ggs_root_teacher.load_uncovered_roots(coverage),
            )

    def test_seeded_cohort_order_is_reproducible(self) -> None:
        roots = ["root-c", "root-a", "root-b", "root-d"]
        first = generate_ggs_root_teacher.select_teacher_roots(roots, 3, 620)
        self.assertEqual(
            first,
            generate_ggs_root_teacher.select_teacher_roots(
                list(reversed(roots)), 3, 620
            ),
        )
        self.assertEqual(3, len(first))
        self.assertEqual(
            ["root-a", "root-b", "root-c"],
            generate_ggs_root_teacher.select_teacher_roots(roots, 3, None),
        )

    def test_excludes_previous_teacher_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_root, _ = canonicalize_board_key(GGS_ROOT)
            excluded_rows = root / "earlier_teacher_rows.txt"
            excluded_rows.write_text(
                "# ggs_root_teacher_v1\n"
                f"{GGS_ROOT} -15 f5:-15\n",
                encoding="utf-8",
                newline="\n",
            )
            self.assertEqual(
                {canonical_root},
                generate_ggs_root_teacher.load_excluded_roots([excluded_rows]),
            )

            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(canonical_root)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            with self.assertRaisesRegex(ValueError, "no uncovered 14-disc roots remain after exclusions"):
                generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    root / "teacher_rows.txt",
                    60.0,
                    28,
                    29,
                    excluded_root_files=[excluded_rows],
                )

    def test_replays_durable_position_update_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            output = root / "teacher_rows.txt"
            expected = generate_ggs_root_teacher._new_state(
                coverage,
                exe,
                [GGS_ROOT],
                60.0,
                28,
                29,
                33,
                74,
                33,
                "hint",
                33,
                0,
                None,
                [],
            )
            generate_ggs_root_teacher._write_outputs(output, expected)
            entry = {
                "move": "f5",
                "score": -15,
                "level": "33",
                "depth": "33@74%",
                "time": "000:00:02.000",
                "nodes": 1,
                "nps": 1,
                "method": "hint_level_33",
            }
            generate_ggs_root_teacher._append_pending_update(
                output, GGS_ROOT, "results", entry
            )
            resumed = generate_ggs_root_teacher._load_state(
                generate_ggs_root_teacher._state_path(output), expected
            )
            self.assertTrue(generate_ggs_root_teacher._apply_pending_updates(output, resumed))
            self.assertEqual(entry, resumed["results"][GGS_ROOT])
            generate_ggs_root_teacher._write_outputs(output, resumed)
            generate_ggs_root_teacher._clear_pending_updates(output)
            self.assertFalse(generate_ggs_root_teacher._pending_updates_path(output).exists())

    def test_progress_report_counts_durable_updates_before_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            output = root / "teacher_rows.txt"
            state = generate_ggs_root_teacher._new_state(
                coverage,
                exe,
                [GGS_ROOT],
                60.0,
                28,
                29,
                33,
                74,
                33,
                "hint",
                33,
                0,
                None,
                [],
            )
            generate_ggs_root_teacher._write_outputs(output, state)
            entry = {
                "move": "f5",
                "score": -15,
                "level": "33",
                "depth": "33@74%",
                "time": "000:00:02.000",
                "nodes": 1,
                "nps": 1,
                "method": "hint_level_33",
            }
            generate_ggs_root_teacher._append_pending_update(
                output, GGS_ROOT, "results", entry
            )
            report = root / "progress.md"
            counts = report_ggs_root_teacher_progress.write_progress_report(
                output.with_suffix(output.suffix + ".state.json"), report
            )
            self.assertEqual(1, counts["accepted"])
            self.assertEqual(1, counts["processed"])
            self.assertEqual(0, counts["compacted_accepted"])
            self.assertEqual(1, counts["pending_records"])
            text = report.read_text(encoding="utf-8")
            self.assertIn("受理済み局面数", text)
            self.assertIn("Accepted positions", text)

    def test_generates_and_resumes_only_with_identical_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            output = root / "teacher_rows.txt"
            result = {
                "move": "f5",
                "score": -15,
                "level": "time",
                "depth": "33@74%",
                "time": "000:00:54.500",
                "nodes": 1,
                "nps": 1,
            }
            with mock.patch.object(generate_ggs_root_teacher, "search_root_at_level", return_value=result) as search:
                self.assertEqual(
                    {"completed": 1, "requested": 1},
                    generate_ggs_root_teacher.generate_teachers(
                        coverage, exe, output, 60.0, 28, 29
                    ),
                )
                search.assert_called_once_with(exe, GGS_ROOT, 33, 28, 29)
            self.assertIn(f"{GGS_ROOT} -15 f5:-15", output.read_text(encoding="utf-8"))
            with mock.patch.object(generate_ggs_root_teacher, "search_root") as search:
                self.assertEqual(
                    {"completed": 1, "requested": 1},
                    generate_ggs_root_teacher.generate_teachers(
                        coverage, exe, output, 60.0, 28, 29, resume=True
                    ),
                )
                search.assert_not_called()
            with self.assertRaisesRegex(ValueError, "resume mismatch for threads"):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, output, 60.0, 27, 29, resume=True
                )

    def test_uses_level_33_fallback_when_time_search_is_shallow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            output = root / "teacher_rows.txt"
            shallow = {
                "move": "f5", "score": -15, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            fallback = {
                "move": "f5", "score": -14, "level": "33", "depth": "33@74%",
                "time": "000:00:02.000", "nodes": 2, "nps": 1,
            }
            with (
                mock.patch.object(generate_ggs_root_teacher, "search_root", return_value=shallow),
                mock.patch.object(generate_ggs_root_teacher, "search_root_at_level", return_value=fallback) as level_search,
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, output, 60.0, 28, 29, fallback_level=33,
                    method="time_then_hint",
                )
            level_search.assert_called_once_with(exe, GGS_ROOT, 33, 28, 29)
            manifest = json.loads(
                output.with_suffix(output.suffix + ".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("hint_level_33", manifest["results"][GGS_ROOT]["method"])

    def test_time_teacher_requires_matching_level_27_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            output = root / "teacher_rows.txt"
            primary = {
                "move": "f5", "score": -15, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            verification = {
                "move": "f5", "score": -14, "level": "27", "depth": "27@74%",
                "time": "000:00:02.000", "nodes": 2, "nps": 1,
            }
            with (
                mock.patch.object(generate_ggs_root_teacher, "search_root", return_value=primary),
                mock.patch.object(generate_ggs_root_teacher, "search_root_at_level", return_value=verification),
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, output, 60.0, 28, 29, min_depth=30,
                    method="time_then_verify", verify_level=27,
                )
            manifest = json.loads(
                output.with_suffix(output.suffix + ".manifest.json").read_text(encoding="utf-8")
            )
            saved = manifest["results"][GGS_ROOT]
            self.assertEqual("time_verified_hint_level_27", saved["method"])
            self.assertEqual("f5", saved["verification"]["move"])
            self.assertEqual("level_27_exact", saved["verification_mode"])
            self.assertEqual(31, manifest["deep_tiebreak_level"])

    def test_time_teacher_retries_shallow_time_search_at_quality_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            output = root / "teacher_rows.txt"
            shallow = {
                "move": "d3", "score": -15, "level": "-", "depth": "28@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            fallback = {
                "move": "f5", "score": -14, "level": "30", "depth": "30@74%",
                "time": "000:00:06.000", "nodes": 2, "nps": 1,
            }
            verification = {
                "move": "f5", "score": -13, "level": "27", "depth": "27@74%",
                "time": "000:00:02.000", "nodes": 3, "nps": 1,
            }
            with (
                mock.patch.object(generate_ggs_root_teacher, "search_root", return_value=shallow),
                mock.patch.object(
                    generate_ggs_root_teacher,
                    "search_root_at_level",
                    side_effect=[fallback, verification],
                ) as level_search,
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, output, 60.0, 28, 29, min_depth=30,
                    fallback_level=30, method="time_then_verify", verify_level=27,
                )
            self.assertEqual(
                [
                    mock.call(exe, GGS_ROOT, 30, 28, 29),
                    mock.call(exe, GGS_ROOT, 27, 28, 29),
                ],
                level_search.call_args_list,
            )
            manifest = json.loads(
                output.with_suffix(output.suffix + ".manifest.json").read_text(encoding="utf-8")
            )
            saved = manifest["results"][GGS_ROOT]
            self.assertEqual(
                "time_fallback_hint_level_30_verified_hint_level_27", saved["method"]
            )
            self.assertEqual("28@74%", saved["primary"]["depth"])
            self.assertEqual("f5", saved["verification"]["move"])

    def test_time_teacher_accepts_matching_level_30_tiebreak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            primary = {
                "move": "b4", "score": -15, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            first_tied = {
                "move": "f5", "score": -15, "level": "27", "depth": "27@74%",
                "time": "000:00:02.000", "nodes": 2, "nps": 1,
            }
            tiebreak = {
                "move": "b4", "score": -15, "level": "30", "depth": "30@74%",
                "time": "000:00:01.000", "nodes": 3, "nps": 1,
            }
            with (
                mock.patch.object(generate_ggs_root_teacher, "search_root", return_value=primary),
                mock.patch.object(
                    generate_ggs_root_teacher,
                    "search_root_at_level",
                    side_effect=[first_tied, tiebreak],
                ),
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, root / "teacher_rows.txt", 60.0, 28, 29, min_depth=30,
                    fallback_level=30, method="time_then_verify", verify_level=27,
                )
            manifest = json.loads(
                (root / "teacher_rows.txt.manifest.json").read_text(encoding="utf-8")
            )
            saved = manifest["results"][GGS_ROOT]
            self.assertEqual("b4", saved["tiebreak"]["move"])
            self.assertEqual("level_30_tiebreak", saved["verification_mode"])

    def test_time_teacher_promotes_agreeing_deep_tiebreak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            primary = {
                "move": "f5", "score": -12, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            verification = {
                "move": "b4", "score": -12, "level": "27", "depth": "27@74%",
                "time": "000:00:02.000", "nodes": 2, "nps": 1,
            }
            tiebreak = {
                "move": "c7", "score": -11, "level": "30", "depth": "30@74%",
                "time": "000:00:03.000", "nodes": 3, "nps": 1,
            }
            deep_tiebreak = {
                "move": "c7", "score": -10, "level": "31", "depth": "31@74%",
                "time": "000:00:04.000", "nodes": 4, "nps": 1,
            }
            with (
                mock.patch.object(generate_ggs_root_teacher, "search_root", return_value=primary),
                mock.patch.object(
                    generate_ggs_root_teacher,
                    "search_root_at_level",
                    side_effect=[verification, tiebreak, deep_tiebreak],
                ),
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, root / "teacher_rows.txt", 60.0, 28, 29, min_depth=30,
                    fallback_level=30, method="time_then_verify", verify_level=27,
                )
            manifest = json.loads(
                (root / "teacher_rows.txt.manifest.json").read_text(encoding="utf-8")
            )
            saved = manifest["results"][GGS_ROOT]
            self.assertEqual("c7", saved["move"])
            self.assertEqual("f5", saved["primary"]["move"])
            self.assertEqual("b4", saved["verification"]["move"])
            self.assertEqual("c7", saved["tiebreak"]["move"])
            self.assertEqual("levels_30_31_tiebreak", saved["verification_mode"])

    def test_time_teacher_rejects_quality_fallback_below_minimum_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            with self.assertRaisesRegex(ValueError, "fallback_level at least min_depth"):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, root / "teacher_rows.txt", 60.0, 28, 29,
                    min_depth=30, fallback_level=29,
                    method="time_then_verify", verify_level=27,
                )

    def test_rejects_mismatched_verification_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            primary = {
                "move": "f5", "score": -15, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            mismatch = {
                "move": "d3", "score": -15, "level": "27", "depth": "27@74%",
                "time": "000:00:02.000", "nodes": 2, "nps": 1,
            }
            tiebreak_mismatch = {
                "move": "d3", "score": -15, "level": "30", "depth": "30@74%",
                "time": "000:00:02.000", "nodes": 3, "nps": 1,
            }
            deep_tiebreak_mismatch = {
                "move": "b4", "score": -15, "level": "31", "depth": "31@74%",
                "time": "000:00:02.000", "nodes": 4, "nps": 1,
            }
            with (
                mock.patch.object(generate_ggs_root_teacher, "search_root", return_value=primary),
                mock.patch.object(
                    generate_ggs_root_teacher,
                    "search_root_at_level",
                    side_effect=[mismatch, tiebreak_mismatch, deep_tiebreak_mismatch],
                ),
            ):
                result = generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, root / "teacher_rows.txt", 60.0, 28, 29, min_depth=30,
                    fallback_level=30, method="time_then_verify", verify_level=27,
                )
            self.assertEqual({"completed": 0, "requested": 1}, result)
            manifest = json.loads(
                (root / "teacher_rows.txt.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, manifest["output"]["rejected"])
            self.assertEqual(
                "level-30 tiebreak d3 does not match level-31 tiebreak b4",
                manifest["rejections"][GGS_ROOT]["reason"],
            )


if __name__ == "__main__":
    unittest.main()
