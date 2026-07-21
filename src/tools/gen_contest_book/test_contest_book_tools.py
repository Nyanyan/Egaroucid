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


INITIAL_BOARD = "---------------------------OX------XO--------------------------- X"


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


if __name__ == "__main__":
    unittest.main()
