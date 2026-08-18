from __future__ import annotations

import json
import hashlib
import random
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
import audit_root_table_matches
import audit_root_teacher_method_benchmark
import benchmark_root_teacher_methods
import book_artifact
import audit_r14_corpus
import build_root_table
import collect_ggs_roots
import generate_ggs_root_teacher
import prepare_root_table_match
import publish_verified_root_table
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
from othello import Board, index_to_coord, normalize_board_text


INITIAL_BOARD = "---------------------------OX------XO--------------------------- X"
GGS_ROOT = "------------------XXXX----XOOX----OXX-----OXO------------------- X"


def write_records(path: Path, transcripts: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for transcript in transcripts:
            f.write(f"initial board: {INITIAL_BOARD}\n")
            f.write(f"transcript: {transcript}\n\n")


def write_adversarial_records(path: Path, records: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for transcript, engine_parity in records:
            f.write(f"generation mode: {generate_records.ADVERSARIAL_GENERATION_MODE}\n")
            f.write(f"engine parity: {engine_parity}\n")
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


def write_teacher_execution_resources(exe: Path) -> None:
    resources = exe.parent / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    (resources / "eval.egev2").write_bytes(b"test evaluation")
    (resources / "eval_move_ordering_end.egev").write_bytes(b"test move ordering")


def write_teacher_executable(exe: Path, content: bytes) -> None:
    exe.write_bytes(content)
    write_teacher_execution_resources(exe)


def saved_teacher_execution_executable(output: Path) -> Path:
    state = json.loads(output.with_suffix(output.suffix + ".state.json").read_text(encoding="utf-8"))
    return Path(
        state["calculation_provenance"]["execution_environment"]["executable"]["snapshot"]["path"]
    )


def write_v13_teacher_manifest(
    teacher: Path,
    *,
    engine: Path | None = None,
    include_match_requirements: bool = False,
) -> None:
    """Create one small, internally consistent v13 teacher artifact for tests."""
    if engine is None:
        engine = teacher.parent / "teacher.exe"
        engine.write_bytes(b"test teacher executable")
    write_teacher_execution_resources(engine)
    provenance = generate_ggs_root_teacher._new_calculation_provenance(teacher, engine, 29)
    source_snapshot = Path(provenance["teacher_script_snapshot"]["path"])
    source_snapshot.write_bytes(Path(generate_ggs_root_teacher.__file__).read_bytes())
    environment = provenance["execution_environment"]
    executable = environment["executable"]
    generate_ggs_root_teacher._copy_or_verify_saved_file(
        executable["source"], executable["snapshot"], "test executable"
    )
    for resource in environment["resources"]:
        generate_ggs_root_teacher._copy_or_verify_saved_file(
            resource["source"], resource["snapshot"], f"test {resource['role']} resource"
        )
    entry = build_root_table.load_root_rows(teacher, 14)[0]
    manifest: dict[str, object] = {
        "schema": generate_ggs_root_teacher.TEACHER_MANIFEST_SCHEMA,
        "output": {
            "sha256": build_root_table.sha256_file(teacher),
            "processed": 1,
            "completed": 1,
            "rejected": 0,
        },
        "calculation_provenance": provenance,
        "random_seed": provenance["random_seed"],
        "results": {
            entry.board: {
                "move": index_to_coord(entry.moves[0][0]),
                "score": entry.value,
            }
        },
        "rejections": {},
    }
    manifest["engine"] = {
        "path": engine.resolve().as_posix(),
        "sha256": build_root_table.sha256_file(engine),
    }
    if include_match_requirements:
        manifest.update(
            {
                "time_seconds": 60,
                "threads": 28,
                "hash_level": 29,
                "min_depth": 30,
                "min_selectivity": 74,
                "verify_level": 31,
            }
        )
    teacher.with_suffix(teacher.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8", newline="\n"
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
            adversarial_games=4,
            adversarial_batch_size=2,
            adversarial_level=21,
            adversarial_reply_margin=2,
            adversarial_reply_width=3,
            adversarial_engine_width=2,
            adversarial_max_rounds=3,
            adversarial_stable_rounds=1,
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

    def test_count_adversarial_records_separates_engine_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_dir = Path(temporary)
            write_records(records_dir / "regular.txt", ["regular"])
            write_adversarial_records(
                records_dir / "adversarial.txt",
                [("attack-a", 0), ("attack-b", 1), ("attack-c", 0)],
            )

            self.assertEqual(
                3,
                generate_records.count_adversarial_records(records_dir, INITIAL_BOARD),
            )
            self.assertEqual(
                2,
                generate_records.count_adversarial_records(records_dir, INITIAL_BOARD, 0),
            )
            self.assertEqual(
                1,
                generate_records.count_adversarial_records(records_dir, INITIAL_BOARD, 1),
            )

    def test_legacy_adversarial_records_do_not_satisfy_v3_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_dir = Path(temporary)
            (records_dir / "legacy.txt").write_text(
                "generation mode: adversarial\n"
                "engine parity: 0\n"
                f"initial board: {INITIAL_BOARD}\n"
                "transcript: legacy-attack\n\n"
                "generation mode: adversarial_v2\n"
                "engine parity: 1\n"
                f"initial board: {INITIAL_BOARD}\n"
                "transcript: legacy-v2-attack\n\n",
                encoding="utf-8",
            )

            self.assertEqual(
                0,
                generate_records.count_adversarial_records(records_dir, INITIAL_BOARD),
            )

    def test_stabilization_certificate_is_tied_to_records_book_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            records_dir.mkdir()
            write_adversarial_records(records_dir / "adversarial.txt", [("attack-a", 0)])
            output = root / "book.egcb"
            output.write_text(valid_book_text(), encoding="utf-8", newline="\n")
            args = self.make_args(games=3)
            profile = generate_records.adversarial_generation_profile(args)
            manifest = {
                "schema": generate_records.GENERATION_MANIFEST_SCHEMA,
                "initial_board": INITIAL_BOARD,
                "baseline": generate_records.records_snapshot(records_dir, INITIAL_BOARD),
                "batches": [],
                "pending_batch": None,
            }
            generate_records.write_adversarial_stabilization(
                manifest,
                records_dir,
                INITIAL_BOARD,
                output,
                profile,
                "stable",
                [],
                1,
                0,
            )

            self.assertTrue(generate_records.adversarial_stabilization_is_current(
                records_dir, INITIAL_BOARD, output, profile
            ))
            write_adversarial_records(
                records_dir / "adversarial.txt",
                [("attack-a", 0), ("attack-b", 1)],
            )
            self.assertFalse(generate_records.adversarial_stabilization_is_current(
                records_dir, INITIAL_BOARD, output, profile
            ))
            write_adversarial_records(records_dir / "adversarial.txt", [("attack-a", 0)])
            self.assertTrue(generate_records.adversarial_stabilization_is_current(
                records_dir, INITIAL_BOARD, output, profile
            ))

            output.write_text(
                valid_book_text().replace("d3:0", "c3:0"),
                encoding="utf-8",
                newline="\n",
            )
            self.assertFalse(generate_records.adversarial_stabilization_is_current(
                records_dir, INITIAL_BOARD, output, profile
            ))
            output.write_text(valid_book_text(), encoding="utf-8", newline="\n")
            changed_profile = json.loads(json.dumps(profile))
            changed_profile["settings"]["adversarial_reply_width"] = 4
            self.assertFalse(generate_records.adversarial_stabilization_is_current(
                records_dir, INITIAL_BOARD, output, changed_profile
            ))

    def test_adversarial_batch_uses_book_and_role_specific_command(self) -> None:
        args = self.make_args(games=3)
        with mock.patch.object(generate_records.subprocess, "run") as run:
            generate_records.run_adversarial_record_batch(
                args,
                INITIAL_BOARD,
                Path("records"),
                2,
                1,
            )

        command = run.call_args.args[0]
        self.assertIn("-contestbook", command)
        self.assertIn("-contestrecordadv", command)
        option_idx = command.index("-contestrecordadv")
        self.assertEqual(
            [INITIAL_BOARD, "2", "records", "2", "3", "2", "30", "1"],
            command[option_idx + 1:option_idx + 9],
        )

    def test_adversarial_generation_balances_parities_and_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            output = root / "book.egcb"
            args = self.make_args(games=3)
            args.adversarial_batch_size = 1
            output.write_text(valid_book_text(), encoding="utf-8")
            batch_calls = 0

            def fake_batch(
                _manifest,
                _out_dir,
                _initial_board,
                _args,
                _profile,
                _output,
                n_batch,
                parity,
                _target,
                before,
            ):
                nonlocal batch_calls
                batch_calls += 1
                after = dict(before)
                if batch_calls <= 2:
                    after["unique_records"] = int(before["unique_records"]) + n_batch
                return after

            with (
                mock.patch.object(
                    generate_records,
                    "prepare_generation_manifest",
                    return_value=({"batches": [], "pending_batch": None}, {"unique_records": 3}),
                ),
                mock.patch.object(generate_records, "can_use_provisional_book", return_value=True),
                mock.patch.object(generate_records, "count_adversarial_records", return_value=4),
                mock.patch.object(
                    generate_records,
                    "adversarial_stabilization_is_current",
                    return_value=False,
                ),
                mock.patch.object(
                    generate_records,
                    "contest_book_policy_fingerprint",
                    return_value={"rows": 1, "sha256": "policy"},
                ),
                mock.patch.object(
                    generate_records,
                    "fingerprint_files",
                    return_value={"count": 1, "sha256": "book", "total_bytes": 1},
                ),
                mock.patch.object(
                    generate_records,
                    "adversarial_record_generation_batch",
                    side_effect=fake_batch,
                ) as batch,
                mock.patch.object(generate_records, "build_provisional_book") as build,
                mock.patch.object(generate_records, "write_adversarial_stabilization") as write_stable,
            ):
                result = generate_records.generate_adversarial_to_target(
                    args,
                    INITIAL_BOARD,
                    records_dir,
                    output,
                )

            self.assertEqual(4, result)
            self.assertEqual(4, batch.call_count)
            self.assertEqual(1, build.call_count)
            self.assertEqual(
                [1, 0, 1, 0],
                [call.args[7] for call in batch.call_args_list],
            )
            self.assertEqual("stable", write_stable.call_args.args[5])

    def test_adversarial_generation_marks_unconverged_round_budget_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            output = root / "book.egcb"
            output.write_text(valid_book_text(), encoding="utf-8")
            args = self.make_args(games=3)
            args.adversarial_max_rounds = 1

            def fake_batch(
                _manifest,
                _out_dir,
                _initial_board,
                _args,
                _profile,
                _output,
                n_batch,
                _parity,
                _target,
                before,
            ):
                after = dict(before)
                after["unique_records"] = int(before["unique_records"]) + n_batch
                return after

            with (
                mock.patch.object(
                    generate_records,
                    "prepare_generation_manifest",
                    return_value=({"batches": [], "pending_batch": None}, {"unique_records": 3}),
                ),
                mock.patch.object(generate_records, "can_use_provisional_book", return_value=True),
                mock.patch.object(generate_records, "count_adversarial_records", return_value=4),
                mock.patch.object(
                    generate_records,
                    "adversarial_stabilization_is_current",
                    return_value=False,
                ),
                mock.patch.object(
                    generate_records,
                    "contest_book_policy_fingerprint",
                    return_value={"rows": 1, "sha256": "policy"},
                ),
                mock.patch.object(
                    generate_records,
                    "fingerprint_files",
                    return_value={"count": 1, "sha256": "book", "total_bytes": 1},
                ),
                mock.patch.object(
                    generate_records,
                    "adversarial_record_generation_batch",
                    side_effect=fake_batch,
                ),
                mock.patch.object(generate_records, "build_provisional_book") as build,
                mock.patch.object(generate_records, "write_adversarial_stabilization") as write_stable,
            ):
                result = generate_records.generate_adversarial_to_target(
                    args, INITIAL_BOARD, records_dir, output
                )

            self.assertEqual(4, result)
            build.assert_called_once()
            self.assertEqual("budget_exhausted", write_stable.call_args.args[5])

    def test_adversarial_generation_carries_stable_sweeps_across_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            output = root / "book.egcb"
            output.write_text(valid_book_text(), encoding="utf-8")
            args = self.make_args(games=3)
            args.adversarial_max_rounds = 1
            args.adversarial_stable_rounds = 2
            snapshot = {"unique_records": 3}
            book_fingerprint = {"count": 1, "sha256": "book", "total_bytes": 1}
            policy_fingerprint = {"rows": 1, "sha256": "policy"}
            profile = generate_records.adversarial_generation_profile(args)
            manifest = {
                "batches": [],
                "pending_batch": None,
                "adversarial_stabilization": {
                    "schema": generate_records.ADVERSARIAL_STABILIZATION_SCHEMA,
                    "status": "budget_exhausted",
                    "generation_mode": generate_records.ADVERSARIAL_GENERATION_MODE,
                    "profile": profile,
                    "records": snapshot,
                    "book": book_fingerprint,
                    "policy": policy_fingerprint,
                    "rounds": [{"round": 0}],
                    "stable_rounds": 1,
                    "new_records": 0,
                },
            }

            with (
                mock.patch.object(
                    generate_records,
                    "prepare_generation_manifest",
                    return_value=(manifest, snapshot),
                ),
                mock.patch.object(generate_records, "can_use_provisional_book", return_value=True),
                mock.patch.object(generate_records, "count_adversarial_records", return_value=4),
                mock.patch.object(
                    generate_records,
                    "adversarial_stabilization_is_current",
                    return_value=False,
                ),
                mock.patch.object(
                    generate_records,
                    "adversarial_generation_profile",
                    return_value=profile,
                ),
                mock.patch.object(
                    generate_records,
                    "contest_book_policy_fingerprint",
                    return_value=policy_fingerprint,
                ),
                mock.patch.object(
                    generate_records,
                    "fingerprint_files",
                    return_value=book_fingerprint,
                ),
                mock.patch.object(
                    generate_records,
                    "adversarial_record_generation_batch",
                    side_effect=lambda *_args: snapshot,
                ) as batch,
                mock.patch.object(generate_records, "build_provisional_book") as build,
                mock.patch.object(generate_records, "write_adversarial_stabilization") as write_stable,
            ):
                result = generate_records.generate_adversarial_to_target(
                    args, INITIAL_BOARD, records_dir, output
                )

            self.assertEqual(4, result)
            self.assertEqual(2, batch.call_count)
            build.assert_not_called()
            self.assertEqual("stable", write_stable.call_args.args[5])
            self.assertEqual(2, write_stable.call_args.args[7])

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
        argv = [
            "generate_all_records.py", "--games", "3", "--resume", "--start-list-order"
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(generate_all_records, "iter_start_boards", return_value=[INITIAL_BOARD]),
            mock.patch.object(generate_all_records, "count_unique_records", return_value=3),
            mock.patch.object(generate_all_records, "count_adversarial_records", return_value=8),
            mock.patch.object(
                generate_all_records,
                "adversarial_stabilization_is_current",
                return_value=True,
            ),
            mock.patch.object(generate_all_records, "ensure_generation_manifest") as ensure_manifest,
            mock.patch.object(generate_all_records.subprocess, "run") as run,
        ):
            self.assertEqual(0, generate_all_records.main())
        run.assert_not_called()
        ensure_manifest.assert_called_once()

    def test_generate_all_priority_manifest_uses_its_order_and_resume(self) -> None:
        second_board = GGS_ROOT
        manifest = Path("priority.jsonl")
        argv = [
            "generate_all_records.py",
            "--games", "3",
            "--resume",
            "--priority-manifest", str(manifest),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                generate_all_records,
                "load_r14_random_setup_priority_manifest",
                return_value=([INITIAL_BOARD, second_board], {}, {}),
            ) as load_priority,
            mock.patch.object(
                generate_all_records,
                "count_unique_records",
                side_effect=[3, 1],
            ),
            mock.patch.object(generate_all_records, "count_adversarial_records", return_value=8),
            mock.patch.object(
                generate_all_records,
                "adversarial_stabilization_is_current",
                side_effect=[True, False],
            ),
            mock.patch.object(generate_all_records, "ensure_generation_manifest") as ensure_manifest,
            mock.patch.object(generate_all_records.subprocess, "run") as run,
        ):
            self.assertEqual(0, generate_all_records.main())

        load_priority.assert_called_once_with(manifest)
        ensure_manifest.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(second_board, command[2])

    def test_generate_all_forwards_executable_override(self) -> None:
        requested_exe = Path("C:/custom/egaroucid.exe")
        argv = [
            "generate_all_records.py",
            "--games", "1",
            "--limit", "1",
            "--start-list-order",
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
        self.assertEqual("5", command[command.index("--adversarial-games") + 1])
        self.assertEqual("19", command[command.index("--adversarial-level") + 1])
        self.assertEqual("2", command[command.index("--adversarial-engine-width") + 1])
        self.assertEqual("3", command[command.index("--adversarial-max-rounds") + 1])
        self.assertEqual("1", command[command.index("--adversarial-stable-rounds") + 1])

    def test_generate_all_resume_runs_missing_adversarial_records(self) -> None:
        argv = [
            "generate_all_records.py",
            "--games", "3",
            "--adversarial-games", "4",
            "--resume",
            "--start-list-order",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(generate_all_records, "iter_start_boards", return_value=[INITIAL_BOARD]),
            mock.patch.object(generate_all_records, "count_unique_records", return_value=3),
            mock.patch.object(generate_all_records, "count_adversarial_records", return_value=1),
            mock.patch.object(
                generate_all_records,
                "adversarial_stabilization_is_current",
                return_value=False,
            ),
            mock.patch.object(generate_all_records.subprocess, "run") as run,
        ):
            self.assertEqual(0, generate_all_records.main())

        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual("4", command[command.index("--adversarial-games") + 1])

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
    def test_publish_requires_fresh_passing_audit_and_rebuilds_table(self) -> None:
        """The publication command must not trust a hand-edited audit JSON."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "teacher_rows.txt"
            snapshot.write_text(
                "# ggs_root_teacher_v1\n"
                f"{GGS_ROOT} -15 f5:-15\n",
                encoding="utf-8",
                newline="\n",
            )
            prepared_dir = root / "prepared"
            table = prepared_dir / "table" / build_root_table.ROOT_TABLE_FILENAME
            build_root_table.build_root_table([], table, root_result_files=[snapshot])
            table_manifest = build_root_table.manifest_path_for_root_table(table)
            prepared = prepared_dir / "prepared_match_input.json"
            prepared.write_text(
                json.dumps(
                    {
                        "schema": prepare_root_table_match.PREPARED_SCHEMA,
                        "snapshot": {
                            "path": snapshot.resolve().as_posix(),
                            "sha256": build_root_table.sha256_file(snapshot),
                        },
                        "table": {
                            "path": table.resolve().as_posix(),
                            "sha256": build_root_table.sha256_file(table),
                            "entries": 1,
                            "manifest_sha256": build_root_table.sha256_file(table_manifest),
                        },
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            results = root / "results.jsonl"
            metadata = root / "matches.meta.json"
            results.write_text("{}\n", encoding="utf-8", newline="\n")
            metadata.write_text("{}\n", encoding="utf-8", newline="\n")
            audit = {
                "schema": audit_root_table_matches.AUDIT_SCHEMA,
                "results": {
                    "path": results.resolve().as_posix(),
                    "sha256": build_root_table.sha256_file(results),
                },
                "prepared_input": {
                    "path": prepared.resolve().as_posix(),
                    "sha256": build_root_table.sha256_file(prepared),
                },
                "metadata": {
                    "path": metadata.resolve().as_posix(),
                    "sha256": build_root_table.sha256_file(metadata),
                },
                "matches": 500,
                "wins": 300,
                "draws": 50,
                "losses": 150,
                "score_rate": 0.65,
                "mean_margin": 2.0,
                "bootstrap_seed": 624,
                "bootstrap_repetitions": 100_000,
                "minimum_processed": 500,
                "minimum_accepted": 500,
                "score_interval": [0.55, 0.75],
                "margin_interval": [0.1, 3.9],
                "checks": {},
                "teacher_calculation": {},
                "level_31_verification_required": True,
                "failures": [],
                "valid": True,
                "eligible_for_adoption": True,
            }
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8", newline="\n")
            target_dir = root / "trained"
            target = target_dir / build_root_table.ROOT_TABLE_FILENAME

            def fresh_audit(*_args: object, **_kwargs: object) -> dict[str, object]:
                return json.loads(audit_path.read_text(encoding="utf-8"))

            with (
                mock.patch.object(publish_verified_root_table, "TRAINED_DIR", target_dir),
                mock.patch.object(
                    publish_verified_root_table.audit_root_table_matches,
                    "audit_match_results",
                    side_effect=fresh_audit,
                ),
            ):
                publication = publish_verified_root_table.publish_verified_root_table(
                    audit_path, output=target
                )
            self.assertEqual(build_root_table.sha256_file(table), build_root_table.sha256_file(target))
            self.assertEqual(1, publication["published_table"]["entries"])
            publication_path = publish_verified_root_table.publication_path_for_root_table(target)
            self.assertTrue(publication_path.is_file())

            target.unlink()
            build_root_table.manifest_path_for_root_table(target).unlink()
            publication_path.unlink()
            audit["eligible_for_adoption"] = False
            audit_path.write_text(json.dumps(audit), encoding="utf-8", newline="\n")
            with mock.patch.object(publish_verified_root_table, "TRAINED_DIR", target_dir):
                with self.assertRaisesRegex(ValueError, "does not permit publication"):
                    publish_verified_root_table.publish_verified_root_table(audit_path, output=target)
            self.assertFalse(target.exists())

            audit["eligible_for_adoption"] = True
            audit_path.write_text(json.dumps(audit), encoding="utf-8", newline="\n")
            with (
                mock.patch.object(publish_verified_root_table, "TRAINED_DIR", target_dir),
                mock.patch.object(
                    publish_verified_root_table.audit_root_table_matches,
                    "audit_match_results",
                    return_value={**audit, "matches": 499},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "differs from a fresh audit"):
                    publish_verified_root_table.publish_verified_root_table(audit_path, output=target)
            self.assertFalse(target.exists())

    def test_builder_cli_rejects_direct_tournament_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trained = Path(temporary) / "trained"
            with (
                mock.patch.object(build_root_table, "TRAINED_DIR", trained),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "build_root_table.py",
                        "--output",
                        str(trained / build_root_table.ROOT_TABLE_FILENAME),
                    ],
                ),
            ):
                with self.assertRaises(SystemExit) as raised:
                    build_root_table.main()
            self.assertEqual(2, raised.exception.code)

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

    def test_formal_comparison_report_must_be_complete_and_match_teacher_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text("{}\n", encoding="utf-8", newline="\n")
            exe = root / "engine.exe"
            write_teacher_executable(exe, b"formal comparison engine")
            environment = {
                "executable": {
                    "source": {
                        "sha256": build_root_table.sha256_file(exe),
                    }
                },
                "resources": [
                    {
                        "role": role,
                        "source": {
                            "sha256": build_root_table.sha256_file(exe.parent / "resources" / relative),
                        },
                    }
                    for role, relative in generate_ggs_root_teacher.RESOURCE_SPECS
                ],
            }
            report_path = root / "formal_comparison_report.json"
            state_path = root / "experiment_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema": generate_ggs_root_teacher.FORMAL_COMPARISON_STATE_SCHEMA,
                        "coverage": {"sha256": build_root_table.sha256_file(coverage)},
                        "execution_environment": environment,
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            report = {
                "schema": generate_ggs_root_teacher.FORMAL_COMPARISON_REPORT_SCHEMA,
                "experiment_state_sha256": build_root_table.sha256_file(state_path),
                "completed_pairs": 4800,
                "decision_protocol": {
                    "candidate_method": "hint_then_verify",
                    "reference_method": "time_then_verify",
                    "required_complete_pairs": 4800,
                },
                "decision_conditions": {"all_pairs_complete": True, "speed": True},
                "level_30_then_level_31_can_continue_to_larger_calculation": True,
            }
            report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
            selection = generate_ggs_root_teacher._load_formal_comparison_selection(
                report_path,
                coverage,
                exe,
                method="hint_then_verify",
                teacher_level=30,
                verify_level=31,
                threads=28,
                hash_level=29,
                min_depth=30,
                min_selectivity=74,
            )
            self.assertEqual("hint_then_verify", selection["selected_method"])
            report["completed_pairs"] = 4799
            report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ValueError, "did not select"):
                generate_ggs_root_teacher._load_formal_comparison_selection(
                    report_path,
                    coverage,
                    exe,
                    method="hint_then_verify",
                    teacher_level=30,
                    verify_level=31,
                    threads=28,
                    hash_level=29,
                    min_depth=30,
                    min_selectivity=74,
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

    def test_root_teacher_holds_an_output_lock_for_the_whole_generation(self) -> None:
        output = Path("C:/temporary/teacher_rows.txt")
        with (
            mock.patch.object(generate_ggs_root_teacher, "file_lock") as lock,
            mock.patch.object(
                generate_ggs_root_teacher,
                "_generate_teachers_unlocked",
                return_value={"completed": 1, "requested": 1},
            ) as unlocked,
        ):
            result = generate_ggs_root_teacher.generate_teachers(
                Path("coverage.json"), Path("teacher.exe"), output, 60.0, 28, 29
            )
        self.assertEqual({"completed": 1, "requested": 1}, result)
        lock.assert_called_once_with(output.with_suffix(output.suffix + ".lock"))
        unlocked.assert_called_once()

    def test_compact_only_replays_durable_updates_without_a_new_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            output = root / "teacher_rows.txt"
            state = generate_ggs_root_teacher._new_state(
                output,
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
            generate_ggs_root_teacher._append_pending_update(
                output,
                GGS_ROOT,
                "results",
                {"move": "f5", "score": -15},
            )
            with mock.patch.object(generate_ggs_root_teacher, "search_root_at_level") as search:
                result = generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    output,
                    60.0,
                    28,
                    29,
                    resume=True,
                    compact_only=True,
                )
            self.assertEqual(
                {"accepted": 1, "rejected": 0, "processed": 1, "requested": 1}, result
            )
            search.assert_not_called()
            self.assertFalse(output.with_suffix(output.suffix + ".pending.jsonl").exists())
            self.assertIn(f"{GGS_ROOT} -15 f5:-15", output.read_text(encoding="utf-8"))

    def test_rejects_missing_endgame_resource_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            exe.write_bytes(b"test teacher")
            resources = root / "resources"
            resources.mkdir()
            (resources / "eval.egev2").write_bytes(b"test evaluation")
            # The endgame move-ordering file is fixed relative to Console and
            # must therefore be present before a teacher search begins.
            with mock.patch.object(generate_ggs_root_teacher, "search_root_at_level") as search:
                with self.assertRaisesRegex(FileNotFoundError, "endgame_move_ordering resource"):
                    generate_ggs_root_teacher.generate_teachers(
                        coverage, exe, root / "teacher_rows.txt", 60.0, 28, 29
                    )
            search.assert_not_called()

    def test_compact_only_repairs_manifest_without_pending_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            output = root / "teacher_rows.txt"
            state = generate_ggs_root_teacher._new_state(
                output, coverage, exe, [GGS_ROOT], 60.0, 28, 29, 33, 74, 33,
                "hint", 33, 0, None, [],
            )
            generate_ggs_root_teacher._write_outputs(output, state)
            manifest_path = output.with_suffix(output.suffix + ".manifest.json")
            manifest_path.write_text("{broken", encoding="utf-8", newline="\n")
            result = generate_ggs_root_teacher.generate_teachers(
                coverage, exe, output, 60.0, 28, 29, resume=True, compact_only=True
            )
            self.assertEqual(
                {"accepted": 0, "rejected": 0, "processed": 0, "requested": 1}, result
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(build_root_table.sha256_file(output), manifest["output"]["sha256"])

    def test_resume_rejects_changed_saved_execution_input_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            output = root / "teacher_rows.txt"
            state = generate_ggs_root_teacher._new_state(
                output, coverage, exe, [GGS_ROOT], 60.0, 28, 29, 33, 74, 33,
                "hint", 33, 0, None, [],
            )
            generate_ggs_root_teacher._write_outputs(output, state)
            saved_evaluation = Path(
                state["calculation_provenance"]["execution_environment"]["resources"][0]["snapshot"]["path"]
            )
            saved_evaluation.write_bytes(b"changed")
            with mock.patch.object(generate_ggs_root_teacher, "search_root_at_level") as search:
                with self.assertRaisesRegex(ValueError, "saved evaluation resource"):
                    generate_ggs_root_teacher.generate_teachers(
                        coverage, exe, output, 60.0, 28, 29, resume=True
                    )
            search.assert_not_called()

    def test_execution_environment_cannot_write_outside_its_content_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            output = root / "teacher_rows.txt"
            state = generate_ggs_root_teacher._new_state(
                output, coverage, exe, [GGS_ROOT], 60.0, 28, 29, 33, 74, 33,
                "hint", 33, 0, None, [],
            )
            outside = root / "must_not_be_created.exe"
            state["calculation_provenance"]["execution_environment"]["executable"][
                "snapshot"
            ]["path"] = outside.resolve().as_posix()
            with self.assertRaisesRegex(ValueError, "unexpected saved executable path"):
                generate_ggs_root_teacher._ensure_execution_environment(output, state)
            self.assertFalse(outside.exists())

    def test_prepares_match_input_from_all_compacted_accepted_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher_rows.txt"
            teacher.write_text(
                "# ggs_root_teacher_v1\n"
                f"{GGS_ROOT} -15 f5:-15\n",
                encoding="utf-8",
                newline="\n",
            )
            write_v13_teacher_manifest(teacher)
            destination = root / "prepared"
            result = prepare_root_table_match.prepare_match_input(
                teacher, destination, minimum_processed=1, minimum_accepted=1
            )
            self.assertEqual(1, result["accepted"])
            self.assertEqual(1, result["table_entries"])
            self.assertEqual(
                {"root_discs": 14, "entries": 1},
                build_root_table.validate_root_table(
                    destination / "table" / build_root_table.ROOT_TABLE_FILENAME,
                    expected_root_discs=14,
                ),
            )
            self.assertEqual(
                f"{GGS_ROOT}\n",
                (destination / "openings" / "roots.txt").read_text(encoding="utf-8"),
            )
            prepared = json.loads(
                (destination / "prepared_match_input.json").read_text(encoding="utf-8")
            )
            self.assertEqual("prepared_root_table_match_input_v5", prepared["schema"])
            self.assertEqual(
                build_root_table.sha256_file(destination / "teacher_manifest.json"),
                prepared["teacher_manifest"]["sha256"],
            )
            self.assertEqual(1, prepared["selection"]["minimum_accepted"])
            self.assertEqual(
                build_root_table.sha256_file(destination / "teacher_calculation_script.py"),
                prepared["teacher_script_snapshot"]["sha256"],
            )
            self.assertEqual(
                {"executable", "evaluation", "endgame_move_ordering"},
                {entry["role"] for entry in prepared["teacher_execution_environment"]["files"]},
            )
            report = (destination / "README.md").read_text(encoding="utf-8")
            self.assertIn("受理局面", report)
            self.assertIn("Every accepted position", report)
            too_small = root / "too_small"
            with self.assertRaisesRegex(ValueError, "below required 2"):
                prepare_root_table_match.prepare_match_input(
                    teacher, too_small, minimum_processed=2, minimum_accepted=1
                )
            self.assertFalse(too_small.exists())
            too_few_accepted = root / "too_few_accepted"
            with self.assertRaisesRegex(ValueError, "accepted 1, below required 2"):
                prepare_root_table_match.prepare_match_input(
                    teacher,
                    too_few_accepted,
                    minimum_processed=1,
                    minimum_accepted=2,
                )
            self.assertFalse(too_few_accepted.exists())
            legacy = root / "legacy_teacher_rows.txt"
            legacy.write_text(
                "# ggs_root_teacher_v1\n" f"{GGS_ROOT} -15 f5:-15\n",
                encoding="utf-8",
                newline="\n",
            )
            legacy.with_suffix(legacy.suffix + ".manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "ggs_root_teacher_manifest_v10",
                        "output": {
                            "sha256": build_root_table.sha256_file(legacy),
                            "processed": 1,
                            "completed": 1,
                            "rejected": 0,
                        },
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            legacy_destination = root / "legacy_prepared"
            with self.assertRaisesRegex(ValueError, "unsupported teacher manifest schema"):
                prepare_root_table_match.prepare_match_input(
                    legacy, legacy_destination, minimum_processed=1, minimum_accepted=1
                )
            self.assertFalse(legacy_destination.exists())
            manifest_path = teacher.with_suffix(teacher.suffix + ".manifest.json")
            seed_mismatch = json.loads(manifest_path.read_text(encoding="utf-8"))
            seed_mismatch["random_seed"] = 621
            manifest_path.write_text(json.dumps(seed_mismatch), encoding="utf-8", newline="\n")
            seed_destination = root / "seed_mismatch"
            with self.assertRaisesRegex(ValueError, "random seed does not match"):
                prepare_root_table_match.prepare_match_input(
                    teacher, seed_destination, minimum_processed=1, minimum_accepted=1
                )
            self.assertFalse(seed_destination.exists())

    def test_audits_color_swapped_root_table_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine.exe"
            engine.write_bytes(b"same binary for both sides")
            binary_sha256 = build_root_table.sha256_file(engine)
            harness = root / "run_root_table_matches.py"
            harness.write_text("# fixed runner\n", encoding="utf-8", newline="\n")
            protocol_root = root / "protocol_worktree"
            protocol_sources: list[dict[str, object]] = []
            for relative in audit_root_table_matches.RUNNER_DEPENDENT_SOURCE_FILES:
                source = protocol_root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"// {relative}\n", encoding="utf-8", newline="\n")
                protocol_sources.append(
                    {
                        "path": source.resolve().as_posix(),
                        "kind": "file",
                        "files": 1,
                        "bytes": source.stat().st_size,
                        "sha256": build_root_table.sha256_file(source),
                        "relative_path": relative,
                    }
                )
            teacher = root / "teacher_rows.txt"
            teacher.write_text(
                "# ggs_root_teacher_v1\n"
                f"{GGS_ROOT} -15 f5:-15\n",
                encoding="utf-8",
                newline="\n",
            )
            write_v13_teacher_manifest(
                teacher,
                engine=engine,
                include_match_requirements=True,
            )
            evaluation = engine.parent / "resources" / "eval.egev2"
            endgame_move_ordering = engine.parent / "resources" / "eval_move_ordering_end.egev"
            prepared_dir = root / "prepared"
            prepare_root_table_match.prepare_match_input(
                teacher,
                prepared_dir,
                minimum_processed=1,
                minimum_accepted=1,
            )
            prepared_path = prepared_dir / "prepared_match_input.json"
            table_dir = prepared_dir / "table"
            _root_discs, table_entries = build_root_table.load_root_table_entries(
                table_dir / build_root_table.ROOT_TABLE_FILENAME,
                14,
            )
            played_board = audit_root_table_matches._expected_game_boards([GGS_ROOT])[0]
            expected_move = audit_root_table_matches._expected_table_move(
                played_board, table_entries
            )

            def complete_record(first_move: str) -> str:
                normalized = normalize_board_text(played_board)
                position = Board.from_text(normalized)
                record = ""
                first = True
                while not position.is_end():
                    legal = position.legal_moves()
                    if not legal:
                        position.pass_turn()
                        continue
                    move = first_move if first else index_to_coord(legal[0])
                    position.play(next(index for index in legal if index_to_coord(index) == move))
                    record += move
                    first = False
                return record

            record = complete_record(expected_move)
            hash_initialization_log = (
                "[ERROR] can't open hash29.eghs\n"
                "[ERROR] can't get hash. you can ignore this error\n"
                "random seed = 620\n"
                "ggs tournament build = true\n"
            )

            def replayed_game(color: str, game_id: int) -> dict[str, object]:
                replay, replay_failures = audit_root_table_matches._replay_game(
                    played_board,
                    {
                        "candidate_color": color,
                        "record": record,
                        "final_discs": None,
                        "candidate_disc_diff": None,
                    },
                )
                self.assertTrue(replay)
                self.assertTrue(replay_failures)
                remaining = {"X": 60000, "O": 60000}
                clock_records: list[dict[str, object]] = []
                for move_number, (side, move) in enumerate(
                    zip(replay["move_sides"], (record[offset:offset + 2] for offset in range(0, len(record), 2))),
                    start=1,
                ):
                    before = dict(remaining)
                    after = dict(before)
                    after[side] -= 1
                    clock_records.append(
                        {
                            "move_number": move_number,
                            "side": side,
                            "role": "candidate" if color == side else "baseline",
                            "move": move,
                            "before_remaining_msec": before,
                            "go_wall_msec": 1,
                            "after_remaining_msec": after,
                        }
                    )
                    remaining = after
                return {
                    "game": game_id,
                    "candidate_color": color,
                    "process_launch_order": (
                        ["candidate", "baseline"]
                        if game_id == 0
                        else ["baseline", "candidate"]
                    ),
                    "candidate_disc_diff": replay["candidate_difference"],
                    "final_discs": replay["final_discs"],
                    "record": record,
                    "external_clock": {
                        "initial_remaining_msec": {"X": 60000, "O": 60000},
                        "records": clock_records,
                        "final_remaining_msec": remaining,
                    },
                }

            def clock_log(game: dict[str, object]) -> str:
                clock = game["external_clock"]
                self.assertIsInstance(clock, dict)
                lines: list[str] = []
                for entry in clock["records"]:
                    self.assertIsInstance(entry, dict)
                    before = entry["before_remaining_msec"]
                    self.assertIsInstance(before, dict)
                    lines.append(f"received cmd: settimems X {before['X']}\n")
                    lines.append(f"received cmd: settimems O {before['O']}\n")
                return "".join(lines)

            first_game = replayed_game("X", 0)
            second_game = replayed_game("O", 1)

            table_log_first = root / "table_first.log"
            table_log_first.write_text(
                hash_initialization_log
                + clock_log(first_game)
                + "contest root table loaded 1 roots\n"
                f"contest root table selected {expected_move} value -15 roots 1 "
                f"{canonicalize_board_key(played_board)[0]}\n"
                f"level Book depth - {expected_move} -15 elapsed 000:00:00.000 nodes 0 nps 0\n",
                encoding="utf-8",
                newline="\n",
            )
            table_log_second = root / "table_second.log"
            table_log_second.write_text(
                hash_initialization_log + clock_log(second_game) + "contest root table loaded 1 roots\n",
                encoding="utf-8",
                newline="\n",
            )
            no_book_first = root / "no_book_first.log"
            no_book_first.write_text(
                hash_initialization_log + clock_log(first_game),
                encoding="utf-8",
                newline="\n",
            )
            no_book_second = root / "no_book_second.log"
            no_book_second.write_text(
                hash_initialization_log + clock_log(second_game),
                encoding="utf-8",
                newline="\n",
            )

            def engine_audit(path: Path) -> dict[str, object]:
                content = path.read_text(encoding="utf-8")
                settimems = [
                    {"color": match.group("color"), "remaining_msec": int(match.group("remaining_msec"))}
                    for match in audit_root_table_matches.SETTIMEMS_COMMAND_RE.finditer(content)
                ]
                return {
                    "log": path.resolve().as_posix(),
                    "log_sha256": build_root_table.sha256_file(path),
                    "exit_code": 0,
                    "timeout_suspected": False,
                    "zero_clock_seen": False,
                    "harness_clock_overrun_msec": 0,
                    "expected_hash_error_lines": 2,
                    "unexpected_error_lines": [],
                    "random_seed_log_lines": 1,
                    "ggs_tournament_build_log_lines": 1,
                    "settimems_commands": settimems,
                }

            results = root / "matches.jsonl"
            first_game["engine_audit"] = {
                "candidate": engine_audit(table_log_first),
                "baseline": engine_audit(no_book_first),
            }
            second_game["engine_audit"] = {
                "candidate": engine_audit(table_log_second),
                "baseline": engine_audit(no_book_second),
            }
            margin = int(first_game["candidate_disc_diff"]) + int(second_game["candidate_disc_diff"])
            row = {
                "match": 0,
                "board": played_board,
                "margin": margin,
                "result": "W" if margin > 0 else "L" if margin < 0 else "D",
                "games": [first_game, second_game],
            }
            results.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
            ordered_positions = results.with_suffix(results.suffix + ".openings.txt")
            ordered_positions.write_text(played_board + "\n", encoding="utf-8", newline="\n")
            ordered_positions_snapshot = {
                "path": ordered_positions.resolve().as_posix(),
                "kind": "file",
                "files": 1,
                "bytes": ordered_positions.stat().st_size,
                "sha256": build_root_table.sha256_file(ordered_positions),
            }
            common_command = [
                str(engine.resolve()),
                "-quiet",
                "-noise",
                "-nobook",
                "-t",
                "8",
                "-hash",
                "29",
                "-seed",
                "620",
                "-eval",
                str(evaluation.resolve()),
                "-time",
                "60",
            ]
            selected_canonical = sorted(
                audit_root_table_matches._d4_canonical_board(board)
                for board in [played_board]
            )
            pool_canonical = sorted(
                audit_root_table_matches._d4_canonical_board(board)
                for board in [GGS_ROOT]
            )
            run_spec = {
                "schema_version": audit_root_table_matches.METADATA_SCHEMA_VERSION,
                "git": {
                    "repository_root": protocol_root.resolve().as_posix(),
                    "commit": "a" * 40,
                    "tracked_worktree_dirty": False,
                    "tracked_status_sha256": hashlib.sha256(b"").hexdigest(),
                },
                "runner_protocol": {
                    "schema": audit_root_table_matches.MATCH_PROTOCOL_SCHEMA,
                    "clean_tracked_worktree_required": True,
                    "source_files": protocol_sources,
                    "external_clock": {
                        "command": "settimems",
                        "initial_remaining_msec": 60000,
                        "measurement": "ceil(monotonic go wall time in milliseconds)",
                        "only_go_commands_decrement_time": True,
                    },
                    "noise_log_lines": {
                        "random_seed": "random seed = 620",
                        "ggs_tournament_build": "ggs tournament build = true",
                    },
                    "process_launch_order": "candidate-first when (match id + game id) is even",
                },
                "parsed_args": {
                    "time": 60,
                    "threads": 8,
                    "hash": 29,
                    "matches": 1,
                    "workers": 1,
                    "seed": 624,
                    "engine_random_seed": 620,
                    "random_symmetry": True,
                    "level_31_verification_required": False,
                    "candidate": str(engine.resolve()),
                    "baseline": str(engine.resolve()),
                    "contestbook": None,
                    "candidate_contestbook": table_dir.resolve().as_posix(),
                    "baseline_contestbook": None,
                    "candidate_extra": "",
                    "baseline_extra": "",
                    "external_clock_control": True,
                },
                "artifacts": {
                    "candidate_binary": {
                        "path": engine.resolve().as_posix(),
                        "sha256": binary_sha256,
                    },
                    "baseline_binary": {
                        "path": engine.resolve().as_posix(),
                        "sha256": binary_sha256,
                    },
                    "evaluation": {
                        "path": evaluation.resolve().as_posix(),
                        "sha256": build_root_table.sha256_file(evaluation),
                    },
                    "endgame_move_ordering": {
                        "path": endgame_move_ordering.resolve().as_posix(),
                        "sha256": build_root_table.sha256_file(endgame_move_ordering),
                    },
                    "ordered_starting_positions": ordered_positions_snapshot,
                },
                "engine_commands": {
                    "candidate": [*common_command, "-contestbook", str(table_dir.resolve())],
                    "baseline": common_command,
                },
                "harness": {
                    "path": harness.resolve().as_posix(),
                    "sha256": build_root_table.sha256_file(harness),
                },
                "openings": {
                    "raw_count": 1,
                    "d4_unique_count": 1,
                    "d4_duplicates_dropped": 0,
                    "canonical_pool_sha256": audit_root_table_matches._sha256_lines(pool_canonical),
                    "selected_count": 1,
                    "ordered_sha256": audit_root_table_matches._sha256_lines([played_board]),
                    "ordered_file": ordered_positions_snapshot,
                    "d4_canonical_set_sha256": audit_root_table_matches._sha256_lines(selected_canonical),
                },
            }
            metadata = {
                "run_spec": run_spec,
                "run_spec_sha256": audit_root_table_matches._canonical_json_sha256(run_spec),
            }
            metadata_path = root / "matches.meta.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8", newline="\n")
            report = root / "report.md"
            payload = audit_root_table_matches.audit_match_results(
                results,
                prepared_path,
                metadata_path,
                report,
                bootstrap_seed=620,
                bootstrap_repetitions=100,
                minimum_processed=1,
                minimum_accepted=1,
            )
            self.assertTrue(payload["valid"])
            self.assertFalse(payload["eligible_for_adoption"])
            self.assertTrue(payload["teacher_calculation"]["ordinary_book_disabled"])
            self.assertTrue(payload["teacher_calculation"]["contest_book_disabled"])
            text = report.read_text(encoding="utf-8")
            self.assertIn("表を使う側の勝ち", text)
            self.assertIn("Table-using side W/D/L", text)
            self.assertIn("Ordinary book during teacher calculation", text)

            altered_clock_row = json.loads(json.dumps(row))
            altered_clock_row["games"][0]["external_clock"]["records"][0][
                "after_remaining_msec"
            ]["X"] += 1
            results.write_text(
                json.dumps(altered_clock_row) + "\n", encoding="utf-8", newline="\n"
            )
            altered_clock = audit_root_table_matches.audit_match_results(
                results,
                prepared_path,
                metadata_path,
                root / "altered_clock.md",
                bootstrap_seed=620,
                bootstrap_repetitions=100,
                minimum_processed=1,
                minimum_accepted=1,
            )
            self.assertFalse(altered_clock["valid"])
            self.assertTrue(
                any("does not subtract only its go time" in failure for failure in altered_clock["failures"])
            )
            results.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")

            original_prepared = prepared_path.read_bytes()
            strict_selection = json.loads(original_prepared.decode("utf-8"))
            strict_selection["selection"]["level_31_verification_required"] = True
            prepared_path.write_text(
                json.dumps(strict_selection), encoding="utf-8", newline="\n"
            )
            altered_level_verification = audit_root_table_matches.audit_match_results(
                results,
                prepared_path,
                metadata_path,
                root / "altered_level_verification.md",
                bootstrap_seed=620,
                bootstrap_repetitions=100,
                minimum_processed=1,
                minimum_accepted=1,
            )
            self.assertFalse(altered_level_verification["valid"])
            self.assertTrue(
                any(
                    "valid level-31 verification" in failure
                    for failure in altered_level_verification["failures"]
                )
            )
            prepared_path.write_bytes(original_prepared)

            frozen_manifest_path = prepared_dir / "teacher_manifest.json"
            original_manifest = frozen_manifest_path.read_bytes()
            prepared = json.loads(original_prepared.decode("utf-8"))
            frozen_manifest = json.loads(original_manifest.decode("utf-8"))
            frozen_manifest["calculation_provenance"]["book_configuration"]["contest_book"][
                "disabled"
            ] = False
            frozen_manifest_path.write_text(
                json.dumps(frozen_manifest), encoding="utf-8", newline="\n"
            )
            prepared["teacher_manifest"]["sha256"] = build_root_table.sha256_file(
                frozen_manifest_path
            )
            prepared_path.write_text(json.dumps(prepared), encoding="utf-8", newline="\n")
            altered_teacher_calculation = audit_root_table_matches.audit_match_results(
                results,
                prepared_path,
                metadata_path,
                root / "altered_teacher_calculation.md",
                bootstrap_seed=620,
                bootstrap_repetitions=100,
                minimum_processed=1,
                minimum_accepted=1,
            )
            self.assertFalse(altered_teacher_calculation["valid"])
            self.assertTrue(
                any(
                    "does not disable both books" in failure
                    for failure in altered_teacher_calculation["failures"]
                )
            )
            frozen_manifest_path.write_bytes(original_manifest)
            prepared_path.write_bytes(original_prepared)

            altered_spec = json.loads(json.dumps(run_spec))
            altered_spec["parsed_args"]["candidate_extra"] = "-t 1"
            metadata_path.write_text(
                json.dumps(
                    {
                        "run_spec": altered_spec,
                        "run_spec_sha256": audit_root_table_matches._canonical_json_sha256(
                            altered_spec
                        ),
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            altered = audit_root_table_matches.audit_match_results(
                results,
                prepared_path,
                metadata_path,
                root / "altered_extra.md",
                bootstrap_seed=620,
                bootstrap_repetitions=100,
                minimum_processed=1,
                minimum_accepted=1,
            )
            self.assertFalse(altered["valid"])
            self.assertTrue(
                any("nonempty candidate_extra" in failure for failure in altered["failures"])
            )

            metadata_path.write_text(json.dumps(metadata), encoding="utf-8", newline="\n")
            other_move = next(
                index_to_coord(index)
                for index in Board.from_text(played_board).legal_moves()
                if index_to_coord(index) != expected_move
            )
            table_log_first.write_text(
                hash_initialization_log
                + clock_log(first_game)
                + "contest root table loaded 1 roots\n"
                f"contest root table selected {other_move} value -15 roots 1 "
                f"{canonicalize_board_key(played_board)[0]}\n"
                f"level Book depth - {other_move} -15 elapsed 000:00:00.000 nodes 0 nps 0\n",
                encoding="utf-8",
                newline="\n",
            )
            row["games"][0]["engine_audit"]["candidate"]["log_sha256"] = (
                build_root_table.sha256_file(table_log_first)
            )
            results.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
            altered_selection = audit_root_table_matches.audit_match_results(
                results,
                prepared_path,
                metadata_path,
                root / "altered_selection.md",
                bootstrap_seed=620,
                bootstrap_repetitions=100,
                minimum_processed=1,
                minimum_accepted=1,
            )
            self.assertFalse(altered_selection["valid"])
            self.assertTrue(
                any("not stored in the temporary table" in failure for failure in altered_selection["failures"])
            )

    def test_match_bootstrap_sorts_score_and_margin_independently(self) -> None:
        rows = [
            {"result": "W", "margin": -8},
            {"result": "L", "margin": 8},
            {"result": "D", "margin": 0},
        ]
        repetitions = 100
        seed = 621
        intervals = audit_root_table_matches._bootstrap_intervals(rows, seed, repetitions)
        generator = random.Random(seed)
        score_samples = []
        margin_samples = []
        points = [(1.0, -8.0), (0.0, 8.0), (0.5, 0.0)]
        for _ in range(repetitions):
            selected = [points[generator.randrange(len(points))] for _ in points]
            score_samples.append(sum(point[0] for point in selected) / len(selected))
            margin_samples.append(sum(point[1] for point in selected) / len(selected))
        score_samples.sort()
        margin_samples.sort()
        self.assertEqual((score_samples[2], score_samples[96]), intervals["score"])
        self.assertEqual((margin_samples[2], margin_samples[96]), intervals["margin"])

    def test_benchmarks_two_root_teacher_methods_on_fixed_population_positions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe = root / "teacher.exe"
            exe.write_bytes(b"teacher executable")
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)),
                encoding="utf-8",
                newline="\n",
            )

            def fake_generate(coverage: Path, _exe: Path, output: Path, *args, **_kwargs) -> dict[str, int]:
                method = args[6]
                board = json.loads(coverage.read_text(encoding="utf-8"))["roots"][0]["canonical_board"]
                move = "f5" if method == "time_then_verify" else "f5"
                output.write_text(
                    "# ggs_root_teacher_v1\n" f"{board} -15 {move}:-15\n",
                    encoding="utf-8",
                    newline="\n",
                )
                output.with_suffix(output.suffix + ".manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": "ggs_root_teacher_manifest_v10",
                            "output": {
                                "sha256": build_root_table.sha256_file(output),
                                "completed": 1,
                                "rejected": 0,
                                "processed": 1,
                            },
                            "results": {board: {"move": move}},
                            "rejections": {},
                        }
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                return {"completed": 1, "requested": 1}

            with mock.patch.object(
                benchmark_root_teacher_methods,
                "generate_teachers",
                side_effect=fake_generate,
            ):
                payload = benchmark_root_teacher_methods.compare_methods(
                    coverage,
                    [],
                    exe,
                    root / "benchmark",
                    1,
                    620,
                    621,
                    622,
                )
            self.assertEqual(1, payload["comparison"]["same_accepted_move"])
            self.assertEqual(0, payload["comparison"]["different_accepted_move"])
            self.assertEqual(1, payload["population"]["count"])
            state = json.loads(
                (root / "benchmark" / "experiment_state.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (root / "benchmark" / "comparison_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                benchmark_root_teacher_methods.EXPERIMENT_STATE_SCHEMA,
                state["schema"],
            )
            self.assertEqual(1, progress["completed_positions"])
            for method_name in (
                benchmark_root_teacher_methods.TIME_METHOD,
                benchmark_root_teacher_methods.LEVEL_METHOD,
            ):
                self.assertGreater(progress["records"][0][method_name]["wall_seconds"], 0.0)
                self.assertEqual(
                    64,
                    len(progress["records"][0][method_name]["output_sha256"]),
                )
            report = (root / "benchmark" / "README.md").read_text(encoding="utf-8")
            self.assertIn("60秒の持ち時間を与える探索", report)
            self.assertIn("level-30/level-31 check", report)
            audit = audit_root_teacher_method_benchmark.audit_benchmark(
                root / "benchmark", root / "benchmark_audit.md"
            )
            self.assertTrue(audit["valid"])
            self.assertEqual(1, audit["counts"]["completed_positions"])
            self.assertEqual(2, audit["counts"]["saved_method_results"])
            self.assertEqual(1, audit["timing_seconds"]["same_session_pairs"])
            self.assertGreater(
                audit["timing_seconds"]["level_30_then_level_31_to_time_managed_search_ratio"],
                0.0,
            )
            self.assertTrue((root / "benchmark_audit.json").is_file())
            with mock.patch.object(
                benchmark_root_teacher_methods,
                "generate_teachers",
                side_effect=AssertionError("completed calculation was unexpectedly repeated"),
            ):
                resumed = benchmark_root_teacher_methods.compare_methods(
                    coverage,
                    [],
                    exe,
                    root / "benchmark",
                    1,
                    620,
                    621,
                    622,
                    resume=True,
                )
            self.assertEqual(payload["positions_detail"], resumed["positions_detail"])
            with self.assertRaisesRegex(ValueError, "resume conditions do not exactly match"):
                benchmark_root_teacher_methods.compare_methods(
                    coverage,
                    [],
                    exe,
                    root / "benchmark",
                    1,
                    623,
                    621,
                    622,
                    resume=True,
                )

    def test_benchmark_resumes_after_archiving_uncheckpointed_method_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe = root / "teacher.exe"
            exe.write_bytes(b"teacher executable")
            second_board = transform_board_text(GGS_ROOT, 1)
            self.assertNotEqual(GGS_ROOT, second_board)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(
                    {
                        "schema": collect_ggs_roots.REPORT_SCHEMA,
                        "root_discs": 14,
                        "roots": [
                            {"canonical_board": GGS_ROOT, "deep_book": False, "root_table": False},
                            {
                                "canonical_board": second_board,
                                "deep_book": False,
                                "root_table": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )

            def write_generated(coverage_path: Path, output: Path, args: tuple[object, ...]) -> tuple[str, str]:
                method = str(args[6])
                board = json.loads(coverage_path.read_text(encoding="utf-8"))["roots"][0][
                    "canonical_board"
                ]
                move = "f5"
                output.write_text(
                    "# ggs_root_teacher_v1\n" f"{board} -15 {move}:-15\n",
                    encoding="utf-8",
                    newline="\n",
                )
                output.with_suffix(output.suffix + ".manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": "ggs_root_teacher_manifest_v10",
                            "output": {
                                "sha256": build_root_table.sha256_file(output),
                                "completed": 1,
                                "rejected": 0,
                                "processed": 1,
                            },
                            "results": {board: {"move": move}},
                            "rejections": {},
                        }
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                return board, method

            initial_calls: list[tuple[str, str]] = []

            def interrupted_generate(coverage_path: Path, _exe: Path, output: Path, *args, **_kwargs):
                initial_calls.append(write_generated(coverage_path, output, args))
                if len(initial_calls) == 3:
                    raise RuntimeError("simulated interruption after output creation")
                return {"completed": 1, "requested": 1}

            output_dir = root / "benchmark"
            with mock.patch.object(
                benchmark_root_teacher_methods,
                "generate_teachers",
                side_effect=interrupted_generate,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    benchmark_root_teacher_methods.compare_methods(
                        coverage, [], exe, output_dir, 2, 620, 621, 622
                    )
            before = json.loads(
                (output_dir / "comparison_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, before["completed_positions"])
            first_before = next(record for record in before["records"] if record["index"] == 1)
            self.assertIn("in_progress", next(record for record in before["records"] if record["index"] == 2))

            resumed_calls: list[tuple[str, str]] = []

            def resume_generate(coverage_path: Path, _exe: Path, output: Path, *args, **_kwargs):
                resumed_calls.append(write_generated(coverage_path, output, args))
                return {"completed": 1, "requested": 1}

            with mock.patch.object(
                benchmark_root_teacher_methods,
                "generate_teachers",
                side_effect=resume_generate,
            ):
                payload = benchmark_root_teacher_methods.compare_methods(
                    coverage, [], exe, output_dir, 2, 620, 621, 622, resume=True
                )
            self.assertEqual(2, len(resumed_calls))
            self.assertTrue(all(call[0] == initial_calls[2][0] for call in resumed_calls))
            archived = (output_dir / "interrupted_attempts" / "archived_attempts.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("calculation output was not present", archived)
            after = json.loads(
                (output_dir / "comparison_progress.json").read_text(encoding="utf-8")
            )
            first_after = next(record for record in after["records"] if record["index"] == 1)
            self.assertEqual(
                first_before[benchmark_root_teacher_methods.TIME_METHOD]["wall_seconds"],
                first_after[benchmark_root_teacher_methods.TIME_METHOD]["wall_seconds"],
            )
            self.assertEqual(2, len(payload["positions_detail"]))

            completed_output = Path(
                first_after[benchmark_root_teacher_methods.TIME_METHOD]["output_path"]
            )
            completed_output.write_text("tampered\n", encoding="utf-8", newline="\n")
            tampered_audit = audit_root_teacher_method_benchmark.audit_benchmark(
                output_dir, root / "tampered_benchmark_audit.md"
            )
            self.assertFalse(tampered_audit["valid"])
            self.assertTrue(
                any("generated output does not match" in failure for failure in tampered_audit["failures"])
            )
            with mock.patch.object(
                benchmark_root_teacher_methods,
                "generate_teachers",
                side_effect=AssertionError("tampered output must not be recalculated silently"),
            ):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    benchmark_root_teacher_methods.compare_methods(
                        coverage, [], exe, output_dir, 2, 620, 621, 622, resume=True
                    )

    def test_forced_move_analysis_reads_played_score_at_requested_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe = root / "teacher.exe"
            exe.write_bytes(b"teacher executable")
            analyzed = (
                "|          Ply|       Player|       Played|        Depth|        Score|\n"
                "|           11|        Black|           f5|       33@74%|           -6|\n"
            )
            completed = SimpleNamespace(returncode=0, stdout=analyzed, stderr="")
            with mock.patch.object(
                benchmark_root_teacher_methods.subprocess,
                "run",
                return_value=completed,
            ) as run:
                result = benchmark_root_teacher_methods.evaluate_forced_move_at_level(
                    exe,
                    GGS_ROOT,
                    "f5",
                    33,
                    root / "forced_move.log",
                )
            self.assertEqual("f5", result["move"])
            self.assertEqual(-6, result["score"])
            self.assertEqual("33@74%", result["depth"])
            self.assertTrue((root / "forced_move.log").is_file())
            command = run.call_args.args[0]
            self.assertIn("-l", command)
            self.assertIn("33", command)
            self.assertIn("-nobook", command)
            self.assertIn("-nocontestbook", command)

    def test_search_root_accepts_console_result_on_stderr(self) -> None:
        table = (
            "|             27|         27@74%|             f5|            -15|  000:00:02.786|      239460993|       85951540|\n"
        )
        completed = SimpleNamespace(returncode=0, stdout="", stderr=table)
        with mock.patch.object(
            generate_ggs_root_teacher.subprocess, "run", return_value=completed
        ) as search:
            result = generate_ggs_root_teacher.search_root(
                Path("C:/teacher.exe"), GGS_ROOT, 60.0, 28, 29
            )
        self.assertEqual("f5", result["move"])
        self.assertEqual(-15, result["score"])
        self.assertEqual(
            generate_ggs_root_teacher._build_search_command(
                "time_limited_search", Path("C:/teacher.exe"), time_seconds=60.0, threads=28, hash_level=29
            ),
            search.call_args.args[0],
        )
        self.assertIn("-nobook", search.call_args.args[0])
        self.assertIn("-nocontestbook", search.call_args.args[0])
        self.assertEqual("620", search.call_args.args[0][search.call_args.args[0].index("-seed") + 1])

    def test_fixed_level_search_uses_the_recorded_command_builder(self) -> None:
        table = (
            "|             31|         31@74%|             f5|            -15|  000:00:02.786|      239460993|       85951540|\n"
        )
        completed = SimpleNamespace(returncode=0, stdout=table, stderr="")
        with mock.patch.object(
            generate_ggs_root_teacher.subprocess, "run", return_value=completed
        ) as search:
            generate_ggs_root_teacher.search_root_at_level(
                Path("C:/teacher.exe"), GGS_ROOT, 31, 28, 29
            )
        self.assertEqual(
            generate_ggs_root_teacher._build_search_command(
                "fixed_level_search", Path("C:/teacher.exe"), level=31, threads=28, hash_level=29
            ),
            search.call_args.args[0],
        )
        self.assertIn("-nobook", search.call_args.args[0])
        self.assertIn("-nocontestbook", search.call_args.args[0])
        self.assertEqual("620", search.call_args.args[0][search.call_args.args[0].index("-seed") + 1])

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
            write_teacher_executable(exe, b"test teacher")
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
            write_teacher_executable(exe, b"test teacher")
            output = root / "teacher_rows.txt"
            expected = generate_ggs_root_teacher._new_state(
                output,
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
            write_teacher_executable(exe, b"test teacher")
            output = root / "teacher_rows.txt"
            state = generate_ggs_root_teacher._new_state(
                output,
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
            self.assertIn("採用局面数（品質検査を通過し、最初の手を記録できた局面）", text)
            self.assertIn("Accepted positions", text)

    def test_generates_and_resumes_only_with_identical_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
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
                    {"accepted": 1, "rejected": 0, "processed": 1, "requested": 1},
                    generate_ggs_root_teacher.generate_teachers(
                        coverage, exe, output, 60.0, 28, 29
                    ),
                )
                search.assert_called_once_with(
                    saved_teacher_execution_executable(output), GGS_ROOT, 33, 28, 29, 620
                )
            teacher_text = output.read_text(encoding="utf-8")
            self.assertIn(f"{GGS_ROOT} -15 f5:-15", teacher_text)
            self.assertIn("# ordinary_book_disabled true", teacher_text)
            self.assertIn("# contest_book_disabled true", teacher_text)
            state_path = output.with_suffix(output.suffix + ".state.json")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                output.with_suffix(output.suffix + ".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(generate_ggs_root_teacher.TEACHER_SCHEMA, state["schema"])
            self.assertEqual(620, state["random_seed"])
            self.assertIn("# random_seed 620", teacher_text)
            self.assertEqual(
                generate_ggs_root_teacher.TEACHER_MANIFEST_SCHEMA,
                manifest["schema"],
            )
            provenance = state["calculation_provenance"]
            self.assertEqual(620, provenance["random_seed"])
            self.assertEqual(provenance, manifest["calculation_provenance"])
            self.assertEqual(
                {
                    "ordinary_book": {"disabled": True, "command_line_option": "-nobook"},
                    "contest_book": {"disabled": True, "command_line_option": "-nocontestbook"},
                },
                provenance["book_configuration"],
            )
            script_snapshot = Path(provenance["teacher_script_snapshot"]["path"])
            self.assertTrue(script_snapshot.is_file())
            self.assertEqual(
                provenance["teacher_script"]["sha256"],
                build_root_table.sha256_file(script_snapshot),
            )
            with mock.patch.object(generate_ggs_root_teacher, "search_root") as search:
                self.assertEqual(
                    {"accepted": 1, "rejected": 0, "processed": 1, "requested": 1},
                    generate_ggs_root_teacher.generate_teachers(
                        coverage, exe, output, 60.0, 28, 29, resume=True
                    ),
                )
                search.assert_not_called()
            with self.assertRaisesRegex(ValueError, "resume mismatch for threads"):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, output, 60.0, 27, 29, resume=True
                )
            with self.assertRaisesRegex(ValueError, "resume mismatch for random_seed"):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, output, 60.0, 28, 29, resume=True, random_seed=621
                )
            state["calculation_provenance"]["teacher_script"]["sha256"] = "0" * 64
            state_path.write_text(json.dumps(state), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ValueError, "resume mismatch for calculation_provenance"):
                generate_ggs_root_teacher.generate_teachers(
                    coverage, exe, output, 60.0, 28, 29, resume=True
                )

    def test_uses_level_33_fallback_when_time_search_is_shallow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
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
            level_search.assert_called_once_with(
                saved_teacher_execution_executable(output), GGS_ROOT, 33, 28, 29, 620
            )
            manifest = json.loads(
                output.with_suffix(output.suffix + ".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("hint_level_33", manifest["results"][GGS_ROOT]["method"])

    def test_hint_then_verify_uses_two_levels_without_time_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            teacher = {
                "move": "f5", "score": -15, "level": "30", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            verification = {
                "move": "f5", "score": -14, "level": "31", "depth": "31@74%",
                "time": "000:00:08.000", "nodes": 2, "nps": 1,
            }
            with (
                mock.patch.object(generate_ggs_root_teacher, "search_root") as time_search,
                mock.patch.object(
                    generate_ggs_root_teacher,
                    "search_root_at_level",
                    side_effect=[teacher, verification],
                ) as level_search,
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    root / "teacher_rows.txt",
                    60.0,
                    28,
                    29,
                    min_depth=30,
                    method="hint_then_verify",
                    teacher_level=30,
                    verify_level=31,
                )
            time_search.assert_not_called()
            self.assertEqual(
                [
                    mock.call(saved_teacher_execution_executable(root / "teacher_rows.txt"), GGS_ROOT, 30, 28, 29, 620),
                    mock.call(saved_teacher_execution_executable(root / "teacher_rows.txt"), GGS_ROOT, 31, 28, 29, 620),
                ],
                level_search.call_args_list,
            )
            manifest = json.loads(
                (root / "teacher_rows.txt.manifest.json").read_text(encoding="utf-8")
            )
            saved = manifest["results"][GGS_ROOT]
            self.assertEqual("hint_level_30_verified_hint_level_31", saved["method"])
            self.assertEqual("level_31_exact", saved["verification_mode"])

    def test_hint_then_verify_uses_repeated_deeper_search_on_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            teacher = {
                "move": "f5", "score": -15, "level": "30", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            verification = {
                "move": "d3", "score": -14, "level": "31", "depth": "31@74%",
                "time": "000:00:08.000", "nodes": 2, "nps": 1,
            }
            verification_repeat = {
                "move": "d3", "score": -13, "level": "31", "depth": "31@74%",
                "time": "000:00:09.000", "nodes": 3, "nps": 1,
            }
            with mock.patch.object(
                generate_ggs_root_teacher,
                "search_root_at_level",
                side_effect=[teacher, verification, verification_repeat],
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    root / "teacher_rows.txt",
                    60.0,
                    28,
                    29,
                    min_depth=30,
                    method="hint_then_verify",
                    teacher_level=30,
                    verify_level=31,
                )
            manifest = json.loads(
                (root / "teacher_rows.txt.manifest.json").read_text(encoding="utf-8")
            )
            saved = manifest["results"][GGS_ROOT]
            self.assertEqual("d3", saved["move"])
            self.assertEqual("f5", saved["teacher"]["move"])
            self.assertEqual("d3", saved["verification"]["move"])
            self.assertEqual(
                "level_31_repeated_after_disagreement", saved["verification_mode"]
            )

    def test_hint_then_verify_rejects_conflicting_repeated_deeper_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            teacher = {
                "move": "f5", "score": -15, "level": "30", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            first = {
                "move": "d3", "score": -14, "level": "31", "depth": "31@74%",
                "time": "000:00:08.000", "nodes": 2, "nps": 1,
            }
            second = {
                "move": "b4", "score": -13, "level": "31", "depth": "31@74%",
                "time": "000:00:09.000", "nodes": 3, "nps": 1,
            }
            with mock.patch.object(
                generate_ggs_root_teacher,
                "search_root_at_level",
                side_effect=[teacher, first, second],
            ):
                result = generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    root / "teacher_rows.txt",
                    60.0,
                    28,
                    29,
                    min_depth=30,
                    method="hint_then_verify",
                    teacher_level=30,
                    verify_level=31,
                )
            self.assertEqual(
                {"accepted": 0, "rejected": 1, "processed": 1, "requested": 1}, result
            )
            manifest = json.loads(
                (root / "teacher_rows.txt.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, manifest["output"]["rejected"])
            self.assertIn("repeated level-31", manifest["rejections"][GGS_ROOT]["reason"])

    def test_time_teacher_rejects_shallow_verification_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            primary = {
                "move": "f5", "score": -15, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            shallow_verification = {
                "move": "f5", "score": -14, "level": "31", "depth": "30@74%",
                "time": "000:00:02.000", "nodes": 2, "nps": 1,
            }
            with (
                mock.patch.object(generate_ggs_root_teacher, "search_root", return_value=primary),
                mock.patch.object(
                    generate_ggs_root_teacher,
                    "search_root_at_level",
                    return_value=shallow_verification,
                ),
                self.assertRaisesRegex(ValueError, "below 31@74%"),
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    root / "teacher_rows.txt",
                    60.0,
                    28,
                    29,
                    min_depth=30,
                    method="time_then_verify",
                    verify_level=31,
                )

    def test_time_teacher_requires_matching_level_27_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            output = root / "teacher_rows.txt"
            primary = {
                "move": "f5", "score": -15, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            verification = {
                "move": "f5", "score": -14, "level": "27", "depth": "30@74%",
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
            write_teacher_executable(exe, b"test teacher")
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
                "move": "f5", "score": -13, "level": "27", "depth": "30@74%",
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
                    mock.call(saved_teacher_execution_executable(output), GGS_ROOT, 30, 28, 29, 620),
                    mock.call(saved_teacher_execution_executable(output), GGS_ROOT, 27, 28, 29, 620),
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
            write_teacher_executable(exe, b"test teacher")
            primary = {
                "move": "b4", "score": -15, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            first_tied = {
                "move": "f5", "score": -15, "level": "27", "depth": "30@74%",
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
            write_teacher_executable(exe, b"test teacher")
            primary = {
                "move": "f5", "score": -12, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            verification = {
                "move": "b4", "score": -12, "level": "27", "depth": "30@74%",
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
            write_teacher_executable(exe, b"test teacher")
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
            write_teacher_executable(exe, b"test teacher")
            primary = {
                "move": "f5", "score": -15, "level": "-", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            mismatch = {
                "move": "d3", "score": -15, "level": "27", "depth": "30@74%",
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
            self.assertEqual(
                {"accepted": 0, "rejected": 1, "processed": 1, "requested": 1}, result
            )
            manifest = json.loads(
                (root / "teacher_rows.txt.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, manifest["output"]["rejected"])
            self.assertEqual(
                "level-30 tiebreak d3 does not match level-31 tiebreak b4",
                manifest["rejections"][GGS_ROOT]["reason"],
            )

    def test_formal_preparation_requires_the_recorded_level_31_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(self.coverage_report(GGS_ROOT)), encoding="utf-8", newline="\n"
            )
            exe = root / "teacher.exe"
            write_teacher_executable(exe, b"test teacher")
            output = root / "teacher_rows.txt"
            teacher = {
                "move": "f5", "score": -15, "level": "30", "depth": "30@74%",
                "time": "000:00:07.000", "nodes": 1, "nps": 1,
            }
            verification = {
                "move": "f5", "score": -14, "level": "31", "depth": "31@74%",
                "time": "000:00:08.000", "nodes": 2, "nps": 1,
            }
            with mock.patch.object(
                generate_ggs_root_teacher,
                "search_root_at_level",
                side_effect=[teacher, verification],
            ):
                generate_ggs_root_teacher.generate_teachers(
                    coverage,
                    exe,
                    output,
                    60.0,
                    28,
                    29,
                    min_depth=30,
                    min_selectivity=74,
                    fallback_level=30,
                    method="hint_then_verify",
                    teacher_level=30,
                    verify_level=31,
                )
            prepared = root / "prepared"
            prepare_root_table_match.prepare_match_input(
                output,
                prepared,
                minimum_processed=1,
                minimum_accepted=1,
                require_level_31_verification=True,
            )
            prepared_payload = json.loads(
                (prepared / "prepared_match_input.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                prepared_payload["selection"]["level_31_verification_required"]
            )

            manifest_path = output.with_suffix(output.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["results"][GGS_ROOT]["verification"]["level"] = "30"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ValueError, "expected 31"):
                prepare_root_table_match.prepare_match_input(
                    output,
                    root / "mismatched_level",
                    minimum_processed=1,
                    minimum_accepted=1,
                    require_level_31_verification=True,
                )

    def test_formal_validation_rejects_time_deep_tiebreak_without_the_initial_disagreement(self) -> None:
        primary = {
            "move": "f5", "score": -15, "level": "-", "depth": "30@74%",
            "time": "000:00:07.000", "nodes": 1, "nps": 1,
            "method": "time_verified_hint_level_31",
        }
        verification = {
            "move": "f5", "score": -14, "level": "31", "depth": "31@74%",
            "time": "000:00:08.000", "nodes": 2, "nps": 1,
        }
        tiebreak = {
            "move": "c7", "score": -13, "level": "30", "depth": "30@74%",
            "time": "000:00:09.000", "nodes": 3, "nps": 1,
        }
        deep = {
            "move": "c7", "score": -12, "level": "31", "depth": "31@74%",
            "time": "000:00:10.000", "nodes": 4, "nps": 1,
            "method": "time_disagreement_tiebreak_levels_30_31",
            "verification_mode": "levels_30_31_tiebreak",
            "primary": primary,
            "verification": verification,
            "tiebreak": tiebreak,
        }
        with self.assertRaisesRegex(ValueError, "deep time tiebreak does not support"):
            generate_ggs_root_teacher.validate_verified_teacher_result(
                GGS_ROOT,
                deep,
                method="time_then_verify",
                min_depth=30,
                min_selectivity=74,
                fallback_level=30,
                teacher_level=30,
                verify_level=31,
            )


if __name__ == "__main__":
    unittest.main()
