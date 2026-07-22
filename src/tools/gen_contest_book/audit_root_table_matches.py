"""Audit color-swapped matches of a root-move table against no book.

The game runner records its table-using side under the historical JSON key
``candidate`` and the no-book side under ``baseline``.  This tool maps those
keys to their concrete roles, verifies artifacts and logs, and writes a
bilingual report without using those ambiguous labels in the report itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any

from build_book import canonicalize_board_key, move_from_representative, transform_board_text
from build_root_table import (
    load_root_rows,
    load_root_table_entries,
    manifest_path_for_root_table,
    sha256_file,
)
from generate_ggs_root_teacher import TEACHER_MANIFEST_SCHEMA, validate_calculation_provenance
from othello import Board, coord_to_index, index_to_coord, normalize_board_text
from prepare_root_table_match import PREPARED_SCHEMA, _validate_manifest_results


AUDIT_SCHEMA = "root_table_match_audit_v1"
BOOK_ZERO_NODES_RE = re.compile(r"level Book depth .* nodes 0", re.IGNORECASE)
TABLE_SELECTION_RE = re.compile(
    r"^contest root table selected\s+(?P<move>[a-h][1-8])\s+"
    r"value\s+(?P<value>[+-]?\d+)\s+roots\s+(?P<roots>\d+)\s+"
    r"(?P<board>[XO-]{64}\s+[XO])\s*$",
    re.MULTILINE | re.IGNORECASE,
)
REQUIRED_TIME_SECONDS = 60
REQUIRED_GAME_THREADS = 8
REQUIRED_GAME_HASH = 29
REQUIRED_TEACHER_THREADS = 28
REQUIRED_MIN_DEPTH = 30
REQUIRED_MIN_SELECTIVITY = 74
REQUIRED_VERIFY_LEVEL = 31
REQUIRED_MATCH_SEED = 624


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _load_rows(results_path: Path) -> list[dict[str, Any]]:
    try:
        lines = results_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read match results {results_path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{results_path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{results_path}:{line_number}: row is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{results_path}: no completed matches")
    return rows


def _load_openings(prepared: dict[str, Any]) -> list[str]:
    openings = prepared.get("openings")
    if not isinstance(openings, dict):
        raise ValueError("prepared input is missing openings")
    path_text = openings.get("path")
    expected_sha256 = openings.get("sha256")
    expected_entries = openings.get("entries")
    if not isinstance(path_text, str) or not isinstance(expected_sha256, str):
        raise ValueError("prepared input has invalid openings provenance")
    path = Path(path_text)
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("prepared starting-position list changed or is missing")
    boards = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not isinstance(expected_entries, int) or expected_entries != len(boards):
        raise ValueError("prepared starting-position count is inconsistent")
    return boards


def _canonical_set(boards: list[str]) -> set[str]:
    result: set[str] = set()
    for board in boards:
        try:
            canonical, _ = canonicalize_board_key(board)
        except ValueError as error:
            raise ValueError(f"invalid starting position {board!r}") from error
        result.add(canonical)
    return result


def _sha256_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _snapshot_file_is_current(snapshot: Any) -> bool:
    if not isinstance(snapshot, dict):
        return False
    path_text = snapshot.get("path")
    expected_sha256 = snapshot.get("sha256")
    if not isinstance(path_text, str) or not isinstance(expected_sha256, str):
        return False
    path = Path(path_text)
    return path.is_file() and sha256_file(path) == expected_sha256


def _expected_table_move(board: str, entries: dict[str, Any]) -> str:
    canonical, source_to_canonical = canonicalize_board_key(board)
    entry = entries.get(canonical)
    if entry is None or not entry.moves:
        raise ValueError("temporary table has no entry for a matched starting position")
    canonical_move = entry.moves[0][0]
    source_move = move_from_representative(canonical_move, source_to_canonical)
    return index_to_coord(source_move)


def _expected_game_boards(openings: list[str]) -> list[str]:
    """Recreate the fixed seed-624 rotation/reflection procedure used for games."""
    generator = random.Random(REQUIRED_MATCH_SEED)
    selected = generator.sample(openings, len(openings))
    return [transform_board_text(board, generator.randrange(8)) for board in selected]


def _d4_canonical_board(board: str) -> str:
    """Match the game runner's rotation/reflection representative exactly."""
    return min(transform_board_text(board, symmetry) for symmetry in range(8))


def _validate_prepared_execution_environment(
    prepared_environment: object,
    provenance: dict[str, Any],
) -> tuple[Path | None, list[str]]:
    """Verify copied Console inputs before using them as audit evidence."""
    if not isinstance(prepared_environment, dict):
        return None, ["prepared input has no saved teacher execution environment"]
    path_text = prepared_environment.get("path")
    file_records = prepared_environment.get("files")
    if not isinstance(path_text, str) or not isinstance(file_records, list):
        return None, ["prepared teacher execution environment is incomplete"]
    actual_root = Path(path_text).resolve()
    environment = provenance.get("execution_environment")
    if not isinstance(environment, dict) or not isinstance(environment.get("directory"), str):
        return None, ["teacher calculation evidence has no execution environment"]
    recorded_root = Path(environment["directory"]).resolve()
    executable = environment.get("executable")
    resources = environment.get("resources")
    if not isinstance(executable, dict) or not isinstance(executable.get("snapshot"), dict) or not isinstance(resources, list):
        return None, ["teacher calculation evidence has incomplete execution inputs"]
    expected: list[tuple[str, dict[str, Any]]] = [("executable", executable["snapshot"])]
    for resource in resources:
        if not isinstance(resource, dict) or not isinstance(resource.get("role"), str) or not isinstance(resource.get("snapshot"), dict):
            return None, ["teacher calculation evidence has an invalid saved resource"]
        expected.append((resource["role"], resource["snapshot"]))
    records_by_role: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for record in file_records:
        if not isinstance(record, dict) or not isinstance(record.get("role"), str):
            failures.append("prepared teacher execution environment has an invalid file record")
            continue
        role = record["role"]
        if role in records_by_role:
            failures.append("prepared teacher execution environment lists a role more than once")
        else:
            records_by_role[role] = record
    if set(records_by_role) != {role for role, _snapshot in expected}:
        failures.append("prepared teacher execution environment has different files from the teacher manifest")
    for role, snapshot in expected:
        path_value = snapshot.get("path")
        digest = snapshot.get("sha256")
        size = snapshot.get("bytes")
        if not isinstance(path_value, str) or not isinstance(digest, str) or not isinstance(size, int):
            failures.append(f"teacher manifest has an invalid saved {role} fingerprint")
            continue
        try:
            relative = Path(path_value).resolve().relative_to(recorded_root)
        except ValueError:
            failures.append(f"teacher manifest saved {role} is outside its execution environment")
            continue
        expected_path = actual_root / relative
        record = records_by_role.get(role)
        if not isinstance(record, dict):
            continue
        if (
            record.get("relative_path") != relative.as_posix()
            or record.get("path") != expected_path.resolve().as_posix()
            or record.get("sha256") != digest
            or record.get("bytes") != size
        ):
            failures.append(f"prepared saved {role} does not match the teacher manifest")
            continue
        if not expected_path.is_file() or sha256_file(expected_path) != digest or expected_path.stat().st_size != size:
            failures.append(f"prepared saved {role} does not match its SHA-256 or byte count")
    return actual_root, failures


def _validate_prepared_input(
    prepared: dict[str, Any],
    minimum_processed: int,
    minimum_accepted: int,
) -> tuple[list[str], dict[str, Any], str | None, dict[str, Any], list[str]]:
    """Recheck every frozen artifact that ties teacher rows to the temporary table."""
    failures: list[str] = []
    teacher_calculation = {
        "ordinary_book_disabled": False,
        "contest_book_disabled": False,
        "teacher_script_sha256": None,
        "teacher_script_snapshot_sha256": None,
        "random_seed": None,
        "evaluation_sha256": None,
        "endgame_move_ordering_sha256": None,
    }
    try:
        openings = _load_openings(prepared)
    except ValueError as error:
        return [], {}, None, teacher_calculation, [str(error)]

    teacher = prepared.get("teacher_results")
    snapshot = prepared.get("snapshot")
    teacher_manifest = prepared.get("teacher_manifest")
    teacher_script_snapshot = prepared.get("teacher_script_snapshot")
    teacher_execution_environment = prepared.get("teacher_execution_environment")
    table = prepared.get("table")
    selection = prepared.get("selection")
    required_sections = (
        teacher,
        snapshot,
        teacher_manifest,
        teacher_script_snapshot,
        teacher_execution_environment,
        table,
        selection,
    )
    if not all(isinstance(value, dict) for value in required_sections):
        return openings, {}, None, teacher_calculation, ["prepared input is missing immutable provenance"]

    accepted = teacher.get("accepted")
    processed = teacher.get("processed")
    rejected = teacher.get("rejected")
    if not all(isinstance(value, int) and value >= 0 for value in (accepted, processed, rejected)):
        failures.append("prepared input has invalid teacher-result counts")
    elif processed != accepted + rejected:
        failures.append("prepared input has inconsistent teacher-result counts")
    elif processed < minimum_processed:
        failures.append(
            f"prepared input processed {processed} positions, below required {minimum_processed}"
        )
    elif accepted < minimum_accepted:
        failures.append(
            f"prepared input accepted {accepted} positions, below required {minimum_accepted}"
        )

    if selection.get("all_accepted_positions_used") is not True:
        failures.append("prepared input does not state that every accepted position was used")
    if selection.get("minimum_processed") != minimum_processed:
        failures.append("prepared input used a different minimum processed-position count")
    if selection.get("minimum_accepted") != minimum_accepted:
        failures.append("prepared input used a different minimum accepted-position count")

    snapshot_path_text = snapshot.get("path")
    if not _snapshot_file_is_current(snapshot):
        failures.append("frozen teacher-row file does not match its SHA-256")
        teacher_rows: list[Any] = []
    else:
        try:
            teacher_rows = load_root_rows(Path(str(snapshot_path_text)), 14)
        except ValueError as error:
            failures.append(f"frozen teacher-row file is invalid: {error}")
            teacher_rows = []

    if isinstance(accepted, int) and len(teacher_rows) != accepted:
        failures.append("frozen teacher-row count does not equal accepted-position count")
    teacher_entries = {entry.board: entry for entry in teacher_rows}
    if len(teacher_entries) != len(teacher_rows):
        failures.append("frozen teacher rows contain duplicate canonical positions")
    try:
        if _canonical_set(openings) != set(teacher_entries) or len(openings) != len(teacher_entries):
            failures.append("starting-position list does not equal the frozen teacher rows")
    except ValueError as error:
        failures.append(str(error))

    teacher_engine_sha256: str | None = None
    if not _snapshot_file_is_current(teacher_manifest):
        failures.append("frozen teacher manifest does not match its SHA-256")
    else:
        try:
            manifest = _read_json(Path(str(teacher_manifest["path"])), "frozen teacher manifest")
        except ValueError as error:
            failures.append(str(error))
            manifest = {}
        output = manifest.get("output") if isinstance(manifest, dict) else None
        engine = manifest.get("engine") if isinstance(manifest, dict) else None
        if manifest.get("schema") != TEACHER_MANIFEST_SCHEMA:
            failures.append("frozen teacher manifest has an unsupported schema")
        if not isinstance(output, dict) or output.get("sha256") != snapshot.get("sha256"):
            failures.append("frozen teacher manifest does not identify the frozen teacher rows")
        elif (
            output.get("processed") != processed
            or output.get("completed") != accepted
            or output.get("rejected") != rejected
        ):
            failures.append("frozen teacher manifest counts do not match the prepared input")
        if not isinstance(engine, dict) or not isinstance(engine.get("sha256"), str):
            failures.append("frozen teacher manifest has no executable SHA-256")
        else:
            teacher_engine_sha256 = engine["sha256"]
        provenance_for_environment = manifest.get("calculation_provenance")
        if isinstance(provenance_for_environment, dict):
            prepared_environment_root, environment_failures = _validate_prepared_execution_environment(
                teacher_execution_environment,
                provenance_for_environment,
            )
            failures.extend(environment_failures)
        else:
            prepared_environment_root = None
            failures.append("frozen teacher manifest has no calculation evidence")
        if not _snapshot_file_is_current(teacher_script_snapshot):
            failures.append("frozen teacher script does not match its SHA-256")
        elif prepared_environment_root is None:
            failures.append("frozen teacher execution environment cannot be verified")
        else:
            script_copy = Path(str(teacher_script_snapshot["path"]))
            try:
                provenance = validate_calculation_provenance(
                    manifest.get("calculation_provenance"),
                    snapshot_path=script_copy,
                    execution_environment_directory=prepared_environment_root,
                )
            except ValueError as error:
                failures.append(f"frozen teacher calculation evidence is invalid: {error}")
            else:
                saved_script = provenance["teacher_script_snapshot"]
                if saved_script["sha256"] != teacher_script_snapshot.get("sha256"):
                    failures.append("frozen teacher script SHA-256 differs from the teacher manifest")
                else:
                    saved_executable = provenance["execution_environment"]["executable"]["snapshot"]
                    if teacher_engine_sha256 is not None and saved_executable["sha256"] != teacher_engine_sha256:
                        failures.append("frozen teacher executable SHA-256 differs from its calculation evidence")
                    book_configuration = provenance["book_configuration"]
                    if manifest.get("random_seed") != provenance["random_seed"]:
                        failures.append("frozen teacher manifest random seed differs from its calculation evidence")
                    resources = {
                        resource["role"]: resource["snapshot"]["sha256"]
                        for resource in provenance["execution_environment"]["resources"]
                    }
                    teacher_calculation = {
                        "ordinary_book_disabled": book_configuration["ordinary_book"]["disabled"],
                        "contest_book_disabled": book_configuration["contest_book"]["disabled"],
                        "teacher_script_sha256": provenance["teacher_script"]["sha256"],
                        "teacher_script_snapshot_sha256": saved_script["sha256"],
                        "random_seed": provenance["random_seed"],
                        "evaluation_sha256": resources["evaluation"],
                        "endgame_move_ordering_sha256": resources["endgame_move_ordering"],
                    }
        try:
            _validate_manifest_results(
                teacher_rows,
                manifest,
                Path(str(snapshot_path_text)),
            )
        except ValueError as error:
            failures.append(f"frozen teacher rows do not match the teacher manifest: {error}")
        for key, expected in (
            ("time_seconds", REQUIRED_TIME_SECONDS),
            ("threads", REQUIRED_TEACHER_THREADS),
            ("hash_level", REQUIRED_GAME_HASH),
            ("min_depth", REQUIRED_MIN_DEPTH),
            ("min_selectivity", REQUIRED_MIN_SELECTIVITY),
            ("verify_level", REQUIRED_VERIFY_LEVEL),
        ):
            value = manifest.get(key)
            if key in {"min_depth", "min_selectivity", "verify_level"}:
                if not isinstance(value, int) or value < expected:
                    failures.append(f"frozen teacher manifest has insufficient {key}")
            elif value != expected:
                failures.append(f"frozen teacher manifest has {key}={value!r}, expected {expected}")

    table_entries: dict[str, Any] = {}
    table_path_text = table.get("path")
    if not _snapshot_file_is_current(table):
        failures.append("temporary table does not match its SHA-256")
    else:
        table_path = Path(str(table_path_text))
        try:
            root_discs, table_entries = load_root_table_entries(table_path, 14)
        except ValueError as error:
            failures.append(f"temporary table is invalid: {error}")
            root_discs = None
            table_entries = {}
        if root_discs != 14 or table.get("entries") != len(table_entries):
            failures.append("temporary table count does not match the prepared input")
        if table_entries != teacher_entries:
            failures.append("temporary-table entries do not equal the frozen teacher rows")
        table_manifest_path = manifest_path_for_root_table(table_path)
        if not table_manifest_path.is_file() or sha256_file(table_manifest_path) != table.get("manifest_sha256"):
            failures.append("temporary-table manifest does not match its SHA-256")
        else:
            try:
                table_manifest = _read_json(table_manifest_path, "temporary-table manifest")
            except ValueError as error:
                failures.append(str(error))
                table_manifest = {}
            sources = table_manifest.get("sources") if isinstance(table_manifest, dict) else None
            if not isinstance(sources, list) or not any(
                isinstance(source, dict)
                and source.get("kind") == "root_result"
                and source.get("sha256") == snapshot.get("sha256")
                for source in sources
            ):
                failures.append("temporary-table manifest does not identify the frozen teacher rows")
    return openings, table_entries, teacher_engine_sha256, teacher_calculation, failures


def _validate_metadata(
    metadata_path: Path,
    prepared: dict[str, Any],
    openings_input: list[str],
    expected_game_boards: list[str],
    teacher_engine_sha256: str | None,
    teacher_calculation: dict[str, Any],
) -> list[str]:
    """Return violations when anything except the temporary table differs."""
    payload = _read_json(metadata_path, "match metadata")
    run_spec = payload.get("run_spec")
    expected_sha = payload.get("run_spec_sha256")
    if not isinstance(run_spec, dict) or expected_sha != _canonical_json_sha256(run_spec):
        return ["match metadata checksum is invalid"]
    parsed = run_spec.get("parsed_args")
    artifacts = run_spec.get("artifacts")
    commands = run_spec.get("engine_commands")
    openings = run_spec.get("openings")
    if not all(isinstance(value, dict) for value in (parsed, artifacts, commands, openings)):
        return ["match metadata is missing required sections"]
    failures: list[str] = []
    for key, expected in (
        ("time", REQUIRED_TIME_SECONDS),
        ("threads", REQUIRED_GAME_THREADS),
        ("hash", REQUIRED_GAME_HASH),
        ("matches", len(expected_game_boards)),
        ("seed", REQUIRED_MATCH_SEED),
    ):
        if parsed.get(key) != expected:
            failures.append(f"game metadata has {key}={parsed.get(key)!r}, expected {expected}")
    workers = parsed.get("workers")
    if workers != 1:
        failures.append("game metadata must use exactly one simultaneous game")
    if parsed.get("random_symmetry") is not True:
        failures.append("game metadata does not enable the fixed rotation/reflection procedure")
    for key in ("candidate_extra", "baseline_extra"):
        if parsed.get(key) != "":
            failures.append(f"game metadata has a nonempty {key}")
    if parsed.get("contestbook") is not None:
        failures.append("game metadata has a shared table directory")

    table = prepared.get("table")
    if not isinstance(table, dict) or not isinstance(table.get("path"), str):
        return failures + ["prepared input has no temporary-table path"]
    prepared_table = Path(table["path"]).resolve()
    prepared_table_dir = prepared_table.parent
    table_dir_text = parsed.get("candidate_contestbook")
    if not isinstance(table_dir_text, str) or Path(table_dir_text).resolve() != prepared_table_dir:
        failures.append("table-using side has a different temporary-table directory")
    elif not prepared_table.is_file() or sha256_file(prepared_table) != table.get("sha256"):
        failures.append("table used for games does not match the prepared table")
    unexpected_books = [
        path for path in prepared_table_dir.rglob("*.egcb") if path.resolve() != prepared_table
    ]
    if unexpected_books:
        failures.append("temporary-table directory contains an individual contest book")
    if parsed.get("baseline_contestbook") is not None:
        failures.append("no-book side has a table directory")

    table_binary = artifacts.get("candidate_binary")
    no_book_binary = artifacts.get("baseline_binary")
    evaluation = artifacts.get("evaluation")
    endgame_move_ordering = artifacts.get("endgame_move_ordering")
    harness = run_spec.get("harness")
    for label, snapshot in (
        ("table-using executable", table_binary),
        ("no-book executable", no_book_binary),
        ("evaluation data", evaluation),
        ("endgame move-ordering data", endgame_move_ordering),
        ("game runner", harness),
    ):
        if not _snapshot_file_is_current(snapshot):
            failures.append(f"{label} does not match its recorded SHA-256")
    if not isinstance(table_binary, dict) or not isinstance(no_book_binary, dict):
        failures.append("game metadata lacks executable fingerprints")
        return failures
    if table_binary.get("sha256") != no_book_binary.get("sha256"):
        failures.append("table-using side and no-book side used different executables")
    if teacher_engine_sha256 is None or table_binary.get("sha256") != teacher_engine_sha256:
        failures.append("game executable does not match the executable that calculated the table")
    if not all(
        isinstance(snapshot, dict) and isinstance(snapshot.get("path"), str)
        for snapshot in (evaluation, endgame_move_ordering)
    ):
        failures.append("game metadata lacks Console-input provenance")
        return failures

    binary_path_text = table_binary.get("path")
    no_book_binary_path_text = no_book_binary.get("path")
    if not isinstance(binary_path_text, str) or not isinstance(no_book_binary_path_text, str):
        failures.append("game metadata lacks executable paths")
        return failures
    binary_path = Path(binary_path_text).resolve()
    no_book_binary_path = Path(no_book_binary_path_text).resolve()
    evaluation_path = Path(evaluation["path"]).resolve()
    endgame_move_ordering_path = Path(endgame_move_ordering["path"]).resolve()
    if binary_path != no_book_binary_path:
        failures.append("table-using side and no-book side use different executable paths")
    if parsed.get("candidate") != str(binary_path) or parsed.get("baseline") != str(no_book_binary_path):
        failures.append("parsed executable paths do not match executable fingerprints")
    if endgame_move_ordering_path != binary_path.parent / "resources" / "eval_move_ordering_end.egev":
        failures.append("game metadata has an unexpected endgame move-ordering path")
    for key, artifact in (
        ("evaluation_sha256", evaluation),
        ("endgame_move_ordering_sha256", endgame_move_ordering),
    ):
        if teacher_calculation.get(key) is None or artifact.get("sha256") != teacher_calculation[key]:
            failures.append(f"game {key.removesuffix('_sha256')} does not match teacher calculation")
    common_command = [
        str(binary_path),
        "-quiet",
        "-noise",
        "-nobook",
        "-t",
        str(REQUIRED_GAME_THREADS),
        "-hash",
        str(REQUIRED_GAME_HASH),
        "-eval",
        str(evaluation_path),
        "-time",
        str(REQUIRED_TIME_SECONDS),
    ]
    if commands.get("candidate") != [
        *common_command,
        "-contestbook",
        str(prepared_table_dir),
    ]:
        failures.append("table-using engine command differs from the prescribed command")
    if commands.get("baseline") != common_command:
        failures.append("no-book engine command differs from the prescribed command")

    selected_canonical = sorted(_d4_canonical_board(board) for board in expected_game_boards)
    pool_canonical = sorted(_d4_canonical_board(board) for board in openings_input)
    expected_opening_metadata = {
        "raw_count": len(pool_canonical),
        "d4_unique_count": len(pool_canonical),
        "d4_duplicates_dropped": 0,
        "canonical_pool_sha256": _sha256_lines(pool_canonical),
        "selected_count": len(expected_game_boards),
        "ordered_sha256": _sha256_lines(expected_game_boards),
        "d4_canonical_set_sha256": _sha256_lines(selected_canonical),
    }
    for key, expected in expected_opening_metadata.items():
        if openings.get(key) != expected:
            failures.append(f"game metadata has an unexpected starting-position {key}")
    return failures


def _bootstrap_intervals(rows: list[dict[str, Any]], seed: int, repetitions: int) -> dict[str, tuple[float, float]]:
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    points = [
        (
            1.0 if row["result"] == "W" else 0.5 if row["result"] == "D" else 0.0,
            float(row["margin"]),
        )
        for row in rows
    ]
    generator = random.Random(seed)
    score_samples: list[float] = []
    margin_samples: list[float] = []
    for _ in range(repetitions):
        selected = [points[generator.randrange(len(points))] for _ in points]
        score_samples.append(sum(point[0] for point in selected) / len(selected))
        margin_samples.append(sum(point[1] for point in selected) / len(selected))
    score_samples.sort()
    margin_samples.sort()
    lower = int(0.025 * repetitions)
    upper = int(0.975 * repetitions) - 1
    return {
        "score": (score_samples[lower], score_samples[upper]),
        "margin": (margin_samples[lower], margin_samples[upper]),
    }


def _replay_game(
    board_text: str,
    game: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Replay one recorded game and recompute its terminal information."""
    failures: list[str] = []
    try:
        board = normalize_board_text(board_text)
        position = Board.from_text(board)
    except ValueError as error:
        return {}, [f"invalid starting position in game record: {error}"]
    side_to_move = board[65]
    candidate_color = game.get("candidate_color")
    if candidate_color not in {"X", "O"}:
        return {}, ["game has an invalid table-side color"]
    record = game.get("record")
    if not isinstance(record, str) or len(record) % 2:
        return {}, ["game has an invalid move record"]
    first_move: str | None = None
    for offset in range(0, len(record), 2):
        while not position.legal_moves():
            if position.is_end():
                failures.append("move record continues after the game ended")
                return {}, failures
            position.pass_turn()
            side_to_move = "O" if side_to_move == "X" else "X"
        move = record[offset:offset + 2].lower()
        try:
            position.play(coord_to_index(move))
        except ValueError as error:
            failures.append(f"move record contains an illegal move {move}: {error}")
            return {}, failures
        if first_move is None:
            first_move = move
        side_to_move = "O" if side_to_move == "X" else "X"
    if not position.is_end():
        failures.append("move record ends before the game ended")
        return {}, failures
    current_player_discs = position.cells.count("X")
    current_opponent_discs = position.cells.count("O")
    if side_to_move == "X":
        black_discs, white_discs = current_player_discs, current_opponent_discs
    else:
        black_discs, white_discs = current_opponent_discs, current_player_discs
    candidate_discs = black_discs if candidate_color == "X" else white_discs
    other_discs = white_discs if candidate_color == "X" else black_discs
    empty_discs = 64 - black_discs - white_discs
    if candidate_discs > other_discs:
        candidate_difference = candidate_discs - other_discs + empty_discs
    elif candidate_discs < other_discs:
        candidate_difference = candidate_discs - other_discs - empty_discs
    else:
        candidate_difference = 0
    final_discs = game.get("final_discs")
    if final_discs != [black_discs, white_discs]:
        failures.append("game final discs do not match replayed moves")
    if game.get("candidate_disc_diff") != candidate_difference:
        failures.append("game table-side disc difference does not match replayed moves")
    return {
        "first_move": first_move,
        "starting_side": board[65],
        "candidate_difference": candidate_difference,
        "final_discs": [black_discs, white_discs],
    }, failures


def _selection_matches(log_text: str) -> list[re.Match[str]]:
    return list(TABLE_SELECTION_RE.finditer(log_text))


def _audit_rows(
    rows: list[dict[str, Any]],
    expected_boards: list[str],
    table_entries: dict[str, Any],
) -> tuple[dict[str, int], list[str]]:
    failures: list[str] = []
    seen_matches: set[int] = set()
    table_log_text: list[str] = []
    no_book_log_text: list[str] = []
    engine_audits: list[dict[str, Any]] = []
    expected_selections = 0
    for row in rows:
        match = row.get("match")
        board = row.get("board")
        games = row.get("games")
        if (
            not isinstance(match, int)
            or match in seen_matches
            or not 0 <= match < len(expected_boards)
        ):
            failures.append("match identifiers are missing or duplicated")
            continue
        seen_matches.add(match)
        if not isinstance(board, str) or not isinstance(games, list) or len(games) != 2:
            failures.append(f"match {match} does not contain one board and two games")
            continue
        if board != expected_boards[match]:
            failures.append(f"match {match} does not use its fixed starting position")
        if row.get("result") not in {"W", "D", "L"} or not isinstance(row.get("margin"), (int, float)):
            failures.append(f"match {match} has invalid result or margin")
        differences: list[float] = []
        colors: set[str] = set()
        game_identifiers: set[int] = set()
        for game in games:
            if not isinstance(game, dict):
                failures.append(f"match {match} has an invalid game record")
                continue
            game_id = game.get("game")
            if not isinstance(game_id, int) or game_id in game_identifiers or game_id not in {0, 1}:
                failures.append(f"match {match} has invalid or duplicated game identifiers")
            else:
                game_identifiers.add(game_id)
            color = game.get("candidate_color")
            if color not in {"X", "O"}:
                failures.append(f"match {match} has an invalid table-side color")
            else:
                colors.add(color)
            replay, replay_failures = _replay_game(board, game)
            for failure in replay_failures:
                failures.append(f"match {match}: {failure}")
            if replay:
                differences.append(float(replay["candidate_difference"]))
            expected_selection = bool(replay and color == replay["starting_side"])
            if expected_selection:
                expected_selections += 1
            engine_audit = game.get("engine_audit")
            if not isinstance(engine_audit, dict) or not isinstance(engine_audit.get("candidate"), dict) or not isinstance(engine_audit.get("baseline"), dict):
                failures.append(f"match {match} is missing engine audit data")
                continue
            table_audit = engine_audit["candidate"]
            no_book_audit = engine_audit["baseline"]
            engine_audits.extend([table_audit, no_book_audit])
            log_contents: dict[str, str] = {}
            for role, audit, logs in (
                ("table", table_audit, table_log_text),
                ("no-book", no_book_audit, no_book_log_text),
            ):
                log_text = audit.get("log")
                log_sha256 = audit.get("log_sha256")
                if not isinstance(log_text, str) or not isinstance(log_sha256, str):
                    failures.append(f"match {match} has {role} log provenance missing")
                    continue
                log_path = Path(log_text)
                if not log_path.is_file() or sha256_file(log_path) != log_sha256:
                    failures.append(f"match {match} {role} log does not match its SHA-256")
                    continue
                content = log_path.read_text(encoding="utf-8", errors="replace")
                log_contents[role] = content
                logs.append(content)
            table_log = log_contents.get("table")
            no_book_log = log_contents.get("no-book")
            if table_log is not None:
                if table_log.count("contest root table loaded ") != 1:
                    failures.append(f"match {match} table-side log did not load exactly one temporary table")
                selections = _selection_matches(table_log)
                if expected_selection:
                    if len(selections) != 1:
                        failures.append(f"match {match} table-side log lacks exactly one starting-position selection")
                    else:
                        selection = selections[0]
                        try:
                            if canonicalize_board_key(selection["board"])[0] != canonicalize_board_key(board)[0]:
                                failures.append(f"match {match} table-side log selected a different starting position")
                            expected_move = _expected_table_move(board, table_entries)
                            if selection["move"].lower() != expected_move:
                                failures.append(f"match {match} table-side log selected a move not stored in the temporary table")
                            if replay.get("first_move") != selection["move"].lower():
                                failures.append(f"match {match} selected table move does not equal the first recorded move")
                        except ValueError as error:
                            failures.append(f"match {match} cannot verify the selected table move: {error}")
                    if len(BOOK_ZERO_NODES_RE.findall(table_log)) != 1:
                        failures.append(f"match {match} table-side starting move was not recorded with zero nodes")
                elif selections:
                    failures.append(f"match {match} table-side log selected a table move when it was not the starting side")
                elif BOOK_ZERO_NODES_RE.findall(table_log):
                    failures.append(f"match {match} table-side log has an unexpected zero-node book move")
            if no_book_log is not None and (
                _selection_matches(no_book_log) or "contest root table loaded " in no_book_log
            ):
                failures.append(f"match {match} no-book log used a temporary table")
        if colors != {"X", "O"}:
            failures.append(f"match {match} did not exchange colors")
        if game_identifiers != {0, 1}:
            failures.append(f"match {match} does not contain game identifiers 0 and 1")
        if len(differences) == 2:
            calculated_margin = sum(differences)
            if not isinstance(row.get("margin"), (int, float)) or abs(calculated_margin - float(row["margin"])) > 1e-9:
                failures.append(f"match {match} margin does not equal its replayed game differences")
            expected_result = "W" if calculated_margin > 0 else "L" if calculated_margin < 0 else "D"
            if row.get("result") != expected_result:
                failures.append(f"match {match} result does not equal its replayed game differences")
    if seen_matches != set(range(len(expected_boards))):
        failures.append("completed matches do not equal the fixed starting-position sequence")
    joined_table_logs = "\n".join(table_log_text)
    joined_no_book_logs = "\n".join(no_book_log_text)
    checks = {
        "engine_executions": len(engine_audits),
        "bad_exit": sum(audit.get("exit_code") != 0 for audit in engine_audits),
        "clock_suspicion": sum(bool(audit.get("timeout_suspected")) for audit in engine_audits),
        "zero_clock": sum(bool(audit.get("zero_clock_seen")) for audit in engine_audits),
        "clock_overrun": sum(audit.get("harness_clock_overrun_msec", 0) != 0 for audit in engine_audits),
        "unexpected_engine_errors": sum(bool(audit.get("unexpected_error_lines")) for audit in engine_audits),
        "table_loaded": joined_table_logs.count("contest root table loaded "),
        "table_selected": joined_table_logs.count("contest root table selected "),
        "table_zero_nodes": len(BOOK_ZERO_NODES_RE.findall(joined_table_logs)),
        "no_book_table_loaded": joined_no_book_logs.count("contest root table loaded "),
        "no_book_table_selected": joined_no_book_logs.count("contest root table selected "),
        "expected_table_selections": expected_selections,
    }
    if checks["bad_exit"]:
        failures.append(f"{checks['bad_exit']} abnormal engine exit(s)")
    if checks["clock_suspicion"]:
        failures.append(f"{checks['clock_suspicion']} clock suspicion(s)")
    if checks["zero_clock"]:
        failures.append(f"{checks['zero_clock']} zero-clock observation(s)")
    if checks["clock_overrun"]:
        failures.append(f"{checks['clock_overrun']} game-manager clock overrun(s)")
    if checks["unexpected_engine_errors"]:
        failures.append(f"{checks['unexpected_engine_errors']} engine audit(s) with unexpected errors")
    expected_loads = 2 * len(rows)
    for key, expected, description in (
        ("table_loaded", expected_loads, "temporary-table loads"),
        ("table_selected", expected_selections, "temporary-table selections"),
        ("table_zero_nodes", expected_selections, "zero-node table selections"),
        ("no_book_table_loaded", 0, "no-book temporary-table loads"),
        ("no_book_table_selected", 0, "no-book table selections"),
    ):
        if checks[key] != expected:
            failures.append(f"{description}: {checks[key]}, expected {expected}")
    return checks, failures


def audit_match_results(
    results_path: Path,
    prepared_input_path: Path,
    metadata_path: Path,
    report_path: Path,
    bootstrap_seed: int,
    bootstrap_repetitions: int = 100_000,
    minimum_processed: int = 500,
    minimum_accepted: int = 500,
) -> dict[str, Any]:
    if minimum_processed < 1:
        raise ValueError("minimum_processed must be positive")
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    rows = _load_rows(results_path)
    prepared = _read_json(prepared_input_path, "prepared match input")
    if prepared.get("schema") != PREPARED_SCHEMA:
        raise ValueError(f"{prepared_input_path}: unsupported prepared-input schema")
    (
        openings,
        table_entries,
        teacher_engine_sha256,
        teacher_calculation,
        failures,
    ) = _validate_prepared_input(
        prepared, minimum_processed, minimum_accepted
    )
    expected_game_boards = _expected_game_boards(openings) if openings else []
    checks, row_failures = _audit_rows(rows, expected_game_boards, table_entries)
    failures.extend(row_failures)
    checks["teacher_ordinary_book_disabled"] = teacher_calculation[
        "ordinary_book_disabled"
    ]
    checks["teacher_contest_book_disabled"] = teacher_calculation[
        "contest_book_disabled"
    ]
    failures.extend(
        _validate_metadata(
            metadata_path,
            prepared,
            openings,
            expected_game_boards,
            teacher_engine_sha256,
            teacher_calculation,
        )
    )
    valid_stat_rows = [
        row
        for row in rows
        if row.get("result") in {"W", "D", "L"}
        and isinstance(row.get("margin"), (int, float))
    ]
    if len(valid_stat_rows) != len(rows):
        failures.append("one or more matches cannot be used for statistical aggregation")
        wins = draws = losses = 0
        score_rate = 0.0
        mean_margin = 0.0
        intervals = {"score": (0.0, 0.0), "margin": (0.0, 0.0)}
    else:
        wins = sum(row["result"] == "W" for row in valid_stat_rows)
        draws = sum(row["result"] == "D" for row in valid_stat_rows)
        losses = sum(row["result"] == "L" for row in valid_stat_rows)
        score_rate = (wins + 0.5 * draws) / len(valid_stat_rows)
        mean_margin = sum(float(row["margin"]) for row in valid_stat_rows) / len(valid_stat_rows)
        intervals = _bootstrap_intervals(valid_stat_rows, bootstrap_seed, bootstrap_repetitions)
    valid = not failures
    eligible = valid and intervals["score"][0] > 0.5 and intervals["margin"][0] > 0.0
    payload = {
        "schema": AUDIT_SCHEMA,
        "results": {"path": results_path.resolve().as_posix(), "sha256": sha256_file(results_path)},
        "prepared_input": {"path": prepared_input_path.resolve().as_posix(), "sha256": sha256_file(prepared_input_path)},
        "metadata": {"path": metadata_path.resolve().as_posix(), "sha256": sha256_file(metadata_path)},
        "matches": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": score_rate,
        "mean_margin": mean_margin,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "minimum_processed": minimum_processed,
        "minimum_accepted": minimum_accepted,
        "score_interval": intervals["score"],
        "margin_interval": intervals["margin"],
        "checks": checks,
        "teacher_calculation": teacher_calculation,
        "failures": failures,
        "valid": valid,
        "eligible_for_adoption": eligible,
    }
    failure_text_ja = "なし" if not failures else "<br>".join(f"検査失敗: {failure}" for failure in failures)
    failure_text_en = "none" if not failures else "<br>".join(failures)
    decision_ja = (
        "有効な対局であり、得点率と平均石差の両方の95%区間下限が中立値を上回った。採用を検討できる。"
        if eligible
        else "大会用ファイルへは追加しない。対局の有効性または事前に定めた95%区間の条件を満たしていない。"
    )
    decision_en = (
        "The games are valid and both lower 95% limits exceed their neutral values. Adoption may be considered."
        if eligible
        else "Do not add moves to the tournament file: validity or the pre-specified 95% interval condition is not satisfied."
    )
    ordinary_book_ja = (
        "無効化を確認" if teacher_calculation["ordinary_book_disabled"] else "無効化を確認できない"
    )
    contest_book_ja = (
        "無効化を確認" if teacher_calculation["contest_book_disabled"] else "無効化を確認できない"
    )
    ordinary_book_en = (
        "disabled and verified"
        if teacher_calculation["ordinary_book_disabled"]
        else "not verified as disabled"
    )
    contest_book_en = (
        "disabled and verified"
        if teacher_calculation["contest_book_disabled"]
        else "not verified as disabled"
    )
    text = f"""# 開始局面用の手の表を使う対局の監査

## 日本語

- match数: {len(rows)}
- 1 matchの定義: 同じ開始局面から、表を使う側を黒石（X）と白石（O）に一度ずつ置く2局
- 最低処理済み局面数・最低採用局面数: {minimum_processed}・{minimum_accepted}
- 表を使う側の勝ち・引分・負け: {wins}・{draws}・{losses}
- 得点率: {score_rate:.2%}
- 平均石差: {mean_margin:+.3f}石
- 得点率の95%区間: {intervals['score'][0]:.2%} から {intervals['score'][1]:.2%}
- 平均石差の95%区間: {intervals['margin'][0]:+.3f} から {intervals['margin'][1]:+.3f}石
- エンジン実行数: {checks['engine_executions']}
- 表の読込み: {checks['table_loaded']}回、表の選択: {checks['table_selected']}回、探索ノード0での表選択: {checks['table_zero_nodes']}回
- 表を使わない側での表の読込み・表の選択: {checks['no_book_table_loaded']}回・{checks['no_book_table_selected']}回
- 教師計算時の通常book: {ordinary_book_ja}
- 教師計算時の大会book: {contest_book_ja}
- 教師計算に使った生成スクリプトのSHA-256: {teacher_calculation['teacher_script_sha256']}
- 保存した生成スクリプトのSHA-256: {teacher_calculation['teacher_script_snapshot_sha256']}
- 教師計算で指定した乱数seed: {teacher_calculation['random_seed']}
- 教師計算に使った主評価ファイルのSHA-256: {teacher_calculation['evaluation_sha256']}
- 教師計算に使った終盤の手順評価ファイルのSHA-256: {teacher_calculation['endgame_move_ordering_sha256']}
- 監査上の問題: {failure_text_ja}

判定: {decision_ja}

この報告は、ローカルのConsoleで両側へ60秒の持ち時間を与えた対局を対象にする。GGSの延長時間30秒を再現した結果ではない。

## English

- Matches: {len(rows)}
- Definition of one match: two games from one starting position, with the table-using side as X once and O once.
- Minimum processed and accepted positions: {minimum_processed} and {minimum_accepted}
- Table-using side W/D/L: {wins}/{draws}/{losses}
- Score rate: {score_rate:.2%}
- Mean disc margin: {mean_margin:+.3f} discs
- 95% score-rate interval: {intervals['score'][0]:.2%} to {intervals['score'][1]:.2%}
- 95% mean-margin interval: {intervals['margin'][0]:+.3f} to {intervals['margin'][1]:+.3f} discs
- Engine executions: {checks['engine_executions']}
- Temporary-table loads: {checks['table_loaded']}; selections: {checks['table_selected']}; zero-node selections: {checks['table_zero_nodes']}
- Temporary-table loads and selections by the no-book side: {checks['no_book_table_loaded']}; {checks['no_book_table_selected']}
- Ordinary book during teacher calculation: {ordinary_book_en}
- Contest book during teacher calculation: {contest_book_en}
- Generator-script SHA-256 used for teacher calculation: {teacher_calculation['teacher_script_sha256']}
- Saved generator-script SHA-256: {teacher_calculation['teacher_script_snapshot_sha256']}
- Random seed specified for teacher calculation: {teacher_calculation['random_seed']}
- Main evaluation-file SHA-256 used for teacher calculation: {teacher_calculation['evaluation_sha256']}
- Endgame move-ordering evaluation-file SHA-256 used for teacher calculation: {teacher_calculation['endgame_move_ordering_sha256']}
- Audit failures: {failure_text_en}

Decision: {decision_en}

This report covers local Console games with a 60-second time allocation for each side. It does not reproduce GGS's 30-second extension time.
"""
    _atomic_write_text(report_path, text)
    _atomic_write_text(
        report_path.with_suffix(report_path.suffix + ".json"),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prepared-input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    parser.add_argument("--minimum-processed", type=int, default=500)
    parser.add_argument("--minimum-accepted", type=int, default=500)
    args = parser.parse_args()
    payload = audit_match_results(
        args.results,
        args.prepared_input,
        args.metadata,
        args.output,
        args.bootstrap_seed,
        args.bootstrap_repetitions,
        args.minimum_processed,
        args.minimum_accepted,
    )
    print(
        f"matches={payload['matches']} valid={payload['valid']} "
        f"eligible_for_adoption={payload['eligible_for_adoption']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
