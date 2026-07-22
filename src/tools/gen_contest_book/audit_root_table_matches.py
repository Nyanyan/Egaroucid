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
from prepare_root_table_match import (
    PREPARED_SCHEMA,
    _validate_level_verified_manifest_results,
    _validate_manifest_results,
)
from run_prepared_root_table_match import (
    EXPECTED_GGS_TOURNAMENT_BUILD_LOG_LINE,
    EXPECTED_RANDOM_SEED_LOG_LINE,
    INITIAL_REMAINING_MSEC,
    MATCH_PROTOCOL_SCHEMA,
    METADATA_SCHEMA_VERSION,
    RUNNER_DEPENDENT_SOURCE_FILES,
)


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
REQUIRED_ENGINE_RANDOM_SEED = 620
REQUIRED_TEACHER_RANDOM_SEED = 620
SETTIMEMS_COMMAND_RE = re.compile(
    r"^(?:>\s*)?received cmd: settimems (?P<color>[XO]) (?P<remaining_msec>\d+)\r?$",
    re.MULTILINE,
)
EXPECTED_HASH29_INITIALIZATION_ERROR_LINES = (
    "[ERROR] can't open hash29.eghs",
    "[ERROR] can't get hash. you can ignore this error",
)


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


def _snapshot_file_with_size_is_current(snapshot: Any) -> bool:
    if not _snapshot_file_is_current(snapshot) or not isinstance(snapshot, dict):
        return False
    size = snapshot.get("bytes")
    path_text = snapshot.get("path")
    return (
        isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and isinstance(path_text, str)
        and Path(path_text).stat().st_size == size
    )


def _validate_runner_protocol_sources(
    git: object,
    protocol: object,
) -> list[str]:
    """Validate the clean-worktree and source-fingerprint contract."""
    failures: list[str] = []
    if not isinstance(git, dict):
        return ["match metadata has no Git provenance"]
    root_text = git.get("repository_root")
    commit = git.get("commit")
    if not isinstance(root_text, str) or not Path(root_text).is_dir():
        failures.append("match metadata has no existing Git worktree root")
        root = None
    else:
        root = Path(root_text).resolve()
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        failures.append("match metadata does not record a full Git commit SHA-1")
    if git.get("tracked_worktree_dirty") is not False:
        failures.append("formal match was not started from a clean tracked worktree")
    if git.get("tracked_status_sha256") != hashlib.sha256(b"").hexdigest():
        failures.append("clean tracked-worktree status SHA-256 is invalid")
    if not isinstance(protocol, dict):
        return failures + ["match metadata has no runner protocol"]
    if protocol.get("schema") != MATCH_PROTOCOL_SCHEMA:
        failures.append("match metadata has an unsupported runner protocol")
    if protocol.get("clean_tracked_worktree_required") is not True:
        failures.append("match metadata does not require a clean tracked worktree")
    clock = protocol.get("external_clock")
    if not isinstance(clock, dict) or (
        clock.get("command") != "settimems"
        or clock.get("initial_remaining_msec") != INITIAL_REMAINING_MSEC
        or clock.get("only_go_commands_decrement_time") is not True
    ):
        failures.append("match metadata has an invalid external-clock protocol")
    noise_lines = protocol.get("noise_log_lines")
    if not isinstance(noise_lines, dict) or (
        noise_lines.get("random_seed") != EXPECTED_RANDOM_SEED_LOG_LINE
        or noise_lines.get("ggs_tournament_build") != EXPECTED_GGS_TOURNAMENT_BUILD_LOG_LINE
    ):
        failures.append("match metadata has invalid required -noise log lines")
    if protocol.get("process_launch_order") != "candidate-first when (match id + game id) is even":
        failures.append("match metadata has an invalid process-launch-order rule")

    records = protocol.get("source_files")
    if not isinstance(records, list):
        return failures + ["match metadata has no runner source fingerprints"]
    by_relative: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("relative_path"), str):
            failures.append("match metadata has an invalid runner source fingerprint")
            continue
        relative = record["relative_path"]
        if relative in by_relative:
            failures.append("match metadata repeats a runner source fingerprint")
        else:
            by_relative[relative] = record
    if set(by_relative) != set(RUNNER_DEPENDENT_SOURCE_FILES):
        failures.append("match metadata has a different runner source-file set")
    if root is not None:
        for relative in RUNNER_DEPENDENT_SOURCE_FILES:
            record = by_relative.get(relative)
            if record is None:
                continue
            path_text = record.get("path")
            expected_path = (root / relative).resolve()
            if not isinstance(path_text, str) or Path(path_text).resolve() != expected_path:
                failures.append(f"runner source fingerprint has an unexpected path: {relative}")
                continue
            if not _snapshot_file_with_size_is_current(record):
                failures.append(f"runner source file does not match its SHA-256: {relative}")
    return failures


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
    level_31_verification_required = selection.get("level_31_verification_required")
    if not isinstance(level_31_verification_required, bool):
        failures.append("prepared input does not state whether level-31 verification was required")

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
                    if provenance["random_seed"] != REQUIRED_TEACHER_RANDOM_SEED:
                        failures.append(
                            "frozen teacher calculation used an unexpected Console random seed"
                        )
        try:
            _validate_manifest_results(
                teacher_rows,
                manifest,
                Path(str(snapshot_path_text)),
            )
        except ValueError as error:
            failures.append(f"frozen teacher rows do not match the teacher manifest: {error}")
        if level_31_verification_required is True:
            verify_level = manifest.get("verify_level")
            if isinstance(verify_level, bool) or not isinstance(verify_level, int) or verify_level < 31:
                failures.append("frozen teacher manifest has verification below level 31")
            else:
                try:
                    _validate_level_verified_manifest_results(
                        teacher_rows,
                        manifest,
                        Path(str(snapshot_path_text)),
                    )
                except ValueError as error:
                    failures.append(
                        "frozen teacher rows do not have valid level-31 verification: "
                        f"{error}"
                    )
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
    results_path: Path,
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
    if run_spec.get("schema_version") != METADATA_SCHEMA_VERSION:
        return ["match metadata has an unsupported schema version"]
    parsed = run_spec.get("parsed_args")
    artifacts = run_spec.get("artifacts")
    commands = run_spec.get("engine_commands")
    openings = run_spec.get("openings")
    if not all(isinstance(value, dict) for value in (parsed, artifacts, commands, openings)):
        return ["match metadata is missing required sections"]
    failures: list[str] = _validate_runner_protocol_sources(
        run_spec.get("git"), run_spec.get("runner_protocol")
    )
    for key, expected in (
        ("time", REQUIRED_TIME_SECONDS),
        ("threads", REQUIRED_GAME_THREADS),
        ("hash", REQUIRED_GAME_HASH),
        ("matches", len(expected_game_boards)),
        ("seed", REQUIRED_MATCH_SEED),
        ("engine_random_seed", REQUIRED_ENGINE_RANDOM_SEED),
    ):
        if parsed.get(key) != expected:
            failures.append(f"game metadata has {key}={parsed.get(key)!r}, expected {expected}")
    workers = parsed.get("workers")
    if workers != 1:
        failures.append("game metadata must use exactly one simultaneous game")
    if parsed.get("random_symmetry") is not True:
        failures.append("game metadata does not enable the fixed rotation/reflection procedure")
    if parsed.get("external_clock_control") is not True:
        failures.append("game metadata does not enable the external millisecond clock")
    selection = prepared.get("selection")
    required_level_verification = (
        selection.get("level_31_verification_required")
        if isinstance(selection, dict)
        else None
    )
    if parsed.get("level_31_verification_required") is not required_level_verification:
        failures.append(
            "game metadata level-31 verification setting does not match the prepared input"
        )
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
    ordered_starting_positions = artifacts.get("ordered_starting_positions")
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
    if not _snapshot_file_with_size_is_current(ordered_starting_positions):
        failures.append("actual starting-position order file does not match its recorded SHA-256")
    elif isinstance(ordered_starting_positions, dict):
        expected_order_path = results_path.with_suffix(results_path.suffix + ".openings.txt").resolve()
        path_text = ordered_starting_positions.get("path")
        if not isinstance(path_text, str) or Path(path_text).resolve() != expected_order_path:
            failures.append("actual starting-position order file has an unexpected path")
        else:
            expected_order_text = "".join(board + "\n" for board in expected_game_boards)
            try:
                actual_order_text = expected_order_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                failures.append(f"cannot read actual starting-position order file: {error}")
            else:
                if actual_order_text != expected_order_text:
                    failures.append("actual starting-position order file differs from the fixed sequence")
            if ordered_starting_positions.get("sha256") != _sha256_lines(expected_game_boards):
                failures.append("actual starting-position order SHA-256 differs from the fixed sequence")
        if openings.get("ordered_file") != ordered_starting_positions:
            failures.append("opening metadata does not repeat the immutable order-file fingerprint")
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
        "-seed",
        str(REQUIRED_ENGINE_RANDOM_SEED),
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
    move_sides: list[str] = []
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
        move_sides.append(side_to_move)
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
        "move_sides": move_sides,
        "candidate_difference": candidate_difference,
        "final_discs": [black_discs, white_discs],
    }, failures


def _selection_matches(log_text: str) -> list[re.Match[str]]:
    return list(TABLE_SELECTION_RE.finditer(log_text))


def _clock_pair(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict) or set(value) != {"X", "O"}:
        return None
    result: dict[str, int] = {}
    for color in ("X", "O"):
        remaining = value.get(color)
        if (
            isinstance(remaining, bool)
            or not isinstance(remaining, int)
            or not 0 <= remaining <= INITIAL_REMAINING_MSEC
        ):
            return None
        result[color] = remaining
    return result


def _settimems_commands_from_log(log_text: str) -> list[tuple[str, int]]:
    return [
        (match.group("color"), int(match.group("remaining_msec")))
        for match in SETTIMEMS_COMMAND_RE.finditer(log_text)
    ]


def _validate_external_clock(
    game: dict[str, Any],
    replay: dict[str, Any],
) -> tuple[list[str], list[tuple[str, int]]]:
    """Replay the external clock; only recorded ``go`` wall time may reduce it."""
    failures: list[str] = []
    clock = game.get("external_clock")
    if not isinstance(clock, dict):
        return ["game lacks external-clock data"], []
    initial = _clock_pair(clock.get("initial_remaining_msec"))
    if initial != {"X": INITIAL_REMAINING_MSEC, "O": INITIAL_REMAINING_MSEC}:
        failures.append("game has an invalid external-clock initial state")
        return failures, []
    records = clock.get("records")
    if not isinstance(records, list):
        return failures + ["game external-clock records are not a list"], []
    move_sides = replay.get("move_sides")
    record_text = game.get("record")
    candidate_color = game.get("candidate_color")
    if (
        not isinstance(move_sides, list)
        or not isinstance(record_text, str)
        or candidate_color not in {"X", "O"}
    ):
        return failures + ["game cannot be replayed for external-clock validation"], []
    if len(records) != len(move_sides):
        failures.append("external-clock record count does not equal the move count")
    current = dict(initial)
    expected_commands: list[tuple[str, int]] = []
    for index, (side, move) in enumerate(
        zip(move_sides, (record_text[offset:offset + 2].lower() for offset in range(0, len(record_text), 2)))
    ):
        if index >= len(records):
            break
        entry = records[index]
        if not isinstance(entry, dict):
            failures.append(f"external-clock record {index + 1} is not an object")
            continue
        if entry.get("move_number") != index + 1:
            failures.append(f"external-clock record {index + 1} has a wrong move number")
        if entry.get("side") != side:
            failures.append(f"external-clock record {index + 1} has a wrong side")
        expected_role = "candidate" if candidate_color == side else "baseline"
        if entry.get("role") != expected_role:
            failures.append(f"external-clock record {index + 1} has a wrong process role")
        if entry.get("move") != move:
            failures.append(f"external-clock record {index + 1} has a wrong move")
        before = _clock_pair(entry.get("before_remaining_msec"))
        if before != current:
            failures.append(f"external-clock record {index + 1} has a wrong pre-go clock")
            continue
        expected_commands.extend([("X", current["X"]), ("O", current["O"])])
        go_wall_msec = entry.get("go_wall_msec")
        if (
            isinstance(go_wall_msec, bool)
            or not isinstance(go_wall_msec, int)
            or go_wall_msec < 1
            or go_wall_msec >= current[side]
        ):
            failures.append(f"external-clock record {index + 1} has an invalid go wall time")
            continue
        expected_after = dict(current)
        expected_after[side] -= go_wall_msec
        after = _clock_pair(entry.get("after_remaining_msec"))
        if after != expected_after:
            failures.append(f"external-clock record {index + 1} does not subtract only its go time")
            continue
        current = expected_after
    if len(records) > len(move_sides):
        failures.append("external-clock has records after the game ended")
    if _clock_pair(clock.get("final_remaining_msec")) != current:
        failures.append("external-clock final state does not match its move records")
    return failures, expected_commands


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
            if isinstance(game_id, int) and game_id in {0, 1}:
                expected_launch_order = (
                    ["candidate", "baseline"]
                    if (match + game_id) % 2 == 0
                    else ["baseline", "candidate"]
                )
                if game.get("process_launch_order") != expected_launch_order:
                    failures.append(f"match {match} game {game_id} has an invalid process launch order")
            if replay:
                clock_failures, expected_settimems = _validate_external_clock(game, replay)
                for failure in clock_failures:
                    failures.append(f"match {match}: {failure}")
            else:
                expected_settimems = []
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
                expected_hash_errors = sum(
                    content.count(line)
                    for line in EXPECTED_HASH29_INITIALIZATION_ERROR_LINES
                )
                if audit.get("expected_hash_error_lines") != expected_hash_errors:
                    failures.append(
                        f"match {match} {role} expected-hash-error count does not match its log"
                    )
                if expected_hash_errors != len(EXPECTED_HASH29_INITIALIZATION_ERROR_LINES):
                    failures.append(
                        f"match {match} {role} lacks the expected hash-29 initialization record"
                    )
                random_seed_lines = content.splitlines().count(EXPECTED_RANDOM_SEED_LOG_LINE)
                if random_seed_lines != 1 or audit.get("random_seed_log_lines") != random_seed_lines:
                    failures.append(
                        f"match {match} {role} lacks the required deterministic random-seed log"
                    )
                tournament_build_lines = content.splitlines().count(
                    EXPECTED_GGS_TOURNAMENT_BUILD_LOG_LINE
                )
                if (
                    tournament_build_lines != 1
                    or audit.get("ggs_tournament_build_log_lines") != tournament_build_lines
                ):
                    failures.append(
                        f"match {match} {role} lacks the required GGS-tournament-build log"
                    )
                actual_settimems = _settimems_commands_from_log(content)
                saved_settimems = [
                    {"color": color, "remaining_msec": remaining_msec}
                    for color, remaining_msec in actual_settimems
                ]
                if actual_settimems != expected_settimems:
                    failures.append(
                        f"match {match} {role} settimems commands do not match the external-clock timeline"
                    )
                if audit.get("settimems_commands") != saved_settimems:
                    failures.append(
                        f"match {match} {role} saved settimems commands do not match its log"
                    )
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
        launch_orders = {
            tuple(game.get("process_launch_order"))
            if isinstance(game.get("process_launch_order"), list)
            else ()
            for game in games
            if isinstance(game, dict)
        }
        if launch_orders != {("candidate", "baseline"), ("baseline", "candidate")}:
            failures.append(f"match {match} did not cancel process launch order across its two games")
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
        "expected_hash_initialization_errors": sum(
            audit.get("expected_hash_error_lines", 0) for audit in engine_audits
        ),
        "deterministic_seed_logs": sum(
            audit.get("random_seed_log_lines") == 1 for audit in engine_audits
        ),
        "tournament_build_logs": sum(
            audit.get("ggs_tournament_build_log_lines") == 1 for audit in engine_audits
        ),
        "settimems_commands": sum(
            len(audit.get("settimems_commands", []))
            for audit in engine_audits
            if isinstance(audit.get("settimems_commands", []), list)
        ),
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
    if checks["deterministic_seed_logs"] != len(engine_audits):
        failures.append("one or more engine logs lack exactly one deterministic random-seed record")
    if checks["tournament_build_logs"] != len(engine_audits):
        failures.append("one or more engine logs lack exactly one GGS-tournament-build record")
    expected_hash_errors = len(engine_audits) * len(EXPECTED_HASH29_INITIALIZATION_ERROR_LINES)
    if checks["expected_hash_initialization_errors"] != expected_hash_errors:
        failures.append(
            "hash-29 initialization records: "
            f"{checks['expected_hash_initialization_errors']}, expected {expected_hash_errors}"
        )
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
            results_path,
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
    selection_payload = prepared.get("selection")
    level_31_verification_required = (
        isinstance(selection_payload, dict)
        and selection_payload.get("level_31_verification_required") is True
    )
    eligible = (
        valid
        and level_31_verification_required
        and intervals["score"][0] > 0.5
        and intervals["margin"][0] > 0.0
    )
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
        "level_31_verification_required": level_31_verification_required,
        "failures": failures,
        "valid": valid,
        "eligible_for_adoption": eligible,
    }
    failure_text_ja = "なし" if not failures else "<br>".join(f"検査失敗: {failure}" for failure in failures)
    failure_text_en = "none" if not failures else "<br>".join(failures)
    decision_ja = (
        "有効な対局であり、得点率と平均石差の両方の95%区間下限が中立値を上回った。採用を検討できる。"
        if eligible
        else "大会用ファイルへは追加しない。各受理局面のlevel 31以上の照合、対局の有効性、または事前に定めた95%区間の条件を満たしていない。"
    )
    decision_en = (
        "The games are valid and both lower 95% limits exceed their neutral values. Adoption may be considered."
        if eligible
        else "Do not add moves to the tournament file: level-31-or-higher verification for every accepted position, validity, or the pre-specified 95% interval condition is not satisfied."
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
- 各受理局面で level 31 以上の照合を必須にしたか: {str(level_31_verification_required).lower()}
- 表を使う側の勝ち・引分・負け: {wins}・{draws}・{losses}
- 得点率: {score_rate:.2%}
- 平均石差: {mean_margin:+.3f}石
- 得点率の95%区間: {intervals['score'][0]:.2%} から {intervals['score'][1]:.2%}
- 平均石差の95%区間: {intervals['margin'][0]:+.3f} から {intervals['margin'][1]:+.3f}石
- エンジン実行数: {checks['engine_executions']}
- `random seed = 620` を1回出したエンジン: {checks['deterministic_seed_logs']}台
- `ggs tournament build = true` を1回出したエンジン: {checks['tournament_build_logs']}台
- `settimems` の記録済みコマンド数: {checks['settimems_commands']}回
- `-hash 29` の期待どおりの初期化エラー記録: {checks['expected_hash_initialization_errors']}行
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

### この報告で新たに使う用語

#### 外部時計

- 出典: この監査に対応する対局実行スクリプト `root_table_match_protocol_v2`。
- 目的: Consoleが入力待ち時間を内部時計へ加える挙動から、持ち時間の比較を切り離す。
- 具体対象: 各実手の直前・直後のX/O残りミリ秒、およびその手の`go`要求から応答までの実測時間。
- 役割: `go`の実測時間だけを手番側から減算し、次の`go`直前に両Consoleへ`settimems`で同じ値を設定する。
- 前後関係: `setboard`の後、各`go`の直前に設定し、`go`応答後に残時間を更新してから次の手へ進む。
- 候補語: 外部時計、対局管理時計、計測時計。
- 初出定義: 本節の「外部時計」は、上記のJSON時系列と`settimems`受信記録で再生可能な対局管理側の残時間を指す。

#### 実際の開始局面順ファイル

- 出典: 同じ対局実行スクリプト。
- 目的: seedから再計算するだけではなく、実際に使った回転・反射後の開始局面順を固定する。
- 具体対象: 結果JSONLと同じ名前に `.openings.txt` を付けたUTF-8ファイル。
- 役割: 各match番号と開始局面を一意に結び、再開時の順序変更を拒否する。
- 前後関係: 結果・メタデータの作成前に一度だけ書き、監査時に内容とSHA-256を照合する。
- 候補語: 実際の開始局面順ファイル、開始局面順固定ファイル、局面順側carファイル。
- 初出定義: 本節では先頭の表記を用い、結果JSONLに記録されたmatch順と同じ行順の局面列を意味する。

#### プロセス起動順の相殺

- 出典: 同じ対局実行スクリプト。
- 目的: 先に起動したConsoleだけが受ける初期化・資源確保の差を、1 match内で片側へ偏らせない。
- 具体対象: 表を使うConsoleと表を使わないConsoleを起動する順番。
- 役割: 2局のうち一方は表を使うConsoleを先、他方は表を使わないConsoleを先に起動する。
- 前後関係: 各局のConsole起動時に決め、2局を含むmatchの監査時に両順が一度ずつあることを確認する。
- 候補語: プロセス起動順の相殺、起動順交替、先起動効果の相殺。
- 初出定義: 本節では先頭の表記を用い、同一matchの2局で`candidate, baseline`と`baseline, candidate`を一度ずつ使うことを指す。

## English

- Matches: {len(rows)}
- Definition of one match: two games from one starting position, with the table-using side as X once and O once.
- Minimum processed and accepted positions: {minimum_processed} and {minimum_accepted}
- Level-31-or-higher verification required for every accepted position: {str(level_31_verification_required).lower()}
- Table-using side W/D/L: {wins}/{draws}/{losses}
- Score rate: {score_rate:.2%}
- Mean disc margin: {mean_margin:+.3f} discs
- 95% score-rate interval: {intervals['score'][0]:.2%} to {intervals['score'][1]:.2%}
- 95% mean-margin interval: {intervals['margin'][0]:+.3f} to {intervals['margin'][1]:+.3f} discs
- Engine executions: {checks['engine_executions']}
- Engines with exactly one `random seed = 620` line: {checks['deterministic_seed_logs']}
- Engines with exactly one `ggs tournament build = true` line: {checks['tournament_build_logs']}
- Recorded `settimems` commands: {checks['settimems_commands']}
- Expected `-hash 29` initialization-error records: {checks['expected_hash_initialization_errors']}
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

### Terms newly used in this report

#### External clock

- Source: `root_table_match_protocol_v2`, the game runner paired with this audit.
- Purpose: separate the comparison clock from Console's input-wait accounting.
- Concrete subject: X/O milliseconds before and after each played move and the measured `go` request duration.
- Role: subtract only that measured duration from the side to move, then send both values through `settimems` before the next `go`.
- Sequence: after `setboard`, before every `go`, then after its response before the next move.
- Candidate terms: external clock, game-manager clock, measurement clock.
- First definition: in this report, it means the game-manager time sequence reproducible from JSON and `settimems` receive records.

#### Actual starting-position order file

- Source: the same game runner.
- Purpose: freeze the transformed board sequence actually used, rather than only the seed used to derive it.
- Concrete subject: the UTF-8 `.openings.txt` sidecar beside the results JSONL.
- Role: bind each match number to one board and reject a changed order on resume.
- Sequence: written once before results and metadata, then content and SHA-256 are audited.
- Candidate terms: actual starting-position order file, frozen opening-order file, opening-order sidecar.
- First definition: here it means the board lines in exactly the same order as match records in the results JSONL.

#### Process-launch-order cancellation

- Source: the same game runner.
- Purpose: prevent first Console startup effects from consistently favoring one side inside a match.
- Concrete subject: the launch order of the table-using and no-book Console processes.
- Role: launch the table-using process first in one game and the no-book process first in the other game.
- Sequence: chosen when each game starts and checked across the two games of a match.
- Candidate terms: process-launch-order cancellation, alternating launch order, first-launch-effect cancellation.
- First definition: here it means using `candidate, baseline` once and `baseline, candidate` once within one match.
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
