"""Generate resumable, contest-time root teachers for uncovered r14 starts.

Input is an immutable JSON report produced either by ``collect_ggs_roots.py``
from actual GGS starts or by ``audit_r14_corpus.py`` from the complete standard
r14 corpus. Every teacher search disables both ordinary and contest books and
uses one engine process per root. A durable per-position journal is written
after each completed root; complete output files are compacted at the selected
interval. The output data rows are accepted directly by ``build_root_table.py
--root-results``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from audit_r14_corpus import CORPUS_REPORT_SCHEMA
from book_artifact import file_lock
from build_root_table import load_root_rows
from collect_ggs_roots import REPORT_SCHEMA, sha256_file
from othello import Board, coord_to_index


TEACHER_SCHEMA = "ggs_root_teacher_state_v11"
TEACHER_MANIFEST_SCHEMA = "ggs_root_teacher_manifest_v11"
TEACHER_FORMAT = "# ggs_root_teacher_v1"
TEACHER_UPDATE_SCHEMA = "ggs_root_teacher_update_v1"
CALCULATION_PROVENANCE_SCHEMA = "ggs_root_teacher_calculation_provenance_v1"
DEEP_TIEBREAK_LEVEL = 31
BOOK_DISABLED_ARGUMENTS = ("-nobook", "-nocontestbook")
TIME_SEARCH_COMMAND_TEMPLATE = [
    "{executable}",
    "-time",
    "{time_seconds}",
    "-t",
    "{threads}",
    "-hash",
    "{hash_level}",
    *BOOK_DISABLED_ARGUMENTS,
]
LEVEL_SEARCH_COMMAND_TEMPLATE = [
    "{executable}",
    "-l",
    "{level}",
    "-t",
    "{threads}",
    "-hash",
    "{hash_level}",
    *BOOK_DISABLED_ARGUMENTS,
]
TIME_SEARCH_INPUT_TEMPLATE = "setboard {board}\ngo\nquit\n"
LEVEL_SEARCH_INPUT_TEMPLATE = "setboard {board}\nhint 1\nquit\n"
RESULT_RE = re.compile(
    r"^\|\s*(?P<level>[^|]+)\|\s*(?P<depth>[^|]+)\|\s*"
    r"(?P<move>[a-h][1-8])\|\s*(?P<score>[+-]?\d+)\|\s*"
    r"(?P<time>[^|]+)\|\s*(?P<nodes>\d+)\|\s*(?P<nps>\d+)\|\s*$",
    re.MULTILINE,
)
DEPTH_RE = re.compile(r"(?P<depth>\d+)@(?P<selectivity>\d+)%")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _state_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".state.json")


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _pending_updates_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".pending.jsonl")


def _lock_path(output: Path) -> Path:
    """Return the OS-lock path guarding all state for one teacher output."""
    return output.with_suffix(output.suffix + ".lock")


def _teacher_script_snapshot_path(output: Path) -> Path:
    """Return the immutable source-copy path associated with one output."""
    return output.with_suffix(output.suffix + ".teacher_script.py")


def _book_configuration() -> dict[str, dict[str, bool | str]]:
    """Return the exact book-related command-line contract for teacher searches."""
    return {
        "ordinary_book": {"disabled": True, "command_line_option": "-nobook"},
        "contest_book": {"disabled": True, "command_line_option": "-nocontestbook"},
    }


def _command_templates() -> dict[str, list[str]]:
    return {
        "time_limited_search": list(TIME_SEARCH_COMMAND_TEMPLATE),
        "fixed_level_search": list(LEVEL_SEARCH_COMMAND_TEMPLATE),
    }


def _standard_input_templates() -> dict[str, str]:
    return {
        "time_limited_search": TIME_SEARCH_INPUT_TEMPLATE,
        "fixed_level_search": LEVEL_SEARCH_INPUT_TEMPLATE,
    }


def _new_calculation_provenance(output: Path) -> dict[str, Any]:
    """Freeze how this output disables books and which script produced it."""
    teacher_script = Path(__file__).resolve()
    teacher_script_sha256 = sha256_file(teacher_script)
    snapshot = _teacher_script_snapshot_path(output).resolve()
    return {
        "schema": CALCULATION_PROVENANCE_SCHEMA,
        "book_configuration": _book_configuration(),
        "command_templates": _command_templates(),
        "standard_input_templates": _standard_input_templates(),
        "teacher_script": {
            "path": teacher_script.as_posix(),
            "sha256": teacher_script_sha256,
        },
        "teacher_script_snapshot": {
            "path": snapshot.as_posix(),
            "sha256": teacher_script_sha256,
        },
    }


def validate_calculation_provenance(
    provenance: object,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable teacher-calculation evidence.

    ``snapshot_path`` is used by a later audit after the script copy has been
    frozen in a separate directory.  With the default, the path recorded in
    the provenance itself is checked.
    """
    if not isinstance(provenance, dict):
        raise ValueError("teacher calculation provenance is not an object")
    if provenance.get("schema") != CALCULATION_PROVENANCE_SCHEMA:
        raise ValueError("teacher calculation provenance has an unsupported schema")
    if provenance.get("book_configuration") != _book_configuration():
        raise ValueError("teacher calculation provenance does not disable both books")
    if provenance.get("command_templates") != _command_templates():
        raise ValueError("teacher calculation provenance has unexpected command templates")
    if provenance.get("standard_input_templates") != _standard_input_templates():
        raise ValueError("teacher calculation provenance has unexpected input templates")
    teacher_script = provenance.get("teacher_script")
    snapshot = provenance.get("teacher_script_snapshot")
    if not isinstance(teacher_script, dict) or not isinstance(snapshot, dict):
        raise ValueError("teacher calculation provenance has no script fingerprints")
    for label, fingerprint in (("teacher script", teacher_script), ("teacher script snapshot", snapshot)):
        if not isinstance(fingerprint.get("path"), str):
            raise ValueError(f"teacher calculation provenance has no {label} path")
        digest = fingerprint.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"teacher calculation provenance has an invalid {label} SHA-256")
    if teacher_script["sha256"] != snapshot["sha256"]:
        raise ValueError("teacher script and its saved copy have different SHA-256 values")
    script_copy = snapshot_path if snapshot_path is not None else Path(snapshot["path"])
    if not script_copy.is_file() or sha256_file(script_copy) != snapshot["sha256"]:
        raise ValueError("saved teacher script does not match its SHA-256")
    try:
        script_text = script_copy.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read saved teacher script {script_copy}: {error}") from error
    if any(option not in script_text for option in BOOK_DISABLED_ARGUMENTS):
        raise ValueError("saved teacher script does not contain both book-disable options")
    return provenance


def _ensure_teacher_script_snapshot(output: Path, state: dict[str, Any]) -> None:
    """Create once, then verify, the exact generator source beside an output."""
    provenance = state.get("calculation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("teacher state has no calculation provenance")
    teacher_script = provenance.get("teacher_script")
    snapshot = provenance.get("teacher_script_snapshot")
    if not isinstance(teacher_script, dict) or not isinstance(snapshot, dict):
        raise ValueError("teacher state has incomplete script provenance")
    source = Path(__file__).resolve()
    expected_snapshot = _teacher_script_snapshot_path(output).resolve()
    if teacher_script.get("path") != source.as_posix() or teacher_script.get("sha256") != sha256_file(source):
        raise ValueError("teacher state was created by a different generator script")
    if snapshot.get("path") != expected_snapshot.as_posix() or snapshot.get("sha256") != teacher_script["sha256"]:
        raise ValueError("teacher state has an unexpected saved-script location or SHA-256")
    if expected_snapshot.exists():
        if sha256_file(expected_snapshot) != snapshot["sha256"]:
            raise ValueError("saved teacher script does not match its SHA-256")
    else:
        _atomic_write_bytes(expected_snapshot, source.read_bytes())
    validate_calculation_provenance(provenance)


def _append_pending_update(
    output: Path,
    board: str,
    field: str,
    entry: dict[str, Any],
) -> None:
    if field not in {"results", "rejections"}:
        raise ValueError(f"invalid pending-update field: {field}")
    path = _pending_updates_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": TEACHER_UPDATE_SCHEMA,
        "board": board,
        "field": field,
        "entry": entry,
    }
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _apply_pending_updates(output: Path, state: dict[str, Any]) -> bool:
    """Replay durable per-position updates not yet compacted into the state."""
    path = _pending_updates_path(output)
    if not path.exists():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read pending updates {path}: {error}") from error
    changed = False
    expected_roots = set(state["roots"])
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if (
            not isinstance(record, dict)
            or record.get("schema") != TEACHER_UPDATE_SCHEMA
            or record.get("field") not in {"results", "rejections"}
            or not isinstance(record.get("board"), str)
            or not isinstance(record.get("entry"), dict)
        ):
            raise ValueError(f"{path}:{line_number}: invalid pending update")
        board = record["board"]
        field = record["field"]
        entry = record["entry"]
        if board not in expected_roots:
            raise ValueError(f"{path}:{line_number}: pending update has an unexpected root")
        other_field = "rejections" if field == "results" else "results"
        if board in state[other_field]:
            raise ValueError(f"{path}:{line_number}: root is both accepted and rejected")
        previous = state[field].get(board)
        if previous is None:
            state[field][board] = entry
            changed = True
        elif previous != entry:
            raise ValueError(f"{path}:{line_number}: conflicting pending update")
    return changed


def _clear_pending_updates(output: Path) -> None:
    path = _pending_updates_path(output)
    if path.exists():
        path.unlink()


def _completed_since_checkpoint(state: dict[str, Any], checkpoint_every: int) -> int:
    """Keep every compaction aligned to the total number of completed roots."""
    return (len(state["results"]) + len(state["rejections"])) % checkpoint_every


def load_uncovered_roots(coverage_path: Path) -> list[str]:
    try:
        report = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read coverage report {coverage_path}: {error}") from error
    if report.get("schema") not in {REPORT_SCHEMA, CORPUS_REPORT_SCHEMA}:
        raise ValueError(f"{coverage_path}: unexpected coverage report schema")
    if report.get("root_discs") != 14:
        raise ValueError(f"{coverage_path}: expected 14-disc roots")
    roots = report.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"{coverage_path}: roots is missing or empty")
    result: set[str] = set()
    for index, entry in enumerate(roots):
        if not isinstance(entry, dict) or not isinstance(entry.get("canonical_board"), str):
            raise ValueError(f"{coverage_path}: invalid root entry {index}")
        if entry.get("deep_book") or entry.get("root_table"):
            continue
        board = entry["canonical_board"]
        if Board.from_text(board).n_discs() != 14:
            raise ValueError(f"{coverage_path}: root entry {index} is not a 14-disc board")
        result.add(board)
    if not result:
        raise ValueError(f"{coverage_path}: no uncovered 14-disc roots")
    return sorted(result)


def load_excluded_roots(paths: list[Path]) -> set[str]:
    """Read canonical 14-disc positions that must not be selected again.

    Accepted teacher output and the published table share a data-row format.
    ``load_root_rows`` also verifies that each stored move is legal before the
    associated starting position is excluded from a later teacher run.
    """
    excluded: set[str] = set()
    for path in paths:
        excluded.update(entry.board for entry in load_root_rows(path, 14))
    return excluded


def root_file_provenance(paths: list[Path]) -> list[dict[str, str]]:
    return [
        {"path": path.resolve().as_posix(), "sha256": sha256_file(path)}
        for path in paths
    ]


def select_teacher_roots(
    roots: list[str], limit: int | None, cohort_seed: int | None
) -> list[str]:
    """Freeze a bounded cohort without relying on source-file ordering."""
    selected = sorted(set(roots))
    if cohort_seed is not None:
        selected = sorted(
            selected,
            key=lambda board: (
                hashlib.sha256(f"{cohort_seed}\0{board}".encode("ascii")).digest(),
                board,
            ),
        )
    if limit is not None:
        selected = selected[:limit]
    return selected


def parse_search_result(output: str, board: str) -> dict[str, int | str]:
    rows = [match.groupdict() for match in RESULT_RE.finditer(output)]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one search-result row, found {len(rows)}")
    row = rows[0]
    move = str(row["move"])
    if coord_to_index(move) not in Board.from_text(board).legal_moves():
        raise ValueError(f"engine returned illegal teacher move {move}")
    return {
        "move": move,
        "score": int(str(row["score"])),
        "level": str(row["level"]).strip(),
        "depth": str(row["depth"]).strip(),
        "time": str(row["time"]).strip(),
        "nodes": int(str(row["nodes"])),
        "nps": int(str(row["nps"])),
    }


def search_root(
    exe: Path,
    board: str,
    time_seconds: float,
    threads: int,
    hash_level: int,
) -> dict[str, int | str]:
    if time_seconds <= 0 or threads <= 0 or not 0 <= hash_level <= 29:
        raise ValueError("invalid time, thread, or hash setting")
    command = [
        str(exe),
        "-time", f"{time_seconds:g}",
        "-t", str(threads),
        "-hash", str(hash_level),
        *BOOK_DISABLED_ARGUMENTS,
    ]
    commands = TIME_SEARCH_INPUT_TEMPLATE.format(board=board)
    try:
        completed = subprocess.run(
            command,
            cwd=exe.parent,
            input=commands,
            text=True,
            capture_output=True,
            timeout=max(60.0, time_seconds * 2.0 + 30.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"teacher search failed for {board}: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher engine exited {completed.returncode} for {board}: {completed.stderr[-400:]}"
        )
    combined_output = completed.stdout + "\n" + completed.stderr
    try:
        return parse_search_result(combined_output, board)
    except ValueError as error:
        raise RuntimeError(
            f"teacher result parsing failed for {board}: {error}; output={combined_output[-400:]}"
        ) from error


def search_root_at_level(
    exe: Path,
    board: str,
    level: int,
    threads: int,
    hash_level: int,
) -> dict[str, int | str]:
    if level < 1 or threads <= 0 or not 0 <= hash_level <= 29:
        raise ValueError("invalid level, thread, or hash setting")
    command = [
        str(exe),
        "-l", str(level),
        "-t", str(threads),
        "-hash", str(hash_level),
        *BOOK_DISABLED_ARGUMENTS,
    ]
    commands = LEVEL_SEARCH_INPUT_TEMPLATE.format(board=board)
    try:
        completed = subprocess.run(
            command,
            cwd=exe.parent,
            input=commands,
            text=True,
            capture_output=True,
            timeout=180.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"teacher fallback failed for {board}: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher fallback exited {completed.returncode} for {board}: {completed.stderr[-400:]}"
        )
    combined_output = completed.stdout + "\n" + completed.stderr
    try:
        return parse_search_result(combined_output, board)
    except ValueError as error:
        raise RuntimeError(
            f"teacher fallback parsing failed for {board}: {error}; output={combined_output[-400:]}"
        ) from error


def validate_quality(result: dict[str, int | str], min_depth: int, min_selectivity: int) -> None:
    if min_depth < 1 or not 1 <= min_selectivity <= 100:
        raise ValueError("invalid minimum teacher quality")
    depth = DEPTH_RE.fullmatch(str(result["depth"]))
    if depth is None:
        raise ValueError(f"invalid teacher depth {result['depth']!r}")
    actual_depth = int(depth.group("depth"))
    actual_selectivity = int(depth.group("selectivity"))
    if actual_depth < min_depth or actual_selectivity < min_selectivity:
        raise ValueError(
            f"teacher search quality {actual_depth}@{actual_selectivity}% is below "
            f"{min_depth}@{min_selectivity}%"
        )


def _new_state(
    output: Path,
    coverage_path: Path,
    exe: Path,
    roots: list[str],
    time_seconds: float,
    threads: int,
    hash_level: int,
    min_depth: int,
    min_selectivity: int,
    fallback_level: int,
    method: str,
    teacher_level: int,
    verify_level: int,
    cohort_seed: int | None,
    excluded_root_files: list[Path],
) -> dict[str, Any]:
    return {
        "schema": TEACHER_SCHEMA,
        "calculation_provenance": _new_calculation_provenance(output),
        "coverage": {
            "path": coverage_path.resolve().as_posix(),
            "sha256": sha256_file(coverage_path),
        },
        "engine": {
            "path": exe.resolve().as_posix(),
            "sha256": sha256_file(exe),
        },
        "time_seconds": time_seconds,
        "threads": threads,
        "hash_level": hash_level,
        "min_depth": min_depth,
        "min_selectivity": min_selectivity,
        "fallback_level": fallback_level,
        "method": method,
        "teacher_level": teacher_level,
        "verify_level": verify_level,
        "cohort_seed": cohort_seed,
        "excluded_root_files": root_file_provenance(excluded_root_files),
        "deep_tiebreak_level": DEEP_TIEBREAK_LEVEL,
        "roots": roots,
        "results": {},
        "rejections": {},
    }


def _load_state(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume teacher state {path}: {error}") from error
    for key in (
        "schema", "calculation_provenance", "coverage", "engine", "time_seconds", "threads", "hash_level",
        "min_depth", "min_selectivity", "fallback_level", "method", "teacher_level",
        "verify_level", "cohort_seed", "excluded_root_files", "deep_tiebreak_level", "roots",
    ):
        if state.get(key) != expected.get(key):
            raise ValueError(f"{path}: resume mismatch for {key}")
    for field in ("results", "rejections"):
        if not isinstance(state.get(field), dict):
            raise ValueError(f"{path}: {field} is invalid")
        if not set(state[field]).issubset(set(expected["roots"])):
            raise ValueError(f"{path}: {field} includes an unexpected root")
    if set(state["results"]) & set(state["rejections"]):
        raise ValueError(f"{path}: a root is both accepted and rejected")
    return state


def _write_outputs(output: Path, state: dict[str, Any]) -> None:
    _ensure_teacher_script_snapshot(output, state)
    roots = state["roots"]
    results = state["results"]
    rejections = state["rejections"]
    provenance = validate_calculation_provenance(state["calculation_provenance"])
    book_configuration = provenance["book_configuration"]
    teacher_script = provenance["teacher_script"]
    teacher_script_snapshot = provenance["teacher_script_snapshot"]
    rows = []
    for board in roots:
        result = results.get(board)
        if result is not None:
            rows.append(f"{board} {result['score']} {result['move']}:{result['score']}")
    teacher_text = "\n".join(
        [
            TEACHER_FORMAT,
            f"# coverage_sha256 {state['coverage']['sha256']}",
            f"# engine_sha256 {state['engine']['sha256']}",
            f"# calculation_provenance_schema {provenance['schema']}",
            f"# ordinary_book_disabled {str(book_configuration['ordinary_book']['disabled']).lower()}",
            f"# ordinary_book_option {book_configuration['ordinary_book']['command_line_option']}",
            f"# contest_book_disabled {str(book_configuration['contest_book']['disabled']).lower()}",
            f"# contest_book_option {book_configuration['contest_book']['command_line_option']}",
            f"# teacher_script_sha256 {teacher_script['sha256']}",
            f"# teacher_script_snapshot_sha256 {teacher_script_snapshot['sha256']}",
            f"# time_seconds {state['time_seconds']:g}",
            f"# threads {state['threads']}",
            f"# hash_level {state['hash_level']}",
            f"# min_depth {state['min_depth']}",
            f"# min_selectivity {state['min_selectivity']}",
            f"# fallback_level {state['fallback_level']}",
            f"# method {state['method']}",
            f"# teacher_level {state['teacher_level']}",
            f"# verify_level {state['verify_level']}",
            f"# cohort_seed {state['cohort_seed']}",
            f"# excluded_root_files {len(state['excluded_root_files'])}",
            f"# deep_tiebreak_level {state['deep_tiebreak_level']}",
            f"# accepted {len(rows)}/{len(roots)}",
            f"# rejected {len(rejections)}/{len(roots)}",
            f"# processed {len(rows) + len(rejections)}/{len(roots)}",
            *rows,
            "",
        ]
    )
    _atomic_write_text(output, teacher_text)
    _atomic_write_text(
        _state_path(output),
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    manifest = {
        "schema": TEACHER_MANIFEST_SCHEMA,
        "output": {
            "path": output.resolve().as_posix(),
            "sha256": sha256_file(output),
            "completed": len(rows),
            "rejected": len(rejections),
            "processed": len(rows) + len(rejections),
            "requested": len(roots),
        },
        "coverage": state["coverage"],
        "engine": state["engine"],
        "calculation_provenance": provenance,
        "time_seconds": state["time_seconds"],
        "threads": state["threads"],
        "hash_level": state["hash_level"],
        "min_depth": state["min_depth"],
        "min_selectivity": state["min_selectivity"],
        "fallback_level": state["fallback_level"],
        "method": state["method"],
        "teacher_level": state["teacher_level"],
        "verify_level": state["verify_level"],
        "cohort_seed": state["cohort_seed"],
        "excluded_root_files": state["excluded_root_files"],
        "deep_tiebreak_level": state["deep_tiebreak_level"],
        "results": {board: results[board] for board in sorted(results)},
        "rejections": {board: rejections[board] for board in sorted(rejections)},
    }
    _atomic_write_text(
        _manifest_path(output),
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _generate_teachers_unlocked(
    coverage_path: Path,
    exe: Path,
    output: Path,
    time_seconds: float,
    threads: int,
    hash_level: int,
    min_depth: int = 33,
    min_selectivity: int = 74,
    fallback_level: int = 33,
    method: str = "hint",
    teacher_level: int = 33,
    verify_level: int = 0,
    resume: bool = False,
    limit: int | None = None,
    cohort_seed: int | None = None,
    excluded_root_files: list[Path] | None = None,
    checkpoint_every: int = 1,
    compact_only: bool = False,
) -> dict[str, int]:
    if not exe.is_file():
        raise FileNotFoundError(f"engine executable not found: {exe}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if compact_only and not resume:
        raise ValueError("compact_only requires resume")
    if fallback_level < 0:
        raise ValueError("fallback_level must not be negative")
    if method not in {"hint", "time_then_hint", "time_then_verify", "hint_then_verify"}:
        raise ValueError(
            "method must be hint, time_then_hint, time_then_verify, or hint_then_verify"
        )
    if teacher_level < 1:
        raise ValueError("teacher_level must be positive")
    if verify_level < 0:
        raise ValueError("verify_level must not be negative")
    if method in {"time_then_verify", "hint_then_verify"} and verify_level < 1:
        raise ValueError(f"{method} requires a positive verify_level")
    if method == "time_then_verify" and fallback_level < min_depth:
        raise ValueError(
            "time_then_verify requires fallback_level at least min_depth "
            "so a shallow time search cannot lower teacher quality"
        )
    if method == "hint_then_verify" and teacher_level < min_depth:
        raise ValueError(
            "hint_then_verify requires teacher_level at least min_depth"
        )
    if excluded_root_files is None:
        excluded_root_files = []
    excluded_root_files = sorted(
        {path.resolve() for path in excluded_root_files}, key=lambda path: path.as_posix()
    )
    excluded_roots = load_excluded_roots(excluded_root_files)
    roots = select_teacher_roots(
        [board for board in load_uncovered_roots(coverage_path) if board not in excluded_roots],
        limit,
        cohort_seed,
    )
    if not roots:
        raise ValueError("no uncovered 14-disc roots remain after exclusions")
    expected = _new_state(
        output, coverage_path, exe, roots, time_seconds, threads, hash_level, min_depth, min_selectivity,
        fallback_level, method, teacher_level, verify_level, cohort_seed, excluded_root_files,
    )
    state_path = _state_path(output)
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state not found: {state_path}")
        state = _load_state(state_path, expected)
        if _pending_updates_path(output).exists():
            _apply_pending_updates(output, state)
            _write_outputs(output, state)
            _clear_pending_updates(output)
    else:
        if state_path.exists() or output.exists():
            raise FileExistsError(f"output exists; use --resume or choose a new output: {output}")
        state = expected
        _write_outputs(output, state)

    completed_since_checkpoint = _completed_since_checkpoint(state, checkpoint_every)
    if compact_only:
        return {"completed": len(state["results"]), "requested": len(roots)}
    for board in roots:
        if board in state["results"] or board in state["rejections"]:
            continue
        if method == "hint":
            result = search_root_at_level(exe, board, teacher_level, threads, hash_level)
            validate_quality(result, max(min_depth, teacher_level), min_selectivity)
            result["method"] = f"hint_level_{teacher_level}"
        elif method == "time_then_hint":
            result = search_root(exe, board, time_seconds, threads, hash_level)
            try:
                validate_quality(result, min_depth, min_selectivity)
                result["method"] = "time"
            except ValueError:
                if fallback_level == 0:
                    raise
                result = search_root_at_level(exe, board, fallback_level, threads, hash_level)
                validate_quality(result, max(min_depth, fallback_level), min_selectivity)
                result["method"] = f"hint_level_{fallback_level}"
        elif method == "hint_then_verify":
            result = search_root_at_level(exe, board, teacher_level, threads, hash_level)
            validate_quality(result, max(min_depth, teacher_level), min_selectivity)
            result["method"] = (
                f"hint_level_{teacher_level}_verified_hint_level_{verify_level}"
            )
            verification = search_root_at_level(
                exe, board, verify_level, threads, hash_level
            )
            validate_quality(verification, max(min_depth, verify_level), min_selectivity)
            if str(result["move"]) != str(verification["move"]):
                verification_repeat = search_root_at_level(
                    exe, board, verify_level, threads, hash_level
                )
                validate_quality(
                    verification_repeat, max(min_depth, verify_level), min_selectivity
                )
                if str(verification["move"]) == str(verification_repeat["move"]):
                    teacher = result
                    result = dict(verification_repeat)
                    result["method"] = (
                        f"hint_level_{teacher_level}_overridden_by_repeated_"
                        f"hint_level_{verify_level}"
                    )
                    result["teacher"] = teacher
                    result["verification"] = verification
                    result["verification_repeat"] = verification_repeat
                    result["verification_mode"] = (
                        f"level_{verify_level}_repeated_after_disagreement"
                    )
                else:
                    state["rejections"][board] = {
                        "reason": (
                            f"level-{verify_level} verification {verification['move']} does not "
                            f"match repeated level-{verify_level} verification "
                            f"{verification_repeat['move']}"
                        ),
                        "teacher": result,
                        "verification": verification,
                        "verification_repeat": verification_repeat,
                    }
                    _append_pending_update(
                        output, board, "rejections", state["rejections"][board]
                    )
                    completed_since_checkpoint += 1
                    if completed_since_checkpoint >= checkpoint_every:
                        _write_outputs(output, state)
                        _clear_pending_updates(output)
                        completed_since_checkpoint = 0
                    print(
                        f"rejected {len(state['rejections'])} {board}",
                        flush=True,
                    )
                    continue
            else:
                result["verification"] = verification
                result["verification_mode"] = f"level_{verify_level}_exact"
        else:
            result = search_root(exe, board, time_seconds, threads, hash_level)
            primary = result
            try:
                validate_quality(result, min_depth, min_selectivity)
                result["method"] = f"time_verified_hint_level_{verify_level}"
            except ValueError:
                result = search_root_at_level(exe, board, fallback_level, threads, hash_level)
                validate_quality(result, max(min_depth, fallback_level), min_selectivity)
                result["method"] = (
                    f"time_fallback_hint_level_{fallback_level}_verified_hint_level_{verify_level}"
                )
                result["primary"] = primary
            verification = search_root_at_level(exe, board, verify_level, threads, hash_level)
            validate_quality(verification, max(min_depth, verify_level), min_selectivity)
            verification_mode = f"level_{verify_level}_exact"
            if str(result["move"]) != str(verification["move"]):
                tiebreak = search_root_at_level(
                    exe, board, fallback_level, threads, hash_level
                )
                validate_quality(tiebreak, max(min_depth, fallback_level), min_selectivity)
                if str(result["move"]) != str(tiebreak["move"]):
                    deep_tiebreak = search_root_at_level(
                        exe, board, DEEP_TIEBREAK_LEVEL, threads, hash_level
                    )
                    validate_quality(
                        deep_tiebreak,
                        max(min_depth, DEEP_TIEBREAK_LEVEL),
                        min_selectivity,
                    )
                    if str(tiebreak["move"]) != str(deep_tiebreak["move"]):
                        state["rejections"][board] = {
                            "reason": (
                                f"level-{fallback_level} tiebreak {tiebreak['move']} does not "
                                f"match level-{DEEP_TIEBREAK_LEVEL} tiebreak {deep_tiebreak['move']}"
                            ),
                            "teacher": result,
                            "verification": verification,
                            "tiebreak": tiebreak,
                            "deep_tiebreak": deep_tiebreak,
                        }
                        _append_pending_update(output, board, "rejections", state["rejections"][board])
                        completed_since_checkpoint += 1
                        if completed_since_checkpoint >= checkpoint_every:
                            _write_outputs(output, state)
                            _clear_pending_updates(output)
                            completed_since_checkpoint = 0
                        print(
                            f"rejected {len(state['rejections'])} {board}",
                            flush=True,
                        )
                        continue
                    deep_tiebreak["method"] = (
                        f"time_disagreement_tiebreak_levels_{fallback_level}_{DEEP_TIEBREAK_LEVEL}"
                    )
                    deep_tiebreak["primary"] = result
                    deep_tiebreak["tiebreak"] = tiebreak
                    result = deep_tiebreak
                    verification_mode = (
                        f"levels_{fallback_level}_{DEEP_TIEBREAK_LEVEL}_tiebreak"
                    )
                else:
                    result["tiebreak"] = tiebreak
                    verification_mode = f"level_{fallback_level}_tiebreak"
            result["verification"] = verification
            result["verification_mode"] = verification_mode
        state["results"][board] = result
        _append_pending_update(output, board, "results", result)
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_every:
            _write_outputs(output, state)
            _clear_pending_updates(output)
            completed_since_checkpoint = 0
        print(f"completed {len(state['results'])}/{len(roots)} {board}", flush=True)
    if completed_since_checkpoint:
        _write_outputs(output, state)
        _clear_pending_updates(output)
    return {"completed": len(state["results"]), "requested": len(roots)}


def generate_teachers(
    coverage_path: Path,
    exe: Path,
    output: Path,
    time_seconds: float,
    threads: int,
    hash_level: int,
    min_depth: int = 33,
    min_selectivity: int = 74,
    fallback_level: int = 33,
    method: str = "hint",
    teacher_level: int = 33,
    verify_level: int = 0,
    resume: bool = False,
    limit: int | None = None,
    cohort_seed: int | None = None,
    excluded_root_files: list[Path] | None = None,
    checkpoint_every: int = 1,
    compact_only: bool = False,
) -> dict[str, int]:
    """Generate one output while holding its OS-owned exclusive lock."""
    with file_lock(_lock_path(output)):
        return _generate_teachers_unlocked(
            coverage_path,
            exe,
            output,
            time_seconds,
            threads,
            hash_level,
            min_depth,
            min_selectivity,
            fallback_level,
            method,
            teacher_level,
            verify_level,
            resume,
            limit,
            cohort_seed,
            excluded_root_files,
            checkpoint_every,
            compact_only,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-seconds", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--hash", dest="hash_level", type=int, default=29)
    parser.add_argument("--min-depth", type=int, default=33)
    parser.add_argument("--min-selectivity", type=int, default=74)
    parser.add_argument("--fallback-level", type=int, default=33)
    parser.add_argument(
        "--method",
        choices=("hint", "time_then_hint", "time_then_verify", "hint_then_verify"),
        default="hint",
    )
    parser.add_argument("--teacher-level", type=int, default=33)
    parser.add_argument("--verify-level", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cohort-seed",
        type=int,
        help="Hash-sort uncovered roots by this fixed seed before applying --limit",
    )
    parser.add_argument(
        "--exclude-root-results",
        type=Path,
        action="append",
        default=[],
        help="Teacher rows or a published table whose positions are excluded (repeatable)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Compact durable per-position updates after this many completed positions",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="With --resume, replay durable updates and publish them without starting another search",
    )
    args = parser.parse_args()
    result = generate_teachers(
        args.coverage,
        args.exe.resolve(),
        args.output,
        args.time_seconds,
        args.threads,
        args.hash_level,
        args.min_depth,
        args.min_selectivity,
        args.fallback_level,
        args.method,
        args.teacher_level,
        args.verify_level,
        args.resume,
        args.limit,
        args.cohort_seed,
        args.exclude_root_results,
        args.checkpoint_every,
        args.compact_only,
    )
    print(f"teacher roots complete {result['completed']}/{result['requested']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
