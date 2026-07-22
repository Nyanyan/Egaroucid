"""Generate resumable, contest-time root teachers for uncovered r14 starts.

Input is an immutable JSON report produced either by ``collect_ggs_roots.py``
from actual GGS starts or by ``audit_r14_corpus.py`` from the complete standard
r14 corpus. Every teacher search disables both ordinary and contest books and
uses one engine process per root. A durable per-position journal is written
after each processed root, whether accepted or rejected; output files are
compacted at the selected interval. The output data rows are accepted directly by ``build_root_table.py
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
from r14_random_setup_probability import (
    R14_RANDOM_SETUP_PROBABILITY_MODEL,
    local_probability_model_source_fingerprints,
    load_r14_random_setup_priority_manifest,
    order_r14_random_setup_boards,
)


TEACHER_SCHEMA = "ggs_root_teacher_state_v17"
TEACHER_MANIFEST_SCHEMA = "ggs_root_teacher_manifest_v17"
TEACHER_FORMAT = "# ggs_root_teacher_v1"
TEACHER_UPDATE_SCHEMA = "ggs_root_teacher_update_v1"
CALCULATION_PROVENANCE_SCHEMA = "ggs_root_teacher_calculation_provenance_v3"
EXECUTION_ENVIRONMENT_SCHEMA = "ggs_root_teacher_execution_environment_v1"
FORMAL_COMPARISON_REPORT_SCHEMA = "formal_root_teacher_method_comparison_report_v1"
FORMAL_COMPARISON_STATE_SCHEMA = "formal_root_teacher_method_comparison_state_v1"
FORMAL_SELECTED_METHOD = "hint_then_verify"
FORMAL_SELECTED_TEACHER_LEVEL = 30
FORMAL_SELECTED_VERIFY_LEVEL = 31
FORMAL_SELECTED_THREADS = 28
FORMAL_SELECTED_HASH_LEVEL = 29
FORMAL_SELECTED_MIN_DEPTH = 30
FORMAL_SELECTED_MIN_SELECTIVITY = 74
DEEP_TIEBREAK_LEVEL = 31
BOOK_DISABLED_ARGUMENTS = ("-nobook", "-nocontestbook")
RESOURCE_SPECS = (
    ("evaluation", Path("eval.egev2")),
    ("endgame_move_ordering", Path("eval_move_ordering_end.egev")),
)
ROOT_ORDER_HASH = "hash"
ROOT_ORDER_GGS_R14_PROBABILITY = "ggs-r14-probability"
ROOT_ORDER_CHOICES = (ROOT_ORDER_HASH, ROOT_ORDER_GGS_R14_PROBABILITY)
TIME_SEARCH_COMMAND_TEMPLATE = [
    "{executable}",
    "-time",
    "{time_seconds}",
    "-t",
    "{threads}",
    "-hash",
    "{hash_level}",
    "-seed",
    "{random_seed}",
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
    "-seed",
    "{random_seed}",
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


def _calculation_input_cache_directory(output: Path) -> Path:
    """Return the shared cache for immutable Console inputs for one output directory."""
    return output.parent / ".teacher_calculation_inputs"


def _fingerprint(path: Path, description: str) -> dict[str, int | str]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return {
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _load_formal_comparison_selection(
    report_path: Path,
    coverage_path: Path,
    exe: Path,
    *,
    method: str,
    teacher_level: int,
    verify_level: int,
    threads: int,
    hash_level: int,
    min_depth: int,
    min_selectivity: int,
) -> dict[str, object]:
    """Verify that a completed formal comparison selected this exact setup.

    The comparison program imports this module, so this deliberately reads its
    persisted JSON rather than importing that program back here.
    """
    report_fingerprint = _fingerprint(report_path, "formal comparison report")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read formal comparison report {report_path}: {error}") from error
    if not isinstance(report, dict) or report.get("schema") != FORMAL_COMPARISON_REPORT_SCHEMA:
        raise ValueError("formal comparison report has an unsupported schema")
    expected_state_sha = report.get("experiment_state_sha256")
    if not isinstance(expected_state_sha, str):
        raise ValueError("formal comparison report has no experiment-state SHA-256")
    state_path = report_path.parent / "experiment_state.json"
    state_fingerprint = _fingerprint(state_path, "formal comparison state")
    if state_fingerprint["sha256"] != expected_state_sha:
        raise ValueError("formal comparison state does not match the report SHA-256")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read formal comparison state {state_path}: {error}") from error
    if not isinstance(state, dict) or state.get("schema") != FORMAL_COMPARISON_STATE_SCHEMA:
        raise ValueError("formal comparison state has an unsupported schema")
    rule = report.get("decision_protocol")
    conditions = report.get("decision_conditions")
    if not isinstance(rule, dict) or not isinstance(conditions, dict):
        raise ValueError("formal comparison report has no decision rule or conditions")
    required_pairs = rule.get("required_complete_pairs")
    if (
        rule.get("candidate_method") != FORMAL_SELECTED_METHOD
        or rule.get("reference_method") != "time_then_verify"
        or isinstance(required_pairs, bool)
        or not isinstance(required_pairs, int)
        or report.get("completed_pairs") != required_pairs
        or report.get("level_30_then_level_31_can_continue_to_larger_calculation") is not True
        or not conditions
        or any(value is not True for value in conditions.values())
    ):
        raise ValueError("formal comparison did not select the level-30/level-31 method")
    expected_setup = {
        "method": FORMAL_SELECTED_METHOD,
        "teacher_level": FORMAL_SELECTED_TEACHER_LEVEL,
        "verify_level": FORMAL_SELECTED_VERIFY_LEVEL,
        "threads": FORMAL_SELECTED_THREADS,
        "hash_level": FORMAL_SELECTED_HASH_LEVEL,
        "min_depth": FORMAL_SELECTED_MIN_DEPTH,
        "min_selectivity": FORMAL_SELECTED_MIN_SELECTIVITY,
    }
    actual_setup = {
        "method": method,
        "teacher_level": teacher_level,
        "verify_level": verify_level,
        "threads": threads,
        "hash_level": hash_level,
        "min_depth": min_depth,
        "min_selectivity": min_selectivity,
    }
    if actual_setup != expected_setup:
        raise ValueError(
            "teacher settings do not match the method selected by the formal comparison"
        )
    coverage = state.get("coverage")
    environment = state.get("execution_environment")
    if not isinstance(coverage, dict) or not isinstance(environment, dict):
        raise ValueError("formal comparison state has incomplete input evidence")
    if coverage.get("sha256") != sha256_file(coverage_path):
        raise ValueError("teacher coverage differs from the formal comparison coverage")
    executable = environment.get("executable")
    resources = environment.get("resources")
    if not isinstance(executable, dict) or not isinstance(resources, list):
        raise ValueError("formal comparison state has incomplete Console evidence")
    formal_exe = executable.get("source")
    current_exe = _fingerprint(exe, "teacher executable")
    if not isinstance(formal_exe, dict) or formal_exe.get("sha256") != current_exe["sha256"]:
        raise ValueError("teacher executable differs from the formal comparison executable")
    formal_resources = {item.get("role"): item.get("source") for item in resources if isinstance(item, dict)}
    for role, relative in RESOURCE_SPECS:
        expected_resource = formal_resources.get(role)
        actual_resource = _fingerprint(exe.parent / "resources" / relative, role)
        if not isinstance(expected_resource, dict) or expected_resource.get("sha256") != actual_resource["sha256"]:
            raise ValueError(f"teacher {role} differs from the formal comparison input")
    return {
        "report": report_fingerprint,
        "experiment_state": state_fingerprint,
        "selected_method": FORMAL_SELECTED_METHOD,
        "selected_setup": expected_setup,
    }


def _environment_identifier(
    executable: dict[str, int | str], resources: list[dict[str, Any]]
) -> str:
    payload = {
        "executable_name": Path(str(executable["path"])).name,
        "executable_sha256": executable["sha256"],
        "resources": [
            {
                "role": resource["role"],
                "relative_path": resource["relative_path"],
                "sha256": resource["source"]["sha256"],
            }
            for resource in resources
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _execution_environment_directory(
    output: Path,
    executable: dict[str, int | str],
    resources: list[dict[str, Any]],
) -> Path:
    return _calculation_input_cache_directory(output) / f"console-{_environment_identifier(executable, resources)}"


def _execution_environment_lock_path(directory: Path) -> Path:
    return directory.parent / f".{directory.name}.lock"


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


def _search_invocation_contract() -> dict[str, Any]:
    """Return the one command/input contract used both for execution and evidence."""
    return {
        "command_templates": _command_templates(),
        "standard_input_templates": _standard_input_templates(),
    }


def _validate_random_seed(random_seed: object) -> int:
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    if not 0 <= random_seed <= 0xFFFFFFFF:
        raise ValueError("random_seed must be between zero and 4294967295")
    return random_seed


def _build_search_command(
    search_kind: str,
    exe: Path,
    *,
    time_seconds: float | None = None,
    level: int | None = None,
    threads: int,
    hash_level: int,
    random_seed: int = 620,
) -> list[str]:
    """Instantiate the same command template that is saved in provenance."""
    random_seed = _validate_random_seed(random_seed)
    template = _search_invocation_contract()["command_templates"].get(search_kind)
    if template is None:
        raise ValueError(f"unknown search kind {search_kind}")
    values = {
        "executable": str(exe),
        "time_seconds": "" if time_seconds is None else f"{time_seconds:g}",
        "level": "" if level is None else str(level),
        "threads": str(threads),
        "hash_level": str(hash_level),
        "random_seed": str(random_seed),
    }
    return [part.format(**values) for part in template]


def _validate_fingerprint_metadata(fingerprint: object, label: str) -> dict[str, int | str]:
    if not isinstance(fingerprint, dict):
        raise ValueError(f"teacher calculation provenance has no {label} fingerprint")
    path = fingerprint.get("path")
    digest = fingerprint.get("sha256")
    size = fingerprint.get("bytes")
    if not isinstance(path, str) or not path:
        raise ValueError(f"teacher calculation provenance has no {label} path")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"teacher calculation provenance has an invalid {label} SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"teacher calculation provenance has an invalid {label} byte count")
    return {"path": path, "sha256": digest, "bytes": size}


def _execution_environment_metadata(
    provenance: dict[str, Any],
    output: Path | None = None,
) -> tuple[
    Path,
    dict[str, int | str],
    dict[str, int | str],
    list[tuple[str, Path, dict[str, int | str], dict[str, int | str]]],
]:
    """Validate paths and fingerprints before copying any saved Console input.

    A resumed state is durable input, not trusted write instructions.  In
    particular, every source file must keep Console's documented layout and
    every snapshot path must stay below the cache directory derived from its
    content fingerprints.
    """
    environment = provenance.get("execution_environment")
    if not isinstance(environment, dict) or environment.get("schema") != EXECUTION_ENVIRONMENT_SCHEMA:
        raise ValueError("teacher calculation provenance has no supported execution environment")
    directory = environment.get("directory")
    if not isinstance(directory, str) or not directory:
        raise ValueError("teacher calculation provenance has no execution-environment directory")
    recorded_root = Path(directory).resolve()
    executable = environment.get("executable")
    if not isinstance(executable, dict):
        raise ValueError("teacher calculation provenance has no execution-environment executable")
    source_executable = _validate_fingerprint_metadata(
        executable.get("source"), "source executable"
    )
    snapshot_executable = _validate_fingerprint_metadata(
        executable.get("snapshot"), "saved executable"
    )
    if (
        source_executable["sha256"] != snapshot_executable["sha256"]
        or source_executable["bytes"] != snapshot_executable["bytes"]
    ):
        raise ValueError("source executable and its saved copy differ")
    expected_executable = recorded_root / Path(str(source_executable["path"])).name
    if Path(str(snapshot_executable["path"])).resolve() != expected_executable.resolve():
        raise ValueError("teacher calculation provenance has an unexpected saved executable path")

    resources = environment.get("resources")
    if not isinstance(resources, list) or len(resources) != len(RESOURCE_SPECS):
        raise ValueError("teacher calculation provenance has an invalid resource list")
    records: list[tuple[str, Path, dict[str, int | str], dict[str, int | str]]] = []
    source_resources: list[dict[str, Any]] = []
    executable_resources = Path(str(source_executable["path"])).resolve().parent / "resources"
    for resource, (expected_role, expected_relative) in zip(resources, RESOURCE_SPECS):
        if not isinstance(resource, dict) or resource.get("role") != expected_role:
            raise ValueError("teacher calculation provenance has resources in an unexpected order")
        if resource.get("relative_path") != expected_relative.as_posix():
            raise ValueError("teacher calculation provenance has an unexpected Console resource")
        source = _validate_fingerprint_metadata(
            resource.get("source"), f"source {expected_role} resource"
        )
        saved = _validate_fingerprint_metadata(
            resource.get("snapshot"), f"saved {expected_role} resource"
        )
        if source["sha256"] != saved["sha256"] or source["bytes"] != saved["bytes"]:
            raise ValueError(f"source and saved {expected_role} resources differ")
        expected_source = executable_resources / expected_relative
        expected_saved = recorded_root / "resources" / expected_relative
        if Path(str(source["path"])).resolve() != expected_source.resolve():
            raise ValueError(f"teacher calculation provenance has an unexpected source {expected_role} path")
        if Path(str(saved["path"])).resolve() != expected_saved.resolve():
            raise ValueError(f"teacher calculation provenance has an unexpected saved {expected_role} path")
        records.append((expected_role, expected_relative, source, saved))
        source_resources.append(
            {
                "role": expected_role,
                "relative_path": expected_relative.as_posix(),
                "source": source,
            }
        )
    if output is not None:
        expected_root = _execution_environment_directory(
            output, source_executable, source_resources
        ).resolve()
        if recorded_root != expected_root:
            raise ValueError("teacher calculation provenance has an unexpected execution-environment directory")
    return recorded_root, source_executable, snapshot_executable, records


def _relative_snapshot_path(
    recorded_root: Path,
    recorded_path: Path,
    actual_root: Path,
    label: str,
) -> Path:
    try:
        relative = recorded_path.resolve().relative_to(recorded_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} is outside the saved execution environment") from error
    return actual_root.resolve() / relative


def _verify_snapshot_file(
    fingerprint: dict[str, int | str], path: Path, label: str
) -> None:
    if not path.is_file() or sha256_file(path) != fingerprint["sha256"]:
        raise ValueError(f"saved {label} does not match its SHA-256")
    if path.stat().st_size != fingerprint["bytes"]:
        raise ValueError(f"saved {label} does not match its byte count")


def _new_calculation_provenance(
    output: Path,
    exe: Path,
    hash_level: int,
    random_seed: int = 620,
) -> dict[str, Any]:
    """Freeze the executable, Console inputs, book options, and generator source."""
    if not 0 <= hash_level <= 29:
        raise ValueError("hash_level must be between zero and 29")
    random_seed = _validate_random_seed(random_seed)
    teacher_script = Path(__file__).resolve()
    teacher_script_sha256 = sha256_file(teacher_script)
    teacher_script_snapshot = _teacher_script_snapshot_path(output).resolve()
    executable = _fingerprint(exe, "teacher executable")
    source_resources: list[dict[str, Any]] = []
    resources_directory = exe.resolve().parent / "resources"
    for role, relative_path in RESOURCE_SPECS:
        source = _fingerprint(resources_directory / relative_path, f"teacher {role} resource")
        source_resources.append(
            {
                "role": role,
                "relative_path": relative_path.as_posix(),
                "source": source,
            }
        )
    environment_directory = _execution_environment_directory(output, executable, source_resources).resolve()
    resources: list[dict[str, Any]] = []
    for resource in source_resources:
        relative_path = Path(str(resource["relative_path"]))
        source = resource["source"]
        resources.append(
            {
                "role": resource["role"],
                "relative_path": resource["relative_path"],
                "source": source,
                "snapshot": {
                    "path": (environment_directory / "resources" / relative_path).as_posix(),
                    "sha256": source["sha256"],
                    "bytes": source["bytes"],
                },
            }
        )
    executable_snapshot = {
        "path": (environment_directory / Path(str(executable["path"])).name).as_posix(),
        "sha256": executable["sha256"],
        "bytes": executable["bytes"],
    }
    return {
        "schema": CALCULATION_PROVENANCE_SCHEMA,
        "book_configuration": _book_configuration(),
        "random_seed": random_seed,
        "search_invocation": _search_invocation_contract(),
        "teacher_script": {
            "path": teacher_script.as_posix(),
            "sha256": teacher_script_sha256,
        },
        "teacher_script_snapshot": {
            "path": teacher_script_snapshot.as_posix(),
            "sha256": teacher_script_sha256,
        },
        "execution_environment": {
            "schema": EXECUTION_ENVIRONMENT_SCHEMA,
            "directory": environment_directory.as_posix(),
            "executable": {
                "source": executable,
                "snapshot": executable_snapshot,
            },
            "resources": resources,
        },
    }


def validate_calculation_provenance(
    provenance: object,
    snapshot_path: Path | None = None,
    execution_environment_directory: Path | None = None,
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
    _validate_random_seed(provenance.get("random_seed"))
    if provenance.get("search_invocation") != _search_invocation_contract():
        raise ValueError("teacher calculation provenance has an unexpected search-invocation contract")
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
    if (
        any(option not in script_text for option in BOOK_DISABLED_ARGUMENTS)
        or "_build_search_command" not in script_text
        or "_search_invocation_contract" not in script_text
    ):
        raise ValueError("saved teacher script does not contain the recorded book-disable command builder")

    recorded_root, _source_executable, snapshot_executable, resources = _execution_environment_metadata(
        provenance
    )
    actual_root = (
        execution_environment_directory
        if execution_environment_directory is not None
        else recorded_root
    )
    executable_path = _relative_snapshot_path(
        recorded_root,
        Path(str(snapshot_executable["path"])),
        actual_root,
        "saved executable",
    )
    _verify_snapshot_file(snapshot_executable, executable_path, "executable")

    for role, _relative_path, _source, saved in resources:
        saved_path = _relative_snapshot_path(
            recorded_root,
            Path(str(saved["path"])),
            actual_root,
            f"saved {role} resource",
        )
        _verify_snapshot_file(saved, saved_path, f"{role} resource")
    return provenance


def _copy_or_verify_saved_file(
    source: dict[str, int | str], snapshot: dict[str, int | str], label: str
) -> None:
    source_path = Path(str(source["path"]))
    snapshot_path = Path(str(snapshot["path"]))
    if snapshot_path.exists():
        _verify_snapshot_file(snapshot, snapshot_path, label)
        return
    try:
        content = source_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read source {label} {source_path}: {error}") from error
    if hashlib.sha256(content).hexdigest() != source["sha256"] or len(content) != source["bytes"]:
        raise ValueError(f"source {label} changed before its saved copy was created")
    _atomic_write_bytes(snapshot_path, content)
    _verify_snapshot_file(snapshot, snapshot_path, label)


def _ensure_teacher_script_snapshot(output: Path, state: dict[str, Any]) -> None:
    """Create once, then verify, the exact generator source beside an output."""
    provenance = state.get("calculation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("teacher state has no calculation provenance")
    teacher_script = provenance.get("teacher_script")
    snapshot = provenance.get("teacher_script_snapshot")
    if not isinstance(teacher_script, dict) or not isinstance(snapshot, dict):
        raise ValueError("teacher state has incomplete script provenance")
    expected_snapshot = _teacher_script_snapshot_path(output).resolve()
    if snapshot.get("path") != expected_snapshot.as_posix() or snapshot.get("sha256") != teacher_script["sha256"]:
        raise ValueError("teacher state has an unexpected saved-script location or SHA-256")
    if expected_snapshot.exists():
        if sha256_file(expected_snapshot) != snapshot["sha256"]:
            raise ValueError("saved teacher script does not match its SHA-256")
    else:
        source = Path(__file__).resolve()
        if teacher_script.get("path") != source.as_posix() or teacher_script.get("sha256") != sha256_file(source):
            raise ValueError("teacher state was created by a different generator script")
        _atomic_write_bytes(expected_snapshot, source.read_bytes())


def _ensure_execution_environment(output: Path, state: dict[str, Any]) -> Path:
    """Create once, then verify, the isolated executable and its two inputs."""
    provenance = state.get("calculation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("teacher state has no calculation provenance")
    root, source_executable, saved_executable, resources = _execution_environment_metadata(
        provenance, output
    )
    with file_lock(_execution_environment_lock_path(root)):
        _copy_or_verify_saved_file(source_executable, saved_executable, "executable")
        for role, _relative_path, source, saved in resources:
            _copy_or_verify_saved_file(source, saved, f"{role} resource")
    return Path(str(saved_executable["path"]))


def _ensure_calculation_snapshots(output: Path, state: dict[str, Any]) -> Path:
    _ensure_teacher_script_snapshot(output, state)
    executable = _ensure_execution_environment(output, state)
    validate_calculation_provenance(state.get("calculation_provenance"))
    return executable


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
    """Keep every compaction aligned to the total number of processed roots."""
    return (len(state["results"]) + len(state["rejections"])) % checkpoint_every


def _generation_status(state: dict[str, Any], roots: list[str]) -> dict[str, int]:
    """Return unambiguous accepted/rejected/processed counts for one invocation."""
    accepted = len(state["results"])
    rejected = len(state["rejections"])
    return {
        "accepted": accepted,
        "rejected": rejected,
        "processed": accepted + rejected,
        "requested": len(roots),
    }


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


def probability_model_source_header_lines(
    prefix: str, sources: dict[str, object] | None
) -> list[str]:
    """Render all frozen local probability-model source hashes in teacher output."""
    if sources is None:
        return [f"# {prefix}_schema -"]
    files = sources.get("files")
    if not isinstance(files, list):
        raise ValueError("probability-model source record has no files")
    lines = [f"# {prefix}_schema {sources['schema']}"]
    for source in files:
        if not isinstance(source, dict):
            raise ValueError("probability-model source record has an invalid file")
        lines.append(f"# {prefix}_{source['id']}_sha256 {source['sha256']}")
    return lines


def select_teacher_roots(
    roots: list[str],
    limit: int | None,
    cohort_seed: int | None,
    root_order: str = ROOT_ORDER_HASH,
    priority_tie_seed: int | None = None,
    priority_boards: list[str] | None = None,
) -> list[str]:
    """Freeze a bounded teacher input without source-file-order dependence.

    ``hash`` preserves the original uniform hashed order.  ``ggs-r14-probability``
    instead orders the ``random_setup(14)`` r14 corpus by the exact probability from the
    repository-local ``random_setup(14)`` reimplementation; it does not claim
    to identify the current GGS server source.  ``cohort_seed`` applies only
    to ``hash``.  The separately named ``priority_tie_seed`` applies only
    within an equal-probability group in ``ggs-r14-probability``.  A frozen
    priority file may contain roots already covered elsewhere: those extras
    are intentionally ignored, but every current teacher root must occur in
    the file.  The selected board list is saved in the state file and is the
    authoritative future resume input.
    """
    if root_order not in ROOT_ORDER_CHOICES:
        raise ValueError(f"root_order must be one of {ROOT_ORDER_CHOICES}")
    available = set(roots)
    if priority_boards is not None:
        if root_order != ROOT_ORDER_GGS_R14_PROBABILITY:
            raise ValueError("priority_boards requires root_order='ggs-r14-probability'")
        if cohort_seed is not None or priority_tie_seed is not None:
            raise ValueError("a frozen priority file cannot be combined with a selection seed")
        selected = [board for board in priority_boards if board in available]
        if set(selected) != available:
            raise ValueError(
                "the frozen priority file does not contain every teacher-population root"
            )
    elif root_order == ROOT_ORDER_GGS_R14_PROBABILITY:
        selected = sorted(available)
        if cohort_seed is not None:
            raise ValueError("cohort_seed is only valid with root_order='hash'")
        selected = order_r14_random_setup_boards(selected, priority_tie_seed)
    else:
        selected = sorted(available)
    if root_order == ROOT_ORDER_HASH and cohort_seed is not None:
        selected = sorted(
            selected,
            key=lambda board: (
                hashlib.sha256(f"{cohort_seed}\0{board}".encode("ascii")).digest(),
                board,
            ),
        )
    elif root_order == ROOT_ORDER_HASH and priority_tie_seed is not None:
        raise ValueError("priority_tie_seed requires root_order='ggs-r14-probability'")
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
    random_seed: int = 620,
) -> dict[str, int | str]:
    if time_seconds <= 0 or threads <= 0 or not 0 <= hash_level <= 29:
        raise ValueError("invalid time, thread, or hash setting")
    random_seed = _validate_random_seed(random_seed)
    command = _build_search_command(
        "time_limited_search",
        exe,
        time_seconds=time_seconds,
        threads=threads,
        hash_level=hash_level,
        random_seed=random_seed,
    )
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
    random_seed: int = 620,
) -> dict[str, int | str]:
    if level < 1 or threads <= 0 or not 0 <= hash_level <= 29:
        raise ValueError("invalid level, thread, or hash setting")
    random_seed = _validate_random_seed(random_seed)
    command = _build_search_command(
        "fixed_level_search",
        exe,
        level=level,
        threads=threads,
        hash_level=hash_level,
        random_seed=random_seed,
    )
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


def _validate_recorded_search_result(
    board: str,
    result: object,
    label: str,
    min_depth: int | None,
    min_selectivity: int,
    expected_level: int | None = None,
) -> dict[str, Any]:
    """Validate one search result embedded in an accepted teacher result."""
    if not isinstance(result, dict):
        raise ValueError(f"{label} is not a search-result object")
    move = result.get("move")
    score = result.get("score")
    if not isinstance(move, str) or not isinstance(score, int):
        raise ValueError(f"{label} has no legal move and integer score")
    try:
        move_index = coord_to_index(move)
    except ValueError as error:
        raise ValueError(f"{label} has an invalid move") from error
    if move_index not in Board.from_text(board).legal_moves():
        raise ValueError(f"{label} has an illegal move")
    if expected_level is not None and result.get("level") != str(expected_level):
        raise ValueError(
            f"{label} has level {result.get('level')!r}, expected {expected_level}"
        )
    if min_depth is not None:
        try:
            validate_quality(result, min_depth, min_selectivity)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label} does not meet its recorded quality") from error
    return result


def validate_verified_teacher_result(
    board: str,
    result: object,
    *,
    method: str,
    min_depth: int,
    min_selectivity: int,
    fallback_level: int,
    teacher_level: int,
    verify_level: int,
) -> None:
    """Require the complete verification trail for a formal teacher result.

    This validator is deliberately stricter than a row-format check.  It
    proves that every accepted move came from one of the two methods that
    actually performs a level verification, rather than merely carrying an
    unrelated ``verify_level`` field in the manifest.
    """
    if min_depth < 1 or not 1 <= min_selectivity <= 100:
        raise ValueError("formal teacher result has invalid minimum quality settings")
    if fallback_level < 0 or teacher_level < 1:
        raise ValueError("formal teacher result has invalid fixed-level settings")
    if method not in {"hint_then_verify", "time_then_verify"}:
        raise ValueError("formal teacher result uses a method without level verification")
    if verify_level < 1:
        raise ValueError("formal teacher result has no positive verification level")
    if method == "hint_then_verify" and teacher_level < min_depth:
        raise ValueError("formal hint teacher level is below the required quality")
    if method == "time_then_verify" and fallback_level < min_depth:
        raise ValueError("formal time fallback level is below the required quality")
    final = _validate_recorded_search_result(board, result, "accepted result", min_depth, min_selectivity)
    result_method = final.get("method")
    verification_mode = final.get("verification_mode")
    if not isinstance(result_method, str) or not isinstance(verification_mode, str):
        raise ValueError("accepted result has no method or verification mode")
    verification = _validate_recorded_search_result(
        board,
        final.get("verification"),
        "verification result",
        max(min_depth, verify_level),
        min_selectivity,
        verify_level,
    )

    if method == "hint_then_verify":
        exact_method = f"hint_level_{teacher_level}_verified_hint_level_{verify_level}"
        repeated_method = (
            f"hint_level_{teacher_level}_overridden_by_repeated_hint_level_{verify_level}"
        )
        if verification_mode == f"level_{verify_level}_exact":
            if result_method != exact_method:
                raise ValueError("exact hint verification has an unexpected selected-method record")
            _validate_recorded_search_result(
                board,
                final,
                "accepted initial hint result",
                max(min_depth, teacher_level),
                min_selectivity,
                teacher_level,
            )
            if final["move"] != verification["move"]:
                raise ValueError("exact hint verification does not match the accepted move")
            return
        if verification_mode != f"level_{verify_level}_repeated_after_disagreement":
            raise ValueError("hint verification has an unexpected verification mode")
        if result_method != repeated_method:
            raise ValueError("repeated hint verification has an unexpected selected-method record")
        teacher = _validate_recorded_search_result(
            board,
            final.get("teacher"),
            "initial hint result",
            max(min_depth, teacher_level),
            min_selectivity,
            teacher_level,
        )
        repeated = _validate_recorded_search_result(
            board,
            final.get("verification_repeat"),
            "repeated verification result",
            max(min_depth, verify_level),
            min_selectivity,
            verify_level,
        )
        _validate_recorded_search_result(
            board,
            final,
            "accepted repeated verification result",
            max(min_depth, verify_level),
            min_selectivity,
            verify_level,
        )
        if teacher["move"] == final["move"]:
            raise ValueError("repeated hint verification did not record the original disagreement")
        if verification["move"] != repeated["move"] or final["move"] != repeated["move"]:
            raise ValueError("repeated hint verification does not support the accepted move")
        return

    direct_method = f"time_verified_hint_level_{verify_level}"
    fallback_method = f"time_fallback_hint_level_{fallback_level}_verified_hint_level_{verify_level}"

    def validate_time_primary(value: dict[str, Any], label: str) -> None:
        """Validate the time-search or fallback result preceding verification."""
        value_method = value.get("method")
        if value_method == direct_method:
            _validate_recorded_search_result(
                board, value, label, min_depth, min_selectivity
            )
            if value.get("level") != "-":
                raise ValueError(f"{label} is not recorded as a time search")
            return
        if value_method == fallback_method:
            _validate_recorded_search_result(
                board,
                value,
                label,
                max(min_depth, fallback_level),
                min_selectivity,
                fallback_level,
            )
            time_result = _validate_recorded_search_result(
                board,
                value.get("primary"),
                f"{label} original time result",
                None,
                min_selectivity,
            )
            if time_result.get("level") != "-":
                raise ValueError(f"{label} original result is not recorded as a time search")
            return
        raise ValueError(f"{label} has an unexpected selected-method record")

    if verification_mode == f"level_{verify_level}_exact":
        validate_time_primary(final, "accepted time result")
        if final["move"] != verification["move"]:
            raise ValueError("exact time verification does not match the accepted move")
        return
    if verification_mode == f"level_{fallback_level}_tiebreak":
        validate_time_primary(final, "accepted time result before tiebreak")
        tiebreak = _validate_recorded_search_result(
            board,
            final.get("tiebreak"),
            "time tiebreak result",
            max(min_depth, fallback_level),
            min_selectivity,
            fallback_level,
        )
        if final["move"] != tiebreak["move"] or final["move"] == verification["move"]:
            raise ValueError("time tiebreak does not support the accepted move")
        return
    if verification_mode == f"levels_{fallback_level}_{DEEP_TIEBREAK_LEVEL}_tiebreak":
        expected_method = (
            f"time_disagreement_tiebreak_levels_{fallback_level}_{DEEP_TIEBREAK_LEVEL}"
        )
        if result_method != expected_method:
            raise ValueError("deep time tiebreak has an unexpected selected-method record")
        _validate_recorded_search_result(
            board,
            final,
            "deep time tiebreak result",
            max(min_depth, DEEP_TIEBREAK_LEVEL),
            min_selectivity,
            DEEP_TIEBREAK_LEVEL,
        )
        primary = _validate_recorded_search_result(
            board, final.get("primary"), "time result before tiebreak", None, min_selectivity
        )
        validate_time_primary(primary, "time result before tiebreak")
        tiebreak = _validate_recorded_search_result(
            board,
            final.get("tiebreak"),
            "first time tiebreak result",
            max(min_depth, fallback_level),
            min_selectivity,
            fallback_level,
        )
        if (
            primary["move"] == tiebreak["move"]
            or primary["move"] == verification["move"]
            or final["move"] != tiebreak["move"]
        ):
            raise ValueError("deep time tiebreak does not support the accepted move")
        return
    raise ValueError("time verification has an unexpected verification mode")


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
    random_seed: int = 620,
    root_order: str = ROOT_ORDER_HASH,
    priority_tie_seed: int | None = None,
    priority_manifest: dict[str, object] | None = None,
    formal_comparison: dict[str, object] | None = None,
) -> dict[str, Any]:
    random_seed = _validate_random_seed(random_seed)
    if root_order not in ROOT_ORDER_CHOICES:
        raise ValueError(f"root_order must be one of {ROOT_ORDER_CHOICES}")
    local_probability_model_sources = (
        local_probability_model_source_fingerprints()
        if root_order == ROOT_ORDER_GGS_R14_PROBABILITY
        else None
    )
    priority_manifest_tie_seed: int | None = None
    if priority_manifest is not None:
        if root_order != ROOT_ORDER_GGS_R14_PROBABILITY:
            raise ValueError("a frozen priority manifest requires probability root order")
        value = priority_manifest.get("tie_seed")
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("frozen priority manifest has an invalid tie seed")
        input_audit = priority_manifest.get("input_audit")
        if (
            not isinstance(input_audit, dict)
            or input_audit.get("local_probability_model_sources")
            != local_probability_model_sources
        ):
            raise ValueError(
                "frozen priority manifest does not match the current local probability-model sources"
            )
        priority_manifest_tie_seed = value
    return {
        "schema": TEACHER_SCHEMA,
        "calculation_provenance": _new_calculation_provenance(
            output, exe, hash_level, random_seed
        ),
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
        "random_seed": random_seed,
        "min_depth": min_depth,
        "min_selectivity": min_selectivity,
        "fallback_level": fallback_level,
        "method": method,
        "teacher_level": teacher_level,
        "verify_level": verify_level,
        "cohort_seed": cohort_seed,
        "root_order": root_order,
        "priority_tie_seed": priority_tie_seed,
        "priority_manifest": priority_manifest,
        "priority_manifest_tie_seed": priority_manifest_tie_seed,
        "formal_comparison": formal_comparison,
        "r14_probability_model": (
            R14_RANDOM_SETUP_PROBABILITY_MODEL
            if root_order == ROOT_ORDER_GGS_R14_PROBABILITY
            else None
        ),
        "r14_local_probability_model_sources": local_probability_model_sources,
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
        "schema", "random_seed", "calculation_provenance", "coverage", "engine", "time_seconds", "threads", "hash_level",
        "min_depth", "min_selectivity", "fallback_level", "method", "teacher_level",
        "verify_level", "cohort_seed", "root_order", "priority_tie_seed", "priority_manifest",
        "priority_manifest_tie_seed",
        "formal_comparison",
        "r14_probability_model", "r14_local_probability_model_sources", "excluded_root_files",
        "deep_tiebreak_level", "roots",
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
    _ensure_calculation_snapshots(output, state)
    roots = state["roots"]
    results = state["results"]
    rejections = state["rejections"]
    provenance = validate_calculation_provenance(state["calculation_provenance"])
    book_configuration = provenance["book_configuration"]
    teacher_script = provenance["teacher_script"]
    teacher_script_snapshot = provenance["teacher_script_snapshot"]
    resources = provenance["execution_environment"]["resources"]
    resource_sha256 = {resource["role"]: resource["snapshot"]["sha256"] for resource in resources}
    priority_sources = (
        state["priority_manifest"]["input_audit"]["local_probability_model_sources"]
        if state["priority_manifest"]
        else None
    )
    probability_source_lines = [
        *probability_model_source_header_lines(
            "priority_manifest_probability_model_sources", priority_sources
        ),
        *probability_model_source_header_lines(
            "r14_probability_model_sources", state["r14_local_probability_model_sources"]
        ),
    ]
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
            f"# evaluation_sha256 {resource_sha256['evaluation']}",
            f"# endgame_move_ordering_sha256 {resource_sha256['endgame_move_ordering']}",
            f"# time_seconds {state['time_seconds']:g}",
            f"# threads {state['threads']}",
            f"# hash_level {state['hash_level']}",
            f"# random_seed {state['random_seed']}",
            f"# min_depth {state['min_depth']}",
            f"# min_selectivity {state['min_selectivity']}",
            f"# fallback_level {state['fallback_level']}",
            f"# method {state['method']}",
            f"# teacher_level {state['teacher_level']}",
            f"# verify_level {state['verify_level']}",
            f"# cohort_seed {state['cohort_seed']}",
            f"# root_order {state['root_order']}",
            f"# priority_tie_seed {state['priority_tie_seed']}",
            f"# priority_manifest_tie_seed {state['priority_manifest_tie_seed']}",
            f"# priority_manifest_sha256 "
            f"{state['priority_manifest']['sha256'] if state['priority_manifest'] else '-'}",
            f"# priority_manifest_metadata_sha256 "
            f"{state['priority_manifest']['metadata_sha256'] if state['priority_manifest'] else '-'}",
            f"# priority_manifest_ordered_roots_sha256 "
            f"{state['priority_manifest']['ordered_roots_sha256'] if state['priority_manifest'] else '-'}",
            f"# formal_comparison_report_sha256 "
            f"{state['formal_comparison']['report']['sha256'] if state['formal_comparison'] else '-'}",
            f"# formal_comparison_state_sha256 "
            f"{state['formal_comparison']['experiment_state']['sha256'] if state['formal_comparison'] else '-'}",
            *probability_source_lines,
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
        "random_seed": state["random_seed"],
        "min_depth": state["min_depth"],
        "min_selectivity": state["min_selectivity"],
        "fallback_level": state["fallback_level"],
        "method": state["method"],
        "teacher_level": state["teacher_level"],
        "verify_level": state["verify_level"],
        "cohort_seed": state["cohort_seed"],
        "root_order": state["root_order"],
        "priority_tie_seed": state["priority_tie_seed"],
        "priority_manifest": state["priority_manifest"],
        "priority_manifest_tie_seed": state["priority_manifest_tie_seed"],
        "formal_comparison": state["formal_comparison"],
        "r14_probability_model": state["r14_probability_model"],
        "r14_local_probability_model_sources": state[
            "r14_local_probability_model_sources"
        ],
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
    random_seed: int = 620,
    root_order: str = ROOT_ORDER_HASH,
    priority_tie_seed: int | None = None,
    priority_manifest_path: Path | None = None,
    formal_comparison_report_path: Path | None = None,
    max_new_positions: int | None = None,
) -> dict[str, int]:
    if not exe.is_file():
        raise FileNotFoundError(f"engine executable not found: {exe}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if max_new_positions is not None and max_new_positions <= 0:
        raise ValueError("max_new_positions must be positive")
    if compact_only and not resume:
        raise ValueError("compact_only requires resume")
    random_seed = _validate_random_seed(random_seed)
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
    formal_comparison: dict[str, object] | None = None
    if formal_comparison_report_path is not None:
        formal_comparison = _load_formal_comparison_selection(
            formal_comparison_report_path,
            coverage_path,
            exe,
            method=method,
            teacher_level=teacher_level,
            verify_level=verify_level,
            threads=threads,
            hash_level=hash_level,
            min_depth=min_depth,
            min_selectivity=min_selectivity,
        )
    if excluded_root_files is None:
        excluded_root_files = []
    excluded_root_files = sorted(
        {path.resolve() for path in excluded_root_files}, key=lambda path: path.as_posix()
    )
    excluded_roots = load_excluded_roots(excluded_root_files)
    priority_boards: list[str] | None = None
    priority_manifest: dict[str, object] | None = None
    if priority_manifest_path is not None:
        priority_boards, _priority_metadata, priority_manifest = load_r14_random_setup_priority_manifest(
            priority_manifest_path
        )
    roots = select_teacher_roots(
        [board for board in load_uncovered_roots(coverage_path) if board not in excluded_roots],
        limit,
        cohort_seed,
        root_order,
        priority_tie_seed,
        priority_boards,
    )
    if not roots:
        raise ValueError("no uncovered 14-disc roots remain after exclusions")
    expected = _new_state(
        output,
        coverage_path,
        exe,
        roots,
        time_seconds,
        threads,
        hash_level,
        min_depth,
        min_selectivity,
        fallback_level,
        method,
        teacher_level,
        verify_level,
        cohort_seed,
        excluded_root_files,
        random_seed,
        root_order,
        priority_tie_seed,
        priority_manifest,
        formal_comparison,
    )
    state_path = _state_path(output)
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state not found: {state_path}")
        state = _load_state(state_path, expected)
        _ensure_calculation_snapshots(output, state)
        _apply_pending_updates(output, state)
        # Always republish all three public files.  This repairs a stop between
        # writing the rows/state and writing the manifest even with no pending row.
        _write_outputs(output, state)
        _clear_pending_updates(output)
    else:
        artifacts = (
            output,
            state_path,
            _manifest_path(output),
            _pending_updates_path(output),
            _teacher_script_snapshot_path(output),
        )
        existing = [path for path in artifacts if path.exists()]
        if existing:
            raise FileExistsError(
                "teacher output artifacts already exist; use --resume or choose a new output: "
                + ", ".join(path.as_posix() for path in existing)
            )
        state = expected
        _write_outputs(output, state)

    completed_since_checkpoint = _completed_since_checkpoint(state, checkpoint_every)
    newly_processed = 0
    if compact_only:
        # The resume branch has already replayed and republished every durable row.
        return _generation_status(state, roots)
    execution_exe = _ensure_calculation_snapshots(output, state)
    for board in roots:
        if board in state["results"] or board in state["rejections"]:
            continue
        if method == "hint":
            result = search_root_at_level(
                execution_exe, board, teacher_level, threads, hash_level, random_seed
            )
            validate_quality(result, max(min_depth, teacher_level), min_selectivity)
            result["method"] = f"hint_level_{teacher_level}"
        elif method == "time_then_hint":
            result = search_root(
                execution_exe, board, time_seconds, threads, hash_level, random_seed
            )
            try:
                validate_quality(result, min_depth, min_selectivity)
                result["method"] = "time"
            except ValueError:
                if fallback_level == 0:
                    raise
                result = search_root_at_level(
                    execution_exe, board, fallback_level, threads, hash_level, random_seed
                )
                validate_quality(result, max(min_depth, fallback_level), min_selectivity)
                result["method"] = f"hint_level_{fallback_level}"
        elif method == "hint_then_verify":
            result = search_root_at_level(
                execution_exe, board, teacher_level, threads, hash_level, random_seed
            )
            validate_quality(result, max(min_depth, teacher_level), min_selectivity)
            result["method"] = (
                f"hint_level_{teacher_level}_verified_hint_level_{verify_level}"
            )
            verification = search_root_at_level(
                execution_exe, board, verify_level, threads, hash_level, random_seed
            )
            validate_quality(verification, max(min_depth, verify_level), min_selectivity)
            if str(result["move"]) != str(verification["move"]):
                verification_repeat = search_root_at_level(
                    execution_exe, board, verify_level, threads, hash_level, random_seed
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
                    newly_processed += 1
                    if completed_since_checkpoint >= checkpoint_every:
                        _write_outputs(output, state)
                        _clear_pending_updates(output)
                        completed_since_checkpoint = 0
                    print(
                        f"rejected {len(state['rejections'])} {board}",
                        flush=True,
                    )
                    if max_new_positions is not None and newly_processed >= max_new_positions:
                        _write_outputs(output, state)
                        _clear_pending_updates(output)
                        return _generation_status(state, roots)
                    continue
            else:
                result["verification"] = verification
                result["verification_mode"] = f"level_{verify_level}_exact"
        else:
            result = search_root(
                execution_exe, board, time_seconds, threads, hash_level, random_seed
            )
            primary = result
            try:
                validate_quality(result, min_depth, min_selectivity)
                result["method"] = f"time_verified_hint_level_{verify_level}"
            except ValueError:
                result = search_root_at_level(
                    execution_exe, board, fallback_level, threads, hash_level, random_seed
                )
                validate_quality(result, max(min_depth, fallback_level), min_selectivity)
                result["method"] = (
                    f"time_fallback_hint_level_{fallback_level}_verified_hint_level_{verify_level}"
                )
                result["primary"] = primary
            verification = search_root_at_level(
                execution_exe, board, verify_level, threads, hash_level, random_seed
            )
            validate_quality(verification, max(min_depth, verify_level), min_selectivity)
            verification_mode = f"level_{verify_level}_exact"
            if str(result["move"]) != str(verification["move"]):
                tiebreak = search_root_at_level(
                    execution_exe, board, fallback_level, threads, hash_level, random_seed
                )
                validate_quality(tiebreak, max(min_depth, fallback_level), min_selectivity)
                if str(result["move"]) != str(tiebreak["move"]):
                    deep_tiebreak = search_root_at_level(
                        execution_exe, board, DEEP_TIEBREAK_LEVEL, threads, hash_level, random_seed
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
                        _append_pending_update(
                            output, board, "rejections", state["rejections"][board]
                        )
                        completed_since_checkpoint += 1
                        newly_processed += 1
                        if completed_since_checkpoint >= checkpoint_every:
                            _write_outputs(output, state)
                            _clear_pending_updates(output)
                            completed_since_checkpoint = 0
                        print(
                            f"rejected {len(state['rejections'])} {board}",
                            flush=True,
                        )
                        if max_new_positions is not None and newly_processed >= max_new_positions:
                            _write_outputs(output, state)
                            _clear_pending_updates(output)
                            return _generation_status(state, roots)
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
        newly_processed += 1
        if completed_since_checkpoint >= checkpoint_every:
            _write_outputs(output, state)
            _clear_pending_updates(output)
            completed_since_checkpoint = 0
        print(f"accepted {len(state['results'])}/{len(roots)} {board}", flush=True)
        if max_new_positions is not None and newly_processed >= max_new_positions:
            _write_outputs(output, state)
            _clear_pending_updates(output)
            return _generation_status(state, roots)
    if completed_since_checkpoint:
        _write_outputs(output, state)
        _clear_pending_updates(output)
    return _generation_status(state, roots)


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
    random_seed: int = 620,
    root_order: str = ROOT_ORDER_HASH,
    priority_tie_seed: int | None = None,
    priority_manifest_path: Path | None = None,
    formal_comparison_report_path: Path | None = None,
    max_new_positions: int | None = None,
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
            random_seed,
            root_order,
            priority_tie_seed,
            priority_manifest_path,
            formal_comparison_report_path,
            max_new_positions,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-seconds", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--hash", dest="hash_level", type=int, default=29)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=620,
        help="Console -seed value recorded and used for every teacher process",
    )
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
        help="With --root-order hash, hash-sort uncovered roots by this seed before --limit",
    )
    parser.add_argument(
        "--root-order",
        choices=ROOT_ORDER_CHOICES,
        default=ROOT_ORDER_HASH,
        help=(
            "hash preserves the existing uniform order; ggs-r14-probability orders the "
            "r14 corpus from the local random_setup(14) reimplementation by "
            "probability"
        ),
    )
    parser.add_argument(
        "--priority-tie-seed",
        type=int,
        help=(
            "With --root-order ggs-r14-probability, use this seed only to order "
            "equal-probability positions before --limit"
        ),
    )
    parser.add_argument(
        "--priority-manifest",
        type=Path,
        help=(
            "With --root-order ggs-r14-probability, use this validated frozen "
            "JSON Lines priority file instead of recomputing an order"
        ),
    )
    parser.add_argument(
        "--formal-comparison-report",
        type=Path,
        help=(
            "Completed formal method-comparison JSON. When supplied, require its "
            "selected level-30/level-31 setup and record its SHA-256."
        ),
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
        help="Compact durable per-position updates after this many processed positions",
    )
    parser.add_argument(
        "--max-new-positions",
        type=int,
        help=(
            "Process at most this many previously unprocessed positions in this "
            "invocation, then compact durable output and stop"
        ),
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
        args.random_seed,
        args.root_order,
        args.priority_tie_seed,
        args.priority_manifest,
        args.formal_comparison_report,
        args.max_new_positions,
    )
    print(
        f"teacher roots processed {result['processed']}/{result['requested']} "
        f"accepted={result['accepted']} rejected={result['rejected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
