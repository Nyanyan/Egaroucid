"""Run the predeclared replicated comparison of two root-teacher methods.

This program is deliberately separate from the ordinary, one-observation
benchmark.  It selects 600 previously unused positions, derives four Console
random seeds from a SHA-256 domain string and the corpus fingerprint, and runs
both methods twice for every position and seed.  The method order is reversed
in the second repetition.  A result is durable only after the complete pair
of methods (and, when needed, the level-33 comparison) has been written.

It is evidence for selecting a calculation method.  It is not a playing
strength experiment and it never writes the tournament starting-position move
table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
import uuid
from typing import Any, Iterable

from audit_r14_corpus import CORPUS_REPORT_SCHEMA
from book_artifact import file_lock
from build_root_table import sha256_file
import generate_ggs_root_teacher as teacher
from generate_ggs_root_teacher import (
    BOOK_DISABLED_ARGUMENTS,
    load_excluded_roots,
    load_uncovered_roots,
    parse_search_result,
    select_teacher_roots,
    validate_quality,
)


FORMAL_COMPARISON_SCHEMA = "formal_root_teacher_method_comparison_v1"
FORMAL_STATE_SCHEMA = "formal_root_teacher_method_comparison_state_v1"
FORMAL_PROGRESS_SCHEMA = "formal_root_teacher_method_comparison_progress_v1"
FORMAL_PAIR_SCHEMA = "formal_root_teacher_method_comparison_pair_v1"
FORMAL_REPORT_SCHEMA = "formal_root_teacher_method_comparison_report_v1"
FORMAL_ENVIRONMENT_SCHEMA = "formal_root_teacher_method_comparison_environment_v1"

POSITION_COUNT = 600
ENGINE_SEED_COUNT = 4
DEEP_SEED_COUNT = 2
REPETITION_COUNT = 2
TIME_SECONDS = 60.0
THREADS = 28
HASH_LEVEL = 29
MIN_DEPTH = 30
MIN_SELECTIVITY = 74
LEVEL_30 = 30
LEVEL_31 = 31
LEVEL_33 = 33
BOOTSTRAP_REPETITIONS = 100_000
TIME_METHOD = "time_then_verify"
LEVEL_METHOD = "hint_then_verify"
METHODS = (TIME_METHOD, LEVEL_METHOD)

ENGINE_SEED_DOMAIN = "Egaroucid formal root-teacher engine-seed v1"
DEEP_SEED_DOMAIN = "Egaroucid formal root-teacher deep-comparison-seed v1"
SELECTION_SEED_DOMAIN = "Egaroucid formal root-teacher position-selection v1"
SELECTION_ID_DOMAIN = "Egaroucid formal root-teacher position-selection-id v1"
ORDER_ID_DOMAIN = "Egaroucid formal root-teacher execution-order-id v1"
PAIR_ID_DOMAIN = "Egaroucid formal root-teacher calculation-pair-id v1"
SCHEDULE_DOMAIN = "Egaroucid formal root-teacher execution-schedule v1"
BOOTSTRAP_SEED_DOMAIN = "Egaroucid formal root-teacher bootstrap-seed v1"

ANALYZE_RESULT_RE = re.compile(
    r"^\|\s*\d+\|\s*(?:Black|White)\|\s*(?P<move>[a-h][1-8])"
    r"\|\s*(?P<depth>\d+@\d+%)\|\s*(?P<score>[+-]?\d+)\|",
    re.MULTILINE,
)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write one complete file or leave its prior version intact."""
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


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} {path} is not a JSON object")
    return payload


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_boards(boards: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(boards).encode("ascii")).hexdigest()


def _derive_digest(domain: str, *parts: str) -> bytes:
    payload = "\0".join((domain, *parts)).encode("ascii")
    return hashlib.sha256(payload).digest()


def _derive_u64(domain: str, *parts: str) -> int:
    return int.from_bytes(_derive_digest(domain, *parts)[:8], "big")


def _fingerprint(path: Path, description: str) -> dict[str, int | str]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return {
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_fingerprint(
    payload: object, description: str, expected_path: Path | None = None
) -> dict[str, int | str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{description} fingerprint is not an object")
    path = payload.get("path")
    digest = payload.get("sha256")
    size = payload.get("bytes")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{description} fingerprint has no path")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{description} fingerprint has an invalid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{description} fingerprint has an invalid byte count")
    recorded = Path(path).resolve()
    if expected_path is not None and recorded != expected_path.resolve():
        raise ValueError(f"{description} fingerprint has an unexpected path")
    return {"path": recorded.as_posix(), "sha256": digest, "bytes": size}


def _copy_or_verify(source: Path, destination: Path, expected: dict[str, int | str]) -> None:
    """Create an immutable copy, never overwriting a differing existing file."""
    if destination.exists():
        actual = _fingerprint(destination, f"saved file {destination}")
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            raise ValueError(f"saved file differs from frozen input: {destination}")
        return
    if _fingerprint(source, f"source file {source}")["sha256"] != expected["sha256"]:
        raise ValueError(f"source file changed while creating its saved copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        copied = _fingerprint(temporary, f"temporary saved file {temporary}")
        if copied["sha256"] != expected["sha256"] or copied["bytes"] != expected["bytes"]:
            raise ValueError(f"copied file differs from frozen input: {source}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _environment_identifier(
    executable: dict[str, int | str], resources: list[dict[str, Any]]
) -> str:
    compact = {
        "executable_name": Path(str(executable["path"])).name,
        "executable_sha256": executable["sha256"],
        "resources": [
            {
                "role": item["role"],
                "relative_path": item["relative_path"],
                "sha256": item["source"]["sha256"],
            }
            for item in resources
        ],
    }
    return hashlib.sha256(
        json.dumps(compact, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _new_execution_environment(output_dir: Path, exe: Path) -> dict[str, Any]:
    """Describe the exact Console executable and two files it reads at start-up."""
    source_exe = _fingerprint(exe, "Console executable")
    source_resources: list[dict[str, Any]] = []
    for role, relative in (
        ("evaluation", Path("eval.egev2")),
        ("endgame_move_ordering", Path("eval_move_ordering_end.egev")),
    ):
        source_resources.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                "source": _fingerprint(exe.resolve().parent / "resources" / relative, role),
            }
        )
    identifier = _environment_identifier(source_exe, source_resources)
    directory = (output_dir / "saved_console_inputs" / f"console-{identifier}").resolve()
    resources: list[dict[str, Any]] = []
    for item in source_resources:
        relative = Path(str(item["relative_path"]))
        source = item["source"]
        resources.append(
            {
                "role": item["role"],
                "relative_path": item["relative_path"],
                "source": source,
                "snapshot": {
                    "path": (directory / "resources" / relative).as_posix(),
                    "sha256": source["sha256"],
                    "bytes": source["bytes"],
                },
            }
        )
    return {
        "schema": FORMAL_ENVIRONMENT_SCHEMA,
        "identifier": identifier,
        "directory": directory.as_posix(),
        "executable": {
            "source": source_exe,
            "snapshot": {
                "path": (directory / Path(str(source_exe["path"])).name).as_posix(),
                "sha256": source_exe["sha256"],
                "bytes": source_exe["bytes"],
            },
        },
        "resources": resources,
    }


def _validate_and_materialize_environment(output_dir: Path, environment: object) -> Path:
    """Verify that the recorded snapshot stays below this output directory."""
    if not isinstance(environment, dict) or environment.get("schema") != FORMAL_ENVIRONMENT_SCHEMA:
        raise ValueError("saved Console environment has an unsupported schema")
    identifier = environment.get("identifier")
    if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{64}", identifier) is None:
        raise ValueError("saved Console environment has an invalid identifier")
    expected_root = (output_dir / "saved_console_inputs" / f"console-{identifier}").resolve()
    if Path(str(environment.get("directory", ""))).resolve() != expected_root:
        raise ValueError("saved Console environment is outside the expected directory")
    executable = environment.get("executable")
    if not isinstance(executable, dict):
        raise ValueError("saved Console environment has no executable")
    source_exe = _validate_fingerprint(executable.get("source"), "source Console executable")
    snapshot_exe = _validate_fingerprint(
        executable.get("snapshot"), "saved Console executable", expected_root / Path(str(source_exe["path"])).name
    )
    if source_exe["sha256"] != snapshot_exe["sha256"] or source_exe["bytes"] != snapshot_exe["bytes"]:
        raise ValueError("source Console executable and saved copy differ in the state")
    resources = environment.get("resources")
    if not isinstance(resources, list) or len(resources) != 2:
        raise ValueError("saved Console environment has an invalid resource list")
    expected_roles = (
        ("evaluation", Path("eval.egev2")),
        ("endgame_move_ordering", Path("eval_move_ordering_end.egev")),
    )
    source_paths: list[tuple[Path, Path, dict[str, int | str]]] = [
        (Path(str(source_exe["path"])), Path(str(snapshot_exe["path"])), source_exe)
    ]
    normalized_resources: list[dict[str, Any]] = []
    for item, (role, relative) in zip(resources, expected_roles):
        if not isinstance(item, dict) or item.get("role") != role or item.get("relative_path") != relative.as_posix():
            raise ValueError("saved Console environment has an unexpected resource")
        source = _validate_fingerprint(
            item.get("source"), f"source {role} resource", Path(str(source_exe["path"])).parent / "resources" / relative
        )
        snapshot = _validate_fingerprint(
            item.get("snapshot"), f"saved {role} resource", expected_root / "resources" / relative
        )
        if source["sha256"] != snapshot["sha256"] or source["bytes"] != snapshot["bytes"]:
            raise ValueError(f"source and saved {role} resource differ in the state")
        source_paths.append((Path(str(source["path"])), Path(str(snapshot["path"])), source))
        normalized_resources.append({"role": role, "relative_path": relative.as_posix(), "source": source})
    if _environment_identifier(source_exe, normalized_resources) != identifier:
        raise ValueError("saved Console environment identifier does not match its inputs")
    for source, snapshot, metadata in source_paths:
        _copy_or_verify(source, snapshot, metadata)
    return Path(str(snapshot_exe["path"])).resolve()


def _load_prior_sample_coverage(path: Path) -> set[str]:
    payload = _read_json(path, "previous fixed sample")
    if payload.get("schema") != CORPUS_REPORT_SCHEMA:
        raise ValueError(f"{path}: previous fixed sample does not use the corpus-report schema")
    roots = payload.get("roots")
    if not isinstance(roots, list):
        raise ValueError(f"{path}: previous fixed sample has no roots list")
    boards: list[str] = []
    for root in roots:
        if not isinstance(root, dict) or not isinstance(root.get("canonical_board"), str):
            raise ValueError(f"{path}: previous fixed sample has an invalid root")
        boards.append(root["canonical_board"])
    if len(boards) != len(set(boards)):
        raise ValueError(f"{path}: previous fixed sample lists one board more than once")
    return set(boards)


def _prior_input_provenance(
    root_results: list[Path], sample_coverages: list[Path]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted({item.resolve() for item in root_results}, key=lambda item: item.as_posix()):
        records.append({"kind": "root_results", **_fingerprint(path, "previous root-result input")})
    for path in sorted({item.resolve() for item in sample_coverages}, key=lambda item: item.as_posix()):
        records.append({"kind": "sample_coverage", **_fingerprint(path, "previous sample-coverage input")})
    return records


def _make_fixed_sample_payload(boards: list[str]) -> dict[str, Any]:
    return {
        "schema": CORPUS_REPORT_SCHEMA,
        "root_discs": 14,
        "roots": [
            {"canonical_board": board, "deep_book": False, "root_table": False}
            for board in boards
        ],
    }


def _derive_engine_seeds(corpus_fingerprint: str) -> list[dict[str, int | str]]:
    derived: list[dict[str, int | str]] = []
    for index in range(1, ENGINE_SEED_COUNT + 1):
        digest = _derive_digest(ENGINE_SEED_DOMAIN, corpus_fingerprint, str(index))
        derived.append(
            {
                "index": index,
                "seed": int.from_bytes(digest[:4], "big"),
                "derivation_sha256": digest.hex(),
            }
        )
    if len({int(item["seed"]) for item in derived}) != ENGINE_SEED_COUNT:
        raise RuntimeError("SHA-256 seed derivation unexpectedly produced duplicate 32-bit seeds")
    return derived


def _derive_deep_seeds(
    corpus_fingerprint: str, main_seeds: list[dict[str, int | str]]
) -> list[dict[str, int | str]]:
    """Derive two seeds reserved exclusively for differing-move level-33 checks."""
    excluded = {int(item["seed"]) for item in main_seeds}
    derived: list[dict[str, int | str]] = []
    for index in range(1, DEEP_SEED_COUNT + 1):
        attempt = 1
        while True:
            digest = _derive_digest(DEEP_SEED_DOMAIN, corpus_fingerprint, str(index), str(attempt))
            seed = int.from_bytes(digest[:4], "big")
            if seed not in excluded:
                excluded.add(seed)
                derived.append(
                    {
                        "index": index,
                        "seed": seed,
                        "derivation_attempt": attempt,
                        "derivation_sha256": digest.hex(),
                    }
                )
                break
            attempt += 1
    return derived


def _method_order(selection_id: str, board: str, engine_seed: int) -> tuple[list[str], str]:
    digest = _derive_digest(ORDER_ID_DOMAIN, selection_id, board, str(engine_seed))
    base = [TIME_METHOD, LEVEL_METHOD] if digest[0] % 2 == 0 else [LEVEL_METHOD, TIME_METHOD]
    return base, digest.hex()


def _pair_plan(
    boards: list[str], selection_id: str, engine_seeds: list[dict[str, int | str]]
) -> list[dict[str, Any]]:
    unscheduled: list[dict[str, Any]] = []
    for position_index, board in enumerate(boards, start=1):
        for seed_item in engine_seeds:
            seed_index = int(seed_item["index"])
            engine_seed = int(seed_item["seed"])
            base_order, order_id = _method_order(selection_id, board, engine_seed)
            for repetition in range(1, REPETITION_COUNT + 1):
                order = base_order if repetition == 1 else list(reversed(base_order))
                pair_id = _derive_digest(
                    PAIR_ID_DOMAIN,
                    selection_id,
                    board,
                    str(seed_index),
                    str(repetition),
                ).hex()
                schedule_key = _derive_digest(SCHEDULE_DOMAIN, selection_id, pair_id).hex()
                unscheduled.append(
                    {
                        "position_index": position_index,
                        "board": board,
                        "engine_seed_index": seed_index,
                        "engine_seed": engine_seed,
                        "repetition": repetition,
                        "execution_order": order,
                        "execution_order_id": order_id,
                        "pair_id": pair_id,
                        "schedule_key": schedule_key,
                    }
                )
    if len(unscheduled) != len(boards) * ENGINE_SEED_COUNT * REPETITION_COUNT:
        raise RuntimeError("unexpected calculation-pair plan size")
    unscheduled.sort(key=lambda item: (str(item["schedule_key"]), str(item["pair_id"])))
    for schedule_index, item in enumerate(unscheduled, start=1):
        item["schedule_index"] = schedule_index
        del item["schedule_key"]
    seen = {str(item["pair_id"]) for item in unscheduled}
    if len(seen) != len(unscheduled):
        raise RuntimeError("calculation-pair identifiers are not unique")
    return unscheduled


def _decision_protocol() -> dict[str, Any]:
    """Return the decision rule before any calculation result is read."""
    return {
        "version": "formal_root_teacher_method_decision_rule_v1",
        "candidate_method": LEVEL_METHOD,
        "reference_method": TIME_METHOD,
        "required_complete_pairs": POSITION_COUNT * ENGINE_SEED_COUNT * REPETITION_COUNT,
        "level_33_required_when_accepted_moves_differ": True,
        "dedicated_level_33_seed_count": DEEP_SEED_COUNT,
        "conditions_all_required": [
            "every planned calculation pair is durably recorded",
            "for every position and main engine seed, repetitions 1 and 2 give the same accepted/rejected status for each method and the same selected move when accepted",
            "no accepted level-method move has a lower forced-move level-33 score than the reference-method move",
            "every dedicated level-33 seed has stable repeated root and forced-move results when accepted moves differ",
            "every stable dedicated level-33 root move is one of the two accepted candidate moves when accepted moves differ",
            "the two dedicated level-33 seeds agree on stable root best move and both candidate forced-move scores when accepted moves differ",
            "no accepted-move difference has an unresolved dedicated level-33 comparison",
            "no pair is accepted only by the reference method",
            "the upper endpoint of the 95 percent paired, position-cluster bootstrap interval for candidate/reference wall time is below 0.90",
        ],
        "bootstrap": {
            "unit": "position",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "confidence_interval": "percentile 95 percent",
        },
        "scope": "calculation_method_selection_only_not_playing_strength",
    }


def _new_state(
    coverage: Path,
    root_results: list[Path],
    sample_coverages: list[Path],
    exe: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], dict[str, Any]]:
    if not root_results and not sample_coverages:
        raise ValueError(
            "a formal comparison must receive at least one --exclude-root-results or --exclude-sample-coverage input"
        )
    all_roots = sorted(load_uncovered_roots(coverage))
    if len(all_roots) != len(set(all_roots)):
        raise ValueError("coverage has duplicate canonical boards")
    corpus_fingerprint = _sha256_boards(all_roots)
    excluded = set(load_excluded_roots(root_results))
    for path in sample_coverages:
        excluded.update(_load_prior_sample_coverage(path))
    population = sorted(board for board in all_roots if board not in excluded)
    if len(population) < POSITION_COUNT:
        raise ValueError(
            f"only {len(population)} positions remain after exclusions; {POSITION_COUNT} are required"
        )
    excluded_fingerprint = _sha256_boards(sorted(excluded))
    selection_seed = _derive_u64(SELECTION_SEED_DOMAIN, corpus_fingerprint, excluded_fingerprint)
    boards = select_teacher_roots(population, POSITION_COUNT, selection_seed)
    if len(boards) != POSITION_COUNT or len(boards) != len(set(boards)):
        raise RuntimeError("position selection did not produce 600 unique boards")
    if set(boards) & excluded:
        raise RuntimeError("position selection included a board supplied for exclusion")
    selection_id = _derive_digest(
        SELECTION_ID_DOMAIN,
        corpus_fingerprint,
        excluded_fingerprint,
        _sha256_boards(boards),
    ).hex()
    engine_seeds = _derive_engine_seeds(corpus_fingerprint)
    deep_seeds = _derive_deep_seeds(corpus_fingerprint, engine_seeds)
    plan = _pair_plan(boards, selection_id, engine_seeds)
    sample_payload = _make_fixed_sample_payload(boards)
    sample_text = _canonical_json(sample_payload)
    environment = _new_execution_environment(output_dir, exe)
    script = Path(__file__).resolve()
    teacher_script = Path(teacher.__file__).resolve()
    state = {
        "schema": FORMAL_STATE_SCHEMA,
        "source_files": {
            "formal_comparison_script": _fingerprint(script, "formal comparison script"),
            "teacher_script": _fingerprint(teacher_script, "teacher script"),
        },
        "coverage": _fingerprint(coverage, "coverage"),
        "prior_inputs": _prior_input_provenance(root_results, sample_coverages),
        "population": {
            "corpus_fingerprint": corpus_fingerprint,
            "coverage_count": len(all_roots),
            "excluded_board_count": len(excluded),
            "excluded_board_fingerprint": excluded_fingerprint,
            "remaining_count": len(population),
            "remaining_fingerprint": _sha256_boards(population),
        },
        "selection": {
            "position_count": POSITION_COUNT,
            "selection_seed": selection_seed,
            "selection_id": selection_id,
            "boards_fingerprint": _sha256_boards(boards),
            "fixed_sample_coverage": {
                "path": (output_dir / "fixed_sample_coverage.json").resolve().as_posix(),
                "sha256": _sha256_text(sample_text),
            },
            "boards": boards,
        },
        "engine_seed_derivation": {
            "sha256_domain_string": ENGINE_SEED_DOMAIN,
            "corpus_fingerprint": corpus_fingerprint,
            "seeds": engine_seeds,
        },
        "deep_seed_derivation": {
            "sha256_domain_string": DEEP_SEED_DOMAIN,
            "corpus_fingerprint": corpus_fingerprint,
            "seeds": deep_seeds,
            "excluded_main_engine_seed_values": sorted(int(item["seed"]) for item in engine_seeds),
        },
        "conditions": {
            "threads": THREADS,
            "hash_level": HASH_LEVEL,
            "books_disabled": list(BOOK_DISABLED_ARGUMENTS),
            "time_method": {
                "name": TIME_METHOD,
                "time_seconds": TIME_SECONDS,
                "fallback_level": LEVEL_30,
                "verify_level": LEVEL_31,
            },
            "level_method": {
                "name": LEVEL_METHOD,
                "teacher_level": LEVEL_30,
                "verify_level": LEVEL_31,
            },
            "different_move_level_33": {
                "level": LEVEL_33,
                "root_searches": 2,
                "forced_move_analyses_per_candidate_move": 2,
            },
        },
        "execution_environment": environment,
        "decision_protocol": _decision_protocol(),
        "pair_plan": plan,
        "pair_plan_sha256": _sha256_text(_canonical_json(plan)),
    }
    return state, boards, plan, sample_payload


def _state_path(output_dir: Path) -> Path:
    return output_dir / "experiment_state.json"


def _progress_path(output_dir: Path) -> Path:
    return output_dir / "comparison_progress.json"


def _lock_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.formal-method-comparison.lock"


def _pair_directory(output_dir: Path, plan: dict[str, Any]) -> Path:
    return output_dir / "pairs" / f"{int(plan['schedule_index']):05d}_{str(plan['pair_id'])[:16]}"


def _pair_result_path(output_dir: Path, plan: dict[str, Any]) -> Path:
    return _pair_directory(output_dir, plan) / "pair_result.json"


def _write_fixed_sample(output_dir: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(output_dir / "fixed_sample_coverage.json", _canonical_json(payload))


def _validate_fixed_sample(output_dir: Path, state: dict[str, Any]) -> None:
    recorded = state["selection"]["fixed_sample_coverage"]
    path = output_dir / "fixed_sample_coverage.json"
    if not path.is_file() or sha256_file(path) != recorded["sha256"]:
        raise ValueError("saved fixed sample coverage does not match the immutable state")
    payload = _read_json(path, "saved fixed sample coverage")
    boards = [item.get("canonical_board") for item in payload.get("roots", []) if isinstance(item, dict)]
    if boards != state["selection"]["boards"]:
        raise ValueError("saved fixed sample coverage has different boards")


def _write_progress(output_dir: Path, state_path: Path, plan: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    payload = {
        "schema": FORMAL_PROGRESS_SCHEMA,
        "experiment_state_sha256": sha256_file(state_path),
        "planned_pairs": len(plan),
        "completed_pairs": len(records),
        "records": records,
    }
    _atomic_write_text(_progress_path(output_dir), _canonical_json(payload))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_positive_seconds(value: object, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{description} must be positive finite seconds")


def _result_from_log(log_text: str, board: str) -> dict[str, int | str]:
    return parse_search_result(log_text, board)


def _validate_query(
    query: object, board: str, engine_seed: int, pair_dir: Path
) -> None:
    if not isinstance(query, dict):
        raise ValueError("saved Console query is not an object")
    kind = query.get("kind")
    if kind not in {"time_limited_search", "fixed_level_search", "forced_move_analysis"}:
        raise ValueError("saved Console query has an unknown kind")
    command = query.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("saved Console query has an invalid command")
    required = {
        "-t": str(THREADS),
        "-hash": str(HASH_LEVEL),
        "-seed": str(engine_seed),
    }
    for option, expected in required.items():
        try:
            position = command.index(option)
        except ValueError as error:
            raise ValueError(f"saved Console query omitted {option}") from error
        if position + 1 >= len(command) or command[position + 1] != expected:
            raise ValueError(f"saved Console query has a different value for {option}")
    for option in BOOK_DISABLED_ARGUMENTS:
        if option not in command:
            raise ValueError(f"saved Console query omitted {option}")
    if kind == "time_limited_search":
        try:
            time_index = command.index("-time")
        except ValueError as error:
            raise ValueError("saved time-limited search omitted -time") from error
        if time_index + 1 >= len(command) or command[time_index + 1] != f"{TIME_SECONDS:g}":
            raise ValueError("saved time-limited search has a different time limit")
        if "-l" in command:
            raise ValueError("saved time-limited search unexpectedly has a fixed level")
    else:
        try:
            level_index = command.index("-l")
        except ValueError as error:
            raise ValueError("saved fixed-level search omitted -l") from error
        if level_index + 1 >= len(command):
            raise ValueError("saved fixed-level search has no level value")
        try:
            level_value = int(command[level_index + 1])
        except ValueError as error:
            raise ValueError("saved fixed-level search has an invalid level value") from error
        if level_value < 1:
            raise ValueError("saved fixed-level search has a nonpositive level")
        if "-time" in command:
            raise ValueError("saved fixed-level search unexpectedly has a time limit")
    standard_input = query.get("standard_input")
    if not isinstance(standard_input, str):
        raise ValueError("saved Console query has no standard input")
    if kind == "time_limited_search":
        if standard_input != f"setboard {board}\ngo\nquit\n":
            raise ValueError("saved time-limited search has different Console input")
    elif kind == "fixed_level_search":
        if standard_input != f"setboard {board}\nhint 1\nquit\n":
            raise ValueError("saved fixed-level search has different Console input")
    else:
        move = query.get("forced_move")
        if not isinstance(move, str):
            raise ValueError("saved forced-move analysis has no move")
        if standard_input != f"setboard {board}\nclearcache\nplay {move}\nanalyze\nquit\n":
            raise ValueError("saved forced-move analysis has different Console input")
    log = query.get("raw_log")
    if not isinstance(log, dict):
        raise ValueError("saved Console query has no raw-log metadata")
    relative = log.get("relative_path")
    digest = log.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError("saved Console query has invalid raw-log metadata")
    path = (pair_dir / relative).resolve()
    if not _is_relative_to(path, pair_dir) or not path.is_file() or sha256_file(path) != digest:
        raise ValueError("saved Console raw log is missing or has a different SHA-256")
    text = path.read_text(encoding="utf-8", errors="replace")
    result = query.get("result")
    if not isinstance(result, dict):
        raise ValueError("saved Console query has no parsed result")
    if kind == "forced_move_analysis":
        matches = [match.groupdict() for match in ANALYZE_RESULT_RE.finditer(text)]
        if len(matches) != 1:
            raise ValueError("saved forced-move raw log has an unexpected analyze result count")
        match = matches[0]
        expected = {
            "move": match["move"],
            "score": int(match["score"]),
            "depth": match["depth"],
        }
    else:
        expected = _result_from_log(text, board)
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"saved Console query result differs from its raw log for {field}")


def _recorded_query_level(query: dict[str, Any]) -> int:
    command = query["command"]
    position = command.index("-l")
    return int(command[position + 1])


def _require_query(
    queries: dict[str, Any], label: str, kind: str, level: int | None = None
) -> dict[str, Any]:
    query = queries.get(label)
    if not isinstance(query, dict) or query.get("kind") != kind:
        raise ValueError(f"saved method result has no expected {label} query")
    if level is not None and _recorded_query_level(query) != level:
        raise ValueError(f"saved method result has a different level for {label}")
    return query


def _validate_time_method_logic(method_result: dict[str, Any]) -> None:
    """Reconstruct the declared time-then-verify branch from saved raw queries."""
    queries = method_result["queries"]
    primary = _require_query(queries, "primary", "time_limited_search")
    verification = _require_query(queries, "verification", "fixed_level_search", LEVEL_31)
    primary_is_usable = True
    try:
        validate_quality(primary["result"], MIN_DEPTH, MIN_SELECTIVITY)
    except ValueError:
        primary_is_usable = False
    candidate_label = "primary"
    candidate = primary
    prefix = "time_result_met_minimum_quality"
    expected_keys = {"primary", "verification"}
    if not primary_is_usable:
        candidate_label = "fallback"
        candidate = _require_query(queries, "fallback", "fixed_level_search", LEVEL_30)
        prefix = "time_result_below_minimum_quality_then_level_30_fallback"
        expected_keys.add("fallback")
    candidate_move = candidate["result"]["move"]
    verification_move = verification["result"]["move"]
    if candidate_move == verification_move:
        expected_status = "accepted"
        expected_selected = candidate_label
        expected_mode = prefix + "_level_31_exact"
    else:
        tiebreak = _require_query(queries, "tiebreak", "fixed_level_search", LEVEL_30)
        expected_keys.add("tiebreak")
        if candidate_move == tiebreak["result"]["move"]:
            expected_status = "accepted"
            expected_selected = candidate_label
            expected_mode = prefix + "_level_30_tiebreak"
        else:
            deep_tiebreak = _require_query(
                queries, "deep_tiebreak", "fixed_level_search", LEVEL_31
            )
            expected_keys.add("deep_tiebreak")
            if tiebreak["result"]["move"] != deep_tiebreak["result"]["move"]:
                expected_status = "rejected"
                expected_selected = None
                expected_mode = None
            else:
                expected_status = "accepted"
                expected_selected = "deep_tiebreak"
                expected_mode = prefix + "_levels_30_31_tiebreak"
    if set(queries) != expected_keys:
        raise ValueError("saved time method has unexpected or missing Console queries")
    if method_result["status"] != expected_status:
        raise ValueError("saved time method status disagrees with its recorded verification branch")
    if expected_status == "accepted":
        if method_result.get("selected_query") != expected_selected:
            raise ValueError("saved time method selected a different query")
        if method_result.get("verification_mode") != expected_mode:
            raise ValueError("saved time method has a different verification mode")
    elif method_result.get("reason") != "level-30 and repeated level-31 tiebreak moves differ":
        raise ValueError("saved time method has a different rejection reason")


def _validate_level_method_logic(method_result: dict[str, Any]) -> None:
    """Reconstruct the declared level-30/level-31 verification branch."""
    queries = method_result["queries"]
    initial = _require_query(queries, "teacher", "fixed_level_search", LEVEL_30)
    verification = _require_query(queries, "verification", "fixed_level_search", LEVEL_31)
    if initial["result"]["move"] == verification["result"]["move"]:
        if set(queries) != {"teacher", "verification"}:
            raise ValueError("saved level method has unexpected or missing Console queries")
        if method_result["status"] != "accepted":
            raise ValueError("saved level method rejected an exact verification")
        if method_result.get("selected_query") != "verification" or method_result.get("verification_mode") != "level_31_exact":
            raise ValueError("saved level method recorded a different exact-verification result")
        return
    repeated = _require_query(queries, "verification_repeat", "fixed_level_search", LEVEL_31)
    if set(queries) != {"teacher", "verification", "verification_repeat"}:
        raise ValueError("saved level method has unexpected or missing Console queries")
    if verification["result"]["move"] != repeated["result"]["move"]:
        if method_result["status"] != "rejected":
            raise ValueError("saved level method accepted unstable level-31 verification")
        if method_result.get("reason") != "two level-31 verification moves differ after a level-30 disagreement":
            raise ValueError("saved level method has a different rejection reason")
        return
    if method_result["status"] != "accepted":
        raise ValueError("saved level method rejected repeated matching verification")
    if (
        method_result.get("selected_query") != "verification_repeat"
        or method_result.get("verification_mode") != "level_31_repeated_after_level_30_disagreement"
    ):
        raise ValueError("saved level method recorded a different repeated-verification result")


def _validate_method_result(
    method_result: object, method_name: str, board: str, engine_seed: int, pair_dir: Path
) -> None:
    if not isinstance(method_result, dict) or method_result.get("method") != method_name:
        raise ValueError("saved method result has a different method name")
    _require_positive_seconds(method_result.get("wall_seconds"), "saved method result wall_seconds")
    status = method_result.get("status")
    if status not in {"accepted", "rejected"}:
        raise ValueError("saved method result has an invalid status")
    queries = method_result.get("queries")
    if not isinstance(queries, dict) or not queries:
        raise ValueError("saved method result has no Console queries")
    for query in queries.values():
        _validate_query(query, board, engine_seed, pair_dir)
    if status == "accepted":
        chosen = method_result.get("selected_query")
        if not isinstance(chosen, str) or not isinstance(queries.get(chosen), dict):
            raise ValueError("accepted method result has no selected query")
        selected = queries[chosen]["result"]
        for field in ("move", "score", "depth"):
            if method_result.get(field) != selected.get(field):
                raise ValueError("accepted method result differs from the selected query")
    else:
        if not isinstance(method_result.get("reason"), str) or not method_result["reason"]:
            raise ValueError("rejected method result has no reason")
        if "selected_query" in method_result:
            raise ValueError("rejected method result unexpectedly selected a query")
    if method_name == TIME_METHOD:
        _validate_time_method_logic(method_result)
    elif method_name == LEVEL_METHOD:
        _validate_level_method_logic(method_result)
    else:
        raise ValueError("saved method result has an unsupported method")


def _deep_seed_status(results: list[dict[str, Any]]) -> str:
    """Combine two independently run deep-seed outcomes without hiding either one."""
    statuses = {str(item["status"]) for item in results}
    if "unresolved" in statuses:
        return "unresolved"
    if "unstable" in statuses:
        return "unstable"
    if "root_move_outside_candidates" in statuses:
        return "root_move_outside_candidates"
    if "level_method_worse" in statuses:
        return "level_method_worse"
    if not _dedicated_seed_results_agree(results):
        return "dedicated_seed_disagreement"
    return "level_method_not_lower"


def _deep_seed_signature(result: dict[str, Any]) -> tuple[object, ...]:
    """Return all substantive level-33 conclusions that must agree across deep seeds."""
    queries = result.get("queries")
    if not isinstance(queries, dict):
        return (result.get("status"),)
    return (
        result.get("status"),
        result.get("root_best_move"),
        queries.get("time_move_first", {}).get("result", {}).get("score"),
        queries.get("level_move_first", {}).get("result", {}).get("score"),
    )


def _dedicated_seed_results_agree(results: list[dict[str, Any]]) -> bool:
    """Require two dedicated deep seeds to agree on outcome, root move, and both scores."""
    return len(results) == DEEP_SEED_COUNT and len({_deep_seed_signature(item) for item in results}) == 1


def _validate_deep_seed_result(
    result: object,
    board: str,
    pair_dir: Path,
    expected_seed: dict[str, int | str],
    time_move: str,
    level_move: str,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("saved dedicated level-33 seed result is not an object")
    if result.get("deep_seed_index") != expected_seed["index"] or result.get("deep_seed") != expected_seed["seed"]:
        raise ValueError("saved dedicated level-33 seed result has a different seed")
    status = result.get("status")
    allowed = {
        "level_method_better",
        "level_method_worse",
        "equal_level_33_score",
        "unstable",
        "root_move_outside_candidates",
        "unresolved",
    }
    if status not in allowed:
        raise ValueError("saved dedicated level-33 seed result has an invalid status")
    _require_positive_seconds(result.get("wall_seconds"), "saved dedicated level-33 seed wall_seconds")
    queries = result.get("queries")
    if not isinstance(queries, dict):
        raise ValueError("saved dedicated level-33 seed result has no queries")
    for query in queries.values():
        _validate_query(query, board, int(expected_seed["seed"]), pair_dir)
    if status == "unresolved":
        if not isinstance(result.get("reason"), str) or not result["reason"]:
            raise ValueError("unresolved dedicated level-33 seed result has no reason")
        return result
    expected_labels = {
        "root_first",
        "root_second",
        "time_move_first",
        "time_move_second",
        "level_move_first",
        "level_move_second",
    }
    if set(queries) != expected_labels:
        raise ValueError("resolved dedicated level-33 seed result has unexpected or missing queries")
    for label in ("root_first", "root_second"):
        _require_query(queries, label, "fixed_level_search", LEVEL_33)
    for label in ("time_move_first", "time_move_second"):
        query = _require_query(queries, label, "forced_move_analysis", LEVEL_33)
        if query.get("forced_move") != time_move:
            raise ValueError("dedicated level-33 seed used a different reference-method move")
    for label in ("level_move_first", "level_move_second"):
        query = _require_query(queries, label, "forced_move_analysis", LEVEL_33)
        if query.get("forced_move") != level_move:
            raise ValueError("dedicated level-33 seed used a different candidate-method move")
    root_move = queries["root_first"]["result"]["move"]
    root_stable = root_move == queries["root_second"]["result"]["move"]
    time_scores = [
        queries["time_move_first"]["result"]["score"],
        queries["time_move_second"]["result"]["score"],
    ]
    level_scores = [
        queries["level_move_first"]["result"]["score"],
        queries["level_move_second"]["result"]["score"],
    ]
    stable_scores = len(set(time_scores)) == 1 and len(set(level_scores)) == 1
    if result.get("root_search_move_is_stable") is not root_stable:
        raise ValueError("dedicated level-33 seed has a different root-stability record")
    if result.get("forced_move_scores_are_stable") is not stable_scores:
        raise ValueError("dedicated level-33 seed has a different score-stability record")
    if result.get("root_best_move") != root_move:
        raise ValueError("dedicated level-33 seed has a different root-best-move record")
    if not root_stable or not stable_scores:
        expected_status = "unstable"
    elif root_move not in {time_move, level_move}:
        expected_status = "root_move_outside_candidates"
    elif level_scores[0] > time_scores[0]:
        expected_status = "level_method_better"
    elif level_scores[0] < time_scores[0]:
        expected_status = "level_method_worse"
    else:
        expected_status = "equal_level_33_score"
    if status != expected_status:
        raise ValueError("dedicated level-33 seed has a different result from its raw logs")
    return result


def _validate_deep_check(
    deep: object,
    board: str,
    pair_dir: Path,
    time_move: str,
    level_move: str,
    deep_seeds: list[dict[str, int | str]],
) -> None:
    if not isinstance(deep, dict):
        raise ValueError("saved level-33 comparison is not an object")
    _require_positive_seconds(deep.get("wall_seconds"), "saved level-33 comparison wall_seconds")
    results = deep.get("dedicated_seed_results")
    if not isinstance(results, list) or len(results) != len(deep_seeds):
        raise ValueError("saved level-33 comparison has a different dedicated-seed result count")
    validated = [
        _validate_deep_seed_result(item, board, pair_dir, seed, time_move, level_move)
        for item, seed in zip(results, deep_seeds)
    ]
    if deep.get("dedicated_seed_results_agree") is not _dedicated_seed_results_agree(validated):
        raise ValueError("saved level-33 comparison has a different dedicated-seed agreement record")
    expected_status = _deep_seed_status(validated)
    if deep.get("status") != expected_status:
        raise ValueError("saved level-33 comparison has a different combined status")


def _validate_pair_payload(
    payload: object,
    plan: dict[str, Any],
    output_dir: Path,
    deep_seeds: list[dict[str, int | str]],
) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != FORMAL_PAIR_SCHEMA:
        raise ValueError("saved calculation pair has an unsupported schema")
    for field in (
        "schedule_index",
        "pair_id",
        "position_index",
        "board",
        "engine_seed_index",
        "engine_seed",
        "repetition",
        "execution_order",
        "execution_order_id",
    ):
        if payload.get(field) != plan.get(field):
            raise ValueError(f"saved calculation pair has a different {field}")
    pair_dir = _pair_directory(output_dir, plan)
    if Path(str(payload.get("pair_directory", ""))).resolve() != pair_dir.resolve():
        raise ValueError("saved calculation pair has a different directory")
    methods = payload.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(METHODS):
        raise ValueError("saved calculation pair has different method results")
    for name in METHODS:
        _validate_method_result(methods[name], name, str(plan["board"]), int(plan["engine_seed"]), pair_dir)
    accepted = [name for name in METHODS if methods[name].get("status") == "accepted"]
    deep = payload.get("level_33_comparison")
    require_deep = (
        len(accepted) == 2 and methods[TIME_METHOD].get("move") != methods[LEVEL_METHOD].get("move")
    )
    if require_deep:
        _validate_deep_check(
            deep,
            str(plan["board"]),
            pair_dir,
            str(methods[TIME_METHOD]["move"]),
            str(methods[LEVEL_METHOD]["move"]),
            deep_seeds,
        )
    elif deep is not None:
        raise ValueError("saved calculation pair has an unnecessary level-33 comparison")


def _load_progress(
    output_dir: Path,
    state_path: Path,
    plan: list[dict[str, Any]],
    deep_seeds: list[dict[str, int | str]],
) -> list[dict[str, Any]]:
    path = _progress_path(output_dir)
    if not path.exists():
        return []
    payload = _read_json(path, "comparison progress")
    if payload.get("schema") != FORMAL_PROGRESS_SCHEMA:
        raise ValueError("comparison progress has an unsupported schema")
    if payload.get("experiment_state_sha256") != sha256_file(state_path):
        raise ValueError("comparison progress was made with a different immutable state")
    if payload.get("planned_pairs") != len(plan):
        raise ValueError("comparison progress has a different planned-pair count")
    records = payload.get("records")
    if not isinstance(records, list) or payload.get("completed_pairs") != len(records):
        raise ValueError("comparison progress has invalid records")
    if len(records) > len(plan):
        raise ValueError("comparison progress records too many pairs")
    for expected, record in zip(plan, records):
        if not isinstance(record, dict):
            raise ValueError("comparison progress has a non-object record")
        if record.get("schedule_index") != expected["schedule_index"] or record.get("pair_id") != expected["pair_id"]:
            raise ValueError("comparison progress is not a prefix of the fixed pair plan")
        pair_path = _pair_result_path(output_dir, expected)
        if record.get("pair_result_path") != pair_path.resolve().as_posix():
            raise ValueError("comparison progress has a different pair-result path")
        if not pair_path.is_file() or sha256_file(pair_path) != record.get("pair_result_sha256"):
            raise ValueError("comparison progress pair result is missing or changed")
        pair_payload = _read_json(pair_path, "saved calculation pair")
        _validate_pair_payload(pair_payload, expected, output_dir, deep_seeds)
    return records


def _archive_incomplete_pair(output_dir: Path, plan: dict[str, Any], reason: str) -> None:
    """Keep incomplete artifacts for inspection and never reuse them as a pair."""
    source = _pair_directory(output_dir, plan)
    if not source.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_dir / "interrupted_attempts" / f"{source.name}_{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    journal = output_dir / "interrupted_attempts" / "archived_attempts.jsonl"
    record = {
        "schema": "formal_root_teacher_method_comparison_archived_attempt_v1",
        "at_utc": _now_utc(),
        "pair_id": plan["pair_id"],
        "schedule_index": plan["schedule_index"],
        "reason": reason,
        "from": source.resolve().as_posix(),
        "to": destination.resolve().as_posix(),
    }
    with journal.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reconcile_incomplete_pairs(
    output_dir: Path, plan: list[dict[str, Any]], complete_records: list[dict[str, Any]]
) -> None:
    complete_count = len(complete_records)
    expected_names = {_pair_directory(output_dir, item).name for item in plan}
    pairs_root = output_dir / "pairs"
    if pairs_root.exists():
        for directory in pairs_root.iterdir():
            if not directory.is_dir() or directory.name not in expected_names:
                raise ValueError(f"unexpected entry in pairs directory: {directory}")
    for item in plan[complete_count:]:
        _archive_incomplete_pair(
            output_dir,
            item,
            "calculation pair was not present in the atomically written comparison progress",
        )


def _build_command(
    exe: Path,
    kind: str,
    engine_seed: int,
    *,
    level: int | None = None,
    time_seconds: float | None = None,
) -> list[str]:
    command = teacher._build_search_command(
        kind,
        exe,
        time_seconds=time_seconds,
        level=level,
        threads=THREADS,
        hash_level=HASH_LEVEL,
        random_seed=engine_seed,
    )
    if "-seed" not in command or command[command.index("-seed") + 1] != str(engine_seed):
        raise RuntimeError("the Console search command did not contain the planned -seed argument")
    if any(option not in command for option in BOOK_DISABLED_ARGUMENTS):
        raise RuntimeError("the Console search command did not disable both books")
    return command


def _run_console(
    exe: Path,
    board: str,
    kind: str,
    engine_seed: int,
    log_path: Path,
    *,
    level: int | None = None,
    time_seconds: float | None = None,
    forced_move: str | None = None,
) -> dict[str, Any]:
    command = _build_command(exe, kind if kind != "forced_move_analysis" else "fixed_level_search", engine_seed, level=level, time_seconds=time_seconds)
    if kind == "time_limited_search":
        standard_input = f"setboard {board}\ngo\nquit\n"
        timeout = max(120.0, float(time_seconds or 0.0) * 3.0 + 60.0)
    elif kind == "fixed_level_search":
        standard_input = f"setboard {board}\nhint 1\nquit\n"
        timeout = 3600.0
    elif kind == "forced_move_analysis":
        if not isinstance(forced_move, str):
            raise ValueError("forced-move analysis needs a move")
        standard_input = f"setboard {board}\nclearcache\nplay {forced_move}\nanalyze\nquit\n"
        timeout = 3600.0
    else:
        raise ValueError(f"unknown Console query kind {kind}")
    try:
        completed = subprocess.run(
            command,
            cwd=exe.parent,
            input=standard_input,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Console {kind} failed for {board}: {error}") from error
    combined = completed.stdout + "\n" + completed.stderr
    _atomic_write_text(log_path, combined)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Console {kind} exited {completed.returncode} for {board}: {combined[-400:]}"
        )
    if kind == "forced_move_analysis":
        matches = [match.groupdict() for match in ANALYZE_RESULT_RE.finditer(combined)]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one analyze result for {board} {forced_move}, found {len(matches)}"
            )
        match = matches[0]
        if match["move"] != forced_move:
            raise RuntimeError(
                f"analyze evaluated {match['move']} rather than requested move {forced_move}"
            )
        result: dict[str, int | str] = {
            "move": match["move"],
            "score": int(match["score"]),
            "depth": match["depth"],
        }
    else:
        try:
            result = _result_from_log(combined, board)
        except ValueError as error:
            raise RuntimeError(f"cannot parse Console {kind} result for {board}: {error}") from error
    if level is not None:
        validate_quality(result, max(MIN_DEPTH, level), MIN_SELECTIVITY)
    relative = log_path.resolve().relative_to(log_path.parent.resolve()).as_posix()
    query: dict[str, Any] = {
        "kind": kind,
        "command": command,
        "standard_input": standard_input,
        "result": result,
        "raw_log": {
            "relative_path": relative,
            "sha256": sha256_file(log_path),
        },
    }
    if kind == "forced_move_analysis":
        query["forced_move"] = forced_move
    return query


def _query_time(exe: Path, board: str, seed: int, pair_dir: Path, label: str) -> dict[str, Any]:
    return _run_console(
        exe,
        board,
        "time_limited_search",
        seed,
        pair_dir / f"{label}.log",
        time_seconds=TIME_SECONDS,
    )


def _query_level(
    exe: Path, board: str, seed: int, pair_dir: Path, label: str, level: int
) -> dict[str, Any]:
    return _run_console(
        exe,
        board,
        "fixed_level_search",
        seed,
        pair_dir / f"{label}.log",
        level=level,
    )


def _query_forced(
    exe: Path, board: str, seed: int, pair_dir: Path, label: str, move: str
) -> dict[str, Any]:
    return _run_console(
        exe,
        board,
        "forced_move_analysis",
        seed,
        pair_dir / f"{label}.log",
        level=LEVEL_33,
        forced_move=move,
    )


def _accepted_method_result(
    method: str,
    queries: dict[str, dict[str, Any]],
    selected_query: str,
    verification_mode: str,
    elapsed: float,
) -> dict[str, Any]:
    selected = queries[selected_query]["result"]
    return {
        "method": method,
        "status": "accepted",
        "selected_query": selected_query,
        "move": selected["move"],
        "score": selected["score"],
        "depth": selected["depth"],
        "verification_mode": verification_mode,
        "queries": queries,
        "wall_seconds": elapsed,
    }


def _rejected_method_result(
    method: str, queries: dict[str, dict[str, Any]], reason: str, elapsed: float
) -> dict[str, Any]:
    return {
        "method": method,
        "status": "rejected",
        "reason": reason,
        "queries": queries,
        "wall_seconds": elapsed,
    }


def _run_time_method(exe: Path, board: str, seed: int, pair_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    queries: dict[str, dict[str, Any]] = {}
    primary = _query_time(exe, board, seed, pair_dir, "time_primary")
    queries["primary"] = primary
    candidate_label = "primary"
    candidate = primary
    try:
        validate_quality(primary["result"], MIN_DEPTH, MIN_SELECTIVITY)
    except ValueError:
        fallback = _query_level(exe, board, seed, pair_dir, "time_fallback_level_30", LEVEL_30)
        queries["fallback"] = fallback
        candidate_label = "fallback"
        candidate = fallback
        verification_mode = "time_result_below_minimum_quality_then_level_30_fallback"
    else:
        verification_mode = "time_result_met_minimum_quality"
    verification = _query_level(exe, board, seed, pair_dir, "time_verification_level_31", LEVEL_31)
    queries["verification"] = verification
    if candidate["result"]["move"] == verification["result"]["move"]:
        return _accepted_method_result(
            TIME_METHOD,
            queries,
            candidate_label,
            verification_mode + "_level_31_exact",
            time.monotonic() - started,
        )
    tiebreak = _query_level(exe, board, seed, pair_dir, "time_tiebreak_level_30", LEVEL_30)
    queries["tiebreak"] = tiebreak
    if candidate["result"]["move"] == tiebreak["result"]["move"]:
        return _accepted_method_result(
            TIME_METHOD,
            queries,
            candidate_label,
            verification_mode + "_level_30_tiebreak",
            time.monotonic() - started,
        )
    deep_tiebreak = _query_level(exe, board, seed, pair_dir, "time_tiebreak_level_31", LEVEL_31)
    queries["deep_tiebreak"] = deep_tiebreak
    if tiebreak["result"]["move"] != deep_tiebreak["result"]["move"]:
        return _rejected_method_result(
            TIME_METHOD,
            queries,
            "level-30 and repeated level-31 tiebreak moves differ",
            time.monotonic() - started,
        )
    return _accepted_method_result(
        TIME_METHOD,
        queries,
        "deep_tiebreak",
        verification_mode + "_levels_30_31_tiebreak",
        time.monotonic() - started,
    )


def _run_level_method(exe: Path, board: str, seed: int, pair_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    queries: dict[str, dict[str, Any]] = {}
    teacher_result = _query_level(exe, board, seed, pair_dir, "level_teacher_level_30", LEVEL_30)
    queries["teacher"] = teacher_result
    verification = _query_level(exe, board, seed, pair_dir, "level_verification_level_31", LEVEL_31)
    queries["verification"] = verification
    if teacher_result["result"]["move"] == verification["result"]["move"]:
        return _accepted_method_result(
            LEVEL_METHOD,
            queries,
            "verification",
            "level_31_exact",
            time.monotonic() - started,
        )
    repeated = _query_level(exe, board, seed, pair_dir, "level_verification_level_31_repeat", LEVEL_31)
    queries["verification_repeat"] = repeated
    if verification["result"]["move"] != repeated["result"]["move"]:
        return _rejected_method_result(
            LEVEL_METHOD,
            queries,
            "two level-31 verification moves differ after a level-30 disagreement",
            time.monotonic() - started,
        )
    return _accepted_method_result(
        LEVEL_METHOD,
        queries,
        "verification_repeat",
        "level_31_repeated_after_level_30_disagreement",
        time.monotonic() - started,
    )


def _run_one_dedicated_level_33_seed(
    exe: Path,
    board: str,
    deep_seed: dict[str, int | str],
    time_move: str,
    level_move: str,
    pair_dir: Path,
) -> dict[str, Any]:
    """Run the complete level-33 evidence set under one seed reserved for this purpose."""
    started = time.monotonic()
    queries: dict[str, dict[str, Any]] = {}
    seed = int(deep_seed["seed"])
    label_prefix = f"level_33_deep_seed_{int(deep_seed['index'])}"
    try:
        root_first = _query_level(exe, board, seed, pair_dir, f"{label_prefix}_root_first", LEVEL_33)
        queries["root_first"] = root_first
        root_second = _query_level(exe, board, seed, pair_dir, f"{label_prefix}_root_second", LEVEL_33)
        queries["root_second"] = root_second
        time_first = _query_forced(exe, board, seed, pair_dir, f"{label_prefix}_time_move_first", time_move)
        queries["time_move_first"] = time_first
        time_second = _query_forced(exe, board, seed, pair_dir, f"{label_prefix}_time_move_second", time_move)
        queries["time_move_second"] = time_second
        level_first = _query_forced(exe, board, seed, pair_dir, f"{label_prefix}_level_move_first", level_move)
        queries["level_move_first"] = level_first
        level_second = _query_forced(exe, board, seed, pair_dir, f"{label_prefix}_level_move_second", level_move)
        queries["level_move_second"] = level_second
    except (RuntimeError, ValueError) as error:
        return {
            "deep_seed_index": deep_seed["index"],
            "deep_seed": seed,
            "status": "unresolved",
            "reason": str(error),
            "queries": queries,
            "wall_seconds": time.monotonic() - started,
        }
    root_stable = root_first["result"]["move"] == root_second["result"]["move"]
    time_scores = [time_first["result"]["score"], time_second["result"]["score"]]
    level_scores = [level_first["result"]["score"], level_second["result"]["score"]]
    stable_scores = len(set(time_scores)) == 1 and len(set(level_scores)) == 1
    payload: dict[str, Any] = {
        "deep_seed_index": deep_seed["index"],
        "deep_seed": seed,
        "level": LEVEL_33,
        "root_search_move_is_stable": root_stable,
        "forced_move_scores_are_stable": stable_scores,
        "root_best_move": root_first["result"]["move"],
        "queries": queries,
        "wall_seconds": time.monotonic() - started,
    }
    if not root_stable or not stable_scores:
        payload["status"] = "unstable"
        payload["reason"] = "repeated level-33 root move or forced-move score was not stable"
    elif root_first["result"]["move"] not in {time_move, level_move}:
        payload["status"] = "root_move_outside_candidates"
        payload["reason"] = "stable level-33 root move is neither accepted candidate move"
    elif level_scores[0] > time_scores[0]:
        payload["status"] = "level_method_better"
    elif level_scores[0] < time_scores[0]:
        payload["status"] = "level_method_worse"
    else:
        payload["status"] = "equal_level_33_score"
    return payload


def _run_level_33_comparison(
    exe: Path,
    board: str,
    deep_seeds: list[dict[str, int | str]],
    time_move: str,
    level_move: str,
    pair_dir: Path,
) -> dict[str, Any]:
    """Use both predeclared dedicated seeds; neither is the pair's main seed."""
    started = time.monotonic()
    results = [
        _run_one_dedicated_level_33_seed(
            exe, board, seed, time_move, level_move, pair_dir
        )
        for seed in deep_seeds
    ]
    return {
        "level": LEVEL_33,
        "dedicated_seed_results": results,
        "dedicated_seed_results_agree": _dedicated_seed_results_agree(results),
        "status": _deep_seed_status(results),
        "wall_seconds": time.monotonic() - started,
    }


def _run_pair(
    exe: Path,
    output_dir: Path,
    plan: dict[str, Any],
    deep_seeds: list[dict[str, int | str]],
) -> dict[str, Any]:
    """Run exactly both methods, then the required level-33 check, without interleaving another pair."""
    pair_dir = _pair_directory(output_dir, plan)
    if pair_dir.exists():
        raise FileExistsError(f"pair directory already exists before a new attempt: {pair_dir}")
    pair_dir.mkdir(parents=True)
    _atomic_write_text(
        pair_dir / "pair_started.json",
        _canonical_json(
            {
                "schema": "formal_root_teacher_method_comparison_pair_started_v1",
                "at_utc": _now_utc(),
                "pair_id": plan["pair_id"],
                "schedule_index": plan["schedule_index"],
                "execution_order": plan["execution_order"],
            }
        ),
    )
    methods: dict[str, dict[str, Any]] = {}
    for method_name in plan["execution_order"]:
        if method_name == TIME_METHOD:
            methods[method_name] = _run_time_method(
                exe, str(plan["board"]), int(plan["engine_seed"]), pair_dir
            )
        elif method_name == LEVEL_METHOD:
            methods[method_name] = _run_level_method(
                exe, str(plan["board"]), int(plan["engine_seed"]), pair_dir
            )
        else:
            raise RuntimeError("pair plan has an unknown method")
    level_33: dict[str, Any] | None = None
    if (
        methods[TIME_METHOD]["status"] == "accepted"
        and methods[LEVEL_METHOD]["status"] == "accepted"
        and methods[TIME_METHOD]["move"] != methods[LEVEL_METHOD]["move"]
    ):
        level_33 = _run_level_33_comparison(
            exe,
            str(plan["board"]),
            deep_seeds,
            str(methods[TIME_METHOD]["move"]),
            str(methods[LEVEL_METHOD]["move"]),
            pair_dir,
        )
    payload: dict[str, Any] = {
        "schema": FORMAL_PAIR_SCHEMA,
        "pair_directory": pair_dir.resolve().as_posix(),
        "completed_at_utc": _now_utc(),
        **{key: plan[key] for key in (
            "schedule_index",
            "pair_id",
            "position_index",
            "board",
            "engine_seed_index",
            "engine_seed",
            "repetition",
            "execution_order",
            "execution_order_id",
        )},
        "methods": methods,
        "level_33_comparison": level_33,
    }
    _atomic_write_text(_pair_result_path(output_dir, plan), _canonical_json(payload))
    return payload


def _pair_outcome_counts(pair_payloads: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "both_accepted": 0,
        "only_time_method_accepted": 0,
        "only_level_method_accepted": 0,
        "both_rejected": 0,
        "same_accepted_move": 0,
        "different_accepted_move": 0,
        "level_method_not_lower": 0,
        "level_method_worse": 0,
        "unstable_level_33": 0,
        "root_move_outside_candidates": 0,
        "unresolved_level_33": 0,
        "dedicated_deep_seed_disagreement": 0,
    }
    for payload in pair_payloads:
        methods = payload["methods"]
        time_accepted = methods[TIME_METHOD]["status"] == "accepted"
        level_accepted = methods[LEVEL_METHOD]["status"] == "accepted"
        if time_accepted and level_accepted:
            counts["both_accepted"] += 1
            if methods[TIME_METHOD]["move"] == methods[LEVEL_METHOD]["move"]:
                counts["same_accepted_move"] += 1
            else:
                counts["different_accepted_move"] += 1
                deep = payload["level_33_comparison"]
                if not deep["dedicated_seed_results_agree"]:
                    counts["dedicated_deep_seed_disagreement"] += 1
                status = deep["status"]
                if status == "level_method_not_lower":
                    counts["level_method_not_lower"] += 1
                elif status == "level_method_worse":
                    counts["level_method_worse"] += 1
                elif status == "dedicated_seed_disagreement":
                    # The separate flag above records this failure even when a more
                    # serious lower-score result also occurs in another deep seed.
                    pass
                elif status == "unstable":
                    counts["unstable_level_33"] += 1
                elif status == "root_move_outside_candidates":
                    counts["root_move_outside_candidates"] += 1
                elif status == "unresolved":
                    counts["unresolved_level_33"] += 1
                else:
                    raise ValueError("saved level-33 comparison has an unknown combined status")
        elif time_accepted:
            counts["only_time_method_accepted"] += 1
        elif level_accepted:
            counts["only_level_method_accepted"] += 1
        else:
            counts["both_rejected"] += 1
    return counts


def _dedicated_deep_seed_counts(pair_payloads: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Keep the two reserved deep seeds separate in the final report."""
    result: dict[str, dict[str, int]] = {}
    allowed = (
        "level_method_better",
        "level_method_worse",
        "equal_level_33_score",
        "unstable",
        "root_move_outside_candidates",
        "unresolved",
    )
    for payload in pair_payloads:
        deep = payload.get("level_33_comparison")
        if not isinstance(deep, dict):
            continue
        for item in deep["dedicated_seed_results"]:
            key = str(item["deep_seed_index"])
            if key not in result:
                result[key] = {status: 0 for status in allowed}
            result[key][item["status"]] += 1
    return result


def _repetition_consistency(pair_payloads: list[dict[str, Any]]) -> dict[str, int]:
    """Require both repetitions to give one reproducible outcome per board and main seed."""
    groups: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for payload in pair_payloads:
        key = (str(payload["board"]), int(payload["engine_seed_index"]))
        repetitions = groups.setdefault(key, {})
        repetition = int(payload["repetition"])
        if repetition in repetitions:
            raise ValueError("duplicate repetition for one board and main engine seed")
        repetitions[repetition] = payload
    expected_groups = POSITION_COUNT * ENGINE_SEED_COUNT
    if len(groups) != expected_groups:
        raise ValueError("completed results do not contain every board and main engine-seed group")
    summary = {
        "groups": len(groups),
        "time_method_disagreements": 0,
        "level_method_disagreements": 0,
        "groups_with_any_method_disagreement": 0,
    }
    for repetitions in groups.values():
        if set(repetitions) != {1, 2}:
            raise ValueError("board and main engine-seed group does not have both repetitions")
        any_disagreement = False
        for method, field in (
            (TIME_METHOD, "time_method_disagreements"),
            (LEVEL_METHOD, "level_method_disagreements"),
        ):
            first = repetitions[1]["methods"][method]
            second = repetitions[2]["methods"][method]
            disagreement = first["status"] != second["status"]
            if not disagreement and first["status"] == "accepted":
                disagreement = first["move"] != second["move"]
            if disagreement:
                summary[field] += 1
                any_disagreement = True
        if any_disagreement:
            summary["groups_with_any_method_disagreement"] += 1
    return summary


def _bootstrap_cluster_time_ratio(
    pair_payloads: list[dict[str, Any]], bootstrap_seed: int
) -> dict[str, float | int]:
    """Bootstrap by starting position so its eight calculations stay together."""
    grouped: dict[int, list[tuple[float, float]]] = {}
    for payload in pair_payloads:
        methods = payload["methods"]
        grouped.setdefault(int(payload["position_index"]), []).append(
            (float(methods[LEVEL_METHOD]["wall_seconds"]), float(methods[TIME_METHOD]["wall_seconds"]))
        )
    if len(grouped) != POSITION_COUNT or any(len(rows) != ENGINE_SEED_COUNT * REPETITION_COUNT for rows in grouped.values()):
        raise ValueError("completed results do not contain eight pairs for every selected position")
    clusters = [
        (sum(level for level, _ in rows), sum(reference for _, reference in rows))
        for _index, rows in sorted(grouped.items())
    ]
    if any(reference <= 0.0 for _, reference in clusters):
        raise ValueError("reference wall time must be positive")
    generator = random.Random(bootstrap_seed)
    ratios: list[float] = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        level_total = 0.0
        reference_total = 0.0
        for _ in clusters:
            level, reference = clusters[generator.randrange(len(clusters))]
            level_total += level
            reference_total += reference
        ratios.append(level_total / reference_total)
    ratios.sort()
    level_total = sum(level for level, _ in clusters)
    reference_total = sum(reference for _, reference in clusters)
    return {
        "point_estimate": level_total / reference_total,
        "lower_95_percent": ratios[math.floor(0.025 * (BOOTSTRAP_REPETITIONS - 1))],
        "upper_95_percent": ratios[math.ceil(0.975 * (BOOTSTRAP_REPETITIONS - 1))],
        "seed": bootstrap_seed,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "cluster_count": len(clusters),
    }


def _selection_conditions(
    rule: dict[str, Any],
    counts: dict[str, int],
    repetitions: dict[str, int],
    timing: dict[str, float | int],
    completed_pairs: int,
) -> dict[str, bool]:
    """Apply only the condition values frozen in the immutable decision rule."""
    return {
        "all_pairs_complete": completed_pairs == int(rule["required_complete_pairs"]),
        "repetitions_consistent_for_both_methods": repetitions["groups_with_any_method_disagreement"] == 0,
        "no_level_method_worse_at_level_33": counts["level_method_worse"] == 0,
        "no_unstable_dedicated_level_33_result": counts["unstable_level_33"] == 0,
        "no_dedicated_level_33_root_move_outside_candidates": counts["root_move_outside_candidates"] == 0,
        "dedicated_level_33_seeds_agree": counts["dedicated_deep_seed_disagreement"] == 0,
        "no_unresolved_level_33": counts["unresolved_level_33"] == 0,
        "no_reference_only_acceptance": counts["only_time_method_accepted"] == 0,
        "paired_time_ratio_upper_95_percent_below_0_90": float(timing["upper_95_percent"]) < 0.90,
    }


def _summarize_completed_pairs(
    output_dir: Path, plan: list[dict[str, Any]], records: list[dict[str, Any]], state: dict[str, Any]
) -> dict[str, Any]:
    pair_payloads = [_read_json(_pair_result_path(output_dir, item), "saved calculation pair") for item in plan]
    counts = _pair_outcome_counts(pair_payloads)
    deep_seed_counts = _dedicated_deep_seed_counts(pair_payloads)
    repetitions = _repetition_consistency(pair_payloads)
    bootstrap_seed = _derive_u64(
        BOOTSTRAP_SEED_DOMAIN,
        str(state["selection"]["selection_id"]),
        str(state["pair_plan_sha256"]),
    )
    timing = _bootstrap_cluster_time_ratio(pair_payloads, bootstrap_seed)
    rule = state["decision_protocol"]
    selection_conditions = _selection_conditions(
        rule, counts, repetitions, timing, len(records)
    )
    can_select = all(selection_conditions.values())
    if can_select:
        summary_ja = (
            "事前に固定した判定規則の全条件を満たしたため、この比較の対象である残り局面の計算には"
            "level 30・level 31照合を使える。ただし、これは計算方法の選択だけを扱い、対戦での強さを示さない。"
        )
        summary_en = (
            "Every precommitted decision condition is met, so the level-30/level-31 verification method "
            "may be used for calculation of the remaining positions covered by this comparison. This is a "
            "calculation-method decision only and is not evidence of playing strength."
        )
    else:
        summary_ja = (
            "事前に固定した判定規則の少なくとも一条件を満たさなかったため、この比較だけを根拠に"
            "level 30・level 31照合を残り局面の計算方法として選ばない。"
        )
        summary_en = (
            "At least one precommitted decision condition is not met, so this comparison alone does not "
            "select the level-30/level-31 verification method for the remaining positions."
        )
    return {
        "schema": FORMAL_REPORT_SCHEMA,
        "experiment_state_sha256": sha256_file(_state_path(output_dir)),
        "completed_pairs": len(records),
        "pair_outcomes": counts,
        "dedicated_level_33_seed_outcomes": deep_seed_counts,
        "repetition_consistency": repetitions,
        "paired_wall_time": timing,
        "decision_protocol": rule,
        "decision_conditions": selection_conditions,
        "level_30_then_level_31_can_continue_to_larger_calculation": can_select,
        "summary_ja": summary_ja,
        "summary_en": summary_en,
    }


def _term_definitions_markdown() -> str:
    """Provide the seven requested definition fields for every nonordinary report term."""
    return """## 用語の定義 / Definitions of terms used in this report

### 位置選択ID / position-selection ID

- 出典 / Source: このスクリプトの `SELECTION_ID_DOMAIN` と、対象局面集合・除外集合・選ばれた600局面の SHA-256 値から作る識別子。 / The identifier is made by this script from `SELECTION_ID_DOMAIN` and SHA-256 values of the corpus, exclusions, and selected 600 positions.
- 目的 / Purpose: 再開時や監査時に、同じ600局面を指していることを確認する。 / It confirms that resume and audit refer to the same 600 positions.
- 具体対象 / Concrete target: `experiment_state.json` の `selection`。 / `selection` in `experiment_state.json`.
- 役割 / Role: 局面の選択を、計算実行順や Console の乱数 seed と区別して固定する。 / It fixes position selection separately from execution order and Console random seeds.
- 前後関係 / Context: 除外済み局面を取り除いた後、計算組を作る前に作成する。 / It is created after exclusions are applied and before calculation pairs are made.
- 候補語 / Candidate terms: 局面選択番号、標本ID。 / Candidate terms considered: position-selection number, sample ID.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.

### 実行順ID / execution-order ID

- 出典 / Source: このスクリプトの `ORDER_ID_DOMAIN`、位置選択ID、局面、Console の乱数 seed から作る SHA-256 値。 / This SHA-256 value is made from `ORDER_ID_DOMAIN`, the position-selection ID, a board, and a Console random seed.
- 目的 / Purpose: 二つの計算方法をどちらから行うかを後で確認可能にする。 / It makes the first of the two calculation methods auditable.
- 具体対象 / Concrete target: 一つの局面と一つの Console の乱数 seed に対する、反復1の二方法の順序。 / The order of the two methods in repetition 1 for one board and one Console random seed.
- 役割 / Role: 反復2ではこの順序を必ず反転させ、順序による偏りを二回の反復で打ち消す。 / Repetition 2 must reverse this order, balancing order effects across the two repetitions.
- 前後関係 / Context: 位置選択IDの作成後、計算組の計画作成時に作成する。 / It is created when the pair plan is made after the position-selection ID.
- 候補語 / Candidate terms: 順序番号、実施順ID。 / Candidate terms considered: order number, run-order ID.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.

### 計算組 / calculation pair

- 出典 / Source: このスクリプトの `pair_plan` と `PAIR_ID_DOMAIN`。 / The `pair_plan` and `PAIR_ID_DOMAIN` of this script.
- 目的 / Purpose: 同じ局面・同じ Console の乱数 seed・同じ反復で二つの計算方法を連続して比べる最小単位を明確にする。 / It defines the smallest unit that compares both methods consecutively under one board, Console seed, and repetition.
- 具体対象 / Concrete target: `pair_result.json` 一ファイルに記録する二方法の結果と、必要時の level 33 比較。 / The two method results and, if needed, level-33 comparison recorded in one `pair_result.json`.
- 役割 / Role: 片方だけ終わった状態を結果として使わず、中断時には丸ごと再計算させる。 / It prevents one-method results from being used; an interrupted pair is rerun as a whole.
- 前後関係 / Context: 600局面、4 seed、2反復から 4,800 組を作る。 / The plan creates 4,800 pairs from 600 boards, four seeds, and two repetitions.
- 候補語 / Candidate terms: 比較単位、二方法組。 / Candidate terms considered: comparison unit, two-method unit.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.

### 事前固定の判定規則 / precommitted decision rule

- 出典 / Source: `experiment_state.json` の `decision_protocol`。 / `decision_protocol` in `experiment_state.json`.
- 目的 / Purpose: 結果を見る前に、どの条件なら計算方法を選べるかを固定する。 / It fixes the conditions for selecting a calculation method before results are seen.
- 具体対象 / Concrete target: 4,800組の完了、反復間一致、二つの専用deep seedによる level 33比較、受理状況、対応を保った時間比の95%区間。 / Completion of 4,800 pairs, repetition consistency, level-33 checks under two dedicated deep seeds, acceptance outcomes, and the paired 95% wall-time interval.
- 役割 / Role: 後から都合のよい条件だけを選ぶことを防ぐ。 / It prevents choosing favorable conditions after results are known.
- 前後関係 / Context: 計算開始前に状態ファイルへ保存し、全組完了後にだけ適用する。 / It is saved before calculation starts and applied only after every pair completes.
- 候補語 / Candidate terms: 事前登録規則、選択条件。 / Candidate terms considered: preregistration rule, selection conditions.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.

### level 33 比較 / level-33 comparison

- 出典 / Source: このスクリプトの `different_move_level_33` 条件と `_run_level_33_comparison`。 / The `different_move_level_33` condition and `_run_level_33_comparison` in this script.
- 目的 / Purpose: 二方法が受理した最初の手が異なるとき、その二手をより深い一定条件で比べる。 / It compares the two accepted moves at a deeper fixed condition when the methods choose different moves.
- 具体対象 / Concrete target: level 33 の最初の手探索を2回と、各候補手に対する `setboard`、`play`、`analyze` を各2回。 / Two level-33 root searches plus two `setboard`, `play`, `analyze` evaluations for each candidate move.
- 役割 / Role: 受理した異なる二手を、main engine seed とは別の二つの専用deep seed、28スレッド、hash 29、book無効で比較する。 / It compares the different accepted moves with two dedicated deep seeds distinct from the main engine seed, 28 threads, hash 29, and both books disabled.
- 前後関係 / Context: 計算組内で二方法を連続実行した後、かつ両方が異なる手を受理した場合だけ実行する。 / It runs after both methods in a calculation pair and only when both accept different moves.
- 候補語 / Candidate terms: 深い照合、level 33検査。 / Candidate terms considered: deep verification, level-33 check.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.

### main engine seed / 主計算seed

- 出典 / Source: `engine_seed_derivation` の SHA-256 domain string と corpus fingerprint。 / The SHA-256 domain string and corpus fingerprint in `engine_seed_derivation`.
- 目的 / Purpose: 二方法の通常計算を、同じ Console の乱数初期状態で比較する。 / It compares the ordinary calculations of both methods from the same Console random initial state.
- 具体対象 / Concrete target: 各計算組の `engine_seed`。 / `engine_seed` in each calculation pair.
- 役割 / Role: 二方法の連続実行と反復1・反復2の両方に使う4個の seed である。 / These are the four seeds used for consecutive method execution in both repetitions.
- 前後関係 / Context: 局面選択後に計算組を作るときに割り当て、異手時の level 33 比較には使わない。 / They are assigned while making pairs after position selection and are not used for level-33 comparisons of differing moves.
- 候補語 / Candidate terms: 通常seed、比較seed。 / Candidate terms considered: ordinary seed, comparison seed.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.

### 専用deep seed / dedicated deep seed

- 出典 / Source: `deep_seed_derivation` の `DEEP_SEED_DOMAIN` と corpus fingerprint。 / `DEEP_SEED_DOMAIN` and corpus fingerprint in `deep_seed_derivation`.
- 目的 / Purpose: 異なる二手の level 33 比較を、通常計算とは別の乱数初期状態で二回確認する。 / It checks a level-33 comparison of different moves twice from random initial states separate from ordinary calculation.
- 具体対象 / Concrete target: `level_33_comparison.dedicated_seed_results` の二つの `deep_seed`。 / The two `deep_seed` values in `level_33_comparison.dedicated_seed_results`.
- 役割 / Role: 各 seed で最初の手探索2回と両候補手の forced analyze 各2回を実行し、二つの seed の最善手と両候補手の評価も一致させる。 / For each seed it runs two root searches and two forced analyses of each candidate move, then requires the two seeds to agree on the root move and both candidate scores.
- 前後関係 / Context: 二方法が異なる手を受理した後にだけ使い、main engine seed と同じ値にならないよう事前導出する。 / It is used only after both methods accept different moves and is derived in advance to differ from every main engine seed.
- 候補語 / Candidate terms: 深い照合seed、独立seed。 / Candidate terms considered: deep-check seed, independent seed.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.

### 反復間一致 / repetition consistency

- 出典 / Source: このスクリプトの `_repetition_consistency` と `decision_protocol`。 / `_repetition_consistency` and `decision_protocol` in this script.
- 目的 / Purpose: 同じ局面と main engine seed で反復1・反復2が同じ結論を出すことを確認する。 / It checks that repetitions 1 and 2 give the same conclusion for one board and main engine seed.
- 具体対象 / Concrete target: 各方法の受理・却下、受理時の採用手。 / Each method's accepted/rejected status and, when accepted, selected move.
- 役割 / Role: 一方の方法だけの反復不一致でも候補方法を選ばない条件にする。 / A discrepancy in either method prevents selection of the candidate method.
- 前後関係 / Context: 全4,800組が完了した後、時間比の集計と同時に判定する。 / It is evaluated after all 4,800 pairs complete alongside the wall-time summary.
- 候補語 / Candidate terms: 再現一致、二反復一致。 / Candidate terms considered: reproducibility agreement, two-repetition agreement.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.

### 対応を保った時間比 / paired wall-time ratio

- 出典 / Source: このスクリプトの `_bootstrap_cluster_time_ratio` と `decision_protocol.bootstrap`。 / `_bootstrap_cluster_time_ratio` and `decision_protocol.bootstrap` in this script.
- 目的 / Purpose: 同じ開始局面に属する8組の計測を一緒に扱い、二方法の実時間を比べる。 / It compares wall time while keeping together the eight pairs belonging to each starting position.
- 具体対象 / Concrete target: 各局面についての level 30・level 31照合の実時間合計を、60秒探索の実時間合計で割った値。 / For each board, the total level-30/level-31 verification time divided by the total 60-second-search time.
- 役割 / Role: 600局面を復元抽出する95%区間の単位を、個々の検索ではなく開始局面にする。 / It makes the starting position, not an individual search, the resampling unit for the 95% interval.
- 前後関係 / Context: 全4,800組が終わった後にだけ集計し、事前固定の判定規則の速度条件に使う。 / It is calculated only after all 4,800 pairs finish and is used by the speed condition of the precommitted decision rule.
- 候補語 / Candidate terms: 対応時間比、局面単位時間比。 / Candidate terms considered: matched time ratio, position-level time ratio.
- 初出定義 / First definition: 本節のこの項目。 / First defined here in this item.
"""


def _write_readme(output_dir: Path, state: dict[str, Any], records: list[dict[str, Any]]) -> None:
    selection = state["selection"]
    plan = state["pair_plan"]
    completed = len(records)
    final = completed == len(plan)
    report_section = ""
    if final:
        report = _summarize_completed_pairs(output_dir, plan, records, state)
        _atomic_write_text(output_dir / "formal_comparison_report.json", _canonical_json(report))
        outcomes = report["pair_outcomes"]
        deep_seed_outcomes = report["dedicated_level_33_seed_outcomes"]
        repetitions = report["repetition_consistency"]
        timing = report["paired_wall_time"]
        report_section = f"""
## 結果 / Results

### 日本語

- 両方で受理: {outcomes['both_accepted']}組
- 60秒探索だけで受理: {outcomes['only_time_method_accepted']}組
- level 30・level 31照合だけで受理: {outcomes['only_level_method_accepted']}組
- 両方で却下: {outcomes['both_rejected']}組
- 両方で受理され、手が異なる: {outcomes['different_accepted_move']}組
- 反復1・反復2の不一致: 60秒探索 {repetitions['time_method_disagreements']}、level 30・level 31照合 {repetitions['level_method_disagreements']}、どちらかが不一致の局面×main seed組 {repetitions['groups_with_any_method_disagreement']} / {repetitions['groups']}
- level 33で level 30・level 31照合の手が低い: {outcomes['level_method_worse']}組
- 専用deep seedの不安定: {outcomes['unstable_level_33']}組
- 専用deep seedの最善手が候補外: {outcomes['root_move_outside_candidates']}組
- 二つの専用deep seedの結論不一致: {outcomes['dedicated_deep_seed_disagreement']}組
- level 33で判定不能: {outcomes['unresolved_level_33']}組
- 専用deep seed別の結果: `{json.dumps(deep_seed_outcomes, ensure_ascii=False, sort_keys=True)}`
- 対応を保った時間比（level 30・level 31照合 ÷ 60秒探索）の点推定: {timing['point_estimate']:.4f}
- 同時間比の95%区間: {timing['lower_95_percent']:.4f} から {timing['upper_95_percent']:.4f}

{report['summary_ja']}

### English

- Accepted by both: {outcomes['both_accepted']} pairs
- Accepted only by the 60-second search: {outcomes['only_time_method_accepted']} pairs
- Accepted only by the level-30/level-31 verification: {outcomes['only_level_method_accepted']} pairs
- Rejected by both: {outcomes['both_rejected']} pairs
- Accepted by both with different moves: {outcomes['different_accepted_move']} pairs
- Repetition-1/repetition-2 disagreements: 60-second search {repetitions['time_method_disagreements']}, level-30/level-31 verification {repetitions['level_method_disagreements']}, board × main-seed groups with either disagreement {repetitions['groups_with_any_method_disagreement']} / {repetitions['groups']}
- Level-30/level-31 move lower at level 33: {outcomes['level_method_worse']} pairs
- Unstable dedicated deep-seed result: {outcomes['unstable_level_33']} pairs
- Dedicated deep-seed root move outside both candidates: {outcomes['root_move_outside_candidates']} pairs
- Disagreement between the two dedicated deep seeds: {outcomes['dedicated_deep_seed_disagreement']} pairs
- Unresolved at level 33: {outcomes['unresolved_level_33']} pairs
- Outcomes by dedicated deep seed: `{json.dumps(deep_seed_outcomes, ensure_ascii=False, sort_keys=True)}`
- Paired wall-time ratio (level-30/level-31 verification / 60-second search), point estimate: {timing['point_estimate']:.4f}
- 95% interval for that ratio: {timing['lower_95_percent']:.4f} to {timing['upper_95_percent']:.4f}

{report['summary_en']}
"""
    else:
        report_section = f"""
## 進捗 / Progress

### 日本語

- 完了した計算組: {completed} / {len(plan)}
- 未完了の計算組: {len(plan) - completed}

まだ全4,800組が完了していないため、計算方法を選ぶ判定は行わない。中断後は同じ引数で `--resume` を指定する。進捗に記録されていない途中の計算組は保存資料へ移し、最初から再計算する。

### English

- Completed calculation pairs: {completed} / {len(plan)}
- Calculation pairs not complete: {len(plan) - completed}

No method-selection decision is made before all 4,800 pairs complete. Resume with the same arguments and `--resume`. Any interrupted pair absent from durable progress is preserved as evidence and rerun from the beginning.
"""
    environment = state["execution_environment"]
    text = f"""# 最初の手を計算する二方法の事前固定比較 / Precommitted comparison of two root-move calculation methods

## 日本語

この文書は、開始局面用の最初の手を計算する二方法を比べるための記録です。対戦の強さを測る実験ではありません。選ばれた600局面ごとに、SHA-256から導いた4個の main engine seed で、二方法を二回ずつ計算します。各計算組では同じ main engine seed と保存済み Console・評価ファイルを使い、二方法を連続して実行します。反復2の実行順は反復1と逆です。両方が異なる手を受理したときは、main engine seed とは別に事前導出した2個の専用deep seedのそれぞれで level 33 の比較を行います。

- 選択した局面数: {selection['position_count']}
- Console の乱数 seed 数: {ENGINE_SEED_COUNT}
- 専用deep seed数: {DEEP_SEED_COUNT}
- 反復数: {REPETITION_COUNT}
- 計算組数: {len(plan)}
- スレッド数: {THREADS}
- hash: {HASH_LEVEL}
- 通常 book とコンテスト book: 無効 (`-nobook`, `-nocontestbook`)
- 保存済み Console: `{environment['executable']['snapshot']['path']}`
- 保存済み評価ファイル: `eval.egev2` と `eval_move_ordering_end.egev`
- 位置選択ID: `{selection['selection_id']}`

## English

This document records a comparison of two methods for calculating a root move for starting positions. It is not an experiment measuring playing strength. For every selected one of 600 positions, both methods are calculated twice under each of four main engine seeds derived from SHA-256. Each calculation pair uses the same main engine seed and saved Console/evaluation files and runs its two methods consecutively. Repetition 2 reverses the order used in repetition 1. When both methods accept different moves, a level-33 comparison is run under each of two dedicated deep seeds derived in advance and distinct from the main engine seeds.

- Selected positions: {selection['position_count']}
- Console random seeds: {ENGINE_SEED_COUNT}
- Dedicated deep seeds: {DEEP_SEED_COUNT}
- Repetitions: {REPETITION_COUNT}
- Calculation pairs: {len(plan)}
- Threads: {THREADS}
- Hash: {HASH_LEVEL}
- Ordinary and contest books: disabled (`-nobook`, `-nocontestbook`)
- Saved Console: `{environment['executable']['snapshot']['path']}`
- Saved evaluation files: `eval.egev2` and `eval_move_ordering_end.egev`
- Position-selection ID: `{selection['selection_id']}`

{_term_definitions_markdown()}

## 事前固定の条件 / Precommitted conditions

### 日本語

`experiment_state.json` に、局面、除外入力、Console と二つの評価ファイルの SHA-256、四つのmain engine seed、二つの専用deep seed、各計算組の実行順、判定規則を保存します。再開時にはこれらが現在の入力と完全に一致し、保存済みファイルの SHA-256 も一致しなければなりません。

### English

`experiment_state.json` stores the positions, exclusion inputs, SHA-256 values for Console and both evaluation files, four main engine seeds, two dedicated deep seeds, the execution order of every calculation pair, and the decision rule. On resume, these must exactly match the current inputs and the SHA-256 values of the saved files.

{report_section}
"""
    _atomic_write_text(output_dir / "README.md", text)


def _run_unlocked(
    coverage: Path,
    root_results: list[Path],
    sample_coverages: list[Path],
    exe: Path,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    if not exe.is_file():
        raise FileNotFoundError(f"Console executable not found: {exe}")
    expected_state, _boards, plan, sample_payload = _new_state(
        coverage, root_results, sample_coverages, exe, output_dir
    )
    state_path = _state_path(output_dir)
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError("--resume needs an existing experiment_state.json")
        state = _read_json(state_path, "experiment state")
        if state != expected_state:
            raise ValueError("resume inputs do not exactly match experiment_state.json")
        _validate_fixed_sample(output_dir, state)
    else:
        if output_dir.exists():
            raise FileExistsError(f"formal comparison output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        _write_fixed_sample(output_dir, sample_payload)
        _atomic_write_text(state_path, _canonical_json(expected_state))
        state = expected_state
    saved_exe = _validate_and_materialize_environment(output_dir, state["execution_environment"])
    deep_seeds = state["deep_seed_derivation"].get("seeds")
    if not isinstance(deep_seeds, list) or len(deep_seeds) != DEEP_SEED_COUNT:
        raise ValueError("immutable state has an invalid dedicated level-33 seed list")
    records = _load_progress(output_dir, state_path, plan, deep_seeds)
    _reconcile_incomplete_pairs(output_dir, plan, records)
    _write_progress(output_dir, state_path, plan, records)
    _write_readme(output_dir, state, records)
    for item in plan[len(records):]:
        payload = _run_pair(saved_exe, output_dir, item, deep_seeds)
        _validate_pair_payload(payload, item, output_dir, deep_seeds)
        pair_path = _pair_result_path(output_dir, item)
        records.append(
            {
                "schedule_index": item["schedule_index"],
                "pair_id": item["pair_id"],
                "pair_result_path": pair_path.resolve().as_posix(),
                "pair_result_sha256": sha256_file(pair_path),
            }
        )
        _write_progress(output_dir, state_path, plan, records)
        _write_readme(output_dir, state, records)
    if len(records) != len(plan):
        raise RuntimeError("formal comparison ended without every planned calculation pair")
    report = _summarize_completed_pairs(output_dir, plan, records, state)
    _atomic_write_text(output_dir / "formal_comparison_report.json", _canonical_json(report))
    _write_readme(output_dir, state, records)
    return report


def run_formal_comparison(
    coverage: Path,
    root_results: list[Path],
    sample_coverages: list[Path],
    exe: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Run exclusively, preserving any interrupted pair for later inspection."""
    with file_lock(_lock_path(output_dir)):
        return _run_unlocked(
            coverage,
            root_results,
            sample_coverages,
            exe.resolve(),
            output_dir,
            resume,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument(
        "--exclude-root-results",
        type=Path,
        action="append",
        default=[],
        help="Prior root-result or root-table input whose boards must not enter the new 600-position sample (repeatable)",
    )
    parser.add_argument(
        "--exclude-sample-coverage",
        type=Path,
        action="append",
        default=[],
        help="Prior fixed_sample_coverage.json whose boards must not enter the new sample (repeatable)",
    )
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when every frozen input, saved Console file, and saved evaluation file matches",
    )
    args = parser.parse_args()
    report = run_formal_comparison(
        args.coverage,
        args.exclude_root_results,
        args.exclude_sample_coverage,
        args.exe,
        args.output_dir,
        args.resume,
    )
    print(
        f"completed_pairs={report['completed_pairs']} "
        f"can_select={report['level_30_then_level_31_can_continue_to_larger_calculation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
