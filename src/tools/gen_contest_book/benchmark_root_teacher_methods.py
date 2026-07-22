"""Compare two root-move calculation methods on an unbiased fixed sample.

The sample is selected from every target 14-disc position before either method
is run.  Every selected position is calculated by both methods in a balanced
per-position order.  Positions with different accepted moves receive a deeper
check: two level-33 root searches and two level-33 ``analyze`` evaluations for
each of the two specified moves.

The output is evidence about calculation quality and wall time, not evidence
about playing strength.  It never changes the tournament move table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from audit_r14_corpus import CORPUS_REPORT_SCHEMA
from book_artifact import file_lock
from build_root_table import sha256_file
import generate_ggs_root_teacher
from generate_ggs_root_teacher import (
    DEPTH_RE,
    generate_teachers,
    load_excluded_roots,
    load_uncovered_roots,
    root_file_provenance,
    search_root_at_level,
    select_teacher_roots,
    validate_quality,
)


BENCHMARK_SCHEMA = "root_teacher_method_benchmark_v2"
EXPERIMENT_STATE_SCHEMA = "root_teacher_method_benchmark_state_v1"
EXPERIMENT_PROGRESS_SCHEMA = "root_teacher_method_benchmark_progress_v1"
TIME_SECONDS = 60.0
THREADS = 28
HASH_LEVEL = 29
MIN_DEPTH = 30
MIN_SELECTIVITY = 74
LEVEL_30 = 30
LEVEL_31 = 31
LEVEL_33 = 33
BOOTSTRAP_REPETITIONS = 100_000
ANALYZE_RESULT_RE = re.compile(
    r"^\|\s*\d+\|\s*(?:Black|White)\|\s*(?P<move>[a-h][1-8])"
    r"\|\s*(?P<depth>\d+@\d+%)\|\s*(?P<score>[+-]?\d+)\|",
    re.MULTILINE,
)
TIME_METHOD = "time_managed_search"
LEVEL_METHOD = "level_30_then_level_31"


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


def _write_coverage(path: Path, boards: list[str]) -> None:
    payload = {
        "schema": CORPUS_REPORT_SCHEMA,
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
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} {path} is not an object")
    return payload


def _load_fixed_coverage_boards(path: Path) -> list[str]:
    payload = _read_json(path, "fixed sample coverage")
    if payload.get("schema") != CORPUS_REPORT_SCHEMA:
        raise ValueError(f"{path}: unexpected fixed-coverage schema")
    roots = payload.get("roots")
    if not isinstance(roots, list):
        raise ValueError(f"{path}: fixed-coverage roots are invalid")
    boards: list[str] = []
    for root in roots:
        if not isinstance(root, dict) or not isinstance(root.get("canonical_board"), str):
            raise ValueError(f"{path}: fixed-coverage root is invalid")
        boards.append(root["canonical_board"])
    if len(boards) != len(set(boards)):
        raise ValueError(f"{path}: fixed-coverage boards are not unique")
    return boards


def _experiment_state_path(output_dir: Path) -> Path:
    return output_dir / "experiment_state.json"


def _experiment_progress_path(output_dir: Path) -> Path:
    return output_dir / "comparison_progress.json"


def _experiment_lock_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.lock"


def _output_path(output_dir: Path, method_name: str, index: int) -> Path:
    directory = {
        TIME_METHOD: output_dir / "time_managed_search_results",
        LEVEL_METHOD: output_dir / "level_30_then_level_31_results",
    }.get(method_name)
    if directory is None:
        raise ValueError(f"unknown calculation method {method_name}")
    return directory / f"{index:04d}.txt"


def _source_file_provenance() -> dict[str, dict[str, str]]:
    teacher = Path(generate_ggs_root_teacher.__file__)
    benchmark = Path(__file__)
    return {
        "benchmark_script": {
            "path": benchmark.resolve().as_posix(),
            "sha256": sha256_file(benchmark),
        },
        "teacher_script": {
            "path": teacher.resolve().as_posix(),
            "sha256": sha256_file(teacher),
        },
    }


def _conditions() -> dict[str, Any]:
    return {
        "threads": THREADS,
        "hash": HASH_LEVEL,
        "minimum_depth": MIN_DEPTH,
        "minimum_selectivity": MIN_SELECTIVITY,
        TIME_METHOD: {
            "method": "time_then_verify",
            "time_seconds": TIME_SECONDS,
            "fallback_level": LEVEL_30,
            "verify_level": LEVEL_31,
        },
        LEVEL_METHOD: {
            "method": "hint_then_verify",
            "teacher_level": LEVEL_30,
            "verify_level": LEVEL_31,
        },
        "different_move_check": {
            "level": LEVEL_33,
            "root_searches": 2,
            "forced_move_analyses_per_move": 2,
            "books_disabled": True,
        },
    }


def _immutable_state(
    coverage: Path,
    excluded_root_results: list[Path],
    exe: Path,
    population: list[str],
    boards: list[str],
    orders: dict[str, tuple[str, str]],
    sample_seed: int,
    order_seed: int,
    bootstrap_seed: int,
    fixed_coverage: Path,
) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_STATE_SCHEMA,
        "engine": {
            "path": exe.resolve().as_posix(),
            "sha256": sha256_file(exe),
        },
        "source_files": _source_file_provenance(),
        "population": {
            "coverage": {
                "path": coverage.resolve().as_posix(),
                "sha256": sha256_file(coverage),
            },
            "excluded_root_results": root_file_provenance(excluded_root_results),
            "count": len(population),
            "sha256": hashlib.sha256("\n".join(population).encode("ascii")).hexdigest(),
        },
        "positions": {
            "population_count": len(population),
            "requested": len(boards),
            "sample_seed": sample_seed,
            "order_seed": order_seed,
            "bootstrap_seed": bootstrap_seed,
            "fixed_coverage_path": fixed_coverage.resolve().as_posix(),
            "fixed_coverage_sha256": sha256_file(fixed_coverage),
            "sha256": hashlib.sha256("\n".join(boards).encode("ascii")).hexdigest(),
            "boards": boards,
            "execution_orders": [list(orders[board]) for board in boards],
        },
        "conditions": _conditions(),
    }


def _write_experiment_progress(
    output_dir: Path,
    state_path: Path,
    requested: int,
    records: list[dict[str, Any]],
) -> None:
    at_least_one = [
        record
        for record in records
        if any(_has_method_result(record, method_name) for method_name in (TIME_METHOD, LEVEL_METHOD))
    ]
    both_methods = [record for record in records if _record_has_both_methods(record)]
    completed = [record for record in records if _record_is_complete(record)]
    payload = {
        "schema": EXPERIMENT_PROGRESS_SCHEMA,
        "experiment_state_sha256": sha256_file(state_path),
        "requested": requested,
        "positions_started": len(records),
        "positions_with_a_completed_calculation": len(at_least_one),
        "positions_with_both_calculations_complete": len(both_methods),
        "completed_positions": len(completed),
        "records": records,
    }
    _atomic_write_text(
        _experiment_progress_path(output_dir),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    summary = _summarize_positions(completed)
    report = f"""# 最初の手を計算する方法の比較: 実行中の進捗

## 日本語

- 固定標本: {requested}局面
- 計算を開始した局面: {len(records)}局面
- 少なくとも一方の計算が完了した局面: {len(at_least_one)}局面
- 両方の計算が完了した局面: {len(both_methods)}局面
- 必要なlevel 33確認まで完了した局面: {len(completed)}局面
- 未完了: {requested - len(completed)}局面
- 両方で採用: {summary['both_accepted']}
- 60秒の持ち時間を与える探索だけで採用: {summary['only_time_managed_search_accepted']}
- level 30・level 31の照合だけで採用: {summary['only_level_30_then_level_31_accepted']}
- 両方で却下: {summary['both_rejected']}
- 両方で採用され、最初の手が異なる: {summary['different_accepted_move']}

この文書は途中経過であり、計算方法の選択や開始局面用の手の表への登録には使わない。`experiment_state.json` は固定した局面・条件を、`comparison_progress.json` は完了済み局面の実時間を含む結果を記録する。中断後は同じ条件で `--resume` を指定して再開できる。

## English

- Fixed sample: {requested} positions
- Positions started: {len(records)}
- Positions with at least one calculation complete: {len(at_least_one)}
- Positions with both calculations complete: {len(both_methods)}
- Positions complete including any required level-33 check: {len(completed)}
- Not complete: {requested - len(completed)}
- Accepted by both: {summary['both_accepted']}
- Accepted only by the search given 60 seconds of remaining game time: {summary['only_time_managed_search_accepted']}
- Accepted only by the level-30/level-31 check: {summary['only_level_30_then_level_31_accepted']}
- Rejected by both: {summary['both_rejected']}
- Accepted by both with different first moves: {summary['different_accepted_move']}

This is progress only and is not used to select a calculation method or enter a move into the starting-position table. `experiment_state.json` records the fixed positions and conditions; `comparison_progress.json` records durable completed-position results including wall times. Use the same conditions with `--resume` after an interruption.
"""
    _atomic_write_text(output_dir / "COMPARISON_PROGRESS.md", report)


def _load_experiment_progress(
    output_dir: Path, state_path: Path, requested: int
) -> list[dict[str, Any]]:
    path = _experiment_progress_path(output_dir)
    if not path.exists():
        return []
    payload = _read_json(path, "comparison progress")
    if payload.get("schema") != EXPERIMENT_PROGRESS_SCHEMA:
        raise ValueError(f"{path}: unexpected comparison-progress schema")
    if payload.get("experiment_state_sha256") != sha256_file(state_path):
        raise ValueError(f"{path}: experiment-state SHA-256 does not match")
    if payload.get("requested") != requested:
        raise ValueError(f"{path}: requested-position count does not match")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: records is invalid")
    at_least_one = [
        record
        for record in records
        if isinstance(record, dict)
        and any(_has_method_result(record, method_name) for method_name in (TIME_METHOD, LEVEL_METHOD))
    ]
    both_methods = [record for record in records if isinstance(record, dict) and _record_has_both_methods(record)]
    completed = [record for record in records if isinstance(record, dict) and _record_is_complete(record)]
    if payload.get("positions_started") != len(records):
        raise ValueError(f"{path}: started-position count does not match records")
    if payload.get("positions_with_a_completed_calculation") != len(at_least_one):
        raise ValueError(f"{path}: completed-calculation position count does not match records")
    if payload.get("positions_with_both_calculations_complete") != len(both_methods):
        raise ValueError(f"{path}: both-calculation position count does not match records")
    if payload.get("completed_positions") != len(completed):
        raise ValueError(f"{path}: completed-position count does not match records")
    return records


def _archive_uncheckpointed_files(
    output_dir: Path,
    index: int,
    files: list[Path],
    reason: str,
) -> None:
    """Preserve output whose elapsed time was not durably checkpointed."""
    if not files:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_dir / "interrupted_attempts" / f"{index:04d}_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    moved: list[dict[str, str]] = []
    for path in files:
        target = destination / path.name
        source = path.resolve().as_posix()
        os.replace(path, target)
        moved.append({"from": source, "to": target.resolve().as_posix()})
    audit_path = output_dir / "interrupted_attempts" / "archived_attempts.jsonl"
    record = {
        "schema": "root_teacher_method_benchmark_archived_attempt_v1",
        "index": index,
        "reason": reason,
        "files": moved,
    }
    with audit_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _archive_uncheckpointed_method(
    output_dir: Path, index: int, method_name: str
) -> None:
    output = _output_path(output_dir, method_name, index)
    _archive_uncheckpointed_files(
        output_dir,
        index,
        sorted(output.parent.glob(f"{output.name}*")),
        "calculation output was not present in the durable comparison-progress checkpoint",
    )


def _archive_uncheckpointed_deep_check(output_dir: Path, index: int) -> None:
    directory = output_dir / "level_33_checks"
    _archive_uncheckpointed_files(
        output_dir,
        index,
        sorted(directory.glob(f"{index:04d}_*.log*")),
        "level-33 check output was not present in the durable comparison-progress checkpoint",
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read generated manifest {manifest_path}: {error}") from error
    output = manifest.get("output") if isinstance(manifest, dict) else None
    if not isinstance(output, dict) or output.get("sha256") != sha256_file(path):
        raise ValueError(f"generated output does not match its manifest: {path}")
    if manifest.get("schema") != "ggs_root_teacher_manifest_v10":
        raise ValueError(f"generated manifest has an unsupported schema: {path}")
    return manifest


def _method_settings(method_name: str) -> tuple[str, int, int, int]:
    if method_name == TIME_METHOD:
        return ("time_then_verify", LEVEL_30, LEVEL_30, LEVEL_31)
    if method_name == LEVEL_METHOD:
        return ("hint_then_verify", LEVEL_30, LEVEL_30, LEVEL_31)
    raise ValueError(f"unknown calculation method {method_name}")


def _run_one_method(
    coverage: Path,
    board: str,
    exe: Path,
    output: Path,
    method_name: str,
    session_id: str,
) -> dict[str, Any]:
    method, fallback_level, teacher_level, verify_level = _method_settings(method_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    generate_teachers(
        coverage,
        exe,
        output,
        TIME_SECONDS,
        THREADS,
        HASH_LEVEL,
        MIN_DEPTH,
        MIN_SELECTIVITY,
        fallback_level,
        method,
        teacher_level,
        verify_level,
        checkpoint_every=1,
    )
    elapsed_seconds = time.monotonic() - started
    finished_at_utc = datetime.now(timezone.utc).isoformat()
    outcome = _outcome_from_manifest(board, output)
    outcome.update(
        {
            "wall_seconds": elapsed_seconds,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "session_id": session_id,
        }
    )
    return outcome


def _outcome_from_manifest(board: str, output: Path) -> dict[str, Any]:
    """Return the calculation result stored in one verified teacher output."""
    manifest = _load_manifest(output)
    output_info = manifest["output"]
    if output_info.get("processed") != 1:
        raise ValueError(f"{output}: expected exactly one processed position")
    results = manifest.get("results")
    rejections = manifest.get("rejections")
    if not isinstance(results, dict) or not isinstance(rejections, dict):
        raise ValueError(f"{output}: missing accepted or rejected results")
    if board in results and board not in rejections:
        result = results[board]
        if not isinstance(result, dict) or not isinstance(result.get("move"), str):
            raise ValueError(f"{output}: invalid accepted result")
        return {
            "status": "accepted",
            "move": result["move"],
            "score": result.get("score"),
            "depth": result.get("depth"),
            "nodes": result.get("nodes"),
            "output_path": output.resolve().as_posix(),
            "output_sha256": output_info.get("sha256"),
        }
    if board in rejections and board not in results:
        rejection = rejections[board]
        if not isinstance(rejection, dict):
            raise ValueError(f"{output}: invalid rejected result")
        return {
            "status": "rejected",
            "reason": rejection.get("reason"),
            "output_path": output.resolve().as_posix(),
            "output_sha256": output_info.get("sha256"),
        }
    raise ValueError(f"{output}: result does not describe {board}")


def _has_method_result(record: dict[str, Any], method_name: str) -> bool:
    return isinstance(record.get(method_name), dict)


def _record_requires_deep_check(record: dict[str, Any]) -> bool:
    if not all(_has_method_result(record, method_name) for method_name in (TIME_METHOD, LEVEL_METHOD)):
        return False
    time_result = record[TIME_METHOD]
    level_result = record[LEVEL_METHOD]
    return (
        time_result.get("status") == "accepted"
        and level_result.get("status") == "accepted"
        and time_result.get("move") != level_result.get("move")
    )


def _record_has_both_methods(record: dict[str, Any]) -> bool:
    return all(_has_method_result(record, method_name) for method_name in (TIME_METHOD, LEVEL_METHOD))


def _record_is_complete(record: dict[str, Any]) -> bool:
    return _record_has_both_methods(record) and (
        not _record_requires_deep_check(record) or isinstance(record.get("level_33_check"), dict)
    )


def _validate_elapsed_metadata(
    payload: dict[str, Any], description: str
) -> None:
    elapsed = payload.get("wall_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
    ):
        raise ValueError(f"{description}: wall_seconds must be a positive finite number")
    for field in ("started_at_utc", "finished_at_utc", "session_id"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"{description}: {field} is invalid")


def _validate_saved_method_record(
    record: dict[str, Any],
    board: str,
    output_dir: Path,
    index: int,
    method_name: str,
) -> None:
    saved = record.get(method_name)
    if not isinstance(saved, dict):
        return
    _validate_elapsed_metadata(saved, f"position {index} {method_name}")
    expected = _outcome_from_manifest(board, _output_path(output_dir, method_name, index))
    for field, expected_value in expected.items():
        if saved.get(field) != expected_value:
            raise ValueError(
                f"position {index} {method_name}: saved progress does not match verified output "
                f"for {field}"
            )


def _validate_saved_deep_check(record: dict[str, Any], index: int) -> None:
    check = record.get("level_33_check")
    if not isinstance(check, dict):
        return
    _validate_elapsed_metadata(check, f"position {index} level-33 check")
    if check.get("status") not in {
        "level_30_then_level_31_better",
        "level_30_then_level_31_worse",
        "equal_level_33_score",
        "unresolved",
    }:
        raise ValueError(f"position {index}: invalid stored level-33 status")
    forced = check.get("forced_move_analyses")
    if forced is None:
        if check.get("status") != "unresolved":
            raise ValueError(f"position {index}: completed level-33 check has no forced-move records")
        return
    if not isinstance(forced, dict):
        raise ValueError(f"position {index}: invalid stored forced-move records")
    for move, analyses in forced.items():
        if not isinstance(move, str) or not isinstance(analyses, list):
            raise ValueError(f"position {index}: invalid stored forced-move analysis")
        for analysis in analyses:
            if not isinstance(analysis, dict):
                raise ValueError(f"position {index}: invalid stored forced-move result")
            log_path = analysis.get("log_path")
            log_sha256 = analysis.get("log_sha256")
            if not isinstance(log_path, str) or not isinstance(log_sha256, str):
                raise ValueError(f"position {index}: forced-move log provenance is invalid")
            path = Path(log_path)
            if not path.is_file() or sha256_file(path) != log_sha256:
                raise ValueError(f"position {index}: forced-move log no longer matches its SHA-256")


def _validate_progress_records(
    records: list[dict[str, Any]],
    boards: list[str],
    orders: dict[str, tuple[str, str]],
    output_dir: Path,
) -> dict[int, dict[str, Any]]:
    records_by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("comparison progress contains a non-object record")
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(boards):
            raise ValueError("comparison progress has an invalid position index")
        if index in records_by_index:
            raise ValueError(f"comparison progress records position {index} more than once")
        board = boards[index - 1]
        if record.get("board") != board:
            raise ValueError(f"comparison progress position {index} has a different board")
        if record.get("execution_order") != list(orders[board]):
            raise ValueError(f"comparison progress position {index} has a different execution order")
        in_progress = record.get("in_progress")
        if in_progress is not None:
            if not isinstance(in_progress, dict):
                raise ValueError(f"comparison progress position {index} has invalid in_progress")
            kind = in_progress.get("kind")
            if kind == "method":
                method_name = in_progress.get("method")
                if method_name not in (TIME_METHOD, LEVEL_METHOD) or _has_method_result(record, method_name):
                    raise ValueError(f"comparison progress position {index} has invalid in-progress method")
            elif kind != "level_33_check":
                raise ValueError(f"comparison progress position {index} has invalid in-progress kind")
            for field in ("started_at_utc", "session_id"):
                if not isinstance(in_progress.get(field), str) or not in_progress[field]:
                    raise ValueError(f"comparison progress position {index} has invalid in-progress {field}")
        for method_name in (TIME_METHOD, LEVEL_METHOD):
            _validate_saved_method_record(record, board, output_dir, index, method_name)
        if "level_33_check" in record:
            if not _record_requires_deep_check(record):
                raise ValueError(f"comparison progress position {index} has an unnecessary level-33 check")
            _validate_saved_deep_check(record, index)
        if in_progress is not None and in_progress.get("kind") == "level_33_check":
            if not _record_requires_deep_check(record) or "level_33_check" in record:
                raise ValueError(f"comparison progress position {index} has invalid in-progress level-33 check")
        records_by_index[index] = record
    if records_by_index and sorted(records_by_index) != list(range(1, max(records_by_index) + 1)):
        raise ValueError("comparison progress has a gap in its started-position prefix")
    return records_by_index


def _order_boards(boards: list[str], order_seed: int) -> dict[str, tuple[str, str]]:
    """Assign exactly half of an even sample to each execution order."""
    ordered = sorted(
        boards,
        key=lambda board: (
            hashlib.sha256(f"{order_seed}\0{board}".encode("ascii")).digest(),
            board,
        ),
    )
    orders: dict[str, tuple[str, str]] = {}
    for index, board in enumerate(ordered):
        orders[board] = (
            (TIME_METHOD, LEVEL_METHOD)
            if index % 2 == 0
            else (LEVEL_METHOD, TIME_METHOD)
        )
    return orders


def _validate_at_requested_level(result: dict[str, int | str], level: int) -> None:
    validate_quality(result, max(MIN_DEPTH, level), MIN_SELECTIVITY)


def evaluate_forced_move_at_level(
    exe: Path,
    board: str,
    move: str,
    level: int,
    log_path: Path,
) -> dict[str, int | str]:
    """Evaluate exactly one played initial move with Console ``analyze``.

    ``hint`` only chooses its displayed top moves.  ``setboard``, ``play``,
    then ``analyze`` instead evaluates the concrete move supplied here.  A new
    Console process, disabled books, and ``clearcache`` give each move the same
    starting condition.
    """
    command = [
        str(exe),
        "-l", str(level),
        "-t", str(THREADS),
        "-hash", str(HASH_LEVEL),
        "-nobook",
        "-nocontestbook",
    ]
    commands = f"setboard {board}\nclearcache\nplay {move}\nanalyze\nquit\n"
    try:
        completed = subprocess.run(
            command,
            input=commands,
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"forced-move level-{level} analysis failed for {board} {move}: {error}") from error
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    _atomic_write_text(log_path, combined_output)
    if completed.returncode != 0:
        raise RuntimeError(
            f"forced-move level-{level} analysis returned {completed.returncode} for {board} {move}: "
            f"{combined_output[-400:]}"
        )
    rows = [match.groupdict() for match in ANALYZE_RESULT_RE.finditer(combined_output)]
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one analyze-result row for {board} {move}, found {len(rows)}"
        )
    row = rows[0]
    if row["move"] != move:
        raise ValueError(
            f"analyze reported {row['move']} while evaluating requested move {move}"
        )
    result: dict[str, int | str] = {
        "move": row["move"],
        "score": int(row["score"]),
        "depth": row["depth"],
        "log_path": log_path.resolve().as_posix(),
        "log_sha256": sha256_file(log_path),
    }
    _validate_at_requested_level(result, level)
    return result


def _deep_check_different_moves(
    exe: Path,
    board: str,
    time_move: str,
    level_move: str,
    log_directory: Path,
    position_index: int,
) -> dict[str, Any]:
    """Compare two different accepted moves conservatively at level 33."""
    try:
        root_first = search_root_at_level(exe, board, LEVEL_33, THREADS, HASH_LEVEL)
        root_second = search_root_at_level(exe, board, LEVEL_33, THREADS, HASH_LEVEL)
        _validate_at_requested_level(root_first, LEVEL_33)
        _validate_at_requested_level(root_second, LEVEL_33)
        forced: dict[str, list[dict[str, int | str]]] = {}
        for move in (time_move, level_move):
            forced[move] = [
                evaluate_forced_move_at_level(
                    exe,
                    board,
                    move,
                    LEVEL_33,
                    log_directory / f"{position_index:04d}_{move}_first.log",
                ),
                evaluate_forced_move_at_level(
                    exe,
                    board,
                    move,
                    LEVEL_33,
                    log_directory / f"{position_index:04d}_{move}_second.log",
                ),
            ]
    except (RuntimeError, ValueError) as error:
        return {"status": "unresolved", "reason": str(error)}
    time_scores = [int(entry["score"]) for entry in forced[time_move]]
    level_scores = [int(entry["score"]) for entry in forced[level_move]]
    root_stable = str(root_first["move"]) == str(root_second["move"])
    score_stable = len(set(time_scores)) == 1 and len(set(level_scores)) == 1
    payload: dict[str, Any] = {
        "level": LEVEL_33,
        "root_searches": [root_first, root_second],
        "forced_move_analyses": forced,
        "root_search_move_is_stable": root_stable,
        "forced_move_scores_are_stable": score_stable,
    }
    if not root_stable or not score_stable:
        payload["status"] = "unresolved"
        payload["reason"] = "repeated level-33 search or forced-move score was not stable"
        return payload
    if level_scores[0] > time_scores[0]:
        payload["status"] = "level_30_then_level_31_better"
    elif level_scores[0] < time_scores[0]:
        payload["status"] = "level_30_then_level_31_worse"
    else:
        payload["status"] = "equal_level_33_score"
    return payload


def _one_sided_binomial_upper(failures: int, total: int, confidence: float = 0.95) -> float:
    """Return the exact Clopper-Pearson one-sided upper bound."""
    if total < 1 or not 0 <= failures <= total or not 0.0 < confidence < 1.0:
        raise ValueError("invalid binomial interval arguments")
    if failures == total:
        return 1.0
    target = 1.0 - confidence

    def cdf(probability: float) -> float:
        return sum(
            math.comb(total, count)
            * probability**count
            * (1.0 - probability) ** (total - count)
            for count in range(failures + 1)
        )

    low = failures / total
    high = 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if cdf(middle) > target:
            low = middle
        else:
            high = middle
    return high


def _bootstrap_paired_time_ratio(
    records: list[dict[str, Any]], seed: int, repetitions: int = BOOTSTRAP_REPETITIONS
) -> dict[str, float | int]:
    """Bootstrap the mean wall-time ratio while retaining each position pair."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    pairs = [
        (
            float(record[LEVEL_METHOD]["wall_seconds"]),
            float(record[TIME_METHOD]["wall_seconds"]),
        )
        for record in records
    ]
    if not pairs or any(time_seconds <= 0.0 for _, time_seconds in pairs):
        raise ValueError("wall times must be positive")
    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(repetitions):
        level_total = 0.0
        time_total = 0.0
        for _ in pairs:
            level_seconds, time_seconds = pairs[generator.randrange(len(pairs))]
            level_total += level_seconds
            time_total += time_seconds
        ratios.append(level_total / time_total)
    ratios.sort()
    return {
        "point_estimate": sum(level_seconds for level_seconds, _ in pairs)
        / sum(time_seconds for _, time_seconds in pairs),
        "lower_95_percent": ratios[math.floor(0.025 * (repetitions - 1))],
        "upper_95_percent": ratios[math.ceil(0.975 * (repetitions - 1))],
        "seed": seed,
        "repetitions": repetitions,
    }


def _records_with_same_session_pair(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only position pairs measured in one uninterrupted benchmark invocation."""
    return [
        record
        for record in records
        if record[TIME_METHOD].get("session_id") == record[LEVEL_METHOD].get("session_id")
    ]


def _timing_by_execution_order(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Keep the two balanced per-position execution orders visible in the output."""
    groups = {
        f"{TIME_METHOD}_first": [
            record for record in records if record["execution_order"][0] == TIME_METHOD
        ],
        f"{LEVEL_METHOD}_first": [
            record for record in records if record["execution_order"][0] == LEVEL_METHOD
        ],
    }
    result: dict[str, dict[str, float | int]] = {}
    for name, group in groups.items():
        result[name] = {
            "positions": len(group),
            TIME_METHOD: sum(float(record[TIME_METHOD]["wall_seconds"]) for record in group),
            LEVEL_METHOD: sum(float(record[LEVEL_METHOD]["wall_seconds"]) for record in group),
        }
    return result


def _summarize_positions(records: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "both_accepted": 0,
        "only_time_managed_search_accepted": 0,
        "only_level_30_then_level_31_accepted": 0,
        "both_rejected": 0,
        "same_accepted_move": 0,
        "different_accepted_move": 0,
        "level_30_then_level_31_better": 0,
        "level_30_then_level_31_worse": 0,
        "equal_level_33_score": 0,
        "unresolved_level_33": 0,
    }
    for record in records:
        time_result = record[TIME_METHOD]
        level_result = record[LEVEL_METHOD]
        time_accepted = time_result["status"] == "accepted"
        level_accepted = level_result["status"] == "accepted"
        if time_accepted and level_accepted:
            summary["both_accepted"] += 1
            if time_result["move"] == level_result["move"]:
                summary["same_accepted_move"] += 1
            else:
                summary["different_accepted_move"] += 1
                status = record.get("level_33_check", {}).get("status")
                if status in summary:
                    summary[status] += 1
                elif status == "unresolved":
                    summary["unresolved_level_33"] += 1
        elif time_accepted:
            summary["only_time_managed_search_accepted"] += 1
        elif level_accepted:
            summary["only_level_30_then_level_31_accepted"] += 1
        else:
            summary["both_rejected"] += 1
    return summary


def _write_report(payload: dict[str, Any], path: Path) -> None:
    positions = payload["positions"]
    comparison = payload["comparison"]
    timing = payload["timing_seconds"]
    decision = payload["calculation_method_decision"]
    report = f"""# 14石開始局面で最初の手を計算する二つの方法の比較

## 日本語

### 用語の定義記録

#### 対象母集団（英語: population）

- 出典: `r14_corpus_current_audit_20260722.json` と、この実験に指定した除外結果ファイル。
- 目的: どの局面へ計算方法を一般化しようとしているかを固定する。
- 具体対象: 既存の開始局面用の手の表と指定した既計算結果に含まれる局面を除いた {positions['population_count']} 局面。
- 役割: 固定標本を選ぶ元になる全体集合。
- 前後関係: まず対象母集団を確定し、その後で固定標本を選び、最後に二つの計算方法を実行する。
- 候補語: 対象集合、比較対象全体、population。
- 初出定義: この節の本項。以後の「対象母集団」はこの {positions['population_count']} 局面だけを指す。

#### 固定標本（英語: fixed sample）

- 出典: 上記対象母集団と、記録した選択用の整数 `{positions['sample_seed']}`。
- 目的: 計算結果や対局結果を見てから局面を選ぶ偏りを防ぐ。
- 具体対象: 対象母集団を SHA-256 で事前に並べ、先頭から選んだ {positions['requested']} 局面。
- 役割: 二つの計算方法を同じ条件で比べる分母。
- 前後関係: 対象母集団の確定後、どちらの計算も始める前に作成し、以後は追加・再抽選しない。
- 候補語: 事前固定局面、比較用局面、fixed sample。
- 初出定義: この節の本項。以後の「固定標本」はこの {positions['requested']} 局面だけを指す。

#### 深さ33の確認（英語: level-33 check）

- 出典: Console の `hint`、`setboard`、`play`、`analyze` コマンドと本実験の設定。
- 目的: 二つの方法で最初の手が異なった場合に、一方を根拠なく不利と決めない。
- 具体対象: 通常bookとcontest bookを無効にした28スレッド・hash 29の深さ33探索2回、および各指定手に対する `setboard` → `play` → `analyze` の深さ33評価2回ずつ。
- 役割: 異なる二手の評価値と再現性を記録し、低い評価または判断不能を検出する。
- 前後関係: 両方法が同じ局面で採用し、かつ最初の手が異なった後にだけ実行する。
- 候補語: 追加深い探索、異手確認、level-33 check。
- 初出定義: この節の本項。以後の「深さ33の確認」はこの反復探索と指定手評価の組を指す。

#### 採用・却下（英語: accepted / rejected）

- 出典: `generate_ggs_root_teacher.py` が出力する各局面の結果。
- 目的: 手が決まった局面と、確認不一致などで手を決めなかった局面を混同しない。
- 具体対象: 各計算方法による固定標本の各局面の結果。
- 役割: 両方採用・片方だけ採用・両方却下を同じ分母で集計する。
- 前後関係: 各局面で二つの計算方法を実行した直後に記録し、却下局面も固定標本から除かない。
- 候補語: 結果あり／結果なし、accepted / rejected。
- 初出定義: この節の本項。却下は対局の敗北を意味しない。

### 短い要約

- **対象母集団**: 指定した局面一覧から、既存の開始局面用の手の表と指定した既計算結果に含まれる局面を除いた {positions['population_count']} 局面。
- **固定標本**: 対象母集団の各局面を、`{positions['sample_seed']}\\0局面文字列` の SHA-256 で並べ、先頭から選んだ {positions['requested']} 局面。計算結果や対局結果を見る前に固定した。
- **採用**: その方法が、設定した品質条件を満たす最初の手を決定したこと。
- **却下**: 照合する探索が一致しないなどの理由で、その方法が手を決定しなかったこと。対局の敗北ではない。
- **深さ33の確認**: 最初の手が異なった局面で、通常bookとcontest bookを無効にして、28スレッド・hash 29で深さ33の最善手探索を2回、さらに各指定手を `setboard` → `play` → `analyze` で2回ずつ評価した確認。

### 固定条件

- 実行ファイル SHA-256: `{payload['engine']['sha256']}`
- 28スレッド、hash 29、通常book無効、contest book無効。
- 「60秒の持ち時間を与える探索」と「level 30で選びlevel 31で照合する探索」を、局面ごとに直列で実行した。局面ごとの実行順は別の固定 SHA-256 順で交互にしたため、片方だけが常に先に実行されることはない。
- level 31の照合は実際に深さ31以上、深さ33の確認は実際に深さ33以上へ達したことを検査した。

### 結果

- 固定標本数: {positions['requested']}
- 両方で採用: {comparison['both_accepted']}
- 60秒の持ち時間を与える探索だけで採用: {comparison['only_time_managed_search_accepted']}
- level 30・level 31の照合だけで採用: {comparison['only_level_30_then_level_31_accepted']}
- 両方で却下: {comparison['both_rejected']}
- 両方で採用され、最初の手が一致: {comparison['same_accepted_move']}
- 両方で採用され、最初の手が異なる: {comparison['different_accepted_move']}
- 異なる手について、level 30・level 31の照合の手が深さ33で低い評価: {comparison['level_30_then_level_31_worse']}
- 異なる手について、深さ33で同じ評価: {comparison['equal_level_33_score']}
- 異なる手について、深さ33でlevel 30・level 31の照合の手が高い評価: {comparison['level_30_then_level_31_better']}
- 異なる手について、深さ33の再現性が足りず判断できない: {comparison['unresolved_level_33']}
- 60秒の持ち時間を与える探索の合計実時間: {timing[TIME_METHOD]['total']:.3f}秒
- level 30・level 31の照合の合計実時間: {timing[LEVEL_METHOD]['total']:.3f}秒
- 対応を保った実時間比（level 30・level 31の照合 ÷ 60秒の持ち時間を与える探索）: {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['point_estimate']:.4f}
- 上記実時間比の95%区間: {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['lower_95_percent']:.4f} ～ {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['upper_95_percent']:.4f}（復元抽出 {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['repetitions']:,}回、乱数の種 {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['seed']}）
- 60秒の持ち時間を与える探索を先に実行した局面数: {timing['by_execution_order'][f'{TIME_METHOD}_first']['positions']}
- level 30・level 31の照合を先に実行した局面数: {timing['by_execution_order'][f'{LEVEL_METHOD}_first']['positions']}
- 深さ33の追加確認の実時間: {timing['level_33_checks']:.3f}秒
- 「深さ33で低い評価」または「判断できない」の局面数: {decision['unsupported_or_unresolved_count']}
- 60秒の持ち時間を与える探索だけが採用した局面数: {decision['level_30_then_level_31_coverage_disadvantage']}
- この数が0件だった場合の、固定標本における未確認率の片側95%上限: {decision['one_sided_95_percent_upper_bound'] * 100:.3f}%

### この結果から分かることと、分からないこと

この比較は、最初の手を計算する二つの方法の一致・却下率・実時間を調べる実験であり、エンジンの対局上の強さを測る実験ではない。`{decision['summary_ja']}`

大会用の開始局面用の手の表へ入れる前には、別途、同じ開始局面から先後を入れ替えて2局対戦する。ここでいう1 matchは、その2局を一組として集計する単位である。

## English

### Terms and scope

- **Population**: {positions['population_count']} positions remaining after excluding positions already covered by the existing starting-position move table and the specified result files.
- **Fixed sample**: {positions['requested']} positions selected before calculation or game results by sorting every population position with SHA-256 of `{positions['sample_seed']}\\0board text`.
- **Accepted**: the method determined a first move satisfying its configured quality condition.
- **Rejected**: the method did not determine a move, for example because confirmation searches disagreed. It is not a game loss.
- **Level-33 check**: for different accepted moves, two independent level-33 root searches plus two `setboard` → `play` → `analyze` evaluations of each specified move, with both books disabled, 28 threads, and hash 29.

### Results

- Fixed sample: {positions['requested']}
- Accepted by both methods: {comparison['both_accepted']}
- Accepted only by the search given 60 seconds of remaining game time: {comparison['only_time_managed_search_accepted']}
- Accepted only by the level-30/level-31 check: {comparison['only_level_30_then_level_31_accepted']}
- Rejected by both: {comparison['both_rejected']}
- Same accepted move: {comparison['same_accepted_move']}
- Different accepted move: {comparison['different_accepted_move']}
- Level-30/level-31 move lower at level 33: {comparison['level_30_then_level_31_worse']}
- Equal level-33 score: {comparison['equal_level_33_score']}
- Level-30/level-31 move higher at level 33: {comparison['level_30_then_level_31_better']}
- Level-33 check unresolved: {comparison['unresolved_level_33']}
- Total wall time, search given 60 seconds of remaining game time: {timing[TIME_METHOD]['total']:.3f} seconds
- Total wall time, level-30/level-31 check: {timing[LEVEL_METHOD]['total']:.3f} seconds
- Paired wall-time ratio (level-30/level-31 check ÷ search given 60 seconds of remaining game time): {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['point_estimate']:.4f}
- 95% interval for that paired ratio: {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['lower_95_percent']:.4f} to {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['upper_95_percent']:.4f}, from {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['repetitions']:,} resamples using seed {timing['paired_level_30_then_level_31_to_time_managed_search_ratio']['seed']}
- Positions where the time-managed search ran first: {timing['by_execution_order'][f'{TIME_METHOD}_first']['positions']}
- Positions where the level-30/level-31 check ran first: {timing['by_execution_order'][f'{LEVEL_METHOD}_first']['positions']}
- Additional level-33-check wall time: {timing['level_33_checks']:.3f} seconds
- Positions lower or unresolved at level 33: {decision['unsupported_or_unresolved_count']}
- Positions accepted only by the time-managed search: {decision['level_30_then_level_31_coverage_disadvantage']}
- One-sided 95% upper bound on that unconfirmed rate: {decision['one_sided_95_percent_upper_bound'] * 100:.3f}%

This comparison measures agreement, rejection rate, and wall time of two calculation methods; it does not measure playing strength. {decision['summary_en']}

Before a move enters the tournament starting-position move table, a separate match plays two games from the same starting position with colors exchanged.
"""
    report += f"""
## 中断時の計時の扱い / Treatment of interruptions in timing

### 日本語

- 同じプログラム起動中に二つの計算を完了した局面: {timing['positions_with_same_session_pairs']}局面
- 中断の前後に二つの計算がまたがったため、時間比の集計から除外した局面: {timing['positions_excluded_from_paired_time_ratio_due_to_interruption']}局面
- 後者が1局面でもある場合、この比較は残りの局面に使う計算方法を決めない。局面ごとの手の一致・不一致の集計には残す。

### English

- Positions whose two calculation times were both measured in one program invocation: {timing['positions_with_same_session_pairs']}
- Positions excluded from the paired time ratio because their two calculations crossed an interruption: {timing['positions_excluded_from_paired_time_ratio_due_to_interruption']}
- If the latter is nonzero, this comparison does not select a method for the remaining positions. The positions remain in the move-agreement and disagreement counts.
"""
    _atomic_write_text(path, report)


def _compare_methods_unlocked(
    coverage: Path,
    excluded_root_results: list[Path],
    exe: Path,
    output_dir: Path,
    positions: int,
    sample_seed: int,
    order_seed: int,
    bootstrap_seed: int,
    resume: bool,
) -> dict[str, Any]:
    if positions <= 0:
        raise ValueError("positions must be positive")
    if not exe.is_file():
        raise FileNotFoundError(f"engine executable not found: {exe}")
    all_roots = load_uncovered_roots(coverage)
    excluded = load_excluded_roots(excluded_root_results)
    population = sorted(board for board in all_roots if board not in excluded)
    if len(population) < positions:
        raise ValueError(
            f"population has {len(population)} positions, below requested {positions}"
        )
    boards = select_teacher_roots(population, positions, sample_seed)
    if len(set(boards)) != len(boards):
        raise ValueError("fixed sample is not unique")
    orders = _order_boards(boards, order_seed)
    fixed_coverage = output_dir / "fixed_sample_coverage.json"
    state_path = _experiment_state_path(output_dir)
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"benchmark output directory does not exist for --resume: {output_dir}")
        if not state_path.is_file():
            raise FileNotFoundError(
                f"{state_path} is required to resume; this directory was not created by the resumable benchmark"
            )
        if not fixed_coverage.is_file():
            raise FileNotFoundError(f"{fixed_coverage} is required to resume")
        saved_state = _read_json(state_path, "experiment state")
        expected_state = _immutable_state(
            coverage,
            excluded_root_results,
            exe,
            population,
            boards,
            orders,
            sample_seed,
            order_seed,
            bootstrap_seed,
            fixed_coverage,
        )
        if saved_state != expected_state:
            raise ValueError(
                "resume conditions do not exactly match experiment_state.json; "
                "start a new output directory instead"
            )
        if _load_fixed_coverage_boards(fixed_coverage) != boards:
            raise ValueError("fixed_sample_coverage.json does not match the saved fixed sample")
    else:
        if output_dir.exists():
            raise FileExistsError(f"benchmark output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        _write_coverage(fixed_coverage, boards)
        _atomic_write_text(
            state_path,
            json.dumps(
                _immutable_state(
                    coverage,
                    excluded_root_results,
                    exe,
                    population,
                    boards,
                    orders,
                    sample_seed,
                    order_seed,
                    bootstrap_seed,
                    fixed_coverage,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    input_directory = output_dir / "per_position_inputs"
    deep_directory = output_dir / "level_33_checks"
    records = _load_experiment_progress(output_dir, state_path, positions)
    records_by_index = _validate_progress_records(records, boards, orders, output_dir)
    session_id = uuid.uuid4().hex

    def checkpoint() -> None:
        _write_experiment_progress(
            output_dir,
            state_path,
            positions,
            [records_by_index[item] for item in sorted(records_by_index)],
        )

    for index, board in enumerate(boards, start=1):
        record = records_by_index.get(index)
        if record is None:
            record = {
                "index": index,
                "board": board,
                "execution_order": list(orders[board]),
            }
            records_by_index[index] = record
        in_progress = record.get("in_progress")
        if isinstance(in_progress, dict):
            if in_progress["kind"] == "method":
                _archive_uncheckpointed_method(output_dir, index, str(in_progress["method"]))
            else:
                _archive_uncheckpointed_deep_check(output_dir, index)
            del record["in_progress"]
            checkpoint()
        one_position_coverage = input_directory / f"{index:04d}.json"
        _write_coverage(one_position_coverage, [board])
        for method_name in orders[board]:
            if _has_method_result(record, method_name):
                continue
            _archive_uncheckpointed_method(output_dir, index, method_name)
            record["in_progress"] = {
                "kind": "method",
                "method": method_name,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
            }
            checkpoint()
            record[method_name] = _run_one_method(
                one_position_coverage,
                board,
                exe,
                _output_path(output_dir, method_name, index),
                method_name,
                session_id,
            )
            del record["in_progress"]
            checkpoint()
        if _record_requires_deep_check(record) and not isinstance(record.get("level_33_check"), dict):
            _archive_uncheckpointed_deep_check(output_dir, index)
            record["in_progress"] = {
                "kind": "level_33_check",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
            }
            checkpoint()
            time_result = record[TIME_METHOD]
            level_result = record[LEVEL_METHOD]
            started = time.monotonic()
            check = _deep_check_different_moves(
                exe,
                board,
                str(time_result["move"]),
                str(level_result["move"]),
                deep_directory,
                index,
            )
            check.update(
                {
                    "wall_seconds": time.monotonic() - started,
                    "started_at_utc": record["in_progress"]["started_at_utc"],
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "session_id": session_id,
                }
            )
            record["level_33_check"] = check
            del record["in_progress"]
            checkpoint()
    records = [records_by_index[index] for index in sorted(records_by_index)]
    if len(records) != positions or any(not _record_is_complete(record) for record in records):
        raise RuntimeError("benchmark ended with an incomplete position record")
    deep_elapsed = sum(
        float(record.get("level_33_check", {}).get("wall_seconds", 0.0))
        for record in records
    )
    timing_records = _records_with_same_session_pair(records)
    cross_session_pairs = len(records) - len(timing_records)
    if not timing_records:
        raise RuntimeError("no position has both calculation times from one benchmark invocation")
    comparison = _summarize_positions(records)
    paired_time_ratio = _bootstrap_paired_time_ratio(timing_records, bootstrap_seed)
    timing_by_order = _timing_by_execution_order(timing_records)
    unsupported_or_unresolved = (
        comparison["level_30_then_level_31_worse"]
        + comparison["unresolved_level_33"]
    )
    upper_bound = _one_sided_binomial_upper(unsupported_or_unresolved, len(records))
    direct_coverage_disadvantage = comparison["only_time_managed_search_accepted"]
    direct_rejections = direct_coverage_disadvantage + comparison["both_rejected"]
    time_rejections = (
        comparison["only_level_30_then_level_31_accepted"] + comparison["both_rejected"]
    )
    can_continue = (
        unsupported_or_unresolved == 0
        and direct_coverage_disadvantage == 0
        and cross_session_pairs == 0
        and float(paired_time_ratio["upper_95_percent"]) < 0.90
    )
    if can_continue:
        summary_ja = (
            "固定標本では、level 30・level 31の照合による手が深さ33で低い評価になった局面も、"
            "判断できない局面もなく、60秒の持ち時間を与える探索だけが採用した局面もなかった。"
            "さらに、対応を保った実時間比の95%区間上端が0.90未満だった。"
            "ただし、手の表への採用と強さの確認には別の先後入替2局対戦が必要である。"
        )
        summary_en = (
            "In this fixed sample, no level-30/level-31 move was lower at level 33 or unresolved, "
            "and there was no position accepted only by the time-managed search. The upper end of the "
            "paired 95% wall-time ratio was below 0.90. A separate color-swapped two-game match is still "
            "required before table adoption or any strength conclusion."
        )
    else:
        summary_ja = (
            "この固定標本だけでは、level 30・level 31の照合を残り局面の計算方法として選ばない。"
            "低い評価、判断不能、60秒の持ち時間を与える探索だけが採用した局面、または十分な速度差の"
            "少なくとも一つがあったためである。"
        )
        summary_en = (
            "This fixed sample does not support selecting the level-30/level-31 check for the remaining "
            "positions, because it had a lower evaluation, an unresolved check, a coverage disadvantage, "
            "or insufficient speed evidence."
        )
    payload: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "engine": {
            "path": exe.resolve().as_posix(),
            "sha256": sha256_file(exe),
        },
        "source_files": {
            "benchmark_script": {
                "path": Path(__file__).resolve().as_posix(),
                "sha256": sha256_file(Path(__file__)),
            },
            "teacher_script": {
                "path": Path(generate_ggs_root_teacher.__file__).resolve().as_posix(),
                "sha256": sha256_file(Path(generate_ggs_root_teacher.__file__)),
            },
        },
        "population": {
            "coverage": {
                "path": coverage.resolve().as_posix(),
                "sha256": sha256_file(coverage),
            },
            "excluded_root_results": root_file_provenance(excluded_root_results),
            "count": len(population),
            "sha256": hashlib.sha256("\n".join(population).encode("ascii")).hexdigest(),
        },
        "positions": {
            "population_count": len(population),
            "requested": positions,
            "sample_seed": sample_seed,
            "order_seed": order_seed,
            "bootstrap_seed": bootstrap_seed,
            "fixed_coverage_path": fixed_coverage.resolve().as_posix(),
            "fixed_coverage_sha256": sha256_file(fixed_coverage),
            "sha256": hashlib.sha256("\n".join(boards).encode("ascii")).hexdigest(),
        },
        "conditions": {
            "threads": THREADS,
            "hash": HASH_LEVEL,
            "minimum_depth": MIN_DEPTH,
            "minimum_selectivity": MIN_SELECTIVITY,
            TIME_METHOD: {
                "method": "time_then_verify",
                "time_seconds": TIME_SECONDS,
                "fallback_level": LEVEL_30,
                "verify_level": LEVEL_31,
            },
            LEVEL_METHOD: {
                "method": "hint_then_verify",
                "teacher_level": LEVEL_30,
                "verify_level": LEVEL_31,
            },
            "different_move_check": {
                "level": LEVEL_33,
                "root_searches": 2,
                "forced_move_analyses_per_move": 2,
                "books_disabled": True,
            },
        },
        "timing_seconds": {
            TIME_METHOD: {
                "total": sum(float(record[TIME_METHOD]["wall_seconds"]) for record in records),
            },
            LEVEL_METHOD: {
                "total": sum(float(record[LEVEL_METHOD]["wall_seconds"]) for record in records),
            },
            "level_33_checks": deep_elapsed,
            "positions_with_same_session_pairs": len(timing_records),
            "positions_excluded_from_paired_time_ratio_due_to_interruption": cross_session_pairs,
            "paired_level_30_then_level_31_to_time_managed_search_ratio": paired_time_ratio,
            "by_execution_order": timing_by_order,
        },
        "comparison": comparison,
        "calculation_method_decision": {
            "level_30_then_level_31_can_continue_to_larger_calculation": can_continue,
            "unsupported_or_unresolved_count": unsupported_or_unresolved,
            "one_sided_95_percent_upper_bound": upper_bound,
            "time_managed_search_rejections": time_rejections,
            "level_30_then_level_31_rejections": direct_rejections,
            "level_30_then_level_31_coverage_disadvantage": direct_coverage_disadvantage,
            "positions_excluded_from_paired_time_ratio_due_to_interruption": cross_session_pairs,
            "summary_ja": summary_ja,
            "summary_en": summary_en,
        },
        "positions_detail": records,
    }
    _atomic_write_text(
        output_dir / "benchmark.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_report(payload, output_dir / "README.md")
    return payload


def compare_methods(
    coverage: Path,
    excluded_root_results: list[Path],
    exe: Path,
    output_dir: Path,
    positions: int,
    sample_seed: int,
    order_seed: int,
    bootstrap_seed: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Compare the fixed two-method experiment while exclusively owning its output directory."""
    with file_lock(_experiment_lock_path(output_dir)):
        return _compare_methods_unlocked(
            coverage,
            excluded_root_results,
            exe,
            output_dir,
            positions,
            sample_seed,
            order_seed,
            bootstrap_seed,
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
        help="Accepted root-result file or existing root table to exclude (repeatable)",
    )
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=600)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=622)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when every frozen condition and every saved result still matches",
    )
    args = parser.parse_args()
    payload = compare_methods(
        args.coverage,
        args.exclude_root_results,
        args.exe.resolve(),
        args.output_dir,
        args.positions,
        args.sample_seed,
        args.order_seed,
        args.bootstrap_seed,
        args.resume,
    )
    comparison = payload["comparison"]
    print(
        f"positions={payload['positions']['requested']} "
        f"same_move={comparison['same_accepted_move']} "
        f"different_move={comparison['different_accepted_move']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
