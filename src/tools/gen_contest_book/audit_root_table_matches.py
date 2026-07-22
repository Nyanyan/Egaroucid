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

from build_book import canonicalize_board_key
from build_root_table import ROOT_TABLE_FILENAME, sha256_file


AUDIT_SCHEMA = "root_table_match_audit_v1"
BOOK_ZERO_NODES_RE = re.compile(r"level Book depth .* nodes 0", re.IGNORECASE)


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


def _validate_metadata(metadata_path: Path, prepared: dict[str, Any], match_count: int) -> list[str]:
    """Return concrete-condition violations, never ambiguous side labels."""
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
    for key, expected in (("time", 60), ("threads", 8), ("hash", 29), ("matches", match_count)):
        if parsed.get(key) != expected:
            failures.append(f"game metadata has {key}={parsed.get(key)!r}, expected {expected}")
    table_dir_text = parsed.get("candidate_contestbook")
    if not isinstance(table_dir_text, str):
        failures.append("table-using side has no temporary table directory")
    else:
        table_path = Path(table_dir_text) / ROOT_TABLE_FILENAME
        table = prepared.get("table")
        if not isinstance(table, dict) or not isinstance(table.get("sha256"), str):
            failures.append("prepared input has no table SHA-256")
        elif not table_path.is_file() or sha256_file(table_path) != table["sha256"]:
            failures.append("table used for games does not match the prepared table")
    if parsed.get("baseline_contestbook") is not None:
        failures.append("no-book side has a table directory")
    table_binary = artifacts.get("candidate_binary")
    no_book_binary = artifacts.get("baseline_binary")
    if not isinstance(table_binary, dict) or not isinstance(no_book_binary, dict):
        failures.append("game metadata lacks executable fingerprints")
    elif table_binary.get("sha256") != no_book_binary.get("sha256"):
        failures.append("table-using side and no-book side used different executables")
    for role in ("candidate", "baseline"):
        command = commands.get(role)
        if not isinstance(command, list) or "-nobook" not in command:
            failures.append(f"{role} engine command does not disable ordinary books")
    if openings.get("selected_count") != match_count:
        failures.append("metadata starting-position count differs from completed matches")
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


def _audit_rows(rows: list[dict[str, Any]], expected_boards: set[str]) -> tuple[dict[str, int], list[str]]:
    failures: list[str] = []
    seen_matches: set[int] = set()
    result_boards: list[str] = []
    table_log_text: list[str] = []
    no_book_log_text: list[str] = []
    engine_audits: list[dict[str, Any]] = []
    expected_selections = 0
    for row in rows:
        match = row.get("match")
        board = row.get("board")
        games = row.get("games")
        if not isinstance(match, int) or match in seen_matches:
            failures.append("match identifiers are missing or duplicated")
            continue
        seen_matches.add(match)
        if not isinstance(board, str) or not isinstance(games, list) or len(games) != 2:
            failures.append(f"match {match} does not contain one board and two games")
            continue
        result_boards.append(board)
        if row.get("result") not in {"W", "D", "L"} or not isinstance(row.get("margin"), (int, float)):
            failures.append(f"match {match} has invalid result or margin")
        differences: list[float] = []
        side_to_move = board[65] if len(board) == 66 and board[64] == " " else None
        colors: set[str] = set()
        for game in games:
            if not isinstance(game, dict):
                failures.append(f"match {match} has an invalid game record")
                continue
            color = game.get("candidate_color")
            if color not in {"X", "O"}:
                failures.append(f"match {match} has an invalid table-side color")
            else:
                colors.add(color)
                if color == side_to_move:
                    expected_selections += 1
            difference = game.get("candidate_disc_diff")
            if not isinstance(difference, (int, float)):
                failures.append(f"match {match} has an invalid table-side disc difference")
            else:
                differences.append(float(difference))
            engine_audit = game.get("engine_audit")
            if not isinstance(engine_audit, dict) or not isinstance(engine_audit.get("candidate"), dict) or not isinstance(engine_audit.get("baseline"), dict):
                failures.append(f"match {match} is missing engine audit data")
                continue
            table_audit = engine_audit["candidate"]
            no_book_audit = engine_audit["baseline"]
            engine_audits.extend([table_audit, no_book_audit])
            for audit, logs in ((table_audit, table_log_text), (no_book_audit, no_book_log_text)):
                log_text = audit.get("log")
                log_sha256 = audit.get("log_sha256")
                if not isinstance(log_text, str) or not isinstance(log_sha256, str):
                    failures.append(f"match {match} has log provenance missing")
                    continue
                log_path = Path(log_text)
                if not log_path.is_file() or sha256_file(log_path) != log_sha256:
                    failures.append(f"match {match} log does not match its SHA-256")
                    continue
                logs.append(log_path.read_text(encoding="utf-8", errors="replace"))
        if colors != {"X", "O"}:
            failures.append(f"match {match} did not exchange colors")
        if len(differences) == 2 and abs(sum(differences) - float(row["margin"])) > 1e-9:
            failures.append(f"match {match} margin does not equal its two game differences")
    if _canonical_set(result_boards) != expected_boards or len(result_boards) != len(expected_boards):
        failures.append("completed matches do not equal the prepared starting-position set")
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
) -> dict[str, Any]:
    if minimum_processed < 1:
        raise ValueError("minimum_processed must be positive")
    rows = _load_rows(results_path)
    prepared = _read_json(prepared_input_path, "prepared match input")
    if prepared.get("schema") != "prepared_root_table_match_input_v1":
        raise ValueError(f"{prepared_input_path}: unsupported prepared-input schema")
    expected_boards = _canonical_set(_load_openings(prepared))
    checks, failures = _audit_rows(rows, expected_boards)
    teacher_results = prepared.get("teacher_results")
    if not isinstance(teacher_results, dict) or not isinstance(teacher_results.get("processed"), int):
        failures.append("prepared input does not record a processed-position count")
    elif teacher_results["processed"] < minimum_processed:
        failures.append(
            f"prepared input processed {teacher_results['processed']} positions, "
            f"below required {minimum_processed}"
        )
    failures.extend(_validate_metadata(metadata_path, prepared, len(rows)))
    wins = sum(row["result"] == "W" for row in rows)
    draws = sum(row["result"] == "D" for row in rows)
    losses = sum(row["result"] == "L" for row in rows)
    score_rate = (wins + 0.5 * draws) / len(rows)
    mean_margin = sum(float(row["margin"]) for row in rows) / len(rows)
    intervals = _bootstrap_intervals(rows, bootstrap_seed, bootstrap_repetitions)
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
        "score_interval": intervals["score"],
        "margin_interval": intervals["margin"],
        "checks": checks,
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
    text = f"""# 開始局面用の手の表を使う対局の監査

## 日本語

- match数: {len(rows)}
- 表を使う側の勝ち・引分・負け: {wins}・{draws}・{losses}
- 得点率: {score_rate:.2%}
- 平均石差: {mean_margin:+.3f}石
- 得点率の95%区間: {intervals['score'][0]:.2%} から {intervals['score'][1]:.2%}
- 平均石差の95%区間: {intervals['margin'][0]:+.3f} から {intervals['margin'][1]:+.3f}石
- エンジン実行数: {checks['engine_executions']}
- 表の読込み: {checks['table_loaded']}回、表の選択: {checks['table_selected']}回、探索ノード0での表選択: {checks['table_zero_nodes']}回
- bookを使わない側での表選択: {checks['no_book_table_selected']}回
- 監査上の問題: {failure_text_ja}

判定: {decision_ja}

## English

- Matches: {len(rows)}
- Table-using side W/D/L: {wins}/{draws}/{losses}
- Score rate: {score_rate:.2%}
- Mean disc margin: {mean_margin:+.3f} discs
- 95% score-rate interval: {intervals['score'][0]:.2%} to {intervals['score'][1]:.2%}
- 95% mean-margin interval: {intervals['margin'][0]:+.3f} to {intervals['margin'][1]:+.3f} discs
- Engine executions: {checks['engine_executions']}
- Temporary-table loads: {checks['table_loaded']}; selections: {checks['table_selected']}; zero-node selections: {checks['table_zero_nodes']}
- Temporary-table selections by the no-book side: {checks['no_book_table_selected']}
- Audit failures: {failure_text_en}

Decision: {decision_en}
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
    args = parser.parse_args()
    payload = audit_match_results(
        args.results,
        args.prepared_input,
        args.metadata,
        args.output,
        args.bootstrap_seed,
        args.bootstrap_repetitions,
        args.minimum_processed,
    )
    print(
        f"matches={payload['matches']} valid={payload['valid']} "
        f"eligible_for_adoption={payload['eligible_for_adoption']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
