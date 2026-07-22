"""Audit durable progress from a fixed root-teacher calculation-method comparison.

The comparison program validates saved work before resuming.  This independent
reader makes the same important artifacts visible while a long comparison is
still in progress: fixed conditions, saved-result SHA-256 values, and the
count of positions for which both calculations (and any required level-33
check) are complete.  It never starts an engine or changes comparison files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import benchmark_root_teacher_methods as benchmark
from build_root_table import sha256_file


AUDIT_SCHEMA = "root_teacher_method_benchmark_audit_v1"


def _atomic_write_text(path: Path, text: str) -> None:
    benchmark._atomic_write_text(path, text)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    return benchmark._read_json(path, description)


def _current_file_matches(provenance: Any) -> bool:
    if not isinstance(provenance, dict):
        return False
    path_text = provenance.get("path")
    expected_sha256 = provenance.get("sha256")
    if not isinstance(path_text, str) or not isinstance(expected_sha256, str):
        return False
    path = Path(path_text)
    return path.is_file() and sha256_file(path) == expected_sha256


def _format_failures(failures: list[str]) -> str:
    if not failures:
        return "- なし\n"
    return "".join(f"- `{failure}`\n" for failure in failures)


def _write_report(payload: dict[str, Any], path: Path) -> None:
    counts = payload["counts"]
    timing = payload["timing_seconds"]
    failures = payload["failures"]
    status_ja = "有効" if payload["valid"] else "無効"
    status_en = "valid" if payload["valid"] else "invalid"
    if timing["same_session_pairs"]:
        ratio_ja = f"{timing['level_30_then_level_31_to_time_managed_search_ratio']:.4f}"
        ratio_en = ratio_ja
    else:
        ratio_ja = "該当なし"
        ratio_en = "not available"
    report = f"""# 固定局面による計算方法比較の保存状態の監査

## 日本語

- 監査結果: **{status_ja}**
- 固定局面数: {counts['requested']}
- 少なくとも一方の計算を保存した局面: {counts['positions_with_a_completed_calculation']}
- 両方の計算を保存した局面: {counts['positions_with_both_calculations_complete']}
- 必要なlevel 33確認まで保存した局面: {counts['completed_positions']}
- 確認した結果ファイル: {counts['saved_method_results']}
- 手が異なり、level 33確認が必要な局面: {counts['different_accepted_moves']}
- 進行中の計算: {counts['in_progress_positions']}
- 同じプログラム起動中に両方の計時を得た局面: {timing['same_session_pairs']}
- その局面での60秒の持ち時間を与える探索の合計実時間: {timing['time_managed_search_total']:.3f}秒
- その局面でのlevel 30・level 31の照合の合計実時間: {timing['level_30_then_level_31_total']:.3f}秒
- 合計実時間の比（level 30・level 31の照合 ÷ 60秒の持ち時間を与える探索）: {ratio_ja}

この監査は、状態ファイル、固定局面一覧、進捗ファイル、保存済みの結果ファイルとmanifestのSHA-256、実行ファイル、二つのPythonスクリプトを照合する。対象110,766局面の一覧そのものは、状態ファイルに記録された入力SHA-256を照合することで確認する。探索、対局、手の表の作成は行わない。

上記の実時間は完了済み局面の途中集計であり、600局面が完了するまでは方法選択に使わない。

### 検出した問題

{_format_failures(failures)}
## English

- Audit result: **{status_en}**
- Fixed positions: {counts['requested']}
- Positions with at least one saved calculation: {counts['positions_with_a_completed_calculation']}
- Positions with both calculations saved: {counts['positions_with_both_calculations_complete']}
- Positions saved through any required level-33 check: {counts['completed_positions']}
- Saved result files checked: {counts['saved_method_results']}
- Positions with different accepted moves that require a level-33 check: {counts['different_accepted_moves']}
- Calculations in progress: {counts['in_progress_positions']}
- Positions whose two times were measured in one program invocation: {timing['same_session_pairs']}
- Total wall time for the search given 60 seconds of remaining game time on those positions: {timing['time_managed_search_total']:.3f} seconds
- Total wall time for the level-30/level-31 check on those positions: {timing['level_30_then_level_31_total']:.3f} seconds
- Total wall-time ratio (level-30/level-31 check ÷ search given 60 seconds of remaining game time): {ratio_en}

This audit compares the state file, fixed-position list, progress file, saved-result and manifest SHA-256 values, executable, and two Python scripts. It checks the 110,766-position input through the input SHA-256 stored in the state file; it does not parse that entire input again. It does not start an engine, play a game, or build a move table.

The wall-time values above are an interim aggregate of completed positions and are not used to select a method until all 600 positions are complete.

### Detected problems

{_format_failures(failures)}"""
    _atomic_write_text(path, report)


def audit_benchmark(output_dir: Path, report_path: Path) -> dict[str, Any]:
    """Return and write an evidence-only audit of one comparison output directory."""
    failures: list[str] = []
    state_path = output_dir / "experiment_state.json"
    progress_path = output_dir / "comparison_progress.json"
    state: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    boards: list[str] = []
    orders: dict[str, tuple[str, str]] = {}
    requested = 0

    try:
        state = _read_json(state_path, "experiment state")
        if state.get("schema") != benchmark.EXPERIMENT_STATE_SCHEMA:
            failures.append("unexpected experiment-state schema")
        positions = state.get("positions")
        if not isinstance(positions, dict):
            failures.append("experiment state has invalid positions")
        else:
            raw_boards = positions.get("boards")
            if not isinstance(raw_boards, list) or not all(isinstance(board, str) for board in raw_boards):
                failures.append("experiment state has invalid fixed boards")
            else:
                boards = list(raw_boards)
                requested = positions.get("requested") if isinstance(positions.get("requested"), int) else 0
                if requested != len(boards) or len(boards) != len(set(boards)):
                    failures.append("experiment state has inconsistent fixed-board count")
                expected_sha256 = hashlib.sha256("\n".join(boards).encode("ascii")).hexdigest()
                if positions.get("sha256") != expected_sha256:
                    failures.append("experiment state fixed-board SHA-256 does not match")
                order_seed = positions.get("order_seed")
                if isinstance(order_seed, int):
                    orders = benchmark._order_boards(boards, order_seed)
                    if positions.get("execution_orders") != [list(orders[board]) for board in boards]:
                        failures.append("experiment state execution orders do not match their seed")
                else:
                    failures.append("experiment state has invalid execution-order seed")
                fixed_path_text = positions.get("fixed_coverage_path")
                if not isinstance(fixed_path_text, str):
                    failures.append("experiment state has no fixed coverage path")
                else:
                    fixed_path = Path(fixed_path_text)
                    if not fixed_path.is_file() or positions.get("fixed_coverage_sha256") != sha256_file(fixed_path):
                        failures.append("fixed coverage file does not match its recorded SHA-256")
                    else:
                        try:
                            if benchmark._load_fixed_coverage_boards(fixed_path) != boards:
                                failures.append("fixed coverage file does not match state boards")
                        except ValueError as error:
                            failures.append(str(error))
        if state.get("conditions") != benchmark._conditions():
            failures.append("experiment state calculation conditions differ from this program")
        if not _current_file_matches(state.get("engine")):
            failures.append("engine executable does not match recorded SHA-256")
        source_files = state.get("source_files")
        if not isinstance(source_files, dict):
            failures.append("experiment state has invalid source-file provenance")
        else:
            for name in ("benchmark_script", "teacher_script"):
                if not _current_file_matches(source_files.get(name)):
                    failures.append(f"{name} does not match recorded SHA-256")
    except ValueError as error:
        failures.append(str(error))

    if state and boards and orders and requested > 0:
        try:
            records = benchmark._load_experiment_progress(output_dir, state_path, requested)
            records_by_index = benchmark._validate_progress_records(records, boards, orders, output_dir)
            if sorted(records_by_index) != list(range(1, len(records_by_index) + 1)):
                failures.append("progress records are not a continuous prefix")
        except ValueError as error:
            failures.append(str(error))
    elif not failures:
        failures.append("state does not provide a usable fixed sample")

    completed = [record for record in records if benchmark._record_is_complete(record)]
    with_one = [
        record
        for record in records
        if any(benchmark._has_method_result(record, method) for method in (benchmark.TIME_METHOD, benchmark.LEVEL_METHOD))
    ]
    with_both = [record for record in records if benchmark._record_has_both_methods(record)]
    saved_method_results = sum(
        1
        for record in records
        for method in (benchmark.TIME_METHOD, benchmark.LEVEL_METHOD)
        if benchmark._has_method_result(record, method)
    )
    different_accepted_moves = sum(
        1 for record in with_both if benchmark._record_requires_deep_check(record)
    )
    in_progress_positions = sum(1 for record in records if isinstance(record.get("in_progress"), dict))
    same_session_pairs = benchmark._records_with_same_session_pair(completed)
    time_managed_search_total = sum(
        float(record[benchmark.TIME_METHOD]["wall_seconds"]) for record in same_session_pairs
    )
    level_30_then_level_31_total = sum(
        float(record[benchmark.LEVEL_METHOD]["wall_seconds"]) for record in same_session_pairs
    )
    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "output_dir": output_dir.resolve().as_posix(),
        "experiment_state_path": state_path.resolve().as_posix(),
        "experiment_state_sha256": sha256_file(state_path) if state_path.is_file() else None,
        "comparison_progress_path": progress_path.resolve().as_posix(),
        "comparison_progress_sha256": sha256_file(progress_path) if progress_path.is_file() else None,
        "valid": not failures,
        "failures": failures,
        "counts": {
            "requested": requested,
            "positions_with_a_completed_calculation": len(with_one),
            "positions_with_both_calculations_complete": len(with_both),
            "completed_positions": len(completed),
            "saved_method_results": saved_method_results,
            "different_accepted_moves": different_accepted_moves,
            "in_progress_positions": in_progress_positions,
        },
        "timing_seconds": {
            "same_session_pairs": len(same_session_pairs),
            "time_managed_search_total": time_managed_search_total,
            "level_30_then_level_31_total": level_30_then_level_31_total,
            "level_30_then_level_31_to_time_managed_search_ratio": (
                level_30_then_level_31_total / time_managed_search_total
                if time_managed_search_total > 0.0
                else None
            ),
        },
    }
    _atomic_write_text(
        report_path.with_suffix(".json"),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_report(payload, report_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_benchmark(args.output_dir, args.report)
    print(
        f"valid={str(payload['valid']).lower()} "
        f"completed={payload['counts']['completed_positions']}/"
        f"{payload['counts']['requested']}"
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
