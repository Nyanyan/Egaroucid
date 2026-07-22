"""Write a compact bilingual progress report for root-move precomputation.

The generator keeps every completed position in a durable JSON-lines companion
file until the selected compaction interval. This tool reads both that file
and the last compacted state, without modifying either, so its counts remain
accurate between compactions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any

import generate_ggs_root_teacher


STATE_SUFFIX = ".state.json"


def output_path_from_state(state_path: Path) -> Path:
    if not state_path.name.endswith(STATE_SUFFIX):
        raise ValueError(f"state path must end with {STATE_SUFFIX}: {state_path}")
    return state_path.with_name(state_path.name[: -len(STATE_SUFFIX)])


def _read_state(state_path: Path) -> dict[str, Any]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read state {state_path}: {error}") from error
    if state.get("schema") != generate_ggs_root_teacher.TEACHER_SCHEMA:
        raise ValueError(f"{state_path}: unsupported state schema")
    for field in ("roots", "results", "rejections"):
        expected_type = list if field == "roots" else dict
        if not isinstance(state.get(field), expected_type):
            raise ValueError(f"{state_path}: invalid {field}")
    return state


def _pending_record_count(output_path: Path) -> int:
    path = generate_ggs_root_teacher._pending_updates_path(output_path)
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read pending updates {path}: {error}") from error


def progress_counts(state_path: Path) -> dict[str, int | str]:
    """Return counts from the compacted state plus durable pending updates."""
    state = _read_state(state_path)
    output_path = output_path_from_state(state_path)
    compacted_accepted = len(state["results"])
    compacted_rejected = len(state["rejections"])
    effective = copy.deepcopy(state)
    generate_ggs_root_teacher._apply_pending_updates(output_path, effective)
    accepted = len(effective["results"])
    rejected = len(effective["rejections"])
    requested = len(effective["roots"])
    processed = accepted + rejected
    return {
        "requested": requested,
        "accepted": accepted,
        "rejected": rejected,
        "processed": processed,
        "remaining": requested - processed,
        "compacted_accepted": compacted_accepted,
        "compacted_rejected": compacted_rejected,
        "pending_records": _pending_record_count(output_path),
        "state_schema": str(state["schema"]),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_progress_report(state_path: Path, report_path: Path) -> dict[str, int | str]:
    counts = progress_counts(state_path)
    output_path = output_path_from_state(state_path)
    pending_path = generate_ggs_root_teacher._pending_updates_path(output_path)
    text = f"""# 開始局面の最初の手の事前計算：進捗報告

## 日本語

- 対象局面数: {counts['requested']}
- 採用局面数（品質検査を通過し、最初の手を記録できた局面）: {counts['accepted']}
- 不採用局面数（品質検査を通過しなかった局面）: {counts['rejected']}
- 処理済み局面数: {counts['processed']}/{counts['requested']}
- 未処理局面数: {counts['remaining']}
- 状態ファイルへ統合済みの採用局面数: {counts['compacted_accepted']}
- 状態ファイルへ統合済みの不採用局面数: {counts['compacted_rejected']}
- 状態ファイルへの統合待ちとして安全に追記済みの記録数: {counts['pending_records']}
- 状態ファイル形式: `{counts['state_schema']}`

状態ファイルへ統合済みの件数には、直近の統合以降に完了した局面は含まれない。一方、追記ファイル `{pending_path.name}` の各記録は局面の完了時点で安全に保存される。`--resume` を指定して再開すると、この追記ファイルの内容を最初に状態ファイルへ反映する。したがって、「採用局面数」「不採用局面数」「処理済み局面数」は、統合待ちの安全な追記記録も含む実際に再開可能な件数である。

この報告書は進捗ファイルを読むだけであり、探索結果、状態ファイル、大会用の開始局面の手の表を変更しない。

## English

- Requested positions: {counts['requested']}
- Accepted positions, including durable append records: {counts['accepted']}
- Rejected positions, including durable append records: {counts['rejected']}
- Processed positions, including durable append records: {counts['processed']}/{counts['requested']}
- Remaining positions: {counts['remaining']}
- Accepted positions already compacted into the state file: {counts['compacted_accepted']}
- Rejected positions already compacted into the state file: {counts['compacted_rejected']}
- Records in the durable append file: {counts['pending_records']}
- State-file format: `{counts['state_schema']}`

The compacted counts do not include positions completed after the latest compaction. Records in `{pending_path.name}` are saved when each position completes and are applied to the state file first by `--resume`. The counts described as including durable append records are the actual resumable processed counts.

This report only reads files; it does not change search results, the state file, or the tournament table of moves for starting positions.
"""
    _atomic_write_text(report_path, text)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = write_progress_report(args.state, args.output)
    print(
        f"processed {counts['processed']}/{counts['requested']} "
        f"accepted={counts['accepted']} rejected={counts['rejected']} "
        f"pending={counts['pending_records']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
