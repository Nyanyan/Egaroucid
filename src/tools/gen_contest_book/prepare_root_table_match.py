"""Prepare immutable input files for matches that test a root-move table.

The teacher generator updates its public result file only at a durable
compaction.  This tool copies one such coherent result, verifies its manifest,
builds a table from every accepted row, and records every input/output hash.
It deliberately does not run games or modify ``trained``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from build_root_table import (
    ROOT_TABLE_FILENAME,
    build_root_table,
    load_root_rows,
    sha256_file,
    validate_root_table,
)
from othello import normalize_board_text


PREPARED_SCHEMA = "prepared_root_table_match_input_v2"


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


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _manifest_path(teacher_results: Path) -> Path:
    return teacher_results.with_suffix(teacher_results.suffix + ".manifest.json")


def _load_manifest_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        content = path.read_bytes()
        manifest = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read teacher manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("output"), dict):
        raise ValueError(f"{path}: invalid teacher manifest")
    return content, manifest


def _accepted_start_boards(teacher_results: Path) -> list[str]:
    boards: list[str] = []
    try:
        lines = teacher_results.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read teacher results {teacher_results}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{teacher_results}:{line_number}: invalid teacher row")
        boards.append(normalize_board_text(parts[0] + " " + parts[1]))
    return boards


def _read_coherent_teacher_results(
    teacher_results: Path,
) -> tuple[bytes, str, bytes, str, dict[str, Any]]:
    manifest_path = _manifest_path(teacher_results)
    manifest_content, manifest = _load_manifest_bytes(manifest_path)
    try:
        content = teacher_results.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read teacher results {teacher_results}: {error}") from error
    content_sha256 = _sha256_bytes(content)
    if manifest["output"].get("sha256") != content_sha256:
        raise ValueError(
            f"{manifest_path}: output SHA-256 does not match {teacher_results}; "
            "wait for the next completed output update"
        )
    if sha256_file(teacher_results) != content_sha256:
        raise ValueError(f"{teacher_results}: changed while it was being copied")
    return (
        content,
        content_sha256,
        manifest_content,
        _sha256_bytes(manifest_content),
        manifest,
    )


def prepare_match_input(
    teacher_results: Path,
    output_dir: Path,
    minimum_processed: int,
    minimum_accepted: int,
) -> dict[str, int | str]:
    if minimum_processed < 1:
        raise ValueError("minimum_processed must be positive")
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    (
        teacher_content,
        teacher_sha256,
        teacher_manifest_content,
        teacher_manifest_sha256,
        teacher_manifest,
    ) = _read_coherent_teacher_results(teacher_results)
    output_metadata = teacher_manifest["output"]
    processed = output_metadata.get("processed")
    completed = output_metadata.get("completed")
    rejected = output_metadata.get("rejected")
    if not all(isinstance(value, int) and value >= 0 for value in (processed, completed, rejected)):
        raise ValueError(f"{_manifest_path(teacher_results)}: invalid output counts")
    if processed < minimum_processed:
        raise ValueError(
            f"{teacher_results}: processed {processed}, below required {minimum_processed}"
        )
    if processed != completed + rejected:
        raise ValueError(f"{_manifest_path(teacher_results)}: inconsistent output counts")
    if completed < minimum_accepted:
        raise ValueError(
            f"{teacher_results}: accepted {completed}, below required {minimum_accepted}"
        )

    snapshot = output_dir / "teacher_rows.txt"
    _atomic_write_bytes(snapshot, teacher_content)
    if sha256_file(snapshot) != teacher_sha256:
        raise RuntimeError(f"{snapshot}: copied SHA-256 mismatch")
    manifest_snapshot = output_dir / "teacher_manifest.json"
    _atomic_write_bytes(manifest_snapshot, teacher_manifest_content)
    if sha256_file(manifest_snapshot) != teacher_manifest_sha256:
        raise RuntimeError(f"{manifest_snapshot}: copied SHA-256 mismatch")

    entries = load_root_rows(snapshot, 14)
    boards = _accepted_start_boards(snapshot)
    if len(entries) != completed or len(boards) != completed:
        raise ValueError(
            f"{teacher_results}: accepted rows {len(entries)} do not match manifest {completed}"
        )
    canonical_boards = {entry.board for entry in entries}
    if len(canonical_boards) != len(entries):
        raise ValueError(f"{teacher_results}: duplicate canonical accepted roots")

    openings = output_dir / "openings" / "roots.txt"
    _atomic_write_text(openings, "\n".join(boards) + "\n")
    table = output_dir / "table" / ROOT_TABLE_FILENAME
    table_result = build_root_table([], table, root_result_files=[snapshot])
    table_validation = validate_root_table(table, expected_root_discs=14)
    if table_validation["entries"] != completed:
        raise RuntimeError("built table has an unexpected accepted-root count")

    prepared = {
        "schema": PREPARED_SCHEMA,
        "teacher_results": {
            "path": teacher_results.resolve().as_posix(),
            "sha256": teacher_sha256,
            "processed": processed,
            "accepted": completed,
            "rejected": rejected,
        },
        "teacher_manifest": {
            "path": manifest_snapshot.resolve().as_posix(),
            "sha256": sha256_file(manifest_snapshot),
        },
        "snapshot": {
            "path": snapshot.resolve().as_posix(),
            "sha256": sha256_file(snapshot),
        },
        "openings": {
            "path": openings.resolve().as_posix(),
            "sha256": sha256_file(openings),
            "entries": len(boards),
        },
        "table": {
            "path": table.resolve().as_posix(),
            "sha256": str(table_result["output_sha256"]),
            "entries": int(table_result["entries"]),
            "manifest_sha256": sha256_file(table.with_suffix(table.suffix + ".manifest.json")),
        },
        "selection": {
            "minimum_processed": minimum_processed,
            "minimum_accepted": minimum_accepted,
            "all_accepted_positions_used": True,
        },
    }
    _atomic_write_text(
        output_dir / "prepared_match_input.json",
        json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    report = f"""# 開始局面用の手の表を使う対局の入力ファイル

## 日本語

このディレクトリは、`{teacher_results.name}` の一貫した出力を複製して作成した。元の出力の処理済み局面数は {processed}、受理局面数は {completed}、不採用局面数は {rejected} である。受理局面はすべて一時的な開始局面用の手の表と開始局面一覧へ入れた。対局結果を見て局面を選び直していない。

このディレクトリは対局の入力専用であり、大会用の `trained/` は変更しない。入力と出力のSHA-256は `prepared_match_input.json` に記録した。

## English

This directory was created by copying one coherent output from `{teacher_results.name}`. The source output processed {processed} positions, accepted {completed}, and rejected {rejected}. Every accepted position was placed in the temporary table of moves for starting positions and in the starting-position list. No position was selected after observing game results.

This directory is only match input and does not modify tournament `trained/`. SHA-256 values for every input and output are recorded in `prepared_match_input.json`.
"""
    _atomic_write_text(output_dir / "README.md", report)
    return {
        "processed": processed,
        "accepted": completed,
        "rejected": rejected,
        "table_entries": int(table_result["entries"]),
        "table_sha256": str(table_result["output_sha256"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-processed", type=int, default=500)
    parser.add_argument("--minimum-accepted", type=int, default=500)
    args = parser.parse_args()
    result = prepare_match_input(
        args.teacher_results,
        args.output_dir,
        args.minimum_processed,
        args.minimum_accepted,
    )
    print(
        f"prepared processed={result['processed']} accepted={result['accepted']} "
        f"rejected={result['rejected']} table_entries={result['table_entries']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
