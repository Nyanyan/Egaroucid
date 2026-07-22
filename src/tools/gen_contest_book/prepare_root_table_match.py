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

from build_book import canonicalize_board_key, move_to_representative
from build_root_table import (
    ROOT_TABLE_FILENAME,
    build_root_table,
    load_root_rows,
    sha256_file,
    validate_root_table,
)
from generate_ggs_root_teacher import (
    TEACHER_MANIFEST_SCHEMA,
    validate_calculation_provenance,
    validate_verified_teacher_result,
)
from othello import Board, coord_to_index, normalize_board_text


PREPARED_SCHEMA = "prepared_root_table_match_input_v5"


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
    if manifest.get("schema") != TEACHER_MANIFEST_SCHEMA:
        raise ValueError(f"{path}: unsupported teacher manifest schema")
    try:
        provenance = validate_calculation_provenance(manifest.get("calculation_provenance"))
    except ValueError as error:
        raise ValueError(f"{path}: invalid teacher-calculation evidence: {error}") from error
    if manifest.get("random_seed") != provenance["random_seed"]:
        raise ValueError(f"{path}: random seed does not match teacher-calculation evidence")
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


def _validate_manifest_results(
    entries: list[Any],
    manifest: dict[str, Any],
    teacher_results: Path,
) -> dict[str, tuple[str, dict[str, Any], int]]:
    """Require every exported row to agree with the frozen calculation result."""
    results = manifest.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{_manifest_path(teacher_results)}: accepted results are missing")
    boards = {entry.board for entry in entries}
    canonical_results: dict[str, tuple[str, dict[str, Any], int]] = {}
    for raw_board, result in results.items():
        if not isinstance(raw_board, str) or not isinstance(result, dict):
            raise ValueError(f"{_manifest_path(teacher_results)}: invalid accepted result")
        try:
            normalized_board = normalize_board_text(raw_board)
            canonical_board, symmetry = canonicalize_board_key(normalized_board)
        except ValueError as error:
            raise ValueError(
                f"{_manifest_path(teacher_results)}: invalid accepted-result board"
            ) from error
        if canonical_board in canonical_results:
            raise ValueError(
                f"{_manifest_path(teacher_results)}: duplicate canonical accepted result"
            )
        canonical_results[canonical_board] = (normalized_board, result, symmetry)
    if set(canonical_results) != boards:
        raise ValueError(
            f"{_manifest_path(teacher_results)}: accepted results do not equal teacher rows"
        )
    for entry in entries:
        raw_board, result, symmetry = canonical_results[entry.board]
        move = result.get("move")
        score = result.get("score")
        if not isinstance(move, str) or not isinstance(score, int):
            raise ValueError(f"{_manifest_path(teacher_results)}: invalid move or score for {entry.board}")
        try:
            source_move = coord_to_index(move)
        except ValueError as error:
            raise ValueError(
                f"{_manifest_path(teacher_results)}: invalid move for {entry.board}"
            ) from error
        if source_move not in Board.from_text(raw_board).legal_moves():
            raise ValueError(f"{_manifest_path(teacher_results)}: illegal move for {entry.board}")
        expected_moves = ((move_to_representative(source_move, symmetry), score),)
        if entry.value != score or entry.moves != expected_moves:
            raise ValueError(
                f"{_manifest_path(teacher_results)}: teacher row does not match result for {entry.board}"
            )
    return canonical_results


def _validate_level_verified_manifest_results(
    entries: list[Any],
    manifest: dict[str, Any],
    teacher_results: Path,
) -> None:
    """Require a complete level-verification trail for every accepted row."""
    required_integer_settings = (
        "min_depth",
        "min_selectivity",
        "fallback_level",
        "teacher_level",
        "verify_level",
    )
    settings: dict[str, int] = {}
    for key in required_integer_settings:
        value = manifest.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{_manifest_path(teacher_results)}: {key} is missing or invalid")
        settings[key] = value
    method = manifest.get("method")
    if not isinstance(method, str):
        raise ValueError(f"{_manifest_path(teacher_results)}: method is missing or invalid")
    canonical_results = _validate_manifest_results(entries, manifest, teacher_results)
    for entry in entries:
        raw_board, result, _symmetry = canonical_results[entry.board]
        try:
            validate_verified_teacher_result(
                raw_board,
                result,
                method=method,
                min_depth=settings["min_depth"],
                min_selectivity=settings["min_selectivity"],
                fallback_level=settings["fallback_level"],
                teacher_level=settings["teacher_level"],
                verify_level=settings["verify_level"],
            )
        except ValueError as error:
            raise ValueError(
                f"{_manifest_path(teacher_results)}: accepted result for {entry.board} "
                f"has no valid level-verification trail: {error}"
            ) from error


def _teacher_script_snapshot_bytes(manifest: dict[str, Any]) -> tuple[bytes, str]:
    provenance = manifest["calculation_provenance"]
    snapshot = provenance["teacher_script_snapshot"]
    snapshot_path = Path(snapshot["path"])
    try:
        content = snapshot_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read saved teacher script {snapshot_path}: {error}") from error
    digest = _sha256_bytes(content)
    if digest != snapshot["sha256"]:
        raise ValueError("saved teacher script changed while preparing match input")
    return content, digest


def _teacher_execution_environment_files(
    manifest: dict[str, Any],
) -> list[tuple[str, Path, bytes, dict[str, Any]]]:
    """Return the saved executable and Console inputs, preserving their layout."""
    provenance = manifest["calculation_provenance"]
    environment = provenance["execution_environment"]
    directory = environment["directory"]
    if not isinstance(directory, str):
        raise ValueError("saved teacher execution environment has no directory")
    root = Path(directory).resolve()
    raw_files: list[tuple[str, dict[str, Any]]] = []
    executable = environment.get("executable")
    if not isinstance(executable, dict) or not isinstance(executable.get("snapshot"), dict):
        raise ValueError("saved teacher execution environment has no executable")
    raw_files.append(("executable", executable["snapshot"]))
    resources = environment.get("resources")
    if not isinstance(resources, list):
        raise ValueError("saved teacher execution environment has no resources")
    for resource in resources:
        if not isinstance(resource, dict) or not isinstance(resource.get("role"), str):
            raise ValueError("saved teacher execution environment has an invalid resource")
        snapshot = resource.get("snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("saved teacher execution environment has an invalid resource copy")
        raw_files.append((resource["role"], snapshot))

    files: list[tuple[str, Path, bytes, dict[str, Any]]] = []
    for role, fingerprint in raw_files:
        path_text = fingerprint.get("path")
        digest = fingerprint.get("sha256")
        size = fingerprint.get("bytes")
        if not isinstance(path_text, str) or not isinstance(digest, str) or not isinstance(size, int):
            raise ValueError(f"saved teacher {role} fingerprint is invalid")
        source = Path(path_text).resolve()
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise ValueError(f"saved teacher {role} is outside its execution environment") from error
        try:
            content = source.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read saved teacher {role} {source}: {error}") from error
        if _sha256_bytes(content) != digest or len(content) != size:
            raise ValueError(f"saved teacher {role} changed while preparing match input")
        files.append((role, relative, content, fingerprint))
    return files


def prepare_match_input(
    teacher_results: Path,
    output_dir: Path,
    minimum_processed: int,
    minimum_accepted: int,
    require_level_31_verification: bool = False,
) -> dict[str, int | str]:
    if minimum_processed < 1:
        raise ValueError("minimum_processed must be positive")
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    if not isinstance(require_level_31_verification, bool):
        raise ValueError("require_level_31_verification must be a boolean")
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

    entries = load_root_rows(teacher_results, 14)
    boards = _accepted_start_boards(teacher_results)
    if sha256_file(teacher_results) != teacher_sha256:
        raise ValueError(f"{teacher_results}: changed while its rows were being checked")
    if len(entries) != completed or len(boards) != completed:
        raise ValueError(
            f"{teacher_results}: accepted rows {len(entries)} do not match manifest {completed}"
        )
    canonical_boards = {entry.board for entry in entries}
    if len(canonical_boards) != len(entries):
        raise ValueError(f"{teacher_results}: duplicate canonical accepted roots")
    _validate_manifest_results(entries, teacher_manifest, teacher_results)
    if require_level_31_verification:
        verify_level = teacher_manifest.get("verify_level")
        if isinstance(verify_level, bool) or not isinstance(verify_level, int) or verify_level < 31:
            raise ValueError(
                f"{_manifest_path(teacher_results)}: formal matches require verify_level at least 31"
            )
        _validate_level_verified_manifest_results(
            entries,
            teacher_manifest,
            teacher_results,
        )
    teacher_script_content, teacher_script_sha256 = _teacher_script_snapshot_bytes(
        teacher_manifest
    )
    teacher_execution_files = _teacher_execution_environment_files(teacher_manifest)

    snapshot = output_dir / "teacher_rows.txt"
    _atomic_write_bytes(snapshot, teacher_content)
    if sha256_file(snapshot) != teacher_sha256:
        raise RuntimeError(f"{snapshot}: copied SHA-256 mismatch")
    manifest_snapshot = output_dir / "teacher_manifest.json"
    _atomic_write_bytes(manifest_snapshot, teacher_manifest_content)
    if sha256_file(manifest_snapshot) != teacher_manifest_sha256:
        raise RuntimeError(f"{manifest_snapshot}: copied SHA-256 mismatch")
    teacher_script_snapshot = output_dir / "teacher_calculation_script.py"
    _atomic_write_bytes(teacher_script_snapshot, teacher_script_content)
    if sha256_file(teacher_script_snapshot) != teacher_script_sha256:
        raise RuntimeError(f"{teacher_script_snapshot}: copied SHA-256 mismatch")
    teacher_execution_environment = output_dir / "teacher_execution_environment"
    prepared_execution_files: list[dict[str, Any]] = []
    for role, relative, content, fingerprint in teacher_execution_files:
        destination = teacher_execution_environment / relative
        _atomic_write_bytes(destination, content)
        if sha256_file(destination) != fingerprint["sha256"] or destination.stat().st_size != fingerprint["bytes"]:
            raise RuntimeError(f"{destination}: copied teacher {role} mismatch")
        prepared_execution_files.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                "path": destination.resolve().as_posix(),
                "sha256": fingerprint["sha256"],
                "bytes": fingerprint["bytes"],
            }
        )
    try:
        validate_calculation_provenance(
            teacher_manifest["calculation_provenance"],
            snapshot_path=teacher_script_snapshot,
            execution_environment_directory=teacher_execution_environment,
        )
    except ValueError as error:
        raise RuntimeError(f"prepared teacher calculation evidence is inconsistent: {error}") from error

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
        "teacher_script_snapshot": {
            "path": teacher_script_snapshot.resolve().as_posix(),
            "sha256": sha256_file(teacher_script_snapshot),
        },
        "teacher_execution_environment": {
            "path": teacher_execution_environment.resolve().as_posix(),
            "files": prepared_execution_files,
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
            "level_31_verification_required": require_level_31_verification,
        },
    }
    _atomic_write_text(
        output_dir / "prepared_match_input.json",
        json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    report = f"""# 開始局面用の手の表を使う対局の入力ファイル

## 日本語

このディレクトリは、`{teacher_results.name}` の一貫した出力を複製して作成した。元の出力の処理済み局面数は {processed}、受理局面数は {completed}、不採用局面数は {rejected} である。受理局面はすべて一時的な開始局面用の手の表と開始局面一覧へ入れた。対局結果を見て局面を選び直していない。教師計算時に通常bookと大会bookを無効にした起動条件と、使用した生成スクリプトの保存コピーもSHA-256で照合して固定した。

各受理局面で level 31 以上の照合を必須にしたか: {str(require_level_31_verification).lower()}。true の場合、各採用手について、manifestに記録された固定levelの探索結果、手、探索品質、設定値の対応を検査してから、この入力を作成している。

このディレクトリは対局の入力専用であり、大会用の `trained/` は変更しない。教師計算で使ったConsole実行ファイル、主評価ファイル、終盤の手順評価ファイルの保存コピーもSHA-256で照合して保存した。入力と出力のSHA-256は `prepared_match_input.json` に記録した。

## English

This directory was created by copying one coherent output from `{teacher_results.name}`. The source output processed {processed} positions, accepted {completed}, and rejected {rejected}. Every accepted position was placed in the temporary table of moves for starting positions and in the starting-position list. No position was selected after observing game results. The immutable manifest records that both books were disabled during teacher calculation, and this directory contains a SHA-256-checked copy of the generator script.

Level-31-or-higher verification required for every accepted position: {str(require_level_31_verification).lower()}. When true, this input is created only after checking that each accepted move is supported by the recorded fixed-level searches, moves, search quality, and manifest settings.

This directory is only match input and does not modify tournament `trained/`. It also preserves SHA-256-checked copies of the Console executable, the main evaluation file, and the endgame move-ordering evaluation file used for teacher calculation. SHA-256 values for every input and output are recorded in `prepared_match_input.json`.
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
    parser.add_argument("--require-level-31-verification", action="store_true")
    args = parser.parse_args()
    result = prepare_match_input(
        args.teacher_results,
        args.output_dir,
        args.minimum_processed,
        args.minimum_accepted,
        args.require_level_31_verification,
    )
    print(
        f"prepared processed={result['processed']} accepted={result['accepted']} "
        f"rejected={result['rejected']} table_entries={result['table_entries']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
