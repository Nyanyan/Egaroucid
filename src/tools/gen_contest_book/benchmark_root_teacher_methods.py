"""Compare two reproducible root-move calculations on the same r14 positions.

The input is a compacted teacher-output file.  Its first requested number of
accepted positions is copied into a small coverage report, then both methods
calculate every one of those positions independently:

* a search given 60 seconds of remaining game time, followed by level-30/31
  checks when necessary;
* a level-30 search followed by a level-31 check, without the initial
  time-managed search.

This tool writes only to a new experiment directory.  It never changes the
tournament table or the reference teacher output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from audit_r14_corpus import CORPUS_REPORT_SCHEMA
from build_root_table import load_root_rows, sha256_file
from generate_ggs_root_teacher import (
    generate_teachers,
    search_root_at_level,
    validate_quality,
)


BENCHMARK_SCHEMA = "root_teacher_method_benchmark_v1"
TIME_SECONDS = 60.0
THREADS = 28
HASH_LEVEL = 29
MIN_DEPTH = 30
MIN_SELECTIVITY = 74
LEVEL_30 = 30
LEVEL_31 = 31
LEVEL_33 = 33


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


def _load_coherent_reference(reference: Path) -> tuple[list[str], dict[str, Any]]:
    manifest_path = reference.with_suffix(reference.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read reference manifest {manifest_path}: {error}") from error
    output = manifest.get("output") if isinstance(manifest, dict) else None
    if not isinstance(output, dict) or output.get("sha256") != sha256_file(reference):
        raise ValueError("reference output does not match its manifest")
    if manifest.get("schema") != "ggs_root_teacher_manifest_v10":
        raise ValueError("reference manifest has an unsupported schema")
    rows = load_root_rows(reference, 14)
    if output.get("completed") != len(rows):
        raise ValueError("reference manifest accepted-position count does not match its rows")
    return [entry.board for entry in rows], manifest


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


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read generated manifest {manifest_path}: {error}") from error
    output = manifest.get("output") if isinstance(manifest, dict) else None
    if not isinstance(output, dict) or output.get("sha256") != sha256_file(path):
        raise ValueError(f"generated output does not match its manifest: {path}")
    return manifest


def _run_method(
    coverage: Path,
    exe: Path,
    output: Path,
    method: str,
) -> tuple[dict[str, Any], float]:
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
        LEVEL_30,
        method,
        LEVEL_30,
        LEVEL_31,
        checkpoint_every=1,
    )
    elapsed_seconds = time.monotonic() - started
    return _load_manifest(output), elapsed_seconds


def _moves_by_board(manifest: dict[str, Any]) -> dict[str, str]:
    results = manifest.get("results")
    if not isinstance(results, dict):
        raise ValueError("generated manifest has no results")
    moves: dict[str, str] = {}
    for board, result in results.items():
        if not isinstance(board, str) or not isinstance(result, dict) or not isinstance(result.get("move"), str):
            raise ValueError("generated manifest has an invalid accepted result")
        moves[board] = result["move"]
    return moves


def compare_methods(
    reference_results: Path,
    exe: Path,
    output_dir: Path,
    positions: int,
) -> dict[str, Any]:
    if positions <= 0:
        raise ValueError("positions must be positive")
    if output_dir.exists():
        raise FileExistsError(f"benchmark output directory already exists: {output_dir}")
    if not exe.is_file():
        raise FileNotFoundError(f"engine executable not found: {exe}")
    reference_boards, reference_manifest = _load_coherent_reference(reference_results)
    reference_engine = reference_manifest.get("engine")
    if not isinstance(reference_engine, dict) or reference_engine.get("sha256") != sha256_file(exe):
        raise ValueError("reference results were calculated with a different executable")
    if len(reference_boards) < positions:
        raise ValueError(
            f"reference has {len(reference_boards)} accepted positions, below requested {positions}"
        )
    boards = reference_boards[:positions]
    if len(set(boards)) != len(boards):
        raise ValueError("reference positions are not unique")
    output_dir.mkdir(parents=True)
    coverage = output_dir / "fixed_positions_coverage.json"
    _write_coverage(coverage, boards)
    time_output = output_dir / "time_managed_60_second_search.txt"
    level_output = output_dir / "level_30_then_level_31.txt"
    time_manifest, time_elapsed = _run_method(
        coverage, exe, time_output, "time_then_verify"
    )
    level_manifest, level_elapsed = _run_method(
        coverage, exe, level_output, "hint_then_verify"
    )
    time_moves = _moves_by_board(time_manifest)
    level_moves = _moves_by_board(level_manifest)
    common = sorted(set(time_moves) & set(level_moves))
    differing = [
        {
            "board": board,
            "time_managed_move": time_moves[board],
            "level_30_then_31_move": level_moves[board],
        }
        for board in common
        if time_moves[board] != level_moves[board]
    ]
    deep_started = time.monotonic()
    for difference in differing:
        deep_result = search_root_at_level(
            exe, difference["board"], LEVEL_33, THREADS, HASH_LEVEL
        )
        validate_quality(deep_result, MIN_DEPTH, MIN_SELECTIVITY)
        difference["level_33_move"] = deep_result["move"]
        difference["time_managed_matches_level_33"] = (
            difference["time_managed_move"] == deep_result["move"]
        )
        difference["level_30_then_31_matches_level_33"] = (
            difference["level_30_then_31_move"] == deep_result["move"]
        )
    deep_elapsed = time.monotonic() - deep_started
    time_output_info = time_manifest["output"]
    level_output_info = level_manifest["output"]
    payload: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "reference": {
            "path": reference_results.resolve().as_posix(),
            "sha256": sha256_file(reference_results),
            "manifest_engine_sha256": reference_manifest.get("engine", {}).get("sha256"),
        },
        "engine": {
            "path": exe.resolve().as_posix(),
            "sha256": sha256_file(exe),
        },
        "positions": {
            "requested": positions,
            "coverage_path": coverage.resolve().as_posix(),
            "coverage_sha256": sha256_file(coverage),
            "sha256": hashlib.sha256("\n".join(boards).encode("utf-8")).hexdigest(),
        },
        "conditions": {
            "threads": THREADS,
            "hash": HASH_LEVEL,
            "minimum_depth": MIN_DEPTH,
            "minimum_selectivity": MIN_SELECTIVITY,
            "time_managed_search": {
                "description": "A search given 60 seconds of remaining game time, then level 30 and level 31 checks when required.",
                "method": "time_then_verify",
                "time_seconds": TIME_SECONDS,
                "fallback_level": LEVEL_30,
                "verify_level": LEVEL_31,
            },
            "level_search": {
                "description": "A level-30 search followed by a level-31 check, with a repeated level-31 check when they disagree.",
                "method": "hint_then_verify",
                "teacher_level": LEVEL_30,
                "verify_level": LEVEL_31,
            },
        },
        "time_managed_search": {
            "wall_seconds": time_elapsed,
            "accepted": time_output_info.get("completed"),
            "rejected": time_output_info.get("rejected"),
            "output_sha256": time_output_info.get("sha256"),
        },
        "level_30_then_31": {
            "wall_seconds": level_elapsed,
            "accepted": level_output_info.get("completed"),
            "rejected": level_output_info.get("rejected"),
            "output_sha256": level_output_info.get("sha256"),
        },
        "comparison": {
            "accepted_by_both": len(common),
            "same_move": len(common) - len(differing),
            "different_move": len(differing),
            "level_33_check_wall_seconds": deep_elapsed,
            "time_managed_matches_level_33": sum(
                bool(item["time_managed_matches_level_33"]) for item in differing
            ),
            "level_30_then_31_matches_level_33": sum(
                bool(item["level_30_then_31_matches_level_33"]) for item in differing
            ),
            "level_30_then_31_supported_for_every_different_move": all(
                bool(item["level_30_then_31_matches_level_33"])
                for item in differing
            ),
            "differences": differing,
        },
    }
    _atomic_write_text(
        output_dir / "benchmark.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    report = f"""# 最初の手を計算する二つの方法の比較

## 日本語

この比較では、`{reference_results.name}` の先頭から固定した {positions} 局面を、両方の方法で計算する。対局結果を見て局面を選び直さない。

- 60秒の持ち時間を与える探索: 60秒の残り持ち時間を渡す探索を行い、品質条件に届かない場合はlevel 30、最後にlevel 31で照合する。
- level 30・level 31の照合: level 30で手を選び、level 31で照合する。手が異なる場合はlevel 31をもう一度実行し、2回のlevel 31が一致したときだけその手を採用する。
- 両方で採用された局面数: {len(common)}
- 最初の手が一致した局面数: {len(common) - len(differing)}
- 最初の手が異なった局面数: {len(differing)}
- 手が異なった局面でlevel 33と一致した数（60秒の持ち時間を与える探索・level 30とlevel 31の照合）: {payload['comparison']['time_managed_matches_level_33']}・{payload['comparison']['level_30_then_31_matches_level_33']}
- 60秒の持ち時間を与える探索の実時間: {time_elapsed:.3f}秒
- level 30・level 31の照合の実時間: {level_elapsed:.3f}秒
- 手が異なった局面に対するlevel 33の追加探索時間: {deep_elapsed:.3f}秒

この比較だけでは大会用の表を変更しない。採用するかどうかは、同じ開始局面から先後を入れ替えた2局の対局による別の検証で決める。

## English

This comparison recalculates the first {positions} fixed positions from `{reference_results.name}` with both methods. It does not select positions after observing game results.

- Time-managed search: give the engine 60 seconds of remaining game time, use level 30 if its quality is insufficient, then verify with level 31.
- Level-30/31 check: choose with level 30 and verify with level 31. When those moves differ, repeat level 31 and accept that move only if the two level-31 searches agree.
- Positions accepted by both methods: {len(common)}
- Positions with the same first move: {len(common) - len(differing)}
- Positions with different first moves: {len(differing)}
- Different-move positions matching level 33 (time-managed search; level-30/31 check): {payload['comparison']['time_managed_matches_level_33']}; {payload['comparison']['level_30_then_31_matches_level_33']}
- Time-managed-search wall time: {time_elapsed:.3f} seconds
- Level-30/31-check wall time: {level_elapsed:.3f} seconds
- Additional level-33-check wall time for different-move positions: {deep_elapsed:.3f} seconds

This comparison does not modify the tournament table. A separate color-swapped two-game match from each same starting position decides whether a table may be adopted.
"""
    _atomic_write_text(output_dir / "README.md", report)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=100)
    args = parser.parse_args()
    payload = compare_methods(
        args.reference_results,
        args.exe.resolve(),
        args.output_dir,
        args.positions,
    )
    print(
        f"positions={payload['positions']['requested']} "
        f"same_move={payload['comparison']['same_move']} "
        f"different_move={payload['comparison']['different_move']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
