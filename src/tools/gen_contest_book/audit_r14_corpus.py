"""Audit a complete r14 start corpus before expanding a contest root table.

The r14 setup files can contain duplicates and different D4 orientations of
the same playable position.  This tool preserves a hash of every input file,
normalises every start in the same way as the runtime root-table lookup, and
records the exact canonical population against which book coverage is
measured.  It intentionally does *not* create teacher moves: quality-gated
teacher generation consumes a frozen report produced here in a later step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Iterable

from build_book import CPP_REPRESENTATIVE_ORDER, _cpp_coord_to_representative
from build_root_table import (
    ROOT_TABLE_DEFAULT_N_DISCS,
    ROOT_TABLE_FILENAME,
    _iter_book_files,
    load_root_from_book,
    load_root_table_entries,
)
from config import START_DIR, TRAINED_DIR
from othello import Board, normalize_board_text


CORPUS_REPORT_SCHEMA = "r14_corpus_coverage_v1"


def _source_indices_for_cpp_symmetry(symmetry: int) -> tuple[int, ...]:
    """Return source string positions in destination-string order.

    ``canonicalize_board_key`` operates bit by bit, which is ideal for book
    generation but unnecessarily expensive for a one-time 111,534-position
    corpus audit.  These permutations reproduce the same C++ bitboard
    representative mapping while allowing each transformed board to be built
    as one string join.
    """
    source_for_destination = [0] * 64
    for source in range(64):
        source_cpp_cell = 63 - source
        destination = 63 - _cpp_coord_to_representative(source_cpp_cell, symmetry)
        source_for_destination[destination] = source
    return tuple(source_for_destination)


CPP_SOURCE_INDICES = tuple(
    _source_indices_for_cpp_symmetry(symmetry)
    for symmetry in CPP_REPRESENTATIVE_ORDER
)
PLAYER_BIT_KEY = str.maketrans("-OX", "001")
OPPONENT_BIT_KEY = str.maketrans("-OX", "010")


def canonicalize_relative_key(relative_key: str) -> str:
    """Return the exact runtime representative for a relative ``<cells> X`` key."""
    if len(relative_key) != 66 or relative_key[64:] != " X":
        raise ValueError("expected a normalized relative board key")
    cells = relative_key[:64]
    candidates = ["".join(cells[index] for index in indices) for indices in CPP_SOURCE_INDICES]
    # representative_board() compares the whole player bitboard before the
    # opponent bitboard.  It is not the same ordering as printable cell text
    # (which would interleave player and opponent bits at each square).
    return min(
        candidates,
        key=lambda candidate: (
            candidate.translate(PLAYER_BIT_KEY),
            candidate.translate(OPPONENT_BIT_KEY),
        ),
    ) + " X"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _start_files(start_dir: Path) -> list[Path]:
    if not start_dir.is_dir():
        raise ValueError(f"start directory does not exist: {start_dir}")
    paths = sorted((path.resolve() for path in start_dir.glob("*.txt")), key=lambda path: path.as_posix())
    if not paths:
        raise ValueError(f"{start_dir}: no .txt start files")
    return paths


def _deep_book_roots(book_dirs: Iterable[Path], root_discs: int) -> set[str]:
    return {
        load_root_from_book(path, root_discs).board
        for path in _iter_book_files(book_dirs)
    }


def audit_corpus(
    start_dir: Path,
    book_dirs: Iterable[Path] = (),
    root_table: Path | None = None,
    root_discs: int = ROOT_TABLE_DEFAULT_N_DISCS,
) -> dict[str, object]:
    """Return a reproducible canonical-population and coverage report."""
    if not 4 <= root_discs <= 64:
        raise ValueError("root_discs must be in [4, 64]")
    resolved_start_dir = start_dir.resolve()
    start_files = _start_files(resolved_start_dir)
    resolved_book_dirs = tuple(path.resolve() for path in book_dirs)

    raw_rows: list[str] = []
    canonical_rows: list[str] = []
    source_row_counts: dict[Path, int] = {}
    for path in start_files:
        count = 0
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise ValueError(f"cannot read start file {path}: {error}") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw_board = normalize_board_text(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid board: {error}") from error
            position = Board.from_text(raw_board)
            if position.n_discs() != root_discs:
                raise ValueError(
                    f"{path}:{line_number}: start has {position.n_discs()} discs, expected {root_discs}"
                )
            if not position.legal_moves():
                raise ValueError(f"{path}:{line_number}: start has no legal move")
            canonical_board = canonicalize_relative_key(position.key())
            raw_rows.append(raw_board)
            canonical_rows.append(canonical_board)
            count += 1
        if count == 0:
            raise ValueError(f"{path}: contains no start boards")
        source_row_counts[path] = count

    canonical_roots = sorted(set(canonical_rows))
    raw_unique = set(raw_rows)
    aliases_by_root = Counter(canonical_rows)
    deep_roots = _deep_book_roots(resolved_book_dirs, root_discs)
    table_roots: set[str] = set()
    root_table_metadata: dict[str, object] = {"present": False}
    if root_table is not None and root_table.is_file():
        resolved_table = root_table.resolve()
        table_discs, table_entries = load_root_table_entries(resolved_table, root_discs)
        table_roots = set(table_entries)
        root_table_metadata = {
            "present": True,
            "path": resolved_table.as_posix(),
            "sha256": sha256_file(resolved_table),
            "root_discs": table_discs,
            "entries": len(table_roots),
        }

    canonical_set = set(canonical_roots)
    covered_by_deep = canonical_set & deep_roots
    covered_by_table = canonical_set & table_roots
    covered_by_any = covered_by_deep | covered_by_table
    return {
        "schema": CORPUS_REPORT_SCHEMA,
        "root_discs": root_discs,
        "start_directory": resolved_start_dir.as_posix(),
        "source_files": [
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": source_row_counts[path],
            }
            for path in start_files
        ],
        "book_directories": [path.as_posix() for path in resolved_book_dirs],
        "deep_book_entries": len(deep_roots),
        "root_table": root_table_metadata,
        "raw_start_rows": len(raw_rows),
        "unique_normalized_starts": len(raw_unique),
        "unique_canonical_roots": len(canonical_roots),
        "duplicate_normalized_rows": len(raw_rows) - len(raw_unique),
        "d4_alias_rows": len(raw_rows) - len(canonical_roots),
        "max_rows_for_one_canonical_root": max(aliases_by_root.values()),
        "canonical_roots_sha256": _digest_lines(canonical_roots),
        "coverage": {
            "deep_book": len(covered_by_deep),
            "root_table": len(covered_by_table),
            "either": len(covered_by_any),
            "uncovered": len(canonical_set - covered_by_any),
        },
        # The sorted list is deliberately retained: later cohort selection can
        # freeze its input by this report hash instead of re-enumerating data.
        "roots": [
            {
                "canonical_board": board,
                "source_rows": aliases_by_root[board],
                "deep_book": board in deep_roots,
                "root_table": board in table_roots,
            }
            for board in canonical_roots
        ],
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-dir", type=Path, default=START_DIR)
    parser.add_argument("--books-dir", type=Path, action="append", default=[])
    parser.add_argument("--root-table", type=Path)
    parser.add_argument("--root-discs", type=int, default=ROOT_TABLE_DEFAULT_N_DISCS)
    parser.add_argument("--expected-canonical-roots", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    book_dirs = args.books_dir or [TRAINED_DIR]
    root_table = args.root_table
    if root_table is None:
        root_table = book_dirs[0] / ROOT_TABLE_FILENAME
    report = audit_corpus(args.start_dir, book_dirs, root_table, args.root_discs)
    if (
        args.expected_canonical_roots is not None
        and report["unique_canonical_roots"] != args.expected_canonical_roots
    ):
        raise ValueError(
            "canonical-root count mismatch: "
            f"expected {args.expected_canonical_roots}, got {report['unique_canonical_roots']}"
        )
    write_report(args.output, report)
    coverage = report["coverage"]
    print(
        f"wrote {report['unique_canonical_roots']} canonical roots from "
        f"{report['raw_start_rows']} start rows: "
        f"deep={coverage['deep_book']} table={coverage['root_table']} "
        f"uncovered={coverage['uncovered']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
