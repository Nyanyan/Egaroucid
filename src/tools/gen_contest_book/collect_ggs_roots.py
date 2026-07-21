"""Collect canonical s8r14 start roots from GGS logs and report book coverage.

The collector is deliberately strict: a ``match start!`` marker must be
followed by the first pending-search board, and that board must have the
declared root-disc count.  This prevents a partial log or a midgame board
from silently becoming teacher input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from build_book import canonicalize_board_key
from build_root_table import (
    ROOT_TABLE_DEFAULT_N_DISCS,
    ROOT_TABLE_FILENAME,
    _iter_book_files,
    load_root_from_book,
    load_root_table_entries,
)
from config import TRAINED_DIR
from othello import Board, normalize_board_text


REPORT_SCHEMA = "ggs_start_root_coverage_v1"
MATCH_START_MARKER = "GGS INFO> match start!"
MATCH_HEADER_RE = re.compile(
    r"^GGS RECV> /os: -\s+(?P<match_id>\S+)\s+.*\bs8r14\b"
)
PENDING_ROOT_RE = re.compile(
    r"^GGS INFO> ggs pending search wait (?P<game_id>\S+) max \d+ "
    r"(?P<cells>[XO-]{64}) (?P<side>[XO])$"
)


@dataclass(frozen=True)
class StartOccurrence:
    source: Path
    line: int
    match_id: str
    game_id: str
    raw_board: str
    canonical_board: str
    symmetry: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ggs_start_roots(path: Path, root_discs: int) -> list[StartOccurrence]:
    """Extract one validated root occurrence for every s8r14 match start."""
    if not 4 <= root_discs <= 64:
        raise ValueError("root_discs must be in [4, 64]")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read GGS log {path}: {error}") from error

    latest_match_id: str | None = None
    waiting_at: tuple[int, str | None] | None = None
    occurrences: list[StartOccurrence] = []
    for line_number, line in enumerate(lines, start=1):
        header = MATCH_HEADER_RE.match(line)
        if header is not None and header.group("match_id") != "match":
            latest_match_id = header.group("match_id")
        if line == MATCH_START_MARKER:
            if waiting_at is not None:
                raise ValueError(
                    f"{path}:{line_number}: new match start before root board for line {waiting_at[0]}"
                )
            waiting_at = (line_number, latest_match_id)
            continue
        if waiting_at is None:
            continue
        root = PENDING_ROOT_RE.match(line)
        if root is None:
            continue
        game_id = root.group("game_id")
        inferred_match_id = game_id.rsplit(".", 1)[0]
        # GGS may assign a synchronised match header a different identifier
        # from either individual game, so retain both instead of conflating
        # them or rejecting a valid pair.
        match_id = waiting_at[1] or inferred_match_id
        raw_board = normalize_board_text(f"{root.group('cells')} {root.group('side')}")
        if Board.from_text(raw_board).n_discs() != root_discs:
            raise ValueError(
                f"{path}:{line_number}: root has {Board.from_text(raw_board).n_discs()} discs, "
                f"expected {root_discs}"
            )
        canonical_board, symmetry = canonicalize_board_key(raw_board)
        occurrences.append(
            StartOccurrence(
                source=path.resolve(),
                line=line_number,
                match_id=match_id,
                game_id=game_id,
                raw_board=raw_board,
                canonical_board=canonical_board,
                symmetry=symmetry,
            )
        )
        waiting_at = None
    if waiting_at is not None:
        raise ValueError(f"{path}: missing root board after match start at line {waiting_at[0]}")
    if not occurrences:
        raise ValueError(f"{path}: no s8r14 start roots found")
    return occurrences


def _deep_book_roots(book_dirs: Iterable[Path], root_discs: int) -> set[str]:
    return {
        load_root_from_book(path, root_discs).board
        for path in _iter_book_files(book_dirs)
    }


def collect_coverage(
    log_paths: Iterable[Path],
    book_dirs: Iterable[Path],
    root_table: Path | None,
    root_discs: int = ROOT_TABLE_DEFAULT_N_DISCS,
) -> dict[str, object]:
    resolved_logs = [path.resolve() for path in log_paths]
    if not resolved_logs:
        raise ValueError("at least one GGS log is required")
    if len(set(resolved_logs)) != len(resolved_logs):
        raise ValueError("the same GGS log was supplied more than once")
    resolved_book_dirs = tuple(path.resolve() for path in book_dirs)

    occurrences = [
        occurrence
        for path in resolved_logs
        for occurrence in parse_ggs_start_roots(path, root_discs)
    ]
    deep_roots = _deep_book_roots(resolved_book_dirs, root_discs)
    table_roots: set[str] = set()
    root_table_metadata: dict[str, object] = {"present": False}
    if root_table is not None and root_table.is_file():
        table_discs, table_entries = load_root_table_entries(root_table, root_discs)
        table_roots = set(table_entries)
        root_table_metadata = {
            "present": True,
            "path": root_table.resolve().as_posix(),
            "sha256": sha256_file(root_table),
            "root_discs": table_discs,
            "entries": len(table_roots),
        }

    grouped: dict[str, list[StartOccurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence.canonical_board, []).append(occurrence)
    roots = []
    for canonical_board in sorted(grouped):
        root_occurrences = grouped[canonical_board]
        roots.append(
            {
                "canonical_board": canonical_board,
                "observed": len(root_occurrences),
                "deep_book": canonical_board in deep_roots,
                "root_table": canonical_board in table_roots,
                "occurrences": [
                    {
                        "source": occurrence.source.as_posix(),
                        "line": occurrence.line,
                        "match_id": occurrence.match_id,
                        "game_id": occurrence.game_id,
                        "raw_board": occurrence.raw_board,
                        "symmetry": occurrence.symmetry,
                    }
                    for occurrence in root_occurrences
                ],
            }
        )

    unique_roots = set(grouped)
    covered_by_deep = unique_roots & deep_roots
    covered_by_table = unique_roots & table_roots
    covered_by_any = covered_by_deep | covered_by_table
    return {
        "schema": REPORT_SCHEMA,
        "root_discs": root_discs,
        "logs": [
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in resolved_logs
        ],
        "book_directories": [path.as_posix() for path in resolved_book_dirs],
        "deep_book_entries": len(deep_roots),
        "root_table": root_table_metadata,
        "observed_start_events": len(occurrences),
        "unique_canonical_roots": len(unique_roots),
        "coverage": {
            "deep_book": len(covered_by_deep),
            "root_table": len(covered_by_table),
            "either": len(covered_by_any),
            "uncovered": len(unique_roots - covered_by_any),
        },
        "roots": roots,
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
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--books-dir", type=Path, action="append", default=[])
    parser.add_argument("--root-table", type=Path)
    parser.add_argument("--root-discs", type=int, default=ROOT_TABLE_DEFAULT_N_DISCS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    book_dirs = args.books_dir or [TRAINED_DIR]
    root_table = args.root_table
    if root_table is None:
        root_table = book_dirs[0] / ROOT_TABLE_FILENAME
    report = collect_coverage(args.log, book_dirs, root_table, args.root_discs)
    write_report(args.output, report)
    coverage = report["coverage"]
    print(
        f"wrote {report['unique_canonical_roots']} canonical roots from "
        f"{report['observed_start_events']} start event(s): "
        f"deep={coverage['deep_book']} table={coverage['root_table']} "
        f"uncovered={coverage['uncovered']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
