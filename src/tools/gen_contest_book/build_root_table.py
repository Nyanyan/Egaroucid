"""Build a verified, one-move contest-root table from individual books.

Deep ``.egcb`` files remain the source of truth for important starts.  This
tool extracts only their root row into one canonical, symmetry-safe table that
the engine can use for every r14 start not covered by a deep file.  A later
teacher pipeline can write shallow source books for all enumerated starts and
use the same builder without changing the runtime format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from build_book import canonicalize_board_key, move_to_representative
from config import TRAINED_DIR, iter_start_boards
from othello import Board, coord_to_index, index_to_coord, normalize_board_text


ROOT_TABLE_FILENAME = "contest_root_table.egcb"
ROOT_TABLE_MANIFEST_SUFFIX = ".manifest.json"
ROOT_TABLE_FORMAT = "# contest_root_table_v1"
ROOT_TABLE_DEFAULT_N_DISCS = 14


@dataclass(frozen=True)
class RootEntry:
    board: str
    value: int
    moves: tuple[tuple[int, int], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path_for_root_table(path: Path) -> Path:
    return path.with_suffix(path.suffix + ROOT_TABLE_MANIFEST_SUFFIX)


def _parse_int(text: str, label: str) -> int:
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"invalid {label}: {text!r}") from error


def _parse_root_row(parts: list[str], source: Path, line_number: int) -> RootEntry:
    if len(parts) < 4:
        raise ValueError(f"{source}:{line_number}: root row has fewer than four fields")
    raw_board = normalize_board_text(parts[0] + " " + parts[1])
    root_board = Board.from_text(raw_board)
    canonical_board, symmetry = canonicalize_board_key(raw_board)
    canonical_position = Board.from_text(canonical_board)
    legal = set(canonical_position.legal_moves())
    value = _parse_int(parts[2], f"root value at {source}:{line_number}")
    moves: dict[int, int] = {}
    for token in parts[3:]:
        if token.count(":") != 1:
            raise ValueError(f"{source}:{line_number}: invalid move score {token!r}")
        coordinate, score_text = token.split(":", 1)
        try:
            source_move = coord_to_index(coordinate)
        except ValueError as error:
            raise ValueError(
                f"{source}:{line_number}: invalid move coordinate {coordinate!r}"
            ) from error
        if source_move not in root_board.legal_moves():
            raise ValueError(f"{source}:{line_number}: illegal source move {coordinate}")
        canonical_move = move_to_representative(source_move, symmetry)
        if canonical_move not in legal:
            raise ValueError(
                f"{source}:{line_number}: canonicalized move {index_to_coord(canonical_move)} is illegal"
            )
        score = _parse_int(score_text, f"move score at {source}:{line_number}")
        previous = moves.get(canonical_move)
        if previous is not None and previous != score:
            raise ValueError(
                f"{source}:{line_number}: conflicting scores for {index_to_coord(canonical_move)}"
            )
        moves[canonical_move] = score
    if not moves:
        raise ValueError(f"{source}:{line_number}: root row has no moves")
    return RootEntry(
        board=canonical_board,
        value=value,
        moves=tuple(sorted(moves.items(), key=lambda item: (-item[1], item[0]))),
    )


def load_root_from_book(path: Path, root_discs: int) -> RootEntry:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    initial: str | None = None
    for line in lines:
        if line.startswith("# initial "):
            initial = normalize_board_text(line[len("# initial "):])
            break
    if initial is None:
        raise ValueError(f"{path}: missing # initial header")
    if Board.from_text(initial).n_discs() != root_discs:
        raise ValueError(
            f"{path}: initial board has {Board.from_text(initial).n_discs()} discs, expected {root_discs}"
        )
    canonical_initial, _ = canonicalize_board_key(initial)
    matching_rows: list[RootEntry] = []
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        entry = _parse_root_row(line.split(), path, line_number)
        if entry.board == canonical_initial:
            matching_rows.append(entry)
    if len(matching_rows) != 1:
        raise ValueError(
            f"{path}: expected exactly one canonical root row for {canonical_initial}, found {len(matching_rows)}"
        )
    return matching_rows[0]


def _iter_book_files(book_dirs: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for directory in book_dirs:
        if not directory.is_dir():
            raise ValueError(f"book directory does not exist: {directory}")
        for path in directory.glob("*.egcb"):
            if path.name != ROOT_TABLE_FILENAME:
                files.add(path.resolve())
    return sorted(files, key=lambda path: path.as_posix())


def load_root_rows(path: Path, root_discs: int) -> list[RootEntry]:
    """Read canonicalizable teacher root rows from a compact text artifact.

    Data rows use the same ``<board> <side> <value> <move>:<score> ...``
    shape as an ``.egcb`` row.  Comment lines are permitted, so a root table
    can also be revalidated/rebuilt directly from its own published output.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read root-result file {path}: {error}") from error
    rows: list[RootEntry] = []
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        entry = _parse_root_row(line.split(), path, line_number)
        if Board.from_text(entry.board).n_discs() != root_discs:
            raise ValueError(
                f"{path}:{line_number}: row has {Board.from_text(entry.board).n_discs()} discs, "
                f"expected {root_discs}"
            )
        rows.append(entry)
    if not rows:
        raise ValueError(f"{path}: no root-result rows")
    return rows


def _entry_line(entry: RootEntry) -> str:
    moves = " ".join(f"{index_to_coord(move)}:{score}" for move, score in entry.moves)
    return f"{entry.board} {entry.value} {moves}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_root_table_entries(
    path: Path, expected_root_discs: int | None = None
) -> tuple[int, dict[str, RootEntry]]:
    """Load every validated canonical entry from a published root table."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read root table {path}: {error}") from error
    if len(lines) < 3 or lines[0] != ROOT_TABLE_FORMAT:
        raise ValueError(f"{path}: invalid root-table format header")
    if not lines[1].startswith("# root_discs ") or not lines[2].startswith("# entries "):
        raise ValueError(f"{path}: missing root-table headers")
    root_discs = _parse_int(lines[1][len("# root_discs "):], "root_discs")
    expected_entries = _parse_int(lines[2][len("# entries "):], "entries")
    if root_discs < 4 or root_discs > 64 or expected_entries <= 0:
        raise ValueError(f"{path}: invalid root-table header values")
    if expected_root_discs is not None and root_discs != expected_root_discs:
        raise ValueError(f"{path}: root_discs {root_discs} != expected {expected_root_discs}")
    entries: dict[str, RootEntry] = {}
    for line_number, line in enumerate(lines[3:], start=4):
        if not line:
            raise ValueError(f"{path}:{line_number}: blank data row")
        entry = _parse_root_row(line.split(), path, line_number)
        if Board.from_text(entry.board).n_discs() != root_discs:
            raise ValueError(f"{path}:{line_number}: row is not a root-disc position")
        if entry.board in entries:
            raise ValueError(f"{path}:{line_number}: duplicate canonical root {entry.board}")
        entries[entry.board] = entry
    if len(entries) != expected_entries:
        raise ValueError(f"{path}: expected {expected_entries} rows, found {len(entries)}")
    return root_discs, entries


def validate_root_table(path: Path, expected_root_discs: int | None = None) -> dict[str, int]:
    root_discs, entries = load_root_table_entries(path, expected_root_discs)
    return {"root_discs": root_discs, "entries": len(entries)}


def build_root_table(
    book_dirs: Iterable[Path],
    output: Path,
    root_discs: int = ROOT_TABLE_DEFAULT_N_DISCS,
    required_starts: Iterable[str] | None = None,
    root_result_files: Iterable[Path] = (),
) -> dict[str, int | str]:
    if not 4 <= root_discs <= 64:
        raise ValueError("root_discs must be in [4, 64]")
    book_paths = _iter_book_files(book_dirs)
    result_paths = sorted({path.resolve() for path in root_result_files}, key=lambda path: path.as_posix())
    if not book_paths and not result_paths:
        raise ValueError("no source .egcb books or root-result files found")
    entries: dict[str, RootEntry] = {}
    source_by_root: dict[str, Path] = {}
    source_entries: list[tuple[Path, str, RootEntry]] = []
    for path in book_paths:
        source_entries.append((path, "book", load_root_from_book(path, root_discs)))
    for path in result_paths:
        source_entries.extend((path, "root_result", entry) for entry in load_root_rows(path, root_discs))
    for path, _source_kind, entry in source_entries:
        previous = entries.get(entry.board)
        if previous is not None:
            if previous != entry:
                raise ValueError(
                    f"conflicting verified roots for {entry.board}: {source_by_root[entry.board]} and {path}"
                )
            continue
        entries[entry.board] = entry
        source_by_root[entry.board] = path
    if required_starts is not None:
        expected = {canonicalize_board_key(normalize_board_text(board))[0] for board in required_starts}
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        if missing or unexpected:
            raise ValueError(
                f"root-table coverage mismatch: missing={len(missing)} unexpected={len(unexpected)}"
            )
    ordered = [entries[key] for key in sorted(entries)]
    text = "\n".join(
        [
            ROOT_TABLE_FORMAT,
            f"# root_discs {root_discs}",
            f"# entries {len(ordered)}",
            *(_entry_line(entry) for entry in ordered),
            "",
        ]
    )
    _atomic_write_text(output, text)
    validation = validate_root_table(output, root_discs)
    sources = []
    for path, source_kind in [
        *((path, "book") for path in book_paths),
        *((path, "root_result") for path in result_paths),
    ]:
        source = {
            "path": path.as_posix(),
            "kind": source_kind,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        source_manifest = path.with_suffix(path.suffix + ".manifest.json")
        if source_manifest.is_file():
            source["manifest"] = {
                "path": source_manifest.as_posix(),
                "sha256": sha256_file(source_manifest),
            }
        sources.append(source)
    manifest = {
        "schema": "contest_root_table_manifest_v1",
        "root_discs": root_discs,
        "entries": validation["entries"],
        "output": {
            "path": output.resolve().as_posix(),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "sources": sources,
        "source_count": len(sources),
        "coverage_required": required_starts is not None,
    }
    _atomic_write_text(
        manifest_path_for_root_table(output),
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "entries": validation["entries"],
        "root_discs": root_discs,
        "sources": len(sources),
        "output_sha256": manifest["output"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--books-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing verified per-start .egcb files (repeatable)",
    )
    parser.add_argument(
        "--root-results",
        type=Path,
        action="append",
        default=[],
        help="Compact verified teacher root rows (repeatable)",
    )
    parser.add_argument("--output", type=Path, default=TRAINED_DIR / ROOT_TABLE_FILENAME)
    parser.add_argument("--root-discs", type=int, default=ROOT_TABLE_DEFAULT_N_DISCS)
    parser.add_argument(
        "--require-starts-dir",
        type=Path,
        help="Require exact D4-canonical coverage of every start in this directory",
    )
    args = parser.parse_args()
    publication_target = (TRAINED_DIR / ROOT_TABLE_FILENAME).resolve()
    if args.output.resolve() == publication_target:
        parser.error(
            "writing trained/contest_root_table.egcb directly is disabled; "
            "use publish_verified_root_table.py after a passed match audit"
        )
    if not args.books_dir and not args.root_results:
        args.books_dir = [TRAINED_DIR]
    required_starts = (
        list(iter_start_boards(args.require_starts_dir))
        if args.require_starts_dir is not None
        else None
    )
    result = build_root_table(
        args.books_dir,
        args.output,
        args.root_discs,
        required_starts,
        args.root_results,
    )
    print(
        f"wrote {result['entries']} verified roots at {result['root_discs']} discs "
        f"from {result['sources']} source artifact(s) to {args.output} sha256={result['output_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
