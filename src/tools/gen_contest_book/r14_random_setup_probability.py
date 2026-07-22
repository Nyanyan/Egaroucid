"""Occurrence probabilities under the local ``random_setup(14)`` reimplementation.

The input directory ``records321_14_random_setup`` contains one spatial
representative for each position produced by the repository-local ``random_setup(14)``
generator reimplementation.  This module restores the probability mass which
was lost when that generator enumerated only one representative for each of
the eight rotations and reflections.  It does not identify the current GGS
server source.

For 14 discs, the local generator reimplementation always occupies the central
four squares and chooses ten of the next twelve squares.  It chooses the number of white discs
from 5 through 9 uniformly, then chooses their locations uniformly.  Thus an
oriented board with ``k`` white discs has probability

    1 / (C(12, 10) * 5 * C(14, k)).

The stored representative stands for every distinct spatial orientation of
that board, so its probability is that value multiplied by its orbit size.
All arithmetic uses :class:`fractions.Fraction`; no floating-point ordering is
used to decide which positions are calculated first.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import uuid
from collections import Counter
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, Iterable

from build_book import canonicalize_board_key, transform_board_text
from othello import normalize_board_text


R14_RANDOM_SETUP_PROBABILITY_SCHEMA = "ggs_random_setup_14_probability_v2"
R14_RANDOM_SETUP_PRIORITY_ROW_SCHEMA = "ggs_random_setup_14_priority_row_v1"
R14_RANDOM_SETUP_PRIORITY_MANIFEST_SCHEMA = "ggs_random_setup_14_priority_manifest_v3"
R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCHEMA = (
    "ggs_random_setup_14_local_probability_model_sources_v1"
)
R14_DISCS = 14
R14_CENTRAL_CELL_COUNT = 4
R14_NEXT_RING_CELL_COUNT = 12
R14_NEXT_RING_SELECTED_COUNT = 10
R14_SILHOUETTE_COUNT = comb(R14_NEXT_RING_CELL_COUNT, R14_NEXT_RING_SELECTED_COUNT)
R14_WHITE_COUNTS = tuple(range(5, 10))
R14_WHITE_COUNT_CHOICES = len(R14_WHITE_COUNTS)
# These three files have separate jobs in the derivation.  Keeping the jobs in
# the immutable metadata prevents a later report from implying that the
# enumerator alone establishes the sampling distribution.
R14_LOCAL_PROBABILITY_MODEL_SOURCE_FILES: tuple[dict[str, str], ...] = (
    {
        "id": "enumerated_population",
        "relative_path": "src/tools/enumerate_ggs_random_boards/random_setup.cpp",
        "role": (
            "enumerates the random_setup(14) support and reduces rotations/reflections "
            "to one spatial representative"
        ),
        "limitation": "does not itself establish how often a supported board is sampled",
    },
    {
        "id": "sampling_steps",
        "relative_path": "src/tools/enumerate_ggs_random_boards/random_setup_2_random.cpp",
        "role": (
            "contains the repository-local cell-selection, white-count, and "
            "color-allocation sampling steps used to derive the r14 mass"
        ),
        "limitation": (
            "is an alternate-generator tool and does not directly execute random_setup(14)"
        ),
    },
    {
        "id": "repository_description",
        "relative_path": "src/tools/enumerate_ggs_random_boards/README.md",
        "role": (
            "documents this repository's claimed correspondence and the boundary "
            "between the primary and alternate constructions"
        ),
        "limitation": "does not authenticate the source or version currently deployed by GGS",
    },
)
R14_LOCAL_PROBABILITY_MODEL_FORMULA_FACTS: tuple[dict[str, object], ...] = (
    {
        "quantity": "C(12,10)",
        "source_ids": ["enumerated_population", "sampling_steps"],
        "role": "ten selectable cells chosen from the twelve cells in the next ring",
    },
    {
        "quantity": "5",
        "source_ids": ["enumerated_population", "sampling_steps"],
        "role": "the five white-disc counts from 5 through 9",
    },
    {
        "quantity": "C(14,k)",
        "source_ids": ["sampling_steps"],
        "role": "uniform allocation of k white discs among fourteen occupied cells",
    },
    {
        "quantity": "orbit_size",
        "source_ids": ["enumerated_population"],
        "role": "restore the rotations/reflections collapsed by enumeration",
    },
)
R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCOPE = (
    "repository-local evidence for this probability derivation; it does not identify "
    "or authenticate the current GGS server source"
)

# This literal is recorded by teacher generation when probability ordering is
# selected.  Keeping it as structured data makes a later audit independent of
# prose wording and of a floating-point implementation.
R14_RANDOM_SETUP_PROBABILITY_MODEL: dict[str, object] = {
    "schema": R14_RANDOM_SETUP_PROBABILITY_SCHEMA,
    "n_discs": R14_DISCS,
    "central_cell_count": R14_CENTRAL_CELL_COUNT,
    "next_ring_cell_count": R14_NEXT_RING_CELL_COUNT,
    "next_ring_selected_count": R14_NEXT_RING_SELECTED_COUNT,
    "silhouette_count": R14_SILHOUETTE_COUNT,
    "white_disc_counts": list(R14_WHITE_COUNTS),
    "white_count_choices": R14_WHITE_COUNT_CHOICES,
    "oriented_board_probability": "1 / (C(12,10) * 5 * C(14,k))",
    "representative_probability": "orbit_size / (C(12,10) * 5 * C(14,k))",
    "spatial_transformations": 8,
    "spatial_identity": (
        "the eight board rotations/reflections; disc colors and side to move are preserved"
    ),
    "source_scope": (
        "exact arithmetic under the repository-local random_setup(14) reimplementation; "
        "not a claim that this is the current GGS server source"
    ),
    "source_files_schema": R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCHEMA,
}


def _ring_distance(index: int) -> int:
    """Match the distance formula used by ``random_setup.cpp`` exactly."""
    x = index % 8
    y = index // 8
    return (max(abs(2 * x - 7), abs(2 * y - 7)) - 1) // 2


@functools.lru_cache(maxsize=262_144)
def validate_r14_random_setup_board(board: str) -> str:
    """Normalize and validate one position generated by ``random_setup(14)``.

    Validation is deliberately strict: applying this model to a board from a
    different GGS random-start algorithm would silently produce a misleading
    priority order.
    """
    if not isinstance(board, str):
        raise ValueError("random_setup(14) board must be text")
    if len(board) != 66 or board[64] != " " or board[65] not in "XO":
        raise ValueError("random_setup(14) board must use 64 cells, one space, and side X or O")
    if any(cell not in "-XO" for cell in board[:64]):
        raise ValueError("random_setup(14) board has an invalid cell character")
    normalized = normalize_board_text(board)
    if board != normalized:
        raise ValueError("random_setup(14) board is not in its required normalized text form")
    cells = normalized[:64]
    side = normalized[65]
    if side != "X":
        raise ValueError("random_setup(14) must have X to move")
    if sum(cell != "-" for cell in cells) != R14_DISCS:
        raise ValueError("random_setup(14) must contain exactly 14 discs")
    white_discs = cells.count("O")
    if white_discs not in R14_WHITE_COUNTS:
        raise ValueError(
            "random_setup(14) must contain one of "
            f"{R14_WHITE_COUNTS} white-disc counts, got {white_discs}"
        )

    central = 0
    next_ring = 0
    for index, cell in enumerate(cells):
        distance = _ring_distance(index)
        if distance == 0:
            if cell == "-":
                raise ValueError("random_setup(14) is missing a central disc")
            central += 1
        elif distance == 1:
            if cell != "-":
                next_ring += 1
        elif cell != "-":
            raise ValueError("random_setup(14) has a disc outside its selectable ring")
    if central != R14_CENTRAL_CELL_COUNT or next_ring != R14_NEXT_RING_SELECTED_COUNT:
        raise ValueError("random_setup(14) has an unexpected occupied-cell silhouette")
    return normalized


@functools.lru_cache(maxsize=262_144)
def r14_random_setup_orbit_size(board: str) -> int:
    """Count distinct rotations/reflections, preserving colors and side to move."""
    normalized = validate_r14_random_setup_board(board)
    return len({transform_board_text(normalized, symmetry) for symmetry in range(8)})


@functools.lru_cache(maxsize=262_144)
def r14_random_setup_probability(board: str) -> Fraction:
    """Return an exact mass under the local random_setup(14) reimplementation."""
    normalized = validate_r14_random_setup_board(board)
    white_discs = normalized[:64].count("O")
    return Fraction(
        r14_random_setup_orbit_size(normalized),
        R14_SILHOUETTE_COUNT * R14_WHITE_COUNT_CHOICES * comb(R14_DISCS, white_discs),
    )


def _tie_break_key(board: str, tie_seed: int | None) -> bytes | str:
    if tie_seed is None:
        return board
    return hashlib.sha256(f"{tie_seed}\0{board}".encode("ascii")).digest()


def order_r14_random_setup_boards(
    boards: Iterable[str], tie_seed: int | None = None
) -> list[str]:
    """Order unique r14 starts by decreasing local-reimplementation probability.

    A supplied ``tie_seed`` affects only positions having exactly the same
    probability.  It therefore diversifies a bounded initial calculation
    without changing the priority between positions of different probability.
    """
    normalized = sorted({validate_r14_random_setup_board(board) for board in boards})
    return sorted(
        normalized,
        key=lambda board: (
            -r14_random_setup_probability(board),
            _tie_break_key(board, tie_seed),
            board,
        ),
    )


def fraction_json(value: Fraction) -> dict[str, int]:
    """Serialize an exact rational value without rounding it."""
    return {"numerator": value.numerator, "denominator": value.denominator}


def ordered_boards_sha256(boards: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for board in boards:
        digest.update(board.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def priority_manifest_metadata_path(path: Path) -> Path:
    """Return the immutable sidecar metadata path for one priority file."""
    return path.with_suffix(path.suffix + ".meta.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_probability_model_source_fingerprints() -> dict[str, object]:
    """Fingerprint every local file used to derive the probability model.

    This records repository-local evidence only.  It intentionally does not
    assert that these files are the source currently deployed by GGS.
    """
    repository_root = Path(__file__).resolve().parents[3]
    files: list[dict[str, int | str]] = []
    for descriptor in R14_LOCAL_PROBABILITY_MODEL_SOURCE_FILES:
        source = repository_root / descriptor["relative_path"]
        if not source.is_file():
            raise FileNotFoundError(f"local probability-model source not found: {source}")
        files.append(
            {
                **descriptor,
                "sha256": _sha256_file(source),
                "bytes": source.stat().st_size,
            }
        )
    return {
        "schema": R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCHEMA,
        "scope": R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCOPE,
        "formula_facts": [dict(fact) for fact in R14_LOCAL_PROBABILITY_MODEL_FORMULA_FACTS],
        "files": files,
    }


def _validated_local_probability_model_source_fingerprints(
    value: object,
    *,
    require_current: bool,
) -> dict[str, object]:
    """Validate the named local source files and, when needed, their hashes."""
    if not isinstance(value, dict):
        raise ValueError("priority input probability-model sources are not an object")
    if value.get("schema") != R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCHEMA:
        raise ValueError("priority input probability-model sources have an unsupported schema")
    if value.get("scope") != R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCOPE:
        raise ValueError("priority input probability-model sources have an unexpected scope")
    expected_facts = [dict(fact) for fact in R14_LOCAL_PROBABILITY_MODEL_FORMULA_FACTS]
    if value.get("formula_facts") != expected_facts:
        raise ValueError("priority input probability-model sources have unexpected formula facts")
    records = value.get("files")
    if not isinstance(records, list) or len(records) != len(R14_LOCAL_PROBABILITY_MODEL_SOURCE_FILES):
        raise ValueError("priority input probability-model sources have unexpected files")
    normalized_records: list[dict[str, int | str]] = []
    for index, (record, descriptor) in enumerate(
        zip(records, R14_LOCAL_PROBABILITY_MODEL_SOURCE_FILES, strict=True)
    ):
        if not isinstance(record, dict) or set(record) != {
            "id", "relative_path", "role", "limitation", "sha256", "bytes"
        }:
            raise ValueError(
                f"priority input probability-model source {index} has unexpected fields"
            )
        if (
            record.get("id") != descriptor["id"]
            or record.get("relative_path") != descriptor["relative_path"]
            or record.get("role") != descriptor["role"]
            or record.get("limitation") != descriptor["limitation"]
        ):
            raise ValueError(
                f"priority input probability-model source {index} has unexpected identity"
            )
        fingerprint = _validate_fingerprint_record(
            record,
            f"priority input probability-model source {index}",
            path_key="relative_path",
        )
        normalized_records.append({**descriptor, **fingerprint})
    normalized: dict[str, object] = {
        "schema": R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCHEMA,
        "scope": R14_LOCAL_PROBABILITY_MODEL_SOURCES_SCOPE,
        "formula_facts": expected_facts,
        "files": normalized_records,
    }
    if require_current and normalized != local_probability_model_source_fingerprints():
        raise ValueError(
            "priority input probability-model sources do not match the current local files"
        )
    return normalized


def _fraction_from_json(value: object, label: str) -> Fraction:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an exact-fraction object")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ValueError(f"{label} has an invalid numerator or denominator")
    return Fraction(numerator, denominator)


def _validate_fingerprint_record(
    value: object,
    label: str,
    *,
    path_key: str = "path",
) -> dict[str, int | str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a fingerprint object")
    path = value.get(path_key)
    digest = value.get("sha256")
    size = value.get("bytes")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label} has no {path_key}")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} has an invalid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label} has an invalid byte count")
    return {path_key: path, "sha256": digest, "bytes": size}


def _validated_input_audit(
    audit: object,
    ordered: list[str],
    *,
    require_current_local_source: bool,
) -> dict[str, Any]:
    """Validate and normalize the exact input evidence tied to priority rows."""
    if not isinstance(audit, dict):
        raise ValueError("priority input audit is not an object")
    if audit.get("schema") != R14_RANDOM_SETUP_PROBABILITY_SCHEMA:
        raise ValueError("priority input audit has an unsupported schema")
    if audit.get("model") != R14_RANDOM_SETUP_PROBABILITY_MODEL:
        raise ValueError("priority input audit has an unexpected probability model")
    start_directory = audit.get("start_directory")
    if not isinstance(start_directory, str) or not start_directory:
        raise ValueError("priority input audit has no start directory")
    source_files = audit.get("source_files")
    if not isinstance(source_files, list):
        raise ValueError("priority input audit has no source-file records")
    normalized_sources: list[dict[str, Any]] = []
    for index, source in enumerate(source_files):
        fingerprint = _validate_fingerprint_record(source, f"priority input source file {index}")
        rows = source.get("rows") if isinstance(source, dict) else None
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(f"priority input source file {index} has an invalid row count")
        normalized_sources.append({**fingerprint, "rows": rows})
    rows = audit.get("rows")
    unique_rows = audit.get("unique_canonical_rows")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or isinstance(unique_rows, bool)
        or not isinstance(unique_rows, int)
        or rows < 1
        or unique_rows < 1
        or rows != len(ordered)
        or unique_rows != len(ordered)
        or sum(source["rows"] for source in normalized_sources) != rows
    ):
        raise ValueError("priority input audit has inconsistent row counts")
    total_probability = _fraction_from_json(audit.get("total_probability"), "priority input total probability")
    expected_total_probability = sum(
        (r14_random_setup_probability(board) for board in ordered), Fraction(0)
    )
    if total_probability != expected_total_probability:
        raise ValueError("priority input audit has a total probability different from its rows")
    ordered_sha256 = audit.get("ordered_roots_sha256")
    if ordered_sha256 != ordered_boards_sha256(ordered):
        raise ValueError("priority input audit has an ordered-root SHA-256 different from its rows")
    local_sources = _validated_local_probability_model_source_fingerprints(
        audit.get("local_probability_model_sources"),
        require_current=require_current_local_source,
    )
    return {
        "schema": R14_RANDOM_SETUP_PROBABILITY_SCHEMA,
        "model": R14_RANDOM_SETUP_PROBABILITY_MODEL,
        "start_directory": start_directory,
        "source_files": normalized_sources,
        "rows": rows,
        "unique_canonical_rows": unique_rows,
        "total_probability": fraction_json(total_probability),
        "ordered_roots_sha256": ordered_sha256,
        "local_probability_model_sources": local_sources,
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


def priority_rows(boards: Iterable[str], tie_seed: int | None) -> tuple[list[str], list[dict[str, Any]]]:
    """Build the ordered, self-checking rows used by a frozen priority file."""
    ordered = order_r14_random_setup_boards(boards, tie_seed)
    rows: list[dict[str, Any]] = []
    for rank, board in enumerate(ordered, start=1):
        rows.append(
            {
                "schema": R14_RANDOM_SETUP_PRIORITY_ROW_SCHEMA,
                "rank": rank,
                "board": board,
                "white_discs": board[:64].count("O"),
                "orbit_size": r14_random_setup_orbit_size(board),
                "probability": fraction_json(r14_random_setup_probability(board)),
            }
        )
    return ordered, rows


def write_r14_random_setup_priority_manifest(
    path: Path,
    audit: dict[str, Any],
    boards: Iterable[str],
    tie_seed: int | None,
) -> dict[str, Any]:
    """Write and return a complete, immutable probability-priority input file.

    The JSON Lines file stores every ranked position and its exact rational
    probability.  Its sidecar JSON records the hash of that file and the input
    audit.  It is tied to the exact local generator-source fingerprint and to
    the single board snapshot used to produce its rows; the source directory
    is not reread after the audit.  A teacher run reads this file rather than
    recomputing an order from whichever Python version happens to be installed
    at resume time.
    """
    resolved = path.resolve()
    ordered, rows = priority_rows(boards, tie_seed)
    input_audit = _validated_input_audit(
        audit, ordered, require_current_local_source=True
    )
    payload = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(resolved, payload)
    output_fingerprint = {
        "path": resolved.as_posix(),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "rows": len(rows),
    }
    metadata = {
        "schema": R14_RANDOM_SETUP_PRIORITY_MANIFEST_SCHEMA,
        "row_schema": R14_RANDOM_SETUP_PRIORITY_ROW_SCHEMA,
        "model": R14_RANDOM_SETUP_PROBABILITY_MODEL,
        "tie_seed": tie_seed,
        "input_audit": input_audit,
        "output": output_fingerprint,
        "ordered_roots_sha256": ordered_boards_sha256(ordered),
    }
    _atomic_write_text(
        priority_manifest_metadata_path(resolved),
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return metadata


def _read_json_snapshot(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    """Read exactly one immutable-in-memory JSON snapshot and parse it."""
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} is not a JSON object")
    return content, value


def load_r14_random_setup_priority_manifest(
    path: Path,
) -> tuple[list[str], dict[str, Any], dict[str, object]]:
    """Load and fully validate a previously frozen r14 priority file.

    File hashes establish byte identity, but they cannot establish that a
    newly supplied file was generated in probability order.  This loader also
    recomputes the complete order from each row and the recorded tie seed.
    """
    resolved = path.resolve()
    metadata_path = priority_manifest_metadata_path(resolved)
    metadata_content, metadata = _read_json_snapshot(metadata_path, "priority metadata")
    if metadata.get("schema") != R14_RANDOM_SETUP_PRIORITY_MANIFEST_SCHEMA:
        raise ValueError("priority metadata has an unsupported schema")
    if metadata.get("row_schema") != R14_RANDOM_SETUP_PRIORITY_ROW_SCHEMA:
        raise ValueError("priority metadata has an unexpected row schema")
    if metadata.get("model") != R14_RANDOM_SETUP_PROBABILITY_MODEL:
        raise ValueError("priority metadata has an unexpected probability model")
    tie_seed = metadata.get("tie_seed")
    if tie_seed is not None and (isinstance(tie_seed, bool) or not isinstance(tie_seed, int)):
        raise ValueError("priority metadata has an invalid tie seed")
    output = metadata.get("output")
    if not isinstance(output, dict):
        raise ValueError("priority metadata has no output fingerprint")
    if output.get("path") != resolved.as_posix():
        raise ValueError("priority metadata has an unexpected output path")
    try:
        priority_content = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read priority file {resolved}: {error}") from error
    priority_sha256 = hashlib.sha256(priority_content).hexdigest()
    if output.get("sha256") != priority_sha256:
        raise ValueError("priority file does not match its recorded SHA-256")
    if output.get("bytes") != len(priority_content):
        raise ValueError("priority file does not match its recorded byte count")

    boards: list[str] = []
    try:
        lines = priority_content.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError(f"cannot read priority file {resolved}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{resolved}:{line_number}: invalid JSON row: {error}") from error
        if not isinstance(row, dict) or row.get("schema") != R14_RANDOM_SETUP_PRIORITY_ROW_SCHEMA:
            raise ValueError(f"{resolved}:{line_number}: unsupported priority row")
        if row.get("rank") != line_number:
            raise ValueError(f"{resolved}:{line_number}: non-contiguous priority rank")
        board = row.get("board")
        if not isinstance(board, str):
            raise ValueError(f"{resolved}:{line_number}: no board")
        board = validate_r14_random_setup_board(board)
        canonical, _ = canonicalize_board_key(board)
        if board != canonical:
            raise ValueError(f"{resolved}:{line_number}: board is not a spatial representative")
        expected_probability = fraction_json(r14_random_setup_probability(board))
        if (
            row.get("white_discs") != board[:64].count("O")
            or row.get("orbit_size") != r14_random_setup_orbit_size(board)
            or row.get("probability") != expected_probability
        ):
            raise ValueError(f"{resolved}:{line_number}: probability details do not match the board")
        boards.append(board)
    if output.get("rows") != len(boards) or not boards:
        raise ValueError("priority metadata has an unexpected row count")
    if len(boards) != len(set(boards)):
        raise ValueError("priority file contains duplicate boards")
    if metadata.get("ordered_roots_sha256") != ordered_boards_sha256(boards):
        raise ValueError("priority file does not match its recorded ordered-root SHA-256")
    if boards != order_r14_random_setup_boards(boards, tie_seed):
        raise ValueError("priority file is not in its recorded probability order")
    _validated_input_audit(
        metadata.get("input_audit"), boards, require_current_local_source=False
    )
    return boards, metadata, {
        "path": resolved.as_posix(),
        "sha256": priority_sha256,
        "bytes": len(priority_content),
        "metadata_path": metadata_path.as_posix(),
        "metadata_sha256": hashlib.sha256(metadata_content).hexdigest(),
        "rows": len(boards),
        "ordered_roots_sha256": metadata["ordered_roots_sha256"],
        "tie_seed": metadata["tie_seed"],
        "model": metadata["model"],
        "input_audit": metadata["input_audit"],
    }


def priority_manifest_provenance(path: Path) -> dict[str, object]:
    """Return fingerprints from the same validated file snapshot as its rows."""
    _boards, _metadata, provenance = load_r14_random_setup_priority_manifest(path)
    return provenance


def _audit_r14_random_setup_directory_with_order(
    start_dir: Path, tie_seed: int | None = 620
) -> tuple[dict[str, Any], list[str]]:
    """Read one directory snapshot and return both its report and fixed order."""
    files = sorted(start_dir.resolve().glob("*.txt"), key=lambda path: path.as_posix())
    if not files:
        raise ValueError(f"{start_dir}: no .txt position files")

    rows: list[str] = []
    canonical_rows: list[str] = []
    source_files: list[dict[str, object]] = []
    for path in files:
        try:
            content = path.read_bytes()
            lines = content.decode("utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise ValueError(f"cannot read {path}: {error}") from error
        count = 0
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                board = validate_r14_random_setup_board(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            rows.append(board)
            canonical, _ = canonicalize_board_key(board)
            canonical_rows.append(canonical)
            count += 1
        source_files.append(
            {
                "path": path.as_posix(),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "rows": count,
            }
        )

    if len(rows) != len(set(rows)):
        raise ValueError("the r14 input has exact duplicate rows")
    if len(canonical_rows) != len(set(canonical_rows)):
        raise ValueError("the r14 input has rotation/reflection duplicate rows")
    if any(board != canonical for board, canonical in zip(rows, canonical_rows)):
        raise ValueError("the r14 input contains a non-representative orientation")

    ordered = order_r14_random_setup_boards(rows, tie_seed)
    total_probability = sum(
        (r14_random_setup_probability(board) for board in ordered), Fraction(0)
    )
    if total_probability != 1:
        raise ValueError(
            "the r14 input does not cover the full random_setup(14) probability mass: "
            f"{total_probability}"
        )

    by_white_count: Counter[int] = Counter()
    by_orbit_size: Counter[int] = Counter()
    probability_by_white_count: dict[int, Fraction] = {}
    probability_by_orbit_size: dict[int, Fraction] = {}
    for board in ordered:
        white_discs = board[:64].count("O")
        orbit_size = r14_random_setup_orbit_size(board)
        probability = r14_random_setup_probability(board)
        by_white_count[white_discs] += 1
        by_orbit_size[orbit_size] += 1
        probability_by_white_count[white_discs] = (
            probability_by_white_count.get(white_discs, Fraction(0)) + probability
        )
        probability_by_orbit_size[orbit_size] = (
            probability_by_orbit_size.get(orbit_size, Fraction(0)) + probability
        )

    checkpoints = sorted(
        {
            count
            for count in (100, 500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, len(ordered))
            if count <= len(ordered)
        }
    )
    cumulative = Fraction(0)
    checkpoint_mass: dict[str, dict[str, int]] = {}
    checkpoint_iter = iter(checkpoints)
    next_checkpoint = next(checkpoint_iter, None)
    for index, board in enumerate(ordered, start=1):
        cumulative += r14_random_setup_probability(board)
        if index == next_checkpoint:
            checkpoint_mass[str(index)] = fraction_json(cumulative)
            next_checkpoint = next(checkpoint_iter, None)

    report = {
        "schema": R14_RANDOM_SETUP_PROBABILITY_SCHEMA,
        "model": R14_RANDOM_SETUP_PROBABILITY_MODEL,
        "local_probability_model_sources": local_probability_model_source_fingerprints(),
        "start_directory": start_dir.resolve().as_posix(),
        "source_files": source_files,
        "rows": len(rows),
        "unique_canonical_rows": len(set(canonical_rows)),
        "tie_seed": tie_seed,
        "ordered_roots_sha256": ordered_boards_sha256(ordered),
        "total_probability": fraction_json(total_probability),
        "counts_by_white_discs": {str(key): value for key, value in sorted(by_white_count.items())},
        "probability_by_white_discs": {
            str(key): fraction_json(value)
            for key, value in sorted(probability_by_white_count.items())
        },
        "counts_by_orbit_size": {str(key): value for key, value in sorted(by_orbit_size.items())},
        "probability_by_orbit_size": {
            str(key): fraction_json(value)
            for key, value in sorted(probability_by_orbit_size.items())
        },
        "cumulative_probability_by_calculation_count": checkpoint_mass,
    }
    _validated_input_audit(report, ordered, require_current_local_source=True)
    return report, ordered


def audit_r14_random_setup_directory(start_dir: Path, tie_seed: int | None = 620) -> dict[str, Any]:
    """Audit a complete local-r14-generator directory and its priority order."""
    report, _ordered = _audit_r14_random_setup_directory_with_order(start_dir, tie_seed)
    return report


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    from config import START_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-dir", type=Path, default=START_DIR)
    parser.add_argument("--tie-seed", type=int, default=620)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--priority-output",
        type=Path,
        help=(
            "Write every ranked board and local-reimplementation probability as a "
            "frozen JSON Lines input"
        ),
    )
    args = parser.parse_args()
    report, ordered = _audit_r14_random_setup_directory_with_order(
        args.start_dir, args.tie_seed
    )
    _atomic_write_json(args.output, report)
    if args.priority_output is not None:
        metadata = write_r14_random_setup_priority_manifest(
            args.priority_output,
            report,
            ordered,
            args.tie_seed,
        )
        print(
            f"wrote r14 priority manifest with {metadata['output']['rows']} roots; "
            f"sha256={metadata['output']['sha256']}"
        )
    print(
        f"wrote r14 probability audit for {report['rows']} roots; "
        f"ordered_roots_sha256={report['ordered_roots_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
