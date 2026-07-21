from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Sequence

from config import GAME_RECORDS_DIR, WORK_DIR
from othello import normalize_board_text


MANIFEST_SCHEMA = "contest_book_build_manifest_v1"
BOOK_HEADER = "# contest_book_v1"
MOVE_SCORE_RE = re.compile(r"[a-h][1-8]:-?\d+")
RECORD_GENERATION_MANIFEST_FILENAME = "generation_manifest.json"
LOCK_TIMEOUT_SECONDS = 30.0
LOCK_POLL_SECONDS = 0.1


class BookValidationError(ValueError):
    pass


class FileLockTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class BookBuildSpec:
    initial_board: str
    records_dir: Path
    output: Path
    max_book_loss: int
    cut_empty: int | None
    include_game_records: bool = True

    def normalized(self) -> "BookBuildSpec":
        return BookBuildSpec(
            initial_board=normalize_board_text(self.initial_board),
            records_dir=self.records_dir.resolve(),
            # Keep the publication path itself rather than following a possible
            # symlink: os.replace() should replace the named artifact, not an
            # unrelated symlink target.
            output=self.output.absolute(),
            max_book_loss=self.max_book_loss,
            cut_empty=self.cut_empty,
            include_game_records=self.include_game_records,
        )


@dataclass(frozen=True)
class BookMetadata:
    records_seen: int
    records_used: int
    cut_empty: int
    board_lines: int


@dataclass(frozen=True)
class BookStatus:
    current: bool
    reason: str


@dataclass(frozen=True)
class BookBuildOutcome:
    built: bool
    previous_status: BookStatus
    metadata: BookMetadata | None


def manifest_path_for_book(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def lock_path_for_book(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".lock")


def _try_lock(handle: BinaryIO) -> bool:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lock(path: Path, timeout_seconds: float = LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    """Take an OS-owned lock; the persistent lock file is harmless after a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()

        acquired = False
        while not acquired:
            acquired = _try_lock(handle)
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise FileLockTimeout(
                    f"timed out waiting {timeout_seconds:g}s for lock {path}; "
                    "the file may remain on disk, but OS locks are released when a process exits"
                )
            time.sleep(LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            _unlock(handle)


def _parse_non_negative_header(line: str, prefix: str) -> int:
    if not line.startswith(prefix):
        raise BookValidationError(f"missing header {prefix.rstrip()}")
    try:
        value = int(line[len(prefix):])
    except ValueError as exc:
        raise BookValidationError(f"invalid integer in {prefix.rstrip()}") from exc
    if value < 0:
        raise BookValidationError(f"negative value in {prefix.rstrip()}")
    return value


def validate_book_file(path: Path, expected_initial_board: str) -> BookMetadata:
    """Validate enough of an .egcb file to reject partial or misrouted output."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BookValidationError(f"cannot read book: {exc}") from exc

    if len(lines) < 5:
        raise BookValidationError("book is truncated before all headers")
    if lines[0] != BOOK_HEADER:
        raise BookValidationError(f"unexpected book header: {lines[0]!r}")

    initial_prefix = "# initial "
    if not lines[1].startswith(initial_prefix):
        raise BookValidationError("missing initial-board header")
    compact_initial = "".join(lines[1][len(initial_prefix):].split())
    if (
        len(compact_initial) != 65
        or any(cell not in "-XO" for cell in compact_initial[:64])
        or compact_initial[64] not in "XO"
    ):
        raise BookValidationError("invalid initial-board header")
    actual_initial = compact_initial[:64] + " " + compact_initial[64]
    expected_initial = normalize_board_text(expected_initial_board)
    if actual_initial != expected_initial:
        raise BookValidationError(
            f"initial-board mismatch: expected {expected_initial}, got {actual_initial}"
        )

    records_seen = _parse_non_negative_header(lines[2], "# records_seen ")
    records_used = _parse_non_negative_header(lines[3], "# records_used ")
    cut_empty = _parse_non_negative_header(lines[4], "# cut_empty ")
    if records_used > records_seen:
        raise BookValidationError("records_used exceeds records_seen")
    if cut_empty >= 64:
        raise BookValidationError("cut_empty must be in [0, 63]")

    board_lines = 0
    for line_number, line in enumerate(lines[5:], start=6):
        parts = line.split()
        if len(parts) < 4:
            raise BookValidationError(f"invalid book row at line {line_number}")
        board, side, value, *moves = parts
        if len(board) != 64 or any(cell not in "-XO" for cell in board):
            raise BookValidationError(f"invalid board at line {line_number}")
        if side not in {"X", "O"}:
            raise BookValidationError(f"invalid side at line {line_number}")
        try:
            int(value)
        except ValueError as exc:
            raise BookValidationError(f"invalid value at line {line_number}") from exc
        if any(MOVE_SCORE_RE.fullmatch(move) is None for move in moves):
            raise BookValidationError(f"invalid move score at line {line_number}")
        board_lines += 1

    return BookMetadata(records_seen, records_used, cut_empty, board_lines)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORK_DIR.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def fingerprint_files(paths: Sequence[Path]) -> dict[str, int | str]:
    aggregate = hashlib.sha256()
    aggregate.update(b"contest-book-input-files-v1\0")
    total_bytes = 0
    sorted_paths = sorted((path.resolve() for path in paths), key=_stable_path)
    for path in sorted_paths:
        stat = path.stat()
        identity = _stable_path(path)
        content_hash = _sha256_file(path)
        aggregate.update(identity.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(stat.st_size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(content_hash.encode("ascii"))
        aggregate.update(b"\n")
        total_bytes += stat.st_size
    return {
        "count": len(sorted_paths),
        "total_bytes": total_bytes,
        "sha256": aggregate.hexdigest(),
    }


def _builder_provenance() -> dict[str, object]:
    source_paths = [
        WORK_DIR / "build_book.py",
        WORK_DIR / "config.py",
        WORK_DIR / "othello.py",
    ]
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "sources": fingerprint_files(source_paths),
    }


def _source_provenance(spec: BookBuildSpec) -> dict[str, object]:
    record_paths = sorted(spec.records_dir.glob("*.txt")) if spec.records_dir.exists() else []
    generation_manifest = spec.records_dir / RECORD_GENERATION_MANIFEST_FILENAME
    generation_paths = [generation_manifest] if generation_manifest.exists() else []
    game_paths: list[Path] = []
    if spec.include_game_records and GAME_RECORDS_DIR.exists():
        game_paths = sorted(GAME_RECORDS_DIR.glob("*.txt"))
    return {
        "book_records": fingerprint_files(record_paths),
        "record_generation": fingerprint_files(generation_paths),
        "game_records": fingerprint_files(game_paths),
    }


def _build_identity(spec: BookBuildSpec) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "initial_board": spec.initial_board,
        "settings": {
            "records_dir": _stable_path(spec.records_dir),
            "max_book_loss": spec.max_book_loss,
            "cut_empty": spec.cut_empty,
            "include_game_records": spec.include_game_records,
        },
        "builder": _builder_provenance(),
        "inputs": _source_provenance(spec),
    }


def _output_provenance(path: Path, metadata: BookMetadata) -> dict[str, object]:
    return {
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "metadata": asdict(metadata),
    }


def _check_book_status_unlocked(spec: BookBuildSpec) -> BookStatus:
    if not spec.output.exists():
        return BookStatus(False, "book missing")
    try:
        metadata = validate_book_file(spec.output, spec.initial_board)
    except BookValidationError as exc:
        return BookStatus(False, f"invalid book: {exc}")

    manifest_path = manifest_path_for_book(spec.output)
    if not manifest_path.exists():
        return BookStatus(False, "manifest missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return BookStatus(False, f"invalid manifest: {exc}")
    if not isinstance(manifest, dict):
        return BookStatus(False, "invalid manifest: top-level JSON value is not an object")

    try:
        expected_identity = _build_identity(spec)
    except OSError as exc:
        return BookStatus(False, f"cannot fingerprint current inputs: {exc}")
    for key, value in expected_identity.items():
        if manifest.get(key) != value:
            return BookStatus(False, f"manifest {key} is stale")
    try:
        output_provenance = _output_provenance(spec.output, metadata)
    except OSError as exc:
        return BookStatus(False, f"cannot fingerprint current book: {exc}")
    if manifest.get("output") != output_provenance:
        return BookStatus(False, "book content does not match manifest")
    return BookStatus(True, "validated manifest and inputs match")


def check_book_status(spec: BookBuildSpec) -> BookStatus:
    spec = spec.normalized()
    if not spec.output.exists():
        return BookStatus(False, "book missing")
    with file_lock(lock_path_for_book(spec.output)):
        return _check_book_status_unlocked(spec)


@contextmanager
def locked_book_status(spec: BookBuildSpec) -> Iterator[BookStatus]:
    """Hold the publication lock while a caller relies on the returned status."""
    spec = _normalize_and_validate_spec(spec)
    with file_lock(lock_path_for_book(spec.output)):
        yield _check_book_status_unlocked(spec)


def write_json_atomically(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_book_atomically_unlocked(
    spec: BookBuildSpec,
    *,
    runner: Callable[..., subprocess.CompletedProcess[object]] | None = None,
) -> BookMetadata:
    identity_before = _build_identity(spec)
    staging = spec.output.with_name(f".{spec.output.name}.{uuid.uuid4().hex}.tmp")
    cmd = [
        sys.executable,
        str(WORK_DIR / "build_book.py"),
        spec.initial_board,
        "--records-dir", str(spec.records_dir),
        "--output", str(staging),
        "--max-book-loss", str(spec.max_book_loss),
    ]
    if spec.cut_empty is not None:
        cmd.extend(["--cut-empty", str(spec.cut_empty)])
    if not spec.include_game_records:
        cmd.append("--no-game-records")

    run = runner or subprocess.run
    try:
        run(cmd, cwd=WORK_DIR, check=True)
        metadata = validate_book_file(staging, spec.initial_board)
        identity_after = _build_identity(spec)
        if identity_after != identity_before:
            raise RuntimeError("book inputs changed while build_book.py was running")

        staging_provenance = _output_provenance(staging, metadata)
        os.replace(staging, spec.output)
        published_provenance = _output_provenance(spec.output, metadata)
        if published_provenance != staging_provenance:
            raise RuntimeError("published book changed before its manifest could be committed")
        manifest = dict(identity_after)
        manifest["output"] = staging_provenance
        write_json_atomically(manifest_path_for_book(spec.output), manifest)
        return metadata
    finally:
        staging.unlink(missing_ok=True)


def _normalize_and_validate_spec(spec: BookBuildSpec) -> BookBuildSpec:
    spec = spec.normalized()
    if spec.max_book_loss < 0:
        raise ValueError("max_book_loss must be non-negative")
    if spec.cut_empty is not None and not (0 <= spec.cut_empty < 64):
        raise ValueError("cut_empty must be in [0, 63]")
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    return spec


def build_book_atomically(
    spec: BookBuildSpec,
    *,
    runner: Callable[..., subprocess.CompletedProcess[object]] | None = None,
) -> BookMetadata:
    """Build, validate, and atomically publish a book plus provenance manifest."""
    spec = _normalize_and_validate_spec(spec)
    with file_lock(lock_path_for_book(spec.output)):
        return _build_book_atomically_unlocked(spec, runner=runner)


def build_book_if_stale(
    spec: BookBuildSpec,
    *,
    runner: Callable[..., subprocess.CompletedProcess[object]] | None = None,
) -> BookBuildOutcome:
    """Check and, if needed, build while holding one output lock throughout."""
    spec = _normalize_and_validate_spec(spec)
    with file_lock(lock_path_for_book(spec.output)):
        status = _check_book_status_unlocked(spec)
        if status.current:
            return BookBuildOutcome(False, status, None)
        metadata = _build_book_atomically_unlocked(spec, runner=runner)
        return BookBuildOutcome(True, status, metadata)
