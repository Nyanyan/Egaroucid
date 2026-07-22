"""Run the fixed local match that evaluates one prepared root-move table.

The input must be the ``prepared_match_input.json`` written by
``prepare_root_table_match.py``.  For every starting position, this runner
plays exactly two games: the table-using process plays as X in one game and as
O in the other.  The other process uses the same saved Console executable and
the same saved Console resources, but has no contest-root table.

The conditions are deliberately fixed instead of being general command-line
knobs.  A completed result can therefore be checked by
``audit_root_table_matches.py`` without guessing which conditions were used:

* 60 seconds per side, 8 Console threads, and hash level 29;
* one match worker (no simultaneous games);
* deterministic opening order and rotation/reflection using seed 624; and
* Console's deterministic internal random seed 620.

The runner records the executable, main evaluation file, and endgame
move-ordering file in its metadata.  ``-hash 29`` is a Console capacity
setting here; it is not a request for a hash-resource file, so this runner
does not require or record such a file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any

from build_book import transform_board_text
from othello import Board, coord_to_index, normalize_board_text


PREPARED_INPUT_SCHEMA = "prepared_root_table_match_input_v5"
METADATA_SCHEMA_VERSION = 3
MATCH_OPENING_SEED = 624
ENGINE_RANDOM_SEED = 620
GAME_TIME_SECONDS = 60
GAME_THREADS = 8
GAME_HASH_LEVEL = 29
GAME_WORKERS = 1

MOVE_RE = re.compile(r"([a-h][1-8])$", re.IGNORECASE)
CLOCK_RE = re.compile(
    r"remaining time\s+"
    r"(\d+):(\d+):(\d+(?:\.\d+)?)\s*/\s*"
    r"(\d+):(\d+):(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparedMatchInput:
    """The verified immutable files needed by this match runner."""

    input_path: Path
    environment_root: Path
    executable: Path
    evaluation: Path
    endgame_move_ordering: Path
    openings: tuple[str, ...]
    table: Path
    table_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_lines(lines: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_snapshot(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "kind": "file",
        "files": 1,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _directory_snapshot(path: Path) -> dict[str, Any]:
    """Fingerprint a directory by file name, byte count, and file digest."""
    path = path.resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    digest = hashlib.sha256()
    total_bytes = 0
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
        digest.update(b"\n")
    return {
        "path": str(path),
        "kind": "directory",
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _path_from_fingerprint(
    value: object,
    label: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    """Verify a prepared file's path, SHA-256, and byte count."""
    if not isinstance(value, dict):
        raise ValueError(f"prepared input has no {label} fingerprint")
    path_text = value.get("path")
    expected_sha256 = value.get("sha256")
    expected_bytes = value.get("bytes")
    if (
        not isinstance(path_text, str)
        or not isinstance(expected_sha256, str)
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValueError(f"prepared {label} fingerprint is invalid")
    path = Path(path_text).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"prepared {label} path does not match its saved layout")
    if not path.is_file():
        raise ValueError(f"prepared {label} file is missing: {path}")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise ValueError(f"prepared {label} file does not match its SHA-256 or byte count")
    return path


def _relative_path(text: object, label: str) -> Path:
    if not isinstance(text, str) or not text:
        raise ValueError(f"prepared {label} has no relative path")
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"prepared {label} has an unsafe relative path")
    return path


def _path_below(root: Path, relative: Path, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"prepared {label} escapes its execution environment") from error
    return path


def _load_execution_environment(value: object) -> tuple[Path, dict[str, Path]]:
    """Load the saved Console executable and evaluation inputs used by games."""
    if not isinstance(value, dict):
        raise ValueError("prepared input has no teacher execution environment")
    root_text = value.get("path")
    files = value.get("files")
    if not isinstance(root_text, str) or not isinstance(files, list):
        raise ValueError("prepared teacher execution environment is invalid")
    root = Path(root_text).resolve()
    if not root.is_dir():
        raise ValueError(f"prepared teacher execution environment is missing: {root}")

    required_roles = {"executable", "evaluation", "endgame_move_ordering"}
    saved: dict[str, Path] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("prepared teacher execution environment has an invalid file entry")
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("prepared teacher execution environment file has no role")
        if role not in required_roles:
            continue
        if role in saved:
            raise ValueError(f"prepared teacher execution environment repeats role {role!r}")
        relative = _relative_path(entry.get("relative_path"), f"teacher {role}")
        expected = _path_below(root, relative, f"teacher {role}")
        saved[role] = _path_from_fingerprint(entry, f"teacher {role}", expected_path=expected)

    missing = sorted(required_roles - set(saved))
    if missing:
        raise ValueError(
            "prepared teacher execution environment is missing required file role(s): "
            + ", ".join(missing)
        )
    if saved["evaluation"] != root / "resources" / "eval.egev2":
        raise ValueError("prepared main evaluation file has an unexpected Console path")
    if saved["endgame_move_ordering"] != root / "resources" / "eval_move_ordering_end.egev":
        raise ValueError("prepared endgame move-ordering file has an unexpected Console path")
    return root, saved


def _load_openings(value: object) -> tuple[str, ...]:
    if not isinstance(value, dict):
        raise ValueError("prepared input has no starting-position list")
    path_text = value.get("path")
    expected_sha256 = value.get("sha256")
    expected_entries = value.get("entries")
    if (
        not isinstance(path_text, str)
        or not isinstance(expected_sha256, str)
        or isinstance(expected_entries, bool)
        or not isinstance(expected_entries, int)
        or expected_entries < 1
    ):
        raise ValueError("prepared starting-position provenance is invalid")
    path = Path(path_text).resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("prepared starting-position list changed or is missing")
    try:
        openings = tuple(
            normalize_board_text(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"cannot read prepared starting-position list {path}: {error}") from error
    if len(openings) != expected_entries:
        raise ValueError("prepared starting-position count is inconsistent")
    representatives = [_d4_representative(board) for board in openings]
    if len(set(representatives)) != len(representatives):
        raise ValueError("prepared starting-position list has rotation/reflection duplicates")
    return openings


def _d4_representative(board: str) -> str:
    return min(transform_board_text(board, symmetry) for symmetry in range(8))


def load_prepared_match_input(path: Path) -> PreparedMatchInput:
    """Verify and return one immutable input directory for a local match."""
    input_path = path.resolve()
    prepared = _read_json(input_path, "prepared match input")
    if prepared.get("schema") != PREPARED_INPUT_SCHEMA:
        raise ValueError(
            f"{input_path}: expected {PREPARED_INPUT_SCHEMA}, got {prepared.get('schema')!r}"
        )
    selection = prepared.get("selection")
    if not isinstance(selection, dict) or selection.get("level_31_verification_required") is not True:
        raise ValueError(
            "prepared input does not require level-31 verification for every accepted root"
        )
    environment_root, environment = _load_execution_environment(
        prepared.get("teacher_execution_environment")
    )
    openings = _load_openings(prepared.get("openings"))

    table = prepared.get("table")
    if not isinstance(table, dict):
        raise ValueError("prepared input has no temporary table")
    table_path_text = table.get("path")
    table_sha256 = table.get("sha256")
    if not isinstance(table_path_text, str) or not isinstance(table_sha256, str):
        raise ValueError("prepared temporary-table provenance is invalid")
    table_path = Path(table_path_text).resolve()
    if not table_path.is_file() or sha256_file(table_path) != table_sha256:
        raise ValueError("prepared temporary table changed or is missing")
    if table_path.suffix.lower() != ".egcb":
        raise ValueError("prepared temporary table is not an Egaroucid contest-book file")
    unexpected_books = [
        candidate
        for candidate in table_path.parent.rglob("*.egcb")
        if candidate.resolve() != table_path
    ]
    if unexpected_books:
        raise ValueError("prepared temporary-table directory contains another contest-book file")

    return PreparedMatchInput(
        input_path=input_path,
        environment_root=environment_root,
        executable=environment["executable"],
        evaluation=environment["evaluation"],
        endgame_move_ordering=environment["endgame_move_ordering"],
        openings=openings,
        table=table_path,
        table_sha256=table_sha256,
    )


def select_starting_boards(openings: tuple[str, ...], seed: int = MATCH_OPENING_SEED) -> list[str]:
    """Use the same fixed seed-624 procedure documented by the auditor."""
    if seed != MATCH_OPENING_SEED:
        raise ValueError(f"starting-position seed must be {MATCH_OPENING_SEED}")
    generator = random.Random(seed)
    selected = generator.sample(list(openings), len(openings))
    return [transform_board_text(board, generator.randrange(8)) for board in selected]


def engine_command(
    prepared: PreparedMatchInput,
    *,
    table_enabled: bool,
    engine_random_seed: int = ENGINE_RANDOM_SEED,
) -> list[str]:
    """Build the exact Console command for one side of a game."""
    if engine_random_seed != ENGINE_RANDOM_SEED:
        raise ValueError(f"Console random seed must be {ENGINE_RANDOM_SEED}")
    command = [
        str(prepared.executable),
        "-quiet",
        "-noise",
        "-nobook",
        "-t",
        str(GAME_THREADS),
        "-hash",
        str(GAME_HASH_LEVEL),
        "-seed",
        str(engine_random_seed),
        "-eval",
        str(prepared.evaluation),
        "-time",
        str(GAME_TIME_SECONDS),
    ]
    if table_enabled:
        command.extend(["-contestbook", str(prepared.table.parent)])
    return command


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run_git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    commit_result = run_git("rev-parse", "HEAD")
    status_result = run_git("status", "--porcelain", "--untracked-files=no")
    commit = (
        commit_result.stdout.decode("ascii", errors="replace").strip()
        if commit_result.returncode == 0
        else None
    )
    status = status_result.stdout if status_result.returncode == 0 else b""
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status),
        "tracked_status_sha256": hashlib.sha256(status).hexdigest(),
    }


def build_run_spec(
    prepared: PreparedMatchInput,
    output: Path,
    boards: list[str],
    move_timeout: float,
) -> dict[str, Any]:
    """Create the immutable metadata that the match auditor consumes."""
    if move_timeout <= 0:
        raise ValueError("move timeout must be positive")
    table_dir = prepared.table.parent.resolve()
    selected_canonical = sorted(_d4_representative(board) for board in boards)
    pool_canonical = sorted(_d4_representative(board) for board in prepared.openings)
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "git": _git_provenance(),
        "harness": _file_snapshot(Path(__file__)),
        "argv": list(sys.argv[1:]),
        "parsed_args": {
            "prepared_input": str(prepared.input_path),
            "output": str(output.resolve()),
            "time": GAME_TIME_SECONDS,
            "threads": GAME_THREADS,
            "hash": GAME_HASH_LEVEL,
            "workers": GAME_WORKERS,
            "seed": MATCH_OPENING_SEED,
            "engine_random_seed": ENGINE_RANDOM_SEED,
            "random_symmetry": True,
            "level_31_verification_required": True,
            "move_timeout": move_timeout,
            "candidate": str(prepared.executable),
            "baseline": str(prepared.executable),
            "contestbook": None,
            "candidate_contestbook": str(table_dir),
            "baseline_contestbook": None,
            "candidate_extra": "",
            "baseline_extra": "",
            "matches": len(boards),
        },
        "engine_commands": {
            "candidate": engine_command(prepared, table_enabled=True),
            "baseline": engine_command(prepared, table_enabled=False),
        },
        "artifacts": {
            "candidate_binary": _file_snapshot(prepared.executable),
            "baseline_binary": _file_snapshot(prepared.executable),
            "evaluation": _file_snapshot(prepared.evaluation),
            "endgame_move_ordering": _file_snapshot(prepared.endgame_move_ordering),
            "candidate_contestbook": _directory_snapshot(table_dir),
            "baseline_contestbook": None,
            "prepared_input": _file_snapshot(prepared.input_path),
        },
        "openings": {
            "raw_count": len(pool_canonical),
            "d4_unique_count": len(pool_canonical),
            "d4_duplicates_dropped": 0,
            "canonical_pool_sha256": _sha256_lines(pool_canonical),
            "selected_count": len(boards),
            "ordered_sha256": _sha256_lines(boards),
            "d4_canonical_set_sha256": _sha256_lines(selected_canonical),
        },
    }


def metadata_path_for(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".meta.json")


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


def write_metadata(path: Path, run_spec: dict[str, Any]) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_sha256": _canonical_json_sha256(run_spec),
        "run_spec": run_spec,
    }
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def validate_or_create_metadata(output: Path, metadata_path: Path, run_spec: dict[str, Any]) -> None:
    expected_sha256 = _canonical_json_sha256(run_spec)
    if output.exists() and not metadata_path.exists():
        raise RuntimeError(
            f"refusing to resume {output}: metadata sidecar is missing ({metadata_path})"
        )
    if not metadata_path.exists():
        write_metadata(metadata_path, run_spec)
        return
    stored = _read_json(metadata_path, "match metadata")
    stored_spec = stored.get("run_spec")
    stored_sha256 = stored.get("run_spec_sha256")
    if not isinstance(stored_spec, dict) or stored_sha256 != _canonical_json_sha256(stored_spec):
        raise RuntimeError(f"resume metadata checksum is invalid: {metadata_path}")
    if stored_sha256 != expected_sha256 or stored_spec != run_spec:
        changed = sorted(
            key
            for key in set(stored_spec) | set(run_spec)
            if stored_spec.get(key) != run_spec.get(key)
        )
        raise RuntimeError(
            "refusing to resume with different conditions; changed metadata sections: "
            + ", ".join(changed)
        )


def _send(proc: subprocess.Popen[bytes], command: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write((command + "\n").encode("ascii"))
    proc.stdin.flush()


def _read_move(proc: subprocess.Popen[bytes], timeout_seconds: float) -> str:
    """Read one Console move while allowing a wall-clock timeout on Windows."""
    result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def reader() -> None:
        try:
            assert proc.stdout is not None
            while True:
                raw = proc.stdout.readline()
                if not raw:
                    raise RuntimeError(
                        f"engine exited while waiting for a move (code {proc.poll()})"
                    )
                match = MOVE_RE.search(raw.decode("utf-8", errors="replace").strip())
                if match:
                    result.put((True, match.group(1).lower()))
                    return
        except BaseException as error:  # deliver failures to the game-manager thread
            result.put((False, error))

    threading.Thread(target=reader, daemon=True).start()
    try:
        ok, value = result.get(timeout=timeout_seconds)
    except queue.Empty as error:
        raise TimeoutError(
            f"engine did not return a move within {timeout_seconds:.3f} seconds"
        ) from error
    if not ok:
        assert isinstance(value, BaseException)
        raise value
    assert isinstance(value, str)
    return value


def _wait_until_initialized(proc: subprocess.Popen[bytes], log_path: Path, timeout_seconds: float) -> int:
    """Wait for the Console startup marker before the first timed move."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    marker = re.compile(r"^initialized\r?$", re.MULTILINE)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"engine exited during initialization (code {proc.poll()}): {log_path}"
            )
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if marker.search(text):
            return round((time.monotonic() - started) * 1000)
        time.sleep(0.01)
    raise TimeoutError(
        f"engine did not initialize within {timeout_seconds:.3f} seconds: {log_path}"
    )


def _clock_text_to_msec(hours: str, minutes: str, seconds: str) -> int:
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def _audit_engine_log(
    log_path: Path,
    own_color: str,
    exit_code: int | None,
    go_durations_msec: list[int],
    startup_wall_msec: int | None,
) -> dict[str, Any]:
    """Record every signal that the match auditor needs to reject a bad game."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    own_clock_samples: list[int] = []
    for match in CLOCK_RE.finditer(text):
        black_msec = _clock_text_to_msec(*match.groups()[0:3])
        white_msec = _clock_text_to_msec(*match.groups()[3:6])
        own_clock_samples.append(black_msec if own_color == "X" else white_msec)
    error_lines = [line for line in text.splitlines() if "[ERROR]" in line]
    total_go_msec = sum(go_durations_msec)
    harness_overrun_msec = max(0, total_go_msec - GAME_TIME_SECONDS * 1000)
    zero_clock_seen = any(value <= 0 for value in own_clock_samples)
    fatal_count = text.count("[FATAL]")
    timeout_suspected = bool(
        exit_code not in (0, None)
        or zero_clock_seen
        or harness_overrun_msec > 1000
        or fatal_count
        or error_lines
    )
    return {
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "exit_code": exit_code,
        "own_color": own_color,
        "own_clock_samples": len(own_clock_samples),
        "startup_wall_msec": startup_wall_msec,
        "min_own_remaining_msec": min(own_clock_samples) if own_clock_samples else None,
        "last_own_remaining_msec": own_clock_samples[-1] if own_clock_samples else None,
        "zero_clock_seen": zero_clock_seen,
        "go_calls": len(go_durations_msec),
        "total_go_wall_msec": total_go_msec,
        "max_go_wall_msec": max(go_durations_msec, default=0),
        "harness_clock_overrun_msec": harness_overrun_msec,
        "verify_timeout_markers": len(re.findall(r"verify-timeout", text, re.IGNORECASE)),
        "timeout_word_markers": len(re.findall(r"\btimeout\b", text, re.IGNORECASE)),
        "terminated_search_markers": len(
            re.findall(r"\bterminated(?:\s+\d+)?\s*(?:ms)?", text, re.IGNORECASE)
        ),
        "expected_hash_error_lines": 0,
        "unexpected_error_lines": error_lines[:20],
        "fatal_markers": fatal_count,
        "timeout_suspected": timeout_suspected,
    }


def _final_discs(position: Board, side_to_move: str) -> tuple[int, int]:
    """Convert Board's side-to-move representation to absolute X/O counts."""
    current_player_discs = position.cells.count("X")
    current_opponent_discs = position.cells.count("O")
    if side_to_move == "X":
        return current_player_discs, current_opponent_discs
    return current_opponent_discs, current_player_discs


def _disc_difference(black_discs: int, white_discs: int, table_color: str) -> int:
    table_discs = black_discs if table_color == "X" else white_discs
    other_discs = white_discs if table_color == "X" else black_discs
    empty_discs = 64 - black_discs - white_discs
    if table_discs > other_discs:
        return table_discs - other_discs + empty_discs
    if table_discs < other_discs:
        return table_discs - other_discs - empty_discs
    return 0


def _other_color(color: str) -> str:
    return "O" if color == "X" else "X"


def _close_process(proc: subprocess.Popen[bytes]) -> int | None:
    if proc.poll() is None:
        try:
            _send(proc, "quit")
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
            proc.wait(timeout=3)
    return proc.poll()


def play_game(
    match_id: int,
    game_id: int,
    board: str,
    table_color: str,
    prepared: PreparedMatchInput,
    log_dir: Path,
    move_timeout: float,
) -> dict[str, Any]:
    """Play one game, using the prepared table only on ``table_color``."""
    commands = {
        "candidate": engine_command(prepared, table_enabled=True),
        "baseline": engine_command(prepared, table_enabled=False),
    }
    # Alternate the process launch order by match.  Both programs are the same
    # saved executable; this prevents a permanent first-launch advantage.
    role_order = ["candidate", "baseline"] if match_id % 2 == 0 else ["baseline", "candidate"]
    log_paths: dict[str, Path] = {}
    log_files: dict[str, Any] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    exit_codes: dict[str, int | None] = {"candidate": None, "baseline": None}
    startup_wall_msec: dict[str, int | None] = {"candidate": None, "baseline": None}
    go_durations_msec: dict[str, list[int]] = {"candidate": [], "baseline": []}
    game_result: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        for role in role_order:
            log_path = log_dir / f"m{match_id:04d}_g{game_id}_{role}.log"
            log_file = log_path.open("wb")
            log_paths[role] = log_path
            log_files[role] = log_file
            processes[role] = subprocess.Popen(
                commands[role],
                cwd=prepared.environment_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_file,
            )
        for role in role_order:
            startup_wall_msec[role] = _wait_until_initialized(
                processes[role], log_paths[role], move_timeout
            )
        for role in role_order:
            _send(processes[role], "setboard " + board)

        position = Board.from_text(board)
        side_to_move = normalize_board_text(board)[65]
        transcript = ""
        while True:
            legal_moves = position.legal_moves()
            if not legal_moves:
                if position.is_end():
                    break
                position.pass_turn()
                side_to_move = _other_color(side_to_move)
                continue
            active = "candidate" if side_to_move == table_color else "baseline"
            passive = "baseline" if active == "candidate" else "candidate"
            move_started = time.monotonic()
            _send(processes[active], "go")
            move = _read_move(processes[active], move_timeout)
            go_durations_msec[active].append(round((time.monotonic() - move_started) * 1000))
            try:
                position.play(coord_to_index(move))
            except ValueError as error:
                raise RuntimeError(f"illegal move {move} after {transcript} on {board}") from error
            transcript += move
            _send(processes[passive], "play " + move)
            side_to_move = _other_color(side_to_move)

        black_discs, white_discs = _final_discs(position, side_to_move)
        game_result = {
            "game": game_id,
            "candidate_color": table_color,
            "process_launch_order": role_order,
            "candidate_disc_diff": _disc_difference(black_discs, white_discs, table_color),
            "final_discs": [black_discs, white_discs],
            "record": transcript,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        for role, process in processes.items():
            exit_codes[role] = _close_process(process)
        for log_file in log_files.values():
            log_file.close()
        if game_result is not None:
            game_result["engine_audit"] = {
                "candidate": _audit_engine_log(
                    log_paths["candidate"],
                    table_color,
                    exit_codes["candidate"],
                    go_durations_msec["candidate"],
                    startup_wall_msec["candidate"],
                ),
                "baseline": _audit_engine_log(
                    log_paths["baseline"],
                    _other_color(table_color),
                    exit_codes["baseline"],
                    go_durations_msec["baseline"],
                    startup_wall_msec["baseline"],
                ),
            }
    if game_result is None:
        raise RuntimeError("game ended without a result")
    return game_result


def play_match(
    match_id: int,
    board: str,
    prepared: PreparedMatchInput,
    log_dir: Path,
    move_timeout: float,
) -> dict[str, Any]:
    """Play one two-game match with the table side as X once and O once."""
    game_specs = [(0, "X"), (1, "O")]
    if match_id % 2:
        game_specs.reverse()
    execution_order = [game_id for game_id, _color in game_specs]
    games = [
        play_game(
            match_id,
            game_id,
            board,
            table_color,
            prepared,
            log_dir,
            move_timeout,
        )
        for game_id, table_color in game_specs
    ]
    games.sort(key=lambda game: int(game["game"]))
    margin = sum(int(game["candidate_disc_diff"]) for game in games)
    return {
        "match": match_id,
        "board": board,
        "game_execution_order": execution_order,
        "margin": margin,
        "result": "W" if margin > 0 else "L" if margin < 0 else "D",
        "games": games,
    }


def validate_resume_result(result: object, boards: list[str], seen: set[int]) -> None:
    """Reject an incomplete or differently configured row before resuming."""
    if not isinstance(result, dict):
        raise RuntimeError("resumed result is not a JSON object")
    match_id = result.get("match")
    if isinstance(match_id, bool) or not isinstance(match_id, int) or not 0 <= match_id < len(boards):
        raise RuntimeError(f"invalid match id in resume output: {match_id!r}")
    if match_id in seen:
        raise RuntimeError(f"duplicate match id in resume output: {match_id}")
    if result.get("board") != boards[match_id]:
        raise RuntimeError(f"starting position mismatch for resumed match {match_id}")
    games = result.get("games")
    if not isinstance(games, list) or len(games) != 2:
        raise RuntimeError(f"resumed match {match_id} does not contain exactly two games")
    colors = sorted(game.get("candidate_color") for game in games if isinstance(game, dict))
    if colors != ["O", "X"]:
        raise RuntimeError(f"resumed match {match_id} is not a table-side X/O color swap")
    if {game.get("game") for game in games if isinstance(game, dict)} != {0, 1}:
        raise RuntimeError(f"resumed match {match_id} does not contain game identifiers 0 and 1")
    differences = [game.get("candidate_disc_diff") for game in games if isinstance(game, dict)]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in differences):
        raise RuntimeError(f"resumed match {match_id} has an invalid table-side disc difference")
    margin = sum(differences)
    if result.get("margin") != margin:
        raise RuntimeError(f"resumed match {match_id} has an inconsistent margin")
    expected_result = "W" if margin > 0 else "L" if margin < 0 else "D"
    if result.get("result") != expected_result:
        raise RuntimeError(f"resumed match {match_id} has an inconsistent result")
    seen.add(match_id)


def _append_result(output: Path, result: dict[str, Any]) -> None:
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record_failure(output: Path, match_id: int, board: str, error: BaseException) -> Path:
    failure_path = output.with_suffix(output.suffix + ".failures.jsonl")
    log_dir = output.parent / (output.stem + "_logs")
    logs = [
        {
            "path": str(log_path.resolve()),
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        }
        for log_path in sorted(log_dir.glob(f"m{match_id:04d}_g*_*.log"))
    ]
    row = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "match": match_id,
        "board": board,
        "exception_type": type(error).__name__,
        "exception": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "logs": logs,
    }
    _append_result(failure_path, row)
    return failure_path


def _print_summary(results: list[dict[str, Any]]) -> None:
    wins = sum(result["result"] == "W" for result in results)
    draws = sum(result["result"] == "D" for result in results)
    losses = sum(result["result"] == "L" for result in results)
    count = len(results)
    score_rate = (wins + 0.5 * draws) / count if count else 0.0
    mean_margin = sum(int(result["margin"]) for result in results) / count if count else 0.0
    if score_rate == 0.0:
        elo = -math.inf
    elif score_rate == 1.0:
        elo = math.inf
    else:
        elo = 400.0 * math.log10(score_rate / (1.0 - score_rate))
    print(
        f"matches={count} table_side_W/D/L={wins}/{draws}/{losses} "
        f"score_rate={score_rate:.4f} mean_margin={mean_margin:+.3f} elo={elo:+.1f}",
        flush=True,
    )


def run_matches(
    prepared: PreparedMatchInput,
    output: Path,
    move_timeout: float,
    opening_seed: int = MATCH_OPENING_SEED,
    engine_random_seed: int = ENGINE_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Run or safely resume every required two-game match."""
    if move_timeout <= 0:
        raise ValueError("move timeout must be positive")
    if opening_seed != MATCH_OPENING_SEED:
        raise ValueError(f"starting-position seed must be {MATCH_OPENING_SEED}")
    if engine_random_seed != ENGINE_RANDOM_SEED:
        raise ValueError(f"Console random seed must be {ENGINE_RANDOM_SEED}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    boards = select_starting_boards(prepared.openings, opening_seed)
    metadata_path = metadata_path_for(output)
    run_spec = build_run_spec(prepared, output, boards, move_timeout)
    validate_or_create_metadata(output, metadata_path, run_spec)
    print(
        f"metadata={metadata_path} run_spec_sha256={_canonical_json_sha256(run_spec)}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    completed: set[int] = set()
    if output.exists():
        for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                raise RuntimeError(f"empty JSON line in {output}:{line_number}")
            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON in {output}:{line_number}: {error}") from error
            validate_resume_result(result, boards, completed)
            assert isinstance(result, dict)
            results.append(result)
        results.sort(key=lambda row: int(row["match"]))
        if results:
            print(f"resuming with {len(results)} completed matches", flush=True)

    log_dir = output.parent / (output.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    for match_id, board in enumerate(boards):
        if match_id in completed:
            continue
        try:
            result = play_match(match_id, board, prepared, log_dir, move_timeout)
        except BaseException as error:
            failure_path = _record_failure(output, match_id, board, error)
            print(
                f"match {match_id} failed; failure record appended to {failure_path}",
                file=sys.stderr,
                flush=True,
            )
            raise
        _append_result(output, result)
        completed.add(match_id)
        results.append(result)
        results.sort(key=lambda row: int(row["match"]))
        _print_summary(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--move-timeout",
        type=float,
        default=180.0,
        help="Maximum wall-clock seconds to wait for one Console move.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MATCH_OPENING_SEED,
        help=f"Fixed starting-position order and rotation/reflection seed ({MATCH_OPENING_SEED}).",
    )
    parser.add_argument(
        "--engine-random-seed",
        type=int,
        default=ENGINE_RANDOM_SEED,
        help=f"Fixed Console -seed value ({ENGINE_RANDOM_SEED}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared = load_prepared_match_input(args.prepared_input)
    results = run_matches(
        prepared,
        args.output,
        args.move_timeout,
        args.seed,
        args.engine_random_seed,
    )
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
