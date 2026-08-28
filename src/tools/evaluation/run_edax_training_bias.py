#!/usr/bin/env python3
"""Run reproducible Edax measurements for evaluation-training diagnostics.

The input is a headered TSV.  By default the position identifier and board
columns are named ``sample_id`` and ``board``; ``--id-column`` and
``--board-column`` make the runner usable with child/sibling TSV files too.
Every other input column is copied to the result with an ``input_`` prefix.

Fixed-depth mode runs levels 0, 1, 2, 4, 5, 6, 8, 9, and 10 by default.
Edax's LEVEL table maps level D to a genuine fixed-depth search only when
``D <= 10`` and ``n_empties > 2 * D``.  All inputs are checked against that
condition before Edax is started.  ``--exact`` is a separate mode and uses
level 60.

Cold transposition tables are guaranteed without relying on undocumented
process state: every unique (board, requested depth) measurement starts a new
Edax process containing exactly one OBF position.  Duplicate boards are run
once and their result is reused for all corresponding input rows.

The output directory receives:

* a normalized TSV (``edax_results.tsv`` by default),
* the exact command lines and working directories (``commands.txt``),
* raw stdout/stderr and one-position OBF files under ``edax_raw/``,
* an environment/measurement manifest (``edax_environment.json``), and
* SHA-256 values for the executable, evaluation file, input, script, and all
  generated artifacts (``edax_sha256.tsv``).

The executable is deliberately restricted to Edax 4.5.5
``wEdax-x86-64-v3.exe``.  Its version banner is checked before measurement.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_VERSION = "1"
EDAX_VERSION = "4.5.5"
EDAX_EXE_NAME = "wEdax-x86-64-v3.exe"
DEFAULT_DEPTHS = (0, 1, 2, 4, 5, 6, 8, 9, 10)
ALLOWED_FIXED_DEPTHS = frozenset(DEFAULT_DEPTHS)

RESULT_RE = re.compile(
    r"^\s*(?P<index>\d+)\|\s*"
    r"(?P<depth>\d+)"
    r"(?:@\s*(?P<selectivity>\d+)%)?\s+"
    r"(?P<bound>[<>=?]?)(?P<score>[+-]\d+)\s+"
    r"(?P<details>.*)$"
)
TIME_AND_NODES_RE = re.compile(
    r"^\s*(?P<search_time>\d+:\d{2}(?:\.\d+)?)"
    r"(?:\s+(?P<nodes>\d+))?"
    r"(?:\s+(?P<nps>\d+))?"
    r"(?P<pv>.*)$"
)
MOVE_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-H][1-8]|PS)(?![A-Za-z0-9])", re.I)
VERSION_RE = re.compile(r"\bEdax\s+version\s+4\.5\.5\b", re.I)


@dataclass
class InputRow:
    sample_id: str
    board: str
    input_line: int
    extra_values: Dict[str, str]
    unique_board_index: int = 0
    cache_hit: bool = False


@dataclass
class UniqueBoard:
    index: int
    board: str
    n_empties: int
    rows: List[InputRow] = field(default_factory=list)


@dataclass
class ProcessCapture:
    returncode: Optional[int]
    timed_out: bool
    elapsed_seconds: float
    stdout: bytes
    stderr: bytes


@dataclass
class EdaxResult:
    requested_depth: int
    reported_depth: Optional[int]
    reported_selectivity: Optional[int]
    value: Optional[int]
    raw_value: Optional[int]
    best_move: str
    bound: str
    complete: bool
    status: str
    returncode: Optional[int]
    timed_out: bool
    elapsed_seconds: float
    search_time: str
    nodes: Optional[int]
    nps: Optional[int]
    pv: str
    stdout_file: str
    stderr_file: str
    problem_file: str
    command_index: int


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def decode_output(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def canonical_board(raw: str, line_number: int) -> Tuple[str, int]:
    text = raw.strip()
    if not text:
        raise ValueError("empty board at input line {}".format(line_number))
    if ";" in text:
        raise ValueError(
            "board at input line {} contains ';'; pass only the 64 cells and side-to-move".format(
                line_number
            )
        )

    parts = text.split()
    if len(parts) == 2:
        cells, side = parts
    elif len(parts) == 1 and len(parts[0]) == 65:
        cells, side = parts[0][:64], parts[0][64]
    else:
        raise ValueError(
            "board at input line {} must be 64 cells plus a side-to-move symbol".format(
                line_number
            )
        )
    if len(cells) != 64 or len(side) != 1:
        raise ValueError(
            "board at input line {} has {} cells; expected 64".format(
                line_number, len(cells)
            )
        )

    cell_map = {
        "X": "X",
        "x": "X",
        "B": "X",
        "b": "X",
        "O": "O",
        "o": "O",
        "W": "O",
        "w": "O",
        "-": "-",
        ".": "-",
    }
    side_map = {
        "X": "X",
        "x": "X",
        "B": "X",
        "b": "X",
        "O": "O",
        "o": "O",
        "W": "O",
        "w": "O",
    }
    try:
        normalized_cells = "".join(cell_map[symbol] for symbol in cells)
    except KeyError as error:
        raise ValueError(
            "invalid board symbol {!r} at input line {}".format(error.args[0], line_number)
        )
    if side not in side_map:
        raise ValueError(
            "invalid side-to-move symbol {!r} at input line {}".format(side, line_number)
        )
    return normalized_cells + " " + side_map[side], normalized_cells.count("-")


def parse_depths(raw_values: Optional[Sequence[str]]) -> Tuple[int, ...]:
    if raw_values is None:
        return DEFAULT_DEPTHS
    depths: List[int] = []
    for raw in raw_values:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                depth = int(token)
            except ValueError:
                raise ValueError("invalid depth {!r}".format(token))
            if depth not in ALLOWED_FIXED_DEPTHS:
                raise ValueError(
                    "unsupported fixed depth {}; allowed depths are {}".format(
                        depth, ",".join(str(value) for value in DEFAULT_DEPTHS)
                    )
                )
            if depth not in depths:
                depths.append(depth)
    if not depths:
        raise ValueError("--depths did not contain a depth")
    return tuple(depths)


def validate_leaf_depths(boards: Iterable[UniqueBoard], depths: Sequence[int]) -> None:
    failures: List[str] = []
    for unique in boards:
        for depth in depths:
            if depth > 10 or unique.n_empties <= 2 * depth:
                failures.append(
                    "board#{}, n_empties={}, depth={} (requires n_empties > {})".format(
                        unique.index, unique.n_empties, depth, 2 * depth
                    )
                )
    if failures:
        preview = "\n  ".join(failures[:20])
        suffix = "" if len(failures) <= 20 else "\n  ... {} more".format(len(failures) - 20)
        raise ValueError(
            "Edax LEVEL would not be a fixed-depth search for these requests:\n  {}{}\n"
            "Use a smaller --depths subset, or run --exact separately at level 60.".format(
                preview, suffix
            )
        )


def prefixed_extra_names(headers: Sequence[str], id_column: str, board_column: str) -> List[str]:
    return ["input_" + header for header in headers if header not in (id_column, board_column)]


def read_input(
    path: Path,
    id_column: str,
    board_column: str,
    max_rows: int,
) -> Tuple[List[InputRow], List[UniqueBoard], List[str]]:
    rows: List[InputRow] = []
    unique_by_board: "OrderedDict[str, UniqueBoard]" = OrderedDict()

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t", strict=True)
        if reader.fieldnames is None:
            raise ValueError("input TSV is empty")
        headers = list(reader.fieldnames)
        if len(headers) != len(set(headers)):
            raise ValueError("input TSV contains duplicate column names")
        if id_column not in headers:
            raise ValueError("missing ID column {!r}".format(id_column))
        if board_column not in headers:
            raise ValueError("missing board column {!r}".format(board_column))
        extra_headers = [header for header in headers if header not in (id_column, board_column)]

        for record_number, record in enumerate(reader, 1):
            if max_rows and len(rows) >= max_rows:
                break
            input_line = record_number + 1
            if None in record:
                raise ValueError("too many TSV fields at input line {}".format(input_line))
            values = [record.get(header) for header in headers]
            if any(value is None for value in values):
                raise ValueError("missing TSV field at input line {}".format(input_line))
            if all(not str(value).strip() for value in values):
                continue

            sample_id = str(record[id_column]).strip()
            if not sample_id:
                raise ValueError("empty ID at input line {}".format(input_line))
            board, n_empties = canonical_board(str(record[board_column]), input_line)
            extras = {"input_" + name: str(record[name]) for name in extra_headers}
            row = InputRow(sample_id, board, input_line, extras)

            unique = unique_by_board.get(board)
            if unique is None:
                unique = UniqueBoard(len(unique_by_board) + 1, board, n_empties)
                unique_by_board[board] = unique
            row.unique_board_index = unique.index
            row.cache_hit = bool(unique.rows)
            unique.rows.append(row)
            rows.append(row)

    if not rows:
        raise ValueError("input TSV has no data rows")
    return rows, list(unique_by_board.values()), prefixed_extra_names(headers, id_column, board_column)


def capture_process(command: Sequence[str], cwd: Path, timeout_seconds: float) -> ProcessCapture:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
            check=False,
        )
        return ProcessCapture(
            completed.returncode,
            False,
            time.perf_counter() - started,
            completed.stdout,
            completed.stderr,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, bytes) else (error.stdout or "").encode()
        stderr = error.stderr if isinstance(error.stderr, bytes) else (error.stderr or "").encode()
        return ProcessCapture(
            None,
            True,
            time.perf_counter() - started,
            stdout,
            stderr,
        )


def command_text(command: Sequence[str], cwd: Path) -> str:
    if os.name == "nt":
        rendered = subprocess.list2cmdline(list(command))
    else:
        try:
            import shlex

            rendered = shlex.join(command)
        except AttributeError:  # Python 3.7/3.8 fallback; harmless on 3.9.
            import shlex

            rendered = " ".join(shlex.quote(value) for value in command)
    # Keep this a single TSV field when write_commands() prefixes its index.
    return "cwd={} :: {}".format(cwd, rendered)


def parse_result_line(stdout: bytes) -> Tuple[Optional[Mapping[str, object]], str]:
    matches: List[Mapping[str, object]] = []
    for line in decode_output(stdout).splitlines():
        match = RESULT_RE.match(line)
        if not match:
            continue
        details = match.group("details")
        detail_match = TIME_AND_NODES_RE.match(details)
        search_time = ""
        nodes: Optional[int] = None
        nps: Optional[int] = None
        pv = details.strip()
        if detail_match:
            search_time = detail_match.group("search_time") or ""
            nodes_text = detail_match.group("nodes")
            nps_text = detail_match.group("nps")
            nodes = int(nodes_text) if nodes_text else None
            nps = int(nps_text) if nps_text else None
            pv = (detail_match.group("pv") or "").strip()
        moves = MOVE_RE.findall(pv)
        matches.append(
            {
                "index": int(match.group("index")),
                "reported_depth": int(match.group("depth")),
                "reported_selectivity": (
                    int(match.group("selectivity")) if match.group("selectivity") else None
                ),
                "bound": match.group("bound") or "",
                "raw_value": int(match.group("score")),
                "best_move": moves[0].upper() if moves else "",
                "search_time": search_time,
                "nodes": nodes,
                "nps": nps,
                "pv": pv,
                "line": line,
            }
        )
    if len(matches) != 1:
        return None, "parsed_{}_result_lines".format(len(matches))
    if int(matches[0]["index"]) != 1:
        return matches[0], "unexpected_problem_index_{}".format(matches[0]["index"])
    return matches[0], ""


def relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_dir.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def run_one(
    executable: Path,
    edax_dir: Path,
    output_dir: Path,
    raw_dir: Path,
    unique: UniqueBoard,
    requested_depth: int,
    exact: bool,
    timeout_seconds: float,
    commands: List[str],
) -> EdaxResult:
    board_dir = raw_dir / "board_{:06d}".format(unique.index)
    board_dir.mkdir(parents=True, exist_ok=True)
    problem_path = board_dir / "position.obf"
    if not problem_path.exists():
        # Path.write_text() gained its newline argument after Python 3.9.
        with problem_path.open("w", encoding="ascii", newline="\n") as destination:
            destination.write(unique.board + "\n")

    label = "exact" if exact else "depth_{:02d}".format(requested_depth)
    stdout_path = board_dir / (label + ".stdout.txt")
    stderr_path = board_dir / (label + ".stderr.txt")
    command = [
        str(executable),
        "-n",
        "1",
        "-h",
        "16",
        "-book-usage",
        "off",
        "-eval-file",
        "data/eval.dat",
        "-l",
        str(requested_depth),
        "-solve",
        str(problem_path.resolve()),
    ]
    commands.append(command_text(command, edax_dir))
    command_index = len(commands)
    capture = capture_process(command, edax_dir, timeout_seconds)
    stdout_path.write_bytes(capture.stdout)
    stderr_path.write_bytes(capture.stderr)

    parsed, parse_status = parse_result_line(capture.stdout)
    reported_depth: Optional[int] = None
    reported_selectivity: Optional[int] = None
    raw_value: Optional[int] = None
    value: Optional[int] = None
    best_move = ""
    bound = ""
    search_time = ""
    nodes: Optional[int] = None
    nps: Optional[int] = None
    pv = ""

    if parsed is not None:
        reported_depth = int(parsed["reported_depth"])
        selectivity_value = parsed["reported_selectivity"]
        reported_selectivity = int(selectivity_value) if selectivity_value is not None else None
        raw_value = int(parsed["raw_value"])
        best_move = str(parsed["best_move"])
        bound = str(parsed["bound"])
        search_time = str(parsed["search_time"])
        nodes_value = parsed["nodes"]
        nps_value = parsed["nps"]
        nodes = int(nodes_value) if nodes_value is not None else None
        nps = int(nps_value) if nps_value is not None else None
        pv = str(parsed["pv"])

    status = "complete"
    if capture.timed_out:
        status = "timeout"
    elif capture.returncode != 0:
        status = "nonzero_exit_{}".format(capture.returncode)
    elif parse_status:
        status = parse_status
    elif bound:
        status = "bounded_result_{}".format(bound)
    elif reported_selectivity is not None:
        status = "unexpected_selectivity_{}".format(reported_selectivity)
    elif reported_depth != (unique.n_empties if exact else requested_depth):
        status = "reported_depth_mismatch"

    complete = status == "complete"
    if complete:
        value = raw_value

    return EdaxResult(
        requested_depth=requested_depth,
        reported_depth=reported_depth,
        reported_selectivity=reported_selectivity,
        value=value,
        raw_value=raw_value,
        best_move=best_move,
        bound=bound,
        complete=complete,
        status=status,
        returncode=capture.returncode,
        timed_out=capture.timed_out,
        elapsed_seconds=capture.elapsed_seconds,
        search_time=search_time,
        nodes=nodes,
        nps=nps,
        pv=pv,
        stdout_file=relative_to_output(stdout_path, output_dir),
        stderr_file=relative_to_output(stderr_path, output_dir),
        problem_file=relative_to_output(problem_path, output_dir),
        command_index=command_index,
    )


BASE_RESULT_HEADER = [
    "sample_id",
    "board",
    "input_line",
    "unique_board_index",
    "duplicate_count",
    "cache_hit",
    "mode",
    "n_empties",
    "requested_depth",
    "reported_depth",
    "reported_selectivity",
    "value",
    "raw_value",
    "best_move",
    "bound",
    "complete",
    "status",
    "returncode",
    "timed_out",
    "elapsed_seconds",
    "search_time",
    "nodes",
    "nps",
    "pv",
    "stdout_file",
    "stderr_file",
    "problem_file",
    "command_index",
]


def optional_field(value: object) -> object:
    return "" if value is None else value


def normalized_row(
    row: InputRow,
    unique: UniqueBoard,
    result: EdaxResult,
    exact: bool,
) -> Dict[str, object]:
    output: Dict[str, object] = {
        "sample_id": row.sample_id,
        "board": row.board,
        "input_line": row.input_line,
        "unique_board_index": unique.index,
        "duplicate_count": len(unique.rows),
        "cache_hit": int(row.cache_hit),
        "mode": "exact" if exact else "fixed_depth",
        "n_empties": unique.n_empties,
        "requested_depth": result.requested_depth,
        "reported_depth": optional_field(result.reported_depth),
        "reported_selectivity": optional_field(result.reported_selectivity),
        "value": optional_field(result.value),
        "raw_value": optional_field(result.raw_value),
        "best_move": result.best_move,
        "bound": result.bound,
        "complete": int(result.complete),
        "status": result.status,
        "returncode": optional_field(result.returncode),
        "timed_out": int(result.timed_out),
        "elapsed_seconds": "{:.6f}".format(result.elapsed_seconds),
        "search_time": result.search_time,
        "nodes": optional_field(result.nodes),
        "nps": optional_field(result.nps),
        "pv": result.pv,
        "stdout_file": result.stdout_file,
        "stderr_file": result.stderr_file,
        "problem_file": result.problem_file,
        "command_index": result.command_index,
    }
    output.update(row.extra_values)
    return output


def ensure_output_targets_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError(
            "refusing to overwrite existing Edax output artifacts:\n  {}".format(
                "\n  ".join(existing)
            )
        )


def write_commands(path: Path, commands: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        destination.write("command_index\tcommand\n")
        for index, command in enumerate(commands, 1):
            destination.write("{}\t{}\n".format(index, command))


def artifact_records(
    roles: Mapping[Path, str],
    generated_roots: Sequence[Path],
    excluded: Sequence[Path],
) -> List[Tuple[str, Path]]:
    excluded_resolved = {path.resolve() for path in excluded}
    result: Dict[Path, str] = {path.resolve(): role for path, role in roles.items()}
    for root in generated_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.resolve() not in excluded_resolved:
                result.setdefault(path.resolve(), "generated_artifact")
    return sorted(((role, path) for path, role in result.items()), key=lambda item: str(item[1]))


def write_sha256(
    path: Path,
    roles: Mapping[Path, str],
    generated_roots: Sequence[Path],
) -> None:
    records = artifact_records(roles, generated_roots, [path])
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["role", "path", "size_bytes", "sha256"])
        for role, artifact in records:
            writer.writerow([role, str(artifact), artifact.stat().st_size, sha256_file(artifact)])


def safe_child(output_dir: Path, name: str, option_name: str) -> Path:
    candidate = Path(name)
    if candidate.name != name or candidate.is_absolute():
        raise ValueError("{} must be a file name, not a path".format(option_name))
    return output_dir / name


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[3]
    default_executable = (
        repository_root
        / "bin"
        / "versions"
        / "edax_4_5_5"
        / "bin"
        / EDAX_EXE_NAME
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--board-column", default="board")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--depths",
        nargs="+",
        help="fixed depths (space- or comma-separated subset of 0,1,2,4,5,6,8,9,10)",
    )
    mode.add_argument("--exact", action="store_true", help="run exact level 60 only")
    parser.add_argument("--edax-exe", type=Path, default=default_executable)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="per-process timeout; 0 means no timeout (default)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="read at most this many nonblank input rows; 0 means all",
    )
    parser.add_argument("--output-name", default="edax_results.tsv")
    parser.add_argument("--commands-name", default="commands.txt")
    parser.add_argument("--environment-name", default="edax_environment.json")
    parser.add_argument("--sha256-name", default="edax_sha256.tsv")
    parser.add_argument("--raw-dir-name", default="edax_raw")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started_at = utc_now()

    try:
        if args.timeout_seconds < 0:
            raise ValueError("--timeout-seconds must be nonnegative")
        if args.max_rows < 0:
            raise ValueError("--max-rows must be nonnegative")
        input_path = args.input_tsv.resolve()
        if not input_path.is_file():
            raise ValueError("input TSV not found: {}".format(input_path))

        executable = args.edax_exe.resolve()
        if not executable.is_file():
            raise ValueError("Edax executable not found: {}".format(executable))
        if executable.name.lower() != EDAX_EXE_NAME.lower():
            raise ValueError(
                "expected {}, got {}".format(EDAX_EXE_NAME, executable.name)
            )
        edax_dir = executable.parent
        eval_path = edax_dir / "data" / "eval.dat"
        if not eval_path.is_file():
            raise ValueError("Edax evaluation file not found: {}".format(eval_path))

        depths = (60,) if args.exact else parse_depths(args.depths)
        rows, unique_boards, extra_headers = read_input(
            input_path, args.id_column, args.board_column, args.max_rows
        )
        if not args.exact:
            validate_leaf_depths(unique_boards, depths)

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = safe_child(output_dir, args.output_name, "--output-name")
        commands_path = safe_child(output_dir, args.commands_name, "--commands-name")
        environment_path = safe_child(
            output_dir, args.environment_name, "--environment-name"
        )
        sha256_path = safe_child(output_dir, args.sha256_name, "--sha256-name")
        raw_dir = safe_child(output_dir, args.raw_dir_name, "--raw-dir-name")
        ensure_output_targets_absent(
            [output_path, commands_path, environment_path, sha256_path, raw_dir]
        )
        raw_dir.mkdir(parents=True)

        commands: List[str] = []
        version_command = [str(executable), "-version"]
        commands.append(command_text(version_command, edax_dir))
        version_capture = capture_process(version_command, edax_dir, 30.0)
        version_stdout_path = raw_dir / "version.stdout.txt"
        version_stderr_path = raw_dir / "version.stderr.txt"
        version_stdout_path.write_bytes(version_capture.stdout)
        version_stderr_path.write_bytes(version_capture.stderr)
        version_text = decode_output(version_capture.stdout) + "\n" + decode_output(
            version_capture.stderr
        )
        if version_capture.timed_out or version_capture.returncode != 0:
            raise ValueError("Edax version probe failed; see edax_raw/version.*.txt")
        if not VERSION_RE.search(version_text):
            raise ValueError(
                "executable did not report Edax {}; see edax_raw/version.*.txt".format(
                    EDAX_VERSION
                )
            )

        cache: Dict[Tuple[str, int], EdaxResult] = {}
        complete_invocations = 0
        invocation_count = 0
        for unique in unique_boards:
            for requested_depth in depths:
                invocation_count += 1
                # One line per invocation makes long, multi-depth runs harder to
                # audit because the useful completion/error messages disappear
                # into megabytes of progress output.  Report once per board at
                # a stable interval instead; every invocation is still recorded
                # verbatim in commands.txt and the normalized TSV.
                report_progress = (
                    requested_depth == depths[0]
                    and (
                        unique.index == 1
                        or unique.index == len(unique_boards)
                        or unique.index % 10 == 0
                    )
                )
                if not args.quiet and report_progress:
                    print(
                        "Edax board {}/{} ({} depths; {} empty squares)".format(
                            unique.index,
                            len(unique_boards),
                            len(depths),
                            unique.n_empties,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                result = run_one(
                    executable,
                    edax_dir,
                    output_dir,
                    raw_dir,
                    unique,
                    requested_depth,
                    args.exact,
                    args.timeout_seconds,
                    commands,
                )
                cache[(unique.board, requested_depth)] = result
                complete_invocations += int(result.complete)

        # Emit in original input order even when identical boards were separated
        # in the source TSV.  Each row still points to the single cached process
        # result for its canonical board and requested depth.
        unique_by_index = {unique.index: unique for unique in unique_boards}
        normalized: List[Dict[str, object]] = []
        for row in rows:
            unique = unique_by_index[row.unique_board_index]
            for requested_depth in depths:
                normalized.append(
                    normalized_row(
                        row,
                        unique,
                        cache[(unique.board, requested_depth)],
                        args.exact,
                    )
                )

        result_header = BASE_RESULT_HEADER + extra_headers
        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=result_header,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(normalized)
        write_commands(commands_path, commands)

        input_hash = sha256_file(input_path)
        executable_hash = sha256_file(executable)
        eval_hash = sha256_file(eval_path)
        script_hash = sha256_file(Path(__file__).resolve())
        environment = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cwd": str(Path.cwd().resolve()),
            "script": str(Path(__file__).resolve()),
            "input_tsv": str(input_path),
            "output_tsv": str(output_path),
            "id_column": args.id_column,
            "board_column": args.board_column,
            "mode": "exact" if args.exact else "fixed_depth",
            "requested_depths": list(depths),
            "fixed_depth_guard": "depth <= 10 and n_empties > 2 * depth",
            "input_rows": len(rows),
            "unique_boards": len(unique_boards),
            "duplicate_rows": len(rows) - len(unique_boards),
            "edax_invocations": invocation_count,
            "complete_invocations": complete_invocations,
            "one_position_per_process": True,
            "n_tasks": 1,
            "hash_table_bits": 16,
            "book_usage": "off",
            "eval_file_argument": "data/eval.dat",
            "edax_executable": str(executable),
            "edax_eval_file": str(eval_path.resolve()),
            "edax_version_required": EDAX_VERSION,
            "timeout_seconds": args.timeout_seconds,
            "max_rows": args.max_rows,
            "sha256": {
                "script": script_hash,
                "input_tsv": input_hash,
                "edax_executable": executable_hash,
                "edax_eval_file": eval_hash,
            },
        }
        # Path.write_text() gained its newline argument after Python 3.9.
        with environment_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            destination.write(
                json.dumps(environment, ensure_ascii=False, indent=2) + "\n"
            )

        roles = {
            Path(__file__).resolve(): "runner_script",
            input_path: "input_tsv",
            executable: "edax_executable",
            eval_path.resolve(): "edax_eval_file",
            output_path.resolve(): "normalized_results",
            commands_path.resolve(): "commands",
            environment_path.resolve(): "environment",
        }
        write_sha256(sha256_path, roles, [raw_dir])

        if not args.quiet:
            print(
                "wrote {} rows; {}/{} Edax invocations complete; {} unique boards".format(
                    len(normalized),
                    complete_invocations,
                    invocation_count,
                    len(unique_boards),
                ),
                file=sys.stderr,
            )
        return 0 if complete_invocations == invocation_count else 2
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
