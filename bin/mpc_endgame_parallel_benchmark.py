#!/usr/bin/env python3
"""Measure fixed-selectivity endgame search on a set of positions.

The benchmark executable is started once per position.  This intentionally
keeps the global transposition table and the thread-local endgame tables cold
for every measured search.  Engine-reported time excludes process startup,
hash allocation, and evaluation-file loading; wall time is recorded
separately.

Example, run from the repository root:

    python bin/mpc_endgame_parallel_benchmark.py 0 8 25 \
        ignored/tmp/example/end_mpc_search_benchmark.exe

MPC level 0 is the compatibility label 74%, and level 6 is 100% (no MPC).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POSITIONS = SCRIPT_DIR / "problem" / "ggs_mpc_endgame_40_20260823.txt"
BOARD_RE = re.compile(r"^[XO-]{64} [XO]$")
MOVE_RE = re.compile(r"^[a-h][1-8]$")


@dataclass(frozen=True)
class Position:
    index: int
    board: str
    empties: int


@dataclass
class Result:
    index: int
    board: str
    empties: int
    return_code: int | None
    timed_out: bool
    error: str
    value: int | None
    move: str
    depth: int | None
    mpc_level: int | None
    nodes: int | None
    time_ms: int | None
    nps: int | None
    wall_time_ms: int
    stdout: str
    stderr: str


class OthelloBoard:
    DIRECTIONS = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def __init__(self, board: str) -> None:
        cells, side = board.split()
        self.cells = cells
        self.side = side

    @staticmethod
    def _inside(row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def legal_moves(self) -> set[str]:
        opponent = "O" if self.side == "X" else "X"
        legal: set[str] = set()
        for pos, cell in enumerate(self.cells):
            if cell != "-":
                continue
            row, col = divmod(pos, 8)
            for drow, dcol in self.DIRECTIONS:
                nrow = row + drow
                ncol = col + dcol
                found_opponent = False
                while self._inside(nrow, ncol):
                    npos = nrow * 8 + ncol
                    if self.cells[npos] != opponent:
                        break
                    found_opponent = True
                    nrow += drow
                    ncol += dcol
                if (
                    found_opponent
                    and self._inside(nrow, ncol)
                    and self.cells[nrow * 8 + ncol] == self.side
                ):
                    legal.add(chr(ord("a") + col) + str(row + 1))
                    break
        return legal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-MPC full-depth endgame benchmark with a fresh "
            "process and cold transposition tables for every position."
        )
    )
    parser.add_argument("mpc_level", type=int, help="MPC level from 0 to 6")
    parser.add_argument("n_threads", type=int, help="search thread count")
    parser.add_argument("hash_level", type=int, help="hash level from 0 to 29")
    parser.add_argument("exe", type=Path, help="end_mpc_search_benchmark executable")
    parser.add_argument(
        "positions",
        nargs="?",
        type=Path,
        default=DEFAULT_POSITIONS,
        help="plain text file containing one '<64 cells> <side>' position per line",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=300.0,
        help="wall-clock timeout for one position (default: 300)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="ask the engine driver to print iteration and YBWC statistics",
    )
    args = parser.parse_args()
    if not 0 <= args.mpc_level <= 6:
        parser.error("mpc_level must be between 0 and 6")
    if args.n_threads <= 0:
        parser.error("n_threads must be positive")
    if not 0 <= args.hash_level <= 29:
        parser.error("hash_level must be between 0 and 29")
    if args.case_timeout_seconds <= 0:
        parser.error("--case-timeout-seconds must be positive")
    return args


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SCRIPT_DIR / path).resolve()


def load_positions(path: Path) -> list[Position]:
    positions: list[Position] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip().upper()
        if not line or line.startswith("#"):
            continue
        if not BOARD_RE.fullmatch(line):
            raise ValueError(f"{path}:{line_number}: invalid board")
        if line in seen:
            raise ValueError(f"{path}:{line_number}: duplicate board")
        seen.add(line)
        empties = line[:64].count("-")
        positions.append(Position(len(positions), line, empties))
    if not positions:
        raise ValueError(f"{path}: no positions")
    return positions


def parse_driver_output(stdout: str) -> dict[str, int | str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("missing benchmark result row")
    header = lines[-2].split("\t")
    values = lines[-1].split("\t")
    expected = ["value", "move", "depth", "mpc_level", "nodes", "time_ms", "nps"]
    if header != expected or len(values) != len(expected):
        raise ValueError("unexpected benchmark result format")
    parsed: dict[str, int | str] = {"move": values[1].lower()}
    for index, key in enumerate(expected):
        if key == "move":
            continue
        parsed[key] = int(values[index])
    return parsed


def validate_result(position: Position, parsed: dict[str, int | str], requested_mpc: int) -> None:
    if parsed["depth"] != position.empties:
        raise ValueError(
            f"reported depth {parsed['depth']} differs from {position.empties} empties"
        )
    if parsed["mpc_level"] != requested_mpc:
        raise ValueError(
            f"reported MPC level {parsed['mpc_level']} differs from {requested_mpc}"
        )
    move = str(parsed["move"])
    legal = OthelloBoard(position.board).legal_moves()
    if legal:
        if not MOVE_RE.fullmatch(move) or move not in legal:
            raise ValueError(f"illegal move {move}; legal moves are {sorted(legal)}")
    elif move not in {"pa", "pass"}:
        raise ValueError(f"expected pass but received {move}")
    if int(parsed["nodes"]) < 0 or int(parsed["time_ms"]) < 0 or int(parsed["nps"]) < 0:
        raise ValueError("negative performance value")


def run_position(
    position: Position,
    exe: Path,
    mpc_level: int,
    n_threads: int,
    hash_level: int,
    timeout_seconds: float,
    diagnostics: bool,
) -> Result:
    command = [
        str(exe),
        str(mpc_level),
        str(n_threads),
        str(hash_level),
        position.board,
    ]
    if diagnostics:
        command.append("--diagnostics")
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        wall_time_ms = round((time.perf_counter() - start) * 1000)
    except subprocess.TimeoutExpired as error:
        wall_time_ms = round((time.perf_counter() - start) * 1000)
        return Result(
            position.index,
            position.board,
            position.empties,
            None,
            True,
            f"timeout after {timeout_seconds:g} seconds",
            None,
            "",
            None,
            None,
            None,
            None,
            None,
            wall_time_ms,
            (error.stdout or "") if isinstance(error.stdout, str) else "",
            (error.stderr or "") if isinstance(error.stderr, str) else "",
        )

    parsed: dict[str, int | str] = {}
    error_message = ""
    if completed.returncode != 0:
        error_message = f"benchmark exited with code {completed.returncode}"
    else:
        try:
            parsed = parse_driver_output(completed.stdout)
            validate_result(position, parsed, mpc_level)
        except ValueError as error:
            error_message = str(error)

    return Result(
        position.index,
        position.board,
        position.empties,
        completed.returncode,
        False,
        error_message,
        int(parsed["value"]) if "value" in parsed else None,
        str(parsed.get("move", "")),
        int(parsed["depth"]) if "depth" in parsed else None,
        int(parsed["mpc_level"]) if "mpc_level" in parsed else None,
        int(parsed["nodes"]) if "nodes" in parsed else None,
        int(parsed["time_ms"]) if "time_ms" in parsed else None,
        int(parsed["nps"]) if "nps" in parsed else None,
        wall_time_ms,
        completed.stdout,
        completed.stderr,
    )


def public_result(result: Result) -> dict[str, object]:
    data = asdict(result)
    data.pop("stdout")
    data.pop("stderr")
    return data


def main() -> int:
    args = parse_args()
    exe = resolve_path(args.exe)
    positions_path = resolve_path(args.positions)
    if not exe.is_file():
        print(f"benchmark executable not found: {exe}", file=sys.stderr)
        return 2
    if not positions_path.is_file():
        print(f"positions file not found: {positions_path}", file=sys.stderr)
        return 2
    try:
        positions = load_positions(positions_path)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    config = {
        "executable": str(exe),
        "positions": str(positions_path),
        "position_count": len(positions),
        "mpc_level": args.mpc_level,
        "threads": args.n_threads,
        "hash_level": args.hash_level,
        "case_timeout_seconds": args.case_timeout_seconds,
        "diagnostics": args.diagnostics,
        "cold_table_method": "fresh process per position",
    }
    print("CONFIG\t" + json.dumps(config, ensure_ascii=False), flush=True)

    results: list[Result] = []
    for position in positions:
        result = run_position(
            position,
            exe,
            args.mpc_level,
            args.n_threads,
            args.hash_level,
            args.case_timeout_seconds,
            args.diagnostics,
        )
        results.append(result)
        marker = "ERROR" if result.error else "RESULT"
        print(
            marker + "\t" + json.dumps(public_result(result), ensure_ascii=False),
            flush=True,
        )
        if result.stdout:
            print(
                "DRIVER_STDOUT\t"
                + json.dumps(
                    {"index": position.index, "text": result.stdout}, ensure_ascii=False
                ),
                flush=True,
            )
        if result.stderr:
            print(
                "DRIVER_STDERR\t"
                + json.dumps(
                    {"index": position.index, "text": result.stderr}, ensure_ascii=False
                ),
                flush=True,
            )
        if result.error:
            break

    valid = [result for result in results if not result.error]
    total_nodes = sum(result.nodes or 0 for result in valid)
    total_time_ms = sum(result.time_ms or 0 for result in valid)
    summary = {
        "attempted_count": len(results),
        "valid_count": len(valid),
        "error_count": len(results) - len(valid),
        "total_nodes": total_nodes,
        "total_time_ms": total_time_ms,
        "aggregate_nps": (
            total_nodes * 1000 // total_time_ms if total_time_ms > 0 else 0
        ),
        "wall_time_ms": sum(result.wall_time_ms for result in results),
    }
    print("TOTAL\t" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if len(valid) == len(positions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
