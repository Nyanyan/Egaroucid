#!/usr/bin/env python3
"""Benchmark early endgame solving with an empty transposition table.

The script extracts positions from Egaroucid GGS game records and starts a
fresh Console process for every position.  A fresh process is intentional:
`clearcache` clears the global transposition table, but the specialized
endgame code also has thread-local local tables.  Process isolation guarantees
that both kinds of tables start empty.

The engine is run with a fixed move time.  The benchmark records whether the
root endgame search was attempted and whether it completed at or above the
requested selectivity.  Engine-reported search time excludes process startup,
hash allocation, and evaluation-file loading.

Examples (run from bin):

    python cold_endgame_benchmark.py

    python cold_endgame_benchmark.py --games "ggs/log/game/2026-08-23-*.txt"

    python cold_endgame_benchmark.py --positions previous/corpus.jsonl \
        --exe Egaroucid_for_Console_clang.exe --threads 20 --hash 29

Use --extract-only to create a reproducible corpus without running searches.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import glob
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GAME_DIR = SCRIPT_DIR / "ggs" / "log" / "game"
DEFAULT_EXE = SCRIPT_DIR / "Egaroucid_for_Console_clang.exe"
DEFAULT_CORPUS = SCRIPT_DIR / "problem" / "cold_endgame_2026-08-23_egrcd.txt"

BOARD_RE = re.compile(r"^[XO-]{64}\s+[XO]$", re.IGNORECASE)
DEPTH_RE = re.compile(r"^(-?\d+)@([0-9]+(?:\.[0-9]+)?)%$")
END_LINE_RE = re.compile(
    r"end\s+depth\s+(\d+)@([0-9]+(?:\.[0-9]+)?)%\s+(.*)",
    re.IGNORECASE,
)
END_TIME_RE = re.compile(r"\btime\s+(\d+)\b", re.IGNORECASE)
MID_LINE_RE = re.compile(
    r"mid\s+depth\s+(\d+)@([0-9]+(?:\.[0-9]+)?)%\s+(.*)",
    re.IGNORECASE,
)
TERMINATED_TIME_RE = re.compile(r"\bterminated\s+(\d+)\s+ms\b", re.IGNORECASE)


@dataclass
class Position:
    board: str
    empties: int
    source: str
    ply: int
    side: str
    actor: str
    game_id: str


@dataclass
class SearchResult:
    index: int
    board: str
    empties: int
    source: str
    ply: int
    side: str
    actor: str
    game_id: str
    return_code: Optional[int]
    timed_out: bool
    error: str
    result_depth: Optional[int]
    result_selectivity: Optional[float]
    move: str
    score: Optional[int]
    engine_time_ms: Optional[int]
    nodes: Optional[int]
    nps: Optional[int]
    wall_time_ms: int
    cpu_time_ms: Optional[int]
    cpu_average_cores: Optional[float]
    cpu_utilization_percent: Optional[float]
    end_attempted: bool
    end_completed: bool
    end_start_time_ms: Optional[int]
    end_search_time_ms: Optional[int]
    first_end_selectivity: Optional[float]
    first_end_time_ms: Optional[int]
    first_end_search_time_ms: Optional[int]
    exact_completed: bool
    first_exact_time_ms: Optional[int]


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

    def __init__(self, cells: str, side: str) -> None:
        cells = cells.upper()
        side = side.upper()
        if len(cells) != 64 or any(cell not in "XO-" for cell in cells):
            raise ValueError("board must contain exactly 64 X/O/- cells")
        if side not in "XO":
            raise ValueError("side to move must be X or O")
        self.cells = list(cells)
        self.side = side

    @staticmethod
    def _inside(row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def legal_moves(self) -> dict[int, list[int]]:
        opponent = "O" if self.side == "X" else "X"
        legal: dict[int, list[int]] = {}
        for pos, cell in enumerate(self.cells):
            if cell != "-":
                continue
            row, col = divmod(pos, 8)
            all_flips: list[int] = []
            for drow, dcol in self.DIRECTIONS:
                nrow = row + drow
                ncol = col + dcol
                line: list[int] = []
                while self._inside(nrow, ncol):
                    npos = nrow * 8 + ncol
                    if self.cells[npos] != opponent:
                        break
                    line.append(npos)
                    nrow += drow
                    ncol += dcol
                if (
                    line
                    and self._inside(nrow, ncol)
                    and self.cells[nrow * 8 + ncol] == self.side
                ):
                    all_flips.extend(line)
            if all_flips:
                legal[pos] = all_flips
        return legal

    def pass_turn(self) -> None:
        self.side = "O" if self.side == "X" else "X"

    def play(self, coord: str) -> None:
        coord = coord.lower()
        if not re.fullmatch(r"[a-h][1-8]", coord):
            raise ValueError(f"invalid move coordinate: {coord}")
        col = ord(coord[0]) - ord("a")
        row = int(coord[1]) - 1
        pos = row * 8 + col
        legal = self.legal_moves()
        if pos not in legal:
            raise ValueError(f"illegal move {coord} for {self.to_string()}")
        self.cells[pos] = self.side
        for flipped in legal[pos]:
            self.cells[flipped] = self.side
        self.pass_turn()

    def empties(self) -> int:
        return self.cells.count("-")

    def to_string(self) -> str:
        return "".join(self.cells) + " " + self.side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract GGS positions and benchmark endgame completion from a "
            "completely cold transposition table."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--games",
        nargs="+",
        help=(
            "Game-record files, directories, or glob patterns. Directories "
            "are searched recursively for *.txt. If omitted, use the "
            "bundled cold-endgame corpus in bin/problem."
        ),
    )
    source.add_argument(
        "--positions",
        type=Path,
        help="Reuse corpus.jsonl or a plain file containing one board per line.",
    )
    parser.add_argument("--min-empty", type=int, default=32)
    parser.add_argument("--max-empty", type=int, default=44)
    parser.add_argument(
        "--player",
        default="egrcd",
        help=(
            "When extracting --games, only positions where this GGS player "
            "is to move, or 'all'."
        ),
    )
    parser.add_argument(
        "--max-per-empty",
        type=int,
        default=5,
        help="Maximum sampled positions for each empties count; 0 means all.",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--extract-only", action="store_true")

    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--hash", dest="hash_level", type=int, default=29)
    parser.add_argument("--movetime-ms", type=int, default=15000)
    parser.add_argument(
        "--required-selectivity",
        type=float,
        default=74.0,
        help=(
            "Minimum completed end-search selectivity counted as a solve. "
            "This classifies results; it does not change the engine schedule."
        ),
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=0.0,
        help="Hard wall timeout per process; 0 chooses movetime + 90 seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New output directory. Default: bin/benchmark_results/<timestamp>.",
    )
    args = parser.parse_args()

    if not 0 <= args.min_empty <= args.max_empty <= 60:
        parser.error("empty range must satisfy 0 <= min <= max <= 60")
    if args.max_per_empty < 0:
        parser.error("--max-per-empty must be nonnegative")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if not 0 <= args.hash_level <= 29:
        parser.error("--hash must be between 0 and 29")
    if args.movetime_ms <= 0:
        parser.error("--movetime-ms must be positive")
    if not 0.0 <= args.required_selectivity <= 100.0:
        parser.error("--required-selectivity must be between 0 and 100")
    if args.case_timeout_seconds < 0.0:
        parser.error("--case-timeout-seconds must be nonnegative")
    return args


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SCRIPT_DIR / path).resolve()


def expand_game_inputs(inputs: Optional[list[str]]) -> list[Path]:
    if not inputs:
        inputs = [str(DEFAULT_GAME_DIR)]
    found: set[Path] = set()
    for raw in inputs:
        raw_path = Path(raw)
        candidate = resolve_input_path(raw_path)
        if candidate.is_dir():
            found.update(path.resolve() for path in candidate.rglob("*.txt"))
            continue
        if candidate.is_file():
            found.add(candidate.resolve())
            continue

        # On Windows the shell normally does not expand globs, so expand them
        # here. Try the current directory first and then the script directory.
        patterns = [str(Path.cwd() / raw), str(SCRIPT_DIR / raw)]
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern, recursive=True))
        for match in matches:
            path = Path(match)
            if path.is_dir():
                found.update(item.resolve() for item in path.rglob("*.txt"))
            elif path.is_file():
                found.add(path.resolve())
    return sorted(found, key=lambda path: str(path).lower())


def parse_game_record(path: Path) -> tuple[str, str, str, str, str]:
    fields: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    game_id = lines[0].strip() if lines else path.stem
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    required = ("black", "white", "initial board", "transcript")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError("missing field(s): " + ", ".join(missing))
    board = fields["initial board"].upper()
    if not BOARD_RE.fullmatch(board):
        raise ValueError("invalid initial board")
    return (
        game_id,
        fields["black"],
        fields["white"],
        board,
        fields["transcript"],
    )


def actor_for_side(side: str, black_name: str, white_name: str) -> str:
    return black_name if side == "X" else white_name


def position_matches_player(actor: str, player_filter: str) -> bool:
    return player_filter.lower() == "all" or actor.lower() == player_filter.lower()


def extract_positions_from_game(
    path: Path,
    min_empty: int,
    max_empty: int,
    player_filter: str,
) -> list[Position]:
    game_id, black_name, white_name, board_text, transcript = parse_game_record(path)
    cells, side = board_text.split()
    board = OthelloBoard(cells, side)
    if len(transcript) % 2:
        raise ValueError("transcript length is not even")
    tokens = [transcript[idx : idx + 2].lower() for idx in range(0, len(transcript), 2)]
    positions: list[Position] = []
    actual_ply = 0

    for token in tokens:
        legal = board.legal_moves()
        if token == "ps":
            if legal:
                raise ValueError(f"explicit pass is illegal at transcript ply {actual_ply}")
            board.pass_turn()
            continue

        n_passes = 0
        while not legal:
            board.pass_turn()
            n_passes += 1
            legal = board.legal_moves()
            if n_passes >= 2 and not legal:
                raise ValueError("transcript continues after game end")

        empties = board.empties()
        actor = actor_for_side(board.side, black_name, white_name)
        if (
            min_empty <= empties <= max_empty
            and position_matches_player(actor, player_filter)
        ):
            positions.append(
                Position(
                    board=board.to_string(),
                    empties=empties,
                    source=str(path),
                    ply=actual_ply,
                    side=board.side,
                    actor=actor,
                    game_id=game_id,
                )
            )

        board.play(token)
        actual_ply += 1
    return positions


def load_positions(path: Path, min_empty: int, max_empty: int) -> list[Position]:
    path = resolve_input_path(path)
    positions: list[Position] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            data = json.loads(line)
            board_text = str(data["board"]).upper()
            source = str(data.get("source", path))
            ply = int(data.get("ply", line_number - 1))
            actor = str(data.get("actor", "unknown"))
            game_id = str(data.get("game_id", Path(source).stem))
        else:
            board_text = line.upper()
            source = str(path)
            ply = line_number - 1
            actor = "unknown"
            game_id = path.stem
        if not BOARD_RE.fullmatch(board_text):
            raise ValueError(f"{path}:{line_number}: invalid board")
        cells, side = board_text.split()
        empties = cells.count("-")
        if min_empty <= empties <= max_empty:
            positions.append(
                Position(
                    board=board_text,
                    empties=empties,
                    source=source,
                    ply=ply,
                    side=side,
                    actor=actor,
                    game_id=game_id,
                )
            )
    return positions


def deduplicate_positions(positions: Iterable[Position]) -> list[Position]:
    by_board: dict[str, Position] = {}
    for position in positions:
        by_board.setdefault(position.board, position)
    return list(by_board.values())


def sample_positions(
    positions: Iterable[Position], max_per_empty: int, seed: int
) -> list[Position]:
    grouped: dict[int, list[Position]] = defaultdict(list)
    for position in positions:
        grouped[position.empties].append(position)
    rng = random.Random(seed)
    selected: list[Position] = []
    for empties in sorted(grouped, reverse=True):
        candidates = sorted(
            grouped[empties], key=lambda pos: (pos.source.lower(), pos.ply, pos.board)
        )
        if max_per_empty and len(candidates) > max_per_empty:
            candidates = rng.sample(candidates, max_per_empty)
            candidates.sort(key=lambda pos: (pos.source.lower(), pos.ply, pos.board))
        selected.extend(candidates)
    return selected


def create_output_dir(requested: Optional[Path]) -> Path:
    if requested is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = SCRIPT_DIR / "benchmark_results" / f"cold_endgame_{stamp}"
    else:
        output = requested if requested.is_absolute() else (Path.cwd() / requested)
        output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    return output


def write_corpus(output: Path, positions: list[Position]) -> None:
    with (output / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for position in positions:
            file.write(json.dumps(asdict(position), ensure_ascii=False) + "\n")
    with (output / "corpus.txt").open("w", encoding="utf-8", newline="\n") as file:
        for position in positions:
            file.write(position.board + "\n")


def parse_duration_ms(value: str) -> Optional[int]:
    match = re.fullmatch(r"(\d+):(\d+):(\d+)(?:\.(\d+))?", value.strip())
    if not match:
        return None
    hours, minutes, seconds = (int(match.group(idx)) for idx in range(1, 4))
    fraction = match.group(4) or ""
    milliseconds = int((fraction + "000")[:3])
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def parse_result_table(stdout: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in stdout.splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.split("|")[1:-1]]
        if len(fields) != 7 or fields[0].lower() == "level":
            continue
        depth_match = DEPTH_RE.fullmatch(fields[1])
        if not depth_match:
            continue
        try:
            parsed = {
                "result_depth": int(depth_match.group(1)),
                "result_selectivity": float(depth_match.group(2)),
                "move": fields[2],
                "score": int(fields[3]),
                "engine_time_ms": parse_duration_ms(fields[4]),
                "nodes": int(fields[5]),
                "nps": int(fields[6]),
            }
        except ValueError:
            continue
    return parsed


def parse_end_iterations(
    stderr: str, empties: int, required_selectivity: float
) -> dict[str, object]:
    end_attempted = False
    end_completed = False
    exact_completed = False
    last_mid_time_ms: Optional[int] = None
    end_start_time_ms: Optional[int] = None
    end_last_time_ms: Optional[int] = None
    first_end_selectivity: Optional[float] = None
    first_end_time_ms: Optional[int] = None
    first_exact_time_ms: Optional[int] = None
    for line in stderr.splitlines():
        mid_match = MID_LINE_RE.search(line)
        if mid_match:
            tail = mid_match.group(3)
            time_match = END_TIME_RE.search(tail)
            if "value" in tail.lower() and time_match:
                last_mid_time_ms = int(time_match.group(1))
            continue
        match = END_LINE_RE.search(line)
        if not match:
            continue
        depth = int(match.group(1))
        selectivity = float(match.group(2))
        tail = match.group(3)
        if depth != empties:
            continue
        end_attempted = True
        if end_start_time_ms is None:
            end_start_time_ms = last_mid_time_ms or 0
        time_match = END_TIME_RE.search(tail)
        terminated_match = TERMINATED_TIME_RE.search(tail)
        if time_match:
            end_last_time_ms = int(time_match.group(1))
        elif terminated_match:
            end_last_time_ms = int(terminated_match.group(1))
        # A completed root search can be followed on the same log line by an
        # auxiliary policy-verification timeout (for example,
        # "value ... narrow-alt@88% terminated").  Only a root tail that
        # starts with "terminated" means the end iteration itself timed out.
        if tail.lstrip().lower().startswith("terminated") or "value" not in tail.lower():
            continue
        if not time_match:
            continue
        iteration_time_ms = int(time_match.group(1))
        if selectivity + 1e-9 >= required_selectivity and not end_completed:
            end_completed = True
            first_end_selectivity = selectivity
            first_end_time_ms = iteration_time_ms
        if selectivity + 1e-9 >= 100.0 and not exact_completed:
            exact_completed = True
            first_exact_time_ms = iteration_time_ms
    end_search_time_ms = None
    if end_start_time_ms is not None and end_last_time_ms is not None:
        end_search_time_ms = max(0, end_last_time_ms - end_start_time_ms)
    first_end_search_time_ms = None
    if end_start_time_ms is not None and first_end_time_ms is not None:
        first_end_search_time_ms = max(0, first_end_time_ms - end_start_time_ms)
    return {
        "end_attempted": end_attempted,
        "end_completed": end_completed,
        "end_start_time_ms": end_start_time_ms,
        "end_search_time_ms": end_search_time_ms,
        "first_end_selectivity": first_end_selectivity,
        "first_end_time_ms": first_end_time_ms,
        "first_end_search_time_ms": first_end_search_time_ms,
        "exact_completed": exact_completed,
        "first_exact_time_ms": first_exact_time_ms,
    }


def get_process_cpu_time_ms(process: Optional[subprocess.Popen[str]]) -> Optional[int]:
    """Return child user+kernel CPU time while its Windows handle is valid."""
    if process is None or sys.platform != "win32" or not hasattr(process, "_handle"):
        return None

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    creation = FileTime()
    exit_time = FileTime()
    kernel = FileTime()
    user = FileTime()
    get_process_times = ctypes.windll.kernel32.GetProcessTimes
    get_process_times.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    ]
    get_process_times.restype = ctypes.c_int
    if not get_process_times(
        ctypes.c_void_p(int(process._handle)),
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        return None

    def ticks(value: FileTime) -> int:
        return (int(value.high) << 32) | int(value.low)

    return round((ticks(kernel) + ticks(user)) / 10_000)


def get_engine_version(exe: Path) -> str:
    try:
        completed = subprocess.run(
            [str(exe), "-v"],
            cwd=str(exe.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"
    return completed.stdout.strip() or completed.stderr.strip()


def run_case(
    index: int,
    position: Position,
    exe: Path,
    threads: int,
    hash_level: int,
    movetime_ms: int,
    required_selectivity: float,
    timeout_seconds: float,
) -> tuple[SearchResult, str, str]:
    command = [
        str(exe),
        "-movetime",
        str(movetime_ms),
        "-hash",
        str(hash_level),
        "-nobook",
        "-nocontestbook",
        "-t",
        str(threads),
        "-seed",
        "0",
        "-noise",
        "-noboard",
    ]
    engine_input = f"setboard {position.board}\ngo\nquit\n"
    start = time.perf_counter()
    stdout = ""
    stderr = ""
    return_code: Optional[int] = None
    timed_out = False
    error_text = ""
    process: Optional[subprocess.Popen[str]] = None
    cpu_time_ms: Optional[int] = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(exe.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = process.communicate(engine_input, timeout=timeout_seconds)
        cpu_time_ms = get_process_cpu_time_ms(process)
        return_code = process.returncode
        if return_code != 0:
            error_text = f"engine exited with code {return_code}"
    except subprocess.TimeoutExpired:
        timed_out = True
        error_text = f"process exceeded {timeout_seconds:.1f}s wall timeout"
        if process is not None:
            process.kill()
            stdout, stderr = process.communicate()
            cpu_time_ms = get_process_cpu_time_ms(process)
            return_code = process.returncode
    except KeyboardInterrupt:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    except OSError as error:
        error_text = str(error)
    wall_time_ms = round((time.perf_counter() - start) * 1000)
    cpu_average_cores = (
        cpu_time_ms / wall_time_ms
        if cpu_time_ms is not None and wall_time_ms > 0
        else None
    )
    cpu_utilization_percent = (
        cpu_average_cores / max(1, threads) * 100.0
        if cpu_average_cores is not None
        else None
    )

    table = parse_result_table(stdout)
    end = parse_end_iterations(stderr, position.empties, required_selectivity)
    if not table and not error_text:
        error_text = "search result table was not found"
    result = SearchResult(
        index=index,
        board=position.board,
        empties=position.empties,
        source=position.source,
        ply=position.ply,
        side=position.side,
        actor=position.actor,
        game_id=position.game_id,
        return_code=return_code,
        timed_out=timed_out,
        error=error_text,
        result_depth=table.get("result_depth"),
        result_selectivity=table.get("result_selectivity"),
        move=str(table.get("move", "")),
        score=table.get("score"),
        engine_time_ms=table.get("engine_time_ms"),
        nodes=table.get("nodes"),
        nps=table.get("nps"),
        wall_time_ms=wall_time_ms,
        cpu_time_ms=cpu_time_ms,
        cpu_average_cores=cpu_average_cores,
        cpu_utilization_percent=cpu_utilization_percent,
        end_attempted=bool(end["end_attempted"]),
        end_completed=bool(end["end_completed"]),
        end_start_time_ms=end["end_start_time_ms"],
        end_search_time_ms=end["end_search_time_ms"],
        first_end_selectivity=end["first_end_selectivity"],
        first_end_time_ms=end["first_end_time_ms"],
        first_end_search_time_ms=end["first_end_search_time_ms"],
        exact_completed=bool(end["exact_completed"]),
        first_exact_time_ms=end["first_exact_time_ms"],
    )
    return result, stdout, stderr


def percentile(values: list[int], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = fraction * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_group(results: list[SearchResult]) -> dict[str, object]:
    completed = [result for result in results if result.end_completed]
    attempted = [result for result in results if result.end_attempted]
    exact = [result for result in results if result.exact_completed]
    end_times = [
        result.first_end_time_ms
        for result in completed
        if result.first_end_time_ms is not None
    ]
    valid = [result for result in results if not result.error]
    nodes = [result.nodes for result in completed if result.nodes is not None]
    valid_nodes = [result.nodes for result in valid if result.nodes is not None]
    nps = [result.nps for result in valid if result.nps is not None]
    completed_nps = [result.nps for result in completed if result.nps is not None]
    end_start_times = [
        result.end_start_time_ms
        for result in attempted
        if result.end_start_time_ms is not None
    ]
    end_search_times = [
        result.end_search_time_ms
        for result in attempted
        if result.end_search_time_ms is not None
    ]
    first_end_search_times = [
        result.first_end_search_time_ms
        for result in completed
        if result.first_end_search_time_ms is not None
    ]
    cpu_average_cores = [
        result.cpu_average_cores
        for result in valid
        if result.cpu_average_cores is not None
    ]
    cpu_utilization = [
        result.cpu_utilization_percent
        for result in valid
        if result.cpu_utilization_percent is not None
    ]
    return {
        "count": len(results),
        "valid": len(valid),
        "attempted": sum(result.end_attempted for result in results),
        "completed": len(completed),
        "completion_rate": len(completed) / len(results) if results else 0.0,
        "exact_completed": len(exact),
        "first_end_time_median_ms": statistics.median(end_times) if end_times else None,
        "first_end_time_p90_ms": percentile(end_times, 0.90),
        "nodes_median": statistics.median(nodes) if nodes else None,
        "valid_nodes_median": statistics.median(valid_nodes) if valid_nodes else None,
        "nps_median": statistics.median(nps) if nps else None,
        "completed_nps_median": statistics.median(completed_nps) if completed_nps else None,
        "end_start_time_median_ms": statistics.median(end_start_times) if end_start_times else None,
        "end_start_time_p90_ms": percentile(end_start_times, 0.90),
        "end_search_time_median_ms": statistics.median(end_search_times) if end_search_times else None,
        "end_search_time_p90_ms": percentile(end_search_times, 0.90),
        "first_end_search_time_median_ms": (
            statistics.median(first_end_search_times) if first_end_search_times else None
        ),
        "first_end_search_time_p90_ms": percentile(first_end_search_times, 0.90),
        "cpu_average_cores_median": (
            statistics.median(cpu_average_cores) if cpu_average_cores else None
        ),
        "cpu_utilization_percent_median": (
            statistics.median(cpu_utilization) if cpu_utilization else None
        ),
    }


def build_summary(results: list[SearchResult], metadata: dict[str, object]) -> dict[str, object]:
    grouped: dict[int, list[SearchResult]] = defaultdict(list)
    for result in results:
        grouped[result.empties].append(result)
    return {
        "metadata": metadata,
        "overall": summarize_group(results),
        "by_empties": {
            str(empties): summarize_group(grouped[empties])
            for empties in sorted(grouped, reverse=True)
        },
    }


def format_optional_number(value: object, digits: int = 0) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_summary(summary: dict[str, object]) -> None:
    print("\nCold-TT endgame benchmark summary")
    print("empty  cases  attempted  completed  rate    exact  median_ms  p90_ms")
    by_empties = summary["by_empties"]
    assert isinstance(by_empties, dict)
    for empty_text, raw_stats in by_empties.items():
        stats = raw_stats
        assert isinstance(stats, dict)
        print(
            f"{int(empty_text):5d}"
            f"  {int(stats['count']):5d}"
            f"  {int(stats['attempted']):9d}"
            f"  {int(stats['completed']):9d}"
            f"  {float(stats['completion_rate']):6.1%}"
            f"  {int(stats['exact_completed']):5d}"
            f"  {format_optional_number(stats['first_end_time_median_ms']):>9}"
            f"  {format_optional_number(stats['first_end_time_p90_ms'], 1):>6}"
        )
    overall = summary["overall"]
    assert isinstance(overall, dict)
    print(
        "overall: "
        f"{overall['completed']}/{overall['count']} completed "
        f"({float(overall['completion_rate']):.1%}), "
        f"{overall['exact_completed']} exact, "
        f"median first-end {format_optional_number(overall['first_end_time_median_ms'])} ms"
    )
    print(
        "timing: median end-start "
        f"{format_optional_number(overall['end_start_time_median_ms'])} ms, "
        "end-search "
        f"{format_optional_number(overall['end_search_time_median_ms'])} ms, "
        "NPS "
        f"{format_optional_number(overall['nps_median'])}, "
        "CPU "
        f"{format_optional_number(overall['cpu_utilization_percent_median'], 1)}% "
        "of requested threads"
    )


def write_results_csv(output: Path, results: list[SearchResult]) -> None:
    fieldnames = list(SearchResult.__dataclass_fields__)
    with (output / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_run_config(output: Path, metadata: dict[str, object]) -> None:
    (output / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output = create_output_dir(args.output)

    extraction_errors: list[str] = []
    if args.positions:
        positions = load_positions(args.positions, args.min_empty, args.max_empty)
        source_description: object = str(resolve_input_path(args.positions))
    elif args.games:
        game_files = expand_game_inputs(args.games)
        if not game_files:
            print("No GGS game-record files found.", file=sys.stderr)
            return 2
        extracted: list[Position] = []
        for path in game_files:
            try:
                extracted.extend(
                    extract_positions_from_game(
                        path, args.min_empty, args.max_empty, args.player
                    )
                )
            except (OSError, ValueError) as error:
                extraction_errors.append(f"{path}: {error}")
        positions = extracted
        source_description = [str(path) for path in game_files]
    else:
        if not DEFAULT_CORPUS.is_file():
            print(f"Bundled corpus not found: {DEFAULT_CORPUS}", file=sys.stderr)
            return 2
        positions = load_positions(DEFAULT_CORPUS, args.min_empty, args.max_empty)
        source_description = str(DEFAULT_CORPUS)

    positions = deduplicate_positions(positions)
    positions = sample_positions(positions, args.max_per_empty, args.seed)
    if not positions:
        print("No positions matched the requested filters.", file=sys.stderr)
        return 2

    write_corpus(output, positions)
    if extraction_errors:
        (output / "extraction_errors.txt").write_text(
            "\n".join(extraction_errors) + "\n", encoding="utf-8"
        )

    exe = resolve_input_path(args.exe)
    metadata: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "cold_tt_method": "fresh process per position",
        "source": source_description,
        "position_count": len(positions),
        "min_empty": args.min_empty,
        "max_empty": args.max_empty,
        "player": args.player,
        "max_per_empty": args.max_per_empty,
        "sampling_seed": args.seed,
        "extraction_error_count": len(extraction_errors),
        "executable": str(exe),
        "threads": args.threads,
        "hash_level": args.hash_level,
        "movetime_ms": args.movetime_ms,
        "required_selectivity": args.required_selectivity,
        "case_timeout_seconds": args.case_timeout_seconds,
    }

    print(f"Corpus: {len(positions)} positions")
    counts: dict[int, int] = defaultdict(int)
    for position in positions:
        counts[position.empties] += 1
    print(
        "By empties: "
        + ", ".join(f"{empty}:{counts[empty]}" for empty in sorted(counts, reverse=True))
    )
    print(f"Output: {output}")

    if args.extract_only:
        metadata["extract_only"] = True
        write_run_config(output, metadata)
        print("Extraction only; searches were not run.")
        return 0

    if not exe.is_file():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 2
    metadata["engine_version"] = get_engine_version(exe)
    write_run_config(output, metadata)

    timeout_seconds = args.case_timeout_seconds
    if timeout_seconds <= 0.0:
        timeout_seconds = args.movetime_ms / 1000.0 + 90.0

    results: list[SearchResult] = []
    stdout_path = output / "engine_stdout.log"
    stderr_path = output / "engine_stderr.log"
    print(
        f"Running fresh process per position: {args.movetime_ms} ms, "
        f"{args.threads} threads, hash {args.hash_level}"
    )
    try:
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_file, \
             stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_file:
            for zero_index, position in enumerate(positions):
                index = zero_index + 1
                result, stdout, stderr = run_case(
                    index,
                    position,
                    exe,
                    args.threads,
                    args.hash_level,
                    args.movetime_ms,
                    args.required_selectivity,
                    timeout_seconds,
                )
                results.append(result)
                header = (
                    f"===== CASE {index}/{len(positions)} empties={position.empties} "
                    f"source={position.source} ply={position.ply} =====\n"
                )
                stdout_file.write(header)
                stdout_file.write(stdout)
                if stdout and not stdout.endswith("\n"):
                    stdout_file.write("\n")
                stdout_file.flush()
                stderr_file.write(header)
                stderr_file.write(stderr)
                if stderr and not stderr.endswith("\n"):
                    stderr_file.write("\n")
                stderr_file.flush()
                status = "SOLVED" if result.end_completed else "not solved"
                if result.error:
                    status = "ERROR: " + result.error
                first_time = (
                    f"{result.first_end_time_ms} ms"
                    if result.first_end_time_ms is not None
                    else "-"
                )
                print(
                    f"[{index:3d}/{len(positions)}] empty {position.empties:2d} "
                    f"{status}; first end {first_time}",
                    flush=True,
                )
                write_results_csv(output, results)
    except KeyboardInterrupt:
        print("\nInterrupted; partial results were saved.", file=sys.stderr)

    summary = build_summary(results, metadata)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_results_csv(output, results)
    print_summary(summary)
    print(f"Results: {output / 'results.csv'}")
    print(f"Summary: {output / 'summary.json'}")
    return 0 if results else 130


if __name__ == "__main__":
    raise SystemExit(main())
