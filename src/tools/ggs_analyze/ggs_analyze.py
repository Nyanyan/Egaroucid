#!/usr/bin/env python3
"""Extract lost / drawn games from Egaroucid GGS game logs and analyze them
with the ``analyzefast`` command of Egaroucid for Console.

See README.md in this directory for usage (in Japanese).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENGINE = REPO_ROOT / "bin" / "Egaroucid_for_Console_ggs_analyze.exe"
DEFAULT_ME = ["egrcd"]
DEFAULT_HASH_LEVEL = 29
DEFAULT_OUT_DIR = REPO_ROOT / "bin" / "ggs" / "log" / "analyze"
VALID_PROBS = ("74", "88", "93", "98", "99", "99.9", "100")
CACHE_VERSION = 1
PASS_TOKENS = {"pa", "ps", "--"}
WINDOWS_CTRL_C_EXIT_CODES = {0xC000013A, -0x3FFFFEC6}  # STATUS_CONTROL_C_EXIT
START_BOARD = "---------------------------OX------XO--------------------------- X"

LOG_FIELD_RE = re.compile(r"^(black|white|initial board|transcript|black's score):\s*(.*)$")
FILE_DATETIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})=(\d{2})=(\d{2})")
PROMPT_RE = re.compile(r"^(?:>\s*)+")
# Hash levels without resources/hash/hashN.eghs (e.g. 29) fall back to random hash numbers
BENIGN_ENGINE_ERROR_RE = re.compile(r"you can ignore this error|can't open hash\d+\.eghs")
ANALYZE_ROW_RE = re.compile(
    r"^\|\s*(\d+)\|\s*(Black|White)\|\s*([a-h][1-8])\|\s*([^|]*?)\s*\|\s*([+-]?\d+)\|"
    r"\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|"
)

PRINT_LOCK = threading.Lock()
ENGINE_WAIT_SECONDS = 30


def log(message: str = "") -> None:
    with PRINT_LOCK:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# Othello rules
# ---------------------------------------------------------------------------

DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def other(side: str) -> str:
    return "O" if side == "X" else "X"


def flips_for(cells: list[str], side: str, move: int) -> list[int]:
    if cells[move] != "-":
        return []
    opponent = other(side)
    row, column = divmod(move, 8)
    result: list[int] = []
    for dr, dc in DIRECTIONS:
        r, c = row + dr, column + dc
        line: list[int] = []
        while 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == opponent:
            line.append(r * 8 + c)
            r += dr
            c += dc
        if line and 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == side:
            result.extend(line)
    return result


def legal_moves(cells: list[str], side: str) -> list[int]:
    return [move for move in range(64) if cells[move] == "-" and flips_for(cells, side, move)]


def coord_to_index(coord: str) -> int:
    if len(coord) != 2 or coord[0] not in "abcdefgh" or coord[1] not in "12345678":
        raise ValueError(f"invalid coordinate {coord!r}")
    return (ord(coord[1]) - ord("1")) * 8 + ord(coord[0]) - ord("a")


def index_to_coord(index: int) -> str:
    return "abcdefgh"[index % 8] + str(index // 8 + 1)


def final_score(cells: str, side: str) -> int:
    """Disc difference for ``side`` with empty squares given to the winner."""
    diff = cells.count(side) - cells.count(other(side))
    empties = cells.count("-")
    if diff > 0:
        diff += empties
    elif diff < 0:
        diff -= empties
    return diff


def board_diagram(cells: str, marks: dict[int, str]) -> str:
    lines = ["  a b c d e f g h"]
    for row in range(8):
        squares = []
        for column in range(8):
            index = row * 8 + column
            squares.append(marks.get(index, {"X": "X", "O": "O", "-": "."}[cells[index]]))
        lines.append(f"{row + 1} " + " ".join(squares))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GGS game logs
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    source: Path
    game_id: str
    black: str
    white: str
    initial_cells: str
    initial_side: str
    moves: list[str]
    black_score: int | None
    finished_at: datetime

    @property
    def dedup_key(self) -> tuple[str, str, str, str]:
        return (self.black, self.white, self.initial_cells + self.initial_side, "".join(self.moves))


@dataclass
class Position:
    index: int
    cells: str
    side: str
    move: int

    @property
    def empties(self) -> int:
        return self.cells.count("-")

    @property
    def n_discs(self) -> int:
        return 64 - self.empties

    @property
    def board_str(self) -> str:
        return self.cells + " " + self.side


@dataclass
class Replay:
    positions: list[Position]
    final_cells: str
    terminal: bool


def parse_file_datetime(path: Path) -> datetime:
    match = FILE_DATETIME_RE.search(path.name)
    if match:
        try:
            return datetime(*(int(value) for value in match.groups()))
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def parse_log_file(path: Path) -> list[GameRecord]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LOG_FIELD_RE.match(line)
        if match is None:
            current = {"id": line}
            blocks.append(current)
            continue
        key, value = match.group(1), match.group(2).strip()
        if current is None or key in current:
            current = {"id": ""}
            blocks.append(current)
        current[key] = value

    finished_at = parse_file_datetime(path)
    records: list[GameRecord] = []
    for block in blocks:
        if not {"black", "white", "initial board", "transcript"} <= block.keys():
            continue
        board = block["initial board"].replace(" ", "")
        if len(board) != 65:
            continue
        cells = board[:64].replace("*", "X")
        side = board[64].upper().replace("*", "X")
        if side not in "XO" or any(cell not in "XO-" for cell in cells):
            continue
        transcript = block["transcript"].replace(" ", "").lower()
        if len(transcript) % 2:
            continue
        moves = [transcript[i:i + 2] for i in range(0, len(transcript), 2)]
        moves = [move for move in moves if move not in PASS_TOKENS]
        black_score: int | None = None
        try:
            black_score = int(block.get("black's score", ""))
        except ValueError:
            black_score = None
        if black_score == -99:
            black_score = None
        records.append(GameRecord(
            source=path,
            game_id=block["id"],
            black=block["black"],
            white=block["white"],
            initial_cells=cells,
            initial_side=side,
            moves=moves,
            black_score=black_score,
            finished_at=finished_at,
        ))
    return records


def replay(record: GameRecord) -> Replay:
    cells = list(record.initial_cells)
    side = record.initial_side
    positions: list[Position] = []
    for coord in record.moves:
        legal = legal_moves(cells, side)
        if not legal:
            side = other(side)
            legal = legal_moves(cells, side)
            if not legal:
                raise ValueError(f"moves remain after the game ended ({coord} at move {len(positions) + 1})")
        move = coord_to_index(coord)
        if move not in legal:
            raise ValueError(f"illegal move {coord} at move {len(positions) + 1}")
        positions.append(Position(len(positions), "".join(cells), side, move))
        for square in flips_for(cells, side, move) + [move]:
            cells[square] = side
        side = other(side)
    terminal = not legal_moves(cells, side) and not legal_moves(cells, other(side))
    return Replay(positions, "".join(cells), terminal)


def collect_log_files(paths: list[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            pattern = "**/*.txt" if recursive else "*.txt"
            files.extend(sorted(p for p in path.glob(pattern) if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"[ERROR] ログが見つかりません: {path}")
    return files


# ---------------------------------------------------------------------------
# GGS client logs (in-game search of our own moves)
# ---------------------------------------------------------------------------

CLIENT_SEARCH_RE = re.compile(
    r"GGS INFO> search (ready|send pending|result|exact ponder|contest book|contest root table|hint only) "
    r"game (\S+) discs \d+ move ([a-h][1-8]) value (-?\d+) depth (\S+) time (\d+)\S*.* board ([XO-]{64} X)"
)
CLIENT_SEARCH_PRIORITY = {"ready": 3, "send pending": 2}  # the move actually decided; other kinds: 1
CLIENT_LOG_CANDIDATES = 3


def normalized_board(board_str: str) -> str:
    """The client log writes boards with the side to move as X."""
    cells, side = board_str[:64], board_str[-1]
    if side == "O":
        cells = cells.translate(str.maketrans("XO", "OX"))
    return cells + " X"


def find_client_logs(log_paths: list[Path], extra: list[Path], recursive: bool) -> list[Path]:
    """Client logs (*_ggs.log) next to the game logs: in the given directories and their parents."""
    found: set[Path] = set()
    for path in log_paths:
        directory = path if path.is_dir() else path.parent
        if path.is_dir():
            found.update(directory.glob("**/*_ggs.log" if recursive else "*_ggs.log"))
        found.update(directory.glob("*_ggs.log"))
        found.update(directory.parent.glob("*_ggs.log"))
    for path in extra:
        found.update(path.glob("**/*_ggs.log") if path.is_dir() else [path])
    return sorted(path.resolve() for path in found if path.is_file())


class ClientLogIndex:
    """Depth, value and time of the in-game search for our own moves, read from GGS client logs."""

    def __init__(self, paths: list[Path]) -> None:
        # (session start from the file name, last write time, path)
        self._files = sorted((parse_file_datetime(path), datetime.fromtimestamp(path.stat().st_mtime), str(path), path)
                             for path in paths)
        self._parsed: dict[Path, dict[tuple[str, str, str], tuple[int, dict]]] = {}

    def _parse(self, path: Path) -> dict[tuple[str, str, str], tuple[int, dict]]:
        data: dict[tuple[str, str, str], tuple[int, dict]] = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "INFO> search " not in line:
                    continue
                match = CLIENT_SEARCH_RE.search(line)
                if match is None:
                    continue
                kind, game_id, move, value, depth, time_ms, board = match.groups()
                priority = CLIENT_SEARCH_PRIORITY.get(kind, 1)
                key = (game_id, board, move)
                if key not in data or priority >= data[key][0]:
                    data[key] = (priority, {"depth": depth, "value": int(value), "time_ms": int(time_ms), "kind": kind})
        return data

    def annotate(self, record: GameRecord, moves: list[dict]) -> bool:
        """Set move["game"] for our moves, searching the client logs of sessions running when the game ended."""
        started = [(end, path) for start, end, _, path in reversed(self._files) if start <= record.finished_at]
        running = [path for end, path in started if end >= record.finished_at]
        candidates = running + [path for _, path in started if path not in running]
        for path in candidates[:CLIENT_LOG_CANDIDATES]:
            if path not in self._parsed:
                self._parsed[path] = self._parse(path)
            data = self._parsed[path]
            found = {}
            for move in moves:
                if move["is_me"]:
                    entry = data.get((record.game_id, normalized_board(move["board"]), move["played"]))
                    if entry is not None:
                        found[move["index"]] = entry[1]
            if found:
                for move in moves:
                    move["game"] = found.get(move["index"])
                return True
        for move in moves:
            move["game"] = None
        return False


# ---------------------------------------------------------------------------
# Game selection
# ---------------------------------------------------------------------------

@dataclass
class SelectedGame:
    record: GameRecord
    my_color: str
    opponent: str
    my_score: int
    reason: str
    match_no: int
    match_id: str
    match_score: int
    match_size: int

    @property
    def me(self) -> str:
        return self.record.black if self.my_color == "X" else self.record.white


def split_names(values: list[str] | None) -> set[str]:
    names: set[str] = set()
    for value in values or []:
        names.update(name.strip().casefold() for name in value.split(",") if name.strip())
    return names


def raw_names(values: list[str] | None) -> str:
    return ", ".join(name.strip() for value in values or [] for name in value.split(",") if name.strip())


def parse_datetime_arg(text: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"日時の形式が不正です: {text!r} (例: 2026-09-26, '2026-09-26 15:00')")


@dataclass
class SelectionStats:
    files: int = 0
    records: int = 0
    duplicates: int = 0
    not_mine: int = 0
    self_play: int = 0
    excluded: int = 0
    unknown_result: int = 0
    out_of_period: int = 0
    no_moves: int = 0
    not_target: int = 0


SYNCHRO_ID_RE = re.compile(r"^(.+)\.[01]$")
MATCH_TIME_TOLERANCE_SECONDS = 60


def match_base_id(game_id: str) -> str:
    """GGS synchro games are named .N.0 / .N.1 and form the match .N."""
    match = SYNCHRO_ID_RE.match(game_id)
    return match.group(1) if match else game_id


def group_matches(candidates: list[tuple[GameRecord, str, str]]) -> list[list[tuple[GameRecord, str, str]]]:
    """Group the two games of a synchro match (same base id and opponent, logged at the same time)."""
    groups: list[list[tuple[GameRecord, str, str]]] = []
    for candidate in candidates:
        record, _, opponent = candidate
        base_id = match_base_id(record.game_id)
        for group in groups:
            first, _, first_opponent = group[0]
            if (
                base_id != record.game_id
                and match_base_id(first.game_id) == base_id
                and first_opponent.casefold() == opponent.casefold()
                and all(other_record.game_id != record.game_id for other_record, _, _ in group)
                and abs((first.finished_at - record.finished_at).total_seconds()) <= MATCH_TIME_TOLERANCE_SECONDS
            ):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    return groups


def select_games(records: list[GameRecord], args: argparse.Namespace, stats: SelectionStats) -> list[SelectedGame]:
    me_names = split_names(args.me)
    vs_names = split_names(args.vs)
    excluded = split_names(args.exclude)
    seen: set[tuple[str, str, str, str]] = set()
    candidates: list[tuple[GameRecord, str, str]] = []
    for record in sorted(records, key=lambda r: (r.finished_at, str(r.source), r.game_id)):
        if record.dedup_key in seen:
            stats.duplicates += 1
            continue
        seen.add(record.dedup_key)
        black_is_me = record.black.casefold() in me_names
        white_is_me = record.white.casefold() in me_names
        if not black_is_me and not white_is_me:
            stats.not_mine += 1
            continue
        if black_is_me and white_is_me:
            stats.self_play += 1
            continue
        my_color = "X" if black_is_me else "O"
        opponent = record.white if black_is_me else record.black
        if opponent.casefold() in excluded:
            stats.excluded += 1
            continue
        if (args.since and record.finished_at < args.since) or (args.until and record.finished_at > args.until):
            stats.out_of_period += 1
            continue
        if not record.moves:
            stats.no_moves += 1
            continue
        candidates.append((record, my_color, opponent))

    groups = group_matches(candidates) if args.unit == "match" else [[candidate] for candidate in candidates]
    selected: list[SelectedGame] = []
    n_matches = 0
    for group in groups:
        if any(record.black_score is None for record, _, _ in group):
            stats.unknown_result += len(group)
            continue
        scores = [record.black_score if my_color == "X" else -record.black_score for record, my_color, _ in group]
        match_score = sum(scores)
        opponent = group[0][2]
        is_vs = opponent.casefold() in vs_names
        if args.vs_only and not is_vs:
            stats.not_target += len(group)
            continue
        if match_score < 0:
            reason = "負け"
        elif match_score == 0 and (is_vs or args.draws):
            reason = "引き分け"
        else:
            stats.not_target += len(group)
            continue
        n_matches += 1
        match_id = match_base_id(group[0][0].game_id) if args.unit == "match" else group[0][0].game_id
        for (record, my_color, _), my_score in sorted(zip(group, scores), key=lambda item: item[0][0].game_id):
            selected.append(SelectedGame(record, my_color, opponent, my_score, reason, n_matches, match_id, match_score, len(group)))
    return selected


# ---------------------------------------------------------------------------
# Search schedule (-dpr)
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    kind: str  # "mid", "end" or "exact"
    empties_hi: int
    empties_lo: int
    depth: int
    prob: str

    @property
    def move_range(self) -> tuple[int, int]:
        # Egaroucid numbers moves by n_discs - 3, i.e. 61 - empties
        return 61 - self.empties_hi, 61 - self.empties_lo

    def describe(self) -> str:
        span = f"{self.empties_hi}〜{self.empties_lo}空き" if self.empties_hi != self.empties_lo else f"{self.empties_hi}空き"
        if self.kind == "mid":
            return f"{span}: 中盤探索 深さ{self.depth} (MPC {self.prob}%)"
        if self.kind == "end":
            return f"{span}: 終盤読み切り (MPC {self.prob}%)"
        return f"{span}: 完全読み (100%)"


@dataclass
class Schedule:
    mid_depth: int
    mid_prob: str
    end_empties: int
    exact_empties: int
    end_probs: list[str]
    phases: list[Phase] = field(default_factory=list)

    def __post_init__(self) -> None:
        phases: list[Phase] = []
        if self.end_empties < 60:
            phases.append(Phase("mid", 60, self.end_empties + 1, self.mid_depth, self.mid_prob))
        n_end = self.end_empties - self.exact_empties
        n_probs = len(self.end_probs)
        if n_end > 0 and n_probs > 0:
            base, remainder = divmod(n_end, n_probs)
            sizes = [base + (1 if i >= n_probs - remainder else 0) for i in range(n_probs)]
            empties_hi = self.end_empties
            for prob, size in zip(self.end_probs, sizes):
                if size == 0:
                    continue
                phases.append(Phase("end", empties_hi, empties_hi - size + 1, 60, prob))
                empties_hi -= size
        if self.exact_empties >= 1:
            phases.append(Phase("exact", self.exact_empties, 1, 60, "100"))
        self.phases = phases

    def dpr_args(self) -> list[str]:
        args: list[str] = []
        for phase in self.phases:
            move_start, move_end = phase.move_range
            args += ["-dpr", str(move_start), str(move_end), str(phase.depth), phase.prob]
        return args


# ---------------------------------------------------------------------------
# Egaroucid for Console
# ---------------------------------------------------------------------------

class EngineError(RuntimeError):
    pass


@dataclass
class AnalyzeRow:
    ply: int
    player: str
    played: str
    played_depth: str
    played_score: int
    alt_move: str | None
    alt_depth: str
    alt_score: int | None
    seconds: float


def parse_analyze_row(line: str) -> AnalyzeRow | None:
    match = ANALYZE_ROW_RE.match(line)
    if match is None:
        return None
    ply, player, played, played_depth, played_score, alt_move, alt_depth, alt_score = match.groups()
    has_alt = re.fullmatch(r"[a-h][1-8]", alt_move) is not None
    return AnalyzeRow(
        ply=int(ply),
        player=player,
        played=played,
        played_depth=played_depth,
        played_score=int(played_score),
        alt_move=alt_move if has_alt else None,
        alt_depth=alt_depth if has_alt else "",
        alt_score=int(alt_score) if has_alt else None,
        seconds=0.0,
    )


class Engine:
    def __init__(self, cmd: list[str], cwd: Path) -> None:
        self.cmd = cmd
        self.errors: list[str] = []
        self.progress = lambda: ""
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.output: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self.output.put(line)
        finally:
            self.output.put(None)

    def send(self, line: str) -> None:
        if self.proc.stdin is None or self.proc.poll() is not None:
            raise EngineError("Egaroucid for Console が終了しています")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def readline(self, on_wait=None) -> str:
        while True:
            try:
                line = self.output.get(timeout=ENGINE_WAIT_SECONDS)
                break
            except queue.Empty:
                if on_wait is not None:
                    on_wait()
        if line is None:
            raise EngineError("Egaroucid for Console が予期せず終了しました")
        return PROMPT_RE.sub("", line.rstrip("\r\n"))

    def analyze(self, board_str: str, moves: str, on_row, label: str = "") -> list[AnalyzeRow]:
        self.send("setboard " + board_str)
        self.send("play " + moves)
        self.send("analyzefast")
        rows: list[AnalyzeRow] = []
        last_time = time.perf_counter()
        in_summary = False
        self.errors = []
        next_empties = board_str.count('-') - len(moves) // 2 + 1

        def on_wait():
            if not in_summary:
                log(f"  {label} 解析中: {next_empties}マス空き、経過{time.perf_counter() - last_time:.0f}秒 {self.progress()}")

        while True:
            line = self.readline(on_wait)
            if "[ERROR]" in line:
                # a failed setboard / play shows up as missing or mismatched rows, checked by the caller
                if not BENIGN_ENGINE_ERROR_RE.search(line):
                    self.errors.append(line.strip())
                    log(f"  [Egaroucid] {line.strip()}")
                continue
            if in_summary:
                if line.startswith("|") and "White" in line:
                    return rows
                continue
            if "Mistake Rate" in line:
                in_summary = True
                continue
            row = parse_analyze_row(line)
            if row is not None:
                now = time.perf_counter()
                row.seconds = now - last_time
                last_time = now
                next_empties = 62 - row.ply
                rows.append(row)
                on_row(row)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("exit")
                self.proc.wait(timeout=10)
            except (EngineError, OSError, subprocess.TimeoutExpired):
                self.proc.kill()

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()


def engine_version(engine: Path) -> str:
    try:
        result = subprocess.run([str(engine), "-v"], cwd=engine.parent, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "(不明)"
    for line in result.stdout.splitlines():
        if line.strip():
            return line.strip()
    return "(不明)"


def check_engine_supports_analyze_dpr(engine: Path) -> None:
    """Verify best-first analysis and the requested depth/probability."""
    cmd = [str(engine), "-quiet", "-nobook", "-nocontestbook", "-t", "1", "-hash", "18", "-dpr", "1", "60", "3", "88"]
    try:
        result = subprocess.run(cmd, cwd=engine.parent, input=f"setboard {START_BOARD}\nplay f5\nanalyzefast\nexit\n",
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"[ERROR] エンジンを起動できません: {engine} ({e})")
    rows = [parse_analyze_row(PROMPT_RE.sub("", line)) for line in result.stdout.splitlines()]
    rows = [row for row in rows if row is not None]
    if not rows or rows[0].played_depth != "3@88%":
        raise SystemExit(
            f"[ERROR] {engine} は analyzefast と -dpr (深さ・確率指定) に対応していません。\n"
            "        最新のソースから Egaroucid for Console をビルドし直してください (README.md 参照)。"
        )


# ---------------------------------------------------------------------------
# Analysis of one game
# ---------------------------------------------------------------------------

def game_cache_key(game: SelectedGame, schedule: Schedule, use_book: bool) -> str:
    record = game.record
    payload = {
        "version": CACHE_VERSION,
        "board": record.initial_cells + " " + record.initial_side,
        "moves": "".join(record.moves),
        "black": record.black,
        "white": record.white,
        "dpr": schedule.dpr_args(),
        "book": use_book,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_path(out_dir: Path, game: SelectedGame, key: str) -> Path:
    game_id = re.sub(r"[^0-9A-Za-z_.-]", "_", game.record.game_id).strip("._") or "game"
    stamp = game.record.finished_at.strftime("%Y%m%d-%H%M%S")
    return out_dir / "cache" / f"{stamp}_{game_id}_{key[:12]}.json"


def color_name(side: str) -> str:
    return "黒" if side == "X" else "白"


def signed(value: int | None) -> str:
    if value is None:
        return "-"
    return f"+{value}" if value > 0 else str(value)


def schedule_cache_key(schedule: Schedule, use_book: bool) -> str:
    payload = {"version": CACHE_VERSION, "dpr": schedule.dpr_args(), "book": use_book}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


ANALYSIS_FIELDS = ("played_score", "played_depth", "alt_move", "alt_score", "alt_depth")


class PositionTable:
    """Analysis results of already analyzed positions, keyed by (board, played move).

    Every result is appended to a JSONL file as soon as the engine reports it,
    so an interrupted run resumes from the last analyzed move. The two games
    of a synchro match start from the same position and often share their
    first moves, so the second game also reuses the results of the first.
    """

    def __init__(self, path: Path | None = None, load: bool = True) -> None:
        self._data: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()
        self._file = None
        if path is not None:
            if load and path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        entry = json.loads(line)
                        self._data[(entry["board"], entry["played"])] = entry
                    except (ValueError, KeyError, TypeError):
                        pass  # e.g. a line cut off by a crash
            self._file = open(path, "a", encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def get(self, board: str, played: str) -> dict | None:
        with self._lock:
            return self._data.get((board, played))

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._data.values())

    def add(self, board: str, played: str, analysis: dict, seconds: float, source: str) -> None:
        entry = {"board": board, "played": played, **{field: analysis[field] for field in ANALYSIS_FIELDS},
                 "seconds": round(seconds, 3), "source": source}
        with self._lock:
            self._data[(board, played)] = entry
            if self._file is not None:
                self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._file.flush()

    def add_game(self, moves: list[dict], source: str) -> None:
        """Load the moves of a game analyzed by an older version (kept in memory only)."""
        with self._lock:
            for move in moves:
                entry = {field: move[field] for field in ANALYSIS_FIELDS}
                entry.update(board=move["board"], played=move["played"], seconds=move.get("seconds", 0.0),
                             source=move.get("reused_from") or source)
                self._data.setdefault((move["board"], move["played"]), entry)


class AnalysisETA:
    """Estimate remaining work by empties, excluding cached/shared positions."""

    def __init__(self, schedule: Schedule, jobs: int, history: list[dict]) -> None:
        self.jobs = jobs
        self._lock = threading.Lock()
        self._remaining: dict[tuple[str, str], tuple[int, int]] = {}
        self._active: dict[tuple[str, str], float] = {}
        self._history: dict[int, list[float]] = {}
        self._current: dict[int, list[float]] = {}
        self.failed = False
        self._phase = {empties: index for index, phase in enumerate(schedule.phases)
                       for empties in range(phase.empties_lo, phase.empties_hi + 1)}
        for entry in history:
            seconds = entry.get("seconds", 0.0)
            if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0:
                self._history.setdefault(entry["board"].count("-"), []).append(seconds)

    def add_game(self, game: SelectedGame, positions: list[Position], known: PositionTable) -> None:
        for position in positions:
            key = position.board_str, index_to_coord(position.move)
            if known.get(*key) is None:
                self._remaining.setdefault(key, (position.empties, game.match_no))

    def start(self, position: Position) -> None:
        key = position.board_str, index_to_coord(position.move)
        with self._lock:
            if key in self._remaining:
                self._active[key] = time.perf_counter()

    def finish(self, position: Position, seconds: float) -> None:
        key = position.board_str, index_to_coord(position.move)
        with self._lock:
            if self._remaining.pop(key, None) is not None:
                self._current.setdefault(position.empties, []).append(seconds)
            self._active.pop(key, None)

    def _prediction(self, empties: int) -> float | None:
        # New measurements take precedence over older runs (e.g. before best-first analysis).
        samples = self._current.get(empties) or self._history.get(empties)
        if not samples:
            phase = self._phase.get(empties)
            for source in (self._current, self._history):
                samples = [seconds for n_empty, values in source.items()
                           if phase is not None and self._phase.get(n_empty) == phase for seconds in values]
                if samples:
                    break
        return sum(samples) / len(samples) if samples else None

    def remaining_seconds(self) -> float | None:
        with self._lock:
            if self.failed:
                return None
            now = time.perf_counter()
            by_match: dict[int, float] = {}
            for key, (empties, match_no) in self._remaining.items():
                seconds = self._prediction(empties)
                if seconds is None:
                    return None
                if key in self._active:
                    seconds = max(1.0, seconds - (now - self._active[key]))
                by_match[match_no] = by_match.get(match_no, 0.0) + seconds
            # --jobs assigns whole matches to workers. A final single match uses one worker.
            workers = [0.0] * self.jobs
            for match_no in sorted(by_match):
                worker = min(range(self.jobs), key=workers.__getitem__)
                workers[worker] += by_match[match_no]
            return max(workers)

    def text(self) -> str:
        seconds = self.remaining_seconds()
        if seconds is None:
            return "ETA: 算出不可（解析失敗あり）" if self.failed else "ETA: 算出待ち（解析時間を収集中）"
        if seconds == 0:
            with self._lock:
                if not self._remaining:
                    return "ETA: 残り0分（解析完了）"
        if seconds < 60:
            duration = "1分未満"
        else:
            hours, minutes = divmod(math.ceil(seconds / 60), 60)
            duration = "約" + (f"{hours}時間" if hours else "") + (f"{minutes}分" if minutes else "")
        finish = (datetime.now() + timedelta(seconds=seconds)).strftime("%m/%d %H:%M")
        return f"ETA: 残り{duration}（完了見込み {finish}）"


def game_label(record: GameRecord) -> str:
    return f"{record.finished_at.strftime('%Y-%m-%d %H:%M')} {record.game_id}"


def move_result(position: Position, game: SelectedGame, played: str, analysis: dict, seconds: float,
                reused_from: str | None) -> dict:
    played_score, alt_score = analysis["played_score"], analysis["alt_score"]
    loss = max(0, alt_score - played_score) if alt_score is not None else 0
    if alt_score is not None and alt_score > played_score:
        best_move, best_score = analysis["alt_move"], alt_score
    else:
        best_move, best_score = played, played_score
    return {
        "index": position.index,
        "empties": position.empties,
        "side": position.side,
        "is_me": position.side == game.my_color,
        "board": position.board_str,
        "played": played,
        **{field: analysis[field] for field in ANALYSIS_FIELDS},
        "best_move": best_move,
        "best_score": best_score,
        "loss": loss,
        "seconds": round(seconds, 3),
        "reused_from": reused_from,
    }


def analyze_game(get_engine, game: SelectedGame, label: str, threshold: int, known: PositionTable,
                 eta: AnalysisETA | None = None) -> list[dict]:
    record = game.record
    source = game_label(record)
    positions = replay(record).positions
    by_ply = {position.n_discs - 3: position for position in positions}

    # positions analyzed before: earlier (interrupted) runs of this game, or the same
    # opening in another game such as the other game of a synchro match
    missing = [position for position in positions if known.get(position.board_str, index_to_coord(position.move)) is None]
    n_known = len(positions) - len(missing)
    if n_known:
        n_other = sum(1 for position in positions
                      if (entry := known.get(position.board_str, index_to_coord(position.move))) and entry["source"] != source)
        log(f"  {label} {n_known}手は解析済み (前回までの途中結果 {n_known - n_other}手, 他の対局からの流用 {n_other}手)")

    mismatches: list[str] = []

    def on_row(row: AnalyzeRow) -> None:
        position = by_ply.get(row.ply)
        if position is None:
            return
        if row.played != index_to_coord(position.move):
            mismatches.append(f"{position.index + 1}手目の着手が一致しません (ログ {index_to_coord(position.move)} / Egaroucid {row.played})")
            return
        analysis = {field: getattr(row, field) for field in ANALYSIS_FIELDS}
        known.add(position.board_str, row.played, analysis, row.seconds, source)
        if eta is not None:
            eta.finish(position, row.seconds)
            next_position = by_ply.get(row.ply - 1)
            if next_position is not None and first <= next_position.index <= last:
                eta.start(next_position)
        is_me = position.side == game.my_color
        loss = max(0, row.alt_score - row.played_score) if row.alt_score is not None else 0
        who = "自分" if is_me else "相手"
        mark = ""
        if loss >= threshold:
            mark = f"  <== {who}の悪手 (損失{loss})"
        elif loss > 0 and is_me:
            mark = f"  <- 自分の疑問手 (損失{loss})"
        alt = f"{row.alt_move} {signed(row.alt_score)} [{row.alt_depth}]" if row.alt_move else "着手と一致"
        log(f"  {label} {position.index + 1:>2}手目 {position.empties:>2}マス空き {color_name(position.side)}({who}) "
            f"{row.played} {signed(row.played_score):>3} [{row.played_depth}] | 最善手 {alt} | "
            f"石損{loss}石 | 解析時間{row.seconds:.1f}秒{mark}" + (f" | {eta.text()}" if eta is not None else ""))

    # analyze each run of consecutive missing moves, the last run first as analyze itself goes backward
    runs: list[list[Position]] = []
    for position in missing:
        if runs and runs[-1][-1].index + 1 == position.index:
            runs[-1].append(position)
        else:
            runs.append([position])
    for run in reversed(runs):
        engine = get_engine()
        first, last = run[0].index, run[-1].index
        if eta is not None:
            eta.start(run[-1])
        engine.analyze(run[0].board_str, "".join(record.moves[first:last + 1]), on_row, label)
        engine_errors = f" (Egaroucid: {' / '.join(engine.errors)})" if engine.errors else ""
        if mismatches:
            raise EngineError(mismatches[0] + engine_errors)
        for position in run:
            if known.get(position.board_str, index_to_coord(position.move)) is None:
                raise EngineError(f"{position.index + 1}手目の解析結果がありません{engine_errors}")

    return moves_from_known(game, positions, known)


def moves_from_known(game: SelectedGame, positions: list[Position], known: PositionTable) -> list[dict] | None:
    """Results of every move of the game from the table, or None if a move is not analyzed yet."""
    source = game_label(game.record)
    results: list[dict] = []
    for position in positions:
        played = index_to_coord(position.move)
        entry = known.get(position.board_str, played)
        if entry is None:
            return None
        reused_from = None if entry["source"] == source else entry["source"]
        results.append(move_result(position, game, played, entry, 0.0 if reused_from else entry.get("seconds", 0.0), reused_from))
    return results


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

@dataclass
class GameAnalysis:
    no: int
    game: SelectedGame
    moves: list[dict]
    final_cells: str
    terminal: bool
    seconds: float

    def side_moves(self, is_me: bool) -> list[dict]:
        return [move for move in self.moves if move["is_me"] == is_me]

    def loss_moves(self, is_me: bool = True) -> list[dict]:
        """Every move that loses stones, regardless of the game's predicted result."""
        return [move for move in self.side_moves(is_me) if move["loss"] > 0]

def match_text(game: SelectedGame) -> str:
    if game.match_size > 1:
        return f"マッチ {game.match_id} 合計 {signed(game.match_score)} ({game.reason})"
    return game.reason


def count_matches(games, reason: str | None = None) -> int:
    return len({game.match_no for game in games if reason is None or game.reason == reason})


def move_label(move: dict) -> str:
    return f"{move['index'] + 1}手目（{move['empties']}マス空き）{color_name(move['side'])} {move['played']}"


def game_search_text(move: dict) -> str:
    """In-game search (depth@probability) that chose our move, from the GGS client log."""
    info = move.get("game")
    if not info:
        return "-"
    if info["depth"].startswith("-"):
        return "contest book" if info["kind"] == "contest book" else "book"
    return info["depth"]


def game_value_text(move: dict) -> str:
    info = move.get("game")
    return signed(info["value"]) if info else "-"


def game_time_text(move: dict) -> str:
    info = move.get("game")
    return f"{info['time_ms'] / 1000:.1f}秒" if info else "-"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_position(out: list[str], title: str, move: dict) -> None:
    out.append(f"**{title}**: {move_label(move)} 評価 {signed(move['played_score'])} → 最善 {move['best_move']} "
               f"評価 {signed(move['best_score'])} (石損 {move['loss']}石、着手の解析深さ {move['played_depth']}、最善手の解析深さ {move['alt_depth']})")
    out.append("")
    if move.get("game"):
        out.append(f"対局時: 読み {game_search_text(move)} / 評価 {game_value_text(move)} / 思考時間 {game_time_text(move)}")
        out.append("")
    out.append("```")
    marks = {coord_to_index(move["played"]): "*", coord_to_index(move["best_move"]): "#"}
    out.append(board_diagram(move["board"][:64], marks))
    out.append(f"手番: {color_name(move['side'])}  (* = 実際に打った手, # = 最善手)")
    out.append("```")
    out.append("")
    out.append(f"盤面: `{move['board']}`")
    out.append("")


def loss_certainty(move: dict) -> str:
    """A loss is exact only when both compared move values were solved exactly."""
    for field in ("played_depth", "alt_depth"):
        depth = re.fullmatch(r"(\d+)@([\d.]+)%", move[field])
        if depth is None or int(depth[1]) < move["empties"] or float(depth[2]) != 100:
            return "近似解析による推定"
    return "完全読みで確認"


def best_depth(move: dict) -> str:
    return move["alt_depth"] if move["loss"] > 0 else move["played_depth"]


def grouped_analyses(analyses: list[GameAnalysis]) -> list[list[GameAnalysis]]:
    groups: dict[int, list[GameAnalysis]] = {}
    for analysis in analyses:
        groups.setdefault(analysis.game.match_no, []).append(analysis)
    return [groups[key] for key in sorted(groups)]


def write_loss_table(out: list[str], analyses: list[GameAnalysis], is_me: bool) -> None:
    losses = [(a, move) for a in analyses for move in a.loss_moves(is_me)]
    out.extend([
        "| 対局ID | 手数 | 空きマス数 | 着手 | 最善手 | 着手の評価値（石） | 最善手の評価値（石） | 石損 | 着手の解析深さ | 最善手の解析深さ | 判定 | 対局時の探索深さ | 対局時の探索時間 |",
        "|---|---:|---:|---|---|---:|---:|---:|---|---|---|---|---|",
    ])
    for a, move in losses:
        out.append(
            f"| `{a.game.record.game_id}` | {move['index'] + 1} | {move['empties']} | {move['played']} | "
            f"{move['best_move']} | {signed(move['played_score'])} | {signed(move['best_score'])} | "
            f"{move['loss']} | {move['played_depth']} | {move['alt_depth']} | {loss_certainty(move)} | "
            f"{game_search_text(move)} | {game_time_text(move)} |"
        )
    if not losses:
        out.append("| 該当する手はない | | | | | | | | | | | | |")
    out.append("")


def write_report(path: Path, analyses: list[GameAnalysis], failures: list[tuple[SelectedGame, str]],
                 unanalyzed: list[SelectedGame], args: argparse.Namespace, schedule: Schedule, engine_lines: list[str],
                 stats: SelectionStats, n_client_logs: int) -> None:
    groups = grouped_analyses(analyses)
    complete = [group for group in groups if len(group) == group[0].game.match_size]
    all_mine = [move for group in complete for a in group for move in a.side_moves(True)]
    all_opp = [move for group in complete for a in group for move in a.side_moves(False)]
    mine_losses = [move for move in all_mine if move["loss"] > 0]
    n_exact = sum(loss_certainty(move) == "完全読みで確認" for move in mine_losses)
    out = ["# 要約", "",
           f"Egaroucid for Consoleで{len(complete)}組の解析を完了した。各着手で、最善手の評価値と実際に打った手の評価値を比べ、1石以上損した手をすべて抽出した。",
           f"Egaroucidの{len(all_mine)}着手のうち、石損があったのは{len(mine_losses)}手、石損の合計は{sum(move['loss'] for move in mine_losses)}石だった。"
           f"そのうち{n_exact}手は完全読みで確認し、{len(mine_losses) - n_exact}手は近似解析による推定である。", "",
           "# 用語の定義", "",
           "| 用語 | 定義 | 要するに |", "|---|---|---|",
           "| 組 | 同じ開始局面で黒白を入れ替えて打つ2局。対象を選ぶときは、2局の最終石差をEgaroucid視点にそろえて足す。 | 2局をまとめた1対戦。 |",
           "| 石損 | 同じ局面・同じ探索設定で求めた最善手の評価値から、実際に打った手の評価値を引いた値。 | 別の手を選べば増やせたと解析された石差。 |",
           "| 近似解析による推定 | 石損を比べた二つの評価値の少なくとも一方が、中盤探索またはMPC確率100%未満の探索で求められた結果。 | 解析の誤差で石損の大きさや有無が変わりうる。 |",
           "| 完全読みで確認 | 石損を比べた二つの評価値が、ともに終局までMPC確率100%で探索された結果。 | 厳密な石差の比較で確認した。 |",
           "| 探索深さの表記 | `35@74%`は探索深さ35、MPC確率74%を表す。深さが空きマス数以上なら終局までの探索である。 | 何手先まで、どのMPC確率で調べたか。 |", "",
           "# 詳細", "", "## 1. 対象と解析方法", "",
           f"- 作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
           *[f"- {line}" for line in engine_lines],
           f"- 自分のアカウント: `{raw_names(args.me)}`",
           "- 対象の抽出単位: " + ("組。2局の最終石差の合計で負け・引き分けを判定。" if args.unit == "match" else "利用者が指定した局単位。"),
           f"- 解析完了: {len(complete)}組。片方だけ解析済みの組: {len(groups) - len(complete)}組。解析失敗: {len(failures)}局。未解析: {len(unanalyzed)}局。",
           f"- 読み込んだ棋譜: {stats.files}ファイル。重複を除外した件数: {stats.duplicates}件。結果不明で除外した件数: {stats.unknown_result}件。", "",
           "同じ解析設定の保存済みの結果を再利用した。新たに探索する局面はConsoleの`analyzefast`で末尾から逆順に解析する。"
           "同じ組の2局には同じConsoleプロセスを使い、対局や局面の切り替えでは置換表を消さない。まず最善手を探索し、実際の着手と同じなら石損0石とする。"
           "異なる場合だけ、実際の着手を同じ局面・同じ探索設定で探索し直し、最初に求めた最善手との差を計算する。石損が1石以上の手をすべて拾った。"
           "勝ち・引き分け・負けの予測が変わったかどうかで手を絞らず、2局の途中の評価値も足していない。", "",
           "解析値は、その着手を打つ側から見た最終石差である。相手の手を掲載する表では相手側の評価値になる。"
           "最善手の評価値と着手の評価値は、同じ局面の同じ探索設定で比べた。", "",
           "| 空きマス数 | 解析設定 |", "|---|---|",
           *[f"| {phase.empties_hi}～{phase.empties_lo} | "
             + (f"探索深さ{phase.depth}、MPC確率{phase.prob}%" if phase.kind == "mid" else
                f"終局まで探索、MPC確率{phase.prob}%" if phase.kind == "end" else "完全読み（MPC確率100%）")
             + " |" for phase in schedule.phases], "",
           "MPC確率100%未満の評価値には誤差がある。特に1～2石の推定石損は、探索設定を強くすると消える場合がある。"
           "石損の合計は各着手で推定された差の合計であり、最終結果の石差とは一致するとは限らない。", "",
           "## 2. 組ごとの集計", "",
           "集計は2局とも解析が終わった組について行った。石損の判定は各着手ごとに行った。", "",
           "| 解析組番号 | 終局日時 | 組ID | 相手 | 最終石差の合計 | Egaroucidの解析着手数 | Egaroucidの石損した手数 | Egaroucidの石損合計 | 相手の石損した手数 | 相手の石損合計 |",
           "|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in complete:
        first = group[0].game
        mine = [move for a in group for move in a.side_moves(True)]
        opp = [move for a in group for move in a.side_moves(False)]
        stamp = max(a.game.record.finished_at for a in group).strftime('%Y-%m-%d %H:%M:%S')
        out.append(f"| [{first.match_no}](#match-{first.match_no}) | {stamp} | `{first.match_id}` | {md_escape(first.opponent)} | "
                   f"{signed(first.match_score)} | {len(mine)} | {sum(move['loss'] > 0 for move in mine)} | {sum(move['loss'] for move in mine)} | "
                   f"{sum(move['loss'] > 0 for move in opp)} | {sum(move['loss'] for move in opp)} |")
    out.extend(["", "## 3. 空きマス数ごとの石損", "",
                "| 空きマス数 | Egaroucidの解析着手数 | 石損した手数 | 石損合計 | 完全読みで確認した手数 |",
                "|---:|---:|---:|---:|---:|"])
    for empties in sorted({move["empties"] for move in all_mine}, reverse=True):
        moves = [move for move in all_mine if move["empties"] == empties]
        losses = [move for move in moves if move["loss"] > 0]
        out.append(f"| {empties} | {len(moves)} | {len(losses)} | {sum(move['loss'] for move in losses)} | "
                   f"{sum(loss_certainty(move) == '完全読みで確認' for move in losses)} |")
    n_matched = sum(bool(move.get("game")) for move in all_mine)
    n_reused = sum(bool(move.get("reused_from")) for a in analyses for move in a.moves)
    out.extend(["", f"対局時の探索記録は{n_client_logs}ファイルを調べ、Egaroucidの{len(all_mine)}着手中{n_matched}着手に対応付けた。"
                f"同じ局面・同じ着手で解析済みの結果を再利用したのは{n_reused}着手である。", "",
                "## 4. 組ごとの石損した手", ""])
    for group in groups:
        first = group[0].game
        out.extend([f'<a id="match-{first.match_no}"></a>', "",
                    f"### 解析組{first.match_no}: `{first.match_id}`、{first.opponent}", "",
                    "Egaroucidが石損したと解析された手は次のとおり。", ""])
        write_loss_table(out, group, True)
        out.extend(["相手が石損したと解析された手は次のとおり。", ""])
        write_loss_table(out, group, False)
        for a in group:
            record = a.game.record
            out.extend([f"#### 対局 `{record.game_id}`", "",
                        f"- Egaroucidの色: {color_name(a.game.my_color)}",
                        f"- 元の棋譜: `{record.source}`",
                        f"- 初期局面: `{record.initial_cells} {record.initial_side}`",
                        f"- 棋譜: `{''.join(record.moves)}`", ""])
            for move in a.loss_moves():
                write_position(out, f"石損した手（{loss_certainty(move)}）", move)
            out.extend(["<details>", "<summary>全着手の解析結果</summary>", "",
                        "| 手数 | 空きマス数 | 手番 | 着手 | 着手の評価値（石） | 最善手 | 最善手の評価値（石） | 石損 | 着手の解析深さ | 最善手の解析深さ |",
                        "|---:|---:|---|---|---:|---|---:|---:|---|---|"])
            for move in a.moves:
                who = "Egaroucid" if move["is_me"] else "相手"
                out.append(f"| {move['index'] + 1} | {move['empties']} | {who} | {move['played']} | "
                           f"{signed(move['played_score'])} | {move['best_move']} | {signed(move['best_score'])} | "
                           f"{move['loss']} | {move['played_depth']} | {best_depth(move)} |")
            out.extend(["", "</details>", ""])
    if failures or unanalyzed:
        out.extend(["## 5. 解析が終わっていない棋譜", ""])
        for game, message in failures:
            out.append(f"- `{game.record.source}`: {message}")
        for game in unanalyzed:
            out.append(f"- `{game.record.source}`: 未解析。")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_csvs(out_dir: Path, analyses: list[GameAnalysis], threshold: int) -> None:
    with open(out_dir / "matches.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["match_no", "match_id", "finished_at", "opponent", "match_score", "reason", "analysis_complete",
                         "games_analyzed", "game_ids", "my_moves", "my_loss_moves", "my_loss_total", "my_exact_loss_moves",
                         "opp_moves", "opp_loss_moves", "opp_loss_total", "analysis_seconds"])
        for group in grouped_analyses(analyses):
            first = group[0].game
            complete = len(group) == first.match_size
            mine = [move for a in group for move in a.side_moves(True)]
            opp = [move for a in group for move in a.side_moves(False)]
            losses = [move for move in mine if move["loss"] > 0]
            writer.writerow([first.match_no, first.match_id, max(a.game.record.finished_at for a in group).isoformat(sep=" "),
                             first.opponent, first.match_score, first.reason, int(complete), len(group),
                             " ".join(a.game.record.game_id for a in group),
                             len(mine) if complete else "", len(losses) if complete else "",
                             sum(move["loss"] for move in mine) if complete else "",
                             sum(loss_certainty(move) == "完全読みで確認" for move in losses) if complete else "",
                             len(opp) if complete else "", sum(move["loss"] > 0 for move in opp) if complete else "",
                             sum(move["loss"] for move in opp) if complete else "", round(sum(a.seconds for a in group), 3)])
    with open(out_dir / "moves.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["match_no", "match_id", "game_id", "source", "move_no", "empties", "color", "player", "is_me",
                         "played", "played_score", "played_depth", "alt_move", "alt_score", "alt_depth", "best_move", "best_score", "best_depth",
                         "loss", "loss_certainty", "board", "seconds", "reused_from", "game_depth", "game_value", "game_time_ms"])
        for a in analyses:
            record = a.game.record
            for move in a.moves:
                game = move.get("game")
                writer.writerow([a.game.match_no, a.game.match_id, record.game_id, str(record.source), move["index"] + 1,
                                 move["empties"], "black" if move["side"] == "X" else "white",
                                 record.black if move["side"] == "X" else record.white, int(move["is_me"]),
                                 move["played"], move["played_score"], move["played_depth"], move["alt_move"] or "",
                                 "" if move["alt_score"] is None else move["alt_score"], move["alt_depth"],
                                 move["best_move"], move["best_score"], best_depth(move), move["loss"], loss_certainty(move) if move["loss"] else "",
                                 move["board"], move["seconds"], move.get("reused_from") or "",
                                 game["depth"] if game else "", game["value"] if game else "", game["time_ms"] if game else ""])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GGS の対局ログから負け・引き分けの対局を抽出し、Egaroucid for Console で解析します。",
    )
    parser.add_argument("logs", nargs="+", type=Path, help="GGS の対局ログ (ファイルまたはディレクトリ)")
    select = parser.add_argument_group("対局の抽出")
    select.add_argument("--me", action="append", help=f"自分のアカウント名 (複数指定可, 既定: {', '.join(DEFAULT_ME)})")
    select.add_argument("--unit", choices=("match", "game"), default="match",
                        help="勝敗を判定する単位。match: シンクロ対局の2局の合計 (既定), game: 1局ごと")
    select.add_argument("--vs", action="append", help="負けに加えて引き分けも抽出する相手 (複数指定・カンマ区切り可)")
    select.add_argument("--vs-only", action="store_true", help="--vs で指定した相手との対局だけを抽出する")
    select.add_argument("--draws", action="store_true", help="全ての相手との引き分けも抽出する")
    select.add_argument("--exclude", action="append", help="除外する相手 (複数指定・カンマ区切り可)")
    select.add_argument("--since", type=parse_datetime_arg, help="この日時以降に終局した対局だけ (例: 2026-09-26)")
    select.add_argument("--until", type=parse_datetime_arg, help="この日時以前に終局した対局だけ")
    select.add_argument("--no-recursive", action="store_true", help="ディレクトリを再帰的に探さない")
    select.add_argument("--max-games", type=int, help="解析する対局数の上限 (新しい順ではなく古い順)")
    select.add_argument("--list", action="store_true", help="抽出結果を表示するだけで解析しない")
    search = parser.add_argument_group("解析設定")
    search.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help=f"Egaroucid for Console の実行ファイル (既定: {DEFAULT_ENGINE})")
    search.add_argument("--threads", type=int, help="1エンジンあたりのスレッド数 (既定: 論理コア数 / --jobs)")
    search.add_argument("--jobs", type=int, default=1, help="同時に動かすエンジンの数 (既定: 1)")
    search.add_argument("--hash", type=int, default=DEFAULT_HASH_LEVEL,
                        help=f"置換表のハッシュレベル (-hash, 既定: {DEFAULT_HASH_LEVEL})")
    search.add_argument("--book", action="store_true", help="book を使う (既定では使わず全局面を探索)")
    search.add_argument("--mid-depth", type=int, default=35, help="中盤探索の深さ (既定: 35)")
    search.add_argument("--mid-prob", default="74", choices=VALID_PROBS, help="中盤探索の MPC 確率 (既定: 74)")
    search.add_argument("--end-empties", type=int, default=43, help="終盤読み切りを始める空きマス数 (既定: 43)")
    search.add_argument("--exact-empties", type=int, default=30, help="完全読みを始める空きマス数 (既定: 30)")
    search.add_argument("--end-probs", default="88,93,98,99",
                        help="終盤読み切り区間を前から順に分割して使う MPC 確率 (既定: 88,93,98,99)")
    search.add_argument("--engine-arg", action="append", default=[], help="エンジンに追加で渡す引数 (例: --engine-arg=-seed --engine-arg=1)")
    output = parser.add_argument_group("出力")
    output.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help=f"出力ディレクトリ (既定: {DEFAULT_OUT_DIR})")
    output.add_argument("--mistake-threshold", type=int, default=4, help="悪手とみなす損失 (既定: 4)")
    output.add_argument("--force", action="store_true", help="解析済みのキャッシュを使わずに解析し直す")
    output.add_argument("--report-only", action="store_true",
                        help="エンジンを動かさず、解析済みのキャッシュからレポートと CSV だけを作り直す")
    output.add_argument("--client-log", action="append", type=Path, default=[],
                        help="対局時の読み深さを取る GGS クライアントのログ (*_ggs.log) かそのディレクトリ。"
                             "既定では対局ログのディレクトリとその親ディレクトリから探す")
    args = parser.parse_args()

    args.me = args.me or list(DEFAULT_ME)
    args.end_probs = [prob.strip() for prob in args.end_probs.split(",") if prob.strip()]
    for prob in args.end_probs:
        if prob not in VALID_PROBS:
            parser.error(f"--end-probs の値 {prob} は {', '.join(VALID_PROBS)} のいずれかにしてください")
    if not 1 <= args.mid_depth <= 60:
        parser.error("--mid-depth は 1〜60 にしてください")
    if not 0 <= args.exact_empties <= args.end_empties <= 60:
        parser.error("0 <= --exact-empties <= --end-empties <= 60 にしてください")
    if args.jobs < 1:
        parser.error("--jobs は 1 以上にしてください")
    if args.threads is None:
        args.threads = max(1, (os.cpu_count() or 1) // args.jobs)
    if args.mistake_threshold < 1:
        parser.error("--mistake-threshold は 1 以上にしてください")
    return args


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    args = parse_args()
    schedule = Schedule(args.mid_depth, args.mid_prob, args.end_empties, args.exact_empties, args.end_probs)

    stats = SelectionStats()
    files = collect_log_files(args.logs, recursive=not args.no_recursive)
    stats.files = len(files)
    records = [record for path in files for record in parse_log_file(path)]
    stats.records = len(records)
    games = select_games(records, args, stats)
    if args.max_games is not None:
        games = games[:args.max_games]

    log(f"ログ {stats.files}ファイル / {stats.records}対局 (重複 {stats.duplicates}, 自分以外 {stats.not_mine}, "
        f"結果不明 {stats.unknown_result}, 除外 {stats.excluded}, 期間外 {stats.out_of_period})")
    log(f"抽出した対局: {len(games)}局 / {count_matches(games)}マッチ "
        f"(負け {count_matches(games, '負け')}, 引き分け {count_matches(games, '引き分け')}, 単位: {args.unit})")
    for i, game in enumerate(games, 1):
        record = game.record
        log(f"  {i:>3}. {record.finished_at.strftime('%Y-%m-%d %H:%M')} {record.game_id:<8} "
            f"{record.black} (黒) vs {record.white} (白)  自分={color_name(game.my_color)} {signed(game.my_score):>3}  "
            f"[{match_text(game)}]")
    if args.list or not games:
        return 0

    engine_path = args.engine.resolve()
    engine_desc = ""
    cmd: list[str] = []
    if not args.report_only:
        if not engine_path.is_file():
            raise SystemExit(f"[ERROR] エンジンが見つかりません: {engine_path} (--engine で指定してください)")
        check_engine_supports_analyze_dpr(engine_path)
        engine_desc = engine_version(engine_path)
        cmd = [str(engine_path), "-quiet", "-nocontestbook", "-t", str(args.threads)]
        if not args.book:
            cmd.append("-nobook")
        cmd += ["-hash", str(args.hash)] + schedule.dpr_args() + args.engine_arg

    log("")
    if not args.report_only:
        log(f"エンジン: {engine_path} ({engine_desc})")
    log("解析設定:")
    for phase in schedule.phases:
        log(f"  {phase.describe()}")
    if not args.report_only:
        log(f"コマンド: {subprocess.list2cmdline(cmd)}")
    log("")

    out_dir = args.out
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)

    settings_key = schedule_cache_key(schedule, args.book)
    # every analyzed position is appended here immediately; --force ignores (but keeps) earlier results
    known_positions = PositionTable(out_dir / "cache" / f"positions_{settings_key[:16]}.jsonl", load=not args.force)
    if not args.force:
        for path in sorted((out_dir / "cache").glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("settings_key") == settings_key:
                    known_positions.add_game(data["moves"], data.get("label", data["game_id"]))
            except (OSError, ValueError, KeyError):
                pass

    analyses: dict[int, GameAnalysis] = {}
    failures: list[tuple[SelectedGame, str]] = []
    unanalyzed: list[SelectedGame] = []
    pending: list[tuple[int, SelectedGame, Path]] = []
    cached_run: tuple[float, dict] = (0.0, {})  # the most recently written game cache
    for no, game in enumerate(games, 1):
        key = game_cache_key(game, schedule, args.book)
        path = cache_path(out_dir, game, key)
        if path.is_file() and not args.force:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                game_replay = replay(game.record)
                for move in data["moves"]:
                    move["is_me"] = move["side"] == game.my_color
                analyses[no] = GameAnalysis(no, game, data["moves"], game_replay.final_cells, game_replay.terminal, data["seconds"])
                cached_run = max(cached_run, (path.stat().st_mtime, data), key=lambda item: item[0])
                log(f"[{no}/{len(games)}] {game.record.game_id} は解析済み (キャッシュ: {path.name})")
                continue
            except (OSError, ValueError, KeyError):
                pass
        if args.report_only:
            # a game whose moves are all in the per-move cache can still be reported
            game_replay = replay(game.record)
            moves = moves_from_known(game, game_replay.positions, known_positions)
            if moves is None:
                unanalyzed.append(game)
                log(f"[{no}/{len(games)}] {game.record.game_id} は解析が終わっていないのでレポートに含めません")
            else:
                seconds = sum(move["seconds"] for move in moves if not move.get("reused_from"))
                analyses[no] = GameAnalysis(no, game, moves, game_replay.final_cells, game_replay.terminal, seconds)
            continue
        pending.append((no, game, path))

    progress_lock = threading.Lock()
    completed_numbers = set(analyses)
    eta = AnalysisETA(schedule, args.jobs, known_positions.entries())
    for _, game, _ in pending:
        try:
            eta.add_game(game, replay(game.record).positions, known_positions)
        except ValueError:
            eta.failed = True  # run() will report the invalid transcript.

    def progress_text() -> str:
        with progress_lock:
            completed_pairs = sum(
                all(number in completed_numbers for number, member in enumerate(games, 1) if member.match_no == match_no)
                for match_no in {member.match_no for member in games}
            )
            counts = f"解析完了 {len(completed_numbers)}/{len(games)}局（{completed_pairs}/{count_matches(games)}組）"
        return counts + (f" | {eta.text()}" if not args.report_only else "")

    def show_progress(no: int | None = None) -> None:
        if no is not None:
            with progress_lock:
                completed_numbers.add(no)
        log(progress_text())

    show_progress()

    if args.report_only:
        # describe the engine and settings used when the cached games were analyzed
        cached_cmd = cached_run[1].get("command") or []
        option = lambda name: cached_cmd[cached_cmd.index(name) + 1] if name in cached_cmd[:-1] else "?"
        engine_lines = [f"エンジン: `{cached_cmd[0] if cached_cmd else '?'}` ({cached_run[1].get('engine', '不明')})",
                        f"スレッド数: {option('-t')} / book: {'不使用' if '-nobook' in cached_cmd else '使用'} / "
                        f"hash level {option('-hash')} (最後に解析した対局のコマンドから。このレポートは --report-only で作成)"]
    else:
        engine_lines = [f"エンジン: `{engine_path}` ({engine_desc})",
                        f"スレッド数: {args.threads} × 並列 {args.jobs} / book: {'使用' if args.book else '不使用'} / "
                        f"hash level {args.hash}"]

    engines: list[Engine] = []
    engines_lock = threading.Lock()
    local = threading.local()
    stopping = threading.Event()

    def get_engine() -> Engine:
        engine = getattr(local, "engine", None)
        if engine is None or engine.proc.poll() is not None:
            engine = Engine(cmd, engine_path.parent)
            engine.progress = progress_text
            log(f"Console起動: PID {engine.proc.pid}、{args.threads}スレッド、hash {args.hash}。対局間で置換表を保持します。")
            local.engine = engine
            with engines_lock:
                engines.append(engine)
        return engine

    def run(no: int, game: SelectedGame, path: Path) -> None:
        if stopping.is_set():
            return
        record = game.record
        label = f"[{no}/{len(games)} {record.game_id}]"
        log(f"{label} 解析開始: {record.black} (黒) vs {record.white} (白), 自分={color_name(game.my_color)} "
            f"{signed(game.my_score)}, {match_text(game)}, {len(record.moves)}手")
        start = time.perf_counter()
        try:
            game_replay = replay(record)
            moves = analyze_game(get_engine, game, label, args.mistake_threshold, known_positions, eta)
        except (ValueError, EngineError) as e:
            engine = getattr(local, "engine", None)
            if isinstance(e, EngineError) and engine is not None:
                try:
                    if engine.proc.wait(timeout=2) in WINDOWS_CTRL_C_EXIT_CODES:
                        stopping.set()  # Ctrl+C in the console also terminates the engines
                except subprocess.TimeoutExpired:
                    pass
            if stopping.is_set():
                return
            log(f"{label} [ERROR] 解析失敗: {e}")
            eta.failed = True
            failures.append((game, str(e)))
            if isinstance(e, EngineError):
                local.engine.kill()
                local.engine = None
            return
        elapsed = time.perf_counter() - start
        seconds = sum(move["seconds"] for move in moves if not move.get("reused_from"))  # includes interrupted runs
        path.write_text(json.dumps({"game_id": record.game_id, "label": game_label(record), "source": str(record.source),
                                    "settings_key": settings_key, "seconds": seconds, "engine": engine_desc,
                                    "command": cmd, "moves": moves},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        analyses[no] = GameAnalysis(no, game, moves, game_replay.final_cells, game_replay.terminal, seconds)
        n_reused = sum(1 for move in moves if move.get("reused_from"))
        log(f"{label} 解析完了 (今回 {elapsed:.0f}秒, 合計 {seconds:.0f}秒" + (f", {n_reused}局面は流用" if n_reused else "") + ")")
        show_progress(no)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    try:
        if executor is None:
            for item in pending:
                run(*item)
                if stopping.is_set():
                    raise KeyboardInterrupt
        else:
            by_match: dict[int, list[tuple[int, SelectedGame, Path]]] = {}
            for item in pending:
                by_match.setdefault(item[1].match_no, []).append(item)

            def run_match(items: list[tuple[int, SelectedGame, Path]]) -> None:
                for item in items:
                    run(*item)

            waiting = {executor.submit(run_match, items) for items in by_match.values()}
            while waiting:
                # poll with a timeout so that Ctrl+C reaches the main thread on Windows
                done, waiting = concurrent.futures.wait(waiting, timeout=0.5)
                for future in done:
                    future.result()
            if stopping.is_set():
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        stopping.set()
        with engines_lock:
            for engine in engines:
                engine.kill()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        log("")
        log("中断しました。同じコマンドをもう一度実行すると、解析済みの局面を飛ばして続きから再開します。")
        return 130
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        with engines_lock:
            for engine in engines:
                engine.close()
        known_positions.close()

    ordered = [analyses[no] for no in sorted(analyses)]
    client_logs = find_client_logs(args.logs, args.client_log, recursive=not args.no_recursive)
    client_index = ClientLogIndex(client_logs)
    n_matched = sum(client_index.annotate(a.game.record, a.moves) for a in ordered)
    log(f"対局時の読み: クライアントログ {len(client_logs)}ファイルから {n_matched}/{len(ordered)}局を対応付けました")
    write_report(out_dir / "report.md", ordered, failures, unanalyzed, args, schedule, engine_lines, stats, len(client_logs))
    write_csvs(out_dir, ordered, args.mistake_threshold)
    log("")
    log(f"レポート: {out_dir / 'report.md'}")
    log(f"CSV: {out_dir / 'matches.csv'}, {out_dir / 'moves.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
