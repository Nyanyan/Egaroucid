#!/usr/bin/env python3
"""Extract lost / drawn games from Egaroucid GGS game logs and analyze them
with the ``analyze`` command of Egaroucid for Console.

See README.md in this directory for usage (in Japanese).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENGINE = REPO_ROOT / "bin" / "Egaroucid_for_Console_clang.exe"
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

    def phase_of(self, empties: int) -> Phase | None:
        for phase in self.phases:
            if phase.empties_lo <= empties <= phase.empties_hi:
                return phase
        return None

    def phase_kind(self, empties: int) -> str:
        phase = self.phase_of(empties)
        return phase.kind if phase else "mid"

    def phase_label(self, kind: str) -> str:
        spans = [phase for phase in self.phases if phase.kind == kind]
        if not spans:
            return kind
        hi, lo = spans[0].empties_hi, spans[-1].empties_lo
        name = {"mid": "中盤探索", "end": "終盤読み切り(MPC)", "exact": "完全読み"}[kind]
        return f"{name} ({hi}〜{lo}空き)"


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

    def send(self, line: str) -> None:
        if self.proc.stdin is None or self.proc.poll() is not None:
            raise EngineError("Egaroucid for Console が終了しています")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def readline(self) -> str:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if line == "":
            raise EngineError("Egaroucid for Console が予期せず終了しました")
        return PROMPT_RE.sub("", line.rstrip("\r\n"))

    def analyze(self, board_str: str, moves: str, on_row) -> list[AnalyzeRow]:
        self.send("clearcache")
        self.send("setboard " + board_str)
        self.send("play " + moves)
        self.send("analyze")
        rows: list[AnalyzeRow] = []
        last_time = time.perf_counter()
        in_summary = False
        self.errors = []
        while True:
            line = self.readline()
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
    """Make sure the engine's analyze command follows -dpr (older builds silently ignore it)."""
    cmd = [str(engine), "-quiet", "-nobook", "-nocontestbook", "-t", "1", "-dpr", "1", "60", "3", "88"]
    try:
        result = subprocess.run(cmd, cwd=engine.parent, input=f"setboard {START_BOARD}\nplay f5\nanalyze\nexit\n",
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"[ERROR] エンジンを起動できません: {engine} ({e})")
    rows = [parse_analyze_row(PROMPT_RE.sub("", line)) for line in result.stdout.splitlines()]
    rows = [row for row in rows if row is not None]
    if not rows or rows[0].played_depth != "3@88%":
        raise SystemExit(
            f"[ERROR] {engine} の analyze コマンドが -dpr (深さ・確率指定) に対応していません。\n"
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


def sign(value: int) -> int:
    return (value > 0) - (value < 0)


def schedule_cache_key(schedule: Schedule, use_book: bool) -> str:
    payload = {"version": CACHE_VERSION, "dpr": schedule.dpr_args(), "book": use_book}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


ANALYSIS_FIELDS = ("played_score", "played_depth", "alt_move", "alt_score", "alt_depth")


class PositionTable:
    """Analysis results of already analyzed positions, keyed by (board, played move).

    The two games of a synchro match start from the same position and often
    share their first moves, so the second game can reuse these results.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._data)

    def get(self, board: str, played: str) -> dict | None:
        with self._lock:
            return self._data.get((board, played))

    def add_game(self, moves: list[dict], source: str) -> None:
        with self._lock:
            for move in moves:
                entry = {field: move[field] for field in ANALYSIS_FIELDS}
                entry["source"] = move.get("reused_from") or source
                self._data.setdefault((move["board"], move["played"]), entry)


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


def analyze_game(get_engine, game: SelectedGame, label: str, threshold: int, known: PositionTable) -> list[dict]:
    record = game.record
    positions = replay(record).positions
    by_ply = {position.n_discs - 3: position for position in positions}

    # reuse the opening shared with an already analyzed game (e.g. the other game of a synchro match)
    reused: list[dict] = []
    for position in positions:
        entry = known.get(position.board_str, index_to_coord(position.move))
        if entry is None:
            break
        reused.append(entry)
    n_reused = len(reused)
    if n_reused:
        log(f"  {label} 1〜{n_reused}手目は解析済みの局面と同じなので結果を流用します ({reused[0]['source']})")

    def on_row(row: AnalyzeRow) -> None:
        position = by_ply.get(row.ply)
        if position is None:
            return
        is_me = position.side == game.my_color
        loss = max(0, row.alt_score - row.played_score) if row.alt_score is not None else 0
        who = "自分" if is_me else "相手"
        mark = ""
        if loss >= threshold:
            mark = f"  <== {who}の悪手 (損失{loss})"
        elif loss > 0 and is_me:
            mark = f"  <- 自分の疑問手 (損失{loss})"
        alt = f"{row.alt_move} {signed(row.alt_score)}" if row.alt_move else "なし"
        log(f"  {label} {position.index + 1:>2}手目 {position.empties:>2}空き {color_name(position.side)}({who}) "
            f"{row.played} {signed(row.played_score):>3} [{row.played_depth}] | 他の最善 {alt} | {row.seconds:.1f}秒{mark}")

    rows_by_ply: dict[int, AnalyzeRow] = {}
    if n_reused < len(positions):
        start = positions[n_reused]
        engine = get_engine()
        rows = engine.analyze(start.board_str, "".join(record.moves[n_reused:]), on_row)
        rows_by_ply = {row.ply: row for row in rows}
        engine_errors = f" (Egaroucid: {' / '.join(engine.errors)})" if engine.errors else ""

    results: list[dict] = []
    for position in positions:
        played = index_to_coord(position.move)
        if position.index < n_reused:
            entry = reused[position.index]
            results.append(move_result(position, game, played, entry, 0.0, entry["source"]))
            continue
        row = rows_by_ply.get(position.n_discs - 3)
        if row is None:
            raise EngineError(f"{position.index + 1}手目の解析結果がありません{engine_errors}")
        if row.played != played:
            raise EngineError(f"{position.index + 1}手目の着手が一致しません (ログ {played} / Egaroucid {row.played}){engine_errors}")
        analysis = {field: getattr(row, field) for field in ANALYSIS_FIELDS}
        results.append(move_result(position, game, played, analysis, row.seconds, None))
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

    def my_eval(self, move: dict) -> int:
        """Evaluation of the position before ``move`` from my point of view."""
        return move["best_score"] if move["is_me"] else -move["best_score"]

    @property
    def start_eval(self) -> int | None:
        return self.my_eval(self.moves[0]) if self.moves else None

    @property
    def final_eval(self) -> int | None:
        """Evaluation after the last move from my point of view (the final score if the game reached the end)."""
        if not self.moves:
            return None
        last = self.moves[-1]
        return last["played_score"] if last["is_me"] else -last["played_score"]

    @property
    def lost_off_board(self) -> bool:
        """The game ended early (time out, resignation, ...) with a result worse than the board evaluation."""
        return not self.terminal and self.final_eval is not None and sign(self.final_eval) > sign(self.game.my_score)

    def side_moves(self, is_me: bool) -> list[dict]:
        return [move for move in self.moves if move["is_me"] == is_me]

    def worsened(self, move: dict) -> bool:
        return move["loss"] > 0 and sign(move["best_score"]) > sign(move["played_score"])

    @property
    def decisive(self) -> dict | None:
        candidates = [move for move in self.side_moves(True) if self.worsened(move)]
        return candidates[-1] if candidates else None

    @property
    def max_loss_move(self) -> dict | None:
        mine = [move for move in self.side_moves(True) if move["loss"] > 0]
        return max(mine, key=lambda move: (move["loss"], -move["index"])) if mine else None


def side_stats(moves: list[dict], threshold: int) -> dict[str, float]:
    losses = [move["loss"] for move in moves]
    return {
        "moves": len(moves),
        "mistakes": sum(1 for loss in losses if loss >= threshold),
        "inaccuracies": sum(1 for loss in losses if 0 < loss < threshold),
        "loss_total": sum(losses),
        "loss_avg": sum(losses) / len(losses) if losses else 0.0,
    }


def result_text(game: SelectedGame) -> str:
    if game.my_score < 0:
        return f"負け ({game.my_score})"
    if game.my_score == 0:
        return "引き分け (0)"
    return f"勝ち (+{game.my_score})"


def match_text(game: SelectedGame) -> str:
    if game.match_size > 1:
        return f"マッチ {game.match_id} 合計 {signed(game.match_score)} ({game.reason})"
    return game.reason


def count_matches(games, reason: str | None = None) -> int:
    return len({game.match_no for game in games if reason is None or game.reason == reason})


def move_label(move: dict) -> str:
    return f"{move['index'] + 1}手目 ({move['empties']}空き) {color_name(move['side'])} {move['played']}"


def judge_text(move: dict, threshold: int) -> str:
    if move["loss"] >= threshold:
        return "悪手"
    if move["loss"] > 0:
        return "疑問手"
    return ""


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_position(out: list[str], title: str, move: dict) -> None:
    out.append(f"**{title}**: {move_label(move)} 評価 {signed(move['played_score'])} → 最善 {move['best_move']} "
               f"評価 {signed(move['best_score'])} (損失 {move['loss']}, 探索 {move['played_depth']})")
    out.append("")
    out.append("```")
    marks = {coord_to_index(move["played"]): "*", coord_to_index(move["best_move"]): "#"}
    out.append(board_diagram(move["board"][:64], marks))
    out.append(f"手番: {color_name(move['side'])}  (* = 実際に打った手, # = 最善手)")
    out.append("```")
    out.append("")
    out.append(f"盤面: `{move['board']}`")
    out.append("")


def write_report(path: Path, analyses: list[GameAnalysis], failures: list[tuple[SelectedGame, str]],
                 args: argparse.Namespace, schedule: Schedule, engine_desc: str, stats: SelectionStats) -> None:
    threshold = args.mistake_threshold
    out: list[str] = []
    out.append("# GGS 対局解析レポート")
    out.append("")
    out.append(f"- 作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out.append(f"- エンジン: `{args.engine}` ({engine_desc})")
    out.append(f"- スレッド数: {args.threads} × 並列 {args.jobs} / book: {'使用' if args.book else '不使用'}"
               + f" / hash level {args.hash}")
    out.append("- 解析設定:")
    for phase in schedule.phases:
        out.append(f"  - {phase.describe()}")
    conditions = [f"自分 = {raw_names(args.me)}", "負け" + (f" (相手: {raw_names(args.vs)})" if args.vs_only else " (全相手)")]
    if args.draws and not args.vs_only:
        conditions.append("引き分け (全相手)")
    elif args.vs:
        conditions.append(f"引き分け (相手: {raw_names(args.vs)})")
    if args.exclude:
        conditions.append(f"除外: {raw_names(args.exclude)}")
    if args.since or args.until:
        conditions.append(f"期間: {args.since or ''} 〜 {args.until or ''}")
    conditions.append("シンクロ対局は2局の合計で勝敗を判定" if args.unit == "match" else "1局ごとに勝敗を判定")
    out.append(f"- 抽出条件: {' / '.join(conditions)}")
    games = [a.game for a in analyses]
    out.append(f"- 解析した対局: {len(analyses)}局 / {count_matches(games)}マッチ "
               f"(負け {count_matches(games, '負け')}, 引き分け {count_matches(games, '引き分け')})"
               + (f" / 解析失敗 {len(failures)}局" if failures else ""))
    out.append(f"- 読み込んだログ: {stats.files}ファイル, {stats.records}対局 (重複 {stats.duplicates}, 結果不明 {stats.unknown_result})")
    n_reused = sum(1 for a in analyses for move in a.moves if move.get("reused_from"))
    if n_reused:
        out.append(f"- 解析済みの対局と同じ局面 (シンクロ対局の共通の序盤など) は結果を流用: {n_reused}局面")
    out.append(f"- 損失 = その局面での最善手の評価値 − 実際に打った手の評価値 (手番側から見た石差)。"
               f"損失{threshold}以上を「悪手」、1〜{threshold - 1}を「疑問手」と呼びます。")
    out.append("")

    if analyses:
        out.append("## まとめ")
        out.append("")
        all_mine = [move for a in analyses for move in a.side_moves(True)]
        all_opp = [move for a in analyses for move in a.side_moves(False)]
        mine, opp = side_stats(all_mine, threshold), side_stats(all_opp, threshold)
        out.append("| | 自分 | 相手 |")
        out.append("|---|---:|---:|")
        out.append(f"| 着手数 | {mine['moves']} | {opp['moves']} |")
        out.append(f"| 悪手 (損失{threshold}以上) | {mine['mistakes']} | {opp['mistakes']} |")
        out.append(f"| 疑問手 (損失1〜{threshold - 1}) | {mine['inaccuracies']} | {opp['inaccuracies']} |")
        out.append(f"| 損失合計 | {mine['loss_total']} | {opp['loss_total']} |")
        out.append(f"| 1手あたり平均損失 | {mine['loss_avg']:.3f} | {opp['loss_avg']:.3f} |")
        out.append("")

        out.append("### 探索フェーズ別の損失 (自分の手)")
        out.append("")
        out.append("| フェーズ | 着手数 | 損失あり | 悪手 | 損失合計 | 1手あたり |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for kind in ("mid", "end", "exact"):
            moves = [move for move in all_mine if schedule.phase_kind(move["empties"]) == kind]
            if not moves and not any(phase.kind == kind for phase in schedule.phases):
                continue
            s = side_stats(moves, threshold)
            out.append(f"| {schedule.phase_label(kind)} | {s['moves']} | {s['mistakes'] + s['inaccuracies']} | "
                       f"{s['mistakes']} | {s['loss_total']} | {s['loss_avg']:.3f} |")
        out.append("")

        out.append("### 相手別")
        out.append("")
        out.append("| 相手 | 局数 | 負けマッチ | 引き分けマッチ | 自分の損失合計 | 相手の損失合計 | 開始局面で既に不利な局 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        opponents = sorted({a.game.opponent for a in analyses}, key=str.casefold)
        for opponent in opponents:
            games_vs = [a for a in analyses if a.game.opponent == opponent]
            my_loss = sum(move["loss"] for a in games_vs for move in a.side_moves(True))
            opp_loss = sum(move["loss"] for a in games_vs for move in a.side_moves(False))
            bad_start = sum(1 for a in games_vs if a.start_eval is not None and a.start_eval < 0)
            selected_vs = [a.game for a in games_vs]
            out.append(f"| {md_escape(opponent)} | {len(games_vs)} | {count_matches(selected_vs, '負け')} | "
                       f"{count_matches(selected_vs, '引き分け')} | {my_loss} | {opp_loss} | {bad_start} |")
        out.append("")

        if args.unit == "match":
            out.append("## マッチ一覧")
            out.append("")
            out.append("シンクロ対局では同じ開始局面を黒白入れ替えて打つので、2局の損失合計を比べると差がどこで付いたか分かります。")
            out.append("")
            out.append("| マッチ | 終局日時 | 相手 | 各局の結果 (自分) | 合計 | 判定 | 自分の損失合計 | 相手の損失合計 |")
            out.append("|---|---|---|---|---:|---|---:|---:|")
            for match_no in sorted({a.game.match_no for a in analyses}):
                in_match = [a for a in analyses if a.game.match_no == match_no]
                first = in_match[0].game
                results = " / ".join(f"[{a.game.record.game_id}](#game-{a.no}) {color_name(a.game.my_color)} {signed(a.game.my_score)}"
                                     for a in in_match)
                my_loss = sum(move["loss"] for a in in_match for move in a.side_moves(True))
                opp_loss = sum(move["loss"] for a in in_match for move in a.side_moves(False))
                out.append(f"| `{first.match_id}` | {first.record.finished_at.strftime('%Y-%m-%d %H:%M')} | {md_escape(first.opponent)} | "
                           f"{results} | {signed(first.match_score)} | {first.reason} | {my_loss} | {opp_loss} |")
            out.append("")

        out.append("## 対局一覧")
        out.append("")
        out.append("| # | 終局日時 | ID | 黒 | 白 | 結果 (自分) | マッチ | 開始局面評価 (自分) | 自分の悪手/疑問手 | 自分の損失計 | 敗着候補 |")
        out.append("|---:|---|---|---|---|---|---|---:|---:|---:|---|")
        for a in analyses:
            record = a.game.record
            s = side_stats(a.side_moves(True), threshold)
            decisive = a.decisive
            decisive_text = f"{move_label(decisive)} (損失{decisive['loss']})" if decisive else "-"
            if a.lost_off_board:
                decisive_text += " / 時間切れ等?"
            black = f"**{md_escape(record.black)}**" if a.game.my_color == "X" else md_escape(record.black)
            white = f"**{md_escape(record.white)}**" if a.game.my_color == "O" else md_escape(record.white)
            out.append(
                f"| [{a.no}](#game-{a.no}) | {record.finished_at.strftime('%Y-%m-%d %H:%M')} | `{record.game_id}` | {black} | {white} | "
                f"{result_text(a.game)} | {match_text(a.game)} | {signed(a.start_eval)} | {s['mistakes']} / {s['inaccuracies']} | "
                f"{s['loss_total']} | {decisive_text} |"
            )
        out.append("")

    if failures:
        out.append("## 解析できなかった対局")
        out.append("")
        for game, message in failures:
            out.append(f"- `{game.record.game_id}` {game.record.black} vs {game.record.white} ({game.record.source}): {message}")
        out.append("")

    if analyses:
        out.append("## 対局詳細")
        out.append("")
    for a in analyses:
        record = a.game.record
        out.append(f'<a id="game-{a.no}"></a>')
        out.append("")
        out.append(f"### {a.no}. `{record.game_id}` {record.black} (黒) vs {record.white} (白) : {result_text(a.game)}")
        out.append("")
        out.append(f"- 自分: {a.game.me} ({color_name(a.game.my_color)}) / 相手: {a.game.opponent} / {match_text(a.game)}")
        out.append(f"- 終局日時: {record.finished_at.strftime('%Y-%m-%d %H:%M:%S')} / ログ: `{record.source}`")
        out.append(f"- 開始局面: `{record.initial_cells} {record.initial_side}`")
        out.append(f"- 棋譜: `{''.join(record.moves)}`" + ("" if a.terminal else " (終局前に対局終了: 時間切れ・投了など)"))
        out.append(f"- 評価 (自分視点): 開始局面 {signed(a.start_eval)} → 最終局面 {signed(a.final_eval)}"
                   + ("" if a.terminal else " (盤上の評価)") + f" / 解析時間: {a.seconds:.0f}秒")
        for is_me, who in ((True, "自分"), (False, "相手")):
            s = side_stats(a.side_moves(is_me), threshold)
            out.append(f"- {who}: 悪手 {s['mistakes']} / 疑問手 {s['inaccuracies']} / 損失合計 {s['loss_total']} / "
                       f"平均損失 {s['loss_avg']:.3f} ({s['moves']}手)")
        out.append("")

        decisive, max_loss = a.decisive, a.max_loss_move
        if a.lost_off_board:
            out.append(f"**注意**: 終局前に対局が終わっており、最終局面の評価 (自分視点 {signed(a.final_eval)}) より結果 "
                       f"({signed(a.game.my_score)}) が悪いです。時間切れ・投了・切断などで負けた可能性があります。")
            out.append("")
        if a.game.my_score <= 0:
            title = "敗着候補" if a.game.my_score < 0 else "勝ちを逃した手の候補"
            if decisive:
                write_position(out, f"{title} (自分の手で最後に勝敗の区分を悪化させた手)", decisive)
            elif not a.terminal:
                out.append(f"**{title}**: 盤上の手で勝敗の区分を悪化させた自分の手はありません。")
                out.append("")
            else:
                out.append(f"**{title}**: 自分の手で勝敗の区分 (勝ち/引き分け/負け) を悪化させた手はありません。"
                           "開始局面の時点で評価が負け (または引き分け) だった可能性があります。")
                out.append("")
        if max_loss and max_loss is not decisive:
            write_position(out, "損失が最大の手", max_loss)

        mine = [move for move in a.side_moves(True) if move["loss"] > 0]
        out.append("#### 自分の損失がある手")
        out.append("")
        if mine:
            out.append("| 手 | 空き | 着手 | 評価 | 最善 | 最善評価 | 損失 | 探索 | 判定 | 区分悪化 |")
            out.append("|---:|---:|---|---:|---|---:|---:|---|---|---|")
            for move in mine:
                out.append(f"| {move['index'] + 1} | {move['empties']} | {move['played']} | {signed(move['played_score'])} | "
                           f"{move['best_move']} | {signed(move['best_score'])} | {move['loss']} | {move['played_depth']} | "
                           f"{judge_text(move, threshold)} | {'○' if a.worsened(move) else ''} |")
        else:
            out.append("なし")
        out.append("")

        opp_mistakes = [move for move in a.side_moves(False) if move["loss"] >= threshold]
        out.append(f"#### 相手の悪手 (損失{threshold}以上)")
        out.append("")
        if opp_mistakes:
            out.append("| 手 | 空き | 着手 | 評価 | 最善 | 最善評価 | 損失 | 探索 |")
            out.append("|---:|---:|---|---:|---|---:|---:|---|")
            for move in opp_mistakes:
                out.append(f"| {move['index'] + 1} | {move['empties']} | {move['played']} | {signed(move['played_score'])} | "
                           f"{move['best_move']} | {signed(move['best_score'])} | {move['loss']} | {move['played_depth']} |")
        else:
            out.append("なし")
        out.append("")

        out.append("<details><summary>全着手の解析</summary>")
        out.append("")
        out.append("評価は手番側から見た石差。「自分視点」はその手を打つ前の局面を自分から見た評価値。"
                   "時間が「流用」の手は、解析済みの対局の同じ局面の結果を使っています。")
        out.append("")
        out.append("| 手 | 空き | 手番 | 着手 | 評価 | 探索 | 他の最善 | 評価 | 損失 | 自分視点 | 時間(秒) |")
        out.append("|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|")
        for move in a.moves:
            who = "自分" if move["is_me"] else "相手"
            out.append(f"| {move['index'] + 1} | {move['empties']} | {color_name(move['side'])}({who}) | {move['played']} | "
                       f"{signed(move['played_score'])} | {move['played_depth']} | {move['alt_move'] or '-'} | "
                       f"{signed(move['alt_score'])} | {move['loss'] or ''} | {signed(a.my_eval(move))} | "
                       f"{'流用' if move.get('reused_from') else format(move['seconds'], '.1f')} |")
        out.append("")
        out.append("</details>")
        out.append("")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_csvs(out_dir: Path, analyses: list[GameAnalysis], schedule: Schedule, threshold: int) -> None:
    with open(out_dir / "games.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "no", "game_id", "finished_at", "source", "black", "white", "me", "my_color", "opponent", "my_score",
            "match_id", "match_score", "reason", "start_eval_me", "my_moves", "my_mistakes", "my_inaccuracies", "my_loss_total",
            "opp_mistakes", "opp_inaccuracies", "opp_loss_total",
            "decisive_move_no", "decisive_empties", "decisive_played", "decisive_best", "decisive_loss",
            "max_loss_move_no", "max_loss_played", "max_loss_best", "max_loss", "terminal", "analysis_seconds",
        ])
        for a in analyses:
            record = a.game.record
            mine, opp = side_stats(a.side_moves(True), threshold), side_stats(a.side_moves(False), threshold)
            decisive, max_loss = a.decisive, a.max_loss_move
            writer.writerow([
                a.no, record.game_id, record.finished_at.isoformat(sep=" "), str(record.source), record.black, record.white,
                a.game.me, a.game.my_color, a.game.opponent, a.game.my_score,
                a.game.match_id, a.game.match_score, {"負け": "loss", "引き分け": "draw"}[a.game.reason], a.start_eval,
                mine["moves"], mine["mistakes"], mine["inaccuracies"], mine["loss_total"],
                opp["mistakes"], opp["inaccuracies"], opp["loss_total"],
                decisive["index"] + 1 if decisive else "", decisive["empties"] if decisive else "",
                decisive["played"] if decisive else "", decisive["best_move"] if decisive else "",
                decisive["loss"] if decisive else "",
                max_loss["index"] + 1 if max_loss else "", max_loss["played"] if max_loss else "",
                max_loss["best_move"] if max_loss else "", max_loss["loss"] if max_loss else "",
                int(a.terminal), round(a.seconds, 1),
            ])
    with open(out_dir / "moves.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "no", "game_id", "move_no", "empties", "phase", "color", "player", "is_me", "played", "played_score",
            "played_depth", "alt_move", "alt_score", "alt_depth", "best_move", "best_score", "loss", "judge",
            "result_worsened", "eval_me_before", "board", "seconds", "reused_from",
        ])
        for a in analyses:
            record = a.game.record
            for move in a.moves:
                player = record.black if move["side"] == "X" else record.white
                writer.writerow([
                    a.no, record.game_id, move["index"] + 1, move["empties"], schedule.phase_kind(move["empties"]),
                    "black" if move["side"] == "X" else "white", player, int(move["is_me"]), move["played"],
                    move["played_score"], move["played_depth"], move["alt_move"] or "", "" if move["alt_score"] is None else move["alt_score"],
                    move["alt_depth"], move["best_move"], move["best_score"], move["loss"],
                    {"悪手": "mistake", "疑問手": "inaccuracy"}.get(judge_text(move, threshold), ""),
                    int(a.worsened(move)), a.my_eval(move), move["board"], move["seconds"], move.get("reused_from") or "",
                ])


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
    if not engine_path.is_file():
        raise SystemExit(f"[ERROR] エンジンが見つかりません: {engine_path} (--engine で指定してください)")
    args.engine = engine_path
    check_engine_supports_analyze_dpr(engine_path)
    engine_desc = engine_version(engine_path)
    cmd = [str(engine_path), "-quiet", "-nocontestbook", "-t", str(args.threads)]
    if not args.book:
        cmd.append("-nobook")
    cmd += ["-hash", str(args.hash)] + schedule.dpr_args() + args.engine_arg

    log("")
    log(f"エンジン: {engine_path} ({engine_desc})")
    log("解析設定:")
    for phase in schedule.phases:
        log(f"  {phase.describe()}")
    log(f"コマンド: {subprocess.list2cmdline(cmd)}")
    log("")

    out_dir = args.out
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)

    settings_key = schedule_cache_key(schedule, args.book)
    known_positions = PositionTable()
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
    pending: list[tuple[int, SelectedGame, Path]] = []
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
                log(f"[{no}/{len(games)}] {game.record.game_id} は解析済み (キャッシュ: {path.name})")
                continue
            except (OSError, ValueError, KeyError):
                pass
        pending.append((no, game, path))

    engines: list[Engine] = []
    engines_lock = threading.Lock()
    local = threading.local()
    stopping = threading.Event()

    def get_engine() -> Engine:
        engine = getattr(local, "engine", None)
        if engine is None or engine.proc.poll() is not None:
            engine = Engine(cmd, engine_path.parent)
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
            moves = analyze_game(get_engine, game, label, args.mistake_threshold, known_positions)
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
            failures.append((game, str(e)))
            if isinstance(e, EngineError):
                local.engine.kill()
                local.engine = None
            return
        seconds = time.perf_counter() - start
        path.write_text(json.dumps({"game_id": record.game_id, "label": game_label(record), "source": str(record.source),
                                    "settings_key": settings_key, "seconds": seconds, "engine": engine_desc,
                                    "command": cmd, "moves": moves},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        known_positions.add_game(moves, game_label(record))
        analyses[no] = GameAnalysis(no, game, moves, game_replay.final_cells, game_replay.terminal, seconds)
        n_reused = sum(1 for move in moves if move.get("reused_from"))
        log(f"{label} 解析完了 ({seconds:.0f}秒" + (f", {n_reused}局面は流用" if n_reused else "") + ")")

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
        log("中断しました。同じコマンドをもう一度実行すると、解析済みの対局を飛ばして続きから再開します。")
        return 130
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        with engines_lock:
            for engine in engines:
                engine.close()

    ordered = [analyses[no] for no in sorted(analyses)]
    write_report(out_dir / "report.md", ordered, failures, args, schedule, engine_desc, stats)
    write_csvs(out_dir, ordered, schedule, args.mistake_threshold)
    log("")
    log(f"レポート: {out_dir / 'report.md'}")
    log(f"CSV: {out_dir / 'games.csv'}, {out_dir / 'moves.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
