"""Repeatedly ask one GGS opponent for the same match.

The script launches Egaroucid's built-in GGS client, waits for its completed
login and subscription setup, and then writes a ``ts ask`` command to the
client's standard input.  The Console forwards that command to GGS.  Each time
the Console reports the end of a match, this script submits the next request.

Credentials are deliberately not stored in this file.  Supply them with
``--ggs-user`` and ``--ggs-password`` or the ``GGS_USER`` and
``GGS_PASSWORD`` environment variables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ENGINE = REPOSITORY_ROOT / "bin" / "Egaroucid_for_Console_clang.exe"
DEFAULT_LOG_DIR = SCRIPT_DIR / "log"
DEFAULT_CONTEST_BOOK_DIR = (
    REPOSITORY_ROOT / "src" / "tools" / "gen_contest_book" / "trained"
)

INITIALIZED_MARKER = "GGS initialization completed; entering main loop"
MATCH_START_MARKER = "match start!"
MATCH_END_MARKER = "match end!"
TIME_CONTROL_PATTERN = re.compile(r"^\d+:\d{2}/(?:\d+:\d{2})?/(?:\d+:\d{2})?$")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GGS_MATCH_RESULT_MARKER = "/os: - match "
SCORE_EPSILON = 1.0e-9


@dataclass(frozen=True)
class MatchResult:
    match_id: str
    disc_difference: float


@dataclass
class MatchStatistics:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    disc_difference_sum: float = 0.0

    @property
    def completed_matches(self) -> int:
        return self.wins + self.draws + self.losses

    def record(self, result: MatchResult) -> None:
        if result.disc_difference > SCORE_EPSILON:
            self.wins += 1
        elif result.disc_difference < -SCORE_EPSILON:
            self.losses += 1
        else:
            self.draws += 1
        self.disc_difference_sum += result.disc_difference

    def win_rate(self) -> float:
        if self.completed_matches == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.completed_matches

    def average_disc_difference(self) -> float:
        if self.completed_matches == 0:
            return 0.0
        return self.disc_difference_sum / self.completed_matches

    def format_summary(
        self,
        label: str,
        max_matches: int = 0,
        latest_result: MatchResult | None = None,
    ) -> str:
        matches = str(self.completed_matches)
        if max_matches:
            matches = f"{matches}/{max_matches}"
        latest_text = ""
        if latest_result is not None:
            latest_text = (
                f" 直近match={latest_result.match_id}"
                f" 直近石差={format_signed_score(latest_result.disc_difference)}"
            )
        return (
            f"{label}: match={matches} 勝ち={self.wins} 引き分け={self.draws} "
            f"負け={self.losses} 勝率={self.win_rate() * 100.0:.1f}% "
            f"平均獲得石差={format_signed_score(self.average_disc_difference())}"
            f"{latest_text}"
        )


def validate_single_token(value: str, option: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{option} must be one non-empty token")
    return value


def validate_time_control(value: str) -> str:
    if not TIME_CONTROL_PATTERN.fullmatch(value):
        raise ValueError(
            "--time-control must have the GGS form initial/increment/extra, "
            "for example 1:00//0:30"
        )
    return value


def format_signed_score(value: float) -> str:
    if abs(value) <= SCORE_EPSILON:
        value = 0.0
    return f"{value:+.2f}"


def make_request(game_type: str, time_control: str, opponent: str) -> str:
    return f"ts ask {game_type} {time_control} {opponent}"


def parse_ggs_match_result(
    line: str, own_player: str, expected_opponent: str
) -> MatchResult | None:
    clean_line = ANSI_ESCAPE_PATTERN.sub("", line)
    marker_index = clean_line.find(GGS_MATCH_RESULT_MARKER)
    if marker_index < 0:
        return None

    tokens = clean_line[marker_index:].split()
    if len(tokens) < 11:
        return None
    if tokens[0:3] != ["/os:", "-", "match"]:
        return None

    player1 = tokens[5]
    player2 = tokens[7]
    expected_players = {own_player.casefold(), expected_opponent.casefold()}
    actual_players = {player1.casefold(), player2.casefold()}
    if actual_players != expected_players:
        return None

    try:
        disc_difference = float(tokens[10])
    except ValueError:
        return None

    if player2.casefold() == own_player.casefold():
        disc_difference = -disc_difference
    elif player1.casefold() != own_player.casefold():
        return None
    if abs(disc_difference) <= SCORE_EPSILON:
        disc_difference = 0.0
    return MatchResult(match_id=tokens[3], disc_difference=disc_difference)


def build_engine_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.engine),
        "-ggs", args.ggs_user, args.ggs_password,
        "-noise",
        "-logdir", str(args.log_dir),
        "-ggslogdir", str(args.log_dir),
        "-ggsgamelogdir", str(args.game_log_dir),
        "-t", str(args.threads),
        "-hash", str(args.hash_level),
        "-nobook",
    ]
    if not args.no_contest_book:
        command.extend(["-contestbook", str(args.contest_book_dir)])
    command.extend(args.engine_arg)
    return command


def redacted_command(command: Sequence[str], password: str) -> str:
    return " ".join("<password>" if part == password else part for part in command)


class ControllerLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, message: str) -> None:
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(line + "\n")


def send_request(
    process: subprocess.Popen[str], request: str, log: ControllerLog
) -> None:
    if process.stdin is None:
        raise RuntimeError("the Egaroucid process has no standard input")
    process.stdin.write(request + "\n")
    process.stdin.flush()
    log.write(f"sent request: {request}")


def terminate_process(process: subprocess.Popen[str], log: ControllerLog) -> None:
    if process.poll() is not None:
        return
    try:
        if process.stdin is not None:
            process.stdin.write("quit\n")
            process.stdin.flush()
        process.wait(timeout=10)
        log.write("Egaroucid exited after quit")
    except (BrokenPipeError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.write("Egaroucid was terminated")


def run(args: argparse.Namespace) -> int:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.game_log_dir.mkdir(parents=True, exist_ok=True)
    if not args.engine.is_file():
        raise FileNotFoundError(f"Egaroucid executable not found: {args.engine}")
    if not args.no_contest_book and not args.contest_book_dir.is_dir():
        raise FileNotFoundError(
            f"contest-book directory not found: {args.contest_book_dir}; "
            "pass --no-contest-book to start without it"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = ControllerLog(args.log_dir / f"{timestamp}_auto_battle.log")
    request = make_request(args.game_type, args.time_control, args.opponent)
    command = build_engine_command(args)
    log.write(f"launching: {redacted_command(command, args.ggs_password)}")
    log.write(f"request template: {request}")

    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    initialized = False
    match_active = False
    completed_matches = 0
    statistics = MatchStatistics()
    pending_match_result: MatchResult | None = None
    try:
        if process.stdout is None:
            raise RuntimeError("the Egaroucid process has no standard output")
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if line:
                log.write(f"console: {line}")
            if not initialized and INITIALIZED_MARKER in line:
                initialized = True
                send_request(process, request, log)
                continue
            if initialized and MATCH_START_MARKER in line:
                match_active = True
                pending_match_result = None
                log.write("match started")
                continue
            parsed_match_result = parse_ggs_match_result(
                line, args.ggs_user, args.opponent
            )
            if initialized and parsed_match_result is not None:
                pending_match_result = parsed_match_result
                continue
            if initialized and MATCH_END_MARKER in line and match_active:
                match_active = False
                completed_matches += 1
                if pending_match_result is not None:
                    statistics.record(pending_match_result)
                    log.write(
                        statistics.format_summary(
                            "対戦結果",
                            args.max_matches,
                            latest_result=pending_match_result,
                        )
                    )
                else:
                    log.write("対戦結果を読み取れませんでした。集計は更新していません。")
                log.write(f"match completed: {completed_matches}")
                if args.max_matches and completed_matches >= args.max_matches:
                    log.write("requested match limit reached")
                    log.write(statistics.format_summary("最終結果", args.max_matches))
                    break
                if args.request_delay:
                    log.write(f"waiting {args.request_delay:g} seconds before the next request")
                    time.sleep(args.request_delay)
                send_request(process, request, log)
        return_code = process.poll()
        if return_code is not None and return_code != 0:
            raise RuntimeError(f"Egaroucid exited with status {return_code}")
        if not initialized:
            raise RuntimeError("Egaroucid ended before GGS initialization completed")
        return 0
    except KeyboardInterrupt:
        log.write("interrupted by user")
        if statistics.completed_matches:
            log.write(statistics.format_summary("現在結果", args.max_matches))
        return 130
    finally:
        terminate_process(process, log)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--ggs-user", default=os.environ.get("GGS_USER"))
    parser.add_argument("--ggs-password", default=os.environ.get("GGS_PASSWORD"))
    parser.add_argument("--opponent", default="nyanyan")
    parser.add_argument("--game-type", default="s8r14")
    parser.add_argument("--time-control", default="1:00//0:30")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hash", dest="hash_level", type=int, default=29)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--game-log-dir", type=Path, default=DEFAULT_LOG_DIR / "game")
    parser.add_argument("--contest-book-dir", type=Path, default=DEFAULT_CONTEST_BOOK_DIR)
    parser.add_argument("--no-contest-book", action="store_true")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=3.0,
        help="seconds to wait after a match ends before requesting the next one",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="stop after this many completed matches; 0 means no limit",
    )
    parser.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        help="one additional argument forwarded to Egaroucid; repeat as needed",
    )
    args = parser.parse_args()
    if not args.ggs_user or not args.ggs_password:
        parser.error("set --ggs-user/--ggs-password or GGS_USER/GGS_PASSWORD")
    try:
        args.ggs_user = validate_single_token(args.ggs_user, "--ggs-user")
        args.ggs_password = validate_single_token(args.ggs_password, "--ggs-password")
        args.opponent = validate_single_token(args.opponent, "--opponent")
        args.game_type = validate_single_token(args.game_type, "--game-type")
        args.time_control = validate_time_control(args.time_control)
    except ValueError as error:
        parser.error(str(error))
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.hash_level < 0:
        parser.error("--hash must be non-negative")
    if args.request_delay < 0:
        parser.error("--request-delay must be non-negative")
    if args.max_matches < 0:
        parser.error("--max-matches must be non-negative")
    args.engine = args.engine.resolve()
    args.log_dir = args.log_dir.resolve()
    args.game_log_dir = args.game_log_dir.resolve()
    args.contest_book_dir = args.contest_book_dir.resolve()
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)
