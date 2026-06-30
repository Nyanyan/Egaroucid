import argparse
import subprocess
from pathlib import Path

from config import (
    CONSOLE_EXE,
    DEFAULT_CUT_EMPTY,
    DEFAULT_GAMES_PER_START,
    DEFAULT_LEVEL,
    DEFAULT_MAX_LOSS_PER_MOVE,
    DEFAULT_MAX_LOSS_TOTAL,
    DEFAULT_THREADS,
    record_dir_for_start,
)
from othello import normalize_board_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_board")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_START)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--exe", type=Path, default=CONSOLE_EXE)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--max-loss-per-move", type=int, default=DEFAULT_MAX_LOSS_PER_MOVE)
    parser.add_argument("--max-loss-total", type=int, default=DEFAULT_MAX_LOSS_TOTAL)
    parser.add_argument("--cut-empty", type=int, default=DEFAULT_CUT_EMPTY)
    parser.add_argument("--use-existing-book", action="store_true")
    args = parser.parse_args()

    initial_board = normalize_board_text(args.initial_board)
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.max_loss_per_move < 0 or args.max_loss_total < 0:
        raise ValueError("loss limits must be non-negative")
    out_dir = args.out_dir or record_dir_for_start(initial_board)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.exe),
        "-l", str(args.level),
        "-thread", str(args.threads),
        "-contestrecord",
        initial_board,
        str(args.games),
        str(out_dir),
        str(args.max_loss_per_move),
        str(args.max_loss_total),
        str(args.cut_empty),
    ]
    if not args.use_existing_book:
        cmd.insert(1, "-nobook")
    subprocess.run(cmd, cwd=CONSOLE_EXE.parents[1], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
