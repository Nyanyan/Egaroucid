import argparse
import subprocess
import sys

from config import (
    DEFAULT_BOOK_MAX_LOSS,
    DEFAULT_CUT_EMPTY,
    DEFAULT_GAMES_PER_START,
    DEFAULT_LEVEL,
    DEFAULT_MAX_LOSS_PER_MOVE,
    DEFAULT_MAX_LOSS_TOTAL,
    DEFAULT_RECORD_BATCH_SIZE,
    DEFAULT_THREADS,
    WORK_DIR,
    iter_start_boards,
)
from othello import normalize_board_text


def find_start_index(boards: list[str], start_board: str) -> int:
    normalized_start = normalize_board_text(start_board)
    for idx, board in enumerate(boards):
        if normalize_board_text(board) == normalized_start:
            return idx
    raise ValueError(f"--start-board not found in start list: {normalized_start}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_START)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RECORD_BATCH_SIZE)
    parser.add_argument("--start-board")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--max-loss-per-move", type=int, default=DEFAULT_MAX_LOSS_PER_MOVE)
    parser.add_argument("--max-loss-total", type=int, default=DEFAULT_MAX_LOSS_TOTAL)
    parser.add_argument("--max-book-loss", type=int, default=DEFAULT_BOOK_MAX_LOSS)
    parser.add_argument("--cut-empty", type=int, default=DEFAULT_CUT_EMPTY)
    parser.add_argument("--use-existing-book", action="store_true")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.skip < 0:
        raise ValueError("--skip must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    boards = list(iter_start_boards())
    start_idx = 0
    if args.start_board:
        start_idx = find_start_index(boards, args.start_board)
    if args.skip:
        start_idx += args.skip
    boards = boards[start_idx:]
    if args.limit is not None:
        boards = boards[:args.limit]

    script = WORK_DIR / "generate_records.py"
    for idx, board in enumerate(boards, start=start_idx):
        initial_board = normalize_board_text(board)
        print(f"[{idx}] generate {initial_board}")
        cmd = [
            sys.executable,
            str(script),
            initial_board,
            "--games", str(args.games),
            "--batch-size", str(args.batch_size),
            "--level", str(args.level),
            "--threads", str(args.threads),
            "--max-loss-per-move", str(args.max_loss_per_move),
            "--max-loss-total", str(args.max_loss_total),
            "--max-book-loss", str(args.max_book_loss),
            "--cut-empty", str(args.cut_empty),
        ]
        if args.use_existing_book:
            cmd.append("--use-existing-book")
        subprocess.run(cmd, cwd=WORK_DIR, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
