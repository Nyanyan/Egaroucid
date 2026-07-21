import argparse
import subprocess
import sys
from pathlib import Path

from config import (
    CONSOLE_EXE,
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
    record_dir_for_start,
)
from generate_records import count_unique_records, ensure_generation_manifest
from othello import normalize_board_text


def find_start_index(boards: list[str], start_board: str) -> int:
    normalized_start = normalize_board_text(start_board)
    for idx, board in enumerate(boards):
        if normalize_board_text(board) == normalized_start:
            return idx
    raise ValueError(f"--start-board not found in start list: {normalized_start}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES_PER_START,
        help="target total number of unique records per start (default: %(default)s)",
    )
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
    parser.add_argument(
        "--exe",
        type=Path,
        default=CONSOLE_EXE,
        help="record-generation executable (default: %(default)s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip starts that already have at least --games unique records",
    )
    args = parser.parse_args()
    args.exe = args.exe.resolve()

    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_loss_per_move < 0 or args.max_loss_total < 0:
        raise ValueError("loss limits must be non-negative")
    if args.max_book_loss < 0:
        raise ValueError("--max-book-loss must be non-negative")
    if not (0 <= args.cut_empty < 64):
        raise ValueError("--cut-empty must be in [0, 63]")
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
        if args.resume:
            records_dir = record_dir_for_start(initial_board)
            n_known = count_unique_records(records_dir, initial_board)
            if n_known >= args.games:
                ensure_generation_manifest(records_dir, initial_board)
                print(f"[{idx}] skip complete known={n_known} target={args.games} {initial_board}")
                continue
            print(f"[{idx}] resume known={n_known} target={args.games} {initial_board}")
        else:
            print(f"[{idx}] generate target={args.games} {initial_board}")
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
            "--exe", str(args.exe),
        ]
        if args.use_existing_book:
            cmd.append("--use-existing-book")
        subprocess.run(cmd, cwd=WORK_DIR, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
