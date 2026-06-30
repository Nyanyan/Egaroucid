import argparse
import subprocess
import sys

from config import (
    DEFAULT_CUT_EMPTY,
    DEFAULT_GAMES_PER_START,
    DEFAULT_LEVEL,
    DEFAULT_MAX_LOSS_PER_MOVE,
    DEFAULT_MAX_LOSS_TOTAL,
    DEFAULT_THREADS,
    WORK_DIR,
    iter_start_boards,
    record_dir_for_start,
)
from othello import normalize_board_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_START)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--max-loss-per-move", type=int, default=DEFAULT_MAX_LOSS_PER_MOVE)
    parser.add_argument("--max-loss-total", type=int, default=DEFAULT_MAX_LOSS_TOTAL)
    parser.add_argument("--cut-empty", type=int, default=DEFAULT_CUT_EMPTY)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    boards = list(iter_start_boards())
    if args.skip:
        boards = boards[args.skip:]
    if args.limit is not None:
        boards = boards[:args.limit]

    script = WORK_DIR / "generate_records.py"
    for idx, board in enumerate(boards, start=args.skip):
        initial_board = normalize_board_text(board)
        out_dir = record_dir_for_start(initial_board)
        if args.resume and out_dir.exists() and any(out_dir.glob("*.txt")):
            print(f"[{idx}] skip existing {initial_board}")
            continue
        print(f"[{idx}] generate {initial_board}")
        cmd = [
            sys.executable,
            str(script),
            initial_board,
            "--games", str(args.games),
            "--level", str(args.level),
            "--threads", str(args.threads),
            "--max-loss-per-move", str(args.max_loss_per_move),
            "--max-loss-total", str(args.max_loss_total),
            "--cut-empty", str(args.cut_empty),
        ]
        subprocess.run(cmd, cwd=WORK_DIR, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
