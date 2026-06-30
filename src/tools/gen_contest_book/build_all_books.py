import argparse
import subprocess
import sys

from config import DEFAULT_BOOK_MAX_LOSS, WORK_DIR, book_path_for_start, iter_start_boards
from othello import normalize_board_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-book-loss", type=int, default=DEFAULT_BOOK_MAX_LOSS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-game-records", action="store_true")
    args = parser.parse_args()

    boards = list(iter_start_boards())
    if args.skip:
        boards = boards[args.skip:]
    if args.limit is not None:
        boards = boards[:args.limit]

    script = WORK_DIR / "build_book.py"
    for idx, board in enumerate(boards, start=args.skip):
        initial_board = normalize_board_text(board)
        output = book_path_for_start(initial_board)
        if args.resume and output.exists():
            print(f"[{idx}] skip existing {output}")
            continue
        print(f"[{idx}] build {initial_board}")
        cmd = [
            sys.executable,
            str(script),
            initial_board,
            "--max-book-loss", str(args.max_book_loss),
        ]
        if args.no_game_records:
            cmd.append("--no-game-records")
        subprocess.run(cmd, cwd=WORK_DIR, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
