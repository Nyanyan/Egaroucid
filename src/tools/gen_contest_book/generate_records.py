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
    TRAINED_DIR,
    WORK_DIR,
    book_path_for_start,
    record_dir_for_start,
)
from othello import normalize_board_text


def count_unique_records(records_dir: Path, initial_board: str) -> int:
    initial_prefix = "initial board: "
    transcript_prefix = "transcript: "
    transcripts: set[str] = set()
    if not records_dir.exists():
        return 0
    for path in sorted(records_dir.glob("*.txt")):
        block_matches = False
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    block_matches = False
                    continue
                if line.startswith(initial_prefix):
                    try:
                        block_matches = normalize_board_text(line[len(initial_prefix):]) == initial_board
                    except ValueError:
                        block_matches = False
                    continue
                if block_matches and line.startswith(transcript_prefix):
                    transcripts.add(line[len(transcript_prefix):].strip())
    return len(transcripts)


def run_record_batch(args: argparse.Namespace, initial_board: str, out_dir: Path, n_games: int, use_contest_book: bool) -> None:
    cmd = [str(args.exe)]
    if not args.use_existing_book:
        cmd.append("-nobook")
    if use_contest_book:
        cmd.extend(["-contestbook", str(TRAINED_DIR)])
    cmd.extend([
        "-l", str(args.level),
        "-thread", str(args.threads),
        "-contestrecord",
        initial_board,
        str(n_games),
        str(out_dir),
        str(args.max_loss_per_move),
        str(args.max_loss_total),
        str(args.cut_empty),
    ])
    subprocess.run(cmd, cwd=CONSOLE_EXE.parents[1], check=True)


def build_provisional_book(args: argparse.Namespace, initial_board: str, out_dir: Path, output: Path) -> None:
    cmd = [
        sys.executable,
        str(WORK_DIR / "build_book.py"),
        initial_board,
        "--records-dir", str(out_dir),
        "--output", str(output),
        "--max-book-loss", str(args.max_book_loss),
        "--cut-empty", str(args.cut_empty),
    ]
    subprocess.run(cmd, cwd=WORK_DIR, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_board")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_START)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RECORD_BATCH_SIZE)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--exe", type=Path, default=CONSOLE_EXE)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--max-loss-per-move", type=int, default=DEFAULT_MAX_LOSS_PER_MOVE)
    parser.add_argument("--max-loss-total", type=int, default=DEFAULT_MAX_LOSS_TOTAL)
    parser.add_argument("--max-book-loss", type=int, default=DEFAULT_BOOK_MAX_LOSS)
    parser.add_argument("--cut-empty", type=int, default=DEFAULT_CUT_EMPTY)
    parser.add_argument("--use-existing-book", action="store_true")
    args = parser.parse_args()

    initial_board = normalize_board_text(args.initial_board)
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_loss_per_move < 0 or args.max_loss_total < 0:
        raise ValueError("loss limits must be non-negative")
    if args.max_book_loss < 0:
        raise ValueError("--max-book-loss must be non-negative")
    out_dir = args.out_dir or record_dir_for_start(initial_board)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = book_path_for_start(initial_board)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_generated = 0
    n_known_records = count_unique_records(out_dir, initial_board)
    use_contest_book = output.exists()
    while n_generated < args.games:
        n_batch = min(args.batch_size, args.games - n_generated)
        print(
            f"generate batch {n_batch} records "
            f"known={n_known_records} generated_this_run={n_generated}/{args.games} "
            f"contest_book={'on' if use_contest_book else 'off'}"
        )
        run_record_batch(args, initial_board, out_dir, n_batch, use_contest_book)

        n_after = count_unique_records(out_dir, initial_board)
        n_new = n_after - n_known_records
        if n_new <= 0:
            print("no new records were generated; stopping")
            break

        n_generated += n_new
        n_known_records = n_after
        print(f"build provisional book after {n_generated}/{args.games} new records")
        build_provisional_book(args, initial_board, out_dir, output)
        use_contest_book = output.exists()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
