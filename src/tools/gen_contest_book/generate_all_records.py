import argparse
import subprocess
import sys
from pathlib import Path

from config import (
    CONSOLE_EXE,
    DATA_DIR,
    DEFAULT_ADVERSARIAL_BATCH_SIZE,
    DEFAULT_ADVERSARIAL_ENGINE_WIDTH,
    DEFAULT_ADVERSARIAL_GAMES_PER_START,
    DEFAULT_ADVERSARIAL_LEVEL,
    DEFAULT_ADVERSARIAL_MAX_ROUNDS,
    DEFAULT_ADVERSARIAL_REPLY_MARGIN,
    DEFAULT_ADVERSARIAL_REPLY_WIDTH,
    DEFAULT_ADVERSARIAL_STABLE_ROUNDS,
    DEFAULT_BOOK_MAX_LOSS,
    DEFAULT_CUT_EMPTY,
    DEFAULT_GAMES_PER_START,
    DEFAULT_LEVEL,
    DEFAULT_MAX_LOSS_PER_MOVE,
    DEFAULT_MAX_LOSS_TOTAL,
    DEFAULT_RECORD_BATCH_SIZE,
    DEFAULT_THREADS,
    WORK_DIR,
    book_path_for_start,
    iter_start_boards,
    record_dir_for_start,
)
from generate_records import (
    adversarial_generation_profile,
    adversarial_stabilization_is_current,
    count_adversarial_records,
    count_unique_records,
    ensure_generation_manifest,
)
from othello import normalize_board_text
from r14_random_setup_probability import load_r14_random_setup_priority_manifest


DEFAULT_R14_PRIORITY_MANIFEST = (
    DATA_DIR / "r14_random_setup_probability_priority_20260722.jsonl"
)


def find_start_index(boards: list[str], start_board: str) -> int:
    normalized_start = normalize_board_text(start_board)
    for idx, board in enumerate(boards):
        if normalize_board_text(board) == normalized_start:
            return idx
    raise ValueError(f"--start-board not found in start list: {normalized_start}")


def load_generation_boards(priority_manifest: Path | None) -> list[str]:
    """Return the requested generation order.

    A priority manifest is validated, including its sidecar metadata, before
    any engine process is launched.  Without one, retain the historical order
    of the ordinary start-position files.
    """
    if priority_manifest is None:
        return list(iter_start_boards())
    boards, _metadata, _provenance = load_r14_random_setup_priority_manifest(
        priority_manifest
    )
    return boards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES_PER_START,
        help="target total number of unique records per start (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RECORD_BATCH_SIZE)
    generation_order = parser.add_mutually_exclusive_group()
    generation_order.add_argument(
        "--priority-manifest",
        type=Path,
        default=DEFAULT_R14_PRIORITY_MANIFEST,
        help=(
            "validated r14 probability-priority JSON Lines file; process its "
            "boards from highest to lowest recorded probability "
            f"(default: {DEFAULT_R14_PRIORITY_MANIFEST})"
        ),
    )
    generation_order.add_argument(
        "--start-list-order",
        action="store_true",
        help="use the historical records321_14_random_setup file order instead",
    )
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
        "--adversarial-games",
        type=int,
        default=DEFAULT_ADVERSARIAL_GAMES_PER_START,
        help=(
            "counterexample probes per stabilization round, split across both "
            "engine parities (default: %(default)s; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--adversarial-batch-size",
        type=int,
        default=DEFAULT_ADVERSARIAL_BATCH_SIZE,
        help="deprecated compatibility option; stabilization rebuilds once per round",
    )
    parser.add_argument(
        "--adversarial-level",
        type=int,
        default=DEFAULT_ADVERSARIAL_LEVEL,
        help="screen opponent replies at this level (default: %(default)s)",
    )
    parser.add_argument(
        "--adversarial-reply-margin",
        type=int,
        default=DEFAULT_ADVERSARIAL_REPLY_MARGIN,
        help="retain opponent replies within this many discs of its best move",
    )
    parser.add_argument(
        "--adversarial-reply-width",
        type=int,
        default=DEFAULT_ADVERSARIAL_REPLY_WIDTH,
        help="maximum opponent replies retained at each branch",
    )
    parser.add_argument(
        "--adversarial-engine-width",
        type=int,
        default=DEFAULT_ADVERSARIAL_ENGINE_WIDTH,
        help="maximum close contest-book moves revalidated at each engine turn",
    )
    parser.add_argument(
        "--adversarial-max-rounds",
        type=int,
        default=DEFAULT_ADVERSARIAL_MAX_ROUNDS,
        help="maximum fixed-point rounds per start and invocation",
    )
    parser.add_argument(
        "--adversarial-stable-rounds",
        type=int,
        default=DEFAULT_ADVERSARIAL_STABLE_ROUNDS,
        help="unchanged full sweeps required for certification",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=CONSOLE_EXE,
        help="record-generation executable (default: %(default)s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip starts that satisfy --games and have a current adversarial "
            "stabilization certificate"
        ),
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
    if args.adversarial_games < 0:
        raise ValueError("--adversarial-games must be non-negative")
    if args.adversarial_batch_size <= 0:
        raise ValueError("--adversarial-batch-size must be positive")
    if args.adversarial_level <= 0:
        raise ValueError("--adversarial-level must be positive")
    if args.adversarial_reply_margin < 0:
        raise ValueError("--adversarial-reply-margin must be non-negative")
    if args.adversarial_reply_width <= 0:
        raise ValueError("--adversarial-reply-width must be positive")
    if args.adversarial_engine_width <= 0:
        raise ValueError("--adversarial-engine-width must be positive")
    if args.adversarial_max_rounds <= 0:
        raise ValueError("--adversarial-max-rounds must be positive")
    if args.adversarial_stable_rounds <= 0:
        raise ValueError("--adversarial-stable-rounds must be positive")
    if not (0 <= args.cut_empty < 64):
        raise ValueError("--cut-empty must be in [0, 63]")
    if args.skip < 0:
        raise ValueError("--skip must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    priority_manifest = None if args.start_list_order else args.priority_manifest
    boards = load_generation_boards(priority_manifest)
    start_idx = 0
    if args.start_board:
        start_idx = find_start_index(boards, args.start_board)
    if args.skip:
        start_idx += args.skip
    boards = boards[start_idx:]
    if args.limit is not None:
        boards = boards[:args.limit]

    script = WORK_DIR / "generate_records.py"
    adversarial_profile = (
        adversarial_generation_profile(args)
        if args.resume and args.adversarial_games > 0
        else None
    )
    for idx, board in enumerate(boards, start=start_idx):
        initial_board = normalize_board_text(board)
        if args.resume:
            records_dir = record_dir_for_start(initial_board)
            n_known = count_unique_records(records_dir, initial_board)
            n_adversarial = count_adversarial_records(records_dir, initial_board)
            regular_complete = n_known >= args.games
            adversarial_complete = (
                args.adversarial_games == 0
                or adversarial_stabilization_is_current(
                    records_dir,
                    initial_board,
                    book_path_for_start(initial_board),
                    adversarial_profile,
                )
            )
            if regular_complete and adversarial_complete:
                ensure_generation_manifest(records_dir, initial_board)
                print(
                    f"[{idx}] skip complete known={n_known} target={args.games} "
                    f"adversarial_stable=yes records={n_adversarial} {initial_board}"
                )
                continue
            print(
                f"[{idx}] resume known={n_known} target={args.games} "
                f"adversarial_stable={'yes' if adversarial_complete else 'no'} "
                f"records={n_adversarial} {initial_board}"
            )
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
            "--adversarial-games", str(args.adversarial_games),
            "--adversarial-batch-size", str(args.adversarial_batch_size),
            "--adversarial-level", str(args.adversarial_level),
            "--adversarial-reply-margin", str(args.adversarial_reply_margin),
            "--adversarial-reply-width", str(args.adversarial_reply_width),
            "--adversarial-engine-width", str(args.adversarial_engine_width),
            "--adversarial-max-rounds", str(args.adversarial_max_rounds),
            "--adversarial-stable-rounds", str(args.adversarial_stable_rounds),
            "--exe", str(args.exe),
        ]
        if args.use_existing_book:
            cmd.append("--use-existing-book")
        subprocess.run(cmd, cwd=WORK_DIR, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
