import argparse

from book_artifact import BookBuildSpec, build_book_atomically, build_book_if_stale
from config import (
    DEFAULT_BOOK_MAX_LOSS,
    book_path_for_start,
    iter_start_boards,
    record_dir_for_start,
)
from othello import normalize_board_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-book-loss", type=int, default=DEFAULT_BOOK_MAX_LOSS)
    parser.add_argument("--cut-empty", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-game-records", action="store_true")
    args = parser.parse_args()

    if args.skip < 0:
        raise ValueError("--skip must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.max_book_loss < 0:
        raise ValueError("--max-book-loss must be non-negative")
    if args.cut_empty is not None and not (0 <= args.cut_empty < 64):
        raise ValueError("--cut-empty must be in [0, 63]")

    boards = list(iter_start_boards())
    if args.skip:
        boards = boards[args.skip:]
    if args.limit is not None:
        boards = boards[:args.limit]

    for idx, board in enumerate(boards, start=args.skip):
        initial_board = normalize_board_text(board)
        output = book_path_for_start(initial_board)
        spec = BookBuildSpec(
            initial_board=initial_board,
            records_dir=record_dir_for_start(initial_board),
            output=output,
            max_book_loss=args.max_book_loss,
            cut_empty=args.cut_empty,
            include_game_records=not args.no_game_records,
        )
        if args.resume:
            outcome = build_book_if_stale(spec)
            if not outcome.built:
                print(f"[{idx}] skip current {output}")
                continue
            print(f"[{idx}] rebuilt {initial_board} ({outcome.previous_status.reason})")
            metadata = outcome.metadata
        else:
            print(f"[{idx}] build {initial_board}")
            metadata = build_book_atomically(spec)
        if metadata is None:
            raise RuntimeError("book build finished without metadata")
        print(
            f"[{idx}] published {output} "
            f"records={metadata.records_used}/{metadata.records_seen} "
            f"boards={metadata.board_lines} cut_empty={metadata.cut_empty}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
