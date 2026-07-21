import argparse
import json
import subprocess
from pathlib import Path

from book_artifact import (
    BookBuildSpec,
    RECORD_GENERATION_MANIFEST_FILENAME,
    build_book_atomically,
    check_book_status,
    file_lock,
    fingerprint_files,
    locked_book_status,
    write_json_atomically,
)
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
    book_path_for_start,
    record_dir_for_start,
)
from othello import normalize_board_text


GENERATION_MANIFEST_SCHEMA = "contest_record_generation_manifest_v1"
GENERATION_LOCK_FILENAME = ".generation.lock"


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


def provisional_book_spec(
    args: argparse.Namespace,
    initial_board: str,
    out_dir: Path,
    output: Path,
) -> BookBuildSpec:
    return BookBuildSpec(
        initial_board=initial_board,
        records_dir=out_dir,
        output=output,
        max_book_loss=args.max_book_loss,
        cut_empty=args.cut_empty,
    )


def build_provisional_book(args: argparse.Namespace, initial_board: str, out_dir: Path, output: Path) -> None:
    build_book_atomically(provisional_book_spec(args, initial_board, out_dir, output))


def can_use_provisional_book(
    args: argparse.Namespace,
    initial_board: str,
    out_dir: Path,
    output: Path,
) -> bool:
    status = check_book_status(provisional_book_spec(args, initial_board, out_dir, output))
    if not status.current:
        print(f"ignore non-current provisional book {output}: {status.reason}")
        return False
    return True


def generation_manifest_path(out_dir: Path) -> Path:
    return out_dir / RECORD_GENERATION_MANIFEST_FILENAME


def records_snapshot(out_dir: Path, initial_board: str) -> dict[str, object]:
    record_paths = sorted(out_dir.glob("*.txt")) if out_dir.exists() else []
    return {
        "unique_records": count_unique_records(out_dir, initial_board),
        "files": fingerprint_files(record_paths),
    }


def generation_profile(args: argparse.Namespace) -> dict[str, object]:
    executable = args.exe.resolve()
    return {
        "executable": {
            "path": executable.as_posix(),
            "fingerprint": fingerprint_files([executable]),
        },
        "driver": fingerprint_files([
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name("config.py"),
        ]),
        "settings": {
            "level": args.level,
            "threads": args.threads,
            "max_loss_per_move": args.max_loss_per_move,
            "max_loss_total": args.max_loss_total,
            "cut_empty": args.cut_empty,
            "use_existing_book": args.use_existing_book,
        },
    }


def prepare_generation_manifest(
    out_dir: Path,
    initial_board: str,
) -> tuple[dict[str, object], dict[str, object]]:
    path = generation_manifest_path(out_dir)
    current = records_snapshot(out_dir, initial_board)
    if not path.exists():
        manifest: dict[str, object] = {
            "schema": GENERATION_MANIFEST_SCHEMA,
            "initial_board": initial_board,
            "baseline": current,
            "batches": [],
            "pending_batch": None,
        }
        write_json_atomically(path, manifest)
        return manifest, current

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid generation manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid generation manifest {path}: top level is not an object")
    if manifest.get("schema") != GENERATION_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported generation manifest schema in {path}")
    if manifest.get("initial_board") != initial_board:
        raise ValueError(f"generation manifest initial-board mismatch in {path}")
    batches = manifest.get("batches")
    if not isinstance(batches, list):
        raise ValueError(f"invalid generation manifest batches in {path}")

    changed = False
    pending = manifest.get("pending_batch")
    if pending is not None:
        if not isinstance(pending, dict):
            raise ValueError(f"invalid pending batch in {path}")
        recovered = dict(pending)
        recovered["after"] = current
        recovered["result"] = "interrupted_or_unobserved"
        batches.append(recovered)
        manifest["pending_batch"] = None
        changed = True

    previous = manifest.get("baseline")
    if not isinstance(previous, dict):
        raise ValueError(f"invalid generation manifest baseline in {path}")
    if batches:
        last_batch = batches[-1]
        if not isinstance(last_batch, dict) or not isinstance(last_batch.get("after"), dict):
            raise ValueError(f"invalid completed batch in {path}")
        previous = last_batch["after"]
    if previous != current:
        batches.append({
            "sequence": len(batches),
            "result": "external_or_unattributed_change",
            "before": previous,
            "after": current,
        })
        changed = True
    if changed:
        write_json_atomically(path, manifest)
    return manifest, current


def ensure_generation_manifest(out_dir: Path, initial_board: str) -> None:
    """Create/reconcile provenance without launching the generation engine."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(out_dir / GENERATION_LOCK_FILENAME):
        prepare_generation_manifest(out_dir, initial_board)


def record_generation_batch(
    manifest: dict[str, object],
    out_dir: Path,
    initial_board: str,
    args: argparse.Namespace,
    profile: dict[str, object],
    output: Path,
    n_batch: int,
    before: dict[str, object],
    use_contest_book: bool,
) -> dict[str, object]:
    batches = manifest["batches"]
    if not isinstance(batches, list):
        raise ValueError("generation manifest batches must be a list")
    contest_book = None
    if use_contest_book:
        contest_book = {
            "path": output.resolve().as_posix(),
            "fingerprint": fingerprint_files([output.resolve()]),
        }
    pending: dict[str, object] = {
        "sequence": len(batches),
        "profile": profile,
        "target_total": args.games,
        "requested_records": n_batch,
        "before": before,
        "contest_book": contest_book,
    }
    manifest["pending_batch"] = pending
    write_json_atomically(generation_manifest_path(out_dir), manifest)

    result = "completed"
    try:
        run_record_batch(args, initial_board, out_dir, n_batch, use_contest_book)
    except BaseException as exc:
        result = f"failed:{type(exc).__name__}"
        raise
    finally:
        after = records_snapshot(out_dir, initial_board)
        completed = dict(pending)
        completed["after"] = after
        completed["result"] = result
        batches.append(completed)
        manifest["pending_batch"] = None
        write_json_atomically(generation_manifest_path(out_dir), manifest)
    return after


def generate_to_target(
    args: argparse.Namespace,
    initial_board: str,
    out_dir: Path,
    output: Path,
) -> int:
    """Generate until the directory contains --games unique records in total."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(out_dir / GENERATION_LOCK_FILENAME):
        manifest, snapshot = prepare_generation_manifest(out_dir, initial_board)
        n_known_records = int(snapshot["unique_records"])
        n_at_start = n_known_records
        if n_known_records >= args.games:
            print(f"target already satisfied: known={n_known_records} target={args.games}")
            return n_known_records

        profile = generation_profile(args)
        while n_known_records < args.games:
            n_batch = min(args.batch_size, args.games - n_known_records)
            spec = provisional_book_spec(args, initial_board, out_dir, output)
            with locked_book_status(spec) as book_status:
                use_contest_book = book_status.current
                if not use_contest_book:
                    print(f"ignore non-current provisional book {output}: {book_status.reason}")
                print(
                    f"generate batch {n_batch} records "
                    f"known={n_known_records}/{args.games} "
                    f"generated_this_run={n_known_records - n_at_start} "
                    f"contest_book={'on' if use_contest_book else 'off'}"
                )
                snapshot = record_generation_batch(
                    manifest,
                    out_dir,
                    initial_board,
                    args,
                    profile,
                    output,
                    n_batch,
                    snapshot,
                    use_contest_book,
                )

            n_after = int(snapshot["unique_records"])
            n_new = n_after - n_known_records
            if n_new <= 0:
                print("no new records were generated; refresh provisional book and stop")
                build_provisional_book(args, initial_board, out_dir, output)
                break

            n_known_records = n_after
            print(
                f"build provisional book at {n_known_records}/{args.games} total records "
                f"({n_known_records - n_at_start} new this run)"
            )
            build_provisional_book(args, initial_board, out_dir, output)
        return n_known_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_board")
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES_PER_START,
        help="target total number of unique records for this start (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RECORD_BATCH_SIZE)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--exe",
        type=Path,
        default=CONSOLE_EXE,
        help="record-generation executable (default: %(default)s)",
    )
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--max-loss-per-move", type=int, default=DEFAULT_MAX_LOSS_PER_MOVE)
    parser.add_argument("--max-loss-total", type=int, default=DEFAULT_MAX_LOSS_TOTAL)
    parser.add_argument("--max-book-loss", type=int, default=DEFAULT_BOOK_MAX_LOSS)
    parser.add_argument("--cut-empty", type=int, default=DEFAULT_CUT_EMPTY)
    parser.add_argument("--use-existing-book", action="store_true")
    args = parser.parse_args()
    args.exe = args.exe.resolve()

    initial_board = normalize_board_text(args.initial_board)
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
    out_dir = args.out_dir or record_dir_for_start(initial_board)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = book_path_for_start(initial_board)
    output.parent.mkdir(parents=True, exist_ok=True)

    generate_to_target(args, initial_board, out_dir, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
