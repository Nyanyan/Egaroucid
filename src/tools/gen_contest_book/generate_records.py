import argparse
import hashlib
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
    DEFAULT_ADVERSARIAL_BATCH_SIZE,
    DEFAULT_ADVERSARIAL_ENGINE_WIDTH,
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
    TRAINED_DIR,
    book_path_for_start,
    record_dir_for_start,
)
from othello import normalize_board_text


GENERATION_MANIFEST_SCHEMA = "contest_record_generation_manifest_v1"
GENERATION_LOCK_FILENAME = ".generation.lock"
ADVERSARIAL_GENERATION_MODE = "adversarial_v3"
ADVERSARIAL_PARITY_ORDER = (1, 0)
ADVERSARIAL_STABILIZATION_SCHEMA = "contest_book_adversarial_stabilization_v1"


def adversarial_targets(n_games: int) -> dict[int, int]:
    return {
        0: (n_games + 1) // 2,
        1: n_games // 2,
    }


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


def count_adversarial_records(
    records_dir: Path,
    initial_board: str,
    engine_parity: int | None = None,
) -> int:
    """Count unique adversarial records, optionally for one engine parity."""
    transcripts: set[str] = set()
    if not records_dir.exists():
        return 0

    def flush(block: dict[str, str]) -> None:
        try:
            block_initial = normalize_board_text(block.get("initial board", ""))
        except ValueError:
            return
        if block_initial != initial_board or block.get("generation mode") != ADVERSARIAL_GENERATION_MODE:
            return
        if engine_parity is not None:
            try:
                block_parity = int(block.get("engine parity", ""))
            except ValueError:
                return
            if block_parity != engine_parity:
                return
        transcript = block.get("transcript")
        if transcript is not None:
            transcripts.add(transcript)

    for path in sorted(records_dir.glob("*.txt")):
        block: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    if block:
                        flush(block)
                        block = {}
                    continue
                if ": " in line:
                    key, value = line.split(": ", 1)
                    block[key] = value
            if block:
                flush(block)
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


def run_adversarial_record_batch(
    args: argparse.Namespace,
    initial_board: str,
    out_dir: Path,
    n_games: int,
    engine_parity: int,
) -> None:
    cmd = [str(args.exe)]
    if not args.use_existing_book:
        cmd.append("-nobook")
    cmd.extend([
        "-contestbook", str(TRAINED_DIR),
        "-l", str(args.adversarial_level),
        "-thread", str(args.threads),
        "-contestrecordadv",
        initial_board,
        str(n_games),
        str(out_dir),
        str(args.adversarial_reply_margin),
        str(args.adversarial_reply_width),
        str(args.adversarial_engine_width),
        str(args.cut_empty),
        str(engine_parity),
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


def contest_book_policy_fingerprint(output: Path) -> dict[str, object]:
    """Hash only playable book rows, excluding changing provenance comments."""
    digest = hashlib.sha256()
    rows = 0
    try:
        with output.open("r", encoding="utf-8") as book_file:
            for raw_line in book_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                digest.update(line.encode("utf-8"))
                digest.update(b"\n")
                rows += 1
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot fingerprint contest book {output}: {exc}") from exc
    return {"rows": rows, "sha256": digest.hexdigest()}


def adversarial_generation_profile(args: argparse.Namespace) -> dict[str, object]:
    profile = generation_profile(args)
    settings = profile["settings"]
    if not isinstance(settings, dict):
        raise ValueError("generation profile settings must be an object")
    settings.update({
        "generation_mode": ADVERSARIAL_GENERATION_MODE,
        "adversarial_games_per_round": args.adversarial_games,
        "adversarial_level": args.adversarial_level,
        "adversarial_reply_margin": args.adversarial_reply_margin,
        "adversarial_reply_width": args.adversarial_reply_width,
        "adversarial_engine_width": args.adversarial_engine_width,
        "adversarial_stable_rounds": args.adversarial_stable_rounds,
    })
    return profile


def adversarial_stabilization_is_current(
    records_dir: Path,
    initial_board: str,
    output: Path,
    profile: dict[str, object],
) -> bool:
    """Return whether the final published book has a matching stable sweep."""
    path = generation_manifest_path(records_dir)
    if not path.exists() or not output.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != GENERATION_MANIFEST_SCHEMA
        or manifest.get("initial_board") != initial_board
    ):
        return False
    stabilization = manifest.get("adversarial_stabilization")
    if not isinstance(stabilization, dict):
        return False
    if (
        stabilization.get("schema") != ADVERSARIAL_STABILIZATION_SCHEMA
        or stabilization.get("status") != "stable"
        or stabilization.get("generation_mode") != ADVERSARIAL_GENERATION_MODE
        or stabilization.get("profile") != profile
    ):
        return False
    try:
        return (
            stabilization.get("records") == records_snapshot(records_dir, initial_board)
            and stabilization.get("book") == fingerprint_files([output.resolve()])
            and stabilization.get("policy") == contest_book_policy_fingerprint(output)
        )
    except ValueError:
        return False


def write_adversarial_stabilization(
    manifest: dict[str, object],
    out_dir: Path,
    initial_board: str,
    output: Path,
    profile: dict[str, object],
    status: str,
    rounds: list[dict[str, object]],
    stable_rounds: int,
    new_records: int,
) -> None:
    manifest["adversarial_stabilization"] = {
        "schema": ADVERSARIAL_STABILIZATION_SCHEMA,
        "status": status,
        "generation_mode": ADVERSARIAL_GENERATION_MODE,
        "profile": profile,
        "records": records_snapshot(out_dir, initial_board),
        "book": fingerprint_files([output.resolve()]),
        "policy": contest_book_policy_fingerprint(output),
        "rounds": rounds,
        "stable_rounds": stable_rounds,
        "new_records": new_records,
    }
    write_json_atomically(generation_manifest_path(out_dir), manifest)


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


def adversarial_record_generation_batch(
    manifest: dict[str, object],
    out_dir: Path,
    initial_board: str,
    args: argparse.Namespace,
    profile: dict[str, object],
    output: Path,
    n_batch: int,
    engine_parity: int,
    target_for_parity: int,
    before: dict[str, object],
) -> dict[str, object]:
    batches = manifest["batches"]
    if not isinstance(batches, list):
        raise ValueError("generation manifest batches must be a list")
    pending: dict[str, object] = {
        "sequence": len(batches),
        "profile": profile,
        "generation_mode": ADVERSARIAL_GENERATION_MODE,
        "engine_parity": engine_parity,
        "target_adversarial_records": target_for_parity,
        "requested_records": n_batch,
        "before": before,
        "contest_book": {
            "path": output.resolve().as_posix(),
            "fingerprint": fingerprint_files([output.resolve()]),
        },
    }
    manifest["pending_batch"] = pending
    write_json_atomically(generation_manifest_path(out_dir), manifest)

    result = "completed"
    try:
        run_adversarial_record_batch(
            args,
            initial_board,
            out_dir,
            n_batch,
            engine_parity,
        )
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


def generate_adversarial_to_target(
    args: argparse.Namespace,
    initial_board: str,
    out_dir: Path,
    output: Path,
) -> int:
    """Probe and rebuild until the final book reaches a bounded fixed point."""
    if args.adversarial_games <= 0:
        return count_adversarial_records(out_dir, initial_board)

    targets = adversarial_targets(args.adversarial_games)
    out_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(out_dir / GENERATION_LOCK_FILENAME):
        manifest, snapshot = prepare_generation_manifest(out_dir, initial_board)
        if not can_use_provisional_book(args, initial_board, out_dir, output):
            print(f"build current book before adversarial generation: {output}")
            build_provisional_book(args, initial_board, out_dir, output)
        if not can_use_provisional_book(args, initial_board, out_dir, output):
            raise ValueError(f"cannot build a current contest book for adversarial generation: {output}")

        profile = adversarial_generation_profile(args)
        if adversarial_stabilization_is_current(
            out_dir,
            initial_board,
            output,
            profile,
        ):
            total = count_adversarial_records(out_dir, initial_board)
            print(f"adversarial stabilization already current: records={total}")
            return total

        rounds: list[dict[str, object]] = []
        stable_rounds = 0
        total_new = 0
        previous_stabilization = manifest.get("adversarial_stabilization")
        if (
            isinstance(previous_stabilization, dict)
            and previous_stabilization.get("schema") == ADVERSARIAL_STABILIZATION_SCHEMA
            and previous_stabilization.get("generation_mode") == ADVERSARIAL_GENERATION_MODE
            and previous_stabilization.get("profile") == profile
            and previous_stabilization.get("records") == snapshot
            and previous_stabilization.get("book") == fingerprint_files([output.resolve()])
            and previous_stabilization.get("policy") == contest_book_policy_fingerprint(output)
        ):
            previous_rounds = previous_stabilization.get("rounds")
            if isinstance(previous_rounds, list):
                rounds = [item for item in previous_rounds if isinstance(item, dict)]
            previous_stable_rounds = previous_stabilization.get("stable_rounds")
            if isinstance(previous_stable_rounds, int) and previous_stable_rounds >= 0:
                stable_rounds = previous_stable_rounds
            previous_new_records = previous_stabilization.get("new_records")
            if isinstance(previous_new_records, int) and previous_new_records >= 0:
                total_new = previous_new_records

        for round_offset in range(args.adversarial_max_rounds):
            round_index = len(rounds)
            before_snapshot = snapshot
            before_unique = int(before_snapshot["unique_records"])
            before_book = fingerprint_files([output.resolve()])
            before_policy = contest_book_policy_fingerprint(output)
            parity_results: list[dict[str, int]] = []

            # Both roles deliberately probe the same immutable book snapshot.
            # Rebuilding between roles would validate the second role against
            # a policy that the first role never saw.
            for parity in ADVERSARIAL_PARITY_ORDER:
                n_probes = targets[parity]
                if n_probes <= 0:
                    continue
                if fingerprint_files([output.resolve()]) != before_book:
                    raise ValueError("contest book changed during adversarial stabilization round")
                parity_before = int(snapshot["unique_records"])
                print(
                    "generate adversarial stabilization probes "
                    f"round={round_index} parity={parity} probes={n_probes}"
                )
                snapshot = adversarial_record_generation_batch(
                    manifest,
                    out_dir,
                    initial_board,
                    args,
                    profile,
                    output,
                    n_probes,
                    parity,
                    n_probes,
                    snapshot,
                )
                parity_after = int(snapshot["unique_records"])
                parity_results.append({
                    "parity": parity,
                    "probes": n_probes,
                    "new_records": parity_after - parity_before,
                })

            round_new = int(snapshot["unique_records"]) - before_unique
            total_new += round_new
            if round_new > 0:
                print(
                    "rebuild provisional book after adversarial stabilization round "
                    f"round={round_index} new_records={round_new}"
                )
                build_provisional_book(args, initial_board, out_dir, output)
            after_book = fingerprint_files([output.resolve()])
            after_policy = contest_book_policy_fingerprint(output)
            policy_stable = before_policy == after_policy
            if round_new == 0 and policy_stable:
                stable_rounds += 1
            else:
                stable_rounds = 0
            rounds.append({
                "round": round_index,
                "before_book": before_book,
                "after_book": after_book,
                "before_policy": before_policy,
                "after_policy": after_policy,
                "parities": parity_results,
                "new_records": round_new,
                "policy_stable": policy_stable,
            })

            status = (
                "stable"
                if stable_rounds >= args.adversarial_stable_rounds
                else "running"
            )
            write_adversarial_stabilization(
                manifest,
                out_dir,
                initial_board,
                output,
                profile,
                status,
                rounds,
                stable_rounds,
                total_new,
            )
            if status == "stable":
                total = count_adversarial_records(out_dir, initial_board)
                print(
                    "adversarial stabilization done "
                    f"rounds={round_offset + 1} stable_rounds={stable_rounds} "
                    f"new_records={total_new} adversarial_records={total}"
                )
                return total

        write_adversarial_stabilization(
            manifest,
            out_dir,
            initial_board,
            output,
            profile,
            "budget_exhausted",
            rounds,
            stable_rounds,
            total_new,
        )
        total = count_adversarial_records(out_dir, initial_board)
        print(
            "adversarial stabilization budget exhausted; keep pending for --resume "
            f"rounds={args.adversarial_max_rounds} new_records={total_new} "
            f"adversarial_records={total}"
        )
        return total


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
    parser.add_argument(
        "--adversarial-games",
        type=int,
        default=0,
        help="counterexample probes per stabilization round (default: disabled)",
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
    )
    parser.add_argument(
        "--adversarial-reply-margin",
        type=int,
        default=DEFAULT_ADVERSARIAL_REPLY_MARGIN,
        help="retain opponent replies within this many discs of its best screened move",
    )
    parser.add_argument(
        "--adversarial-reply-width",
        type=int,
        default=DEFAULT_ADVERSARIAL_REPLY_WIDTH,
        help="maximum opponent replies retained at each adversarial branch",
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
        help="maximum fixed-point rounds per invocation (default: %(default)s)",
    )
    parser.add_argument(
        "--adversarial-stable-rounds",
        type=int,
        default=DEFAULT_ADVERSARIAL_STABLE_ROUNDS,
        help="unchanged full sweeps required for certification (default: %(default)s)",
    )
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
    out_dir = args.out_dir or record_dir_for_start(initial_board)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = book_path_for_start(initial_board)
    output.parent.mkdir(parents=True, exist_ok=True)

    generate_to_target(args, initial_board, out_dir, output)
    generate_adversarial_to_target(args, initial_board, out_dir, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
