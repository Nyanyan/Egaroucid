#!/usr/bin/env python3
"""Split Quest games globally, then route each move by its player's rating."""

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from board_replay import replay_games
from pipeline_common import (BoardWriter, SPLIT_NAMES, TranscriptWriter,
                             dataset_directory, iter_transcripts, load_manifest,
                             output_directory, source_info, write_manifest)
from raw_formats import normalize_transcript
from wthor_dataset import file_sha256, split_counts


MIN_RATING = 800
BAND_WIDTH = 200
UNKNOWN_SCORE = -128
FILENAME = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[^()]+Z)"
    r"\((?P<quest_game_id>[^()]+)\)"
    r"(?P<black_name>[^()]+)\((?P<black_rating>\d+)\)-"
    r"(?P<white_name>[^()]+)\((?P<white_rating>\d+)\)$"
)


@dataclass
class QuestGame:
    timestamp: str
    quest_game_id: str
    black_name: str
    black_rating: int
    white_name: str
    white_rating: int
    transcript: str
    source_files: list = field(default_factory=list)


def parse_filename(path):
    match = FILENAME.fullmatch(Path(path).stem)
    if not match:
        raise ValueError(f"cannot read Quest game ID and black/white ratings: {path}")
    result = match.groupdict()
    for color in ("black", "white"):
        result[color + "_rating"] = int(result[color + "_rating"])
    return result


def rating_band(rating):
    if rating < MIN_RATING:
        return "under_800"
    lower = MIN_RATING + (rating - MIN_RATING) // BAND_WIDTH * BAND_WIDTH
    return f"{lower}-{lower + BAND_WIDTH}"


def band_bounds(band):
    return (None, MIN_RATING) if band == "under_800" else tuple(map(int, band.split("-")))


def same_range_band(rating):
    return "2200_plus" if rating >= 2200 else rating_band(rating)


def extract_same_range_transcripts(args):
    """Select existing Quest games without changing their global IDs or splits.

    Write the numbered pipeline's transcript and transcript_split contracts.
    Board conversion remains the responsibility of 03_generate_board_data.py.
    """
    if args.data_root is None:
        raise ValueError("--same-range-source requires --data-root or POLICY_DATA_ROOT")
    if args.input_dir is not None or args.output_dir is not None or args.complete_games_only:
        raise ValueError("--same-range-source uses --data-root for output; "
                         "choose unfinished-game handling in 03_generate_board_data.py")
    if args.games_per_file <= 0:
        raise ValueError("games_per_file must be positive")
    source = args.same_range_source.resolve()
    original = load_manifest(source, "transcript_split")
    provenance = source_info(source)
    metadata, quest_ids, split_by_transcript = {}, set(), {}
    selected = {}
    started = time.monotonic()
    with (source / "games.jsonl").open(encoding="utf-8") as stream:
        for done, line in enumerate(stream, 1):
            entry = json.loads(line)
            game_id, quest_id = entry["game_id"], entry["quest_game_id"]
            if not isinstance(game_id, int) or game_id < 0 or game_id in metadata or quest_id in quest_ids:
                raise ValueError("duplicate or invalid Quest game ID in source metadata")
            transcript = normalize_quest_transcript(entry["transcript"])
            if transcript != entry["transcript"] or entry["split"] not in SPLIT_NAMES:
                raise ValueError("invalid normalized transcript or source split")
            if split_by_transcript.setdefault(transcript, entry["split"]) != entry["split"]:
                raise ValueError("identical transcripts cross source splits")
            metadata[game_id] = entry
            quest_ids.add(quest_id)
            black_band = same_range_band(entry["black_rating"])
            if black_band == same_range_band(entry["white_rating"]):
                selected.setdefault(black_band, []).append(entry)
            if done % 20000 == 0 or done == original["games"]:
                elapsed = time.monotonic() - started
                eta = elapsed * max(0, original["games"] - done) / done
                print(f"same-range selection: {done:,}/{original['games']:,} games; "
                      f"elapsed {elapsed:.1f}s, ETA {eta:.1f}s", flush=True)
    if len(metadata) != original["games"]:
        raise ValueError("source metadata game count differs from manifest")
    seen = set()
    for split in SPLIT_NAMES:
        count = 0
        for game_id, transcript, _, _ in iter_transcripts(source / split):
            entry = metadata.get(game_id)
            if (game_id in seen or entry is None or entry["split"] != split
                    or entry["transcript"] != transcript):
                raise ValueError("source transcript, metadata and inherited split disagree")
            seen.add(game_id)
            count += 1
        if count != original["splits"][split]["games"]:
            raise ValueError(f"source transcript count mismatch for {split}")
    if seen != set(metadata) or not selected:
        raise ValueError("source metadata coverage mismatch or no same-range games")
    # Validate every output before publishing the first new dataset.
    for band in selected:
        for stage in ("transcript", "transcript_split"):
            destination = dataset_directory(args.data_root, stage, "OQ_samerange_" + band)
            if source == destination or source in destination.parents or destination in source.parents:
                raise ValueError("input and output directories must not contain one another")
            if destination.exists() and any(destination.iterdir()) and not args.replace:
                raise FileExistsError(f"output exists: {destination}")
    results = {}
    for done, (band, entries) in enumerate(sorted(selected.items(), key=lambda item:
            2200 if item[0] == "2200_plus" else band_bounds(item[0])[1] - 1), 1):
        entries.sort(key=lambda entry: entry["game_id"])
        dataset = "OQ_samerange_" + band
        low, high = (2200, None) if band == "2200_plus" else band_bounds(band)
        settings = dict(seed=original["seed"], ratios=original["ratios"],
                        split_unit=original["split_unit"], inherited_split_source=provenance,
                        rating_range=dict(min_inclusive=low, max_exclusive=high),
                        player_filter="both players' ratings fall in the same rating range",
                        game_ids_format="uint64 little-endian; original global Quest game ID")
        destination = dataset_directory(args.data_root, "transcript", dataset)
        with output_directory(destination, args.replace) as output:
            writer = TranscriptWriter(output, args.games_per_file)
            try:
                with (output / "games.jsonl").open("w", encoding="utf-8") as stream:
                    for entry in entries:
                        location = writer.write(entry["game_id"], entry["transcript"])
                        stream.write(json.dumps({**entry, **location}, ensure_ascii=False) + "\n")
            finally:
                writer.close()
            write_manifest(output, "transcript", dataset, source=provenance, **settings,
                           games=len(entries), unique_transcripts=len({e["transcript"] for e in entries}),
                           text_format="one lowercase coordinate transcript per line; implicit passes")
        split_destination = dataset_directory(args.data_root, "transcript_split", dataset)
        with output_directory(split_destination, args.replace) as output:
            writers = {split: TranscriptWriter(output / split, args.games_per_file) for split in SPLIT_NAMES}
            try:
                for entry in entries:
                    writers[entry["split"]].write(entry["game_id"], entry["transcript"])
            finally:
                for writer in writers.values():
                    writer.close()
            counts = {split: dict(games=writers[split].games,
                                  positions=sum(len(e["transcript"]) // 2 for e in entries if e["split"] == split),
                                  unique_transcripts=len({e["transcript"] for e in entries if e["split"] == split}))
                      for split in SPLIT_NAMES}
            write_manifest(output, "transcript_split", dataset, source=source_info(destination),
                           **settings, games=len(entries),
                           unique_transcripts=len({e["transcript"] for e in entries}),
                           splits=counts)
        results[dataset] = dict(games=len(entries), splits=counts,
                                positions=sum(c["positions"] for c in counts.values()))
        elapsed = time.monotonic() - started
        eta = elapsed * (len(selected) - done) / done
        print(f"same-range transcripts: {done}/{len(selected)} datasets; {dataset}: "
              f"{len(entries):,} games; elapsed {elapsed:.1f}s, ETA {eta:.1f}s", flush=True)
    if source_info(source) != provenance:
        raise ValueError("source manifest changed during extraction")
    return dict(source=provenance, datasets=results)


def normalize_quest_transcript(value):
    # Older Quest exports separate coordinates with ';' and mark passes as '-'.
    # Passes are reconstructed from the board, as for compact coordinate files.
    value = "".join(value.split()).lower()
    if ";" in value:
        tokens = value.strip(";").split(";")
        if any(token != "-" and not re.fullmatch(r"[a-h][1-8]", token) for token in tokens):
            raise ValueError("expected semicolon-separated coordinates or '-' pass markers")
        value = "".join(token for token in tokens if token != "-")
    return normalize_transcript(value)


def read_games(source):
    """Deduplicate Quest IDs, requiring all copies' metadata and moves to agree."""
    files = sorted(source.rglob("*.oth"))
    if not files:
        raise FileNotFoundError(f"no .oth files in {source}")
    games, inputs = {}, []
    for index, path in enumerate(files, 1):
        metadata = parse_filename(path)
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"input changed while reading: {path}")
        try:
            transcript = normalize_quest_transcript(raw.decode("utf-8-sig"))
        except (UnicodeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error
        relative = path.relative_to(source).as_posix()
        inputs.append({"path": relative, "size": len(raw), "mtime_ns": after.st_mtime_ns,
                       "sha256": hashlib.sha256(raw).hexdigest()})
        key = metadata["quest_game_id"]
        candidate = QuestGame(**metadata, transcript=transcript)
        if key in games:
            existing = games[key]
            if (existing.transcript != transcript or
                    any(getattr(existing, name) != value for name, value in metadata.items())):
                raise ValueError(f"conflicting copies of Quest game {key}: "
                                 f"{existing.source_files[0]} and {relative}")
            existing.source_files.append(relative)
        else:
            candidate.source_files.append(relative)
            games[key] = candidate
        if index % 10000 == 0 or index == len(files):
            print(f"read: {index:,}/{len(files):,} files, {len(games):,} distinct games", flush=True)
    # Numeric IDs and splitting are independent of the input folder names/order.
    return [games[key] for key in sorted(games)], inputs


def assign_splits(games, seed, ratios):
    transcripts = sorted({game.transcript for game in games})
    counts = split_counts(len(transcripts), *ratios)
    order = np.random.default_rng(seed).permutation(len(transcripts))
    by_transcript, start = {}, 0
    for split, count in enumerate(counts):
        for index in order[start:start + count]:
            by_transcript[transcripts[index]] = split
        start += count
    return np.array([by_transcript[g.transcript] for g in games], dtype=np.int8), counts


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_transcripts(output, games, assignment, group_counts, settings, games_per_file):
    directory = output / "transcript_split" / "all"
    writers = {name: TranscriptWriter(directory / name, games_per_file) for name in SPLIT_NAMES}
    try:
        for game_id, game in enumerate(games):
            writers[SPLIT_NAMES[assignment[game_id]]].write(game_id, game.transcript)
    finally:
        for writer in writers.values():
            writer.close()
    write_jsonl(directory / "games.jsonl", (
        {"game_id": game_id, **asdict(game), "split": SPLIT_NAMES[assignment[game_id]],
         "black_band": rating_band(game.black_rating), "white_band": rating_band(game.white_rating)}
        for game_id, game in enumerate(games)))
    return write_manifest(directory, "transcript_split", "all", **settings, games=len(games),
        unique_transcripts=sum(group_counts),
        splits={name: {"games": writers[name].games, "unique_transcripts": group_counts[i]}
                for i, name in enumerate(SPLIT_NAMES)})


def empty_counts():
    return {"games": 0, "positions": 0, "black_positions": 0, "white_positions": 0,
            "unfinished_games": 0, "unfinished_positions": 0}


def generate_boards(output, games, assignment, args):
    writers, band_counts = {}, {}
    totals = {name: {"input_games": int(np.count_nonzero(assignment == i)),
                    "games": 0, "rejected_games": 0, "positions": 0,
                    "unfinished_games": 0, "unfinished_positions": 0,
                    "under_800_positions": 0}
              for i, name in enumerate(SPLIT_NAMES)}
    try:
        with (output / "rejected_games.jsonl").open("w", encoding="utf-8") as rejected_file:
            for start in range(0, len(games), args.games_per_batch):
                batch = games[start:start + args.games_per_batch]
                boards, ids, rejected = replay_games(
                    [(start + i, game.transcript) for i, game in enumerate(batch)],
                    allow_unfinished=not args.complete_games_only)
                for entry in rejected:
                    split = SPLIT_NAMES[assignment[entry["game_id"]]]
                    totals[split]["rejected_games"] += 1
                    rejected_file.write(json.dumps({**entry, "split": split,
                        "quest_game_id": games[entry["game_id"]].quest_game_id}) + "\n")
                offsets = (ids - start).astype(np.intp)
                black = np.array([g.black_rating for g in batch], dtype=np.int64)
                white = np.array([g.white_rating for g in batch], dtype=np.int64)
                ratings = np.where(boards["color"] == 0, black[offsets], white[offsets])
                bands = np.where(ratings < MIN_RATING, -1,
                                 MIN_RATING + (ratings - MIN_RATING) // BAND_WIDTH * BAND_WIDTH)
                splits = assignment[ids.astype(np.intp)]
                unfinished = boards["score"] == UNKNOWN_SCORE
                for i, name in enumerate(SPLIT_NAMES):
                    selected = splits == i
                    totals[name]["games"] += len(np.unique(ids[selected]))
                    totals[name]["positions"] += int(selected.sum())
                    totals[name]["unfinished_games"] += len(np.unique(ids[selected & unfinished]))
                    totals[name]["unfinished_positions"] += int((selected & unfinished).sum())
                    totals[name]["under_800_positions"] += int((selected & (ratings < MIN_RATING)).sum())
                for lower in np.unique(bands):
                    band = rating_band(int(lower))
                    if band not in writers:
                        writers[band] = {name: BoardWriter(output / "board_split" / band / name,
                                                          args.positions_per_file) for name in SPLIT_NAMES}
                        band_counts[band] = {name: empty_counts() for name in SPLIT_NAMES}
                    for i, name in enumerate(SPLIT_NAMES):
                        selected = (bands == lower) & (splits == i)
                        count = band_counts[band][name]
                        writers[band][name].write(boards[selected], ids[selected])
                        count["games"] += len(np.unique(ids[selected]))
                        count["positions"] += int(selected.sum())
                        count["black_positions"] += int((selected & (boards["color"] == 0)).sum())
                        count["white_positions"] += int((selected & (boards["color"] == 1)).sum())
                        count["unfinished_games"] += len(np.unique(ids[selected & unfinished]))
                        count["unfinished_positions"] += int((selected & unfinished).sum())
                print(f"replayed: {start + len(batch):,}/{len(games):,} games", flush=True)
    finally:
        for split_writers in writers.values():
            for writer in split_writers.values():
                writer.close()
    if not band_counts:
        raise ValueError("no legal positions")
    if sum(c["positions"] for splits in band_counts.values() for c in splits.values()) != sum(
            c["positions"] for c in totals.values()):
        raise ValueError("rating routing did not conserve the number of positions")
    return band_counts, totals


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("POLICY_DATA_ROOT"),
                        help="Root containing raw_data; or set POLICY_DATA_ROOT.")
    parser.add_argument("--input-dir", type=Path, help="Default: <data-root>/raw_data; read .oth recursively.")
    parser.add_argument("--output-dir", type=Path, help="Default: <data-root>/othello_quest_split.")
    parser.add_argument("--same-range-source", type=Path,
                        help="Extract same-band games from an existing transcript_split/all directory; "
                             "preserve its IDs/splits and write OQ_samerange_* transcript datasets under data-root.")
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--train-ratio", type=float, default=.90)
    parser.add_argument("--val-ratio", type=float, default=.05)
    parser.add_argument("--test-ratio", type=float, default=.05)
    parser.add_argument("--games-per-file", type=int, default=10000)
    parser.add_argument("--games-per-batch", type=int, default=4096)
    parser.add_argument("--positions-per-file", type=int, default=1000000)
    parser.add_argument("--complete-games-only", action="store_true",
                        help="Reject unfinished games, as in the WTHOR pipeline; default keeps them with score -128.")
    parser.add_argument("--replace", action="store_true", help="Back up the previous complete output before replacing it.")
    return parser


def run(args):
    if args.same_range_source is not None:
        return extract_same_range_transcripts(args)
    root = Path(args.data_root).resolve() if args.data_root else None
    if root is None and (args.input_dir is None or args.output_dir is None):
        raise ValueError("set POLICY_DATA_ROOT / --data-root, or pass both --input-dir and --output-dir")
    source = (args.input_dir if args.input_dir is not None else root / "raw_data").resolve()
    destination = (args.output_dir if args.output_dir is not None else root / "othello_quest_split").resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("input and output directories must not contain one another")
    if min(args.games_per_file, args.games_per_batch, args.positions_per_file) <= 0:
        raise ValueError("batch and output file limits must be positive")
    ratios = np.array([args.train_ratio, args.val_ratio, args.test_ratio], dtype=float)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0) or not np.isfinite(ratios.sum()):
        raise ValueError("all split ratios must be finite and positive")
    np.random.default_rng(args.seed)  # Validate before reading all input files.
    settings = {"seed": args.seed, "ratios": dict(zip(SPLIT_NAMES, (ratios / ratios.sum()).tolist())),
                "split_unit": "identical_transcript_group_across_all_rating_bands"}
    with output_directory(destination, args.replace) as output:
        games, inputs = read_games(source)
        assignment, group_counts = assign_splits(games, args.seed, ratios)
        write_jsonl(output / "inputs.jsonl", inputs)
        save_transcripts(output, games, assignment, group_counts, settings, args.games_per_file)
        band_counts, totals = generate_boards(output, games, assignment, args)
        transcript_source = {"directory": str(destination / "transcript_split" / "all"),
            "manifest_sha256": file_sha256(output / "transcript_split" / "all" / "manifest.json")}
        for band, counts in sorted(band_counts.items(), key=lambda item: band_bounds(item[0])[1]):
            lower, upper = band_bounds(band)
            write_manifest(output / "board_split" / band, "board_split", band,
                **settings, source=transcript_source, splits=counts,
                rating_range={"min_inclusive": lower, "max_exclusive": upper},
                player_filter="rating of the actual side to move; black=0, white=1",
                record_format="<QQbbb: player, opponent, color, policy, score", record_bytes=19,
                game_ids_format="uint64 little-endian; global Quest game ID shared by all bands",
                score_convention="terminal disc difference with empties awarded to winner; -128 if unfinished",
                unknown_score=UNKNOWN_SCORE, complete_games_only=args.complete_games_only)
        # Detect input additions, deletions or edits during a long conversion.
        if {p.relative_to(source).as_posix() for p in source.rglob("*.oth")} != {r["path"] for r in inputs}:
            raise ValueError("input file list changed during conversion")
        for record in inputs:
            stat = (source / record["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
                raise ValueError(f"input changed during conversion: {record['path']}")
        summary = write_manifest(output, "othello_quest_split", "othello_quest", **settings,
            source_directory=str(source), raw_files=len(inputs), games=len(games),
            duplicate_file_copies=len(inputs) - len(games), unique_transcripts=sum(group_counts),
            rating_threshold=MIN_RATING, rating_width=BAND_WIDTH, below_threshold_band="under_800",
            complete_games_only=args.complete_games_only,
            splits=totals, bands=band_counts)
    for band, counts in sorted(band_counts.items(), key=lambda item: band_bounds(item[0])[1]):
        print(band, ", ".join(f"{name}: {counts[name]['positions']:,} positions" for name in SPLIT_NAMES))
        empty = [name for name in SPLIT_NAMES if not counts[name]["positions"]]
        if empty:
            print(f"NOTE: {band} has empty splits: {', '.join(empty)}; cannot train this band with the existing three-way loader.")
    print("output:", destination)
    return summary


if __name__ == "__main__":
    run(make_argparser().parse_args())
