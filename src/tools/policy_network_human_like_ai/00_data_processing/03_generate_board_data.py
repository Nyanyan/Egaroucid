#!/usr/bin/env python3
"""Expand already split transcript games into 19-byte board records."""

import argparse
from itertools import islice
import json

from board_replay import replay_games
from pipeline_common import (BoardWriter, SPLIT_NAMES, add_dataset_arguments,
    dataset_directory, iter_transcripts, load_manifest, output_directory, source_info, write_manifest)


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--games-per-batch", type=int, default=4096)
    parser.add_argument("--positions-per-file", type=int, default=1000000)
    return parser


def run(args):
    if args.games_per_batch <= 0:
        raise ValueError("games_per_batch must be positive")
    source = dataset_directory(args.data_root, "transcript_split", args.dataset)
    destination = dataset_directory(args.data_root, "board_split", args.dataset)
    original = load_manifest(source, "transcript_split")
    counts, seen = {}, set()
    with output_directory(destination, args.replace) as output:
        with (output / "rejected_games.jsonl").open("w", encoding="utf-8") as rejected_file:
            for name in SPLIT_NAMES:
                writer = BoardWriter(output / name, args.positions_per_file)
                games = iter_transcripts(source / name)
                processed = rejected_count = 0
                try:
                    while True:
                        batch = list(islice(games, args.games_per_batch))
                        if not batch:
                            break
                        for game_id, _, _, _ in batch:
                            if game_id in seen:
                                raise ValueError(f"repeated game ID across transcript splits: {game_id}")
                            seen.add(game_id)
                        boards, ids, rejected = replay_games([(g[0], g[1]) for g in batch])
                        for entry in rejected:
                            rejected_file.write(json.dumps({"split": name, **entry}) + "\n")
                        writer.write(boards, ids)
                        processed += len(batch)
                        rejected_count += len(rejected)
                        print(f"{name}: {processed:,} games, {writer.positions:,} positions, {rejected_count:,} rejected", flush=True)
                finally:
                    writer.close()
                if processed != original["splits"][name]["games"]:
                    raise ValueError(f"transcript count mismatch for {name}")
                if not writer.positions:
                    raise ValueError(f"no valid games in {name}")
                counts[name] = {"input_games": processed, "games": processed - rejected_count,
                                "rejected_games": rejected_count, "positions": writer.positions}
        result = write_manifest(output, "board_split", args.dataset, source=source_info(source),
            seed=original["seed"], ratios=original["ratios"], split_unit=original["split_unit"],
            record_format="<QQbbb: player, opponent, color, policy, score", record_bytes=19,
            game_ids_format="uint64 little-endian; original game ID per position",
            score_convention="terminal disc difference with empties awarded to winner", splits=counts)
    return result


if __name__ == "__main__":
    result = run(make_argparser().parse_args())
    print("board_split:", result["splits"])
