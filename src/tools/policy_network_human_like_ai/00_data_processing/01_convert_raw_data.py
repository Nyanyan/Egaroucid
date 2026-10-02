#!/usr/bin/env python3
"""Convert one raw dataset to numbered, normalized transcript text files."""

import argparse
import json

from pipeline_common import (TranscriptWriter, add_dataset_arguments, dataset_directory,
                             file_info, output_directory, write_manifest)
from raw_formats import RAW_FORMATS


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--format", required=True, choices=sorted(RAW_FORMATS))
    parser.add_argument("--games-per-file", type=int, default=10000)
    return parser


def run(args):
    source = dataset_directory(args.data_root, "raw_data", args.dataset)
    destination = dataset_directory(args.data_root, "transcript", args.dataset)
    pattern, reader = RAW_FORMATS[args.format]
    files = sorted(source.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} files in {source}")
    raw_files = [file_info(p, source) for p in files]
    with output_directory(destination, args.replace) as output:
        writer = TranscriptWriter(output, args.games_per_file)
        try:
            with (output / "games.jsonl").open("w", encoding="utf-8") as provenance:
                for path in files:
                    for transcript, row in reader(path):
                        game_id = writer.games
                        location = writer.write(game_id, transcript)
                        provenance.write(json.dumps({"game_id": game_id, **location,
                            "source_file": path.relative_to(source).as_posix(), "source_row": row}) + "\n")
                    print(f"converted {path.name}: {writer.games:,} games total", flush=True)
        finally:
            writer.close()
        if writer.games == 0:
            raise ValueError("no games found")
        if raw_files != [file_info(p, source) for p in files]:
            raise ValueError("raw files changed during conversion")
        result = write_manifest(output, "transcript", args.dataset, raw_format=args.format,
                                source_directory=str(source), raw_files=raw_files, games=writer.games,
                                text_format="one lowercase coordinate transcript per line; implicit passes",
                                game_ids_format="uint64 little-endian; one per transcript")
    return result


if __name__ == "__main__":
    result = run(make_argparser().parse_args())
    print(f"transcript: {result['games']:,} games")
