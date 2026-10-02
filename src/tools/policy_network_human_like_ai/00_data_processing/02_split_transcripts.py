#!/usr/bin/env python3
"""Assign complete transcripts to train/val/test before board expansion."""

import argparse

import numpy as np

from pipeline_common import (SPLIT_NAMES, TranscriptWriter, add_dataset_arguments,
    dataset_directory, iter_transcripts, load_manifest, output_directory, source_info, write_manifest)
from wthor_dataset import split_counts


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--train-ratio", type=float, default=.90)
    parser.add_argument("--val-ratio", type=float, default=.05)
    parser.add_argument("--test-ratio", type=float, default=.05)
    parser.add_argument("--games-per-file", type=int, default=10000)
    return parser


def run(args):
    source = dataset_directory(args.data_root, "transcript", args.dataset)
    destination = dataset_directory(args.data_root, "transcript_split", args.dataset)
    original = load_manifest(source, "transcript")
    games = list(iter_transcripts(source))
    if len(games) != original["games"] or len({g[0] for g in games}) != len(games):
        raise ValueError("transcript counts or unique game IDs do not match")
    # Keep repeated copies of exactly the same game in one split, without
    # deleting data. Ratios apply to these groups; row counts can differ slightly.
    groups = {}
    for game_id, transcript, _, _ in games:
        groups.setdefault(transcript, []).append(game_id)
    n_train, n_val, _ = split_counts(len(groups), args.train_ratio, args.val_ratio, args.test_ratio)
    order = np.random.default_rng(args.seed).permutation(len(groups))
    assignment = np.empty(len(groups), dtype=np.int8)
    assignment[order[:n_train]] = 0
    assignment[order[n_train:n_train + n_val]] = 1
    assignment[order[n_train + n_val:]] = 2
    by_transcript = dict(zip(groups, assignment.tolist()))
    with output_directory(destination, args.replace) as output:
        writers = {name: TranscriptWriter(output / name, args.games_per_file) for name in SPLIT_NAMES}
        try:
            for game_id, transcript, _, _ in games:
                writers[SPLIT_NAMES[by_transcript[transcript]]].write(game_id, transcript)
        finally:
            for writer in writers.values():
                writer.close()
        counts = {name: {"games": writers[name].games,
                        "unique_transcripts": int(np.count_nonzero(assignment == i))}
                  for i, name in enumerate(SPLIT_NAMES)}
        result = write_manifest(output, "transcript_split", args.dataset, source=source_info(source),
            seed=args.seed, ratios=dict(zip(SPLIT_NAMES, (args.train_ratio, args.val_ratio, args.test_ratio))),
            split_unit="identical_transcript_group", games=len(games), unique_transcripts=len(groups),
            duplicate_copies=len(games) - len(groups), splits=counts)
    return result


if __name__ == "__main__":
    result = run(make_argparser().parse_args())
    print("transcript_split:", result["splits"])
