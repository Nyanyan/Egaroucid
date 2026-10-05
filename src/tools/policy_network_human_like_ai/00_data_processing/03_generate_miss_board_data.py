#!/usr/bin/env python3
"""Analyze split transcripts at level 10 and retain moves losing at least N discs."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
import os
from pathlib import Path

import numpy as np

from board_replay import replay_games
from analysis_pipeline import Progress, collect_games, analyze_missing, transcript_multiplicities
from miss_analysis import (AnalysisCache, AnalyzeEngine, LEVEL, THREADS, default_engine,
                           engine_identity, parse_analysis, transcript_key)
from pipeline_common import (BoardWriter, SPLIT_NAMES, add_dataset_arguments, dataset_directory,
                             iter_transcripts, load_manifest, output_directory, source_info, write_manifest)


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--min-loss", type=int, required=True, metavar="N",
                        help="Keep moves losing >= N discs (1..128); creates <dataset>_MISS_<N>.")
    parser.add_argument("--egaroucid-exe", type=Path,
                        default=os.environ.get("EGAROUCID_EXE") or default_engine())
    parser.add_argument("--workers", type=int, default=32, help="Persistent engines; each uses exactly one thread.")
    parser.add_argument("--hash-level", type=int, default=25,
                        help="Per-engine hash exponent; requires resources/hash/hash<N>.eghs (default: 25).")
    parser.add_argument("--timeout-sec", type=float, default=600, help="Timeout per engine command.")
    parser.add_argument("--progress-seconds", type=float, default=5)
    parser.add_argument("--games-per-batch", type=int, default=4096)
    parser.add_argument("--positions-per-file", type=int, default=1000000)
    return parser


def write_boards(output, accepted, unique, cache, counts, args):
    progress = Progress("board_split", sum(map(len, accepted.values())), args.progress_seconds)
    done = 0
    for name in SPLIT_NAMES:
        writer = BoardWriter(output / name, args.positions_per_file)
        selected_games = 0
        try:
            games = accepted[name]
            for start in range(0, len(games), args.games_per_batch):
                batch = games[start:start + args.games_per_batch]
                boards, ids, rejected = replay_games(batch)
                if rejected:
                    raise ValueError("previously validated game failed replay")
                keep = np.zeros(len(boards), dtype=bool)
                offset = 0
                for game_id, transcript in batch:
                    rows = parse_analysis(cache.get(transcript), transcript, unique[transcript])
                    mask = np.array([row["loss"] >= args.min_loss for row in rows], dtype=bool)
                    keep[offset:offset + len(rows)] = mask
                    selected_games += int(mask.any())
                    offset += len(rows)
                if offset != len(boards):
                    raise ValueError("analysis/replay board count mismatch")
                writer.write(boards[keep], ids[keep])
                done += len(batch)
                progress.show(done, f"{name}: {writer.positions:,} positions")
        finally:
            writer.close()
        # An empty split is a valid result for a strict threshold; never fabricate samples.
        counts[name].update(games=selected_games, positions=writer.positions,
                            games_without_misses=len(accepted[name]) - selected_games)
    progress.show(done, force=True)


def run(args):
    if not 1 <= args.min_loss <= 128:
        raise ValueError("--min-loss must be in 1..128")
    for key in ("workers", "games_per_batch", "positions_per_file", "timeout_sec", "progress_seconds"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive and finite")
    if not 0 <= args.hash_level <= 29:
        raise ValueError("--hash-level must be in 0..29")
    args.egaroucid_exe = Path(args.egaroucid_exe).expanduser().resolve()
    dataset = f"{args.dataset}_MISS_{args.min_loss}"
    source = dataset_directory(args.data_root, "transcript_split", args.dataset)
    destination = dataset_directory(args.data_root, "board_split", dataset)
    cache_dir = dataset_directory(args.data_root, "cache", args.dataset + "_LOSS_WEIGHTED")
    # Refuse existing output before doing expensive work; publication is atomic.
    with output_directory(destination, args.replace) as output:
        print("checking transcript manifest...", flush=True)
        original = load_manifest(source, "transcript_split")
        accepted, unique, rejected, counts = collect_games(source, original, args)
        identity = engine_identity(args.egaroucid_exe, args.hash_level)
        with closing(AnalysisCache(cache_dir, identity)) as cache:
            print(f"Egaroucid level {LEVEL}, {args.workers} processes x {THREADS} thread; cache: {cache.path}", flush=True)
            stats = analyze_missing(unique, cache, args, game_counts=transcript_multiplicities(accepted))
            write_boards(output, accepted, unique, cache, counts, args)
            with (output / "rejected_games.jsonl").open("w", encoding="utf-8") as rejected_file:
                for entry in rejected:
                    rejected_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            result = write_manifest(
                output, "board_split", dataset, source=source_info(source),
                seed=original["seed"], ratios=original["ratios"], split_unit=original["split_unit"],
                record_format="<QQbbb: player, opponent, color, policy, score", record_bytes=19,
                game_ids_format="uint64 little-endian; original game ID per position",
                score_convention="terminal disc difference with empties awarded to winner",
                selection={"minimum_loss": args.min_loss, "comparison": ">=",
                           "loss": "max(played_score, alternative_score) - played_score",
                           "policy": "original human move", "position": "before the human move"},
                analysis={**identity, **stats, "workers": args.workers,
                          "cache": f"cache/{args.dataset}_LOSS_WEIGHTED/analyze.sqlite3",
                          "configuration": cache.configuration}, splits=counts)
    print("saved:", destination, flush=True)
    return result


if __name__ == "__main__":
    result = run(make_argparser().parse_args())
    print("board_split:", result["splits"])
