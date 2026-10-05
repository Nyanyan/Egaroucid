#!/usr/bin/env python3
"""Export every valid position with weight 2 ** (disc_loss / 2), using only cached scores."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path

import numpy as np

from analysis_pipeline import add_replay_arguments, collect_games, Progress, validate_positive
from board_replay import replay_games
from miss_analysis import AnalysisCache, LEVEL, THREADS, parse_analysis, transcript_key
from pipeline_common import (BoardWriter, SPLIT_NAMES, WEIGHT_FORMAT,
                             dataset_directory, load_manifest, output_directory, source_info, write_manifest)


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("POLICY_DATA_ROOT"),
                        help="Pipeline root containing cache and transcript_split; or set POLICY_DATA_ROOT.")
    parser.add_argument("--cache-name", "--dataset", dest="cache_name", required=True,
                        help="Cache folder name, e.g. WTHOR_1977_2024_LOSS_WEIGHTED; also used for the output folder.")
    parser.add_argument("--replace", action="store_true",
                        help="Keep an existing nonempty output as a timestamped backup before replacing it.")
    add_replay_arguments(parser)
    parser.add_argument("--configuration", help="Cache configuration ID; default: latest statistics or sole configuration.")
    parser.add_argument("--positions-per-file", type=int, default=1000000)
    return parser


def loss_weights(losses):
    values = np.asarray(losses, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > 128):
        raise ValueError("disc losses must be finite and in [0, 128]")
    return np.exp2(values / 2)


def run(args):
    validate_positive(args, "games_per_batch", "positions_per_file", "progress_seconds")
    dataset = args.cache_name
    destination = dataset_directory(args.data_root, "board_split", dataset)
    cache_dir = dataset_directory(args.data_root, "cache", args.cache_name)
    report_path = cache_dir / "statistics" / "loss_distribution.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    if report is not None:
        source_dataset = report["dataset"]
    elif args.cache_name.endswith("_LOSS_WEIGHTED"):
        # An interrupted analysis may have saved scores but not yet written statistics.
        source_dataset = args.cache_name[:-len("_LOSS_WEIGHTED")]
    else:
        raise ValueError("cache has no statistics; use a <source_dataset>_LOSS_WEIGHTED cache folder "
                         "or run 03_analyze_transcripts.py first")
    source = dataset_directory(args.data_root, "transcript_split", source_dataset)
    original = load_manifest(source, "transcript_split")
    accepted, unique, rejected, counts = collect_games(source, original, args)
    configuration = args.configuration
    if configuration is None and report is not None:
        if report["source"]["manifest_sha256"] != source_info(source)["manifest_sha256"]:
            raise ValueError("statistics refer to different transcripts; rerun 03_analyze_transcripts.py or specify --configuration")
        configuration = report["configuration"]
    with closing(AnalysisCache.open_existing(cache_dir, configuration)) as cache:
        if cache.identity.get("level") != LEVEL or cache.identity.get("threads") != THREADS:
            raise ValueError("expected a level-10, single-thread analyze cache")
        keys = cache.keys()
        missing = sum(transcript_key(t) not in keys for t in unique)
        if missing:
            raise ValueError(f"{missing:,} games missing from cache; run 03_analyze_transcripts.py first")
        with output_directory(destination, args.replace) as output:
            progress = Progress("weighted board_split", sum(map(len, accepted.values())), args.progress_seconds)
            done = 0
            for name in SPLIT_NAMES:
                writer = BoardWriter(output / name, args.positions_per_file, weighted=True)
                weight_sum, minimum, maximum = 0.0, None, None
                try:
                    games = accepted[name]
                    for start in range(0, len(games), args.games_per_batch):
                        batch = games[start:start + args.games_per_batch]
                        boards, ids, errors = replay_games(batch)
                        if errors:
                            raise ValueError("previously validated game failed replay")
                        losses = []
                        for _, transcript in batch:
                            rows = parse_analysis(cache.get(transcript), transcript, unique[transcript])
                            losses.extend(row["loss"] for row in rows)
                        weights = loss_weights(losses)
                        writer.write(boards, ids, weights)
                        weight_sum += float(weights.sum())
                        minimum = float(weights.min()) if minimum is None else min(minimum, float(weights.min()))
                        maximum = float(weights.max()) if maximum is None else max(maximum, float(weights.max()))
                        done += len(batch)
                        progress.show(done, f"{name}: {writer.positions:,} positions")
                finally:
                    writer.close()
                counts[name].update(games=len(accepted[name]), positions=writer.positions,
                                    weight_sum=weight_sum, minimum_weight=minimum, maximum_weight=maximum)
            progress.show(done, force=True)
            (output / "rejected_games.jsonl").write_text(
                "".join(json.dumps(entry) + "\n" for entry in rejected), encoding="utf-8")
            result = write_manifest(
                output, "board_split", dataset, source=source_info(source),
                seed=original["seed"], ratios=original["ratios"], split_unit=original["split_unit"],
                record_format="<QQbbb: player, opponent, color, policy, score", record_bytes=19,
                game_ids_format="uint64 little-endian; original game ID per position",
                score_convention="terminal disc difference with empties awarded to winner",
                sample_weights={"format": WEIGHT_FORMAT, "suffix": ".weights", "formula": "2 ** (disc_loss / 2)",
                                "normalization": "none in files", "clipping": None},
                analysis={**cache.identity, "configuration": cache.configuration,
                          "cache": f"cache/{args.cache_name}/analyze.sqlite3"}, splits=counts)
    print("saved:", destination, flush=True)
    return result


if __name__ == "__main__":
    run(make_argparser().parse_args())
