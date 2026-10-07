#!/usr/bin/env python3
"""Export nested mistake-only test sets from validated, cached level-10 analyses."""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import json
import os
from pathlib import Path

import numpy as np

from board_replay import replay_games
from miss_analysis import AnalysisCache, parse_analysis
from pipeline_common import (dataset_directory, iter_transcripts, load_manifest,
                             output_directory, source_info, write_manifest)
from wthor_dataset import BOARD_DTYPE, file_sha256

THRESHOLDS = (1, 2, 4, 6)


def read_test_records(directory):
    boards, ids = [], []
    for path in sorted(directory.glob("*.dat")):
        if path.stat().st_size % BOARD_DTYPE.itemsize:
            raise ValueError(f"incomplete board record: {path}")
        part = np.fromfile(path, dtype=BOARD_DTYPE)
        id_path = path.with_suffix(".game_ids")
        if id_path.stat().st_size != len(part) * 8:
            raise ValueError(f"game IDs do not align: {path}")
        boards.append(part)
        ids.append(np.fromfile(id_path, dtype="<u8"))
    return (np.concatenate(boards) if boards else np.empty(0, BOARD_DTYPE),
            np.concatenate(ids) if ids else np.empty(0, "<u8"))


def export_subsets(output, dataset, boards, ids, losses, metadata):
    if len(boards) != len(ids) or len(ids) != len(losses):
        raise ValueError("boards, IDs and losses must align")
    if np.any(losses < 0) or np.any(losses > 128):
        raise ValueError("loss must be in [0, 128]")
    counts = []
    for threshold in THRESHOLDS:
        name = f"{dataset}_MISS_{threshold}"
        directory = output / name
        test = directory / "test"
        test.mkdir(parents=True)
        indices = np.flatnonzero(losses >= threshold).astype("<u8")
        path = test / "0000000.dat"
        boards[indices].tofile(path)
        ids[indices].tofile(path.with_suffix(".game_ids"))
        losses[indices].astype("u1").tofile(path.with_suffix(".losses"))
        indices.tofile(path.with_suffix(".source_indices"))
        games = len(np.unique(ids[indices]))
        row = dict(dataset=name, minimum_loss=threshold, positions=len(indices), games=games,
                   source_positions=len(boards), fraction=len(indices) / len(boards) if len(boards) else 0)
        counts.append(row)
        write_manifest(directory, "board_split", name, **metadata,
                       record_format="<QQbbb: player, opponent, color, policy, score", record_bytes=19,
                       game_ids_format="uint64 little-endian; original game ID per position",
                       losses_format="uint8; cached disc loss per position",
                       source_indices_format="uint64 little-endian; zero-based index in original test",
                       source_split="test", augmentation=False,
                       selection={"minimum_loss": threshold, "comparison": ">=",
                                  "loss": "max(played_score, alternative_score) - played_score"},
                       splits={"test": {"positions": len(indices), "games": games}})
    return counts


def run(args):
    if args.games_per_batch <= 0:
        raise ValueError("--games-per-batch must be positive")
    transcripts = dataset_directory(args.data_root, "transcript_split", args.dataset)
    board_dir = dataset_directory(args.data_root, "board_split", args.dataset)
    cache_dir = args.cache_dir or dataset_directory(args.data_root, "cache", args.dataset + "_LOSS_WEIGHTED")
    print("Validating source manifests and checksums...", flush=True)
    transcript_manifest = load_manifest(transcripts, "transcript_split")
    board_manifest = load_manifest(board_dir, "board_split")
    if board_manifest["source"] != source_info(transcripts):
        raise ValueError("board data was not generated from these split transcripts")
    boards, ids = read_test_records(board_dir / "test")
    if len(boards) != board_manifest["splits"]["test"]["positions"]:
        raise ValueError("test board count differs from manifest")
    games = [(game_id, transcript) for game_id, transcript, _, _ in iter_transcripts(transcripts / "test")]
    if len(games) != transcript_manifest["splits"]["test"]["games"]:
        raise ValueError("test transcript count differs from manifest")
    if len({game_id for game_id, _ in games}) != len(games):
        raise ValueError("duplicate game IDs in test")
    configuration = args.configuration
    statistics_path = cache_dir / "statistics" / "loss_distribution.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8")) if statistics_path.exists() else None
    if configuration is None and statistics is not None:
        if statistics["source"] != source_info(transcripts):
            raise ValueError("analysis statistics refer to different split transcripts")
        configuration = statistics["configuration"]
    losses = np.empty(len(boards), dtype="u1")
    offset = 0
    rejected = []
    with closing(AnalysisCache.open_existing(cache_dir, configuration)) as cache:
        if cache.identity.get("level") != 10:
            raise ValueError("mistake selection requires level-10 analysis")
        for start in range(0, len(games), args.games_per_batch):
            batch = games[start:start + args.games_per_batch]
            replayed, replay_ids, bad = replay_games(batch)
            end = offset + len(replayed)
            if not np.array_equal(replayed, boards[offset:end]) or not np.array_equal(replay_ids, ids[offset:end]):
                raise ValueError("replayed test differs from original test records/IDs")
            bad_ids = {entry["game_id"] for entry in bad}
            rejected.extend(bad)
            local = 0
            for game_id, transcript in batch:
                if game_id in bad_ids:
                    continue
                size = len(transcript) // 2
                rows = parse_analysis(cache.get(transcript), transcript,
                                      replayed["color"][local:local + size].tobytes())
                if [row["played"] for row in rows] != replayed["policy"][local:local + size].tolist():
                    raise ValueError("analyzed policies differ from replay")
                losses[offset + local:offset + local + size] = [row["loss"] for row in rows]
                local += size
            if local != len(replayed):
                raise ValueError("analysis count differs from replay")
            offset = end
            print(f"Validated {min(start + len(batch), len(games)):,}/{len(games):,} games; {offset:,} positions", flush=True)
        if offset != len(boards) or len(rejected) != board_manifest["splits"]["test"]["rejected_games"]:
            raise ValueError("test validation counts differ")
        histogram = np.bincount(losses, minlength=129).tolist()
        if statistics is not None and statistics["configuration"] == cache.configuration:
            if histogram != [row["count"] for row in statistics["distribution"]["test"]["bins"]]:
                raise ValueError("loss histogram differs from existing analysis statistics")
        metadata = dict(source=source_info(board_dir), transcript_source=source_info(transcripts),
                        analysis={"cache": str(cache.path.resolve()), "configuration": cache.configuration,
                                  "compatible_configurations": cache.compatible_configurations,
                                  "identity": cache.identity})
        with output_directory(args.output_dir, args.replace) as output:
            counts = export_subsets(output, args.dataset, boards, ids, losses, metadata)
            (output / "source_test.losses").write_bytes(losses.tobytes())
            (output / "rejected_games.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rejected), encoding="utf-8")
            report = dict(version=1, source_dataset=args.dataset, source_positions=len(boards),
                          thresholds=list(THRESHOLDS), datasets=counts, loss_histogram=histogram,
                          source_loss_sha256=file_sha256(output / "source_test.losses"), **metadata)
            (output / "dataset_counts.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            with (output / "dataset_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(counts[0]))
                writer.writeheader()
                writer.writerows(counts)
    print(json.dumps(counts, indent=2), flush=True)
    return report


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = os.environ.get("POLICY_DATA_ROOT")
    if not default_root and os.environ.get("EGAROUCID_DATA"):
        default_root = Path(os.environ["EGAROUCID_DATA"]) / "20261002_policy_network_human_like_data"
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--dataset", default="WTHOR_1977_2024")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--configuration")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games-per-batch", type=int, default=4096)
    parser.add_argument("--replace", action="store_true")
    return parser


if __name__ == "__main__":
    run(make_argparser().parse_args())
