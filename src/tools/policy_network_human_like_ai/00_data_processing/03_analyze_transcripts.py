#!/usr/bin/env python3
"""Cache all level-10 analyses and report the loss distribution, without board export."""

import argparse
from contextlib import closing
import csv
import json
from pathlib import Path

import numpy as np

from analysis_pipeline import (add_analysis_arguments, analyze_missing, collect_games, Progress,
                               validate_positive, transcript_multiplicities, loss_histogram, format_loss_percentages)
from miss_analysis import AnalysisCache, LEVEL, THREADS, engine_identity, parse_analysis
from pipeline_common import (SPLIT_NAMES, add_dataset_arguments, dataset_directory, load_manifest,
                             output_directory, source_info)


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    add_analysis_arguments(parser)
    return parser


def loss_distribution(accepted, unique, cache, interval):
    histograms = {name: np.zeros(129, dtype=np.int64) for name in SPLIT_NAMES}
    progress = Progress("loss statistics", sum(map(len, accepted.values())), interval)
    done = 0
    all_counts = np.zeros(129, dtype=np.int64)
    for name, games in accepted.items():
        # Count duplicate game copies in the distribution just as in the dataset.
        for _, transcript in games:
            rows = parse_analysis(cache.get(transcript), transcript, unique[transcript])
            histogram = loss_histogram(rows)
            histograms[name] += histogram
            all_counts += histogram
            done += 1
            progress.show(done, f"{name} | {format_loss_percentages(all_counts)}")
    histograms = {"all": all_counts, **histograms}
    result = {}
    for name, counts in histograms.items():
        total = int(counts.sum())
        result[name] = {"positions": total, "bins": [
            {"loss": loss, "count": int(count), "percentage": 100 * int(count) / total if total else 0.0}
            for loss, count in enumerate(counts)]}
    progress.show(done, format_loss_percentages(all_counts), force=True)
    for name in SPLIT_NAMES:
        print(f"{name}: {format_loss_percentages(histograms[name])}", flush=True)
    return result


def plot_distribution(distribution, dataset, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(13, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    frequency, comparison = figure.subplots(1, 2)
    bins = distribution["all"]["bins"]
    maximum = max((row["loss"] for row in bins if row["count"]), default=0)
    x = np.arange(maximum + 1)
    frequency.bar(x, [row["percentage"] for row in bins[:maximum + 1]], color="#2563a6")
    frequency.set(title=f"All positions (n={distribution['all']['positions']:,})",
                  ylabel="Positions (%)", xlabel="Disc loss", xlim=(-.75, max(1, maximum) + .75))
    has_values = False
    for name, color in zip(SPLIT_NAMES, ("#2563a6", "#d97706", "#059669")):
        selected = [row for row in distribution[name]["bins"] if row["count"]]
        if selected:
            has_values = True
            comparison.plot([row["loss"] for row in selected], [row["percentage"] for row in selected],
                            marker=".", label=name, color=color, alpha=.8)
    if has_values:
        comparison.set_yscale("log")
        comparison.legend()
    else:
        comparison.text(.5, .5, "No valid positions", ha="center", transform=comparison.transAxes)
    comparison.set(title="Split distributions (log scale; zero bins omitted)",
                   ylabel="Positions (%)", xlabel="Disc loss", xlim=(-.75, max(1, maximum) + .75))
    for axis in (frequency, comparison):
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle(f"{dataset} | Egaroucid level {LEVEL}")
    figure.savefig(path, dpi=160)


def run(args):
    validate_positive(args, "workers", "games_per_batch", "timeout_sec", "progress_seconds")
    if not 0 <= args.hash_level <= 29:
        raise ValueError("--hash-level must be in 0..29")
    args.egaroucid_exe = Path(args.egaroucid_exe).expanduser().resolve()
    source = dataset_directory(args.data_root, "transcript_split", args.dataset)
    cache_dir = dataset_directory(args.data_root, "cache", args.dataset + "_LOSS_WEIGHTED")
    # Check plotting availability before starting a potentially long search.
    from matplotlib.figure import Figure  # noqa: F401

    original = load_manifest(source, "transcript_split")
    accepted, unique, rejected, counts = collect_games(source, original, args)
    identity = engine_identity(args.egaroucid_exe, args.hash_level)
    with closing(AnalysisCache(cache_dir, identity)) as cache:
        print(f"Egaroucid level {LEVEL}: {args.workers} processes x {THREADS} thread; {cache.path}", flush=True)
        stats = analyze_missing(unique, cache, args, game_counts=transcript_multiplicities(accepted))
        distribution = loss_distribution(accepted, unique, cache, args.progress_seconds)
        report = {"version": 1, "dataset": args.dataset, "source": source_info(source),
                  "configuration": cache.configuration,
                  "analysis": {**identity, **stats, "compatible_cache_configurations": cache.compatible_configurations},
                  "splits": counts, "counting_unit": "position, including duplicate game copies",
                  "loss_formula": "max(played_score, alternative_score) - played_score",
                  "distribution": distribution}
        # Publish a consistent report bundle and preserve the previous one on reruns.
        with output_directory(cache_dir / "statistics", replace=True) as output:
            (output / "loss_distribution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            with (output / "loss_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["split", "loss", "count", "percentage"])
                writer.writeheader()
                for name, data in distribution.items():
                    writer.writerows({"split": name, **row} for row in data["bins"])
            (output / "rejected_games.jsonl").write_text(
                "".join(json.dumps(entry) + "\n" for entry in rejected), encoding="utf-8")
            plot_distribution(distribution, args.dataset, output / "loss_distribution.png")
    print("saved statistics:", cache_dir / "statistics", flush=True)
    return report


if __name__ == "__main__":
    run(make_argparser().parse_args())
