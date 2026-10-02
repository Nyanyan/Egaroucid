#!/usr/bin/env python3
"""Join newly measured agreement and strength; never use the legacy static CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

BASELINES = tuple(f"egaroucid_l{level}" for level in range(1, 20, 2))
LABELS = {"dense512": "Dense 512 x 4", "conv8_bn": "Conv8 + BN"}


def anchored_elos(metadata, replicate_path):
    points = {row["name"]: float(row["paired_set_elo"]) for row in metadata["results"]}
    with replicate_path.open(encoding="utf-8", newline="") as source:
        draws = list(csv.DictReader(source))
    if len(draws) != metadata["bootstrap"]["replicates"]:
        raise ValueError("Bootstrap replicate count mismatch")
    # Separate tournaments have different participant means. Fix the mean of
    # the same ten Egaroucid baselines to 1500 in BOTH points and replicates.
    offset = 1500.0 - np.mean([points[name] for name in BASELINES])
    offsets = 1500.0 - np.asarray([[float(row[name]) for name in BASELINES] for row in draws]).mean(axis=1)
    results = {}
    for name, point in points.items():
        values = np.asarray([float(row[name]) for row in draws]) + offsets
        low, high = np.quantile(values, [0.025, 0.975])
        results[name] = (point + offset, float(low), float(high))
    return results


def model_rows(root, name):
    from game_dataset import file_sha256
    destination = root / name
    human = json.loads((destination / "human" / "random_wthor_blend_summary.json").read_text(encoding="utf-8"))
    strength = destination / "strength"
    metadata = json.loads((strength / "strength_elo_ci95.json").read_text(encoding="utf-8"))
    for filename in ("strength_games.jsonl", "strength_results.json", "strength_manifest.json"):
        if file_sha256(strength / filename) != metadata["source_sha256"][filename]:
            raise ValueError(f"Stale Elo analysis: {strength / filename}")
    replicate_path = strength / "strength_elo_bootstrap_replicates.csv"
    if file_sha256(replicate_path) != metadata["generated_sha256"][replicate_path.name]:
        raise ValueError("Bootstrap replicate checksum mismatch")
    ratings = anchored_elos(metadata, replicate_path)
    rows = []
    for kind, measurements in (("blend", human["results"]), ("console", human["console_level_results"])):
        for measurement in measurements:
            if kind == "console" and measurement["level"] == 21:
                continue  # Already represented by alpha=0.
            participant = (f"alpha_{float(measurement['alpha'])}" if kind == "blend"
                           else f"egaroucid_l{measurement['level']}")
            point, low, high = ratings[participant]
            rows.append({"model": name, "kind": kind, "participant": participant,
                         "alpha": measurement.get("alpha", ""), "level": measurement.get("level", 21),
                         "elo_anchored": point, "elo_ci95_low": low, "elo_ci95_high": high,
                         "positions": measurement["positions"],
                         **{key: measurement[key] for key in ("top1_accuracy", "top1_ci95_lower", "top1_ci95_upper",
                                                               "top3_accuracy", "top3_ci95_lower", "top3_ci95_upper")}})
    return rows, human["sample_global_indices_sha256"], human["sample_records_sha256"]


def write_report(root: Path, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    reference = None
    for name in names:
        model_data, index_hash, records_hash = model_rows(root, name)
        if reference is not None and reference != (index_hash, records_hash):
            raise ValueError("Agreement samples differ between models")
        reference = index_hash, records_hash
        rows.extend(model_data)
    stem = root / ("rating_vs_human_match_" + "_".join(names))
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(12, 8))
    colors = {"dense512": "#c75028", "conv8_bn": "#1764b0"}
    for name in names:
        for kind in ("console", "blend"):
            selected = [row for row in rows if row["model"] == name and row["kind"] == kind]
            x = [row["elo_anchored"] for row in selected]
            y = [100 * row["top1_accuracy"] for row in selected]
            color = colors[name]
            alpha = 1.0 if kind == "blend" else 0.4
            label = LABELS[name] if kind == "blend" else "Egaroucid baselines (" + LABELS[name] + " tournament)"
            axis.plot(x, y, "o-" if kind == "blend" else ".--", color=color, alpha=alpha, label=label)
            for row in selected:
                px, py = row["elo_anchored"], 100 * row["top1_accuracy"]
                axis.plot([row["elo_ci95_low"], row["elo_ci95_high"]], [py, py], color=color, alpha=alpha)
                axis.plot([px, px], [100 * row["top1_ci95_lower"], 100 * row["top1_ci95_upper"]], color=color, alpha=alpha)
                if kind == "blend":
                    axis.annotate(f"a={row['alpha']:g}", (px, py), xytext=(5, 8 if name == "dense512" else -14), textcoords="offset points", fontsize=9)
    axis.set_xlabel("Relative Elo (mean of Egaroucid levels 1,3,...,19 fixed to 1500)")
    axis.set_ylabel("Human top-1 move agreement (%)")
    axis.set_title("Game-held-out WTHOR: policy blending with Egaroucid level 21")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    figure.text(0.5, 0.02, "95% intervals: Elo = XOT-cluster percentile bootstrap; agreement = position-wise Wilson.\n"
                "a=0: Egaroucid only; a=1: policy network only. Separate tournaments share the same baseline levels and XOT schedule.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    figure.savefig(stem.with_suffix(".png"), dpi=180)
    figure.savefig(stem.with_suffix(".svg"))
    plt.close(figure)
    stem.with_suffix(".json").write_text(json.dumps({
        "models": names, "sample_indices_sha256": reference[0], "sample_records_sha256": reference[1],
        "elo_anchor": "mean of common Egaroucid levels 1,3,...,19 = 1500 in each point estimate and bootstrap replicate",
        "elo_ci": "pointwise 95% XOT/set-index cluster percentile bootstrap after re-centering",
        "agreement_ci": "position-wise 95% Wilson; does not account for within-game correlation",
        "interpretation": "descriptive comparison using identical baseline levels; not a joint tournament or a significance test",
        "legacy_results_included": False,
    }, indent=2), encoding="utf-8")
    print(f"Report: {stem.with_suffix('.png')}\nData: {stem.with_suffix('.csv')}", flush=True)
