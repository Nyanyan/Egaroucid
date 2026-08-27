#!/usr/bin/env python3
"""Summarize and calibrate practical midgame ProbCut residuals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


TAIL_PROBABILITY = (0.13, 0.06, 0.035)  # 74%, 88%, 93% two-sided levels


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty quantile")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(errors: list[float], cushion: float) -> dict[str, object]:
    bias = statistics.fmean(errors)
    sigma = math.sqrt(statistics.fmean((value - bias) ** 2 for value in errors))
    rms = math.sqrt(statistics.fmean(value * value for value in errors))
    high_errors = []
    low_errors = []
    for tail in TAIL_PROBABILITY:
        lower = quantile(errors, tail)
        upper = quantile(errors, 1.0 - tail)
        high_errors.append(max(1, math.ceil(cushion * max(0.0, -lower))))
        low_errors.append(max(1, math.ceil(cushion * max(0.0, upper))))
    return {
        "n": len(errors),
        "bias": bias,
        "sigma": sigma,
        "rms": rms,
        "mae": statistics.fmean(abs(value) for value in errors),
        "minimum": min(errors),
        "maximum": max(errors),
        "high_error_74_88_93": high_errors,
        "low_error_74_88_93": low_errors,
    }


def load_rows(paths: list[Path]) -> list[dict[str, int | str]]:
    result = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                converted: dict[str, int | str] = dict(row)
                for field in (
                    "n_discs", "deep_depth", "shallow_depth", "static_value",
                    "shallow_value", "deep_value", "error", "shallow_nodes",
                    "deep_nodes", "shallow_time_ms", "deep_time_ms", "legal_count",
                ):
                    converted[field] = int(row[field])
                result.append(converted)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cushion", type=float, default=1.10)
    args = parser.parse_args()

    rows = load_rows(args.inputs)
    by_depth: dict[int, list[dict[str, int | str]]] = {}
    for row in rows:
        by_depth.setdefault(int(row["deep_depth"]), []).append(row)

    report: dict[str, object] = {
        "inputs": [str(path) for path in args.inputs],
        "cushion": args.cushion,
        "tail_probability_74_88_93": TAIL_PROBABILITY,
        "depths": {},
    }
    for deep_depth, depth_rows in sorted(by_depth.items()):
        shallow_depths = sorted({int(row["shallow_depth"]) for row in depth_rows})
        current_shallow = ((deep_depth * 2 // 5) & ~1) + (deep_depth & 1)
        depth_report: dict[str, object] = {
            "current_shallow_depth": current_shallow,
            "alternatives": {},
        }
        for shallow_depth in shallow_depths:
            selected = [row for row in depth_rows if int(row["shallow_depth"]) == shallow_depth]
            errors = [float(int(row["error"])) for row in selected]
            summary = summarize(errors, args.cushion)
            summary["mean_shallow_nodes"] = statistics.fmean(
                int(row["shallow_nodes"]) for row in selected
            )
            summary["mean_deep_nodes"] = statistics.fmean(
                int(row["deep_nodes"]) for row in selected
            )
            depth_report["alternatives"][str(shallow_depth)] = summary
        report["depths"][str(deep_depth)] = depth_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for depth, values in report["depths"].items():
        current = str(values["current_shallow_depth"])
        summary = values["alternatives"][current]
        print(
            f"depth={depth} shallow={current} n={summary['n']} "
            f"bias={summary['bias']:.3f} sigma={summary['sigma']:.3f} "
            f"high={summary['high_error_74_88_93']} low={summary['low_error_74_88_93']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
