#!/usr/bin/env python3
"""Fit and cross-validate a midgame ProbCut residual-scale model.

The input is one or more TSV files emitted by mid_probcut_dataset_tool.  Boards
are assigned to train/validation before aggregation, so rows for the same board
can never leak across the split.  The fitted quantity is log(RMS residual),
which keeps the runtime sigma positive without clipping a polynomial surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def features(n_discs: int, shallow_depth: int, deep_depth: int) -> list[float]:
    n = (n_discs - 28.0) / 14.0
    shallow = shallow_depth / 13.0
    gap = (deep_depth - shallow_depth) / 13.0
    return [
        1.0,
        n,
        shallow,
        gap,
        n * n,
        shallow * shallow,
        gap * gap,
        n * shallow,
        n * gap,
        shallow * gap,
    ]


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1.0e-12:
            raise ValueError("singular normal matrix")
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def fit(rows: list[dict[str, int | str]], ridge: float) -> list[float]:
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (int(row["n_discs"]), int(row["shallow_depth"]), int(row["deep_depth"]))
        buckets[key].append(int(row["error"]))
    dimension = len(features(0, 0, 0))
    normal = [[0.0] * dimension for _ in range(dimension)]
    target = [0.0] * dimension
    for (n_discs, shallow_depth, deep_depth), errors in buckets.items():
        x = features(n_discs, shallow_depth, deep_depth)
        rms = math.sqrt(sum(error * error for error in errors) / len(errors))
        y = math.log(max(0.25, rms))
        weight = float(len(errors))
        for i in range(dimension):
            target[i] += weight * x[i] * y
            for j in range(dimension):
                normal[i][j] += weight * x[i] * x[j]
    for i in range(dimension):
        normal[i][i] += ridge
    return solve(normal, target)


def sigma(row: dict[str, int | str], coefficients: list[float]) -> float:
    x = features(int(row["n_discs"]), int(row["shallow_depth"]), int(row["deep_depth"]))
    return math.exp(sum(coefficient * value for coefficient, value in zip(coefficients, x)))


def legacy_sigma(row: dict[str, int | str]) -> float:
    n_discs = int(row["n_discs"])
    shallow_depth = int(row["shallow_depth"])
    deep_depth = int(row["deep_depth"])
    value = (
        0.8335834703936896 * (n_discs / 64.0)
        - 4.71778909968251 * (shallow_depth / 60.0)
        + 1.1467905781538477 * (deep_depth / 60.0)
    )
    return (
        -0.5274699259330169 * value**3
        + 6.5091001393587335 * value**2
        + 3.9546352081550378 * value
        + 1.8719077939546169
    )


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metrics(rows: list[dict[str, int | str]], coefficients: list[float]) -> dict[str, object]:
    normalized = [int(row["error"]) / sigma(row, coefficients) for row in rows]
    legacy_normalized = [int(row["error"]) / legacy_sigma(row) for row in rows]
    result: dict[str, object] = {
        "rows": len(rows),
        "rms_normalized": math.sqrt(sum(value * value for value in normalized) / len(normalized)),
        "legacy_rms_normalized": math.sqrt(
            sum(value * value for value in legacy_normalized) / len(legacy_normalized)
        ),
        "mean_normalized": sum(normalized) / len(normalized),
        "tails": {},
    }
    for name, probability in (("74", 0.13), ("88", 0.06), ("93", 0.035)):
        result["tails"][name] = {
            "high_multiplier": max(0.0, -quantile(normalized, probability)),
            "low_multiplier": max(0.0, quantile(normalized, 1.0 - probability)),
        }
    return result


def read_rows(paths: list[Path]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for path in paths:
        prefix = path.read_bytes()[:2]
        encoding = "utf-16" if prefix in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        with path.open(encoding=encoding, newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-modulo", type=int, default=5)
    parser.add_argument("--validation-index", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    args = parser.parse_args()

    rows = read_rows(args.inputs)
    train: list[dict[str, int | str]] = []
    validation: list[dict[str, int | str]] = []
    for row in rows:
        digest = hashlib.sha256(str(row["board"]).encode("utf-8")).digest()
        shard = int.from_bytes(digest[:8], "big") % args.validation_modulo
        (validation if shard == args.validation_index else train).append(row)
    coefficients = fit(train, args.ridge)
    report = {
        "inputs": [str(path) for path in args.inputs],
        "feature_order": [
            "constant", "n", "shallow", "gap", "n2", "shallow2", "gap2",
            "n_shallow", "n_gap", "shallow_gap",
        ],
        "feature_scaling": {
            "n": "(n_discs - 28) / 14",
            "shallow": "shallow_depth / 13",
            "gap": "(deep_depth - shallow_depth) / 13",
        },
        "coefficients": coefficients,
        "train": metrics(train, coefficients),
        "validation": metrics(validation, coefficients),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
