#!/usr/bin/env python3
"""Cross-validate conditional mean and scale models for endgame MPC."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


SELECTIVITY = (0.74, 0.88, 0.93)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def observation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("root_id", "")), str(row["board"]),
        int(row["deep_depth"]), int(row["shallow_depth"]),
    )


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = {}
    for row in rows:
        key = observation_key(row)
        previous = result.get(key)
        if previous is not None:
            for field in ("shallow_value", "deep_value", "d0_value", "legal_count"):
                if int(previous[field]) != int(row[field]):
                    raise ValueError(f"inconsistent {field} for {key}")
        else:
            result[key] = row
    return list(result.values())


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-12:
            raise ValueError("singular normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= divisor
        for row in range(n):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier == 0.0:
                continue
            for j in range(column, n + 1):
                augmented[row][j] -= multiplier * augmented[column][j]
    return [augmented[i][n] for i in range(n)]


FEATURES: dict[str, Callable[[dict[str, Any]], list[float]]] = {
    "identity": lambda row: [],
    "bias": lambda row: [],
    "bias_d0": lambda row: [float(row["d0_value"] - row["shallow_value"])],
    "bias_full": lambda row: [
        float(row["d0_value"] - row["shallow_value"]),
        float(row["deep_depth"]), float(row["shallow_depth"]),
        float(row["legal_count"]),
    ],
    "affine_s": lambda row: [float(row["shallow_value"])],
    "affine_s_d0": lambda row: [float(row["shallow_value"]), float(row["d0_value"])],
    "affine_full": lambda row: [
        float(row["shallow_value"]), float(row["d0_value"]),
        float(row["deep_depth"]), float(row["shallow_depth"]),
        float(row["legal_count"]),
    ],
}


class MeanModel:
    def __init__(
        self, name: str, base_shallow: bool, means: list[float],
        scales: list[float], coefficients: list[float],
    ) -> None:
        self.name = name
        self.base_shallow = base_shallow
        self.means = means
        self.scales = scales
        self.coefficients = coefficients

    def predict(self, row: dict[str, Any]) -> float:
        features = FEATURES[self.name](row)
        value = float(row["shallow_value"]) if self.base_shallow else 0.0
        value += self.coefficients[0]
        for index, feature in enumerate(features):
            value += self.coefficients[index + 1] * (
                (feature - self.means[index]) / self.scales[index]
            )
        return value

    def raw_coefficients(self) -> dict[str, Any]:
        raw = [
            self.coefficients[index + 1] / self.scales[index]
            for index in range(len(self.means))
        ]
        intercept = self.coefficients[0] - sum(
            coefficient * mean for coefficient, mean in zip(raw, self.means)
        )
        return {
            "base_shallow": self.base_shallow,
            "intercept": intercept, "coefficients": raw,
        }


def fit_mean(rows: list[dict[str, Any]], name: str, ridge: float = 0.01) -> MeanModel:
    if name == "identity":
        return MeanModel("identity", True, [], [], [0.0])
    base_shallow = name.startswith("bias")
    feature_function = FEATURES[name]
    raw_features = [feature_function(row) for row in rows]
    dimension = len(raw_features[0]) if raw_features else 0
    means = [
        sum(features[index] for features in raw_features) / len(raw_features)
        for index in range(dimension)
    ]
    scales = []
    for index in range(dimension):
        variance = sum(
            (features[index] - means[index]) ** 2 for features in raw_features
        ) / len(raw_features)
        scales.append(max(1.0, math.sqrt(variance)))
    size = dimension + 1
    normal = [[0.0] * size for _ in range(size)]
    target = [0.0] * size
    for row, features in zip(rows, raw_features):
        design = [1.0] + [
            (features[index] - means[index]) / scales[index]
            for index in range(dimension)
        ]
        value = float(row["deep_value"])
        if base_shallow:
            value -= float(row["shallow_value"])
        for i in range(size):
            target[i] += design[i] * value
            for j in range(size):
                normal[i][j] += design[i] * design[j]
    for index in range(1, size):
        normal[index][index] += ridge
    return MeanModel(name, base_shallow, means, scales, solve(normal, target))


def legal_bucket(row: dict[str, Any]) -> int:
    legal = int(row["legal_count"])
    return 0 if legal <= 3 else 1 if legal <= 6 else 2


def sigma_key(row: dict[str, Any], kind: str) -> tuple[int, ...]:
    deep = int(row["deep_depth"])
    shallow = int(row["shallow_depth"])
    if kind == "global":
        return ()
    if kind == "shallow":
        return (shallow,)
    if kind == "gap_shallow":
        return (deep - shallow, shallow)
    if kind == "deep_shallow":
        return (deep, shallow)
    if kind == "deep2_shallow_legal":
        return (deep // 2, shallow, legal_bucket(row))
    raise ValueError(kind)


class SigmaModel:
    def __init__(
        self, kind: str, global_variance: float,
        shallow_variances: dict[tuple[int, ...], float],
        variances: dict[tuple[int, ...], float],
    ) -> None:
        self.kind = kind
        self.global_variance = global_variance
        self.shallow_variances = shallow_variances
        self.variances = variances

    def sigma(self, row: dict[str, Any]) -> float:
        key = sigma_key(row, self.kind)
        variance = self.variances.get(key)
        if variance is None:
            variance = self.shallow_variances.get(
                (int(row["shallow_depth"]),), self.global_variance
            )
        return max(0.5, math.sqrt(variance))


def fit_sigma(
    rows: list[dict[str, Any]], mean_model: MeanModel,
    kind: str, shrinkage: float,
) -> SigmaModel:
    residuals = [float(row["deep_value"]) - mean_model.predict(row) for row in rows]
    global_variance = sum(value * value for value in residuals) / len(residuals)
    shallow_sums: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row, residual in zip(rows, residuals):
        selected = shallow_sums[(int(row["shallow_depth"]),)]
        selected[0] += residual * residual
        selected[1] += 1.0
    shallow_variances = {
        key: (total + shrinkage * global_variance) / (count + shrinkage)
        for key, (total, count) in shallow_sums.items()
    }
    sums: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row, residual in zip(rows, residuals):
        selected = sums[sigma_key(row, kind)]
        selected[0] += residual * residual
        selected[1] += 1.0
    variances = {}
    for key, (total, count) in sums.items():
        parent = global_variance
        if kind != "global":
            shallow = key[0] if kind == "shallow" else key[1]
            parent = shallow_variances.get((shallow,), global_variance)
        variances[key] = (total + shrinkage * parent) / (count + shrinkage)
    return SigmaModel(kind, global_variance, shallow_variances, variances)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_fold(
    train: list[dict[str, Any]], test: list[dict[str, Any]],
    mean_name: str, sigma_kind: str, shrinkage: float,
) -> dict[str, Any]:
    mean_model = fit_mean(train, mean_name)
    sigma_model = fit_sigma(train, mean_model, sigma_kind, shrinkage)
    train_z = [
        (float(row["deep_value"]) - mean_model.predict(row)) / sigma_model.sigma(row)
        for row in train
    ]
    test_residuals = [float(row["deep_value"]) - mean_model.predict(row) for row in test]
    test_z = [
        residual / sigma_model.sigma(row)
        for row, residual in zip(test, test_residuals)
    ]
    nll = sum(
        math.log(sigma_model.sigma(row)) + 0.5 * z * z
        for row, z in zip(test, test_z)
    ) / len(test)
    tails = {}
    for level, selectivity in enumerate(SELECTIVITY):
        tail = (1.0 - selectivity) / 2.0
        lower = quantile(train_z, tail)
        upper = quantile(train_z, 1.0 - tail)
        lower_violations = sum(value < lower for value in test_z)
        upper_violations = sum(value > upper for value in test_z)
        tails[str(level)] = {
            "lower_multiplier": -lower, "upper_multiplier": upper,
            "lower_violation_rate": lower_violations / len(test_z),
            "upper_violation_rate": upper_violations / len(test_z),
            "target_tail_rate": tail,
        }
    return {
        "count": len(test),
        "rmse": math.sqrt(sum(value * value for value in test_residuals) / len(test)),
        "mae": sum(abs(value) for value in test_residuals) / len(test),
        "bias": sum(test_residuals) / len(test), "nll": nll,
        "tails": tails, "mean_raw": mean_model.raw_coefficients(),
    }


def parse_inputs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, raw_path = value.split("=", 1)
        result[label] = Path(raw_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "shallow", "static"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = parse_inputs(args.input)
    datasets = {}
    for label, path in inputs.items():
        rows = deduplicate(load_jsonl(path))
        if args.mode == "shallow":
            rows = [row for row in rows if int(row["shallow_depth"]) > 0]
        elif args.mode == "static":
            rows = [row for row in rows if int(row["shallow_depth"]) == 0]
        datasets[label] = rows
    labels = sorted(datasets)
    mean_names = (
        "identity", "bias", "bias_d0", "bias_full",
        "affine_s", "affine_s_d0", "affine_full",
    )
    sigma_specs = [
        ("global", 0.0), ("shallow", 16.0), ("shallow", 64.0),
        ("gap_shallow", 16.0), ("gap_shallow", 64.0),
        ("deep_shallow", 16.0), ("deep_shallow", 64.0),
        ("deep2_shallow_legal", 32.0), ("deep2_shallow_legal", 96.0),
    ]
    candidates = []
    for mean_name in mean_names:
        for sigma_kind, shrinkage in sigma_specs:
            folds = []
            for held_out in labels:
                train = [
                    row for label in labels if label != held_out
                    for row in datasets[label]
                ]
                folds.append({
                    "held_out": held_out,
                    **evaluate_fold(
                        train, datasets[held_out], mean_name,
                        sigma_kind, shrinkage,
                    ),
                })
            count = sum(fold["count"] for fold in folds)
            aggregate = {
                metric: sum(fold[metric] * fold["count"] for fold in folds) / count
                for metric in ("rmse", "mae", "bias", "nll")
            }
            worst_tail_excess = max(
                fold["tails"][str(level)][side + "_violation_rate"] -
                fold["tails"][str(level)]["target_tail_rate"]
                for fold in folds for level in range(3)
                for side in ("lower", "upper")
            )
            candidates.append({
                "mean": mean_name, "sigma": sigma_kind,
                "shrinkage": shrinkage, "count": count,
                **aggregate, "worst_tail_excess": worst_tail_excess,
                "folds": folds,
            })
    candidates.sort(key=lambda row: (row["worst_tail_excess"] > 0.01, row["nll"], row["rmse"]))
    report = {
        "mode": args.mode,
        "inputs": {
            label: {"path": str(inputs[label].resolve()), "rows": len(datasets[label])}
            for label in labels
        },
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for rank, candidate in enumerate(candidates[:15], 1):
        print(
            f"{rank:2d} mean={candidate['mean']:<13} sigma={candidate['sigma']:<20} "
            f"shrink={candidate['shrinkage']:>4.0f} rmse={candidate['rmse']:.4f} "
            f"mae={candidate['mae']:.4f} nll={candidate['nll']:.4f} "
            f"worst_tail_excess={candidate['worst_tail_excess']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
