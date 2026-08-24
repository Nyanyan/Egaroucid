#!/usr/bin/env python3
"""Cross-validated MPC threshold, shallow-depth, and admission policy screen."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from screen_end_probcut_v2 import (
    SELECTIVITY,
    deduplicate,
    fit_mean,
    fit_sigma,
    load_jsonl,
    quantile,
)


SELECTIVITY_Z = (1.13, 1.55, 1.81)
PROBCUT_END = (
    -1.3182333120273682, -6.99290557735024,
    -0.05280654146244756, 0.48284187178125065,
    5.289589936037036, 11.940601436361513,
)


def parse_inputs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, raw_path = value.split("=", 1)
        result[label] = Path(raw_path)
    return result


def context_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("root_id", "")), str(row["board"]),
        int(row["deep_depth"]), int(row["trace_shallow_depth"]),
        int(row["alpha"]), int(row["beta"]), str(row["direction"]),
        int(row["mpc_level"]),
    )


def group_contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = context_key(row)
        context = groups.setdefault(key, {"base": row, "depths": {}})
        depth = int(row["shallow_depth"])
        previous = context["depths"].get(depth)
        if previous is not None and int(previous["shallow_value"]) != int(row["shallow_value"]):
            raise ValueError(f"inconsistent shallow value for {key} depth {depth}")
        context["depths"][depth] = row
    return list(groups.values())


def legacy_sigma(row: dict[str, Any]) -> float:
    a, b, c, d, e, f = PROBCUT_END
    value = a * int(row["n_discs"]) / 64.0 + b * int(row["shallow_depth"]) / 60.0
    return c * value**3 + d * value**2 + e * value + f


def correct_and_error(row: dict[str, Any]) -> tuple[bool, int]:
    deep = int(row["deep_value"])
    if row["direction"] == "high":
        boundary = int(row["beta"])
        return deep >= boundary, max(0, boundary - deep)
    boundary = int(row["alpha"])
    return deep <= boundary, max(0, deep - boundary)


def legacy_decision(context: dict[str, Any]) -> tuple[bool, bool, int]:
    row = context["base"]
    shallow_depth = int(row["trace_shallow_depth"])
    sample = context["depths"][shallow_depth]
    error = math.ceil(SELECTIVITY_Z[int(row["mpc_level"])] * legacy_sigma(sample))
    gate_error = max(1, error - 3)
    if row["direction"] == "high":
        gate = int(row["d0_value"]) >= int(row["beta"]) + gate_error
        cut = gate and int(sample["shallow_value"]) >= int(row["beta"]) + error
    else:
        gate = int(row["d0_value"]) <= int(row["alpha"]) - gate_error
        cut = gate and int(sample["shallow_value"]) <= int(row["alpha"]) - error
    return gate, cut, shallow_depth


def model_tail_multipliers(rows: list[dict[str, Any]], mean_model: Any, sigma_model: Any) -> dict[int, tuple[float, float]]:
    z_values = [
        (int(row["deep_value"]) - mean_model.predict(row)) / sigma_model.sigma(row)
        for row in rows
    ]
    result = {}
    for level, selectivity in enumerate(SELECTIVITY):
        tail = (1.0 - selectivity) / 2.0
        result[level] = (-quantile(z_values, tail), quantile(z_values, 1.0 - tail))
    return result


def model_threshold(
    row: dict[str, Any], mean_model: Any, sigma_model: Any,
    multipliers: dict[int, tuple[float, float]], cushion: float,
) -> int:
    raw = mean_model.raw_coefficients()
    lower, upper = multipliers[int(row["mpc_level"])]
    sigma = sigma_model.sigma(row)
    if mean_model.name == "identity":
        if row["direction"] == "high":
            return math.ceil(
                int(row["beta"]) + cushion * lower * sigma - 1.0e-12
            )
        return math.floor(
            int(row["alpha"]) - cushion * upper * sigma + 1.0e-12
        )
    if mean_model.name != "bias_full" or not raw["base_shallow"]:
        raise ValueError("unsupported policy mean model")
    delta_coefficient, deep_coefficient, shallow_coefficient, legal_coefficient = raw["coefficients"]
    shallow_slope = 1.0 - delta_coefficient
    remainder = (
        float(raw["intercept"]) + delta_coefficient * int(row["d0_value"]) +
        deep_coefficient * int(row["deep_depth"]) +
        shallow_coefficient * int(row["shallow_depth"]) +
        legal_coefficient * int(row["legal_count"])
    )
    if row["direction"] == "high":
        value = (int(row["beta"]) + cushion * lower * sigma - remainder) / shallow_slope
        return math.ceil(value - 1.0e-12)
    value = (int(row["alpha"]) - cushion * upper * sigma - remainder) / shallow_slope
    return math.floor(value + 1.0e-12)


def empty_stats() -> dict[str, int]:
    return {
        "contexts": 0, "probes": 0, "cuts": 0, "correct_cuts": 0,
        "wrong_cuts": 0, "wrong_2plus": 0, "wrong_4plus": 0,
        "probe_nodes": 0, "deep_nodes": 0, "simulated_nodes": 0,
    }


def add_result(
    stats: dict[str, int], row: dict[str, Any], gate: bool, cut: bool,
) -> None:
    correct, error = correct_and_error(row)
    deep_nodes = int(row["deep_nodes"])
    probe_nodes = int(row["shallow_nodes"]) if gate else 0
    stats["contexts"] += 1
    stats["probes"] += int(gate)
    stats["cuts"] += int(cut)
    stats["correct_cuts"] += int(cut and correct)
    stats["wrong_cuts"] += int(cut and not correct)
    stats["wrong_2plus"] += int(cut and not correct and error >= 2)
    stats["wrong_4plus"] += int(cut and not correct and error >= 4)
    stats["probe_nodes"] += probe_nodes
    stats["deep_nodes"] += deep_nodes
    stats["simulated_nodes"] += probe_nodes + (0 if cut else deep_nodes)


def finish_stats(stats: dict[str, int]) -> dict[str, Any]:
    result = dict(stats)
    result["cut_rate"] = stats["cuts"] / stats["contexts"] if stats["contexts"] else 0.0
    result["wrong_cut_rate"] = stats["wrong_cuts"] / stats["cuts"] if stats["cuts"] else 0.0
    result["node_ratio"] = stats["simulated_nodes"] / stats["deep_nodes"] if stats["deep_nodes"] else 1.0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--evaluation-input", type=Path)
    parser.add_argument("--mean-model", choices=("identity", "bias_full"), default="bias_full")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = parse_inputs(args.input)
    raw_datasets = {label: load_jsonl(path) for label, path in inputs.items()}
    observation_datasets = {
        label: [row for row in deduplicate(rows) if int(row["shallow_depth"]) > 0]
        for label, rows in raw_datasets.items()
    }
    context_datasets = {
        label: group_contexts(rows) for label, rows in raw_datasets.items()
    }
    labels = sorted(inputs)
    baseline = empty_stats()
    baseline_buckets: dict[tuple[int, int], dict[str, int]] = {}
    configurations: dict[tuple[int, int, float], dict[str, int]] = {}
    bucket_configurations: dict[tuple[int, int, int, int, float], dict[str, int]] = {}
    offsets = (-4, -2, 0, 2, 4)
    slacks = (0, 2, 3, 4, 6, 8, 12, 99)
    cushions = (1.0, 1.05, 1.10, 1.15)
    if args.evaluation_input is not None:
        evaluation_rows = load_jsonl(args.evaluation_input)
        iterations = [(
            [row for label in labels for row in observation_datasets[label]],
            group_contexts(evaluation_rows),
        )]
    else:
        iterations = [
            (
                [
                    row for label in labels if label != held_out
                    for row in observation_datasets[label]
                ],
                context_datasets[held_out],
            )
            for held_out in labels
        ]
    stored_gate_mismatches = 0
    stored_gate_comparisons = 0
    for train_observations, evaluation_contexts in iterations:
        mean_model = fit_mean(train_observations, args.mean_model)
        sigma_model = fit_sigma(
            train_observations, mean_model, "gap_shallow", 64.0
        )
        multipliers = model_tail_multipliers(
            train_observations, mean_model, sigma_model
        )
        for context in evaluation_contexts:
            base = context["base"]
            if int(base["trace_shallow_depth"]) not in context["depths"]:
                continue
            legacy_gate, legacy_cut, legacy_depth = legacy_decision(context)
            if "gate_passed" in base:
                stored_gate_comparisons += 1
                stored_gate_mismatches += int(bool(base["gate_passed"]) != legacy_gate)
            legacy_row = context["depths"][legacy_depth]
            add_result(baseline, legacy_row, legacy_gate, legacy_cut)
            baseline_bucket = baseline_buckets.setdefault(
                (int(base["mpc_level"]), int(base["deep_depth"])), empty_stats()
            )
            add_result(baseline_bucket, legacy_row, legacy_gate, legacy_cut)
            for offset in offsets:
                shallow_depth = legacy_depth + offset
                sample = context["depths"].get(shallow_depth)
                if sample is None or shallow_depth == 0:
                    continue
                for cushion in cushions:
                    threshold = model_threshold(
                        sample, mean_model, sigma_model, multipliers, cushion
                    )
                    for slack in slacks:
                        if base["direction"] == "high":
                            gate = int(base["d0_value"]) >= threshold - slack
                            cut = gate and int(sample["shallow_value"]) >= threshold
                        else:
                            gate = int(base["d0_value"]) <= threshold + slack
                            cut = gate and int(sample["shallow_value"]) <= threshold
                        key = (offset, slack, cushion)
                        stats = configurations.setdefault(key, empty_stats())
                        add_result(stats, sample, gate, cut)
                        bucket_key = (
                            int(base["mpc_level"]), int(base["deep_depth"]),
                            offset, slack, cushion,
                        )
                        bucket_stats = bucket_configurations.setdefault(
                            bucket_key, empty_stats()
                        )
                        add_result(bucket_stats, sample, gate, cut)
    finished = []
    for (offset, slack, cushion), stats in configurations.items():
        if stats["contexts"] != baseline["contexts"]:
            continue
        finished.append({
            "offset": offset, "slack": slack, "cushion": cushion,
            **finish_stats(stats),
        })
    finished.sort(key=lambda row: (
        row["wrong_4plus"], row["wrong_2plus"], row["wrong_cuts"],
        row["simulated_nodes"],
    ))
    buckets = [
        {
            "mpc_level": key[0], "deep_depth": key[1],
            "offset": key[2], "slack": key[3], "cushion": key[4],
            **finish_stats(stats),
        }
        for key, stats in bucket_configurations.items()
    ]
    report = {
        "inputs": {label: str(path.resolve()) for label, path in inputs.items()},
        "baseline": finish_stats(baseline),
        "baseline_buckets": [
            {"mpc_level": key[0], "deep_depth": key[1], **finish_stats(stats)}
            for key, stats in baseline_buckets.items()
        ],
        "stored_gate_comparisons": stored_gate_comparisons,
        "stored_gate_mismatches": stored_gate_mismatches,
        "configurations": finished,
        "bucket_configurations": buckets,
    }
    if args.evaluation_input is not None:
        static_train = [
            row for label in labels
            for row in deduplicate(raw_datasets[label])
            if int(row["shallow_depth"]) == 0
        ]
        static_mean = fit_mean(static_train, "bias")
        static_sigma = fit_sigma(
            static_train, static_mean, "gap_shallow", 64.0
        )
        static_z = [
            (int(row["deep_value"]) - static_mean.predict(row)) /
            static_sigma.sigma(row)
            for row in static_train
        ]
        static_results = []
        for confidence in (0.98, 0.99, 0.995, 0.999):
            tail = (1.0 - confidence) / 2.0
            lower = -quantile(static_z, tail)
            upper = quantile(static_z, 1.0 - tail)
            for cushion in (1.0, 1.05, 1.10, 1.20):
                stats = empty_stats()
                for context in group_contexts(load_jsonl(args.evaluation_input)):
                    sample = context["depths"].get(0)
                    if sample is None:
                        continue
                    bias = static_mean.raw_coefficients()["intercept"]
                    sigma = static_sigma.sigma(sample)
                    if sample["direction"] == "high":
                        threshold = math.ceil(
                            int(sample["beta"]) - bias + cushion * lower * sigma - 1.0e-12
                        )
                        cut = int(sample["d0_value"]) >= threshold
                    else:
                        threshold = math.floor(
                            int(sample["alpha"]) - bias - cushion * upper * sigma + 1.0e-12
                        )
                        cut = int(sample["d0_value"]) <= threshold
                    add_result(stats, sample, False, cut)
                static_results.append({
                    "confidence": confidence, "cushion": cushion,
                    "lower_multiplier": lower, "upper_multiplier": upper,
                    "mean": static_mean.raw_coefficients(),
                    **finish_stats(stats),
                })
        report["static_direct"] = static_results
        legacy_slack_results = []
        evaluation_contexts = group_contexts(load_jsonl(args.evaluation_input))
        for slack in (0, 1, 2, 3, 4, 5, 6, 8, 12, 99):
            stats = empty_stats()
            for context in evaluation_contexts:
                base = context["base"]
                shallow_depth = int(base["trace_shallow_depth"])
                sample = context["depths"].get(shallow_depth)
                if sample is None:
                    continue
                error = math.ceil(
                    SELECTIVITY_Z[int(base["mpc_level"])] * legacy_sigma(sample)
                )
                if base["direction"] == "high":
                    threshold = int(base["beta"]) + error
                    gate = int(base["d0_value"]) >= threshold - slack
                    cut = gate and int(sample["shallow_value"]) >= threshold
                else:
                    threshold = int(base["alpha"]) - error
                    gate = int(base["d0_value"]) <= threshold + slack
                    cut = gate and int(sample["shallow_value"]) <= threshold
                add_result(stats, sample, gate, cut)
            legacy_slack_results.append({"slack": slack, **finish_stats(stats)})
        report["legacy_slack"] = legacy_slack_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("baseline", json.dumps(report["baseline"], sort_keys=True))
    baseline_finished = report["baseline"]
    eligible = [
        row for row in finished
        if (
            row["wrong_4plus"] <= baseline_finished["wrong_4plus"] and
            row["wrong_2plus"] <= baseline_finished["wrong_2plus"] and
            row["wrong_cuts"] <= baseline_finished["wrong_cuts"] and
            row["correct_cuts"] >= math.floor(0.99 * baseline_finished["correct_cuts"])
        )
    ]
    eligible.sort(key=lambda row: row["simulated_nodes"])
    print("eligible configurations")
    for row in eligible[:20]:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
