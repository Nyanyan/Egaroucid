#!/usr/bin/env python3
"""Evaluate the selected recalibrated endgame MPC policy and emit parameters."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from optimize_end_probcut_policy_v2 import (
    add_result,
    context_key,
    correct_and_error,
    empty_stats,
    finish_stats,
    group_contexts,
    legacy_decision,
    model_tail_multipliers,
    model_threshold,
    parse_inputs,
)
from screen_end_probcut_v2 import deduplicate, fit_mean, fit_sigma, load_jsonl, quantile


PLUS_TWO_DEPTHS = {0: set(), 1: set(), 2: set()}
SAME_DEPTH_CUSHION = 1.10
SAME_DEPTH_GATE_SLACK = 4
STATIC_CONFIDENCE = 0.99
STATIC_CUSHION = 1.05

# Parameters currently compiled into multi_probcut.hpp.  Keeping this explicit
# runtime-policy evaluator prevents an implementation/model mismatch from being
# hidden by evaluating a freshly fitted (and therefore different) model.
RUNTIME_SHALLOW_DEPTH = (4, 5, 4, 5, 4, 7, 6, 7, 6)
RUNTIME_SHALLOW_SIGMA = (
    4.784937620133198, 4.404753276200285, 5.2572064673113585,
    4.228319028081452, 5.128290894292946, 3.8714224779536655,
    4.498944083748551, 3.5529833917572193, 3.9725633191799896,
)
RUNTIME_SHALLOW_LOWER_TAIL = (
    1.0550310553705888, 1.4922446086063115, 1.7959223408965452,
)
RUNTIME_SHALLOW_UPPER_TAIL = (
    1.0859804512675133, 1.5634343141830378, 1.8446710563924853,
)


def fit_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = deduplicate(rows)
    shallow_rows = [row for row in observations if int(row["shallow_depth"]) > 0]
    static_rows = [row for row in observations if int(row["shallow_depth"]) == 0]
    # The conditional mean only reduced development RMSE by about 1.6% and
    # made wall-clock search less stable.  Keep the robust identity predictor
    # (exact value = shallow value + residual) and model its residual scale.
    shallow_mean = fit_mean(shallow_rows, "identity")
    shallow_sigma = fit_sigma(
        shallow_rows, shallow_mean, "gap_shallow", 64.0
    )
    shallow_tails = model_tail_multipliers(
        shallow_rows, shallow_mean, shallow_sigma
    )
    static_mean = fit_mean(static_rows, "bias")
    static_sigma = fit_sigma(
        static_rows, static_mean, "gap_shallow", 64.0
    )
    static_z = [
        (int(row["deep_value"]) - static_mean.predict(row)) /
        static_sigma.sigma(row)
        for row in static_rows
    ]
    tail = (1.0 - STATIC_CONFIDENCE) / 2.0
    static_tails = (-quantile(static_z, tail), quantile(static_z, 1.0 - tail))
    return {
        "shallow_mean": shallow_mean, "shallow_sigma": shallow_sigma,
        "shallow_tails": shallow_tails, "static_mean": static_mean,
        "static_sigma": static_sigma, "static_tails": static_tails,
    }


def static_decision(row: dict[str, Any], bundle: dict[str, Any]) -> bool:
    bias = bundle["static_mean"].raw_coefficients()["intercept"]
    sigma = bundle["static_sigma"].sigma(row)
    lower, upper = bundle["static_tails"]
    if row["direction"] == "high":
        threshold = math.ceil(
            int(row["beta"]) - bias + STATIC_CUSHION * lower * sigma - 1.0e-12
        )
        return int(row["d0_value"]) >= threshold
    threshold = math.floor(
        int(row["alpha"]) - bias - STATIC_CUSHION * upper * sigma + 1.0e-12
    )
    return int(row["d0_value"]) <= threshold


def selected_shallow_config(base: dict[str, Any]) -> tuple[int, int, float]:
    level = int(base["mpc_level"])
    deep = int(base["deep_depth"])
    plus_two = level in PLUS_TWO_DEPTHS and deep in PLUS_TWO_DEPTHS[level]
    return (
        (2, 0, 1.0)
        if plus_two
        else (0, SAME_DEPTH_GATE_SLACK, SAME_DEPTH_CUSHION)
    )


def runtime_shallow_threshold(base: dict[str, Any]) -> int:
    deep = int(base["deep_depth"])
    level = int(base["mpc_level"])
    sigma = RUNTIME_SHALLOW_SIGMA[deep - 10]
    if base["direction"] == "high":
        error = math.ceil(
            SAME_DEPTH_CUSHION * RUNTIME_SHALLOW_LOWER_TAIL[level] * sigma
            - 1.0e-12
        )
        return int(base["beta"]) + error
    error = math.ceil(
        SAME_DEPTH_CUSHION * RUNTIME_SHALLOW_UPPER_TAIL[level] * sigma
        - 1.0e-12
    )
    return int(base["alpha"]) - error


def add_final_result(
    stats: dict[str, int], base: dict[str, Any], sample: dict[str, Any],
    static_cut: bool, gate: bool, shallow_cut: bool,
) -> None:
    cut = static_cut or shallow_cut
    correct, error = correct_and_error(base)
    deep_nodes = int(base["deep_nodes"])
    probe_nodes = int(sample["shallow_nodes"]) if gate and not static_cut else 0
    stats["contexts"] += 1
    stats["probes"] += int(gate and not static_cut)
    stats["cuts"] += int(cut)
    stats["correct_cuts"] += int(cut and correct)
    stats["wrong_cuts"] += int(cut and not correct)
    stats["wrong_2plus"] += int(cut and not correct and error >= 2)
    stats["wrong_4plus"] += int(cut and not correct and error >= 4)
    stats["probe_nodes"] += probe_nodes
    stats["deep_nodes"] += deep_nodes
    stats["simulated_nodes"] += probe_nodes + (0 if cut else deep_nodes)


def evaluate_contexts(
    contexts: list[dict[str, Any]], bundle: dict[str, Any]
) -> dict[str, Any]:
    baseline = empty_stats()
    final = empty_stats()
    runtime_final = empty_stats()
    static_stats = empty_stats()
    incomplete = 0
    for context in contexts:
        base = context["base"]
        original_depth = int(base["trace_shallow_depth"])
        if original_depth not in context["depths"]:
            incomplete += 1
            continue
        offset, slack, cushion = selected_shallow_config(base)
        selected_depth = original_depth + offset
        sample = context["depths"].get(selected_depth)
        static_sample = context["depths"].get(0)
        if sample is None or static_sample is None:
            incomplete += 1
            continue
        legacy_gate, legacy_cut, _ = legacy_decision(context)
        add_result(
            baseline, context["depths"][original_depth], legacy_gate, legacy_cut
        )
        static_cut = static_decision(static_sample, bundle)
        add_result(static_stats, static_sample, False, static_cut)
        threshold = model_threshold(
            sample, bundle["shallow_mean"], bundle["shallow_sigma"],
            bundle["shallow_tails"], cushion,
        )
        if base["direction"] == "high":
            gate = int(base["d0_value"]) >= threshold - slack
            shallow_cut = (
                not static_cut and gate and
                int(sample["shallow_value"]) >= threshold
            )
        else:
            gate = int(base["d0_value"]) <= threshold + slack
            shallow_cut = (
                not static_cut and gate and
                int(sample["shallow_value"]) <= threshold
            )
        add_final_result(final, base, sample, static_cut, gate, shallow_cut)
        runtime_depth = RUNTIME_SHALLOW_DEPTH[int(base["deep_depth"]) - 10]
        runtime_sample = context["depths"].get(runtime_depth)
        if runtime_sample is None:
            incomplete += 1
            continue
        runtime_threshold = runtime_shallow_threshold(base)
        if base["direction"] == "high":
            runtime_gate = (
                int(base["d0_value"]) >=
                runtime_threshold - SAME_DEPTH_GATE_SLACK
            )
            runtime_shallow_cut = (
                not static_cut and runtime_gate and
                int(runtime_sample["shallow_value"]) >= runtime_threshold
            )
        else:
            runtime_gate = (
                int(base["d0_value"]) <=
                runtime_threshold + SAME_DEPTH_GATE_SLACK
            )
            runtime_shallow_cut = (
                not static_cut and runtime_gate and
                int(runtime_sample["shallow_value"]) <= runtime_threshold
            )
        add_final_result(
            runtime_final, base, runtime_sample, static_cut,
            runtime_gate, runtime_shallow_cut,
        )
    return {
        "baseline": finish_stats(baseline),
        "fitted_identity": finish_stats(final),
        "runtime_final": finish_stats(runtime_final),
        "static_only": finish_stats(static_stats), "incomplete": incomplete,
    }


def serialize_bundle(bundle: dict[str, Any], template_rows: list[dict[str, Any]]) -> dict[str, Any]:
    shallow_sigma_table = {}
    static_sigma_table = {}
    representatives: dict[tuple[int, int], dict[str, Any]] = {}
    for row in deduplicate(template_rows):
        representatives.setdefault(
            (int(row["deep_depth"]), int(row["shallow_depth"])), row
        )
    for deep in range(10, 19):
        shallow_sigma_table[str(deep)] = {}
        for shallow in range(1, deep):
            if (shallow & 1) != (deep & 1):
                continue
            row = representatives.get((deep, shallow))
            if row is None:
                row = {
                    "deep_depth": deep, "shallow_depth": shallow,
                    "legal_count": 6,
                }
            shallow_sigma_table[str(deep)][str(shallow)] = bundle["shallow_sigma"].sigma(row)
        row0 = representatives.get((deep, 0), {
            "deep_depth": deep, "shallow_depth": 0, "legal_count": 6,
        })
        static_sigma_table[str(deep)] = bundle["static_sigma"].sigma(row0)
    return {
        "shallow_mean": bundle["shallow_mean"].raw_coefficients(),
        "shallow_tails": {
            str(level): {"lower": values[0], "upper": values[1]}
            for level, values in bundle["shallow_tails"].items()
        },
        "shallow_sigma": shallow_sigma_table,
        "static_mean": bundle["static_mean"].raw_coefficients(),
        "static_tails": {
            "lower": bundle["static_tails"][0],
            "upper": bundle["static_tails"][1],
            "confidence": STATIC_CONFIDENCE, "cushion": STATIC_CUSHION,
        },
        "static_sigma": static_sigma_table,
        "plus_two_depths": {
            str(level): sorted(depths) for level, depths in PLUS_TWO_DEPTHS.items()
        },
        "same_depth_policy": {
            "slack": SAME_DEPTH_GATE_SLACK,
            "cushion": SAME_DEPTH_CUSHION,
        },
        "runtime_policy": {
            "shallow_depth": list(RUNTIME_SHALLOW_DEPTH),
            "sigma": list(RUNTIME_SHALLOW_SIGMA),
            "lower_tail": list(RUNTIME_SHALLOW_LOWER_TAIL),
            "upper_tail": list(RUNTIME_SHALLOW_UPPER_TAIL),
        },
        "plus_two_policy": {"slack": 0, "cushion": 1.0},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--evaluation-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = parse_inputs(args.input)
    datasets = {label: load_jsonl(path) for label, path in inputs.items()}
    labels = sorted(datasets)
    folds = []
    for held_out in labels:
        train = [
            row for label in labels if label != held_out for row in datasets[label]
        ]
        bundle = fit_bundle(train)
        folds.append({
            "held_out": held_out,
            **evaluate_contexts(group_contexts(datasets[held_out]), bundle),
        })
    all_rows = [row for label in labels for row in datasets[label]]
    final_bundle = fit_bundle(all_rows)
    evaluation = None
    if args.evaluation_input is not None:
        evaluation_rows = load_jsonl(args.evaluation_input)
        evaluation = evaluate_contexts(
            group_contexts(evaluation_rows), final_bundle
        )
    report = {
        "inputs": {label: str(path.resolve()) for label, path in inputs.items()},
        "folds": folds, "evaluation": evaluation,
        "model": serialize_bundle(final_bundle, all_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for fold in folds:
        print(
            fold["held_out"], "baseline", fold["baseline"],
            "runtime", fold["runtime_final"],
            "fitted-identity", fold["fitted_identity"],
            "static", fold["static_only"],
            "incomplete", fold["incomplete"],
        )
    if evaluation is not None:
        print("evaluation", evaluation)
    print("model", json.dumps(report["model"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
