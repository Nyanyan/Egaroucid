#!/usr/bin/env python3
"""Choose the depth, directional margins, and admission checks for end MPC.

The objective is the quantity used by the search rather than a least-squares
fit of residuals: do not increase wrong cuts of at least two discs, and among
the remaining choices minimize the simulated node count.  The upper and lower
directions are optimized independently.  Root positions used for final
validation are never used to choose a parameter.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit"

DEPTHS = tuple(range(10, 19))
CURRENT_DEPTH = (4, 5, 4, 5, 4, 7, 6, 7, 6)
CURRENT_HIGH = (
    (6, 6, 7, 5, 6, 5, 6, 5, 5),
    (8, 8, 9, 7, 9, 7, 8, 6, 7),
    (10, 9, 11, 9, 11, 8, 9, 8, 8),
)
CURRENT_LOW = (
    (6, 6, 7, 6, 7, 5, 6, 5, 5),
    (9, 8, 10, 8, 9, 7, 8, 7, 7),
    (10, 9, 11, 9, 11, 8, 10, 8, 9),
)
STATIC_HIGH = (18, 19, 20, 21, 21, 19, 20, 19, 19)
STATIC_LOW = (-27, -28, -29, -30, -30, -29, -29, -29, -28)
CURRENT_SLACK = 4
OFFSET_DIRS = ("offset_m2", "offset_p0", "offset_p2", "offset_p4")
TRAIN_FILES = (
    "dev743.jsonl", "dev202607.jsonl", "dev202608a.jsonl",
    "dev202608b.jsonl", "admission_train.jsonl",
)
VALIDATION_FILES = ("holdout202606.jsonl", "admission_holdout.jsonl")


def load_rows(file_names: Iterable[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    contexts: dict[tuple[Any, ...], dict[str, Any]] = {}
    for directory in OFFSET_DIRS:
        for file_name in file_names:
            path = SOURCE / directory / file_name
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                row = json.loads(line)
                key = (
                    str(row.get("root_id", "")), str(row["board"]),
                    int(row["deep_depth"]), int(row["alpha"]),
                    int(row["beta"]), str(row["direction"]),
                    int(row["mpc_level"]),
                )
                context = contexts.setdefault(key, {"base": row, "samples": {}})
                shallow = int(row["shallow_depth"])
                previous = context["samples"].get(shallow)
                if previous is not None:
                    for field in ("shallow_value", "deep_value", "deep_nodes"):
                        if int(previous[field]) != int(row[field]):
                            raise ValueError(f"inconsistent {field} for {key}")
                else:
                    context["samples"][shallow] = row
    return contexts


def static_cut(row: dict[str, Any]) -> bool:
    index = int(row["deep_depth"]) - DEPTHS[0]
    if row["direction"] == "high":
        return int(row["d0_value"]) >= int(row["beta"]) + STATIC_HIGH[index]
    return int(row["d0_value"]) <= int(row["alpha"]) + STATIC_LOW[index]


def correct_and_error(row: dict[str, Any]) -> tuple[bool, int]:
    deep = int(row["deep_value"])
    if row["direction"] == "high":
        boundary = int(row["beta"])
        return deep >= boundary, max(0, boundary - deep)
    boundary = int(row["alpha"])
    return deep <= boundary, max(0, deep - boundary)


def decide(row: dict[str, Any], margin: int, slack: int) -> tuple[bool, bool]:
    if static_cut(row):
        return False, True
    if row["direction"] == "high":
        threshold = int(row["beta"]) + margin
        run_shallow = int(row["d0_value"]) >= threshold - slack
        cut = run_shallow and int(row["shallow_value"]) >= threshold
    else:
        threshold = int(row["alpha"]) - margin
        run_shallow = int(row["d0_value"]) <= threshold + slack
        cut = run_shallow and int(row["shallow_value"]) <= threshold
    return run_shallow, cut


def empty_stats() -> dict[str, int]:
    return {
        "contexts": 0, "shallow_searches": 0, "cuts": 0,
        "wrong_cuts": 0, "wrong_2plus": 0, "wrong_4plus": 0,
        "shallow_nodes": 0, "deep_nodes": 0, "simulated_nodes": 0,
    }


def evaluate(rows: list[dict[str, Any]], margin: int, slack: int) -> dict[str, int]:
    stats = empty_stats()
    for row in rows:
        run_shallow, cut = decide(row, margin, slack)
        correct, error = correct_and_error(row)
        shallow_nodes = int(row["shallow_nodes"]) if run_shallow else 0
        deep_nodes = int(row["deep_nodes"])
        stats["contexts"] += 1
        stats["shallow_searches"] += int(run_shallow)
        stats["cuts"] += int(cut)
        stats["wrong_cuts"] += int(cut and not correct)
        stats["wrong_2plus"] += int(cut and not correct and error >= 2)
        stats["wrong_4plus"] += int(cut and not correct and error >= 4)
        stats["shallow_nodes"] += shallow_nodes
        stats["deep_nodes"] += deep_nodes
        stats["simulated_nodes"] += shallow_nodes + (0 if cut else deep_nodes)
    return stats


def eligible(candidate: dict[str, int], baseline: dict[str, int]) -> bool:
    return (
        candidate["wrong_4plus"] <= baseline["wrong_4plus"] and
        candidate["wrong_2plus"] <= baseline["wrong_2plus"] and
        candidate["wrong_cuts"] <= baseline["wrong_cuts"]
    )


def finish(stats: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(stats)
    result["node_ratio"] = (
        stats["simulated_nodes"] / stats["deep_nodes"]
        if stats["deep_nodes"] else 1.0
    )
    return result


def add(into: dict[str, int], values: dict[str, int]) -> None:
    for key in into:
        into[key] += values[key]


def rows_for(
    contexts: dict[tuple[Any, ...], dict[str, Any]], deep: int,
    direction: str, level: int, shallow: int,
) -> list[dict[str, Any]]:
    result = []
    for context in contexts.values():
        base = context["base"]
        if (
            int(base["deep_depth"]) == deep and
            str(base["direction"]) == direction and
            int(base["mpc_level"]) == level and
            shallow in context["samples"]
        ):
            result.append(context["samples"][shallow])
    return result


def best_parameters(
    rows: list[dict[str, Any]], baseline: dict[str, int], current_margin: int,
    tune_margin: bool, tune_slack: bool,
) -> tuple[int, int, dict[str, int]]:
    choices = []
    # A one-disc reduction is the largest unobserved extrapolation allowed.
    # Larger margins remain available when the data show that the current
    # setting is unsafe.  This prevents a sparse bucket from choosing an
    # unrealistically small margin merely because it happened to contain no
    # wrong cut.
    margins = (
        range(max(1, current_margin - 1), current_margin + 5)
        if tune_margin else (current_margin,)
    )
    slacks = range(0, 9) if tune_slack else (CURRENT_SLACK,)
    for margin in margins:
        for slack in slacks:
            stats = evaluate(rows, margin, slack)
            if eligible(stats, baseline):
                choices.append((stats["simulated_nodes"], abs(margin - current_margin),
                                abs(slack - CURRENT_SLACK), margin, slack, stats))
    if not choices:
        return current_margin, CURRENT_SLACK, baseline
    choices.sort(key=lambda item: item[:-1])
    _, _, _, margin, slack, stats = choices[0]
    return margin, slack, stats


def choose_policy(
    contexts: dict[tuple[Any, ...], dict[str, Any]], *,
    tune_depth: bool, tune_margin: bool, tune_slack: bool,
) -> dict[str, Any]:
    policy: dict[str, Any] = {"depths": {}, "parameters": {}}
    baseline_total = empty_stats()
    candidate_total = empty_stats()
    for deep in DEPTHS:
        index = deep - DEPTHS[0]
        current_depth = CURRENT_DEPTH[index]
        available = sorted({
            shallow
            for context in contexts.values()
            if int(context["base"]["deep_depth"]) == deep
            for shallow in context["samples"]
            if shallow > 0
        })
        if not tune_depth:
            available = [current_depth]
        depth_choices = []
        for shallow in available:
            selected: dict[str, Any] = {}
            combined_baseline = empty_stats()
            combined_candidate = empty_stats()
            complete = True
            for direction in ("high", "low"):
                for level in range(3):
                    current_margin = (
                        CURRENT_HIGH[level][index]
                        if direction == "high" else CURRENT_LOW[level][index]
                    )
                    base_rows = rows_for(
                        contexts, deep, direction, level, current_depth
                    )
                    candidate_rows = rows_for(
                        contexts, deep, direction, level, shallow
                    )
                    if not base_rows or len(candidate_rows) != len(base_rows):
                        complete = False
                        break
                    baseline = evaluate(base_rows, current_margin, CURRENT_SLACK)
                    margin, slack, stats = best_parameters(
                        candidate_rows, baseline, current_margin,
                        tune_margin, tune_slack
                    )
                    selected[f"{direction}_{level}"] = {
                        "margin": margin, "slack": slack,
                        "selection": finish(stats),
                        "baseline": finish(baseline),
                    }
                    add(combined_baseline, baseline)
                    add(combined_candidate, stats)
                if not complete:
                    break
            if complete:
                depth_choices.append((
                    combined_candidate["simulated_nodes"],
                    abs(shallow - current_depth), shallow, selected,
                    combined_baseline, combined_candidate,
                ))
        if not depth_choices:
            raise ValueError(f"no complete candidate at deep depth {deep}")
        depth_choices.sort(key=lambda item: item[:3])
        _, _, shallow, selected, combined_baseline, combined_candidate = depth_choices[0]
        policy["depths"][str(deep)] = shallow
        for key, value in selected.items():
            policy["parameters"][f"{deep}_{key}"] = value
        add(baseline_total, combined_baseline)
        add(candidate_total, combined_candidate)
    policy["selection_baseline"] = finish(baseline_total)
    policy["selection_candidate"] = finish(candidate_total)
    return policy


def evaluate_policy(
    contexts: dict[tuple[Any, ...], dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    baseline_total = empty_stats()
    candidate_total = empty_stats()
    by_direction = {
        "baseline": {"high": empty_stats(), "low": empty_stats()},
        "candidate": {"high": empty_stats(), "low": empty_stats()},
    }
    by_depth = {
        "baseline": {str(deep): empty_stats() for deep in DEPTHS},
        "candidate": {str(deep): empty_stats() for deep in DEPTHS},
    }
    for deep in DEPTHS:
        index = deep - DEPTHS[0]
        shallow = int(policy["depths"][str(deep)])
        for direction in ("high", "low"):
            for level in range(3):
                base_rows = rows_for(
                    contexts, deep, direction, level, CURRENT_DEPTH[index]
                )
                candidate_rows = rows_for(contexts, deep, direction, level, shallow)
                current_margin = (
                    CURRENT_HIGH[level][index]
                    if direction == "high" else CURRENT_LOW[level][index]
                )
                parameter = policy["parameters"][f"{deep}_{direction}_{level}"]
                baseline = evaluate(base_rows, current_margin, CURRENT_SLACK)
                candidate = evaluate(
                    candidate_rows, int(parameter["margin"]), int(parameter["slack"])
                )
                add(baseline_total, baseline)
                add(candidate_total, candidate)
                add(by_direction["baseline"][direction], baseline)
                add(by_direction["candidate"][direction], candidate)
                add(by_depth["baseline"][str(deep)], baseline)
                add(by_depth["candidate"][str(deep)], candidate)
    return {
        "baseline": finish(baseline_total),
        "candidate": finish(candidate_total),
        "by_direction": {
            kind: {direction: finish(stats) for direction, stats in values.items()}
            for kind, values in by_direction.items()
        },
        "by_depth": {
            kind: {depth: finish(stats) for depth, stats in values.items()}
            for kind, values in by_depth.items()
        },
    }


def main() -> int:
    training = load_rows(TRAIN_FILES)
    validation = load_rows(VALIDATION_FILES)
    training_roots = {key[0] for key in training}
    validation_roots = {key[0] for key in validation}
    overlap = training_roots & validation_roots
    if overlap:
        raise ValueError(f"training/validation root overlap: {len(overlap)}")
    variants = {
        "gate_only": (False, False, True),
        "margin_only": (False, True, False),
        "gate_and_margin": (False, True, True),
        "depth_and_margin": (True, True, False),
        "combined": (True, True, True),
    }
    report = {"variants": {}, "counts": {
        "selection_contexts": len(training),
        "validation_contexts": len(validation),
        "selection_roots": len(training_roots),
        "validation_roots": len(validation_roots),
    }}
    for name, (tune_depth, tune_margin, tune_slack) in variants.items():
        policy = choose_policy(
            training, tune_depth=tune_depth,
            tune_margin=tune_margin, tune_slack=tune_slack,
        )
        policy["validation"] = evaluate_policy(validation, policy)
        report["variants"][name] = policy
    output = HERE / "end_policy_variants.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    concise = {name: {
        "depths": policy["depths"],
        "selection_baseline": policy["selection_baseline"],
        "selection_candidate": policy["selection_candidate"],
        "validation_baseline": policy["validation"]["baseline"],
        "validation_candidate": policy["validation"]["candidate"],
    } for name, policy in report["variants"].items()}
    print(json.dumps(concise, indent=2, sort_keys=True))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
