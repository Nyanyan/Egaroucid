#!/usr/bin/env python3
"""Tune the generic endgame MPC model against cut errors and search cost."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INPUTS = (
    REPO / "benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit/"
    "high_depth_samples/samples.jsonl",
    REPO / "benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit/"
    "deep23_26_samples/samples.jsonl",
)
CURRENT = (-1.3182333120273682, -6.99290557735024,
           -0.05280654146244756, 0.48284187178125065,
           5.289589936037036, 11.940601436361513)
Z = (1.13, 1.55, 1.81)
CURRENT_SLACK = 3


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("root_id", "")), str(row["board"]),
        int(row["deep_depth"]), int(row["alpha"]), int(row["beta"]),
        str(row["direction"]), int(row["mpc_level"]),
    )


def load() -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for path in INPUTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            grouped.setdefault(key(row), {})[int(row["shallow_depth"])] = row
    result = []
    for samples in grouped.values():
        shallow = [row for depth, row in samples.items() if depth > 0]
        if len(shallow) == 1:
            result.append(shallow[0])
    return result


def is_confirmation(row: dict[str, Any]) -> bool:
    token = "generic-end-direct-objective|" + str(row.get("root_id", ""))
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 5 == 0


def sigma(coefficients: list[float] | tuple[float, ...], row: dict[str, Any]) -> float:
    a, b, c, d, e, f = coefficients
    value = a * int(row["n_discs"]) / 64.0 + b * int(row["shallow_depth"]) / 60.0
    return max(0.5, c * value**3 + d * value**2 + e * value + f)


def correct_and_error(row: dict[str, Any]) -> tuple[bool, int]:
    deep = int(row["deep_value"])
    if row["direction"] == "high":
        boundary = int(row["beta"])
        return deep >= boundary, max(0, boundary - deep)
    boundary = int(row["alpha"])
    return deep <= boundary, max(0, deep - boundary)


def evaluate(
    rows: list[dict[str, Any]], coefficients: list[float] | tuple[float, ...],
    slacks: dict[int, int],
) -> dict[str, Any]:
    result = {
        "contexts": 0, "shallow_searches": 0, "cuts": 0,
        "wrong_cuts": 0, "wrong_2plus": 0, "wrong_4plus": 0,
        "shallow_nodes": 0, "deep_nodes": 0, "simulated_nodes": 0,
        "shallow_ms": 0, "deep_ms": 0, "simulated_ms": 0,
    }
    for row in rows:
        level = int(row["mpc_level"])
        margin = math.ceil(Z[level] * sigma(coefficients, row) - 1.0e-12)
        gate_margin = max(1, margin - slacks.get(int(row["deep_depth"]), CURRENT_SLACK))
        if row["direction"] == "high":
            run_shallow = int(row["d0_value"]) >= int(row["beta"]) + gate_margin
            cut = run_shallow and int(row["shallow_value"]) >= int(row["beta"]) + margin
        else:
            run_shallow = int(row["d0_value"]) <= int(row["alpha"]) - gate_margin
            cut = run_shallow and int(row["shallow_value"]) <= int(row["alpha"]) - margin
        correct, error = correct_and_error(row)
        shallow_nodes = int(row["shallow_nodes"]) if run_shallow else 0
        shallow_ms = int(row.get("shallow_ms", 0)) if run_shallow else 0
        deep_nodes = int(row["deep_nodes"])
        deep_ms = int(row.get("deep_ms", 0))
        result["contexts"] += 1
        result["shallow_searches"] += int(run_shallow)
        result["cuts"] += int(cut)
        result["wrong_cuts"] += int(cut and not correct)
        result["wrong_2plus"] += int(cut and not correct and error >= 2)
        result["wrong_4plus"] += int(cut and not correct and error >= 4)
        result["shallow_nodes"] += shallow_nodes
        result["deep_nodes"] += deep_nodes
        result["simulated_nodes"] += shallow_nodes + (0 if cut else deep_nodes)
        result["shallow_ms"] += shallow_ms
        result["deep_ms"] += deep_ms
        result["simulated_ms"] += shallow_ms + (0 if cut else deep_ms)
    result["node_ratio"] = result["simulated_nodes"] / result["deep_nodes"]
    result["time_ratio"] = (
        result["simulated_ms"] / result["deep_ms"] if result["deep_ms"] else 1.0
    )
    return result


def allowed(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        candidate["wrong_4plus"] <= baseline["wrong_4plus"] and
        candidate["wrong_2plus"] <= baseline["wrong_2plus"] and
        candidate["wrong_cuts"] <= baseline["wrong_cuts"]
    )


def objective(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    return 0.5 * (
        candidate["simulated_nodes"] / baseline["simulated_nodes"] +
        candidate["simulated_ms"] / baseline["simulated_ms"]
    )


def tune_direction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_slacks = {depth: CURRENT_SLACK for depth in range(19, 26)}
    baseline = evaluate(rows, CURRENT, current_slacks)
    coefficients = list(CURRENT)
    slacks = dict(current_slacks)
    best = baseline

    # Coordinate changes alter the shape of the model.  No common multiplier
    # is applied to all predicted errors.
    starting_steps = (0.08, 0.30, 0.012, 0.05, 0.30, 0.45)
    for reduction in (1.0, 0.5, 0.25):
        changed = True
        while changed:
            changed = False
            for index, base_step in enumerate(starting_steps):
                for sign in (-1.0, 1.0):
                    candidate_coefficients = coefficients[:]
                    candidate_coefficients[index] += sign * base_step * reduction
                    candidate = evaluate(rows, candidate_coefficients, slacks)
                    if allowed(candidate, baseline) and objective(candidate, baseline) + 1e-9 < objective(best, baseline):
                        coefficients, best = candidate_coefficients, candidate
                        changed = True
            for depth in sorted({int(row["deep_depth"]) for row in rows}):
                for value in range(0, 9):
                    candidate_slacks = dict(slacks)
                    candidate_slacks[depth] = value
                    candidate = evaluate(rows, coefficients, candidate_slacks)
                    if allowed(candidate, baseline) and objective(candidate, baseline) + 1e-9 < objective(best, baseline):
                        slacks, best = candidate_slacks, candidate
                        changed = True
    return {
        "coefficients": coefficients,
        "slacks": slacks,
        "baseline": baseline,
        "candidate": best,
    }


def main() -> int:
    rows = load()
    selection = [row for row in rows if not is_confirmation(row)]
    confirmation = [row for row in rows if is_confirmation(row)]
    result: dict[str, Any] = {
        "counts": {"all": len(rows), "selection": len(selection),
                   "confirmation": len(confirmation)},
        "directions": {},
    }
    for direction in ("high", "low"):
        selected_rows = [row for row in selection if row["direction"] == direction]
        checked_rows = [row for row in confirmation if row["direction"] == direction]
        tuned = tune_direction(selected_rows)
        current_slacks = {depth: CURRENT_SLACK for depth in range(19, 26)}
        tuned["confirmation_baseline"] = evaluate(
            checked_rows, CURRENT, current_slacks
        )
        tuned["confirmation_candidate"] = evaluate(
            checked_rows, tuned["coefficients"], tuned["slacks"]
        )
        result["directions"][direction] = tuned
    output = HERE / "end_generic_policy.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
