#!/usr/bin/env python3
"""Choose one shallow-search depth for each generic endgame MPC depth."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
INPUTS = (
    HERE / "end_generic_depths_19_21/samples.jsonl",
    HERE / "end_generic_depths_23_25/samples.jsonl",
)
COEFFICIENTS = HERE / "end_generic_policy.json"
Z = (1.13, 1.55, 1.81)
CURRENT_DEPTHS = {19: 7, 20: 8, 21: 9, 23: 9, 24: 8, 25: 11}


def context_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("root_id", "")), str(row["board"]),
        int(row["deep_depth"]), int(row["alpha"]), int(row["beta"]),
        str(row["direction"]), int(row["mpc_level"]),
    )


def load() -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in INPUTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            context = grouped.setdefault(context_key(row), {"base": row, "samples": {}})
            context["samples"][int(row["shallow_depth"])] = row
    return list(grouped.values())


def is_confirmation(context: dict[str, Any]) -> bool:
    token = "generic-end-depth|" + str(context["base"].get("root_id", ""))
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 5 == 0


def sigma(coefficients: list[float], row: dict[str, Any]) -> float:
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


def empty_stats() -> dict[str, int]:
    return {
        "contexts": 0, "shallow_searches": 0, "cuts": 0,
        "wrong_cuts": 0, "wrong_2plus": 0, "wrong_4plus": 0,
        "shallow_nodes": 0, "deep_nodes": 0, "simulated_nodes": 0,
        "shallow_ms": 0, "deep_ms": 0, "simulated_ms": 0,
    }


def finish(stats: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(stats)
    result["node_ratio"] = stats["simulated_nodes"] / stats["deep_nodes"]
    result["time_ratio"] = stats["simulated_ms"] / stats["deep_ms"]
    return result


def evaluate(
    contexts: list[dict[str, Any]], depths: dict[int, int],
    coefficients: dict[str, list[float]], slacks: dict[str, dict[int, int]],
) -> dict[str, Any]:
    stats = empty_stats()
    for context in contexts:
        base = context["base"]
        deep = int(base["deep_depth"])
        row = context["samples"].get(depths[deep])
        if row is None:
            raise ValueError(f"missing depth {depths[deep]} at deep depth {deep}")
        direction = str(row["direction"])
        margin = math.ceil(
            Z[int(row["mpc_level"])] * sigma(coefficients[direction], row) - 1e-12
        )
        gate_margin = max(1, margin - slacks[direction][deep])
        if direction == "high":
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
        stats["contexts"] += 1
        stats["shallow_searches"] += int(run_shallow)
        stats["cuts"] += int(cut)
        stats["wrong_cuts"] += int(cut and not correct)
        stats["wrong_2plus"] += int(cut and not correct and error >= 2)
        stats["wrong_4plus"] += int(cut and not correct and error >= 4)
        stats["shallow_nodes"] += shallow_nodes
        stats["deep_nodes"] += deep_nodes
        stats["simulated_nodes"] += shallow_nodes + (0 if cut else deep_nodes)
        stats["shallow_ms"] += shallow_ms
        stats["deep_ms"] += deep_ms
        stats["simulated_ms"] += shallow_ms + (0 if cut else deep_ms)
    return finish(stats)


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


def main() -> int:
    contexts = load()
    selection = [context for context in contexts if not is_confirmation(context)]
    confirmation = [context for context in contexts if is_confirmation(context)]
    fitted = json.loads(COEFFICIENTS.read_text(encoding="utf-8"))["directions"]
    coefficients = {
        direction: [float(value) for value in fitted[direction]["coefficients"]]
        for direction in ("high", "low")
    }
    slacks = {
        direction: {int(depth): int(value) for depth, value in fitted[direction]["slacks"].items()}
        for direction in ("high", "low")
    }
    depths = dict(CURRENT_DEPTHS)
    selection_baseline = evaluate(selection, depths, coefficients, slacks)
    choices: dict[str, Any] = {}
    for deep in sorted(depths):
        depth_contexts = [
            context for context in selection
            if int(context["base"]["deep_depth"]) == deep
        ]
        baseline = evaluate(depth_contexts, depths, coefficients, slacks)
        available = sorted(set.intersection(*(
            set(context["samples"]) for context in depth_contexts
        )))
        candidates = []
        for shallow in available:
            candidate_depths = dict(depths)
            candidate_depths[deep] = shallow
            stats = evaluate(depth_contexts, candidate_depths, coefficients, slacks)
            if allowed(stats, baseline):
                candidates.append((objective(stats, baseline),
                                   abs(shallow - CURRENT_DEPTHS[deep]), shallow, stats))
        candidates.sort(key=lambda item: item[:3])
        _, _, selected, stats = candidates[0]
        depths[deep] = selected
        choices[str(deep)] = {
            "available": available, "selected": selected,
            "baseline": baseline, "candidate": stats,
        }
    result = {
        "counts": {"all": len(contexts), "selection": len(selection),
                   "confirmation": len(confirmation)},
        "depths": depths,
        "choices": choices,
        "selection_baseline": selection_baseline,
        "selection_candidate": evaluate(selection, depths, coefficients, slacks),
        "confirmation_baseline": evaluate(
            confirmation, CURRENT_DEPTHS, coefficients, slacks
        ),
        "confirmation_candidate": evaluate(
            confirmation, depths, coefficients, slacks
        ),
        "confirmation_by_depth": {
            str(deep): {
                "baseline": evaluate(
                    [context for context in confirmation
                     if int(context["base"]["deep_depth"]) == deep],
                    CURRENT_DEPTHS, coefficients, slacks,
                ),
                "candidate": evaluate(
                    [context for context in confirmation
                     if int(context["base"]["deep_depth"]) == deep],
                    depths, coefficients, slacks,
                ),
            }
            for deep in sorted(depths)
        },
    }
    output = HERE / "end_generic_depth_policy.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
