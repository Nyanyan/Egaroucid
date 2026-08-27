#!/usr/bin/env python3
"""Compare Egaroucid and Edax fixed-depth scores with exact OBF scores."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from pathlib import Path


EDAX_RESULT_RE = re.compile(r"^\s*(\d+)\|\s*(\d+)\s+([<>=]?)([+-]\d+)\b")
DEPTH_FROM_NAME_RE = re.compile(r"edax_depth_(\d+)\.txt$")
MOVE_SCORE_RE = re.compile(r"\b(?:[A-H][1-8]|PS):([+-]?\d+)", re.IGNORECASE)


def outcome(value: int) -> int:
    return (value > 0) - (value < 0)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = probability * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def pearson(xs: list[int], ys: list[int]) -> float:
    if len(xs) < 2:
        return math.nan
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else math.nan


def score_metrics(predictions: dict[str, int], exact: dict[str, int]) -> dict[str, float | int]:
    keys = sorted(set(predictions) & set(exact))
    errors = [predictions[key] - exact[key] for key in keys]
    absolute = [abs(error) for error in errors]
    predicted_values = [predictions[key] for key in keys]
    exact_values = [exact[key] for key in keys]
    n = len(keys)
    if n == 0:
        raise ValueError("no common positions")
    return {
        "n": n,
        "bias": statistics.fmean(errors),
        "mae": statistics.fmean(absolute),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "median_abs_error": statistics.median(absolute),
        "p90_abs_error": percentile(absolute, 0.90),
        "p95_abs_error": percentile(absolute, 0.95),
        "max_abs_error": max(absolute),
        "exact_score_count": sum(error == 0 for error in errors),
        "within_1_count": sum(error <= 1 for error in absolute),
        "within_2_count": sum(error <= 2 for error in absolute),
        "within_4_count": sum(error <= 4 for error in absolute),
        "outcome_agreement_count": sum(
            outcome(predictions[key]) == outcome(exact[key]) for key in keys
        ),
        "false_nonloss_count": sum(
            exact[key] < 0 <= predictions[key] for key in keys
        ),
        "false_loss_count": sum(
            predictions[key] < 0 <= exact[key] for key in keys
        ),
        "pearson": pearson(predicted_values, exact_values),
    }


def read_obf(path: Path) -> tuple[list[str], dict[str, int], set[int]]:
    boards_by_index: list[str] = []
    exact: dict[str, int] = {}
    pass_indices: set[int] = set()
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for index, line in enumerate(source, 1):
            board = line.split(";", 1)[0].strip()
            boards_by_index.append(board)
            scores = [int(value) for value in MOVE_SCORE_RE.findall(line)]
            if re.search(r";\s*PS:", line, re.IGNORECASE):
                pass_indices.add(index)
                continue
            if not scores:
                raise ValueError(f"no exact move score at OBF line {index}")
            exact[board] = max(scores)
    return boards_by_index, exact, pass_indices


def read_egaroucid(pattern: str) -> tuple[dict[int, dict[str, int]], dict[str, int]]:
    by_depth: dict[int, dict[str, int]] = {}
    deep: dict[str, int] = {}
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"no Egaroucid files matched {pattern}")
    for name in paths:
        with open(name, "r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                board = row["board"]
                depth = int(row["shallow_depth"])
                value = int(row["shallow_value"])
                deep_value = int(row["deep_value"])
                previous = by_depth.setdefault(depth, {}).setdefault(board, value)
                if previous != value:
                    raise ValueError(f"inconsistent Egaroucid value for {board} at depth {depth}")
                previous_deep = deep.setdefault(board, deep_value)
                if previous_deep != deep_value:
                    raise ValueError(f"inconsistent Egaroucid exact value for {board}")
    return by_depth, deep


def read_edax(directory: Path, boards_by_index: list[str], pass_indices: set[int]) -> dict[int, dict[str, int]]:
    by_depth: dict[int, dict[str, int]] = {}
    for path in sorted(directory.glob("edax_depth_*.txt")):
        match = DEPTH_FROM_NAME_RE.search(path.name)
        if not match:
            continue
        expected_depth = int(match.group(1))
        values: dict[str, int] = {}
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                result = EDAX_RESULT_RE.match(line)
                if not result:
                    continue
                index = int(result.group(1))
                reported_depth = int(result.group(2))
                bound = result.group(3)
                value = int(result.group(4))
                if reported_depth != expected_depth:
                    raise ValueError(
                        f"Edax reported depth {reported_depth}, expected {expected_depth}: {line.rstrip()}"
                    )
                if bound:
                    raise ValueError(f"bounded Edax result is unsuitable: {line.rstrip()}")
                if index in pass_indices:
                    continue
                board = boards_by_index[index - 1]
                values[board] = value
        by_depth[expected_depth] = values
    if not by_depth:
        raise ValueError("no Edax result files found")
    return by_depth


def paired_comparison(
    left: dict[str, int], right: dict[str, int], exact: dict[str, int]
) -> dict[str, float | int]:
    keys = sorted(set(left) & set(right) & set(exact))
    differences = [abs(left[key] - exact[key]) - abs(right[key] - exact[key]) for key in keys]
    return {
        "n": len(keys),
        "egaroucid_better": sum(value < 0 for value in differences),
        "equal": sum(value == 0 for value in differences),
        "edax_better": sum(value > 0 for value in differences),
        "mean_abs_error_difference_egaroucid_minus_edax": statistics.fmean(differences),
        "median_abs_error_difference_egaroucid_minus_edax": statistics.median(differences),
    }


def depth_transition(values: dict[int, dict[str, int]], exact: dict[str, int]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    depths = sorted(values)
    for previous_depth, depth in zip(depths, depths[1:]):
        keys = sorted(set(values[previous_depth]) & set(values[depth]) & set(exact))
        changes = [
            abs(values[depth][key] - exact[key]) - abs(values[previous_depth][key] - exact[key])
            for key in keys
        ]
        result[f"{previous_depth}->{depth}"] = {
            "improved": sum(value < 0 for value in changes),
            "unchanged": sum(value == 0 for value in changes),
            "worsened": sum(value > 0 for value in changes),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obf", type=Path, required=True)
    parser.add_argument("--egaroucid-glob", required=True)
    parser.add_argument("--edax-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--deep-is-exact",
        action="store_true",
        help="verify the Egaroucid deep_value column against the OBF exact score",
    )
    args = parser.parse_args()

    boards_by_index, exact, pass_indices = read_obf(args.obf)
    egaroucid, egaroucid_deep = read_egaroucid(args.egaroucid_glob)
    edax = read_edax(args.edax_dir, boards_by_index, pass_indices)
    common_depths = sorted(set(egaroucid) & set(edax))
    empties = sorted({board[:64].count("-") for board in exact})
    if len(empties) != 1:
        raise ValueError(f"mixed empty counts in dataset: {empties}")
    if args.deep_is_exact:
        mismatches = [
            {"board": board, "obf": value, "egaroucid": egaroucid_deep.get(board)}
            for board, value in exact.items()
            if egaroucid_deep.get(board) != value
        ]
        agreement_count = len(exact) - len(mismatches)
    else:
        mismatches = []
        agreement_count = None
    report = {
        "dataset": {
            "obf_positions": len(boards_by_index),
            "pass_positions_excluded": len(pass_indices),
            "measured_positions": len(exact),
            "empties": empties[0],
        },
        "exact_cross_check": {
            "performed": args.deep_is_exact,
            "egaroucid_positions": len(egaroucid_deep),
            "agreement_count": agreement_count,
            "mismatches": mismatches,
        },
        "metrics": {
            "egaroucid": {
                str(depth): score_metrics(egaroucid[depth], exact)
                for depth in common_depths
            },
            "edax": {
                str(depth): score_metrics(edax[depth], exact)
                for depth in common_depths
            },
        },
        "paired": {
            str(depth): paired_comparison(egaroucid[depth], edax[depth], exact)
            for depth in common_depths
        },
        "transition": {
            "egaroucid": depth_transition(egaroucid, exact),
            "edax": depth_transition(edax, exact),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "depth\tengine\tn\tbias\tmae\trmse\tp90_abs\twithin2\twithin4\toutcome_agree\tpearson"
    )
    for depth in common_depths:
        for engine in ("egaroucid", "edax"):
            metric = report["metrics"][engine][str(depth)]
            print(
                f"{depth}\t{engine}\t{metric['n']}\t{metric['bias']:.3f}\t"
                f"{metric['mae']:.3f}\t{metric['rmse']:.3f}\t{metric['p90_abs_error']:.1f}\t"
                f"{metric['within_2_count']}\t{metric['within_4_count']}\t"
                f"{metric['outcome_agreement_count']}\t{metric['pearson']:.4f}"
            )
    if args.deep_is_exact:
        print(
            "exact_cross_check="
            f"{report['exact_cross_check']['agreement_count']}/{len(exact)} "
            f"mismatches={len(report['exact_cross_check']['mismatches'])}"
        )
    else:
        print("exact_cross_check=not_performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
