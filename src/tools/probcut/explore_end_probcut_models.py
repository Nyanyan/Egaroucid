#!/usr/bin/env python3
"""Cross-month exploration of conditional endgame ProbCut quantiles.

This is an offline tool.  It never runs in the engine search path.  Candidate
models use only features that are already available before the single shallow
NWS: depths, d0, direction, legal count, and the existing sigma as a scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SELECTIVITY_MPCT = (1.13, 1.55, 1.81)
SELECTIVITY_PERCENT = (74.0, 88.0, 93.0)
PROBCUT_END = (
    -1.3182333120273682,
    -6.99290557735024,
    -0.05280654146244756,
    0.48284187178125065,
    5.289589936037036,
    11.940601436361513,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sigma_end(row: dict[str, Any]) -> float:
    a, b, c, d, e, f = PROBCUT_END
    value = a * int(row["n_discs"]) / 64.0 + b * int(row["shallow_depth"]) / 60.0
    return c * value**3 + d * value**2 + e * value + f


def boundary(row: dict[str, Any]) -> int:
    return int(row["beta"] if row["direction"] == "high" else row["alpha"])


def legacy_threshold(row: dict[str, Any]) -> int:
    error = math.ceil(SELECTIVITY_MPCT[int(row["mpc_level"])] * sigma_end(row))
    return boundary(row) + error if row["direction"] == "high" else boundary(row) - error


def conservative_quantile(values: list[float], tail: float, direction: str) -> float:
    ordered = sorted(values)
    if direction == "high":
        index = max(0, math.floor(tail * len(ordered)) - 1)
    else:
        index = min(len(ordered) - 1, math.ceil((1.0 - tail) * len(ordered)))
    return float(ordered[index])


def feature_names(base: str) -> list[str]:
    return {
        "ols_s": ["shallow"],
        "ols_s_sigma": ["shallow", "sigma"],
        "ols_s_d0": ["shallow", "d0"],
        "ols_s_d0_sigma": ["shallow", "d0", "sigma"],
        "ols_s_d0_sigma2": ["shallow", "d0", "sigma", "sigma2"],
        "ols_s_d0_deep": ["shallow", "d0", "deep"],
        "ols_s_d0_deep_legal": ["shallow", "d0", "deep", "legal"],
        "ols_s_d0_sigma_legal": ["shallow", "d0", "sigma", "legal"],
    }[base]


def feature_value(row: dict[str, Any], name: str) -> float:
    sigma = sigma_end(row)
    return {
        "shallow": float(row["shallow_value"]),
        "d0": float(row["d0_value"]),
        "sigma": sigma,
        "sigma2": sigma * sigma,
        "deep": float(row["deep_depth"]),
        "legal": float(row["legal_count"]),
    }[name]


def fit_positive_ridge(
    rows: list[dict[str, Any]], names: list[str], ridge: float, loss: str
) -> dict[str, Any]:
    matrix = np.asarray([[feature_value(row, name) for name in names] for row in rows], dtype=float)
    target = np.asarray([float(row["deep_value"]) for row in rows], dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1.0e-9] = 1.0
    normalized = (matrix - means) / scales
    design = np.column_stack((np.ones(len(rows)), normalized))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.ones(len(rows))
    coefficients = np.zeros(design.shape[1])
    iterations = 1 if loss == "least_squares" else 8
    for _ in range(iterations):
        weighted = design * weights[:, None]
        coefficients = np.linalg.solve(
            design.T @ weighted + penalty, weighted.T @ target
        )
        if loss != "least_squares":
            residuals = np.abs(target - design @ coefficients)
            delta = float(loss.removeprefix("huber"))
            weights = np.minimum(1.0, delta / np.maximum(residuals, 1.0e-9))
    raw = coefficients[1:] / scales
    intercept = float(coefficients[0] - np.dot(raw, means))
    if raw[0] <= 1.0 / 256.0:
        # Preserve monotonicity in S.  Refit D-S on the remaining features.
        raw[0] = 1.0
        if len(names) == 1:
            intercept = float(np.mean(target - matrix[:, 0]))
        else:
            rest = matrix[:, 1:]
            rest_means = rest.mean(axis=0)
            rest_scales = rest.std(axis=0)
            rest_scales[rest_scales < 1.0e-9] = 1.0
            rest_design = np.column_stack((np.ones(len(rows)), (rest - rest_means) / rest_scales))
            rest_penalty = np.eye(rest_design.shape[1]) * ridge
            rest_penalty[0, 0] = 0.0
            rest_coefficients = np.linalg.solve(
                rest_design.T @ rest_design + rest_penalty,
                rest_design.T @ (target - matrix[:, 0]),
            )
            raw[1:] = rest_coefficients[1:] / rest_scales
            intercept = float(rest_coefficients[0] - np.dot(raw[1:], rest_means))
    return {"feature_names": names, "coefficients": raw.tolist(), "intercept": intercept}


def residual_bucket(row: dict[str, Any], kind: str) -> str:
    if row["direction"] == "high":
        margin = int(row["d0_value"]) - int(row["beta"])
    else:
        margin = int(row["alpha"]) - int(row["d0_value"])
    margin_part = 0 if margin < 4 else 1 if margin < 8 else 2 if margin < 16 else 3
    legal = int(row["legal_count"])
    legal_part = 0 if legal <= 3 else 1 if legal <= 6 else 2
    if kind == "margin":
        return str(margin_part)
    if kind == "legal":
        return str(legal_part)
    if kind == "marginlegal":
        return f"{margin_part}|{legal_part}"
    raise ValueError(kind)


def fit_local_quantiles(
    rows: list[dict[str, Any]],
    values: list[float],
    global_quantile: float,
    tail: float,
    direction: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    kind = str(spec.get("quantile_bucket", "none"))
    if kind == "none":
        return {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values):
        grouped[residual_bucket(row, kind)].append(value)
    result = {}
    for key, selected in grouped.items():
        if len(selected) < int(spec["min_bucket"]):
            continue
        local = conservative_quantile(selected, tail, direction)
        weight = len(selected) / (len(selected) + float(spec["shrinkage"]))
        result[key] = {
            "count": len(selected),
            "quantile": weight * local + (1.0 - weight) * global_quantile,
        }
    return result


def linear_prediction(row: dict[str, Any], group: dict[str, Any]) -> float:
    return float(group["intercept"]) + sum(
        float(coefficient) * feature_value(row, name)
        for name, coefficient in zip(group["feature_names"], group["coefficients"])
    )


def fit_model(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["mpc_level"]), str(row["direction"]))].append(row)
    result = {"format": 2, "spec": spec, "groups": {}}
    for (level, direction), group_rows in sorted(grouped.items()):
        tail = ((1.0 - SELECTIVITY_PERCENT[level] / 100.0) / 2.0) * float(spec["tail_scale"])
        base = str(spec["base"])
        group: dict[str, Any] = {"count": len(group_rows), "tail": tail}
        if base == "emp_error":
            values = [float(row["deep_value"] - row["shallow_value"]) for row in group_rows]
            group["quantile"] = conservative_quantile(values, tail, direction)
        elif base in ("emp_z", "emp_z_depth2", "emp_z_depth"):
            values = [
                float(row["deep_value"] - row["shallow_value"]) / sigma_end(row)
                for row in group_rows
            ]
            global_quantile = conservative_quantile(values, tail, direction)
            group["quantile"] = global_quantile
            local_quantiles = fit_local_quantiles(
                group_rows, values, global_quantile, tail, direction, spec
            )
            if local_quantiles:
                group["quantile_buckets"] = local_quantiles
            if base != "emp_z":
                bucket_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in group_rows:
                    width = 2 if base == "emp_z_depth2" else 1
                    bucket_rows[int(row["deep_depth"]) // width].append(row)
                group["buckets"] = {}
                for bucket, selected in bucket_rows.items():
                    if len(selected) < int(spec["min_bucket"]):
                        continue
                    local = conservative_quantile(
                        [float(row["deep_value"] - row["shallow_value"]) / sigma_end(row) for row in selected],
                        tail,
                        direction,
                    )
                    weight = len(selected) / (len(selected) + float(spec["shrinkage"]))
                    group["buckets"][str(bucket)] = {
                        "count": len(selected),
                        "quantile": weight * local + (1.0 - weight) * global_quantile,
                    }
        else:
            names = feature_names(base)
            group.update(
                fit_positive_ridge(
                    group_rows, names, float(spec["ridge"]), str(spec["loss"])
                )
            )
            residuals = [float(row["deep_value"]) - linear_prediction(row, group) for row in group_rows]
            group["quantile"] = conservative_quantile(residuals, tail, direction)
            local_quantiles = fit_local_quantiles(
                group_rows, residuals, float(group["quantile"]), tail, direction, spec
            )
            if local_quantiles:
                group["quantile_buckets"] = local_quantiles
        result["groups"][f"{level}|{direction}"] = group
    return result


def learned_threshold_float(row: dict[str, Any], model: dict[str, Any]) -> float:
    spec = model["spec"]
    base = str(spec["base"])
    group = model["groups"][f"{int(row['mpc_level'])}|{row['direction']}"]
    q = float(group["quantile"])
    quantile_kind = str(spec.get("quantile_bucket", "none"))
    if quantile_kind != "none":
        selected = group.get("quantile_buckets", {}).get(
            residual_bucket(row, quantile_kind)
        )
        if selected is not None:
            q = float(selected["quantile"])
    if base == "emp_error":
        value = boundary(row) - q
    elif base in ("emp_z", "emp_z_depth2", "emp_z_depth"):
        if base != "emp_z":
            width = 2 if base == "emp_z_depth2" else 1
            bucket = group.get("buckets", {}).get(str(int(row["deep_depth"]) // width))
            if bucket is not None:
                q = float(bucket["quantile"])
        value = boundary(row) - q * sigma_end(row)
    else:
        names = group["feature_names"]
        coefficients = group["coefficients"]
        slope = float(coefficients[0])
        remainder = float(group["intercept"]) + q
        for name, coefficient in zip(names[1:], coefficients[1:]):
            remainder += float(coefficient) * feature_value(row, name)
        value = (boundary(row) - remainder) / slope
    return value


def model_threshold(row: dict[str, Any], model: dict[str, Any]) -> int:
    learned = learned_threshold_float(row, model)
    legacy = float(legacy_threshold(row))
    blend = float(model["spec"]["blend"])
    blended = (1.0 - blend) * legacy + blend * learned
    return math.ceil(blended) if row["direction"] == "high" else math.floor(blended)


def row_result(row: dict[str, Any], threshold: int) -> tuple[bool, bool, int, int]:
    shallow = int(row["shallow_value"])
    deep = int(row["deep_value"])
    edge = boundary(row)
    if row["direction"] == "high":
        cut = shallow >= threshold
        correct = deep >= edge
        error = max(0, edge - deep)
        margin = shallow - threshold
    else:
        cut = shallow <= threshold
        correct = deep <= edge
        error = max(0, deep - edge)
        margin = threshold - shallow
    return cut, correct, error, margin


def evaluate(rows: list[dict[str, Any]], model: dict[str, Any] | None) -> dict[str, Any]:
    fields = ("cuts", "correct_cuts", "wrong_cuts", "wrong_2plus", "wrong_4plus", "wrong_error_sum")
    total: dict[str, Any] = {field: 0 for field in fields}
    total["count"] = len(rows)
    cuts_with_margin: list[tuple[int, bool, int]] = []
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        threshold = legacy_threshold(row) if model is None else model_threshold(row, model)
        cut, correct, error, margin = row_result(row, threshold)
        key = (row["direction"], int(row["mpc_level"]), int(row["deep_depth"]), int(row["shallow_depth"]))
        stats = groups.setdefault(key, {"direction": key[0], "mpc_level": key[1], "deep_depth": key[2], "shallow_depth": key[3], "count": 0, **{field: 0 for field in fields}})
        stats["count"] += 1
        if cut:
            for target in (total, stats):
                target["cuts"] += 1
                target["correct_cuts"] += int(correct)
                target["wrong_cuts"] += int(not correct)
                target["wrong_2plus"] += int((not correct) and error >= 2)
                target["wrong_4plus"] += int((not correct) and error >= 4)
                target["wrong_error_sum"] += error
            cuts_with_margin.append((margin, correct, error))
    for stats in [total, *groups.values()]:
        stats["cut_rate"] = stats["cuts"] / stats["count"] if stats["count"] else 0.0
        stats["wrong_cut_rate"] = stats["wrong_cuts"] / stats["cuts"] if stats["cuts"] else 0.0
    return {"overall": total, "groups": list(groups.values()), "cuts_with_margin": cuts_with_margin}


def equal_cut(legacy: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    count = min(len(legacy["cuts_with_margin"]), len(candidate["cuts_with_margin"]))
    def summarize(rows: list[tuple[int, bool, int]]) -> dict[str, int]:
        selected = sorted(rows, key=lambda item: item[0], reverse=True)[:count]
        return {
            "cuts": count,
            "wrong_cuts": sum(not correct for _, correct, _ in selected),
            "wrong_2plus": sum((not correct) and error >= 2 for _, correct, error in selected),
            "wrong_4plus": sum((not correct) and error >= 4 for _, correct, error in selected),
        }
    return {"cut_count": count, "legacy": summarize(legacy["cuts_with_margin"]), "candidate": summarize(candidate["cuts_with_margin"])}


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    bases = ("emp_error", "emp_z", "emp_z_depth2", "emp_z_depth", "ols_s", "ols_s_sigma", "ols_s_d0", "ols_s_d0_sigma", "ols_s_d0_sigma2", "ols_s_d0_deep", "ols_s_d0_deep_legal", "ols_s_d0_sigma_legal")
    for base in bases:
        ridges = (0.0, 1.0, 10.0) if base.startswith("ols_") else (0.0,)
        shrinkages = (16.0, 64.0) if base in ("emp_z_depth2", "emp_z_depth") else (0.0,)
        for tail_scale in (0.25, 0.5, 0.75, 1.0):
            for blend in (0.125, 0.25, 0.5, 0.75, 1.0):
                for ridge in ridges:
                    for shrinkage in shrinkages:
                        specs.append({"base": base, "tail_scale": tail_scale, "blend": blend, "ridge": ridge, "shrinkage": shrinkage, "min_bucket": 24, "loss": "least_squares", "quantile_bucket": "none"})
    for base in ("emp_z", "ols_s_d0_sigma", "ols_s_d0_sigma_legal"):
        losses = ("least_squares",) if base == "emp_z" else ("least_squares", "huber2", "huber4")
        for loss in losses:
            for quantile_bucket in ("margin", "legal", "marginlegal"):
                for tail_scale in (0.5, 0.75, 1.0):
                    for blend in (0.125, 0.25, 0.5):
                        for shrinkage in (32.0, 64.0):
                            specs.append({"base": base, "tail_scale": tail_scale, "blend": blend, "ridge": 1.0 if base.startswith("ols_") else 0.0, "shrinkage": shrinkage, "min_bucket": 32, "loss": loss, "quantile_bucket": quantile_bucket})
    return specs


def spec_name(spec: dict[str, Any]) -> str:
    return f"{spec['base']}_{spec['loss']}_q{spec['quantile_bucket']}_tail{spec['tail_scale']}_blend{spec['blend']}_ridge{spec['ridge']}_shrink{spec['shrinkage']}"


def add_totals(target: dict[str, int], source: dict[str, Any]) -> None:
    for key in ("count", "cuts", "correct_cuts", "wrong_cuts", "wrong_2plus", "wrong_4plus", "wrong_error_sum"):
        target[key] = target.get(key, 0) + int(source[key])


def bootstrap_selected(
    datasets: dict[str, list[dict[str, Any]]],
    labels: list[str],
    spec: dict[str, Any],
    iterations: int = 5000,
) -> dict[str, Any]:
    fields = ("cuts", "correct_cuts", "wrong_cuts", "wrong_2plus", "wrong_4plus")
    by_month: dict[str, list[dict[str, int]]] = {}
    raw = []
    for held_out in labels:
        train = [row for label in labels if label != held_out for row in datasets[label]]
        model = fit_model(train, spec)
        roots: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in datasets[held_out]:
            roots[str(row["root_id"])].append(row)
        deltas = []
        for root_id, rows in roots.items():
            old = evaluate(rows, None)["overall"]
            new = evaluate(rows, model)["overall"]
            delta = {field: int(new[field]) - int(old[field]) for field in fields}
            deltas.append(delta)
            raw.append({"month": held_out, "root_id": root_id, **delta})
        by_month[held_out] = deltas
    generator = random.Random(20260824)
    draws: dict[str, list[int]] = {field: [] for field in fields}
    for _ in range(iterations):
        totals = {field: 0 for field in fields}
        for label in labels:
            roots = by_month[label]
            for _ in roots:
                selected = roots[generator.randrange(len(roots))]
                for field in fields:
                    totals[field] += selected[field]
        for field in fields:
            draws[field].append(totals[field])
    def percentile(values: list[int], probability: float) -> int:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))]
    return {
        "seed": 20260824,
        "iterations": iterations,
        "root_count": sum(len(rows) for rows in by_month.values()),
        "delta": {
            field: {
                "median": percentile(values, 0.5),
                "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
                "probability_nonpositive": sum(value <= 0 for value in values) / len(values),
            }
            for field, values in draws.items()
        },
        "per_root": raw,
    }


def extrapolation_audit(model: dict[str, Any]) -> dict[str, Any]:
    deltas = []
    by_depth: dict[int, list[int]] = defaultdict(list)
    for deep in range(38, 45):
        shallow = ((deep * 2 // 5) & ~1) + (deep & 1)
        for level in range(3):
            for direction in ("high", "low"):
                row = {
                    "deep_depth": deep,
                    "shallow_depth": shallow,
                    "n_discs": 64 - deep,
                    "mpc_level": level,
                    "direction": direction,
                    "alpha": -1,
                    "beta": 1,
                    "shallow_value": 0,
                    "deep_value": 0,
                }
                error = abs(legacy_threshold(row) - boundary(row))
                for margin in (max(1, error - 3), error, error + 4, error + 8):
                    for legal in (1, 4, 8):
                        selected = dict(row)
                        selected["d0_value"] = boundary(selected) + (
                            margin if direction == "high" else -margin
                        )
                        selected["legal_count"] = legal
                        delta = model_threshold(selected, model) - legacy_threshold(selected)
                        deltas.append(delta)
                        by_depth[deep].append(delta)
    return {
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
        "maximum_absolute_delta": max(abs(value) for value in deltas),
        "by_depth": {
            str(deep): {"minimum": min(values), "maximum": max(values)}
            for deep, values in by_depth.items()
        },
    }


def screen(inputs: list[str], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    datasets: dict[str, list[dict[str, Any]]] = {}
    input_meta = []
    for item in inputs:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        datasets[label] = load_jsonl(path)
        input_meta.append({"label": label, "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(datasets[label]), "roots": len({row['root_id'] for row in datasets[label]})})
    labels = sorted(datasets)
    all_rows = [row for label in labels for row in datasets[label]]
    legacy_by_fold = {label: evaluate(datasets[label], None) for label in labels}
    legacy_total: dict[str, int] = {}
    for label in labels:
        add_totals(legacy_total, legacy_by_fold[label]["overall"])
    summaries = []
    specs = candidate_specs()
    for index, spec in enumerate(specs):
        aggregate: dict[str, int] = {}
        equal_aggregate = {"legacy": defaultdict(int), "candidate": defaultdict(int)}
        folds = []
        for held_out in labels:
            train = [row for label in labels if label != held_out for row in datasets[label]]
            model = fit_model(train, spec)
            result = evaluate(datasets[held_out], model)
            comparison = equal_cut(legacy_by_fold[held_out], result)
            add_totals(aggregate, result["overall"])
            for side in ("legacy", "candidate"):
                for key, value in comparison[side].items():
                    equal_aggregate[side][key] += int(value)
            folds.append({"held_out": held_out, "legacy_overall": legacy_by_fold[held_out]["overall"], "overall": result["overall"], "equal_cut": comparison})
        aggregate["cut_rate"] = aggregate["cuts"] / aggregate["count"]
        aggregate["wrong_cut_rate"] = aggregate["wrong_cuts"] / aggregate["cuts"] if aggregate["cuts"] else 0.0
        stable_by_fold = all(
            fold["overall"]["wrong_4plus"] <= fold["legacy_overall"]["wrong_4plus"]
            and fold["overall"]["wrong_2plus"] <= fold["legacy_overall"]["wrong_2plus"]
            and fold["overall"]["correct_cuts"]
            >= math.floor(0.995 * fold["legacy_overall"]["correct_cuts"])
            for fold in folds
        )
        eligible = (
            aggregate["wrong_4plus"] <= legacy_total["wrong_4plus"]
            and aggregate["wrong_2plus"] <= legacy_total["wrong_2plus"]
            and aggregate["wrong_cuts"] < legacy_total["wrong_cuts"]
            and equal_aggregate["candidate"]["wrong_4plus"] <= equal_aggregate["legacy"]["wrong_4plus"]
            and equal_aggregate["candidate"]["wrong_2plus"] <= equal_aggregate["legacy"]["wrong_2plus"]
            and equal_aggregate["candidate"]["wrong_cuts"] <= equal_aggregate["legacy"]["wrong_cuts"]
            and aggregate["correct_cuts"] >= math.floor(0.995 * legacy_total["correct_cuts"])
            and stable_by_fold
        )
        summaries.append({"name": spec_name(spec), "spec": spec, "eligible": eligible, "stable_by_fold": stable_by_fold, "overall": aggregate, "equal_cut": {side: dict(values) for side, values in equal_aggregate.items()}, "folds": folds})
        if (index + 1) % 100 == 0:
            print(f"screened {index + 1}/{len(specs)}", flush=True)
    extrapolating_bases = {
        "emp_z",
        "emp_z_depth2",
        "emp_z_depth",
        "ols_s_sigma",
        "ols_s_d0_sigma",
        "ols_s_d0_sigma2",
        "ols_s_d0_sigma_legal",
    }
    for item in summaries:
        item["extrapolation_safe"] = False
        if item["eligible"] and item["spec"]["base"] in extrapolating_bases:
            audit = extrapolation_audit(fit_model(all_rows, item["spec"]))
            item["extrapolation"] = audit
            item["extrapolation_safe"] = audit["maximum_absolute_delta"] <= 2
    eligible = [item for item in summaries if item["eligible"] and item["extrapolation_safe"]]
    pool = eligible if eligible else summaries
    selected = min(pool, key=lambda item: (item["overall"]["wrong_4plus"], item["overall"]["wrong_2plus"], item["overall"]["wrong_cuts"], -item["overall"]["correct_cuts"], -item["overall"]["cuts"]))
    final_model = fit_model(all_rows, selected["spec"])
    final_model["selection"] = {"method": "leave-one-root-set-out", "selected_name": selected["name"], "eligible": selected["eligible"], "extrapolation_safe": selected.get("extrapolation_safe", False), "development_rows": len(all_rows), "development_months": labels, "external_test_inspected": False}
    (output / "selected_model.json").write_text(json.dumps(final_model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "all_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for item in summaries:
            file.write(json.dumps(item, sort_keys=True) + "\n")
    report = {"inputs": input_meta, "candidate_count": len(summaries), "eligible_extrapolation_safe_count": len(eligible), "legacy": legacy_total, "selected": selected, "bootstrap": bootstrap_selected(datasets, labels, selected["spec"]), "top_candidates": sorted(summaries, key=lambda item: (not (item["eligible"] and item.get("extrapolation_safe", False)), item["overall"]["wrong_4plus"], item["overall"]["wrong_2plus"], item["overall"]["wrong_cuts"], -item["overall"]["correct_cuts"]))[:30]}
    (output / "screen_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"legacy": legacy_total, "selected": selected["name"], "selected_overall": selected["overall"], "eligible": selected["eligible"]}, sort_keys=True))


def evaluate_external(test: Path, model_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    rows = load_jsonl(test)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    legacy = evaluate(rows, None)
    candidate = evaluate(rows, model)
    report = {
        "test": {"path": str(test.resolve()), "sha256": hashlib.sha256(test.read_bytes()).hexdigest(), "rows": len(rows), "roots": len({row['root_id'] for row in rows})},
        "model": {"path": str(model_path.resolve()), "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "selection": model.get("selection", {}), "spec": model["spec"]},
        "legacy": {key: value for key, value in legacy.items() if key != "cuts_with_margin"},
        "candidate": {key: value for key, value in candidate.items() if key != "cuts_with_margin"},
        "equal_cut": equal_cut(legacy, candidate),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"legacy": legacy["overall"], "candidate": candidate["overall"], "equal_cut": report["equal_cut"]}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument("--input", action="append", required=True, help="LABEL=JSONL")
    screen_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--test", type=Path, required=True)
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "screen":
        screen(args.input, args.output)
    else:
        evaluate_external(args.test, args.model, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
