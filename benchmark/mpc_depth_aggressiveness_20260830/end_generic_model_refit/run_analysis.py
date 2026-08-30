#!/usr/bin/env python3
"""Refit and evaluate the legacy generic endgame MPC sigma polynomial.

The script intentionally uses only the Python standard library so that the
analysis can be rerun with the repository's bundled Python installation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CURRENT_COEFFICIENTS = (
    -1.3182333120273682,
    -6.99290557735024,
    -0.05280654146244756,
    0.48284187178125065,
    5.289589936037036,
    11.940601436361513,
)
END_Z = (1.13, 1.55, 1.81, 2.32, 2.57, 3.29)
END_PERCENT = (
    74.15237755199644,
    87.88584839958820,
    92.97042128319224,
    97.96591226625605,
    98.98301485020178,
    99.89981261724286,
)

HISTORICAL_DEV = {
    "dev743": Path("benchmark/end_mpc_enriched_dev_743/samples.jsonl"),
    "dev202607": Path("benchmark/end_mpc_enriched_dev_202607/samples.jsonl"),
    "dev202608a": Path("benchmark/end_mpc_enriched_dev_202608a/samples.jsonl"),
    "dev202608b": Path("benchmark/end_mpc_enriched_dev_202608b/samples.jsonl"),
}
HISTORICAL_HOLDOUT = Path(
    "benchmark/end_mpc_enriched_holdout_202606/samples.jsonl"
)
NEW_HIGH = Path(
    "benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit/"
    "high_depth_samples/samples.jsonl"
)
NEW_DEEP = Path(
    "benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit/"
    "deep23_26_samples/samples.jsonl"
)
KNOWN_DEEP30 = Path(
    "benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit/"
    "deep30_known_samples.jsonl"
)
TRACE_USAGE = Path(
    "benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit/"
    "trace_usage/trace_depth_usage.json"
)
SEARCH_LOGS = (
    Path("bin/ggs/log/2026-08-23-15=47=18_search.log"),
    Path("bin/ggs/log/2026-08-25-15=52=06_search.log"),
    Path("bin/ggs/log/2026-08-26-14=37=39_search.log"),
    Path("bin/ggs/log/2026-08-27-23=02=28_search.log"),
    Path("bin/ggs/log/2026-08-29-22=38=57_search.log"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generic_shallow_depth(deep: int) -> int:
    return ((deep * 2 // 5) & ~1) + (deep & 1)


def root_key(row: dict[str, Any]) -> str:
    return str(row.get("root_id", row.get("root_board", row["board"])))


def root_board_key(row: dict[str, Any]) -> str:
    return str(row.get("root_board", row["board"]))


def domain_for(deep: int) -> str:
    if deep <= 18:
        return "deep10_18"
    if deep <= 22:
        return "deep19_22"
    return "deep23_26"


def observation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        root_key(row), str(row["board"]), int(row["deep_depth"]),
        int(row["shallow_depth"]),
    )


def deduplicate_observations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        deep = int(row["deep_depth"])
        shallow = int(row["shallow_depth"])
        if shallow != generic_shallow_depth(deep) or shallow <= 0:
            continue
        key = observation_key(row)
        previous = selected.get(key)
        if previous is not None:
            for field in ("shallow_value", "deep_value", "n_discs"):
                if int(previous[field]) != int(row[field]):
                    raise ValueError(f"inconsistent {field}: {key}")
        else:
            copied = dict(row)
            copied["domain"] = domain_for(deep)
            selected[key] = copied
    return list(selected.values())


def locked_holdout(row: dict[str, Any]) -> bool:
    token = f"generic-end-holdout-v1|{row['domain']}|{root_key(row)}"
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 5 == 0


def cv_fold(row: dict[str, Any]) -> int:
    token = f"generic-end-cv-v1|{row['domain']}|{root_key(row)}"
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 5


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-14:
            raise ValueError("singular normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for j in range(column, size + 1):
            augmented[column][j] /= divisor
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier == 0.0:
                continue
            for j in range(column, size + 1):
                augmented[row][j] -= multiplier * augmented[column][j]
    return [augmented[i][size] for i in range(size)]


def observation_weights(
    rows: list[dict[str, Any]], mode: str
) -> dict[tuple[Any, ...], float]:
    root_counts = Counter((str(row["domain"]), root_key(row)) for row in rows)
    domain_roots: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        domain_roots[str(row["domain"])].add(root_key(row))
    result = {}
    for row in rows:
        domain = str(row["domain"])
        root = root_key(row)
        weight = 1.0 / root_counts[(domain, root)]
        if mode == "domain_equal":
            weight /= len(domain_roots[domain]) * len(domain_roots)
        elif mode == "root_equal":
            weight /= sum(len(roots) for roots in domain_roots.values())
        else:
            raise ValueError(mode)
        result[observation_key(row)] = weight
    return result


def sigma_current(n_discs: int, shallow: int) -> float:
    a, b, c, d, e, f = CURRENT_COEFFICIENTS
    u = a * n_discs / 64.0 + b * shallow / 60.0
    return c * u * u * u + d * u * u + e * u + f


def normalize_coefficients_a_minus_one(
    coefficients: Iterable[float],
) -> list[float]:
    a, b, c, d, e, f = coefficients
    scale = -1.0 / a
    return [
        -1.0,
        scale * b,
        c / (scale ** 3),
        d / (scale ** 2),
        e / scale,
        f,
    ]


def sigma_model(model: dict[str, Any], n_discs: int, shallow: int) -> float:
    if model["name"] == "current":
        return max(0.5, sigma_current(n_discs, shallow))
    a, b, c, d, e, f = model["coefficients"]
    u = a * n_discs / 64.0 + b * shallow / 60.0
    return max(0.5, c * u * u * u + d * u * u + e * u + f)


def bucket_targets(
    rows: list[dict[str, Any]], mode: str
) -> list[tuple[int, int, float, float]]:
    weights = observation_weights(rows, mode)
    sums: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        key = (int(row["n_discs"]), int(row["shallow_depth"]))
        weight = weights[observation_key(row)]
        residual = float(row["deep_value"]) - float(row["shallow_value"])
        sums[key][0] += weight * residual * residual
        sums[key][1] += weight
    return [
        (n_discs, shallow, math.sqrt(total / weight), weight)
        for (n_discs, shallow), (total, weight) in sorted(sums.items())
    ]


def fit_cubic_for_ratio(
    buckets: list[tuple[int, int, float, float]], ratio: float,
    prior_strength: float,
) -> tuple[list[float], float]:
    size = 4
    normal = [[0.0] * size for _ in range(size)]
    target = [0.0] * size
    for n_discs, shallow, sigma, weight in buckets:
        u = n_discs / 64.0 + ratio * shallow / 60.0
        design = [u * u * u, u * u, u, 1.0]
        for i in range(size):
            target[i] += weight * design[i] * sigma
            for j in range(size):
                normal[i][j] += weight * design[i] * design[j]
    # The old fit was made from a substantially broader private corpus.  A
    # tunable prediction-space prior lets root-CV decide how far the smaller
    # exact-labelled corpus can safely move away from that established shape.
    if prior_strength > 0.0:
        prior_weight = prior_strength / 35.0
        for deep in range(10, 45):
            n_discs = 64 - deep
            shallow = generic_shallow_depth(deep)
            sigma = sigma_current(n_discs, shallow)
            u = n_discs / 64.0 + ratio * shallow / 60.0
            design = [u * u * u, u * u, u, 1.0]
            for i in range(size):
                target[i] += prior_weight * design[i] * sigma
                for j in range(size):
                    normal[i][j] += prior_weight * design[i] * design[j]
    trace = sum(normal[index][index] for index in range(size))
    for index in range(size):
        normal[index][index] += max(1.0e-14, trace * 1.0e-12)
    coefficients = solve(normal, target)
    objective = 0.0
    for n_discs, shallow, sigma, weight in buckets:
        u = n_discs / 64.0 + ratio * shallow / 60.0
        predicted = (
            coefficients[0] * u * u * u + coefficients[1] * u * u +
            coefficients[2] * u + coefficients[3]
        )
        objective += weight * (predicted - sigma) ** 2
    if prior_strength > 0.0:
        prior_weight = prior_strength / 35.0
        for deep in range(10, 45):
            n_discs = 64 - deep
            shallow = generic_shallow_depth(deep)
            target_sigma = sigma_current(n_discs, shallow)
            u = n_discs / 64.0 + ratio * shallow / 60.0
            predicted = (
                coefficients[0] * u * u * u + coefficients[1] * u * u +
                coefficients[2] * u + coefficients[3]
            )
            objective += prior_weight * (predicted - target_sigma) ** 2
    # Prevent a numerically attractive fit from producing unusable negative
    # sigmas on the complete depth range where generic endgame MPC can run.
    for deep in range(10, 45):
        for shallow in (generic_shallow_depth(deep),):
            u = (64 - deep) / 64.0 + ratio * shallow / 60.0
            predicted = (
                coefficients[0] * u * u * u + coefficients[1] * u * u +
                coefficients[2] * u + coefficients[3]
            )
            if predicted < 0.5:
                objective += 1.0e4 * (0.5 - predicted) ** 2
    return coefficients, objective


def fit_model(
    rows: list[dict[str, Any]], mode: str, prior_strength: float = 0.0
) -> dict[str, Any]:
    buckets = bucket_targets(rows, mode)
    best: tuple[float, float, list[float]] | None = None
    center = 0.0
    radius = 30.0
    for _ in range(6):
        for index in range(241):
            ratio = center - radius + 2.0 * radius * index / 240.0
            coefficients, objective = fit_cubic_for_ratio(
                buckets, ratio, prior_strength
            )
            candidate = (objective, ratio, coefficients)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        center = best[1]
        radius /= 10.0
    assert best is not None
    objective, ratio, coefficients = best
    return {
        "name": (
            f"refit_{mode}"
            if prior_strength == 0.0
            else f"refit_{mode}_prior{prior_strength:g}"
        ),
        "weighting": mode,
        "prior_strength": prior_strength,
        # a=-1 fixes the exact scale non-identifiability between (a,b) and
        # the outer cubic coefficients.  fit_cubic_for_ratio uses v=x+r*y;
        # the emitted engine coefficients use u=-v.
        "coefficients": [
            -1.0, -ratio, -coefficients[0], coefficients[1],
            -coefficients[2], coefficients[3],
        ],
        "fit_objective": objective,
        "bucket_count": len(buckets),
    }


def weighted_metrics(
    rows: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, float]:
    weights = observation_weights(rows, "root_equal")
    total = sum(weights.values())
    residual_sq = residual_abs = nll = z_sq = 0.0
    for row in rows:
        weight = weights[observation_key(row)]
        residual = float(row["deep_value"]) - float(row["shallow_value"])
        sigma = sigma_model(
            model, int(row["n_discs"]), int(row["shallow_depth"])
        )
        residual_sq += weight * residual * residual
        residual_abs += weight * abs(residual)
        z_sq += weight * (residual / sigma) ** 2
        nll += weight * (math.log(sigma) + 0.5 * (residual / sigma) ** 2)

    buckets = bucket_targets(rows, "root_equal")
    bucket_weight = sum(row[3] for row in buckets)
    sigma_sq = sigma_abs = 0.0
    for n_discs, shallow, target, weight in buckets:
        predicted = sigma_model(model, n_discs, shallow)
        sigma_sq += weight * (predicted - target) ** 2
        sigma_abs += weight * abs(predicted - target)
    return {
        "roots": float(len({root_key(row) for row in rows})),
        "observations": float(len(rows)),
        "residual_rmse": math.sqrt(residual_sq / total),
        "residual_mae": residual_abs / total,
        "sigma_rmse": math.sqrt(sigma_sq / bucket_weight),
        "sigma_mae": sigma_abs / bucket_weight,
        "standardized_rms": math.sqrt(z_sq / total),
        "gaussian_nll": nll / total,
    }


def cross_validate(
    rows: list[dict[str, Any]], specs: list[tuple[str, float]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        train = [row for row in rows if cv_fold(row) != fold]
        test = [row for row in rows if cv_fold(row) == fold]
        models = [{"name": "current", "coefficients": list(CURRENT_COEFFICIENTS)}]
        models.extend(fit_model(train, mode, prior) for mode, prior in specs)
        for model in models:
            fold_rows.append({
                "fold": fold,
                "model": model["name"],
                **weighted_metrics(test, model),
            })
    aggregate: list[dict[str, Any]] = []
    model_names = ["current"] + [
        fit_model(rows, mode, prior)["name"] for mode, prior in specs
    ]
    for name in model_names:
        selected = [row for row in fold_rows if row["model"] == name]
        weight = sum(float(row["roots"]) for row in selected)
        result: dict[str, Any] = {
            "model": name,
            "folds": len(selected),
            "roots": int(weight),
            "observations": int(sum(float(row["observations"]) for row in selected)),
        }
        for metric in (
            "residual_rmse", "residual_mae", "sigma_rmse", "sigma_mae",
            "standardized_rms", "gaussian_nll",
        ):
            result[metric] = sum(
                float(row[metric]) * float(row["roots"]) for row in selected
            ) / weight
        aggregate.append(result)
    return fold_rows, aggregate


def context_key(row: dict[str, Any], include_level: bool) -> tuple[Any, ...]:
    base: tuple[Any, ...] = (
        root_key(row), str(row["board"]), int(row["deep_depth"]),
        int(row["alpha"]), int(row["beta"]), str(row["direction"]),
    )
    return base + ((int(row["mpc_level"]),) if include_level else ())


def select_contexts(
    rows: Iterable[dict[str, Any]], roots: set[str] | None,
    include_level: bool,
) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if "alpha" not in row or "beta" not in row:
            continue
        deep = int(row["deep_depth"])
        if int(row["shallow_depth"]) != generic_shallow_depth(deep):
            continue
        if roots is not None and root_key(row) not in roots:
            continue
        selected.setdefault(context_key(row, include_level), row)
    return list(selected.values())


def simulate_cuts(
    contexts: list[dict[str, Any]], model: dict[str, Any], level: int,
    use_recorded_level: bool,
) -> dict[str, Any]:
    stats = Counter()
    for row in contexts:
        if use_recorded_level and int(row["mpc_level"]) != level:
            continue
        deep_nodes = int(row["deep_nodes"])
        sigma = sigma_model(
            model, int(row["n_discs"]), int(row["shallow_depth"])
        )
        error = math.ceil(END_Z[level] * sigma - 1.0e-12)
        gate_error = max(1, error - 3)
        direction = str(row["direction"])
        if direction == "high":
            boundary = int(row["beta"])
            gate = int(row["d0_value"]) >= boundary + gate_error
            cut = gate and int(row["shallow_value"]) >= boundary + error
            correct = int(row["deep_value"]) >= boundary
            wrong_margin = max(0, boundary - int(row["deep_value"]))
        else:
            boundary = int(row["alpha"])
            gate = int(row["d0_value"]) <= boundary - gate_error
            cut = gate and int(row["shallow_value"]) <= boundary - error
            correct = int(row["deep_value"]) <= boundary
            wrong_margin = max(0, int(row["deep_value"]) - boundary)
        probe_nodes = int(row["shallow_nodes"]) if gate else 0
        stats["contexts"] += 1
        stats["probes"] += int(gate)
        stats["cuts"] += int(cut)
        stats["wrong_cuts"] += int(cut and not correct)
        stats["wrong_2plus"] += int(cut and not correct and wrong_margin >= 2)
        stats["wrong_4plus"] += int(cut and not correct and wrong_margin >= 4)
        stats["baseline_nodes"] += deep_nodes
        stats["estimated_nodes"] += probe_nodes + (0 if cut else deep_nodes)
    baseline = stats["baseline_nodes"]
    return {
        "model": model["name"],
        "level": level,
        "selectivity_percent": END_PERCENT[level],
        "contexts": stats["contexts"],
        "probes": stats["probes"],
        "cuts": stats["cuts"],
        "wrong_cuts": stats["wrong_cuts"],
        "wrong_2plus": stats["wrong_2plus"],
        "wrong_4plus": stats["wrong_4plus"],
        "estimated_node_ratio": stats["estimated_nodes"] / baseline if baseline else 0.0,
    }


END_LINE = re.compile(r"end depth (\d+)@([0-9.]+)%\s+([^\r\n]*)")


def analyze_logs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in SEARCH_LOGS:
        for match in END_LINE.finditer(path.read_text(encoding="utf-8", errors="replace")):
            depth = int(match.group(1))
            percent = float(match.group(2))
            suffix = match.group(3)
            rows.append({
                "log": path.name,
                "depth": depth,
                "selectivity_percent": percent,
                "completed": int(suffix.startswith("value ")),
                "terminated": int("terminated" in suffix),
            })
    buckets = ((10, 18), (19, 22), (23, 26), (27, 30), (31, 34), (35, 38), (39, 64))
    summary = []
    for low, high in buckets:
        selected = [row for row in rows if low <= int(row["depth"]) <= high]
        summary.append({
            "depth_range": f"{low}-{high if high < 64 else '以上'}",
            "attempts": len(selected),
            "completed": sum(int(row["completed"]) for row in selected),
            "terminated": sum(int(row["terminated"]) for row in selected),
        })
    return rows, summary


def depth_calibration_rows(
    datasets: dict[str, list[dict[str, Any]]],
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, rows in datasets.items():
        for deep in sorted({int(row["deep_depth"]) for row in rows}):
            selected = [row for row in rows if int(row["deep_depth"]) == deep]
            weights = observation_weights(selected, "root_equal")
            total = sum(weights.values())
            rms = math.sqrt(sum(
                weights[observation_key(row)] *
                (float(row["deep_value"]) - float(row["shallow_value"])) ** 2
                for row in selected
            ) / total)
            for model in models:
                result.append({
                    "dataset": label,
                    "deep_depth": deep,
                    "shallow_depth": generic_shallow_depth(deep),
                    "roots": len({root_key(row) for row in selected}),
                    "observations": len(selected),
                    "empirical_residual_rms": rms,
                    "model": model["name"],
                    "predicted_sigma": sigma_model(
                        model, 64 - deep, generic_shallow_depth(deep)
                    ),
                })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_audit(
    label: str, path: Path, raw: list[dict[str, Any]],
    observations: list[dict[str, Any]], role: str,
) -> dict[str, Any]:
    roots = {root_key(row) for row in observations}
    return {
        "dataset": label,
        "role": role,
        "path": str(path),
        "sha256": sha256_file(path),
        "raw_rows": len(raw),
        "unique_roots": len(roots),
        "unique_actual_shallow_observations": len(observations),
        "min_deep": min((int(row["deep_depth"]) for row in observations), default=""),
        "max_deep": max((int(row["deep_depth"]) for row in observations), default=""),
        "min_shallow": min((int(row["shallow_depth"]) for row in observations), default=""),
        "max_shallow": max((int(row["shallow_depth"]) for row in observations), default=""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "benchmark/mpc_depth_aggressiveness_20260830/"
            "end_generic_model_refit/results"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    historical_raw = {label: load_jsonl(path) for label, path in HISTORICAL_DEV.items()}
    historical_obs = {
        label: deduplicate_observations(rows)
        for label, rows in historical_raw.items()
    }
    june_raw = load_jsonl(HISTORICAL_HOLDOUT)
    june_obs = deduplicate_observations(june_raw)
    high_raw = load_jsonl(NEW_HIGH)
    high_obs = deduplicate_observations(high_raw)
    deep_raw = load_jsonl(NEW_DEEP)
    deep_obs = deduplicate_observations(deep_raw)
    deep30_raw = load_jsonl(KNOWN_DEEP30)
    deep30_obs = deduplicate_observations(deep30_raw)

    new_all_obs = high_obs + deep_obs
    new_train_obs = [row for row in new_all_obs if not locked_holdout(row)]
    new_holdout_obs = [row for row in new_all_obs if locked_holdout(row)]
    train_obs = [row for rows in historical_obs.values() for row in rows] + new_train_obs

    current = {"name": "current", "coefficients": list(CURRENT_COEFFICIENTS)}
    prior_grid = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    cv_specs = [
        (mode, prior)
        for mode in ("root_equal", "domain_equal")
        for prior in prior_grid
    ]
    fold_rows, cv_rows = cross_validate(train_obs, cv_specs)
    cv_by_name = {str(row["model"]): row for row in cv_rows}
    selected_regularized: dict[str, float] = {}
    for mode in ("root_equal", "domain_equal"):
        candidates = [
            (prior, cv_by_name[
                f"refit_{mode}" if prior == 0.0 else f"refit_{mode}_prior{prior:g}"
            ])
            for prior in prior_grid if prior > 0.0
        ]
        selected_regularized[mode] = min(
            candidates, key=lambda pair: float(pair[1]["gaussian_nll"])
        )[0]
    final_models = [current]
    for mode in ("root_equal", "domain_equal"):
        final_models.append(fit_model(train_obs, mode, 0.0))
        final_models.append(
            fit_model(train_obs, mode, selected_regularized[mode])
        )

    coefficient_rows = []
    for model in final_models:
        a, b, c, d, e, f = model["coefficients"]
        coefficient_rows.append({
            "model": model["name"],
            "representation": "original" if model["name"] == "current" else "a=-1",
            "a": a, "b": b, "c": c,
            "d": d, "e": e, "f": f,
        })
        if model["name"] == "current":
            na, nb, nc, nd, ne, nf = normalize_coefficients_a_minus_one(
                model["coefficients"]
            )
            coefficient_rows.append({
                "model": "current",
                "representation": "equivalent a=-1",
                "a": na, "b": nb, "c": nc,
                "d": nd, "e": ne, "f": nf,
            })

    holdout_rows: list[dict[str, Any]] = []
    holdouts = {
        "historical_202606_deep10_18": june_obs,
        "locked_new_deep19_26": new_holdout_obs,
        "known_exact_deep30": deep30_obs,
    }
    for label, rows in holdouts.items():
        for model in final_models:
            holdout_rows.append({
                "dataset": label, "model": model["name"],
                **weighted_metrics(rows, model),
            })

    high_holdout_roots = {root_key(row) for row in new_holdout_obs}
    high_contexts = select_contexts(
        high_raw + deep_raw, high_holdout_roots, include_level=True
    )
    june_contexts = select_contexts(june_raw, None, include_level=False)
    cut_rows: list[dict[str, Any]] = []
    for model in final_models:
        for level in range(3):
            cut_rows.append({
                "dataset": "locked_new_deep19_26",
                **simulate_cuts(high_contexts, model, level, True),
            })
        for level in range(3, 6):
            cut_rows.append({
                "dataset": "historical_202606_deep10_18_simulated_high_selectivity",
                **simulate_cuts(june_contexts, model, level, False),
            })

    audit_rows = []
    for label, path in HISTORICAL_DEV.items():
        audit_rows.append(dataset_audit(
            label, path, historical_raw[label], historical_obs[label], "development"
        ))
    audit_rows.extend((
        dataset_audit(
            "holdout202606", HISTORICAL_HOLDOUT, june_raw, june_obs,
            "historical independent holdout",
        ),
        dataset_audit(
            "new_deep19_22", NEW_HIGH, high_raw, high_obs,
            "new hash-split development/holdout",
        ),
        dataset_audit(
            "new_deep23_26", NEW_DEEP, deep_raw, deep_obs,
            "new hash-split development/holdout",
        ),
        dataset_audit(
            "known_exact_deep30", KNOWN_DEEP30, deep30_raw, deep30_obs,
            "external exact holdout",
        ),
    ))
    all_dataset_rows = {
        **historical_raw,
        "holdout202606": june_raw,
        "new_deep19_22": high_raw,
        "new_deep23_26": deep_raw,
        "known_exact_deep30": deep30_raw,
    }
    root_intersections = []
    dataset_labels = list(all_dataset_rows)
    for left_index, left in enumerate(dataset_labels):
        left_roots = {root_board_key(row) for row in all_dataset_rows[left]}
        for right in dataset_labels[left_index + 1:]:
            right_roots = {root_board_key(row) for row in all_dataset_rows[right]}
            root_intersections.append({
                "left": left,
                "right": right,
                "shared_root_boards": len(left_roots & right_roots),
            })

    log_rows, log_summary = analyze_logs()
    log_selectivity_summary = []
    for percent in sorted({float(row["selectivity_percent"]) for row in log_rows}):
        selected = [
            row for row in log_rows
            if float(row["selectivity_percent"]) == percent
        ]
        log_selectivity_summary.append({
            "selectivity_percent": percent,
            "attempts": len(selected),
            "completed": sum(int(row["completed"]) for row in selected),
            "terminated": sum(int(row["terminated"]) for row in selected),
            "mean_depth": sum(int(row["depth"]) for row in selected) / len(selected),
            "min_depth": min(int(row["depth"]) for row in selected),
            "max_depth": max(int(row["depth"]) for row in selected),
        })
    calibration_rows = depth_calibration_rows(holdouts, final_models)
    metadata = {
        "training_roots": len({(str(row["domain"]), root_key(row)) for row in train_obs}),
        "training_observations": len(train_obs),
        "locked_new_holdout_roots": len({root_key(row) for row in new_holdout_obs}),
        "locked_new_holdout_observations": len(new_holdout_obs),
        "holdout_rule": "sha256('generic-end-holdout-v1|domain|root_id') mod 5 == 0",
        "cv_rule": "sha256('generic-end-cv-v1|domain|root_id') mod 5",
        "trace_usage": json.loads(TRACE_USAGE.read_text(encoding="utf-8")),
        "search_logs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in SEARCH_LOGS
        ],
        "models": final_models,
        "prior_grid": list(prior_grid),
        "regularized_strength_selected_by_development_cv_gaussian_nll":
            selected_regularized,
    }

    write_csv(args.output / "dataset_audit.csv", audit_rows)
    write_csv(args.output / "root_intersections.csv", root_intersections)
    write_csv(args.output / "coefficients.csv", coefficient_rows)
    write_csv(args.output / "cv_folds.csv", fold_rows)
    write_csv(args.output / "cv_summary.csv", cv_rows)
    write_csv(args.output / "holdout_metrics.csv", holdout_rows)
    write_csv(args.output / "cut_simulation.csv", cut_rows)
    write_csv(args.output / "depth_calibration.csv", calibration_rows)
    write_csv(args.output / "log_end_iterations.csv", log_rows)
    write_csv(args.output / "log_depth_summary.csv", log_summary)
    write_csv(args.output / "log_selectivity_summary.csv", log_selectivity_summary)
    (args.output / "results.json").write_text(
        json.dumps({
            "metadata": metadata,
            "dataset_audit": audit_rows,
            "root_intersections": root_intersections,
            "coefficients": coefficient_rows,
            "cv_summary": cv_rows,
            "holdout_metrics": holdout_rows,
            "cut_simulation": cut_rows,
            "depth_calibration": calibration_rows,
            "log_depth_summary": log_summary,
            "log_selectivity_summary": log_selectivity_summary,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "training_observations": len(train_obs),
        "new_holdout_observations": len(new_holdout_obs),
        "models": coefficient_rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
