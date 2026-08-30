#!/usr/bin/env python3
"""中盤MPCの誤差モデルを、対局単位で分離して再学習・評価する。

標準偏差へ後付け倍率を掛ける処理は行わない。実行時と同じ

    x = a*n/64 + b*shallow/60 + c*deep/60
    sigma = d*x^3 + e*x^2 + f*x + g

の7係数を直接最適化する。出力は機械可読JSON/CSVとC++定数候補。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

LEGACY = np.array([
    0.8335834703936896,
    -4.71778909968251,
    1.1467905781538477,
    -0.5274699259330169,
    6.5091001393587335,
    3.9546352081550378,
    1.8719077939546169,
], dtype=np.float64)

# 現行の中盤用z値。標準偏差モデルの比較なので、この値は変更しない。
Z_LEVELS = {"74": 1.13, "88": 1.55, "93": 1.81}
OFFSETS = (-4, -2, 0, 2, 4)


@dataclass(frozen=True)
class Row:
    board: str
    root: str
    source: str
    n_discs: int
    deep_depth: int
    shallow_depth: int
    shallow_value: int
    deep_value: int
    error: int
    shallow_nodes: int
    deep_nodes: int


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metadata() -> tuple[dict[str, str], dict[str, set[str]]]:
    files = {
        "model_dev": ROOT / "benchmark/mid_mpc_model_dev_20260827.jsonl",
        "ggs_dev": ROOT / "benchmark/midgame_ggs_dev_20260827.jsonl",
        "ggs_holdout": ROOT / "benchmark/midgame_ggs_holdout_20260827.jsonl",
        "tuning_dev": ROOT / "benchmark/mid_tuning_dev_20260827.jsonl",
        "tuning_validation": ROOT / "benchmark/mid_tuning_validation_20260827.jsonl",
        "tuning_final": ROOT / "benchmark/mid_tuning_final_holdout_20260827.jsonl",
    }
    board_roots: dict[str, set[str]] = defaultdict(set)
    sets: dict[str, set[str]] = {}
    for name, path in files.items():
        rows = read_jsonl(path)
        sets[name] = {str(row["board"]) for row in rows}
        for row in rows:
            board_roots[str(row["board"])].add(str(row.get("game", row["board"])))
    # 同一盤面が複数対局に現れた場合、どちらか一方の対局名を選ぶとCVを跨ぐ。
    # 盤面を共有した対局をunionし、その連結成分全体を一つのrootとして扱う。
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    for roots in board_roots.values():
        ordered = sorted(roots)
        for root in ordered:
            parent.setdefault(root, root)
        for root in ordered[1:]:
            union(ordered[0], root)
    components: dict[str, list[str]] = defaultdict(list)
    for root in parent:
        components[find(root)].append(root)
    component_name = {
        member: "component:" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()
        for members in components.values()
        for member in members
    }
    return {
        board: component_name[next(iter(roots))]
        for board, roots in board_roots.items()
    }, sets


def tsv_rows(paths: list[Path], source: str, board_to_root: dict[str, str]) -> list[Row]:
    result: list[Row] = []
    for path in paths:
        if not path.exists():
            continue
        prefix = path.read_bytes()[:2]
        encoding = "utf-16" if prefix in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        with path.open(encoding=encoding, newline="") as handle:
            for raw in csv.DictReader(handle, delimiter="\t"):
                board = raw["board"]
                root = board_to_root.get(board, "board:" + hashlib.sha256(board.encode()).hexdigest())
                result.append(Row(
                    board=board,
                    root=root,
                    source=source,
                    n_discs=int(raw["n_discs"]),
                    deep_depth=int(raw["deep_depth"]),
                    shallow_depth=int(raw["shallow_depth"]),
                    shallow_value=int(raw["shallow_value"]),
                    deep_value=int(raw["deep_value"]),
                    error=int(raw["error"]),
                    shallow_nodes=int(raw["shallow_nodes"]),
                    deep_nodes=int(raw["deep_nodes"]),
                ))
    return result


def current_shallow(deep: int, offset: int) -> int:
    value = ((deep * 2 // 5) & ~1) + (deep & 1) + offset
    return min(deep - 2, max(deep & 1, value))


def select_candidate(rows: list[Row], offset: int, include_static: bool = True) -> list[Row]:
    return [
        row for row in rows
        if row.shallow_depth == current_shallow(row.deep_depth, offset)
        or (include_static and row.shallow_depth == 0)
    ]


def root_fold(root: str, folds: int) -> int:
    digest = hashlib.sha256(root.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def arrays(rows: list[Row]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.array([
        [row.n_discs / 64.0, row.shallow_depth / 60.0, row.deep_depth / 60.0]
        for row in rows
    ], dtype=np.float64)
    errors = np.array([row.error for row in rows], dtype=np.float64)
    root_counts = Counter(row.root for row in rows)
    weights = np.array([1.0 / root_counts[row.root] for row in rows], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return features, errors, weights


def sigma(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    x = features @ coefficients[:3]
    return ((coefficients[3] * x + coefficients[4]) * x + coefficients[5]) * x + coefficients[6]


def normalize_projection(coefficients: np.ndarray) -> np.ndarray:
    """a,b,cの尺度不定性だけを除き、表すsigma曲面は変えない。"""
    result = coefficients.copy()
    wanted = float(np.linalg.norm(LEGACY[:3]))
    actual = float(np.linalg.norm(result[:3]))
    if actual < 1.0e-12:
        return LEGACY.copy()
    k = wanted / actual
    result[:3] *= k
    result[3] /= k ** 3
    result[4] /= k ** 2
    result[5] /= k
    return result


def objective_gradient(
    features: np.ndarray,
    errors: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray]:
    x = features @ coefficients[:3]
    derivative_x = 3.0 * coefficients[3] * x * x + 2.0 * coefficients[4] * x + coefficients[5]
    predicted = ((coefficients[3] * x + coefficients[4]) * x + coefficients[5]) * x + coefficients[6]
    if np.any(predicted <= 0.10) or not np.all(np.isfinite(predicted)):
        return 1.0e12, np.zeros(7, dtype=np.float64)
    ratio2 = (errors / predicted) ** 2
    per_row = np.log(predicted) + 0.5 * ratio2
    loss = float(np.average(per_row, weights=weights))
    dl_dsigma = (1.0 - ratio2) / predicted
    scaled = weights * dl_dsigma / weights.sum()
    gradient = np.empty(7, dtype=np.float64)
    gradient[:3] = features.T @ (scaled * derivative_x)
    gradient[3] = np.sum(scaled * x ** 3)
    gradient[4] = np.sum(scaled * x ** 2)
    gradient[5] = np.sum(scaled * x)
    gradient[6] = np.sum(scaled)
    # 曲面を現行から不必要に遠ざけない弱い正則化。
    scale = np.maximum(np.abs(LEGACY), 1.0)
    delta = (coefficients - LEGACY) / scale
    loss += ridge * float(delta @ delta)
    gradient += 2.0 * ridge * delta / scale
    return loss, gradient


def fit(rows: list[Row], ridge: float, seed: int) -> np.ndarray:
    features, errors, weights = arrays(rows)
    # 係数が参照するのは(n_discs, shallow, deep)だけなので、同一特徴を
    # E[e^2]と重みへ集約する。Gaussian NLLと勾配は行単位計算と同一。
    features, inverse = np.unique(features, axis=0, return_inverse=True)
    bucket_weights = np.bincount(inverse, weights=weights)
    bucket_error2 = np.bincount(inverse, weights=weights * errors * errors)
    errors = np.sqrt(bucket_error2 / bucket_weights)
    weights = bucket_weights
    rng = np.random.default_rng(seed)
    starts = [LEGACY.copy()]
    for _ in range(2):
        trial = LEGACY.copy()
        trial[:3] *= 1.0 + rng.normal(0.0, 0.08, 3)
        trial[3:] *= 1.0 + rng.normal(0.0, 0.08, 4)
        starts.append(normalize_projection(trial))
    best_loss = math.inf
    best = LEGACY.copy()
    for start in starts:
        coefficients = start.copy()
        first = np.zeros(7)
        second = np.zeros(7)
        local_best_loss = math.inf
        local_best = coefficients.copy()
        stale = 0
        for iteration in range(1, 3501):
            loss, gradient = objective_gradient(features, errors, weights, coefficients, ridge)
            if loss < local_best_loss:
                local_best_loss = loss
                local_best = coefficients.copy()
                stale = 0
            else:
                stale += 1
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            corrected_first = first / (1.0 - 0.9 ** iteration)
            corrected_second = second / (1.0 - 0.999 ** iteration)
            rate = 0.003 * (0.25 + 0.75 * (1.0 - iteration / 3501.0))
            candidate = coefficients - rate * corrected_first / (np.sqrt(corrected_second) + 1.0e-8)
            candidate = normalize_projection(candidate)
            candidate_loss, _ = objective_gradient(features, errors, weights, candidate, ridge)
            if candidate_loss >= 1.0e11:
                first *= 0.5
                second *= 0.5
                continue
            coefficients = candidate
            if stale >= 700 and iteration >= 1500:
                break
        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best = local_best
    return normalize_projection(best)


def evaluate(rows: list[Row], coefficients: np.ndarray) -> dict[str, object]:
    features, errors, weights = arrays(rows)
    predicted = sigma(features, coefficients)
    normalized = errors / predicted
    result: dict[str, object] = {
        "rows": len(rows),
        "boards": len({row.board for row in rows}),
        "roots": len({row.root for row in rows}),
        "weighted_gaussian_nll": float(np.average(np.log(predicted) + 0.5 * normalized ** 2, weights=weights)),
        "weighted_normalized_rms": float(math.sqrt(np.average(normalized ** 2, weights=weights))),
        "weighted_normalized_mean": float(np.average(normalized, weights=weights)),
        "sigma_min": float(predicted.min()),
        "sigma_mean": float(np.average(predicted, weights=weights)),
        "sigma_max": float(predicted.max()),
        "levels": {},
    }
    nonstatic = np.array([row.shallow_depth != 0 for row in rows])
    static = ~nonstatic
    result["static_normalized_rms"] = (
        float(math.sqrt(np.average(normalized[static] ** 2, weights=weights[static])))
        if static.any() else 0.0
    )
    result["shallow_normalized_rms"] = (
        float(math.sqrt(np.average(normalized[nonstatic] ** 2, weights=weights[nonstatic])))
        if nonstatic.any() else 0.0
    )
    deep_nodes = np.array([row.deep_nodes for row in rows], dtype=np.float64)
    shallow_nodes = np.array([row.shallow_nodes for row in rows], dtype=np.float64)
    shallow_value = np.array([row.shallow_value for row in rows], dtype=np.int64)
    deep_value = np.array([row.deep_value for row in rows], dtype=np.int64)
    beta_grid = np.arange(-16, 17, dtype=np.int64)
    result["nonstatic_rows"] = int(nonstatic.sum())
    result["static_rows"] = int(static.sum())
    result["measured_shallow_nodes_sum"] = int(shallow_nodes[nonstatic].sum())
    result["measured_shallow_nodes_mean"] = float(shallow_nodes[nonstatic].mean()) if nonstatic.any() else 0.0
    for name, z in Z_LEVELS.items():
        margin = np.ceil(z * predicted).astype(np.int64)
        excess = np.maximum(0, np.abs(errors).astype(np.int64) - margin)
        selected_excess = excess[nonstatic]
        # NWSのbetaを-16..16で一様に置いた場合の、浅い探索によるcut数。
        cut_count = 0
        wrong_cut_count = 0
        estimated_nodes = 0.0
        no_mpc_nodes = 0.0
        for index in np.flatnonzero(nonstatic):
            high_cut = shallow_value[index] >= beta_grid + margin[index]
            low_cut = shallow_value[index] <= beta_grid - 1 - margin[index]
            high_wrong = high_cut & (deep_value[index] < beta_grid)
            low_wrong = low_cut & (deep_value[index] > beta_grid - 1)
            cuts = int(high_cut.sum() + low_cut.sum())
            wrong = int(high_wrong.sum() + low_wrong.sum())
            scenarios = 2 * len(beta_grid)
            cut_count += cuts
            wrong_cut_count += wrong
            no_mpc_nodes += scenarios * deep_nodes[index]
            estimated_nodes += scenarios * (deep_nodes[index] + shallow_nodes[index]) - cuts * deep_nodes[index]
        result["levels"][name] = {
            "margin_mean_nonstatic": float(np.mean(margin[nonstatic])) if nonstatic.any() else 0.0,
            "residual_outside_margin": int(np.sum(selected_excess >= 1)),
            "margin_excess_ge_2": int(np.sum(selected_excess >= 2)),
            "margin_excess_ge_4": int(np.sum(selected_excess >= 4)),
            "simulated_cut_count": cut_count,
            "simulated_wrong_cut_count": wrong_cut_count,
            "simulated_wrong_cut_rate": wrong_cut_count / cut_count if cut_count else 0.0,
            "estimated_nodes": estimated_nodes,
            "no_mpc_nodes": no_mpc_nodes,
            "estimated_node_ratio": estimated_nodes / no_mpc_nodes if no_mpc_nodes else 1.0,
        }
    return result


def calibrate_g_on_training(rows: list[Row], coefficients: np.ndarray) -> tuple[np.ndarray, float]:
    """2石/4石超過を現行以下に保つ最小のg補正を、学習側だけで決める。

    これは丸め済みmarginへの倍率ではなく、7係数モデルの切片gを変更する。
    統計的RMS fitそのものと混同しないよう、別候補として出力する。
    """
    nonstatic = [row for row in rows if row.shallow_depth != 0]
    features, errors, _ = arrays(nonstatic)
    absolute = np.abs(errors).astype(np.int64)

    def counts(model: np.ndarray) -> tuple[tuple[int, int, int], ...] | None:
        predicted = sigma(features, model)
        if np.any(predicted <= 0.10):
            return None
        result = []
        for z in Z_LEVELS.values():
            margin = np.ceil(z * predicted).astype(np.int64)
            excess = np.maximum(0, absolute - margin)
            result.append((int(np.sum(excess >= 1)), int(np.sum(excess >= 2)), int(np.sum(excess >= 4))))
        return tuple(result)

    baseline = counts(LEGACY)
    assert baseline is not None
    # 0.02石刻み。ge2/ge4は現行以下、1石以上は現行+1%（最低2件）まで。
    for delta in np.arange(-3.0, 6.0001, 0.02):
        candidate = coefficients.copy()
        candidate[6] += float(delta)
        value = counts(candidate)
        if value is None:
            continue
        safe = all(
            got[1] <= old[1]
            and got[2] <= old[2]
            and got[0] <= old[0] + max(2, math.ceil(old[0] * 0.01))
            for got, old in zip(value, baseline)
        )
        if safe:
            return candidate, float(delta)
    raise RuntimeError("gの探索範囲内に学習側安全性制約を満たす候補がありません")


def overlap_report(metadata_sets: dict[str, set[str]], board_to_root: dict[str, str]) -> dict[str, object]:
    names = sorted(metadata_sets)
    result: dict[str, object] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            left_boards = metadata_sets[left]
            right_boards = metadata_sets[right]
            left_roots = {board_to_root[board] for board in left_boards}
            right_roots = {board_to_root[board] for board in right_boards}
            result[f"{left}__{right}"] = {
                "board_overlap": len(left_boards & right_boards),
                "root_overlap": len(left_roots & right_roots),
            }
    return result


def write_coefficients(models: dict[str, list[float]]) -> None:
    lines = [
        "#pragma once",
        "// fit_mid_model.py が出力した比較用係数。ソース本体には未適用。",
        "// gは最終的な実効値。MPC_PROBCUT_G_OFFSETを追加してはならない。",
        "namespace mid_mpc_refit_20260830 {",
        "struct Coefficients { double a, b, c, d, e, f, g; };",
        "inline constexpr Coefficients current{",
        "    " + ", ".join(f"{value:.17g}" for value in LEGACY) + "};",
    ]
    for name, values in models.items():
        lines += [
            f"inline constexpr Coefficients {name}{{",
            "    " + ", ".join(f"{value:.17g}" for value in values) + "};",
        ]
    lines += ["} // namespace mid_mpc_refit_20260830", ""]
    (OUT / "candidate_coefficients.hpp").write_text("\n".join(lines), encoding="utf-8")


def write_tables(report: dict[str, object]) -> None:
    with (OUT / "coefficients.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "shallow_offset", "a", "b", "c", "d", "e", "f", "g"])
        writer.writerow(["current", "", *LEGACY.tolist()])
        for offset, item in report["candidates"].items():
            writer.writerow([f"rms_refit_{offset}", offset, *item["rms_coefficients"]])
            writer.writerow([f"safety_refit_{offset}", offset, *item["safety_coefficients"]])
    with (OUT / "cv_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "offset", "model", "level", "rows", "roots", "nll", "normalized_rms",
            "measured_shallow_nodes_sum", "measured_shallow_nodes_mean",
            "margin_mean", "outside", "excess_ge_2", "excess_ge_4", "cuts", "wrong_cuts",
            "wrong_cut_rate", "estimated_node_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, item in report["candidates"].items():
            for model_name in ("current", "rms_refit", "safety_refit"):
                metrics = item["cv_aggregate"][model_name]
                for level, values in metrics["levels"].items():
                    writer.writerow({
                        "offset": offset,
                        "model": model_name,
                        "level": level,
                        "rows": metrics["rows"],
                        "roots": metrics["roots"],
                        "nll": metrics["weighted_gaussian_nll"],
                        "normalized_rms": metrics["weighted_normalized_rms"],
                        "measured_shallow_nodes_sum": metrics["measured_shallow_nodes_sum"],
                        "measured_shallow_nodes_mean": metrics["measured_shallow_nodes_mean"],
                        "margin_mean": values["margin_mean_nonstatic"],
                        "outside": values["residual_outside_margin"],
                        "excess_ge_2": values["margin_excess_ge_2"],
                        "excess_ge_4": values["margin_excess_ge_4"],
                        "cuts": values["simulated_cut_count"],
                        "wrong_cuts": values["simulated_wrong_cut_count"],
                        "wrong_cut_rate": values["simulated_wrong_cut_rate"],
                        "estimated_node_ratio": values["estimated_node_ratio"],
                    })


def aggregate_fold_metrics(parts: list[dict[str, object]]) -> dict[str, object]:
    # 数え上げ値は加算。連続値は行数重み付き平均。node比は近似値なので同じく行数重み。
    total_rows = sum(int(part["rows"]) for part in parts)
    result: dict[str, object] = {
        "rows": total_rows,
        "boards": sum(int(part["boards"]) for part in parts),
        "roots": sum(int(part["roots"]) for part in parts),
        "nonstatic_rows": sum(int(part["nonstatic_rows"]) for part in parts),
        "static_rows": sum(int(part["static_rows"]) for part in parts),
        "measured_shallow_nodes_sum": sum(int(part["measured_shallow_nodes_sum"]) for part in parts),
    }
    result["measured_shallow_nodes_mean"] = (
        result["measured_shallow_nodes_sum"] / result["nonstatic_rows"]
        if result["nonstatic_rows"] else 0.0
    )
    for key in (
        "weighted_gaussian_nll", "weighted_normalized_rms", "weighted_normalized_mean",
        "static_normalized_rms", "shallow_normalized_rms", "sigma_min", "sigma_mean", "sigma_max",
    ):
        if key == "sigma_min":
            result[key] = min(float(part[key]) for part in parts)
        elif key == "sigma_max":
            result[key] = max(float(part[key]) for part in parts)
        else:
            # 各fold内の連続指標はroot-equal weightで計算済み。fold統合もroot数で重み付けする。
            weight_key = "roots"
            denominator = sum(int(part[weight_key]) for part in parts)
            if key.endswith("normalized_rms"):
                result[key] = math.sqrt(
                    sum(float(part[key]) ** 2 * int(part[weight_key]) for part in parts) / denominator
                )
            else:
                result[key] = sum(float(part[key]) * int(part[weight_key]) for part in parts) / denominator
    result["levels"] = {}
    for level in Z_LEVELS:
        level_parts = [part["levels"][level] for part in parts]
        cuts = sum(int(part["simulated_cut_count"]) for part in level_parts)
        wrong = sum(int(part["simulated_wrong_cut_count"]) for part in level_parts)
        estimated_nodes = sum(float(part["estimated_nodes"]) for part in level_parts)
        no_mpc_nodes = sum(float(part["no_mpc_nodes"]) for part in level_parts)
        result["levels"][level] = {
            "margin_mean_nonstatic": sum(float(part["margin_mean_nonstatic"]) * int(parts[i]["nonstatic_rows"]) for i, part in enumerate(level_parts)) / result["nonstatic_rows"],
            "residual_outside_margin": sum(int(part["residual_outside_margin"]) for part in level_parts),
            "margin_excess_ge_2": sum(int(part["margin_excess_ge_2"]) for part in level_parts),
            "margin_excess_ge_4": sum(int(part["margin_excess_ge_4"]) for part in level_parts),
            "simulated_cut_count": cuts,
            "simulated_wrong_cut_count": wrong,
            "simulated_wrong_cut_rate": wrong / cuts if cuts else 0.0,
            "estimated_nodes": estimated_nodes,
            "no_mpc_nodes": no_mpc_nodes,
            "estimated_node_ratio": estimated_nodes / no_mpc_nodes if no_mpc_nodes else 1.0,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    args = parser.parse_args()

    board_to_root, metadata_sets = metadata()
    model_dev = tsv_rows(sorted((ROOT / "benchmark").glob("mid_mpc_model_dev_d*_20260827.tsv")), "model_dev", board_to_root)
    probcut_dev = tsv_rows(sorted((ROOT / "benchmark").glob("mid_probcut_dev_d*_20260827.tsv")), "probcut_dev", board_to_root)
    original_holdout = tsv_rows(sorted((ROOT / "benchmark").glob("mid_probcut_holdout_d*_20260827.tsv")), "holdout_original", board_to_root)
    expanded = sorted(OUT.glob("holdout_all_d*.tsv"))
    holdout_paths = expanded or sorted((ROOT / "benchmark").glob("mid_probcut_holdout_d*_20260827.tsv"))
    holdout = tsv_rows(holdout_paths, "holdout", board_to_root)
    development_raw = model_dev + probcut_dev
    holdout_roots = {row.root for row in holdout}
    holdout_boards = {row.board for row in holdout}
    leaked_roots = {row.root for row in development_raw} & holdout_roots
    leaked_boards = {row.board for row in development_raw} & holdout_boards
    # holdoutの誤差値はfitや候補作成に一切使わない。metadata監査で判明した
    # 重複rootだけを開発集合から丸ごと隔離する。
    development = [row for row in development_raw if row.root not in leaked_roots]
    expanded_by_key = {
        (row.board, row.deep_depth, row.shallow_depth): row
        for row in holdout
    }
    common_original = [
        row for row in original_holdout
        if (row.board, row.deep_depth, row.shallow_depth) in expanded_by_key
    ]
    regeneration_mismatches = 0
    for old in common_original:
        new = expanded_by_key[(old.board, old.deep_depth, old.shallow_depth)]
        if (old.shallow_value, old.deep_value, old.error) != (new.shallow_value, new.deep_value, new.error):
            regeneration_mismatches += 1

    report: dict[str, object] = {
        "method": {
            "formula": "sigma=(d*x+e)*x^2+f*x+g, x=a*n_discs/64+b*shallow_depth/60+c*deep_depth/60",
            "fit_loss": "root-equal weighted zero-mean Gaussian negative log likelihood",
            "posthoc_scale": False,
            "fold_assignment": "sha256(game/root) mod folds",
            "folds": args.folds,
            "ridge": args.ridge,
            "z_levels_unchanged": Z_LEVELS,
            "wrong_cut_proxy": "NWS beta=-16..16; shallow cutoff contradicting deep score",
        },
        "inputs": {
            "model_dev_rows": len(model_dev),
            "probcut_dev_rows": len(probcut_dev),
            "holdout_rows": len(holdout),
            "holdout_expanded": bool(expanded),
            "holdout_original_rows_rechecked": len(common_original),
            "holdout_regeneration_score_mismatches": regeneration_mismatches,
            "development_rows_before_quarantine": len(development_raw),
            "development_rows_removed_by_root_quarantine": len(development_raw) - len(development),
            "development_boards_removed_by_root_quarantine": len({row.board for row in development_raw if row.root in leaked_roots}),
            "development_roots_removed_by_root_quarantine": len(leaked_roots),
            "development_boards": len({row.board for row in development}),
            "development_roots": len({row.root for row in development}),
            "holdout_boards": len({row.board for row in holdout}),
            "holdout_roots": len({row.root for row in holdout}),
            "metadata_sets": {
                name: {
                    "boards": len(boards),
                    "root_components": len({board_to_root[board] for board in boards}),
                }
                for name, boards in metadata_sets.items()
            },
        },
        "metadata_overlap": overlap_report(metadata_sets, board_to_root),
        "development_holdout_overlap": {
            "before_quarantine_boards": len(leaked_boards),
            "before_quarantine_roots": len(leaked_roots),
            "after_quarantine_boards": len({row.board for row in development} & holdout_boards),
            "after_quarantine_roots": len({row.root for row in development} & holdout_roots),
        },
        "current_coefficients": LEGACY.tolist(),
        "current_g_breakdown": {
            "source_base_g": 1.5719077939546169,
            "default_MPC_PROBCUT_G_OFFSET": 0.3,
            "effective_g_used_for_comparison": float(LEGACY[6]),
            "candidate_g_is_effective_and_must_not_add_offset": True,
        },
        "candidates": {},
    }

    models: dict[str, list[float]] = {}
    for offset in OFFSETS:
        selected_dev = select_candidate(development, offset)
        selected_holdout = select_candidate(holdout, offset)
        fold_reports_current: list[dict[str, object]] = []
        fold_reports_refit: list[dict[str, object]] = []
        fold_reports_safety: list[dict[str, object]] = []
        fold_coefficients: list[list[float]] = []
        fold_safety_coefficients: list[list[float]] = []
        fold_g_adjustments: list[float] = []
        for fold in range(args.folds):
            train = [row for row in selected_dev if root_fold(row.root, args.folds) != fold]
            validation = [row for row in selected_dev if root_fold(row.root, args.folds) == fold]
            coefficients = fit(train, args.ridge, 1000 + offset * 10 + fold)
            safety_coefficients, g_adjustment = calibrate_g_on_training(train, coefficients)
            fold_coefficients.append(coefficients.tolist())
            fold_safety_coefficients.append(safety_coefficients.tolist())
            fold_g_adjustments.append(g_adjustment)
            fold_reports_current.append(evaluate(validation, LEGACY))
            fold_reports_refit.append(evaluate(validation, coefficients))
            fold_reports_safety.append(evaluate(validation, safety_coefficients))
        final_coefficients = fit(selected_dev, args.ridge, 2000 + offset)
        final_safety_coefficients, final_g_adjustment = calibrate_g_on_training(selected_dev, final_coefficients)
        suffix = 'm' + str(-offset) if offset < 0 else 'p' + str(offset)
        models[f"rms_offset_{suffix}"] = final_coefficients.tolist()
        models[f"safety_offset_{suffix}"] = final_safety_coefficients.tolist()
        report["candidates"][str(offset)] = {
            "selected_development_rows": len(selected_dev),
            "selected_development_boards": len({row.board for row in selected_dev}),
            "selected_development_roots": len({row.root for row in selected_dev}),
            "depth_pairs": sorted({(row.deep_depth, row.shallow_depth) for row in selected_dev}),
            "rms_coefficients": final_coefficients.tolist(),
            "safety_coefficients": final_safety_coefficients.tolist(),
            "safety_g_adjustment": final_g_adjustment,
            "fold_coefficients": fold_coefficients,
            "fold_safety_coefficients": fold_safety_coefficients,
            "fold_g_adjustments": fold_g_adjustments,
            "cv_aggregate": {
                "current": aggregate_fold_metrics(fold_reports_current),
                "rms_refit": aggregate_fold_metrics(fold_reports_refit),
                "safety_refit": aggregate_fold_metrics(fold_reports_safety),
            },
            "holdout": {
                "rows": len(selected_holdout),
                "boards": len({row.board for row in selected_holdout}),
                "roots": len({row.root for row in selected_holdout}),
                "current": evaluate(selected_holdout, LEGACY),
                "rms_refit": evaluate(selected_holdout, final_coefficients),
                "safety_refit": evaluate(selected_holdout, final_safety_coefficients),
            },
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_coefficients(models)
    write_tables(report)
    print(json.dumps({
        "outputs": ["fit_results.json", "candidate_coefficients.hpp", "coefficients.csv", "cv_metrics.csv"],
        "development_holdout_overlap": report["development_holdout_overlap"],
        "holdout_expanded": report["inputs"]["holdout_expanded"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
