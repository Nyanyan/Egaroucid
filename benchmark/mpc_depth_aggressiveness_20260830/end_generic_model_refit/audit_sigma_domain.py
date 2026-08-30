#!/usr/bin/env python3
"""Audit raw generic-end MPC sigma predictions over observed/runtime domains."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


BASE = Path("benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit")
RESULTS = BASE / "results"
COEFFICIENTS = RESULTS / "coefficients.csv"
DATASET_AUDIT = RESULTS / "dataset_audit.csv"
OUTPUT_CSV = RESULTS / "sigma_domain_audit.csv"
OUTPUT_MD = RESULTS / "sigma_domain_audit.md"

MODEL_LABELS = {
    ("current", "original"): "現行（元の係数表現）",
    ("current", "equivalent a=-1"): "現行（a=-1の等価表現）",
    ("refit_root_equal", "a=-1"): "root_equal再学習",
    ("refit_root_equal_prior3", "a=-1"): "root_equal再学習・prior 3",
    ("refit_domain_equal", "a=-1"): "domain_equal再学習",
    ("refit_domain_equal_prior3", "a=-1"): "domain_equal再学習・prior 3",
}

ENGINE_VARIANTS = {
    ("current", "original"): "0",
    ("current", "equivalent a=-1"): "0（等価表現）",
    ("refit_root_equal_prior3", "a=-1"): "1",
    ("refit_domain_equal_prior3", "a=-1"): "2",
}

SCOPE_LABELS = {
    "collected_nonstatic_unique": "収集済みの非static浅い探索context",
    "code_route_all_deep3_60": "コード上の全列挙（深い探索3～60手）",
    "runtime_generic_selectivity_74_88_93":
        "実行時の汎用式（74%・88%・93%）",
    "runtime_generic_selectivity_98_99_99_9":
        "実行時の汎用式（98%・99%・99.9%）",
}


def shallow_depth(deep_depth: int) -> int:
    """Production formula with both experiment offsets equal to zero."""
    result = ((deep_depth * 2 // 5) & ~1) + (deep_depth & 1)
    return min(max(result, deep_depth & 1), deep_depth - 2)


def sigma(model: dict[str, str], n_discs: int, shallow: int) -> float:
    a, b, c, d, e, f = (
        float(model[name]) for name in ("a", "b", "c", "d", "e", "f")
    )
    u = a * n_discs / 64.0 + b * shallow / 60.0
    return c * u**3 + d * u**2 + e * u + f


def root_key(row: dict[str, Any]) -> str:
    return str(row.get("root_id", row.get("root_board", row["board"])))


def observation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        root_key(row), str(row["board"]), int(row["deep_depth"]),
        int(row["shallow_depth"]),
    )


def load_observations() -> tuple[list[dict[str, Any]], int, int, list[str]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_rows = 0
    inconsistent_n_discs = 0
    paths: list[str] = []
    with DATASET_AUDIT.open(encoding="utf-8", newline="") as file:
        datasets = list(csv.DictReader(file))
    for dataset in datasets:
        path = Path(dataset["path"])
        paths.append(path.as_posix())
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                deep = int(row["deep_depth"])
                shallow = int(row["shallow_depth"])
                # Historical files have no static_probe field.  A positive
                # shallow depth identifies their actual shallow-search row.
                if bool(row.get("static_probe", False)):
                    continue
                if shallow <= 0 or shallow != shallow_depth(deep):
                    continue
                source_rows += 1
                if int(row["n_discs"]) != 64 - deep:
                    inconsistent_n_discs += 1
                key = observation_key(row)
                previous = selected.get(key)
                if previous is None:
                    selected[key] = row
                else:
                    for field in ("n_discs", "deep_depth", "shallow_depth"):
                        if int(previous[field]) != int(row[field]):
                            raise ValueError(f"inconsistent {field}: {key}")
    return list(selected.values()), source_rows, inconsistent_n_discs, paths


def route_points(depths: Iterable[int]) -> list[dict[str, int]]:
    return [
        {
            "deep_depth": deep,
            "n_discs": 64 - deep,
            "shallow_depth": shallow_depth(deep),
        }
        for deep in depths
    ]


def summarize(
    model: dict[str, str], scope: str, points: list[dict[str, Any]]
) -> dict[str, Any]:
    values = [
        (
            sigma(model, int(point["n_discs"]), int(point["shallow_depth"])),
            int(point["deep_depth"]),
            int(point["n_discs"]),
            int(point["shallow_depth"]),
        )
        for point in points
    ]
    minimum = min(values)
    maximum = max(values)
    nonpositive = [value for value in values if value[0] <= 0.0]
    return {
        "model": model["model"],
        "representation": model["representation"],
        "engine_variant": ENGINE_VARIANTS.get(
            (model["model"], model["representation"]), ""
        ),
        "scope": scope,
        "point_count": len(points),
        "min_sigma": f"{minimum[0]:.12f}",
        "min_deep_depth": minimum[1],
        "min_n_discs": minimum[2],
        "min_shallow_depth": minimum[3],
        "max_sigma": f"{maximum[0]:.12f}",
        "max_deep_depth": maximum[1],
        "max_n_discs": maximum[2],
        "max_shallow_depth": maximum[3],
        "nonpositive_count": len(nonpositive),
        "positive_on_scope": "yes" if not nonpositive else "no",
    }


def nonpositive_depths(
    model: dict[str, str], points: list[dict[str, int]]
) -> str:
    depths = [
        point["deep_depth"]
        for point in points
        if sigma(model, point["n_discs"], point["shallow_depth"]) <= 0.0
    ]
    return "なし" if not depths else "、".join(str(depth) for depth in depths)


def markdown_table(rows: list[dict[str, Any]], scope: str) -> list[str]:
    selected = [row for row in rows if row["scope"] == scope]
    lines = [
        "| モデル | 点数 | 最小sigma | 最小となる（深さ, 石数, 浅い深さ） | "
        "最大sigma | 最大となる（深さ, 石数, 浅い深さ） | 0以下 |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for row in selected:
        label = MODEL_LABELS[(row["model"], row["representation"])]
        lines.append(
            f"| {label} | {row['point_count']} | {float(row['min_sigma']):.6f} | "
            f"({row['min_deep_depth']}, {row['min_n_discs']}, "
            f"{row['min_shallow_depth']}) | {float(row['max_sigma']):.6f} | "
            f"({row['max_deep_depth']}, {row['max_n_discs']}, "
            f"{row['max_shallow_depth']}) | {row['nonpositive_count']} |"
        )
    return lines


def main() -> None:
    with COEFFICIENTS.open(encoding="utf-8", newline="") as file:
        models = list(csv.DictReader(file))

    observations, source_rows, inconsistent, paths = load_observations()
    all_route = route_points(range(3, 61))
    generic_low = route_points(
        deep for deep in range(3, 61) if not 10 <= deep <= 18
    )
    generic_high = all_route
    scopes = {
        "collected_nonstatic_unique": observations,
        "code_route_all_deep3_60": all_route,
        "runtime_generic_selectivity_74_88_93": generic_low,
        "runtime_generic_selectivity_98_99_99_9": generic_high,
    }
    summaries = [
        summarize(model, scope, points)
        for scope, points in scopes.items()
        for model in models
    ]

    fields = [
        "model", "representation", "engine_variant", "scope", "point_count", "min_sigma",
        "min_deep_depth", "min_n_discs", "min_shallow_depth", "max_sigma",
        "max_deep_depth", "max_n_discs", "max_shallow_depth",
        "nonpositive_count", "positive_on_scope",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    current_original = next(
        model for model in models
        if model["model"] == "current" and model["representation"] == "original"
    )
    current_equivalent = next(
        model for model in models
        if model["model"] == "current" and model["representation"] != "original"
    )
    equivalence_error = max(
        abs(
            sigma(current_original, point["n_discs"], point["shallow_depth"])
            - sigma(current_equivalent, point["n_discs"], point["shallow_depth"])
        )
        for point in all_route
    )

    runtime_rows = {
        (row["model"], row["representation"]): row
        for row in summaries
        if row["scope"] == "runtime_generic_selectivity_98_99_99_9"
    }
    positive_models = [
        MODEL_LABELS[key]
        for key, row in runtime_rows.items()
        if row["positive_on_scope"] == "yes"
    ]
    nonpositive_models = [
        MODEL_LABELS[key]
        for key, row in runtime_rows.items()
        if row["positive_on_scope"] == "no"
    ]

    md: list[str] = [
        "# 汎用終盤MPC sigma式の定義域監査",
        "",
        "## 計算方法",
        "",
        "`coefficients.csv` の各係数を、エンジン本体と同じ次式へ代入した。",
        "",
        "```text",
        "u = a * n_discs / 64 + b * shallow_depth / 60",
        "sigma = c * u^3 + d * u^2 + e * u + f",
        "```",
        "",
        "ここで調べたのは式が返す生の値である。解析コード内で精度指標を"
        "計算するときの `max(0.5, sigma)` は、エンジン本体の"
        " `probcut_sigma_end()` にはないため適用していない。"
        " `MPC_SIGMA_SCALE` は既定値1.0としている。",
        "",
        "収集データは `dataset_audit.csv` に記録された8個のJSONLから読み込んだ。"
        " `static_probe=true`、浅い探索深さ0、現行の浅い探索深さと一致しない行を"
        "除き、`root・盤面・深い探索深さ・浅い探索深さ` が同じ行を1件にまとめた。",
        "",
        f"- 入力JSONL: {len(paths)}個",
        f"- 条件に合う入力行: {source_rows:,}行",
        f"- 重複除去後: {len(observations):,} context",
        f"- `n_discs != 64 - deep_depth`: {inconsistent}件",
        "",
        "コード上の列挙では、深い探索深さを3～60手、石数を"
        " `64 - deep_depth` とし、浅い探索深さを現在の式"
        " `((deep_depth * 2 / 5) & ~1) + (deep_depth & 1)`"
        "（整数除算、実装と同じ上下限制約）で求めた。",
        "",
        "## 収集された非static浅い探索context",
        "",
        *markdown_table(summaries, "collected_nonstatic_unique"),
        "",
        "## コード上の全列挙",
        "",
        "この表は深い探索3～60手の58点すべてに汎用式を数値として適用した結果である。"
        "実際の分岐による除外は次節に示す。",
        "",
        *markdown_table(summaries, "code_route_all_deep3_60"),
        "",
        "## special-table分岐を反映した実行時の範囲",
        "",
        "`use_recalibrated_end_mpc()` により、74%・88%・93%では深い探索10～18手に"
        "汎用sigma式を使わず、`END_MPC_SHALLOW_ERROR_HIGH/LOW` の専用表を使う。"
        "したがって、この3段階で汎用式が使われる列挙点は"
        "3～9手と19～60手の49点である。98%・99%・99.9%では専用表へ分岐しないため、"
        "汎用式の列挙点は3～60手の58点である。",
        "",
        "### 74%・88%・93%（汎用式が実際に使われる49点）",
        "",
        *markdown_table(summaries, "runtime_generic_selectivity_74_88_93"),
        "",
        "### 98%・99%・99.9%（汎用式が実際に使われる58点）",
        "",
        *markdown_table(summaries, "runtime_generic_selectivity_98_99_99_9"),
        "",
        "### 0以下となる深い探索深さ",
        "",
        "| モデル | 74%・88%・93%の汎用式範囲 | 98%・99%・99.9%の汎用式範囲 |",
        "|---|---|---|",
    ]
    for model in models:
        label = MODEL_LABELS[(model["model"], model["representation"])]
        md.append(
            f"| {label} | {nonpositive_depths(model, generic_low)} | "
            f"{nonpositive_depths(model, generic_high)} |"
        )
    md.extend([
        "",
        "### エンジン本体のコンパイル時variantとの対応",
        "",
        "| `END_MPC_SIGMA_MODEL_VARIANT` | 係数 | 汎用式の全58点で正値 |",
        "|---:|---|:---:|",
        "| 0 | 現行（元の係数表現） | はい |",
        "| 1 | root_equal再学習・prior 3 | はい |",
        "| 2 | domain_equal再学習・prior 3 | はい |",
        "",
        "`root_equal再学習` と `domain_equal再学習` のpriorなし二候補は"
        " `coefficients.csv` には記録されているが、現在の本体では"
        " `END_MPC_SIGMA_MODEL_VARIANT` の選択肢に接続されていない。",
        "",
        "深い探索10～18手・74%～93%の専用分岐では、上表のどの汎用式も"
        "呼ばれない。既定の専用表に記録された `END_MPC_SHALLOW_SIGMA` は"
        "3.552983～5.257206で、実際の判定に使う"
        " `END_MPC_SHALLOW_ERROR_HIGH/LOW` は5～11石である。",
        "",
        "## 正値を保つ範囲の整理",
        "",
        "収集済みcontextだけでなく、どの選択率でも実行時に汎用式が使われ得る"
        "全58点でsigmaが0より大きいモデルは次のとおり。",
        "",
        *[f"- {label}" for label in positive_models],
        "",
        "同じ全58点のどこかでsigmaが0以下になるモデルは次のとおり。",
        "",
        *([f"- {label}" for label in nonpositive_models] or ["- なし"]),
        "",
        "現行の二つの係数表現は同じ関数を表す。深い探索3～60手の58点で比較した"
        f"最大絶対差は `{equivalence_error:.3e}` だった。",
        "",
        "詳細な数値は [sigma_domain_audit.csv](sigma_domain_audit.csv) に保存した。",
        "この監査は各モデルの数値範囲を記録するもので、モデルの採否判定は含めない。",
        "",
        "## 再実行",
        "",
        "```powershell",
        "C:\\Users\\yaman\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe `",
        "  benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit/audit_sigma_domain.py",
        "```",
    ])
    OUTPUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
