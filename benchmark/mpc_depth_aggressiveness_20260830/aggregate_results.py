#!/usr/bin/env python3
"""MPC 比較結果を CSV と日本語 Markdown に集約する。

このスクリプトは実験を実行しない。既存の summary.csv / summary.json を読み、
次の三種類を混同しない形で表へ変換する。

1. 標準偏差モデルの係数 a～g（終盤の旧式は a～f）を再推定した比較
2. MPC の浅い探索深度を変え、候補ごとに誤差モデルを再学習した比較
3. 標準正規分布の z 値へ固定倍率を掛けただけの参考測定

各比較の入力が揃っているかを検査し、report.md の「測定の完了状況」に示す。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parents[1]


VARIANT_LABELS = {
    "control": "現行",
    "depth_m4": "浅い探索深度 -4手",
    "depth_m2": "浅い探索深度 -2手",
    "depth_refit0": "浅い探索深度 ±0手（再学習）",
    "depth_p0": "浅い探索深度 ±0手（再学習）",
    "depth_p2": "浅い探索深度 +2手",
    "depth_p4": "浅い探索深度 +4手",
    "refit_-4": "係数再推定・浅い探索深度 -4手",
    "refit_-2": "係数再推定・浅い探索深度 -2手",
    "refit_0": "係数再推定・浅い探索深度 ±0手",
    "refit_2": "係数再推定・浅い探索深度 +2手",
    "refit_4": "係数再推定・浅い探索深度 +4手",
}

MID_MODEL_RUNTIME_VARIANTS = {
    "control": "current",
    "m4": "refit_-4",
    "m2": "refit_-2",
    "p0": "refit_0",
    "p2": "refit_2",
    "p4": "refit_4",
}

END_GENERIC_MODEL_LABELS = {
    "current": "現行",
    "constrained_bernstein_root_equal_min0.5_prior0.03": "開始局面を均等に重み付けした係数",
    "constrained_bernstein_domain_equal_min0.5_prior3": "探索深度範囲を均等に重み付けした係数",
}

END_DATASET_LABELS = {
    "historical_202606_deep10_18": "2026年6月・深さ10～18",
    "locked_new_deep19_26": "新規独立確認・深さ19～26",
    "known_exact_deep30": "既知完全値・深さ30",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ratio(value: float, baseline: float) -> float | None:
    return value / baseline if baseline else None


def fmt_int(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{integer(value):,}"


def fmt_float(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "—"
    val = number(value, math.nan)
    if not math.isfinite(val):
        return "—"
    return f"{val:.{digits}f}"


def fmt_percent(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "—"
    val = number(value, math.nan)
    if not math.isfinite(val):
        return "—"
    return f"{100.0 * val:.{digits}f}%"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_escape(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "| 状態 |\n|---|\n| 対応する集計結果がありません |"
    header = "| " + " | ".join(title for _, title in columns) + " |"
    separator = "|" + "|".join("---:" if key not in {"label", "source", "dataset", "model", "status", "measurement", "required", "representation"} else "---" for key, _ in columns) + "|"
    body = [
        "| " + " | ".join(md_escape(row.get(key)) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def aggregate_accuracy(path: Path, source: str) -> list[dict[str, Any]]:
    """固定深度 accuracy summary を variant 単位で合計する。"""
    # `complete` は真偽値ではなく完了した局面数である。部分実行を予定局面数
    # `positions` の実測結果として混ぜないよう、両者が一致する行だけを使う。
    rows = [
        row
        for row in read_csv(path)
        if integer(row.get("positions")) > 0
        and integer(row.get("complete")) == integer(row.get("positions"))
    ]
    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    metadata: dict[tuple[str, str], dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        variant = row.get("variant", "")
        if not variant:
            continue
        dataset = row.get("dataset", "all") or "all"
        group = groups[(dataset, variant)]
        for field in (
            "candidate_nodes",
            "candidate_time_ms",
            "positions",
            "regret_ge_2",
            "regret_ge_4",
        ):
            group[field] += number(row.get(field))
        positions = number(row.get("positions"))
        group["weighted_regret"] += number(row.get("mean_regret")) * positions
        group["weighted_agreement"] += number(row.get("agreement")) * positions
        group["runs"] += 1
        for field in (
            "depth",
            "reference_depth",
            "mpc_level",
            "threads",
            "hash_level",
            "repetition",
        ):
            metadata[(dataset, variant)][field].add(integer(row.get(field)))

    result: list[dict[str, Any]] = []
    for (dataset, variant), values in sorted(groups.items(), key=lambda item: (item[0][0], variant_sort_key(item[0][1]))):
        control = groups.get((dataset, "control"), {})
        nodes = values["candidate_nodes"]
        time_ms = values["candidate_time_ms"]
        positions = values["positions"]
        result.append(
            {
                "source": f"{source}:{dataset}",
                "variant": variant,
                "label": display_variant(variant),
                "runs": int(values["runs"]),
                "positions": int(positions),
                "nodes": int(nodes),
                "node_ratio": ratio(nodes, control.get("candidate_nodes", 0.0)),
                "time_ms": int(time_ms),
                "time_ratio": ratio(time_ms, control.get("candidate_time_ms", 0.0)),
                "nps": nodes * 1000.0 / time_ms if time_ms else None,
                "regret_ge_2": int(values["regret_ge_2"]),
                "regret_ge_4": int(values["regret_ge_4"]),
                "mean_regret": values["weighted_regret"] / positions if positions else None,
                "agreement": values["weighted_agreement"] / positions if positions else None,
                **{
                    output_field: ",".join(
                        str(value) for value in sorted(metadata[(dataset, variant)][input_field])
                    )
                    for input_field, output_field in (
                        ("depth", "depths"),
                        ("reference_depth", "reference_depths"),
                        ("mpc_level", "mpc_levels"),
                        ("threads", "thread_counts"),
                        ("hash_level", "hash_levels"),
                        ("repetition", "repetitions"),
                    )
                },
            }
        )
    return result


def aggregate_mid_model_accuracy(path: Path, source: str) -> list[dict[str, Any]]:
    rows = aggregate_accuracy(path, source)
    for row in rows:
        mapped = MID_MODEL_RUNTIME_VARIANTS.get(str(row.get("variant")))
        if mapped is None:
            continue
        row["variant"] = mapped
        row["label"] = (
            "現行係数"
            if mapped == "current"
            else VARIANT_LABELS.get(mapped, mapped)
        )
    return rows


def variant_sort_key(variant: str) -> tuple[int, float, str]:
    if variant == "control":
        return (0, 0, variant)
    if variant == "current":
        return (0, 0, variant)
    refit_match = re.fullmatch(r"refit_(-?\d+)", variant)
    if refit_match:
        return (1, int(refit_match.group(1)), variant)
    if variant in {"depth_refit0", "depth_p0"}:
        return (1, 0, variant)
    depth_match = re.search(r"depth_(m|p)?(\d+)", variant)
    if depth_match:
        sign = -1 if depth_match.group(1) == "m" else 1
        return (1, sign * int(depth_match.group(2)), variant)
    z_match = re.fullmatch(r"z(\d+)", variant)
    if z_match:
        return (2, -int(z_match.group(1)), variant)
    return (3, 0, variant)


def display_variant(variant: str) -> str:
    if variant in VARIANT_LABELS:
        return VARIANT_LABELS[variant]
    z_match = re.fullmatch(r"z(\d+)", variant)
    if z_match:
        return f"分位点 z の倍率 {int(z_match.group(1)) / 100.0:.2f}"
    return variant


def display_measurement_source(source: str) -> str:
    """内部データ名をレポート向けの測定群名へ置き換える。"""
    dataset_labels = {
        "ggs": "GGS実戦由来局面",
        "all": "全データ",
        "dev": "開発用データ",
        "validation": "確認用データ",
    }
    if ":" not in source:
        return source
    measurement, dataset = source.rsplit(":", 1)
    return f"{measurement}：{dataset_labels.get(dataset, dataset)}"


def collect_fixed_time_depth(repo_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parent = repo_dir / "benchmark" / "mpc_sigma_sweep_20260829"
    for path in sorted(parent.glob("fixed_time_mpc0830_mid_depth_*_256p/summary.json")):
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        candidate = Path(str(data.get("candidate", ""))).stem
        match = re.search(r"depth_(m\d+|p\d+)", candidate)
        variant = f"depth_{match.group(1)}" if match else candidate
        confidence = data.get("candidate_pair_score_bootstrap_95pct") or [None, None]
        rows.append(
            {
                "variant": variant,
                "label": display_variant(variant),
                "pairs": integer(data.get("pairs")),
                "games": integer(data.get("games")),
                "move_time_ms": integer(data.get("move_time_ms")),
                "threads": integer(data.get("threads_per_engine")),
                "hash_level": integer(data.get("hash_level")),
                "opening_start": integer(data.get("opening_start")),
                "pair_wins": integer(data.get("candidate_pair_wins")),
                "pair_draws": integer(data.get("candidate_pair_draws")),
                "pair_losses": integer(data.get("candidate_pair_losses")),
                "pair_score_rate": number(data.get("candidate_pair_score_rate")),
                "score_ci_low": confidence[0] if len(confidence) > 0 else None,
                "score_ci_high": confidence[1] if len(confidence) > 1 else None,
                "disc_diff_per_game": number(data.get("candidate_average_disc_diff_per_game")),
                "error_count": len(data.get("errors") or []),
                "early_exit_count": len(
                    data.get("engines_exited_before_shutdown") or []
                ),
                "source": path.relative_to(repo_dir).as_posix(),
            }
        )
    return sorted(rows, key=lambda row: variant_sort_key(str(row["variant"])))


def collect_fixed_time_mid_model(repo_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parent = repo_dir / "benchmark" / "mpc_sigma_sweep_20260829"
    for path in sorted(parent.glob("fixed_time_mpc0830_mid_model_*_256p/summary.json")):
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        candidate = Path(str(data.get("candidate", ""))).stem
        match = re.search(r"mpc_model_(m4|m2|p0|p2|p4)$", candidate)
        if match is None:
            continue
        variant = MID_MODEL_RUNTIME_VARIANTS[match.group(1)]
        confidence = data.get("candidate_pair_score_bootstrap_95pct") or [None, None]
        rows.append(
            {
                "variant": variant,
                "label": VARIANT_LABELS.get(variant, variant),
                "pairs": integer(data.get("pairs")),
                "games": integer(data.get("games")),
                "move_time_ms": integer(data.get("move_time_ms")),
                "threads": integer(data.get("threads_per_engine")),
                "hash_level": integer(data.get("hash_level")),
                "opening_start": integer(data.get("opening_start")),
                "pair_wins": integer(data.get("candidate_pair_wins")),
                "pair_draws": integer(data.get("candidate_pair_draws")),
                "pair_losses": integer(data.get("candidate_pair_losses")),
                "pair_score_rate": number(data.get("candidate_pair_score_rate")),
                "score_ci_low": confidence[0] if len(confidence) > 0 else None,
                "score_ci_high": confidence[1] if len(confidence) > 1 else None,
                "disc_diff_per_game": number(data.get("candidate_average_disc_diff_per_game")),
                "error_count": len(data.get("errors") or []),
                "early_exit_count": len(
                    data.get("engines_exited_before_shutdown") or []
                ),
                "source": path.relative_to(repo_dir).as_posix(),
            }
        )
    return sorted(rows, key=lambda row: variant_sort_key(str(row["variant"])))


def collect_fixed_time_end_models(
    repo_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """終盤専用深度モデルと終盤汎用係数モデルの固定時間対戦を集計する。"""
    depth_rows: list[dict[str, Any]] = []
    generic_rows: list[dict[str, Any]] = []
    parent = repo_dir / "benchmark" / "mpc_sigma_sweep_20260829"
    depth_map = {
        "m2": "depth_m2",
        "p0": "depth_refit0",
        "p2": "depth_p2",
        "p4": "depth_p4",
    }
    generic_map = {
        "rootconstrained": "end_model_root",
        "domainconstrained": "end_model_domain",
    }
    generic_labels = {
        "end_model_root": "開始局面を均等に重み付けした係数",
        "end_model_domain": "探索深度範囲を均等に重み付けした係数",
    }

    for path in sorted(parent.glob("fixed_time_mpc0830_end_*_256p/summary.json")):
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        candidate = Path(str(data.get("candidate", ""))).stem
        depth_match = re.search(r"console_end_depth_(m2|p0|p2|p4)$", candidate)
        generic_match = re.search(
            r"console_end_mpc_model_(rootconstrained|domainconstrained)$",
            candidate,
        )
        if depth_match:
            variant = depth_map[depth_match.group(1)]
            label = display_variant(variant)
            destination = depth_rows
        elif generic_match:
            variant = generic_map[generic_match.group(1)]
            label = generic_labels[variant]
            destination = generic_rows
        else:
            continue

        confidence = data.get("candidate_pair_score_bootstrap_95pct") or [None, None]
        destination.append(
            {
                "variant": variant,
                "label": label,
                "pairs": integer(data.get("pairs")),
                "games": integer(data.get("games")),
                "move_time_ms": integer(data.get("move_time_ms")),
                "threads": integer(data.get("threads_per_engine")),
                "hash_level": integer(data.get("hash_level")),
                "opening_start": integer(data.get("opening_start")),
                "pair_wins": integer(data.get("candidate_pair_wins")),
                "pair_draws": integer(data.get("candidate_pair_draws")),
                "pair_losses": integer(data.get("candidate_pair_losses")),
                "pair_score_rate": number(data.get("candidate_pair_score_rate")),
                "score_ci_low": confidence[0] if len(confidence) > 0 else None,
                "score_ci_high": confidence[1] if len(confidence) > 1 else None,
                "disc_diff_per_game": number(
                    data.get("candidate_average_disc_diff_per_game")
                ),
                "error_count": len(data.get("errors") or []),
                "early_exit_count": len(
                    data.get("engines_exited_before_shutdown") or []
                ),
                "source": path.relative_to(repo_dir).as_posix(),
            }
        )

    depth_rows.sort(key=lambda row: variant_sort_key(str(row["variant"])))
    generic_rows.sort(key=lambda row: str(row["variant"]))
    return depth_rows, generic_rows


def collect_end_depth_offline(base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(base_dir / "end_offline_refit" / "summary.csv"):
        candidate = row.get("Candidate", "")
        variant = {"m2": "depth_m2", "p0": "depth_refit0", "p2": "depth_p2", "p4": "depth_p4"}.get(candidate, candidate)
        rows.append(
            {
                "variant": variant,
                "label": display_variant(variant),
                "cv_contexts": integer(row.get("CvContexts")),
                "cv_cuts": integer(row.get("CvCuts")),
                "cv_wrong": integer(row.get("CvWrong")),
                "cv_wrong_2": integer(row.get("CvWrong2")),
                "cv_wrong_4": integer(row.get("CvWrong4")),
                "cv_node_ratio": number(row.get("CvNodeRatio")),
                "holdout_contexts": integer(row.get("HoldoutContexts")),
                "holdout_cuts": integer(row.get("HoldoutCuts")),
                "holdout_wrong": integer(row.get("HoldoutWrong")),
                "holdout_wrong_2": integer(row.get("HoldoutWrong2")),
                "holdout_wrong_4": integer(row.get("HoldoutWrong4")),
                "holdout_node_ratio": number(row.get("HoldoutNodeRatio")),
            }
        )
    return sorted(rows, key=lambda row: variant_sort_key(str(row["variant"])))


def collect_end_depth_parameters(base_dir: Path) -> list[dict[str, Any]]:
    source = base_dir / "end_offline_refit" / "parameters_with_admission.csv"
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(source):
        groups[(row.get("Candidate", ""), integer(row.get("DeepDepth")))].append(row)
    result: list[dict[str, Any]] = []
    for (candidate, deep), rows in groups.items():
        ordered = sorted(rows, key=lambda row: integer(row.get("Level")))
        if not ordered:
            continue
        variant = {
            "m2": "depth_m2",
            "p0": "depth_refit0",
            "p2": "depth_p2",
            "p4": "depth_p4",
        }.get(candidate, candidate)
        result.append(
            {
                "variant": variant,
                "label": display_variant(variant),
                "deep_depth": deep,
                "shallow_depth": integer(ordered[0].get("ShallowDepth")),
                "sigma": number(ordered[0].get("Sigma")),
                "lower_tails": " / ".join(fmt_float(row.get("LowerTail"), 6) for row in ordered),
                "upper_tails": " / ".join(fmt_float(row.get("UpperTail"), 6) for row in ordered),
                "high_errors": " / ".join(str(integer(row.get("HighError"))) for row in ordered),
                "low_errors": " / ".join(str(integer(row.get("LowError"))) for row in ordered),
            }
        )
    return sorted(
        result,
        key=lambda row: (variant_sort_key(str(row["variant"])), integer(row["deep_depth"])),
    )


def collect_end_depth_runtime(base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(base_dir.glob("end_runtime_clean_*/summary.json")):
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        config = data.get("config", {})
        summary = data.get("summary", {})
        overall = summary.get("groups", {}).get("overall", {})
        candidate = str(config.get("candidate_label", ""))
        if candidate not in overall:
            keys = [key for key in overall if key != "scale100"]
            candidate = keys[0] if len(keys) == 1 else candidate
        candidate_values = overall.get(candidate, {})
        control_values = overall.get("scale100", {})
        pair = summary.get("pair_groups", {}).get("overall", {})
        if not candidate_values:
            continue
        rows.append(
            {
                "variant": candidate,
                "label": display_variant(candidate),
                "runs": integer(candidate_values.get("attempted")),
                "completed": integer(candidate_values.get("completed")),
                "timed_out": integer(candidate_values.get("timed_out")),
                "nodes": integer(candidate_values.get("nodes")),
                "control_nodes": integer(control_values.get("nodes")),
                "node_ratio_sum": number(pair.get("node_ratio_sum")),
                "node_ratio_geomean": number(pair.get("node_ratio_geomean")),
                "time_ms": integer(candidate_values.get("time_ms")),
                "control_time_ms": integer(control_values.get("time_ms")),
                "time_ratio_sum": number(pair.get("time_ratio_sum")),
                "time_ratio_geomean": number(pair.get("time_ratio_geomean")),
                "nps": integer(candidate_values.get("aggregate_nps")),
                "control_nps": integer(control_values.get("aggregate_nps")),
                "nps_ratio": ratio(
                    number(candidate_values.get("aggregate_nps")),
                    number(control_values.get("aggregate_nps")),
                ),
                "exact_value_matches": integer(candidate_values.get("exact_value_matches")),
                "exact_move_matches": integer(candidate_values.get("exact_move_matches")),
                "absolute_error_sum": integer(candidate_values.get("absolute_error_sum")),
                "count": integer(config.get("count")),
                "repetitions": integer(config.get("repetitions")),
                "levels": ",".join(str(value) for value in config.get("levels", [])),
                "threads": integer(config.get("threads")),
                "hash_level": integer(config.get("hash_level")),
                "timeout_seconds": number(config.get("timeout_seconds")),
                "cold_tt": str(config.get("cold_tt", "")),
                "source": path.relative_to(base_dir).as_posix(),
            }
        )
    return sorted(rows, key=lambda row: variant_sort_key(str(row["variant"])))


def collect_end_generic_runtime(base_dir: Path) -> list[dict[str, Any]]:
    labels = {
        "end_model_root": "開始局面を均等に重み付けした係数",
        "end_model_domain": "探索深度範囲を均等に重み付けした係数",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(base_dir.glob("end_generic_runtime*/summary.json")):
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        config = data.get("config", {})
        summary = data.get("summary", {})
        overall = summary.get("groups", {}).get("overall", {})
        candidate = str(config.get("candidate_label", ""))
        candidate_values = overall.get(candidate, {})
        control_values = overall.get("scale100", {})
        pair = summary.get("pair_groups", {}).get("overall", {})
        if not candidate_values:
            continue
        rows.append(
            {
                "variant": candidate,
                "label": labels.get(candidate, candidate),
                "runs": integer(candidate_values.get("attempted")),
                "completed": integer(candidate_values.get("completed")),
                "timed_out": integer(candidate_values.get("timed_out")),
                "nodes": integer(candidate_values.get("nodes")),
                "control_nodes": integer(control_values.get("nodes")),
                "node_ratio_sum": number(pair.get("node_ratio_sum")),
                "node_ratio_geomean": number(pair.get("node_ratio_geomean")),
                "time_ms": integer(candidate_values.get("time_ms")),
                "control_time_ms": integer(control_values.get("time_ms")),
                "time_ratio_sum": number(pair.get("time_ratio_sum")),
                "time_ratio_geomean": number(pair.get("time_ratio_geomean")),
                "nps": integer(candidate_values.get("aggregate_nps")),
                "control_nps": integer(control_values.get("aggregate_nps")),
                "nps_ratio": ratio(
                    number(candidate_values.get("aggregate_nps")),
                    number(control_values.get("aggregate_nps")),
                ),
                "exact_value_matches": integer(candidate_values.get("exact_value_matches")),
                "exact_move_matches": integer(candidate_values.get("exact_move_matches")),
                "absolute_error_sum": integer(candidate_values.get("absolute_error_sum")),
                "count": integer(config.get("count")),
                "repetitions": integer(config.get("repetitions")),
                "levels": ",".join(str(value) for value in config.get("levels", [])),
                "threads": integer(config.get("threads")),
                "hash_level": integer(config.get("hash_level")),
                "timeout_seconds": number(config.get("timeout_seconds")),
                "cold_tt": str(config.get("cold_tt", "")),
                "source": path.relative_to(base_dir).as_posix(),
            }
        )
    return rows


def collect_cold_end_runtime(
    base_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """32～44マス空きの実戦相当・固定時間終盤測定を集計する。"""
    label_by_executable = {
        "console_mpc_model_control": "現行",
        "console_end_depth_m2": "浅い探索深度 -2手",
        "console_end_depth_p0": "浅い探索深度 ±0手（再学習）",
        "console_end_depth_p2": "浅い探索深度 +2手",
        "console_end_depth_p4": "浅い探索深度 +4手",
        "console_end_mpc_model_rootconstrained": "開始局面を均等に重み付けした係数",
        "console_end_mpc_model_domainconstrained": "探索深度範囲を均等に重み付けした係数",
    }
    overall_rows: list[dict[str, Any]] = []
    by_empty_rows: list[dict[str, Any]] = []
    for path in sorted(base_dir.glob("cold_end_runtime_*/summary.json")):
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        metadata = data.get("metadata", {})
        overall = data.get("overall", {})
        by_empties = data.get("by_empties", {})
        if not isinstance(metadata, dict) or not isinstance(overall, dict):
            continue
        executable = Path(str(metadata.get("executable", ""))).stem
        label = label_by_executable.get(executable, executable or path.parent.name)
        result_rows = read_csv(path.parent / "results.csv")

        def result_depth_stats(empties: int | None) -> tuple[float | None, int | None]:
            depths = [
                integer(row.get("result_depth"))
                for row in result_rows
                if row.get("result_depth") not in (None, "")
                and not str(row.get("error", "")).strip()
                and (
                    empties is None
                    or integer(row.get("empties"), -1) == empties
                )
            ]
            if not depths:
                return None, None
            return statistics.median(depths), max(depths)

        def make_row(stats: Mapping[str, Any], empties: int | None) -> dict[str, Any]:
            result_depth_median, result_depth_max = result_depth_stats(empties)
            return {
                "variant": executable,
                "label": label,
                "empties": empties,
                "count": integer(stats.get("count")),
                "valid": integer(stats.get("valid")),
                "attempted": integer(stats.get("attempted")),
                "completed": integer(stats.get("completed")),
                "completion_rate": number(stats.get("completion_rate")),
                "exact_completed": integer(stats.get("exact_completed")),
                "first_end_time_median_ms": stats.get("first_end_time_median_ms"),
                "first_end_time_p90_ms": stats.get("first_end_time_p90_ms"),
                "nodes_median": stats.get("nodes_median"),
                "valid_nodes_median": stats.get("valid_nodes_median"),
                "nps_median": stats.get("nps_median"),
                "completed_nps_median": stats.get("completed_nps_median"),
                "end_start_time_median_ms": stats.get("end_start_time_median_ms"),
                "end_search_time_median_ms": stats.get("end_search_time_median_ms"),
                "cpu_average_cores_median": stats.get("cpu_average_cores_median"),
                "cpu_utilization_percent_median": stats.get(
                    "cpu_utilization_percent_median"
                ),
                "result_depth_median": result_depth_median,
                "result_depth_max": result_depth_max,
                "threads": integer(metadata.get("threads")),
                "movetime_ms": integer(metadata.get("movetime_ms")),
                "hash_level": integer(metadata.get("hash_level")),
                "min_empty": integer(metadata.get("min_empty")),
                "max_empty": integer(metadata.get("max_empty")),
                "max_per_empty": integer(metadata.get("max_per_empty")),
                "sampling_seed": integer(metadata.get("sampling_seed")),
                "extraction_error_count": integer(
                    metadata.get("extraction_error_count")
                ),
                "required_selectivity": number(
                    metadata.get("required_selectivity"), math.nan
                ),
                "case_timeout_seconds": number(
                    metadata.get("case_timeout_seconds"), math.nan
                ),
                "cold_tt": str(metadata.get("cold_tt_method", "")),
                "source": path.relative_to(base_dir).as_posix(),
            }

        overall_rows.append(make_row(overall, None))
        if isinstance(by_empties, dict):
            for empties, stats in by_empties.items():
                if isinstance(stats, dict):
                    by_empty_rows.append(make_row(stats, integer(empties)))

    order = {name: index for index, name in enumerate(label_by_executable)}
    overall_rows.sort(key=lambda row: order.get(str(row["variant"]), 999))
    by_empty_rows.sort(
        key=lambda row: (order.get(str(row["variant"]), 999), -integer(row["empties"]))
    )
    return overall_rows, by_empty_rows


def collect_end_z_reference(base_dir: Path) -> list[dict[str, Any]]:
    folder = base_dir / "end_aggression_refit"
    paths = [
        folder / "cv_summary.csv",
        folder / "holdout_202606_summary.csv",
        folder / "admission_holdout_summary.csv",
    ]
    result: list[dict[str, Any]] = []
    for path in paths:
        for row in read_csv(path):
            result.append(
                {
                    "dataset": row.get("scope", path.stem),
                    "candidate": row.get("candidate", ""),
                    "z_scale": number(row.get("z_scale")),
                    "contexts": integer(row.get("contexts")),
                    "cuts": integer(row.get("cuts")),
                    "wrong_cuts": integer(row.get("wrong_cuts")),
                    "wrong_2plus": integer(row.get("wrong_2plus")),
                    "wrong_4plus": integer(row.get("wrong_4plus")),
                    "simulated_nodes": integer(row.get("simulated_nodes")),
                    "node_ratio": number(row.get("node_ratio")),
                }
            )
    return result


def collect_mid_coefficients(base_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    folder = base_dir / "mid_model_refit"
    coeffs: list[dict[str, Any]] = []
    filtered_path = folder / "cv_filtered_coefficients.csv"
    if filtered_path.is_file():
        # These are the exact ridge choices compiled into
        # MID_MPC_RECALIBRATED_VARIANT.  coefficients.csv also contains other
        # exploratory ridge fits, so using it here can make the report disagree
        # with the executable even though both files are individually valid.
        filtered_rows = read_csv(filtered_path)
        current = next(
            (row for row in filtered_rows if row.get("kind") == "current"),
            None,
        )
        if current is not None:
            coeffs.append(
                {
                    "model": "current",
                    "label": "現行",
                    "shallow_offset": "",
                    "ridge": "",
                    **{
                        letter: number(
                            current.get("g_effective" if letter == "g" else letter)
                        )
                        for letter in "abcdefg"
                    },
                }
            )
        for row in filtered_rows:
            if row.get("kind") != "refit":
                continue
            offset = integer(row.get("offset"))
            variant = f"refit_{offset}"
            coeffs.append(
                {
                    "model": variant,
                    "label": VARIANT_LABELS.get(variant, variant),
                    "shallow_offset": offset,
                    "ridge": number(row.get("ridge")),
                    **{
                        letter: number(
                            row.get("g_effective" if letter == "g" else letter)
                        )
                        for letter in "abcdefg"
                    },
                }
            )
    else:
        for row in read_csv(folder / "coefficients.csv"):
            variant = row.get("variant", "")
            coeffs.append(
                {
                    "model": variant,
                    "label": VARIANT_LABELS.get(variant, "現行" if variant == "current" else variant),
                    "shallow_offset": row.get("shallow_offset", ""),
                    "ridge": row.get("ridge", ""),
                    **{letter: number(row.get(letter)) for letter in "abcdefg"},
                }
            )
    metrics: list[dict[str, Any]] = []
    for row in read_csv(folder / "cv_metrics.csv"):
        metrics.append(
            {
                "offset": integer(row.get("offset")),
                "model": row.get("model", ""),
                "level": integer(row.get("level")),
                "rows": integer(row.get("rows")),
                "roots": integer(row.get("roots")),
                "nll": number(row.get("nll")),
                "normalized_rms": number(row.get("normalized_rms")),
                "margin_mean": number(row.get("margin_mean")),
                "wrong_cuts": integer(row.get("wrong_cuts")),
                "wrong_cut_rate": number(row.get("wrong_cut_rate")),
                "excess_ge_2": integer(row.get("excess_ge_2")),
                "excess_ge_4": integer(row.get("excess_ge_4")),
                "estimated_node_ratio": number(row.get("estimated_node_ratio")),
            }
        )
    metadata = read_json(folder / "fit_results.json") or {}
    return coeffs, metrics, metadata


def collect_end_generic(base_dir: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    constrained = base_dir / "end_generic_model_refit" / "results_constrained"
    folder = constrained if (constrained / "coefficients.csv").is_file() else (
        base_dir / "end_generic_model_refit" / "results"
    )
    coeffs = []
    for row in read_csv(folder / "coefficients.csv"):
        model = row.get("model", "")
        coeffs.append(
            {
                "model": model,
                "label": END_GENERIC_MODEL_LABELS.get(model, model),
                "representation": row.get("representation", "") or (
                    "現行の多項式" if model == "current" else "正値制約から多項式へ変換"
                ),
                "global_sigma_multiplier": number(
                    row.get("global_sigma_multiplier"), 1.0
                ),
                **{letter: number(row.get(letter)) for letter in "abcdef"},
            }
        )
    selected_models = {str(row["model"]) for row in coeffs}
    cv = []
    for row in read_csv(folder / "cv_summary.csv"):
        if row.get("model", "") not in selected_models:
            continue
        cv.append(
            {
                "model": row.get("model", ""),
                "label": END_GENERIC_MODEL_LABELS.get(
                    row.get("model", ""), row.get("model", "")
                ),
                "folds": integer(row.get("folds")),
                "roots": integer(row.get("roots")),
                "observations": integer(row.get("observations")),
                "sigma_rmse": number(row.get("sigma_rmse")),
                "sigma_mae": number(row.get("sigma_mae")),
                "standardized_rms": number(row.get("standardized_rms")),
                "gaussian_nll": number(row.get("gaussian_nll")),
            }
        )
    holdout = []
    for row in read_csv(folder / "holdout_metrics.csv"):
        if row.get("model", "") not in selected_models:
            continue
        holdout.append(
            {
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "label": END_GENERIC_MODEL_LABELS.get(
                    row.get("model", ""), row.get("model", "")
                ),
                "roots": integer(row.get("roots")),
                "observations": integer(row.get("observations")),
                "sigma_rmse": number(row.get("sigma_rmse")),
                "sigma_mae": number(row.get("sigma_mae")),
                "standardized_rms": number(row.get("standardized_rms")),
                "gaussian_nll": number(row.get("gaussian_nll")),
            }
        )
    domain = []
    for row in read_csv(folder / "domain_audit.csv"):
        if row.get("model", "") not in selected_models:
            continue
        domain.append(
            {
                "model": row.get("model", ""),
                "label": END_GENERIC_MODEL_LABELS.get(
                    row.get("model", ""), row.get("model", "")
                ),
                "table_entries": integer(row.get("table_entries")),
                "minimum_raw_sigma": number(row.get("minimum_raw_sigma")),
                "minimum_n_discs": integer(row.get("minimum_n_discs")),
                "minimum_shallow_depth": integer(row.get("minimum_shallow_depth")),
                "maximum_raw_sigma": number(row.get("maximum_raw_sigma")),
                "nonpositive_entries": integer(row.get("nonpositive_entries")),
                "below_0_5_entries": integer(row.get("below_0_5_entries")),
            }
        )
    cuts = []
    for row in read_csv(folder / "cut_simulation.csv"):
        if row.get("model", "") not in selected_models:
            continue
        cuts.append(
            {
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "label": END_GENERIC_MODEL_LABELS.get(
                    row.get("model", ""), row.get("model", "")
                ),
                "level": integer(row.get("level")),
                "selectivity_percent": number(row.get("selectivity_percent")),
                "contexts": integer(row.get("contexts")),
                "probes": integer(row.get("probes")),
                "cuts": integer(row.get("cuts")),
                "wrong_cuts": integer(row.get("wrong_cuts")),
                "wrong_2plus": integer(row.get("wrong_2plus")),
                "wrong_4plus": integer(row.get("wrong_4plus")),
                "estimated_node_ratio": number(row.get("estimated_node_ratio")),
            }
        )
    return coeffs, cv, holdout, domain, cuts


def fixed_time_rows_complete(
    rows: Sequence[Mapping[str, Any]], expected_variants: set[str]
) -> bool:
    """2局1組対戦が表題の固定条件で全件完了したかを検査する。"""
    return (
        {str(row.get("variant")) for row in rows} == expected_variants
        and all(
            integer(row.get("pairs")) == 256
            and integer(row.get("games")) == 512
            and integer(row.get("move_time_ms")) == 100
            and integer(row.get("threads")) == 1
            and integer(row.get("hash_level")) == 20
            and integer(row.get("opening_start")) == 5000
            and integer(row.get("pair_wins"))
            + integer(row.get("pair_draws"))
            + integer(row.get("pair_losses"))
            == 256
            and integer(row.get("error_count")) == 0
            and integer(row.get("early_exit_count")) == 0
            for row in rows
        )
    )


def end_runtime_row_complete(row: Mapping[str, Any]) -> bool:
    """30マス空き8局面×3選択率の実機測定条件を検査する。"""
    runs = integer(row.get("runs"))
    return (
        integer(row.get("count")) == 8
        and integer(row.get("repetitions")) == 1
        and str(row.get("levels")) == "0,1,2"
        and integer(row.get("threads")) == 1
        and integer(row.get("hash_level")) == 29
        and math.isclose(number(row.get("timeout_seconds")), 180.0)
        and "fresh process" in str(row.get("cold_tt", "")).lower()
        and runs == 24
        and integer(row.get("completed")) + integer(row.get("timed_out")) == runs
    )


def runtime_status(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows or not all(end_runtime_row_complete(row) for row in rows):
        return "未完了"
    timed_out = sum(integer(row.get("timed_out")) for row in rows)
    return "完了" if timed_out == 0 else f"完了（時間切れ{timed_out}件を含む）"


def build_status(
    base_dir: Path,
    repo_dir: Path,
    mid_metadata: Mapping[str, Any],
    mid_model_depth: Sequence[Mapping[str, Any]],
    mid_model_fixed_time: Sequence[Mapping[str, Any]],
    end_runtime: Sequence[Mapping[str, Any]],
    end_depth_fixed_time: Sequence[Mapping[str, Any]],
    end_generic_runtime: Sequence[Mapping[str, Any]],
    end_generic_fixed_time: Sequence[Mapping[str, Any]],
    cold_end_overall: Sequence[Mapping[str, Any]],
    cold_end_by_empty: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    expanded = bool(mid_metadata.get("inputs", {}).get("holdout_expanded"))
    mid_model_fixed_time_complete = fixed_time_rows_complete(
        mid_model_fixed_time,
        {"refit_-4", "refit_-2", "refit_0", "refit_2", "refit_4"},
    )
    end_depth_fixed_time_complete = fixed_time_rows_complete(
        end_depth_fixed_time,
        {"depth_m2", "depth_refit0", "depth_p2", "depth_p4"},
    )
    end_generic_fixed_time_complete = fixed_time_rows_complete(
        end_generic_fixed_time, {"end_model_root", "end_model_domain"}
    )
    expected_cold_variants = {
        "console_mpc_model_control",
        "console_end_depth_m2",
        "console_end_depth_p0",
        "console_end_depth_p2",
        "console_end_depth_p4",
        "console_end_mpc_model_rootconstrained",
        "console_end_mpc_model_domainconstrained",
    }
    cold_empty_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cold_end_by_empty:
        cold_empty_groups[str(row.get("variant"))].append(row)
    cold_end_complete = (
        {str(row.get("variant")) for row in cold_end_overall}
        == expected_cold_variants
        and all(
            integer(row.get("count")) == 65
            and integer(row.get("valid")) == 65
            and integer(row.get("threads")) == 20
            and integer(row.get("movetime_ms")) == 15000
            and integer(row.get("hash_level")) == 29
            and integer(row.get("min_empty")) == 32
            and integer(row.get("max_empty")) == 44
            and integer(row.get("max_per_empty")) == 5
            and integer(row.get("sampling_seed")) == 20260823
            and integer(row.get("extraction_error_count")) == 0
            and math.isclose(number(row.get("required_selectivity")), 74.0)
            and math.isclose(number(row.get("case_timeout_seconds")), 105.0)
            and "fresh process" in str(row.get("cold_tt", "")).lower()
            for row in cold_end_overall
        )
        and set(cold_empty_groups) == expected_cold_variants
        and all(
            {integer(row.get("empties")) for row in rows} == set(range(32, 45))
            and all(
                integer(row.get("count")) == 5
                and integer(row.get("valid")) == 5
                for row in rows
            )
            for rows in cold_empty_groups.values()
        )
    )
    mid_model_depth_complete = (
        {str(row.get("variant")) for row in mid_model_depth}
        == {"current", "refit_-4", "refit_-2", "refit_0", "refit_2", "refit_4"}
        and all(integer(row.get("positions")) == 1800 for row in mid_model_depth)
        and all(integer(row.get("runs")) == 6 for row in mid_model_depth)
        and all(str(row.get("depths")) == "16" for row in mid_model_depth)
        and all(
            str(row.get("reference_depths")) == "16" for row in mid_model_depth
        )
        and all(
            str(row.get("mpc_levels")) == "0,1,2,3,4,5"
            for row in mid_model_depth
        )
        and all(str(row.get("thread_counts")) == "1" for row in mid_model_depth)
        and all(str(row.get("hash_levels")) == "20" for row in mid_model_depth)
        and all(str(row.get("repetitions")) == "1" for row in mid_model_depth)
    )
    constrained_audit = (
        base_dir / "end_generic_model_refit" / "results_constrained" /
        "domain_audit.csv"
    )
    constrained_domain_rows = read_csv(constrained_audit)
    constrained_domain_complete = (
        len(constrained_domain_rows) >= 3
        and all(integer(row.get("below_0_5_entries"), 1) == 0 for row in constrained_domain_rows)
    )
    regression = read_json(
        base_dir / "coefficient_variant_regression_full_precalc" / "results.json"
    )
    regression_complete = (
        isinstance(regression, dict)
        and bool(regression.get("all_passed"))
        and integer(regression.get("total")) >= 24
    )
    checks = [
        (
            "係数候補の実装が24構成で動作することの確認",
            "完了" if regression_complete else "未完了",
            "事前計算なし／あり、全候補、固定倍率併用拒否を含む24構成が全成功",
        ),
        (
            "中盤の係数再推定：独立確認用データの全浅い探索深度",
            "完了" if expanded and (base_dir / "mid_model_refit" / "report.md").is_file() else "未完了",
            "独立確認用データについて全浅い探索深度を収集済みであることと、日本語の詳細レポート",
        ),
        (
            "中盤の係数再推定候補：固定深度の探索時間・訪問局面数・探索速度・着手精度",
            "完了" if mid_model_depth_complete else "未完了",
            "現行と係数再推定候補を同一局面・同一深度で比較した集計結果",
        ),
        (
            "中盤の係数再推定候補：1手100ミリ秒の2局1組対戦",
            "完了" if mid_model_fixed_time_complete else "未完了",
            "各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20、開始番号5000、実行エラーなし",
        ),
        (
            "終盤の浅い探索深度：実行ファイルによる比較 -2手",
            runtime_status(
                [row for row in end_runtime if row.get("variant") == "depth_m2"]
            ),
            "30マス空き8局面、選択率74%・88%・93%、置換表を空にした比較",
        ),
        (
            "終盤の浅い探索深度：実行ファイルによる比較 ±0手（再学習）",
            runtime_status(
                [
                    row
                    for row in end_runtime
                    if row.get("variant") == "depth_refit0"
                ]
            ),
            "30マス空き8局面、選択率74%・88%・93%、置換表を空にした比較",
        ),
        (
            "終盤の浅い探索深度：実行ファイルによる比較 +2手",
            runtime_status(
                [row for row in end_runtime if row.get("variant") == "depth_p2"]
            ),
            "30マス空き8局面、選択率74%・88%・93%、置換表を空にした比較",
        ),
        (
            "終盤の浅い探索深度：実行ファイルによる比較 +4手",
            runtime_status(
                [row for row in end_runtime if row.get("variant") == "depth_p4"]
            ),
            "30マス空き8局面、選択率74%・88%・93%、置換表を空にした比較",
        ),
        (
            "終盤の a～f 係数再推定候補：実行ファイルで測った探索時間・訪問局面数・探索速度",
            runtime_status(end_generic_runtime)
            if {str(row.get("variant")) for row in end_generic_runtime}
            == {"end_model_root", "end_model_domain"}
            else "未完了",
            "現行と係数再推定候補を同一局面で交互に実行した集計結果",
        ),
        (
            "終盤の浅い探索深度候補：1手100ミリ秒の2局1組対戦",
            "完了" if end_depth_fixed_time_complete else "未完了",
            "4候補について各256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20、開始番号5000、実行エラーなし",
        ),
        (
            "終盤の a～f 係数再推定候補：1手100ミリ秒の2局1組対戦",
            "完了" if end_generic_fixed_time_complete else "未完了",
            "2候補について各256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20、開始番号5000、実行エラーなし",
        ),
        (
            "終盤の a～f 係数再推定候補：適用範囲外で標準偏差が正になることの保証",
            "完了" if constrained_domain_complete else "未完了",
            "盤上石数0～64、浅い探索深度0～60の全3,965組合せで標準偏差 σ が0.5以上",
        ),
        (
            "終盤の実戦相当局面：20スレッド・15秒の読み切り数と最終探索深度",
            "完了" if cold_end_complete else "未完了",
            "7設定、32～44マス空き各5局面、20スレッド、15秒、置換表容量設定29、置換表を空にした表",
        ),
    ]
    return [
        {"measurement": measurement, "status": status, "required_output": required}
        for measurement, status, required in checks
    ]


def render_report(
    *,
    mid_model_depth: Sequence[Mapping[str, Any]],
    mid_model_fixed_time: Sequence[Mapping[str, Any]],
    mid_depth: Sequence[Mapping[str, Any]],
    mid_fixed_time: Sequence[Mapping[str, Any]],
    end_depth_offline: Sequence[Mapping[str, Any]],
    end_depth_parameters: Sequence[Mapping[str, Any]],
    end_depth_runtime: Sequence[Mapping[str, Any]],
    end_depth_fixed_time: Sequence[Mapping[str, Any]],
    mid_z: Sequence[Mapping[str, Any]],
    end_z: Sequence[Mapping[str, Any]],
    mid_coeffs: Sequence[Mapping[str, Any]],
    mid_cv: Sequence[Mapping[str, Any]],
    mid_metadata: Mapping[str, Any],
    end_coeffs: Sequence[Mapping[str, Any]],
    end_cv: Sequence[Mapping[str, Any]],
    end_holdout: Sequence[Mapping[str, Any]],
    end_domain: Sequence[Mapping[str, Any]],
    end_cut_simulation: Sequence[Mapping[str, Any]],
    end_generic_runtime: Sequence[Mapping[str, Any]],
    end_generic_fixed_time: Sequence[Mapping[str, Any]],
    cold_end_overall: Sequence[Mapping[str, Any]],
    cold_end_by_empty: Sequence[Mapping[str, Any]],
    status: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# Multi-ProbCut（MPC）の浅い探索深度・標準偏差モデル・枝刈り確率の比較",
            "",
            "この文書は測定値を並べるための集計レポートである。どの設定を本体へ反映するかは記載しない。",
            "",
            "## 用語の定義",
            "",
            "| 用語・記号 | この文書での意味 |",
            "|---|---|",
            "| MPC | Multi-ProbCutの略。浅い探索の結果と誤差分布から、深い探索を省略してよいか判定する選択的枝刈り法。 |",
            "| 深い探索深度 | 本来求めたい探索の深度。 |",
            "| 浅い探索深度 | MPCが深い探索を省略できるか判定するため、先に実行する探索の深度。 |",
            "| MPC判定機会 | ある局面・探索窓・選択率で、MPCによる枝刈りを検討した1回。 |",
            "| 枝刈り | 浅い探索または静的評価に基づき、深い探索を行わずに値が探索窓の外側にあると判定すること。表の枝刈り回数は、この判定が成立した回数。 |",
            "| 誤った枝刈り | 枝刈りした方向と完全読みの値が矛盾したもの。`2石以上`と`4石以上`は、完全読みの値が探索窓の境界から外れた大きさ。 |",
            "| 訪問局面数 | 探索中に処理した局面の延べ数。実装内部でノード数と呼んでいる値。 |",
            "| 推定訪問局面数比 | 浅い探索の訪問局面数と、枝刈りできなかった場合の深い探索の訪問局面数を合計し、すべてを深く探索した場合の訪問局面数で割った値。経過時間の比ではない。 |",
            "| 探索速度 | 1秒当たりの訪問局面数。 |",
            "| 標準偏差 `σ` | 浅い探索値と深い探索値の誤差の広がりを予測する値。 |",
            "| 分位点 `z` | 指定した枝刈り確率を標準正規分布上の距離へ変換した値。MPCの判定幅は主に `σ×z` で決まる。 |",
            "| 選択率 | MPCの判定幅を決める信頼水準。高いほど枝刈り条件が厳しくなり、100%ではMPCによる枝刈りを行わない。 |",
            "| 選択率レベル | 実装内部で選択率を指定する整数番号。表には対応する選択率も併記する。 |",
            "| 整数誤差幅 | MPC判定で評価値に加減する石数単位の境界幅。標準偏差と上下の分位点から求め、整数に丸めた値。 |",
            "| 交差検証 | 学習用データを複数群に分け、一部で学習し、残りで誤差を測る手順を群を入れ替えて繰り返す検証方法。 |",
            "| 開始局面群 | 同じ元の開始局面から得た誤差標本をまとめた単位。交差検証では同じ群を学習側と検証側へ分割しない。 |",
            "| 独立確認用データ | 係数や設定の選択には使わず、選択後の性能確認だけに使うデータ。 |",
            "| 事前分布強度 | 係数が極端な値へ動くことを抑える正則化の強さ。0は正則化なしを表す。 |",
            "| 正規分布負対数尤度 | 予測した標準偏差と観測誤差の適合度。小さいほど適合が良い。 |",
            "| 標準化残差の二乗平均平方根 | 観測誤差を予測標準偏差で割った値の二乗平均平方根。1に近いほど予測した誤差幅と観測誤差の大きさが一致する。 |",
            "| 通常窓探索 | 下限と上限の間に幅を持つ探索窓を使う探索。 |",
            "| ゼロ窓探索 | 候補値が境界を超えるかだけを調べる、幅1の探索窓を使う探索。 |",
            "| 固定深度の着手損失 | 完全読みを基準として、候補が選んだ手の評価値が最善手より何石低いかを表す値。 |",
            "| 固定時間対戦 | 同じ開始局面を先後交替で2局行い、その2局を1組として勝ち・引き分け・負けを数える測定。 |",
            "| 組スコア率 | 2局1組対戦で、勝ちを1点、引き分けを0.5点、負けを0点として組数で割った値。 |",
            "| 読み切り | 指定した選択率の終盤探索が、制限時間内に終局まで完了すること。 |",
            "| 完全読み | 選択率100%の終盤探索が終局まで完了し、厳密な評価値を求めること。 |",
            "| 置換表容量設定 | 置換表に割り当てる容量を実装の設定番号で表したもの。 |",
            "| 中央処理装置使用率 | 指定した探索スレッド数に対し、実際に中央処理装置を使った割合。表ではCPU使用率と略記する。 |",
            "| GGS | Generic Game Serverの名称。このレポートでは、GGS対戦から抽出した局面の測定群を示すときに使う。 |",
            "",
            "## 比較の区分",
            "",
            "次の三つは変更内容が異なるため、表を分けている。",
            "",
            "1. **標準偏差モデルの係数再推定**：中盤では `a`～`g`、終盤の汎用式では `a`～`f` をデータから再推定する。標準正規分布の分位点 `z` は変えない。",
            "2. **浅い探索深度の変更**：MPCが境界判定に使う浅い探索の深度を変える。終盤の深さ10～18手については、候補深度ごとに標準偏差・上下分位点・整数誤差幅を学習し直している。",
            "3. **分位点 `z` へ固定倍率を掛けた参考測定**：標準偏差モデルの係数は変えず、すべての分位点 `z` に同じ倍率を掛ける。この測定は係数再推定の結果ではない。",
            "",
            "## 1. 標準偏差モデルの係数再推定",
            "",
            "### 1.1 中盤の a～g 係数",
            "",
            "使用式は `x = a×(石数/64) + b×(浅い探索深度/60) + c×(深い探索深度/60)`、`σ = (d×x+e)×x² + f×x + g` である。下表の候補は係数そのものを再推定しており、標準偏差 `σ` の計算後に固定倍率を掛けていない。固定深度測定で使う選択率6段階の分位点 `z` は、現行と同じ 1.13 / 1.55 / 1.81 / 2.088 / 2.1845 / 2.7965 である。",
            "",
        ]
    )
    mid_coeff_table = []
    for row in mid_coeffs:
        mid_coeff_table.append(
            {
                "label": row["label"],
                "shallow_offset": row["shallow_offset"] if row["shallow_offset"] != "" else "—",
                "ridge": fmt_float(row.get("ridge"), 4) if row.get("ridge") != "" else "—",
                **{letter: fmt_float(row[letter], 6) for letter in "abcdefg"},
            }
        )
    expanded = bool(mid_metadata.get("inputs", {}).get("holdout_expanded"))
    if not expanded:
        lines.extend(["この表は追加の独立確認用データについて全浅い探索深度の収集が終わる前の暫定出力である。最終の係数表としては扱わない。", ""])
    lines.append(markdown_table(mid_coeff_table, [("label", "設定"), ("shallow_offset", "浅い探索深度差"), ("ridge", "リッジ回帰の正則化係数"), *((letter, letter) for letter in "abcdefg")]))
    lines.extend(["", "実行ファイルへ組み込んだ係数の全桁は[中盤の係数一覧](mid_model_refit/cv_filtered_coefficients.csv)にある。リッジ回帰の正則化係数は、開始局面群単位の5分割交差検証だけで選んだ。独立確認用データの誤差は、係数や正則化係数の選択に使っていない。学習用と独立確認用の間には、同一盤面、8対称変換、色反転を含む盤面重複がないことを確認した。", ""])

    selected_cv = sorted(
        [
            row
            for row in mid_cv
            if str(row.get("model")) in {"current", "safety_refit"}
        ],
        key=lambda row: (
            integer(row.get("offset")),
            0 if str(row.get("model")) == "current" else 1,
            integer(row.get("level")),
        ),
    )
    mid_cv_table = [
        {
            "offset": f"{integer(row['offset']):+d}手",
            "model": "現行係数" if row["model"] == "current" else "候補専用の再推定係数",
            "level": f"{row['level']}%",
            "rows": fmt_int(row["rows"]),
            "roots": fmt_int(row["roots"]),
            "nll": fmt_float(row["nll"], 4),
            "norm_rms": fmt_float(row["normalized_rms"], 4),
            "wrong": fmt_int(row["wrong_cuts"]),
            "wrong_rate": fmt_percent(row["wrong_cut_rate"], 3),
            "wrong2": fmt_int(row["excess_ge_2"]),
            "wrong4": fmt_int(row["excess_ge_4"]),
            "node_ratio": fmt_float(row["estimated_node_ratio"], 4),
        }
        for row in selected_cv
    ]
    lines.extend(
        [
            "各浅い探索深度について、現行係数をそのまま使った場合と、その深度専用に再推定した係数を使った場合の交差検証値を示す。標準化残差の二乗平均平方根は、残差を予測標準偏差で割った値の二乗平均平方根であり、1に近いほど予測した誤差幅と観測誤差の大きさが一致する。",
            "",
            markdown_table(mid_cv_table, [("offset", "浅い探索深度差"), ("model", "モデル"), ("level", "選択率"), ("rows", "誤差標本"), ("roots", "開始局面群"), ("nll", "正規分布負対数尤度"), ("norm_rms", "標準化残差の二乗平均平方根"), ("wrong", "誤った枝刈り"), ("wrong_rate", "誤った枝刈りの割合"), ("wrong2", "2石以上"), ("wrong4", "4石以上"), ("node_ratio", "推定訪問局面数比")]),
            "",
            "独立確認用データを含む全表は[中盤係数再推定の詳細レポート](mid_model_refit/report.md)にある。",
            "",
        ]
    )
    lines.append(f"追加の独立確認用データについて全浅い探索深度を収集済みか: **{'はい' if expanded else 'いいえ（未完了）'}**。")

    model_depth_table = [
        {
            "label": row["label"],
            "positions": fmt_int(row["positions"]),
            "nodes": fmt_int(row["nodes"]),
            "node_ratio": fmt_float(row["node_ratio"], 4),
            "time_ms": fmt_int(row["time_ms"]),
            "time_ratio": fmt_float(row["time_ratio"], 4),
            "nps": fmt_int(row["nps"]),
            "wrong2": fmt_int(row["regret_ge_2"]),
            "wrong4": fmt_int(row["regret_ge_4"]),
            "mean_regret": fmt_float(row["mean_regret"], 3),
            "agreement": fmt_percent(row["agreement"], 2),
        }
        for row in mid_model_depth
    ]
    lines.extend(
        [
            "",
            "#### 1.1.1 再推定した a～g を使う固定深度測定",
            "",
            "独立確認用の300局面を、中盤深さ16手、選択率6段階、1スレッド、各探索の開始時に置換表を空にして測定した。各行の局面×選択率は1,800である。表に示した浅い探索深度差と、その深度専用に再推定した a～g を同時に使っている。訪問局面数と探索時間は1,800探索の合計であり、探索速度は合計訪問局面数を合計探索時間で割った値である。現行係数のまま深度だけを変えた2章の初期測定とは別である。",
            "",
            markdown_table(model_depth_table, [("label", "設定"), ("positions", "局面×選択率"), ("nodes", "合計訪問局面数"), ("node_ratio", "訪問局面数比"), ("time_ms", "合計探索時間（ミリ秒）"), ("time_ratio", "時間比"), ("nps", "探索速度（局面/秒）"), ("wrong2", "2石以上の着手損失"), ("wrong4", "4石以上の着手損失"), ("mean_regret", "平均着手損失"), ("agreement", "完全読みとの着手一致率")]),
            "",
        ]
    )

    model_fixed_time_table = [
        {
            "label": row["label"],
            "pairs": fmt_int(row["pairs"]),
            "wdl": f"{row['pair_wins']} / {row['pair_draws']} / {row['pair_losses']}",
            "score": fmt_percent(row["pair_score_rate"], 2),
            "ci": f"{fmt_percent(row['score_ci_low'], 2)} ～ {fmt_percent(row['score_ci_high'], 2)}",
            "disc": fmt_float(row["disc_diff_per_game"], 3),
        }
        for row in mid_model_fixed_time
    ]
    lines.extend(
        [
            "#### 1.1.2 再推定した a～g を使う1手100ミリ秒対戦",
            "",
            "各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20で測定した。開始局面は同じ一覧の番号5000～5255を使った。1組は同じ開始局面を先後交替で行う2局である。勝ち・引き分け・負けは、候補から見た組単位の数である。",
            "",
            markdown_table(model_fixed_time_table, [("label", "設定"), ("pairs", "組数"), ("wdl", "組の勝ち / 引き分け / 負け"), ("score", "組スコア率"), ("ci", "復元抽出法による95%信頼区間"), ("disc", "1局当たり平均石差")]),
            "",
        ]
    )
    lines.extend(["", "### 1.2 終盤の汎用 a～f 係数", ""])
    lines.extend(
        [
            "対象式は `x = 石数/64`、`y = 浅い探索深度/60`、`u = a×x + b×y`、`σ = c×u³ + d×u² + e×u + f` である。主な使用箇所は深い探索深度19手以上である。深い探索深度10～18手・選択率74%～93%の専用表とは別である。",
            "",
            "係数再推定に使った新規データの中心は深い探索深度19～26手である。履歴データには深い探索深度10～18手があり、深さ30手は8局面を独立確認に使った。候補式は、三次バーンスタイン基底の4係数をそれぞれ0.5以上に制約して学習し、実行時には従来と同じ三次多項式の c～f へ変換している。このため指数関数や実行時の範囲判定は増えない。学習用と各独立確認用データの間には、開始局面、同一盤面、8対称変換、色反転を含む重複がないことを確認した。",
            "",
            "`u` の尺度と c～f の尺度が重複するため、候補では a=-1に固定してb～fを推定した。事前分布強度が0より大きい候補だけを交差検証で比較した。正則化なし、すなわち事前分布強度0の設定では、開始局面を均等に重み付けするモデルの b が-31.6665まで動き、対象範囲外への外挿を抑制できなかったため、候補選択から除外した。したがって、下表の候補は、事前分布強度0を含む全設定から交差検証で選んだ最良値ではない。",
            "",
        ]
    )
    end_coeff_table = [
        {
            "model": row["label"],
            "representation": row["representation"],
            "multiplier": fmt_float(row["global_sigma_multiplier"], 1),
            **{letter: fmt_float(row[letter], 6) for letter in "abcdef"},
        }
        for row in end_coeffs
    ]
    lines.append(markdown_table(end_coeff_table, [("model", "モデル"), ("representation", "表現"), *((letter, letter) for letter in "abcdef"), ("multiplier", "標準偏差への固定倍率")]))
    lines.extend(["", "係数の全桁は[終盤の係数一覧](end_generic_model_refit/results_constrained/coefficients.csv)にある。標準偏差への固定倍率は全候補で1.0であり、候補間で異なるのは a～f だけである。", ""])

    end_domain_table = [
        {
            "model": row["label"],
            "entries": fmt_int(row["table_entries"]),
            "minimum": fmt_float(row["minimum_raw_sigma"], 6),
            "at": f"石数{row['minimum_n_discs']}・浅い探索深度{row['minimum_shallow_depth']}",
            "maximum": fmt_float(row["maximum_raw_sigma"], 6),
            "nonpositive": fmt_int(row["nonpositive_entries"]),
            "below": fmt_int(row["below_0_5_entries"]),
        }
        for row in end_domain
    ]
    lines.extend(
        [
            "石数0～64と浅い探索深度0～60の直積、全3,965組合せへ多項式を直接代入した監査結果である。",
            "",
            markdown_table(end_domain_table, [("model", "モデル"), ("entries", "検査組合せ"), ("minimum", "最小標準偏差"), ("at", "最小位置"), ("maximum", "最大標準偏差"), ("nonpositive", "標準偏差が0以下"), ("below", "標準偏差が0.5未満")]),
            "",
        ]
    )
    end_cv_table = [
        {
            "model": row["label"],
            "roots": fmt_int(row["roots"]),
            "obs": fmt_int(row["observations"]),
            "sigma_rmse": fmt_float(row["sigma_rmse"], 4),
            "sigma_mae": fmt_float(row["sigma_mae"], 4),
            "std_rms": fmt_float(row["standardized_rms"], 4),
            "nll": fmt_float(row["gaussian_nll"], 4),
        }
        for row in end_cv
    ]
    lines.append(markdown_table(end_cv_table, [("model", "モデル"), ("roots", "開始局面群"), ("obs", "誤差標本"), ("sigma_rmse", "標準偏差予測の二乗平均平方根誤差"), ("sigma_mae", "標準偏差予測の平均絶対誤差"), ("std_rms", "標準化残差の二乗平均平方根"), ("nll", "正規分布負対数尤度")]))
    lines.extend(["", "独立確認用データ別の値は次の通りである。", ""])
    end_holdout_table = [
        {
            "dataset": END_DATASET_LABELS.get(str(row["dataset"]), row["dataset"]),
            "model": row["label"],
            "roots": fmt_int(row["roots"]),
            "obs": fmt_int(row["observations"]),
            "sigma_rmse": fmt_float(row["sigma_rmse"], 4),
            "sigma_mae": fmt_float(row["sigma_mae"], 4),
            "std_rms": fmt_float(row["standardized_rms"], 4),
            "nll": fmt_float(row["gaussian_nll"], 4),
        }
        for row in end_holdout
    ]
    lines.append(markdown_table(end_holdout_table, [("dataset", "データ"), ("model", "モデル"), ("roots", "開始局面群"), ("obs", "誤差標本"), ("sigma_rmse", "標準偏差予測の二乗平均平方根誤差"), ("sigma_mae", "標準偏差予測の平均絶対誤差"), ("std_rms", "標準化残差の二乗平均平方根"), ("nll", "正規分布負対数尤度")]))

    end_cut_table = [
        {
            "dataset": (
                "新規独立確認・深さ19～26"
                if str(row["dataset"]) == "locked_new_deep19_26"
                else "2026年6月・深さ10～18"
            ),
            "model": row["label"],
            "level": row["level"],
            "selectivity": f"{number(row['selectivity_percent']):.2f}%",
            "contexts": fmt_int(row["contexts"]),
            "probes": fmt_int(row["probes"]),
            "cuts": fmt_int(row["cuts"]),
            "wrong": fmt_int(row["wrong_cuts"]),
            "wrong2": fmt_int(row["wrong_2plus"]),
            "wrong4": fmt_int(row["wrong_4plus"]),
            "ratio": fmt_float(row["estimated_node_ratio"], 4),
        }
        for row in end_cut_simulation
    ]
    lines.extend(
        [
            "",
            "完全読み値を使ってMPC判定を再現した結果を、選択率ごとに示す。推定訪問局面数比は、その判定機会だけを同じ条件で比較する、記録済みデータからの事後計算であり、実行ファイルで測った探索時間比ではない。",
            "",
            markdown_table(end_cut_table, [("dataset", "データ"), ("model", "モデル"), ("level", "選択率レベル"), ("selectivity", "選択率"), ("contexts", "判定機会"), ("probes", "浅い探索実行"), ("cuts", "枝刈り"), ("wrong", "誤った枝刈り"), ("wrong2", "2石以上"), ("wrong4", "4石以上"), ("ratio", "推定訪問局面数比")]),
            "",
        ]
    )

    generic_runtime_table = [
        {
            "label": row["label"],
            "runs": f"{row['completed']} / {row['runs']}",
            "timeout": fmt_int(row["timed_out"]),
            "control_nodes": fmt_int(row["control_nodes"]),
            "nodes": fmt_int(row["nodes"]),
            "node_sum": fmt_float(row["node_ratio_sum"], 4),
            "node_geo": fmt_float(row["node_ratio_geomean"], 4),
            "control_time": fmt_int(row["control_time_ms"]),
            "time": fmt_int(row["time_ms"]),
            "time_sum": fmt_float(row["time_ratio_sum"], 4),
            "time_geo": fmt_float(row["time_ratio_geomean"], 4),
            "control_nps": fmt_int(row["control_nps"]),
            "nps": fmt_int(row["nps"]),
            "nps_ratio": fmt_float(row["nps_ratio"], 4),
            "value": f"{row['exact_value_matches']} / {row['completed']}",
            "move": f"{row['exact_move_matches']} / {row['completed']}",
            "error": fmt_int(row["absolute_error_sum"]),
        }
        for row in end_generic_runtime
    ]
    lines.extend(
        [
            "",
            "#### 1.2.1 終盤汎用 a～f の実行ファイルによる比較",
            "",
            "30マス空き8局面と選択率74%・88%・93%の組合せ、合計24条件について、現行と候補を各1回ずつ実行した（計48実行）。1スレッドで、置換表は毎回空にした。現行・候補それぞれの合計訪問局面数と合計時間は、その側で完了した実行だけの合計であり、時間切れした実行の180秒は加えていない。総和比と比の幾何平均は、現行と候補の両方が完了した条件だけで計算した。各探索速度は、各側の完了分の合計訪問局面数を合計時間で割った値である。完全読み値・手の一致数の分母は、候補側で完了した実行数である。",
            "",
            markdown_table(generic_runtime_table, [("label", "設定"), ("runs", "候補完了 / 実行"), ("timeout", "候補時間切れ"), ("control_nodes", "現行完了分・合計訪問局面数"), ("nodes", "候補完了分・合計訪問局面数"), ("node_sum", "両方が完了した条件での訪問局面数総和比"), ("node_geo", "両方が完了した条件での訪問局面数比の幾何平均"), ("control_time", "現行完了分・合計時間（ミリ秒）"), ("time", "候補完了分・合計時間（ミリ秒）"), ("time_sum", "両方が完了した条件での時間総和比"), ("time_geo", "両方が完了した条件での時間比の幾何平均"), ("control_nps", "現行の探索速度（局面/秒）"), ("nps", "候補の探索速度（局面/秒）"), ("nps_ratio", "探索速度比"), ("value", "完全読み値一致 / 候補完了"), ("move", "完全読み手一致 / 候補完了"), ("error", "候補完了分・絶対誤差合計")]),
            "",
        ]
    )

    generic_fixed_time_table = [
        {
            "label": row["label"],
            "pairs": fmt_int(row["pairs"]),
            "wdl": f"{row['pair_wins']} / {row['pair_draws']} / {row['pair_losses']}",
            "score": fmt_percent(row["pair_score_rate"], 2),
            "ci": f"{fmt_percent(row['score_ci_low'], 2)} ～ {fmt_percent(row['score_ci_high'], 2)}",
            "disc": fmt_float(row["disc_diff_per_game"], 3),
        }
        for row in end_generic_fixed_time
    ]
    lines.extend(
        [
            "#### 1.2.2 終盤汎用 a～f を使う1手100ミリ秒対戦",
            "",
            "各候補を現行係数と比較した。各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20で、開始局面は番号5000～5255を使った。1組は同じ開始局面を先後交替で行う2局であり、勝ち・引き分け・負けは候補から見た組単位の数である。",
            "",
            markdown_table(generic_fixed_time_table, [("label", "設定"), ("pairs", "組数"), ("wdl", "組の勝ち / 引き分け / 負け"), ("score", "組スコア率"), ("ci", "復元抽出法による95%信頼区間"), ("disc", "1局当たり平均石差")]),
            "",
        ]
    )

    lines.extend(["", "## 2. MPCの浅い探索深度", "", "### 2.1 中盤・固定深度", ""])
    mid_depth_table = [
        {
            "source": display_measurement_source(str(row["source"])),
            "label": row["label"],
            "positions": fmt_int(row["positions"]),
            "nodes": fmt_int(row["nodes"]),
            "node_ratio": fmt_float(row["node_ratio"], 4),
            "time_ms": fmt_int(row["time_ms"]),
            "time_ratio": fmt_float(row["time_ratio"], 4),
            "nps": fmt_int(row["nps"]),
            "wrong2": fmt_int(row["regret_ge_2"]),
            "wrong4": fmt_int(row["regret_ge_4"]),
            "mean_regret": fmt_float(row["mean_regret"], 3),
            "agreement": fmt_percent(row["agreement"], 2),
        }
        for row in mid_depth
    ]
    lines.extend(
        [
            "100局面確認の条件は中盤深さ16手、100局面×選択率6段階、1スレッド、各探索の開始時に置換表を空にした測定である。開発用・確認用の測定結果が完成した場合は、データ別の行を同じ表へ追加する。比率は各測定群・各データ内の現行値を1とした。実際の局面×選択率数は表に示す。訪問局面数と探索時間は全探索の合計であり、探索速度は合計訪問局面数を合計探索時間で割った値である。",
            "この固定深度表と次の固定時間対戦は、浅い探索深度だけを変え、既存の a～g 係数を共通に使った初期測定である。1.1節の候補別再推定係数はまだ反映していない。",
            "",
            markdown_table(mid_depth_table, [("source", "測定群"), ("label", "設定"), ("positions", "局面×選択率"), ("nodes", "合計訪問局面数"), ("node_ratio", "訪問局面数比"), ("time_ms", "合計探索時間（ミリ秒）"), ("time_ratio", "時間比"), ("nps", "探索速度（局面/秒）"), ("wrong2", "2石以上の着手損失"), ("wrong4", "4石以上の着手損失"), ("mean_regret", "平均着手損失"), ("agreement", "完全読みとの着手一致率")]),
            "",
            "元データは[中盤の浅い探索深度に関する測定結果](screen_depth_ggs100/summary.csv)にある。",
            "",
            "### 2.2 中盤・1手100ミリ秒の2局1組対戦",
            "",
        ]
    )
    fixed_time_table = [
        {
            "label": row["label"],
            "pairs": fmt_int(row["pairs"]),
            "wdl": f"{row['pair_wins']} / {row['pair_draws']} / {row['pair_losses']}",
            "score": fmt_percent(row["pair_score_rate"], 2),
            "ci": f"{fmt_percent(row['score_ci_low'], 2)} ～ {fmt_percent(row['score_ci_high'], 2)}",
            "disc": fmt_float(row["disc_diff_per_game"], 3),
        }
        for row in mid_fixed_time
    ]
    lines.extend(
        [
            "各候補を現行と比較した。各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20で、開始局面は番号5000～5255を使った。1組は同じ開始局面の先後を入れ替えた2局であり、表の勝ち・引き分け・負けは組単位である。",
            "",
            markdown_table(fixed_time_table, [("label", "設定"), ("pairs", "組数"), ("wdl", "組の勝ち / 引き分け / 負け"), ("score", "組スコア率"), ("ci", "復元抽出法による95%信頼区間"), ("disc", "1局当たり平均石差")]),
            "",
            "### 2.3 終盤深さ10～18手・候補別の再学習による事後計算",
            "",
            "この表は浅い探索深度を変えた後、各候補専用の標準偏差・上下分位点・整数誤差幅を再学習した結果である。現行モデルへ深度差だけを与えた比較ではない。推定訪問局面数比の浅い探索部分は、置換表を空にした通常窓探索の訪問局面数である。実際のMPCはゼロ窓探索を使うため、実行ファイルによる測定表と分けて読む必要がある。",
            "",
        ]
    )
    end_offline_table = [
        {
            "label": row["label"],
            "cv": fmt_int(row["cv_contexts"]),
            "cv_cuts": fmt_int(row["cv_cuts"]),
            "cv_wrong": fmt_int(row["cv_wrong"]),
            "cv_wrong2": fmt_int(row["cv_wrong_2"]),
            "cv_wrong4": fmt_int(row["cv_wrong_4"]),
            "cv_ratio": fmt_float(row["cv_node_ratio"], 4),
            "ho": fmt_int(row["holdout_contexts"]),
            "ho_cuts": fmt_int(row["holdout_cuts"]),
            "ho_wrong": fmt_int(row["holdout_wrong"]),
            "ho_wrong2": fmt_int(row["holdout_wrong_2"]),
            "ho_wrong4": fmt_int(row["holdout_wrong_4"]),
            "ho_ratio": fmt_float(row["holdout_node_ratio"], 4),
        }
        for row in end_depth_offline
    ]
    lines.extend(
        [
            markdown_table(end_offline_table, [("label", "設定"), ("cv", "交差検証の判定機会"), ("cv_cuts", "交差検証の枝刈り"), ("cv_wrong", "交差検証の誤った枝刈り"), ("cv_wrong2", "同2石以上"), ("cv_wrong4", "同4石以上"), ("cv_ratio", "交差検証の推定訪問局面数比"), ("ho", "6月確認の判定機会"), ("ho_cuts", "6月確認の枝刈り"), ("ho_wrong", "6月確認の誤った枝刈り"), ("ho_wrong2", "同2石以上"), ("ho_wrong4", "同4石以上"), ("ho_ratio", "6月確認の推定訪問局面数比")]),
            "",
            "下表の分位点と誤差幅は、選択率74% / 88% / 93%の順である。標準偏差・上下分位点・整数誤差幅は、候補ごと、深い探索深度ごとに別々に再推定している。",
            "",
            markdown_table(
                [
                    {
                        "label": row["label"],
                        "deep": row["deep_depth"],
                        "shallow": row["shallow_depth"],
                        "sigma": fmt_float(row["sigma"], 6),
                        "lower": row["lower_tails"],
                        "upper": row["upper_tails"],
                        "high": row["high_errors"],
                        "low": row["low_errors"],
                    }
                    for row in end_depth_parameters
                ],
                [("label", "設定"), ("deep", "深い探索深度"), ("shallow", "浅い探索深度"), ("sigma", "標準偏差 σ"), ("lower", "下側分位点"), ("upper", "上側分位点"), ("high", "上側整数誤差幅"), ("low", "下側整数誤差幅")],
            ),
            "",
            "丸め前の値と一覧は[終盤深さ10～18手の再学習パラメータ](end_offline_refit/parameters_with_admission.csv)にある。",
            "",
            "### 2.4 終盤・30マス空き8局面の実行ファイルによる比較",
            "",
            "30マス空き8局面と選択率74%・88%・93%の組合せ、合計24条件について、現行と候補を各1回ずつ実行した（計48実行）。条件は1スレッドで、置換表は毎回空にした。現行・候補それぞれの合計訪問局面数と合計時間は、その側で完了した実行だけの合計であり、時間切れした実行の180秒は加えていない。総和比と比の幾何平均は、現行と候補の両方が完了した条件だけで計算した。各探索速度は、各側の完了分の合計訪問局面数を合計時間で割った値である。完全読み値・手の一致数の分母は、候補側で完了した実行数である。",
            "",
        ]
    )
    runtime_table = [
        {
            "label": row["label"],
            "runs": f"{row['completed']} / {row['runs']}",
            "timeout": fmt_int(row["timed_out"]),
            "control_nodes": fmt_int(row["control_nodes"]),
            "nodes": fmt_int(row["nodes"]),
            "node_sum": fmt_float(row["node_ratio_sum"], 4),
            "node_geo": fmt_float(row["node_ratio_geomean"], 4),
            "control_time": fmt_int(row["control_time_ms"]),
            "time": fmt_int(row["time_ms"]),
            "time_sum": fmt_float(row["time_ratio_sum"], 4),
            "time_geo": fmt_float(row["time_ratio_geomean"], 4),
            "control_nps": fmt_int(row["control_nps"]),
            "nps": fmt_int(row["nps"]),
            "nps_ratio": fmt_float(row["nps_ratio"], 4),
            "value": f"{row['exact_value_matches']} / {row['completed']}",
            "move": f"{row['exact_move_matches']} / {row['completed']}",
            "error": fmt_int(row["absolute_error_sum"]),
        }
        for row in end_depth_runtime
    ]
    lines.append(markdown_table(runtime_table, [("label", "設定"), ("runs", "候補完了 / 実行"), ("timeout", "候補時間切れ"), ("control_nodes", "現行完了分・合計訪問局面数"), ("nodes", "候補完了分・合計訪問局面数"), ("node_sum", "両方が完了した条件での訪問局面数総和比"), ("node_geo", "両方が完了した条件での訪問局面数比の幾何平均"), ("control_time", "現行完了分・合計時間（ミリ秒）"), ("time", "候補完了分・合計時間（ミリ秒）"), ("time_sum", "両方が完了した条件での時間総和比"), ("time_geo", "両方が完了した条件での時間比の幾何平均"), ("control_nps", "現行の探索速度（局面/秒）"), ("nps", "候補の探索速度（局面/秒）"), ("nps_ratio", "探索速度比"), ("value", "完全読み値一致 / 候補完了"), ("move", "完全読み手一致 / 候補完了"), ("error", "候補完了分・絶対誤差合計")]))

    end_depth_fixed_time_table = [
        {
            "label": row["label"],
            "pairs": fmt_int(row["pairs"]),
            "wdl": f"{row['pair_wins']} / {row['pair_draws']} / {row['pair_losses']}",
            "score": fmt_percent(row["pair_score_rate"], 2),
            "ci": f"{fmt_percent(row['score_ci_low'], 2)} ～ {fmt_percent(row['score_ci_high'], 2)}",
            "disc": fmt_float(row["disc_diff_per_game"], 3),
        }
        for row in end_depth_fixed_time
    ]
    lines.extend(
        [
            "",
            "### 2.5 終盤深さ10～18手専用モデル・1手100ミリ秒の2局1組対戦",
            "",
            "各候補を現行と比較した。各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20で、開始局面は番号5000～5255を使った。浅い探索深度だけでなく、その候補専用に再推定した標準偏差・上下分位点・整数誤差幅も同時に使っている。勝ち・引き分け・負けは候補から見た組単位の数である。",
            "",
            markdown_table(end_depth_fixed_time_table, [("label", "設定"), ("pairs", "組数"), ("wdl", "組の勝ち / 引き分け / 負け"), ("score", "組スコア率"), ("ci", "復元抽出法による95%信頼区間"), ("disc", "1局当たり平均石差")]),
        ]
    )

    cold_overall_table = [
        {
            "label": row["label"],
            "cases": fmt_int(row["count"]),
            "completed": f"{row['completed']} / {row['count']}",
            "rate": fmt_percent(row["completion_rate"], 1),
            "exact": fmt_int(row["exact_completed"]),
            "first": fmt_int(row["first_end_time_median_ms"]),
            "p90": fmt_int(row["first_end_time_p90_ms"]),
            "depth_median": fmt_float(row["result_depth_median"], 1),
            "depth_max": fmt_int(row["result_depth_max"]),
            "nodes": fmt_int(row["valid_nodes_median"]),
            "nps": fmt_int(row["nps_median"]),
            "cores": fmt_float(row["cpu_average_cores_median"], 2),
            "cpu": fmt_float(row["cpu_utilization_percent_median"], 1),
        }
        for row in cold_end_overall
    ]
    cold_by_empty_table = [
        {
            "label": row["label"],
            "empties": row["empties"],
            "completed": f"{row['completed']} / {row['count']}",
            "exact": fmt_int(row["exact_completed"]),
            "first": fmt_int(row["first_end_time_median_ms"]),
            "depth_median": fmt_float(row["result_depth_median"], 1),
            "depth_max": fmt_int(row["result_depth_max"]),
            "nodes": fmt_int(row["valid_nodes_median"]),
            "nps": fmt_int(row["nps_median"]),
        }
        for row in cold_end_by_empty
    ]
    lines.extend(
        [
            "",
            "### 2.6 実戦相当局面・20スレッド・1手15秒",
            "",
            "32～44マス空きについて各5局面、合計65局面を使う。各局面は新しいプロセスで開始するため置換表は空である。`読み切り` は選択率74%以上の終盤探索が時間内に完了した件数、`完全読み` は選択率100%が完了した件数である。",
            "",
            markdown_table(cold_overall_table, [("label", "設定"), ("cases", "局面数"), ("completed", "読み切り / 局面"), ("rate", "読み切り率"), ("exact", "完全読み"), ("first", "最初の読み切り時刻の中央値（ミリ秒）"), ("p90", "同90%点（ミリ秒）"), ("depth_median", "最終探索深度中央値"), ("depth_max", "最終探索深度最大"), ("nodes", "訪問局面数中央値"), ("nps", "探索速度中央値（局面/秒）"), ("cores", "使用コア数中央値"), ("cpu", "指定20スレッドに対するCPU使用率中央値（%）")]),
            "",
            "マス空き数別の内訳は次の通りである。",
            "",
            markdown_table(cold_by_empty_table, [("label", "設定"), ("empties", "マス空き"), ("completed", "読み切り / 局面"), ("exact", "完全読み"), ("first", "最初の読み切り時刻の中央値（ミリ秒）"), ("depth_median", "最終探索深度中央値"), ("depth_max", "最終探索深度最大"), ("nodes", "訪問局面数中央値"), ("nps", "探索速度中央値（局面/秒）")]),
        ]
    )

    lines.extend(
        [
            "",
            "## 3. 分位点 z へ固定倍率を掛けた参考測定（係数再推定ではない）",
            "",
            "この章だけは標準偏差モデルの係数を変えていない。現行の分位点 z へ全体で同じ倍率を掛けたときの感度を見る参考表であり、1章の係数再推定とは別実験である。",
            "この固定倍率機能は現在の本体ソースには残していない。この章は測定済みファイルの記録だけであり、本体の現在状態を表すものではない。",
            "",
            "### 3.1 中盤・固定深度",
            "",
        ]
    )
    mid_z_table = [
        {
            "source": display_measurement_source(str(row["source"])),
            "label": row["label"],
            "positions": fmt_int(row["positions"]),
            "nodes": fmt_int(row["nodes"]),
            "node_ratio": fmt_float(row["node_ratio"], 4),
            "time": fmt_int(row["time_ms"]),
            "time_ratio": fmt_float(row["time_ratio"], 4),
            "nps": fmt_int(row["nps"]),
            "wrong2": fmt_int(row["regret_ge_2"]),
            "wrong4": fmt_int(row["regret_ge_4"]),
            "mean": fmt_float(row["mean_regret"], 3),
        }
        for row in mid_z
    ]
    lines.extend(
        [
            "条件は中盤深さ16手、選択率6段階、1スレッド、各探索の開始時に置換表を空にした固定深度測定である。`z=0.98`は現行の分位点 z の98%を使うという意味である。測定群は別々に実行しているため、訪問局面数比・時間比は各群内の現行値を1としている。訪問局面数と探索時間は各測定群の全探索の合計であり、探索速度は合計訪問局面数を合計探索時間で割った値である。",
            "",
            markdown_table(mid_z_table, [("source", "測定群"), ("label", "設定"), ("positions", "局面×選択率"), ("nodes", "合計訪問局面数"), ("node_ratio", "訪問局面数比"), ("time", "合計探索時間（ミリ秒）"), ("time_ratio", "時間比"), ("nps", "探索速度（局面/秒）"), ("wrong2", "2石以上の着手損失"), ("wrong4", "4石以上の着手損失"), ("mean", "平均着手損失")]),
            "",
            "### 3.2 終盤深さ10～18手・記録済みデータによる参考値",
            "",
        ]
    )
    end_z_dataset_labels = {
        "cv_total": "開発データの交差検証合計",
        "holdout202606": "2026年6月の独立確認用データ",
        "admission_holdout": "追加の独立確認用データ",
    }
    end_z_table = [
        {
            "dataset": end_z_dataset_labels.get(str(row["dataset"]), row["dataset"]),
            "scale": fmt_float(row["z_scale"], 2),
            "contexts": fmt_int(row["contexts"]),
            "cuts": fmt_int(row["cuts"]),
            "wrong": fmt_int(row["wrong_cuts"]),
            "wrong2": fmt_int(row["wrong_2plus"]),
            "wrong4": fmt_int(row["wrong_4plus"]),
            "nodes": fmt_int(row["simulated_nodes"]),
            "ratio": fmt_float(row["node_ratio"], 4),
        }
        for row in end_z
    ]
    lines.extend(
        [
            "浅い探索深度は全候補で現行と同じである。分位点 z の倍率ごとに丸め前の残差分位点を取り直し、整数誤差幅を再計算した。ここでも係数は共通である。",
            "",
            markdown_table(end_z_table, [("dataset", "データ"), ("scale", "分位点 z の倍率"), ("contexts", "判定機会"), ("cuts", "枝刈り"), ("wrong", "誤った枝刈り"), ("wrong2", "2石以上"), ("wrong4", "4石以上"), ("nodes", "推定訪問局面数"), ("ratio", "推定訪問局面数比")]),
            "",
            "確率への変換と整数誤差幅は[終盤の固定倍率に関する詳細レポート](end_aggression_refit/report.md)にある。",
            "",
            "## 4. 測定の完了状況",
            "",
        ]
    )
    status_table = [
        {"measurement": row["measurement"], "status": row["status"], "required": row["required_output"]}
        for row in status
    ]
    lines.extend(
        [
            markdown_table(status_table, [("measurement", "測定"), ("status", "状態"), ("required", "完了条件")]),
            "",
            "## 5. 再生成",
            "",
            "```powershell",
            "python benchmark/mpc_depth_aggressiveness_20260830/aggregate_results.py",
            "```",
            "",
            "上のコマンドは既存の測定結果を読み、入力条件と件数を検査してから、集計用の表形式ファイルとこのレポートを更新する。追加の測定結果が作成された場合も、同じコマンドで表へ反映される。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="MPC 比較結果を日本語 Markdown 表へ集約する")
    parser.add_argument("--base", type=Path, default=BASE_DIR, help="実験ディレクトリ")
    parser.add_argument("--repo", type=Path, default=REPO_DIR, help="リポジトリルート")
    parser.add_argument("--output", type=Path, default=None, help="集計 CSV の出力先")
    parser.add_argument("--report", type=Path, default=None, help="Markdown レポートの出力先")
    args = parser.parse_args()

    base_dir = args.base.resolve()
    repo_dir = args.repo.resolve()
    output_dir = (args.output or (base_dir / "generated")).resolve()
    report_path = (args.report or (base_dir / "report.md")).resolve()

    mid_model_depth = []
    for path in sorted(base_dir.rglob("mid_model_runtime*/summary.csv")):
        mid_model_depth.extend(
            aggregate_mid_model_accuracy(path, path.parent.name)
        )
    mid_model_depth.sort(key=lambda row: variant_sort_key(str(row["variant"])))
    mid_model_fixed_time = collect_fixed_time_mid_model(repo_dir)

    mid_depth = []
    mid_depth.extend(aggregate_accuracy(base_dir / "screen_depth_ggs100" / "summary.csv", "100局面確認"))
    mid_depth.extend(aggregate_accuracy(base_dir / "matrix_depth_dev_validation" / "summary.csv", "開発・確認行列"))
    mid_fixed_time = collect_fixed_time_depth(repo_dir)
    end_depth_offline = collect_end_depth_offline(base_dir)
    end_depth_parameters = collect_end_depth_parameters(base_dir)
    end_depth_runtime = collect_end_depth_runtime(base_dir)
    end_generic_runtime = collect_end_generic_runtime(base_dir)
    end_depth_fixed_time, end_generic_fixed_time = collect_fixed_time_end_models(
        repo_dir
    )
    cold_end_overall, cold_end_by_empty = collect_cold_end_runtime(base_dir)

    mid_z = []
    mid_z.extend(aggregate_accuracy(base_dir / "screen_z_ggs100" / "summary.csv", "z 0.90～1.00・100局面確認"))
    mid_z.extend(aggregate_accuracy(base_dir / "screen_z_aggressive_ggs100" / "summary.csv", "z 0.80～0.88・100局面確認"))
    mid_z.extend(aggregate_accuracy(base_dir / "matrix_z_dev_validation" / "summary.csv", "z 0.90～1.00・開発/確認行列"))
    end_z = collect_end_z_reference(base_dir)

    mid_coeffs, mid_cv, mid_metadata = collect_mid_coefficients(base_dir)
    end_coeffs, end_cv, end_holdout, end_domain, end_cut_simulation = (
        collect_end_generic(base_dir)
    )
    status = build_status(
        base_dir,
        repo_dir,
        mid_metadata,
        mid_model_depth,
        mid_model_fixed_time,
        end_depth_runtime,
        end_depth_fixed_time,
        end_generic_runtime,
        end_generic_fixed_time,
        cold_end_overall,
        cold_end_by_empty,
    )

    accuracy_fields = ["source", "variant", "label", "runs", "positions", "depths", "reference_depths", "mpc_levels", "thread_counts", "hash_levels", "repetitions", "nodes", "node_ratio", "time_ms", "time_ratio", "nps", "regret_ge_2", "regret_ge_4", "mean_regret", "agreement"]
    write_csv(output_dir / "mid_model_fixed_depth.csv", mid_model_depth, accuracy_fields)
    fixed_time_fields = ["variant", "label", "pairs", "games", "move_time_ms", "threads", "hash_level", "opening_start", "pair_wins", "pair_draws", "pair_losses", "pair_score_rate", "score_ci_low", "score_ci_high", "disc_diff_per_game", "error_count", "early_exit_count", "source"]
    write_csv(output_dir / "mid_model_fixed_time_pairs.csv", mid_model_fixed_time, fixed_time_fields)
    write_csv(output_dir / "mid_shallow_depth_fixed_depth.csv", mid_depth, accuracy_fields)
    write_csv(output_dir / "mid_z_multiplier_reference.csv", mid_z, accuracy_fields)
    write_csv(output_dir / "mid_shallow_depth_fixed_time_pairs.csv", mid_fixed_time, fixed_time_fields)
    write_csv(output_dir / "end_shallow_depth_offline.csv", end_depth_offline, ["variant", "label", "cv_contexts", "cv_cuts", "cv_wrong", "cv_wrong_2", "cv_wrong_4", "cv_node_ratio", "holdout_contexts", "holdout_cuts", "holdout_wrong", "holdout_wrong_2", "holdout_wrong_4", "holdout_node_ratio"])
    write_csv(output_dir / "end_shallow_depth_parameters.csv", end_depth_parameters, ["variant", "label", "deep_depth", "shallow_depth", "sigma", "lower_tails", "upper_tails", "high_errors", "low_errors"])
    end_runtime_fields = ["variant", "label", "runs", "completed", "timed_out", "nodes", "control_nodes", "node_ratio_sum", "node_ratio_geomean", "time_ms", "control_time_ms", "time_ratio_sum", "time_ratio_geomean", "control_nps", "nps", "nps_ratio", "exact_value_matches", "exact_move_matches", "absolute_error_sum", "count", "repetitions", "levels", "threads", "hash_level", "timeout_seconds", "cold_tt", "source"]
    write_csv(output_dir / "end_shallow_depth_runtime.csv", end_depth_runtime, end_runtime_fields)
    write_csv(output_dir / "end_generic_model_runtime.csv", end_generic_runtime, end_runtime_fields)
    write_csv(output_dir / "end_shallow_depth_fixed_time_pairs.csv", end_depth_fixed_time, fixed_time_fields)
    write_csv(output_dir / "end_generic_model_fixed_time_pairs.csv", end_generic_fixed_time, fixed_time_fields)
    cold_fields = ["variant", "label", "empties", "count", "valid", "attempted", "completed", "completion_rate", "exact_completed", "first_end_time_median_ms", "first_end_time_p90_ms", "nodes_median", "valid_nodes_median", "nps_median", "completed_nps_median", "end_start_time_median_ms", "end_search_time_median_ms", "cpu_average_cores_median", "cpu_utilization_percent_median", "result_depth_median", "result_depth_max", "threads", "movetime_ms", "hash_level", "min_empty", "max_empty", "max_per_empty", "sampling_seed", "extraction_error_count", "required_selectivity", "case_timeout_seconds", "cold_tt", "source"]
    write_csv(output_dir / "cold_end_runtime_overall.csv", cold_end_overall, cold_fields)
    write_csv(output_dir / "cold_end_runtime_by_empty.csv", cold_end_by_empty, cold_fields)
    write_csv(output_dir / "end_z_multiplier_reference.csv", end_z, ["dataset", "candidate", "z_scale", "contexts", "cuts", "wrong_cuts", "wrong_2plus", "wrong_4plus", "simulated_nodes", "node_ratio"])
    write_csv(output_dir / "mid_model_coefficients.csv", mid_coeffs, ["model", "label", "shallow_offset", "ridge", "a", "b", "c", "d", "e", "f", "g"])
    write_csv(output_dir / "mid_model_cv.csv", mid_cv, ["offset", "model", "level", "rows", "roots", "nll", "normalized_rms", "margin_mean", "wrong_cuts", "wrong_cut_rate", "excess_ge_2", "excess_ge_4", "estimated_node_ratio"])
    write_csv(output_dir / "end_generic_model_coefficients.csv", end_coeffs, ["model", "label", "representation", "a", "b", "c", "d", "e", "f", "global_sigma_multiplier"])
    write_csv(output_dir / "end_generic_model_cv.csv", end_cv, ["model", "label", "folds", "roots", "observations", "sigma_rmse", "sigma_mae", "standardized_rms", "gaussian_nll"])
    write_csv(output_dir / "end_generic_model_holdout.csv", end_holdout, ["dataset", "model", "label", "roots", "observations", "sigma_rmse", "sigma_mae", "standardized_rms", "gaussian_nll"])
    write_csv(output_dir / "end_generic_model_domain_audit.csv", end_domain, ["model", "label", "table_entries", "minimum_raw_sigma", "minimum_n_discs", "minimum_shallow_depth", "maximum_raw_sigma", "nonpositive_entries", "below_0_5_entries"])
    write_csv(output_dir / "end_generic_model_cut_simulation.csv", end_cut_simulation, ["dataset", "model", "label", "level", "selectivity_percent", "contexts", "probes", "cuts", "wrong_cuts", "wrong_2plus", "wrong_4plus", "estimated_node_ratio"])
    write_csv(output_dir / "measurement_status.csv", status, ["measurement", "status", "required_output"])

    report = render_report(
        mid_model_depth=mid_model_depth,
        mid_model_fixed_time=mid_model_fixed_time,
        mid_depth=mid_depth,
        mid_fixed_time=mid_fixed_time,
        end_depth_offline=end_depth_offline,
        end_depth_parameters=end_depth_parameters,
        end_depth_runtime=end_depth_runtime,
        end_depth_fixed_time=end_depth_fixed_time,
        mid_z=mid_z,
        end_z=end_z,
        mid_coeffs=mid_coeffs,
        mid_cv=mid_cv,
        mid_metadata=mid_metadata,
        end_coeffs=end_coeffs,
        end_cv=end_cv,
        end_holdout=end_holdout,
        end_domain=end_domain,
        end_cut_simulation=end_cut_simulation,
        end_generic_runtime=end_generic_runtime,
        end_generic_fixed_time=end_generic_fixed_time,
        cold_end_overall=cold_end_overall,
        cold_end_by_empty=cold_end_by_empty,
        status=status,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    print(f"wrote {report_path}")
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
