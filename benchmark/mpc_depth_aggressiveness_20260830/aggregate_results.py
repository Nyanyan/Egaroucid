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
    text_columns = {
        "label", "source", "dataset", "model", "status", "measurement",
        "required", "representation", "term", "strict", "plain",
    }
    separator = "|" + "|".join(
        "---:" if key not in text_columns else "---" for key, _ in columns
    ) + "|"
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
        reference_weight_match = re.search(r"_prior([0-9.]+)$", model)
        coeffs.append(
            {
                "model": model,
                "label": END_GENERIC_MODEL_LABELS.get(model, model),
                "representation": (
                    "三次式をそのまま使用"
                    if model == "current"
                    else "q₀～q₃を0.5以上に制約して計算し、c～fへ変換"
                ),
                "global_sigma_multiplier": number(
                    row.get("global_sigma_multiplier"), 1.0
                ),
                "reference_weight": (
                    number(reference_weight_match.group(1))
                    if reference_weight_match else ""
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
            "各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20、開始番号5000、実行失敗なし",
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
            "4候補について各256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20、開始番号5000、実行失敗なし",
        ),
        (
            "終盤の a～f 係数再推定候補：1手100ミリ秒の2局1組対戦",
            "完了" if end_generic_fixed_time_complete else "未完了",
            "2候補について各256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20、開始番号5000、実行失敗なし",
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


def build_report_intro() -> list[str]:
    """Return the reader-facing summary and glossary used by the report."""
    search_terms = [
        {
            "term": "評価値",
            "strict": "探索または評価関数が返す、石差単位の数値。同じ局面・同じ手番から比較し、値が大きい手を上位とする。",
            "plain": "ある手が別の手より何石分よいと計算されたかを表す数値。",
        },
        {
            "term": "石数・マス空き",
            "strict": "石数は盤上で黒石または白石が置かれているマスの合計。マス空きは `64−石数`。",
            "plain": "ゲームがどこまで進んだかを表す。マス空きが少ないほど終局に近い。",
        },
        {
            "term": "探索深度",
            "strict": "探索が先の着手を再帰的に調べる回数を指定する整数。深度を1増やすと、原則として着手を1回先まで追加して調べる。",
            "plain": "何手先まで調べるかを表す数値。",
        },
        {
            "term": "静的評価",
            "strict": "子局面を生成する探索を行わず、評価関数を現在局面へ1回適用して評価値を得る処理。",
            "plain": "先を読まず、現在の盤面だけで点数を付ける処理。",
        },
        {
            "term": "探索窓 `[α, β]`",
            "strict": "アルファ・ベータ探索へ渡す下限 `α` と上限 `β`。探索結果を `α` 以下、`α` と `β` の間、`β` 以上のいずれかとして扱う。",
            "plain": "評価値の全範囲ではなく、判定に必要な境界だけを指定する仕組み。",
        },
        {
            "term": "通常窓探索",
            "strict": "`β−α` が1より大きい探索窓を使い、窓内の評価値も求める探索。",
            "plain": "境界を超えたかだけでなく、境界の間にある評価値も調べる探索。",
        },
        {
            "term": "ゼロ窓探索",
            "strict": "整数評価で `β=α+1` とし、評価値が境界以下か境界以上かだけを判定する探索。",
            "plain": "指定した境界を超えるかだけを調べる探索。",
        },
        {
            "term": "Multi-ProbCut（MPC）",
            "strict": "浅い探索値と誤差モデルから深い探索値が探索窓の外にあると判定した場合に、その深い探索を実行せず上限値または下限値を返す処理。",
            "plain": "短い探索で境界を超えると判断できた場合に、長い探索を省略する処理。",
        },
        {
            "term": "深い探索深度",
            "strict": "MPCを使わない場合に、その探索呼出しが実行する予定の探索深度。",
            "plain": "MPCが省略できるか判定する対象の探索深度。",
        },
        {
            "term": "浅い探索深度",
            "strict": "MPC判定の入力値を得るため、深い探索より先に実行する探索の深度。",
            "plain": "深い探索を省略できるか調べるために先に読む深度。",
        },
        {
            "term": "MPC判定機会",
            "strict": "局面、探索窓、深い探索深度、浅い探索深度、選択率が1組与えられ、MPC条件を1回評価したこと。",
            "plain": "MPCが深い探索を省略するか判断した回数。",
        },
        {
            "term": "枝刈り",
            "strict": "MPC判定により深い探索を実行せず、評価値が探索窓の上側または下側にあるという境界値を返すこと。",
            "plain": "深い探索を省略したこと。",
        },
        {
            "term": "誤った枝刈り",
            "strict": "上側へ枝刈りしたのに完全読み値が上側境界未満だった場合、または下側へ枝刈りしたのに完全読み値が下側境界を超えていた場合。",
            "plain": "省略してよいというMPCの判定が完全読みと矛盾した回数。",
        },
        {
            "term": "誤った枝刈りの割合",
            "strict": "`誤った枝刈り回数÷枝刈り回数×100`。枝刈り回数が0の場合は0とする。",
            "plain": "深い探索を省略した全回数のうち、完全読みと矛盾した割合。",
        },
        {
            "term": "予測誤差幅を2石以上・4石以上超過",
            "strict": "残差を `r`、予測した整数誤差幅を `m` とし、`max(0,abs(r)−m)` が2以上または4以上だった誤差標本数。",
            "plain": "浅い探索の実際の誤差が、予測した範囲を2石以上または4石以上超えた回数。",
        },
        {
            "term": "誤った枝刈りが境界から2石以上・4石以上",
            "strict": "上側の誤枝刈りでは `上側境界−完全読み値`、下側の誤枝刈りでは `完全読み値−下側境界` が2以上または4以上だった回数。",
            "plain": "MPCの誤判定が探索窓の境界から2石以上または4石以上離れていた回数。",
        },
        {
            "term": "浅い探索実行",
            "strict": "MPC判定機会のうち、静的評価による事前条件を満たし、MPC判定用の浅い探索を実際に呼び出した回数。",
            "plain": "MPC判定のための浅い探索を実行した回数。",
        },
        {
            "term": "訪問局面数",
            "strict": "探索関数が処理した局面の延べ回数。実装内部の `nodes` と同じ値。",
            "plain": "探索中に計算した局面数。通常は少ないほど計算量が少ない。",
        },
        {
            "term": "探索速度",
            "strict": "`合計訪問局面数÷合計探索秒数`。単位は局面/秒。",
            "plain": "1秒に何局面を処理したか。大きいほど同じ時間で多く計算している。",
        },
        {
            "term": "置換表",
            "strict": "探索済み局面を盤面識別値で検索し、その評価値・探索深度・境界種別・着手を再利用する表。",
            "plain": "同じ局面を再計算しないための探索結果の保存領域。",
        },
        {
            "term": "置換表容量設定",
            "strict": "実行ファイルが置換表の要素数を決めるために受け取る整数設定値。",
            "plain": "探索結果を何件保存できるかを決める実行設定。",
        },
        {
            "term": "スレッド数・使用コア数・中央処理装置使用率",
            "strict": "スレッド数は並列に実行可能な処理単位数。使用コア数は測定中に計算へ使った物理コア数の時間平均。使用率は `使用コア数÷指定スレッド数×100`。",
            "plain": "並列計算をいくつ指定し、そのうち平均していくつ分を実際に使ったかを表す。",
        },
    ]

    model_terms = [
        {
            "term": "現行・候補",
            "strict": "現行は比較基準の実行ファイルまたは設定。候補は比較する変更後の実行ファイルまたは設定。",
            "plain": "表では現行を基準値1として、変更後が増減した量を読む。",
        },
        {
            "term": "モデル・係数",
            "strict": "モデルは入力から予測値を計算する式。係数はその式へ代入する固定値で、この文書では `a`～`g` などで表す。",
            "plain": "モデルは計算式、係数は同じ入力に対する計算結果を決める数値。",
        },
        {
            "term": "係数再推定・学習",
            "strict": "収集した誤差標本を目的関数へ代入し、その目的関数を最小にする係数を数値計算で求める処理。",
            "plain": "実測した誤差に合うように、計算式の数値を決め直す処理。",
        },
        {
            "term": "重み `wᵢ`",
            "strict": "目的関数の第 `i` 標本の項へ掛ける非負の数値。重み付き平均は `Σwᵢxᵢ÷Σwᵢ`。",
            "plain": "各標本を係数計算へどれだけ反映するかを指定する数値。",
        },
        {
            "term": "残差 `rᵢ`",
            "strict": "第 `i` 標本の `深い探索値−浅い探索値`。この実験では残差の平均を0と仮定する。",
            "plain": "浅い探索値が深い探索値から何石ずれたか。",
        },
        {
            "term": "標準偏差 `σᵢ`",
            "strict": "第 `i` 標本についてモデルが予測する残差の二乗平均平方根。常に正の値を使う。",
            "plain": "その条件で浅い探索値が通常どの程度ずれると予測したか。",
        },
        {
            "term": "標準偏差モデル",
            "strict": "石数、浅い探索深度、深い探索深度を入力し、予測標準偏差 `σ` を返す式と係数の組。各節に実際の式を記載する。",
            "plain": "盤面段階と探索深度から、浅い探索の誤差量を予測する計算式。",
        },
        {
            "term": "数式記号 `Σ`・`abs`・`sqrt`・`ceil`・`min`・`max`・`exp`",
            "strict": "`Σ` は指定範囲の合計。`abs(x)` はxの絶対値。`sqrt(x)` は二乗するとxになる非負値。`ceil(x)` はx以上の最小整数。`min` と `max` は引数の最小値と最大値。`exp(y)` は `ln(x)=y` を満たす正のx。",
            "plain": "この文書の式で使う、合計・絶対値・平方根・切上げ・最小・最大・指数の計算。",
        },
        {
            "term": "自然対数 `ln(x)`",
            "strict": "正の `x` に対する `1` から `x` までの `1/t` の定積分。",
            "plain": "目的関数で使用する数学関数。",
        },
        {
            "term": "正規分布 `N(0, σ²)`",
            "strict": "残差 `r` の確率密度を `exp(−r²÷(2σ²))÷(σ×sqrt(2π))` とする、平均0・分散 `σ²` の確率分布。`π` は円周率。",
            "plain": "0に近い誤差ほど生じやすいという仮定。",
        },
        {
            "term": "標準正規分布",
            "strict": "平均0、標準偏差1の正規分布 `N(0,1)`。",
            "plain": "誤差を標準偏差で割った後に使う共通の確率分布。",
        },
        {
            "term": "分位点 `z`",
            "strict": "標準正規分布に従う変数を `Z` としたとき、指定確率 `q` に対して `P(Z≤z)=q` を満たす値。",
            "plain": "選択率を石差の判定幅へ変換するための数値。",
        },
        {
            "term": "上側・下側分位点",
            "strict": "標準化残差 `r÷σ` を昇順に並べ、指定した上側確率と下側確率に対応する位置から別々に求めた値。",
            "plain": "浅い探索が大き過ぎる場合と小さ過ぎる場合の誤差量を別々に表す数値。",
        },
        {
            "term": "選択率",
            "strict": "MPC判定に使う中央確率。実装の各段階では対応する分位点 `z` が定まり、100%ではMPCを実行しない。",
            "plain": "大きくすると深い探索を省略する回数が減り、誤った省略も通常は減る設定。",
        },
        {
            "term": "選択率レベル",
            "strict": "実装が複数の選択率を区別するために使う整数番号。表には番号と実際の選択率を併記する。",
            "plain": "選択率を指定する実装内部の番号。",
        },
        {
            "term": "整数誤差幅",
            "strict": "終盤の浅い探索では `ceil(1.10×分位点×σ−10⁻¹²)` で求める石差単位の整数。上側と下側を別々に計算する。",
            "plain": "MPC判定の境界へ加える石差。大きいほど枝刈り成立回数が減る。",
        },
        {
            "term": "誤差標本・開始局面群",
            "strict": "誤差標本は1回のMPC判定機会から得た残差。開始局面群は同じ開始局面から派生した誤差標本の集合。",
            "plain": "1個の誤差データと、同じ対局開始点から得た誤差データのまとまり。",
        },
        {
            "term": "5分割交差検証",
            "strict": "開始局面群を5組へ分け、4組で係数を求め、残る1組で指標を計算する処理を、検証側を変えて5回行う方法。",
            "plain": "係数を決めたデータとは別の部分でも誤差予測が合うかを確認する方法。",
        },
        {
            "term": "独立確認用データ",
            "strict": "係数、浅い探索深度、`λ`、`p` の選択へ使用せず、選択完了後の指標計算だけに使用するデータ。",
            "plain": "設定決定に使わず、最後の確認にだけ使うデータ。",
        },
        {
            "term": "中盤係数再推定の目的関数 `J`",
            "strict": "`J=Σwᵢ[ln(σᵢ)+rᵢ²÷(2σᵢ²)]÷Σwᵢ+λD`。`Σ` は全標本についての合計。係数は `J` を最小にするよう求める。",
            "plain": "誤差予測の不一致と現行係数からの差を1個の数値にしたもの。小さい係数組を選ぶ。",
        },
        {
            "term": "現行係数との差 `D`",
            "strict": "`D=Σ((θⱼ−θⱼ⁽⁰⁾)÷max(abs(θⱼ⁽⁰⁾),1))²`。`θⱼ` は候補係数、`θⱼ⁽⁰⁾` は現行係数、`max` は大きい方を返す関数。",
            "plain": "候補係数が現行係数から変化した量を、係数ごとに尺度をそろえて合計した値。",
        },
        {
            "term": "`λ`",
            "strict": "中盤の目的関数 `J` の項 `λD` で `D` に掛ける非負の値。今回の候補値は1、3、10。",
            "plain": "目的関数の中で、現行係数との差を何倍で数えるかを指定する値。",
        },
        {
            "term": "現行式から生成した121個の追加データ",
            "strict": "石数を11段階、浅い探索深度を11段階にした121組について、現行式の `σ` を目標値として作成した終盤係数学習用データ。",
            "plain": "実測データが少ない入力でも、現行式の値を係数計算へ含めるためのデータ。",
        },
        {
            "term": "追加データの合計重み `p`",
            "strict": "121個の追加データへそれぞれ `p÷121` を掛けたときの重みの合計。`p=0` では追加データを使わない。",
            "plain": "121個の追加データ全体を係数計算へどれだけ反映するかを指定する値。",
        },
        {
            "term": "実測標準偏差",
            "strict": "同じ条件の残差が `n` 個あるときの `sqrt(Σrᵢ²÷n)`。残差平均を0とする今回の仮定に対応する。",
            "plain": "実際に観測した浅い探索誤差の大きさ。",
        },
        {
            "term": "正規分布負対数尤度",
            "strict": "各標本の `ln(σᵢ)+rᵢ²÷(2σᵢ²)` の重み付き平均。全候補で同じ定数項は省く。",
            "plain": "予測標準偏差と観測残差の一致を測る値。小さい方が一致している。",
        },
        {
            "term": "標準化残差の二乗平均平方根",
            "strict": "`sqrt(Σ(rᵢ÷σᵢ)²÷n)`。重み付きの場合は対応する重み付き平均を使う。",
            "plain": "1に近いほど、予測した誤差量と観測した誤差量の全体的な大きさが一致している。",
        },
        {
            "term": "標準偏差予測の平均絶対誤差",
            "strict": "条件ごとの予測標準偏差を `σ̂ₖ`、実測標準偏差を `sₖ`、条件数を `K` としたときの `Σabs(σ̂ₖ−sₖ)÷K`。",
            "plain": "標準偏差の予測が平均で何石ずれたか。小さい方がよい。",
        },
        {
            "term": "標準偏差予測の二乗平均平方根誤差",
            "strict": "条件数を `K` としたときの `sqrt(Σ(σ̂ₖ−sₖ)²÷K)`。",
            "plain": "標準偏差予測の誤差を、大きなずれへ大きい比重を付けて集計した値。小さい方がよい。",
        },
        {
            "term": "固定倍率",
            "strict": "モデルが計算した全 `σ` または全分位点 `z` へ、入力条件に関係なく同じ数を掛ける変更。",
            "plain": "係数を学習し直さず、判定幅だけを一律に変更する方法。",
        },
    ]

    measurement_terms = [
        {
            "term": "固定深度測定",
            "strict": "すべての比較設定へ同一局面、同一探索深度、同一選択率集合を与える測定。",
            "plain": "同じ深さまで読ませ、時間・訪問局面数・選んだ手を比べる測定。",
        },
        {
            "term": "局面×選択率",
            "strict": "`局面数×各局面で実行した選択率数` で求めた探索実行数。",
            "plain": "表の合計値に含まれる探索の回数。",
        },
        {
            "term": "推定訪問局面数比",
            "strict": "`(浅い探索の訪問局面数+枝刈り不成立時の深い探索訪問局面数)÷全判定で深い探索した場合の訪問局面数`。",
            "plain": "記録済みデータから計算した訪問局面数の比。実行時間の実測値ではない。",
        },
        {
            "term": "事後計算",
            "strict": "保存済みの浅い探索値、完全読み値、訪問局面数を式へ代入して求め、候補実行ファイルを動かさない計算。",
            "plain": "過去の記録から求めた値であり、実行ファイルの速度測定ではない。",
        },
        {
            "term": "訪問局面数比・時間比・探索速度比",
            "strict": "同じ条件で `候補の値÷現行の値` とした比。",
            "plain": "1で同じ。時間比と訪問局面数比は1未満なら候補が少なく、探索速度比は1超なら候補が速い。",
        },
        {
            "term": "総和比",
            "strict": "現行と候補が両方完了した条件だけについて、`候補の合計÷現行の合計` とした比。",
            "plain": "両方が完了した条件全体で、候補が現行の何倍だったか。",
        },
        {
            "term": "幾何平均",
            "strict": "正の比 `x₁`～`xₙ` に対する `(x₁×…×xₙ)^(1÷n)`。",
            "plain": "各条件の倍率を、条件数の多い範囲に偏らない1個の倍率へまとめた値。",
        },
        {
            "term": "固定深度の着手損失・平均着手損失",
            "strict": "着手損失は `完全読み最善手の評価値−候補が選んだ手の完全読み評価値`。単位は石。平均着手損失は全局面の着手損失合計を局面数で割った値。",
            "plain": "候補が選んだ手によって最善手から何石失ったか。0が最善手で、平均値も小さい方がよい。",
        },
        {
            "term": "完全読みとの着手一致率",
            "strict": "`候補の着手が完全読み最善手と一致した局面数÷測定局面数×100`。",
            "plain": "候補が完全読みと同じ手を選んだ割合。",
        },
        {
            "term": "固定時間対戦・2局1組",
            "strict": "同じ開始局面で先後を交換して2局実行し、その2局の合計結果を1組の勝ち、引き分け、負けに分類する測定。",
            "plain": "先手・後手の差を減らすため、同じ開始局面を両方の色で対戦する測定。",
        },
        {
            "term": "組スコア率",
            "strict": "`(組の勝ち数+0.5×組の引き分け数)÷組数×100`。",
            "plain": "50%なら同点、50%を超えれば測定結果では候補側の得点が多い。",
        },
        {
            "term": "1局当たり平均石差",
            "strict": "候補側から見た各局の最終石差を合計し、対局数で割った値。",
            "plain": "1局につき候補が平均で何石多く、または少なく終えたか。",
        },
        {
            "term": "復元抽出法による95%信頼区間",
            "strict": "観測した対戦組から元と同じ組数を重複ありで反復抽出し、各反復の組スコア率の2.5%点から97.5%点までを取った区間。",
            "plain": "対戦組を取り直したときに組スコア率が変動する範囲。50%を含む場合、この測定だけでは優劣を確定できない。",
        },
        {
            "term": "読み切り・読み切り率",
            "strict": "2.6節では、選択率74%以上の終盤探索が制限時間内に終局まで完了した局面を読み切りとして数える。読み切り率は `完了局面数÷全局面数×100`。",
            "plain": "選択率74%以上の終盤探索が時間内に最後まで計算できた局面数と割合。",
        },
        {
            "term": "完全読み",
            "strict": "選択率100%で終局まで探索し、MPCによる枝刈りを使わず厳密な評価値を得ること。",
            "plain": "省略を使わず、最終結果を確定した探索。",
        },
        {
            "term": "完全読み値の絶対誤差",
            "strict": "`abs(候補の完全読み値−現行の完全読み値)`。",
            "plain": "候補と現行が求めた確定石差の違い。両方が正しく完了すれば0になる。",
        },
        {
            "term": "候補完了 / 実行・候補時間切れ",
            "strict": "候補完了 / 実行は、制限時間内に完了した候補の実行数と開始した全実行数。候補時間切れは `全実行数−完了数`。",
            "plain": "候補が何件中何件を時間内に終え、何件を終えられなかったか。",
        },
        {
            "term": "完全読み値一致・完全読み手一致",
            "strict": "現行と候補の両方が完了した条件で、返した完全読み値が同じだった件数と、返した着手が同じだった件数。",
            "plain": "候補が現行と同じ確定石差と着手を返した回数。",
        },
        {
            "term": "中央値・90%点",
            "strict": "中央値は昇順に並べた値の50%点。90%点は昇順に並べた値の90%がその値以下になる境界。",
            "plain": "中央値は中央の事例、90%点は遅い側から10%の事例が始まる境界。",
        },
        {
            "term": "最初の読み切り時刻・最終探索深度",
            "strict": "最初の読み切り時刻は探索開始から、選択率74%以上で最初に終局まで完了した結果を得るまでのミリ秒。最終探索深度は制限時間内に最後に完了して出力された探索結果の深度。",
            "plain": "終盤の結果を最初に得た時刻と、時間内に完了した最後の探索の深さ。",
        },
        {
            "term": "測定群",
            "strict": "同一の局面集合、実行条件、集計手順で得た行の集合。比は同じ測定群内で計算する。",
            "plain": "互いに直接比較できる測定結果のまとまり。",
        },
        {
            "term": "GGS",
            "strict": "Generic Game Serverの名称。この文書ではGGS対戦記録から抽出した局面集合を示す。",
            "plain": "実戦由来の局面を取得した対局サーバー。",
        },
    ]

    def term_table(rows: Sequence[Mapping[str, str]]) -> str:
        return markdown_table(
            rows,
            [
                ("term", "用語・記号"),
                ("strict", "厳密な定義"),
                ("plain", "平易な説明（つまり何か）"),
            ],
        )

    return [
        "# Multi-ProbCut（MPC）の浅い探索深度・標準偏差モデル・枝刈り確率の比較",
        "",
        "この文書は、変更内容ごとの測定値を比較するためのレポートである。採用する設定は決めない。最初に主要結果を示し、その後に測定条件と全結果を示す。",
        "",
        "## 主要結果",
        "",
        "| 変更内容 | 実行ファイルで得た主な数値 | この数値から直接言えること |",
        "|---|---|---|",
        "| 中盤の標準偏差モデルを再推定し、浅い探索深度を変えない候補 | 固定深度の時間比1.0034、2石以上の着手損失7件、4石以上1件。現行は時間比1.0000、7件、1件。1手100ミリ秒対戦の組スコア率49.22%、95%信頼区間44.92%～53.52%。 | 固定深度では現行とほぼ同じ時間・着手精度だった。対戦の信頼区間は50%を含む。 |",
        "| 中盤の浅い探索深度だけを変更 | 固定深度では全候補の時間比が1を超えた。1手100ミリ秒対戦では全4候補の95%信頼区間が50%を含んだ。 | 測定した候補には、固定深度の時間を現行より減らした設定がない。対戦だけでは候補間の優劣を確定できない。 |",
        "| 終盤19手以上の標準偏差モデルを再推定 | 1手15秒・65局面の読み切りは現行38局面、開始局面均等候補36局面、探索深度範囲均等候補34局面。30マス空きの時間総和比はそれぞれ1.0641と1.1180。 | この測定では両候補とも現行より読み切り数が少なく、両方が完了した条件の合計時間も多い。 |",
        "| 終盤MPCの浅い探索深度を変更 | 1手15秒・65局面の読み切りは現行38、−2手37、再学習±0手34、+2手33、+4手27。 | この測定では全候補の読み切り数が現行より少ない。 |",
        "| 分位点 `z` へ0.98を掛けた参考測定 | 中盤固定深度の時間比0.9488。2石以上の着手損失4件、4石以上0件、平均着手損失0.018で、同じ測定群の現行と同数。 | この100局面測定では時間が5.12%減り、表にある着手精度指標は変わらなかった。固定倍率機能は現在の本体プログラムに入っていない。 |",
        "",
        "## 表を読む順序",
        "",
        "1. 実行ファイルで測った時間は、`合計探索時間`、`時間比`、`1手100ミリ秒対戦`、`1手15秒`の表で確認する。`推定訪問局面数比`は実行時間ではない。",
        "2. 固定深度では、時間比と訪問局面数比が1未満なら現行より小さく、探索速度比が1を超えれば現行より1秒当たりの処理局面数が多い。着手損失は小さい方、完全読みとの着手一致率は大きい方が正確である。",
        "3. 固定時間対戦では、組スコア率50%が同点である。95%信頼区間が50%を含む場合、この対戦数だけでは優劣を確定できない。",
        "4. 同じ表の現行と候補だけを比較する。局面集合や測定条件が異なる別表の絶対時間は直接比較しない。",
        "",
        "## 用語の定義",
        "",
        "用語を先にすべて読む必要はない。表で不明な語が出たときに参照できるよう、厳密な定義と平易な説明を併記する。",
        "",
        "### 探索処理",
        "",
        term_table(search_terms),
        "",
        "### 誤差モデルと係数計算",
        "",
        term_table(model_terms),
        "",
        "### 測定値",
        "",
        term_table(measurement_terms),
        "",
        "## 比較した三種類の変更",
        "",
        "1. **標準偏差モデルの係数再推定**：中盤では `a`～`g`、終盤の深い探索深度19手以上では `a`～`f` を誤差標本から計算し直した。分位点 `z` は変更していない。",
        "2. **MPCの浅い探索深度の変更**：深い探索を省略できるか判断するために先に実行する探索の深度を変更した。終盤の深い探索深度10～18手では、候補ごとに標準偏差、上下分位点、整数誤差幅も計算し直した。",
        "3. **分位点 `z` への固定倍率**：標準偏差モデルの係数は変更せず、全分位点 `z` へ同じ倍率を掛けた。この変更は参考測定だけに使い、現在の本体プログラムには残していない。",
        "",
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
            "| MPC | Multi-ProbCutの略。浅い探索の結果と、浅い探索値・深い探索値の誤差の分布を使い、深い探索を省略できるか判定する枝刈り法。 |",
            "| 静的評価 | 先の手を探索せず、現在の盤面だけから評価値を計算すること。 |",
            "| 探索窓 | 探索で詳しく調べる評価値の範囲。値が下限以下か、範囲内か、上限以上かを判定する。 |",
            "| 深い探索深度 | 本来求めたい探索の深度。 |",
            "| 浅い探索深度 | MPCが深い探索を省略できるか判定するため、先に実行する探索の深度。 |",
            "| 固定深度測定 | すべての局面を同じ探索深度まで調べて設定を比較する測定。 |",
            "| MPC判定機会 | ある局面・探索窓・選択率で、MPCによる枝刈りを検討した1回。 |",
            "| 枝刈り | 浅い探索または静的評価に基づき、深い探索を行わずに値が探索窓の外側にあると判定すること。表の枝刈り回数は、この判定が成立した回数。 |",
            "| 誤った枝刈り | 枝刈りした方向と完全読みの値が矛盾したもの。`2石以上`と`4石以上`は、完全読みの値が探索窓の境界から外れた大きさ。 |",
            "| 訪問局面数 | 探索中に処理した局面の延べ数。実装内部でノード数と呼んでいる値。 |",
            "| 推定訪問局面数比 | 浅い探索の訪問局面数と、枝刈りできなかった場合の深い探索の訪問局面数を合計し、すべてを深く探索した場合の訪問局面数で割った値。経過時間の比ではない。 |",
            "| 探索速度 | 1秒当たりの訪問局面数。 |",
            "| 現行 | 比較の基準にした、変更前の実行ファイルまたは設定。 |",
            "| 候補 | 現行と比較するために変更を加えた実行ファイルまたは設定。 |",
            "| モデル | 入力値から予測値を計算する式と、その式で使う係数の組。 |",
            "| 標準偏差モデル | 石数、浅い探索深度、深い探索深度から、浅い探索値と深い探索値の誤差の標準偏差を予測する計算式。 |",
            "| 係数 | 標準偏差モデルの式に入る固定値。この文書では `a`～`g` などで表す。 |",
            "| 係数再推定 | 収集した誤差データを使い、標準偏差モデルの係数を計算し直すこと。 |",
            "| 学習 | 収集した誤差データと予測値のずれが小さくなるように、モデルの係数を決める計算。 |",
            "| 重み付け | 各誤差標本が係数の計算へ与える影響を、掛け算で増減すること。 |",
            "| 中盤係数再推定の目的関数 `J` | `J = Σwᵢ[ln(σᵢ)+rᵢ²/(2σᵢ²)]/Σwᵢ + λD`。`rᵢ` は残差、`σᵢ` は予測標準偏差、`wᵢ` は各誤差標本の重みである。係数は `J` が最小になるように決める。 |",
            "| 現行係数との差 `D` | `D = Σ((θⱼ−θⱼ⁽⁰⁾)/max(|θⱼ⁽⁰⁾|,1))²`。`θⱼ` は再推定する係数、`θⱼ⁽⁰⁾` は対応する現行係数である。 |",
            "| `λ` | 中盤係数再推定の目的関数 `J` で、現行係数との差 `D` に掛ける値。表の1、3、10などがこの値である。 |",
            "| 標準偏差 `σ` | 浅い探索値と深い探索値の誤差の広がりを予測する値。 |",
            "| 正規分布 | 平均の付近が最も多く、平均から離れるほど少なくなる左右対称の確率分布。平均と標準偏差で形が決まる。 |",
            "| 標準正規分布 | 平均が0、標準偏差が1の正規分布。 |",
            "| 分位点 `z` | 指定した確率に対応する標準正規分布上の境界値。MPCの判定幅は主に `σ×z` で決まる。 |",
            "| 上側・下側分位点 | 残差を予測標準偏差で割った値を並べ、指定した確率に対応する上側と下側の境界をそれぞれ求めた値。 |",
            "| 選択率 | MPCの枝刈り判断を誤らないために設定する目標確率。高いほど枝刈り条件が厳しくなり、100%ではMPCによる枝刈りを行わない。 |",
            "| 選択率レベル | 実装内部で選択率を指定する整数番号。表には対応する選択率も併記する。 |",
            "| 整数誤差幅 | MPC判定で評価値に加減する石数単位の境界幅。標準偏差と上下の分位点から求め、整数に丸めた値。 |",
            "| 交差検証 | 学習用データを複数群に分け、一部で学習し、残りで誤差を測る手順を群を入れ替えて繰り返す検証方法。 |",
            "| 開始局面群 | 同じ元の開始局面から得た誤差標本をまとめた単位。交差検証では同じ群を学習側と検証側へ分割しない。 |",
            "| 誤差標本 | 1回のMPC判定機会について記録した、浅い探索値と深い探索値の差。 |",
            "| 独立確認用データ | 係数や設定の選択には使わず、選択後の性能確認だけに使うデータ。 |",
            "| 現行式から生成した121個の追加データ | 盤上石数を0～64の11段階、浅い探索深度を0～60の11段階に分けた全121組合せについて、現行式の標準偏差を目標値としたデータ。終盤係数の学習用データへ加える。 |",
            "| 121個の追加データの合計重み `p` | 121個の追加データが係数計算へ与える重みの合計。各追加データの重みは `p/121`。`p=0` なら追加しない。 |",
            "| 残差 | 深い探索値から浅い探索値を引いた値。この学習では残差の平均を0とみなし、標準偏差モデルは残差の広がりだけを予測する。 |",
            "| 実測標準偏差 | 石数と浅い探索深度が同じ誤差標本をまとめ、残差を二乗して平均し、平方根を取った値。 |",
            "| 正規分布負対数尤度 | 残差を `r`、予測標準偏差を `σ` としたとき、各標本の `ln(σ)+r²/(2σ²)` を平均した値。`ln` は自然対数を表す。小さいほど予測標準偏差が観測誤差によく合う。全候補に共通する定数は省いている。 |",
            "| 標準化残差の二乗平均平方根 | 残差を予測標準偏差で割り、その値を二乗して平均し、平方根を取った値。1に近いほど予測した誤差幅と観測誤差の大きさが一致する。 |",
            "| 標準偏差予測の平均絶対誤差 | 予測標準偏差と実測標準偏差の差の絶対値を平均した値。小さいほどよい。 |",
            "| 標準偏差予測の二乗平均平方根誤差 | 予測標準偏差と実測標準偏差の差を二乗して平均し、平方根を取った値。大きな誤差の影響を強く受け、小さいほどよい。 |",
            "| 通常窓探索 | 下限と上限の間に幅を持つ探索窓を使う探索。 |",
            "| ゼロ窓探索 | 候補値が境界を超えるかだけを調べる、幅1の探索窓を使う探索。 |",
            "| 固定深度の着手損失 | 完全読みを基準として、候補が選んだ手の評価値が最善手より何石低いかを表す値。 |",
            "| 固定時間対戦 | 同じ開始局面を先後交替で2局行い、その2局を1組として勝ち・引き分け・負けを数える測定。 |",
            "| 組スコア率 | 2局1組対戦で、勝ちを1点、引き分けを0.5点、負けを0点として組数で割った値。 |",
            "| 1局当たり平均石差 | 各対局の最終石差を候補側から見た符号で合計し、対局数で割った値。 |",
            "| 復元抽出法による95%信頼区間 | 対戦結果から同じ組数を重複を許して何度も抽出し直し、各回の組スコア率を計算したとき、中央95%が入る範囲。 |",
            "| 幾何平均 | 複数の比をすべて掛け、その個数に応じた累乗根を取った平均。 |",
            "| 総和比 | 現行と候補の両方が完了した条件だけを使い、候補の合計値を現行の合計値で割った値。 |",
            "| 訪問局面数比・時間比・探索速度比 | 同じ測定群で候補の値を現行の値で割った値。1なら同じ、1未満なら候補の方が小さい。 |",
            "| 測定群 | 同じ局面集合と同じ探索条件でまとめて実行した測定結果の単位。 |",
            "| 局面×選択率 | 局面数に、各局面で試した選択率の個数を掛けた探索実行数。 |",
            "| 完全読みとの着手一致率 | 候補が選んだ手と、完全読みで得た最善手が同じだった局面の割合。 |",
            "| 候補完了 / 実行 | 候補が制限時間内に完了した条件数と、候補を実行した全条件数。 |",
            "| 中央値 | 値を小さい順に並べたとき中央に来る値。 |",
            "| 90%点 | 値を小さい順に並べたとき、全体の90%がその値以下になる境界。 |",
            "| 読み切り | 指定した選択率の終盤探索が、制限時間内に終局まで完了すること。 |",
            "| 読み切り率 | 読み切った局面数を測定した全局面数で割った値。 |",
            "| 完全読み | 選択率100%の終盤探索が終局まで完了し、厳密な評価値を求めること。 |",
            "| 完全読み値の絶対誤差 | 候補の完全読み値と現行の完全読み値の差から符号を除いた値。 |",
            "| 置換表 | 以前に探索した局面と結果を保存し、同じ局面を再び探索するときに再利用する表。 |",
            "| 置換表容量設定 | 置換表へ保存できる項目数を、実装の設定番号で表したもの。 |",
            "| スレッド数 | 1個の実行ファイルの中で、探索を並列に進める処理単位の数。 |",
            "| 使用コア数 | 中央処理装置が同時に計算を進める物理的な処理単位を、探索中に平均して何個使ったかを表す値。 |",
            "| 中央処理装置使用率 | 指定した探索スレッド数に対し、実際に中央処理装置を使った割合。 |",
            "| GGS | Generic Game Serverの名称。このレポートでは、GGS対戦から抽出した局面の測定群を示すときに使う。 |",
            "",
            "## 比較の区分",
            "",
            "次の三つは変更内容が異なるため、表を分けている。",
            "",
            "1. **標準偏差モデルの係数再推定**：中盤では `a`～`g`、終盤の深い探索深度19手以上で使う式では `a`～`f` をデータから再推定する。標準正規分布の分位点 `z` は変えない。",
            "2. **浅い探索深度の変更**：MPCが境界判定に使う浅い探索の深度を変える。終盤の深さ10～18手については、候補深度ごとに標準偏差・上下分位点・整数誤差幅を学習し直している。",
            "3. **分位点 `z` へ固定倍率を掛けた参考測定**：標準偏差モデルの係数は変えず、すべての分位点 `z` に同じ倍率を掛ける。この測定は係数再推定の結果ではない。",
            "",
            "## 1. 標準偏差モデルの係数再推定",
            "",
            "### 1.1 中盤の a～g 係数",
            "",
            "使用式は `x=a×(石数÷64)+b×(浅い探索深度÷60)+c×(深い探索深度÷60)`、`σ=(d×x+e)×x²+f×x+g` である。",
            "",
            "**比較対象:** 現行係数と、浅い探索深度を−4、−2、±0、+2、+4手にした各条件専用の再推定係数。",
            "",
            "**算出方法:** 各条件の誤差標本から a～g を再推定した。`σ` の計算後に固定倍率を掛けていない。分位点 `z` は全条件で現行と同じ 1.13 / 1.55 / 1.81 / 2.088 / 2.1845 / 2.7965 である。",
            "",
            "**表の読み方:** a～g は式へ代入する値であり、係数の大小だけでは速度や正確さを判定できない。速度と着手精度は1.1.1、固定時間の対戦結果は1.1.2で比較する。",
            "",
        ]
    )
    first_detail = lines.index("## 1. 標準偏差モデルの係数再推定")
    lines = build_report_intro() + lines[first_detail:]

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
    lines.append(markdown_table(mid_coeff_table, [("label", "設定"), ("shallow_offset", "浅い探索深度差"), ("ridge", "λ"), *((letter, letter) for letter in "abcdefg")]))
    lines.extend(["", "実行ファイルへ組み込んだ係数の全桁は[中盤の係数一覧](mid_model_refit/cv_filtered_coefficients.csv)にある。表の `λ` は目的関数の項 `λD` へ代入した値である。開始局面群単位の5分割交差検証で `λ` を選び、独立確認用データは係数または `λ` の選択に使っていない。学習用と独立確認用の間には、同じ盤面、盤面を回転または鏡映した盤面、黒石と白石を入れ替えた盤面の重複が0件だった。", ""])

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
            "**比較対象:** 各浅い探索深度で現行係数を使った場合と、同じ深度専用に再推定した係数を使った場合。",
            "",
            "**算出方法:** 開始局面群単位の5分割交差検証。各指標の式は用語表に示した。",
            "",
            "**表の読み方:** 正規分布負対数尤度と誤った枝刈りは小さい方がよい。標準化残差の二乗平均平方根は1に近い方が予測量と観測量の大きさが一致する。推定訪問局面数比は実行時間ではない。",
            "",
            markdown_table(mid_cv_table, [("offset", "浅い探索深度差"), ("model", "モデル"), ("level", "選択率"), ("rows", "誤差標本"), ("roots", "開始局面群"), ("nll", "正規分布負対数尤度"), ("norm_rms", "標準化残差の二乗平均平方根"), ("wrong", "誤った枝刈り"), ("wrong_rate", "誤った枝刈りの割合"), ("wrong2", "予測誤差幅を2石以上超過"), ("wrong4", "予測誤差幅を4石以上超過"), ("node_ratio", "推定訪問局面数比")]),
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
            "**比較対象:** 現行と、浅い探索深度および a～g を同時に変更した5候補。",
            "",
            "**測定方法:** 独立確認用300局面、中盤深さ16手、選択率6段階、1スレッド。各探索の開始時に置換表を空にした。各行は1,800探索の合計である。",
            "",
            "**表の読み方:** 時間比と訪問局面数比は1未満なら現行より少ない。着手損失は小さい方、完全読みとの着手一致率は大きい方が正確である。この表は、現行係数のまま浅い探索深度だけを変えた2.1節とは変更内容が異なる。",
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
            "**比較対象:** 浅い探索深度および a～g を同時に変更した各候補と現行。",
            "",
            "**測定方法:** 各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20。開始局面番号5000～5255を使い、同じ開始局面を先後交替した2局を1組とした。",
            "",
            "**表の読み方:** 勝ち・引き分け・負けは候補側から見た組数である。組スコア率50%が同点であり、95%信頼区間が50%を含む場合はこの対戦数だけで優劣を確定できない。",
            "",
            markdown_table(model_fixed_time_table, [("label", "設定"), ("pairs", "組数"), ("wdl", "組の勝ち / 引き分け / 負け"), ("score", "組スコア率"), ("ci", "復元抽出法による95%信頼区間"), ("disc", "1局当たり平均石差")]),
            "",
        ]
    )
    lines.extend(["", "### 1.2 終盤の深い探索深度19手以上で使う a～f 係数", ""])
    lines.extend(
        [
            "対象式は `x=石数÷64`、`y=浅い探索深度÷60`、`u=a×x+b×y`、`σ=c×u³+d×u²+e×u+f` である。主な使用箇所は深い探索深度19手以上であり、深い探索深度10～18手・選択率74%～93%の専用表とは別である。",
            "",
            "`開始局面を均等に重み付けした係数`は、各開始局面群の標本数が違っても、1群当たりの学習への影響を同じにした候補である。`探索深度範囲を均等に重み付けした係数`は、深い探索深度10～18手、19～26手、30手などの収集元ごとに学習への影響を同じにし、さらに各収集元の中では開始局面群の影響を同じにした候補である。",
            "",
            "係数再推定に使った新規データの中心は深い探索深度19～26手である。履歴データには深い探索深度10～18手があり、深さ30手は8局面を独立確認に使った。学習時は `ρ=−b`、`v=x+ρy` とした。石数と浅い探索深度の全範囲で得る `v` の最小値を `vₘᵢₙ`、最大値を `vₘₐₓ` とし、`t=(v−vₘᵢₙ)÷(vₘₐₓ−vₘᵢₙ)` としたため、tは0以上1以下になる。標準偏差は `σ=q₀(1−t)³+3q₁t(1−t)²+3q₂t²(1−t)+q₃t³` と表し、`q₀`～`q₃` をすべて0.5以上にした。q₀～q₃に掛かる `(1−t)³`、`3t(1−t)²`、`3t²(1−t)`、`t³` はそれぞれ0以上で、合計は1である。このため、すべてのtで `σ` は0.5以上になる。学習後に同じ `σ` を返す c～f を式の展開によって計算したため、探索中の計算手順は現行と同じである。学習用と各独立確認用データの間には、同じ開始局面、同じ盤面、盤面を回転または鏡映した盤面、黒石と白石を入れ替えた盤面の重複が0件だった。",
            "",
            "`u=a×x+b×y` を一定倍し、c～fをその倍率に対応して変換すると、すべての入力で同じ `σ` を返す別の係数組を作れる。このため a～fを同時に自由変数にすると係数組が一意に決まらない。候補では a=-1に固定し、b～fだけを推定した。実測データに加え、用語表で定義した121個の追加データを使用した。表の `p` は追加データの合計重みであり、開始局面を均等に重み付けした候補は0.03、探索深度範囲を均等に重み付けした候補は3である。候補選択処理は `p=0` を比較対象から除外するよう実装されていた。`p=0` で計算した場合は両方の重み付け方法で b=-31.6665だった。したがって、下表の2候補は `p=0` を含む全設定から交差検証値だけで選んだ結果ではない。",
            "",
            "**比較対象:** 現行係数、開始局面群の合計重みを均等にした候補、探索深度範囲ごとの合計重みを均等にした候補。",
            "",
            "**算出方法:** 上記の誤差標本と121個の追加データから a～f を再推定した。標準偏差への固定倍率は全候補で1.0。",
            "",
            "**表の読み方:** a～f と `p` は係数計算の結果と条件であり、速度・誤差・対戦結果は後続の表で比較する。",
            "",
        ]
    )
    end_coeff_table = [
        {
            "model": row["label"],
            "representation": row["representation"],
            "reference_weight": (
                fmt_float(row["reference_weight"], 2)
                if row.get("reference_weight") != "" else "—"
            ),
            "multiplier": fmt_float(row["global_sigma_multiplier"], 1),
            **{letter: fmt_float(row[letter], 6) for letter in "abcdef"},
        }
        for row in end_coeffs
    ]
    lines.append(markdown_table(end_coeff_table, [("model", "モデル"), ("representation", "学習時と実行時の計算方法"), ("reference_weight", "p"), *((letter, letter) for letter in "abcdef"), ("multiplier", "標準偏差への固定倍率")]))
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
            "**検査対象:** 各係数式へ、石数0～64と浅い探索深度0～60の全3,965組合せを代入した。",
            "",
            "**表の読み方:** 最小標準偏差が0.5以上なら、検査した全入力で実装が要求する下限0.5を満たす。この検査は誤差予測の正確さや探索速度を測っていない。",
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
    lines.extend([
        "**比較対象:** 5分割交差検証で得た、現行と2候補の標準偏差予測。",
        "",
        "**表の読み方:** 二乗平均平方根誤差、平均絶対誤差、正規分布負対数尤度は小さい方がよい。標準化残差の二乗平均平方根は1に近い方がよい。",
        "",
        markdown_table(end_cv_table, [("model", "モデル"), ("roots", "開始局面群"), ("obs", "誤差標本"), ("sigma_rmse", "標準偏差予測の二乗平均平方根誤差"), ("sigma_mae", "標準偏差予測の平均絶対誤差"), ("std_rms", "標準化残差の二乗平均平方根"), ("nll", "正規分布負対数尤度")]),
        "",
        "**比較対象:** 係数選択に使わなかった三つの独立確認用データについて、同じ4指標をデータ別に計算した。",
        "",
        "**表の読み方:** 各データ内で現行と2候補を比較する。開始局面群と誤差標本が少ない行は、少数の観測値から計算した結果である。",
        "",
    ])
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
            "**比較対象:** 完全読み値を保存してある判定機会で、現行と2候補のMPC判定を選択率別に再現した。",
            "",
            "**算出方法:** 保存済みの浅い探索値、完全読み値、訪問局面数を使う事後計算。候補実行ファイルは動かしていない。",
            "",
            "**表の読み方:** 誤った枝刈りは少ない方がよい。推定訪問局面数比は小さい方が推定訪問局面数は少ないが、実測した探索時間比ではない。",
            "",
            markdown_table(end_cut_table, [("dataset", "データ"), ("model", "モデル"), ("level", "選択率レベル"), ("selectivity", "選択率"), ("contexts", "判定機会"), ("probes", "浅い探索実行"), ("cuts", "枝刈り"), ("wrong", "誤った枝刈り"), ("wrong2", "誤枝刈りが境界から2石以上"), ("wrong4", "誤枝刈りが境界から4石以上"), ("ratio", "推定訪問局面数比")]),
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
            "#### 1.2.1 終盤の深い探索深度19手以上で使う a～f の実行ファイルによる比較",
            "",
            "**比較対象:** 30マス空き8局面と選択率74%・88%・93%の合計24条件で、現行と終盤係数2候補を比較した。",
            "",
            "**測定方法:** 各条件を現行と候補で1回ずつ実行した。1スレッド、置換表は毎回空、制限時間180秒。合計値は完了した実行だけを含み、時間切れの180秒は含めない。総和比と幾何平均は両方が完了した条件だけで計算した。",
            "",
            "**表の読み方:** 完了数を先に確認する。時間総和比と訪問局面数総和比は1未満なら候補が少ない。完全読み値・手の一致数の分母は候補が完了した実行数。",
            "",
            markdown_table(generic_runtime_table, [("label", "設定"), ("runs", "候補完了 / 実行"), ("timeout", "候補時間切れ"), ("control_nodes", "現行完了分・合計訪問局面数"), ("nodes", "候補完了分・合計訪問局面数"), ("node_sum", "両方が完了した条件での訪問局面数総和比"), ("node_geo", "両方が完了した条件での訪問局面数比の幾何平均"), ("control_time", "現行完了分・合計時間（ミリ秒）"), ("time", "候補完了分・合計時間（ミリ秒）"), ("time_sum", "両方が完了した条件での時間総和比"), ("time_geo", "両方が完了した条件での時間比の幾何平均"), ("control_nps", "現行の探索速度（局面/秒）"), ("nps", "候補の探索速度（局面/秒）"), ("nps_ratio", "探索速度比"), ("value", "完全読み値一致 / 候補完了"), ("move", "完全読み手一致 / 候補完了"), ("error", "候補完了分・完全読み値の絶対誤差合計")]),
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
            "#### 1.2.2 終盤の深い探索深度19手以上で使う a～f の1手100ミリ秒対戦",
            "",
            "**比較対象:** 終盤係数2候補と現行係数。",
            "",
            "**測定方法:** 各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20。開始局面番号5000～5255を使い、先後交替した2局を1組とした。",
            "",
            "**表の読み方:** 勝ち・引き分け・負けは候補側から見た組数。組スコア率50%が同点であり、95%信頼区間が50%を含む場合はこの対戦数だけで優劣を確定できない。",
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
            "**比較対象:** 現行と、MPCの浅い探索深度だけを−4、−2、+2、+4手変更した候補。全候補で既存の a～g 係数を共通に使う。",
            "",
            "**測定方法:** GGS実戦由来100局面、中盤深さ16手、選択率6段階、1スレッド。各探索の開始時に置換表を空にした。訪問局面数と時間は600探索の合計。",
            "",
            "**表の読み方:** 時間比と訪問局面数比は1未満なら現行より少ない。着手損失は小さい方、完全読みとの着手一致率は大きい方が正確である。1.1節の再推定係数は使っていない。",
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
            "**比較対象:** 浅い探索深度だけを変更した4候補と現行。",
            "",
            "**測定方法:** 各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20。開始局面番号5000～5255を使い、先後交替した2局を1組とした。",
            "",
            "**表の読み方:** 勝ち・引き分け・負けは候補側から見た組数。組スコア率50%が同点であり、95%信頼区間が50%を含む場合はこの対戦数だけで優劣を確定できない。",
            "",
            markdown_table(fixed_time_table, [("label", "設定"), ("pairs", "組数"), ("wdl", "組の勝ち / 引き分け / 負け"), ("score", "組スコア率"), ("ci", "復元抽出法による95%信頼区間"), ("disc", "1局当たり平均石差")]),
            "",
            "### 2.3 終盤深さ10～18手・候補別の再学習による事後計算",
            "",
            "**比較対象:** 浅い探索深度−2、±0、+2、+4手の各条件について、その条件専用に標準偏差、上下分位点、整数誤差幅を計算し直した結果。",
            "",
            "**算出方法:** 保存済みデータによる事後計算。浅い探索部分の訪問局面数は、置換表を空にした通常窓探索で測った。実際のMPCが使うゼロ窓探索の実測時間ではない。",
            "",
            "**表の読み方:** 誤った枝刈りは少ない方がよい。推定訪問局面数比は小さい方が事後計算上の訪問局面数が少ない。実行時間は2.4節で確認する。",
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
            markdown_table(end_offline_table, [("label", "設定"), ("cv", "交差検証の判定機会"), ("cv_cuts", "交差検証の枝刈り"), ("cv_wrong", "交差検証の誤った枝刈り"), ("cv_wrong2", "交差検証の誤枝刈りが境界から2石以上"), ("cv_wrong4", "交差検証の誤枝刈りが境界から4石以上"), ("cv_ratio", "交差検証の推定訪問局面数比"), ("ho", "6月確認の判定機会"), ("ho_cuts", "6月確認の枝刈り"), ("ho_wrong", "6月確認の誤った枝刈り"), ("ho_wrong2", "6月確認の誤枝刈りが境界から2石以上"), ("ho_wrong4", "6月確認の誤枝刈りが境界から4石以上"), ("ho_ratio", "6月確認の推定訪問局面数比")]),
            "",
            "**表の内容:** 次の表は、上の事後計算で各候補・各深い探索深度へ使用した浅い探索深度、標準偏差、上下分位点、整数誤差幅を示す。分位点と誤差幅は選択率74% / 88% / 93%の順。",
            "",
            "**表の読み方:** この表はMPC判定へ代入した数値の一覧であり、速度または正確さの結果表ではない。",
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
            "丸め前の値と一覧は[終盤深さ10～18手で学習し直した数値一覧](end_offline_refit/parameters_with_admission.csv)にある。",
            "",
            "### 2.4 終盤・30マス空き8局面の実行ファイルによる比較",
            "",
            "**比較対象:** 30マス空き8局面と選択率74%・88%・93%の合計24条件で、現行と浅い探索深度4候補を比較した。各候補は専用に再計算した標準偏差、上下分位点、整数誤差幅を使う。",
            "",
            "**測定方法:** 各条件を現行と候補で1回ずつ実行した。1スレッド、置換表は毎回空、制限時間180秒。合計値は完了した実行だけを含み、時間切れの180秒は含めない。総和比と幾何平均は両方が完了した条件だけで計算した。",
            "",
            "**表の読み方:** 完了数を先に確認する。時間総和比と訪問局面数総和比は1未満なら候補が少ない。完全読み値・手の一致数の分母は候補が完了した実行数。",
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
    lines.append(markdown_table(runtime_table, [("label", "設定"), ("runs", "候補完了 / 実行"), ("timeout", "候補時間切れ"), ("control_nodes", "現行完了分・合計訪問局面数"), ("nodes", "候補完了分・合計訪問局面数"), ("node_sum", "両方が完了した条件での訪問局面数総和比"), ("node_geo", "両方が完了した条件での訪問局面数比の幾何平均"), ("control_time", "現行完了分・合計時間（ミリ秒）"), ("time", "候補完了分・合計時間（ミリ秒）"), ("time_sum", "両方が完了した条件での時間総和比"), ("time_geo", "両方が完了した条件での時間比の幾何平均"), ("control_nps", "現行の探索速度（局面/秒）"), ("nps", "候補の探索速度（局面/秒）"), ("nps_ratio", "探索速度比"), ("value", "完全読み値一致 / 候補完了"), ("move", "完全読み手一致 / 候補完了"), ("error", "候補完了分・完全読み値の絶対誤差合計")]))

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
            "**比較対象:** 浅い探索深度と、その条件専用の標準偏差、上下分位点、整数誤差幅を同時に変更した4候補と現行。",
            "",
            "**測定方法:** 各候補256組・512局、1手100ミリ秒、1スレッド、置換表容量設定20。開始局面番号5000～5255を使い、先後交替した2局を1組とした。",
            "",
            "**表の読み方:** 勝ち・引き分け・負けは候補側から見た組数。組スコア率50%が同点であり、95%信頼区間が50%を含む場合はこの対戦数だけで優劣を確定できない。",
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
            "**比較対象:** 現行、浅い探索深度4候補、終盤係数2候補の計7設定。",
            "",
            "**測定方法:** 32～44マス空きについて各5局面、合計65局面。20スレッド、1手15秒。各局面で実行ファイルを起動し直し、置換表を空にした。",
            "",
            "**表の読み方:** 読み切り数と読み切り率は大きい方が、同じ15秒で終局まで計算できた局面が多い。最初の読み切り時刻は小さい方が早い。訪問局面数と探索速度の中央値は同一局面の対になった比ではないため、読み切り数と併せて読む。",
            "",
            markdown_table(cold_overall_table, [("label", "設定"), ("cases", "局面数"), ("completed", "読み切り / 局面"), ("rate", "読み切り率"), ("exact", "完全読み"), ("first", "最初の読み切り時刻の中央値（ミリ秒）"), ("p90", "最初の読み切り時刻の90%点（ミリ秒）"), ("depth_median", "最終探索深度中央値"), ("depth_max", "最終探索深度最大"), ("nodes", "訪問局面数中央値"), ("nps", "探索速度中央値（局面/秒）"), ("cores", "使用コア数中央値"), ("cpu", "指定20スレッドに対する中央処理装置使用率の中央値（%）")]),
            "",
            "**次の表の内容:** 上の65局面をマス空き数別に5局面ずつ分けた内訳。各マス空き数の標本数は5なので、1局面の成否で読み切り数が1変わる。",
            "",
            markdown_table(cold_by_empty_table, [("label", "設定"), ("empties", "マス空き"), ("completed", "読み切り / 局面"), ("exact", "完全読み"), ("first", "最初の読み切り時刻の中央値（ミリ秒）"), ("depth_median", "最終探索深度中央値"), ("depth_max", "最終探索深度最大"), ("nodes", "訪問局面数中央値"), ("nps", "探索速度中央値（局面/秒）")]),
        ]
    )

    lines.extend(
        [
            "",
            "## 3. 分位点 z へ固定倍率を掛けた参考測定（係数再推定ではない）",
            "",
            "この章だけは標準偏差モデルの係数を変えていない。現行の分位点 z へ全体で同じ倍率を掛けたとき、測定値がどのように変わるかを示す参考表であり、1章の係数再推定とは別実験である。",
            "この固定倍率機能は現在の本体プログラムには残していない。この章は測定済みファイルの記録だけであり、本体の現在状態を表すものではない。",
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
            "**比較対象:** 現行の全分位点 `z` に0.80～1.00の固定倍率を掛けた候補。`z=0.98`は全分位点を現行の98%にする。",
            "",
            "**測定方法:** 中盤深さ16手、100局面、選択率6段階、1スレッド。各探索の開始時に置換表を空にした。倍率範囲ごとに別の測定群で実行した。",
            "",
            "**表の読み方:** 各測定群内の現行を1として比を読む。時間比は小さい方が時間が少なく、着手損失は小さい方が正確である。この固定倍率機能は現在の本体プログラムに残っていない。",
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
            "**比較対象:** 終盤の深い探索深度10～18手で、浅い探索深度と標準偏差モデルを変えず、分位点 `z` の固定倍率だけを0.90～1.00へ変更した候補。",
            "",
            "**算出方法:** 倍率ごとに残差分位点と整数誤差幅を再計算し、保存済みデータから枝刈り回数と推定訪問局面数を求めた事後計算。",
            "",
            "**表の読み方:** 誤った枝刈りは少ない方がよい。推定訪問局面数比は小さい方が事後計算上の訪問局面数が少ないが、実測時間ではない。",
            "",
            markdown_table(end_z_table, [("dataset", "データ"), ("scale", "分位点 z の倍率"), ("contexts", "判定機会"), ("cuts", "枝刈り"), ("wrong", "誤った枝刈り"), ("wrong2", "誤枝刈りが境界から2石以上"), ("wrong4", "誤枝刈りが境界から4石以上"), ("nodes", "推定訪問局面数"), ("ratio", "推定訪問局面数比")]),
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
            "この表は、予定した測定が指定件数まで実行されたかを示す。速度または着手精度の比較結果ではない。",
            "",
            markdown_table(status_table, [("measurement", "測定"), ("status", "状態"), ("required", "完了条件")]),
            "",
            "## 5. 再生成",
            "",
            "```powershell",
            "python benchmark/mpc_depth_aggressiveness_20260830/aggregate_results.py",
            "```",
            "",
            "上の実行手順は既存の測定結果を読み、入力条件と件数を検査してから、集計用の表形式ファイルとこのレポートを更新する。追加の測定結果が作成された場合も、同じ実行手順で表へ反映される。",
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
    write_csv(output_dir / "end_generic_model_coefficients.csv", end_coeffs, ["model", "label", "representation", "reference_weight", "a", "b", "c", "d", "e", "f", "global_sigma_multiplier"])
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
