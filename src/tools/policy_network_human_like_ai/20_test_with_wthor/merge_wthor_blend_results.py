#!/usr/bin/env python3
"""
Merge shard outputs from evaluate_wthor_blend_human_match.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def resolve_result_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "wthor_blend_human_match.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_results(inputs: Sequence[Path]) -> List[dict]:
    results = []
    for path in inputs:
        result_path = resolve_result_path(path)
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        result["_source_file"] = str(result_path)
        results.append(result)
    if not results:
        raise ValueError("no input results")
    return results


def read_input_list(path: Path) -> List[Path]:
    inputs: List[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            inputs.append(Path(stripped))
    return inputs


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_topn(results: Sequence[dict]) -> List[dict]:
    merged: Dict[Tuple[float, int], dict] = {}
    for result in results:
        for row in result["topn"]:
            key = (float(row["blend_param"]), int(row["top_n"]))
            dst = merged.setdefault(
                key,
                {
                    "blend_param": key[0],
                    "top_n": key[1],
                    "exact_hits": 0,
                    "symmetric_hits": 0,
                    "positions": 0,
                },
            )
            dst["exact_hits"] += int(row["exact_hits"])
            dst["symmetric_hits"] += int(row["symmetric_hits"])
            dst["positions"] += int(row["positions"])
    rows = []
    for key in sorted(merged):
        row = merged[key]
        positions = row["positions"]
        hits = row["symmetric_hits"]
        accuracy = hits / positions if positions else 0.0
        rows.append(
            {
                **row,
                "hits": hits,
                "accuracy": accuracy,
                "exact_accuracy": row["exact_hits"] / positions if positions else 0.0,
                "symmetric_accuracy": accuracy,
            }
        )
    return rows


def merge_buckets(results: Sequence[dict]) -> List[dict]:
    merged: Dict[Tuple[float, str, int], dict] = {}
    for result in results:
        for row in result["move_bucket_topn"]:
            key = (float(row["blend_param"]), str(row["move_bucket"]), int(row["top_n"]))
            dst = merged.setdefault(
                key,
                {
                    "blend_param": key[0],
                    "move_bucket": key[1],
                    "top_n": key[2],
                    "symmetric_hits": 0,
                    "positions": 0,
                },
            )
            dst["symmetric_hits"] += int(row["symmetric_hits"])
            dst["positions"] += int(row["positions"])
    rows = []
    for key in sorted(merged):
        row = merged[key]
        positions = row["positions"]
        rows.append({**row, "symmetric_accuracy": row["symmetric_hits"] / positions if positions else 0.0})
    return rows


def merge_hint_cache_stats(results: Sequence[dict]) -> dict:
    merged = {
        "path": None,
        "lookups": 0,
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "rows": None,
    }
    for result in results:
        stats = result.get("hint_cache_stats", {})
        if not stats:
            continue
        merged["path"] = stats.get("path") or result.get("hint_cache_db") or merged["path"]
        for key in ("lookups", "hits", "misses", "writes"):
            merged[key] += int(stats.get(key) or 0)
        if stats.get("rows") is not None:
            merged["rows"] = max(int(stats["rows"]), int(merged["rows"] or 0))
    return merged


def merge_results(results: Sequence[dict]) -> dict:
    first = results[0]
    topn = merge_topn(results)
    bucket_rows = merge_buckets(results)
    raw_hint_samples = []
    worker_summaries = []
    for result in results:
        raw_hint_samples.extend(result.get("raw_hint_samples", []))
        worker_summaries.extend(result.get("worker_summaries", []))
    return {
        "board_data_dir": first["board_data_dir"],
        "weights": first["weights"],
        "egaroucid_exe": first["egaroucid_exe"],
        "egaroucid_level": first["egaroucid_level"],
        "blend_params": first["blend_params"],
        "available_positions": first["available_positions"],
        "range_start": min(int(result.get("range_start", 0)) for result in results),
        "range_end": max(int(result.get("range_end", result.get("available_positions", 0))) for result in results),
        "sample_positions": None,
        "sample_seed": None,
        "jobs": sum(int(result.get("jobs", 1)) for result in results),
        "hint_cache_db": first.get("hint_cache_db"),
        "hint_cache_stats": merge_hint_cache_stats(results),
        "agreement_definition": {
            "primary_metric": "symmetry_aware",
            "description": (
                "手番側と相手側の石配置をそれぞれ不変に保つ盤面対称変換で、"
                "人間の実着手から移る合法手を同値手とする。"
                "同値手のいずれかが上位N手に入れば一致と数える。"
            ),
            "exact_metric_role": "診断用。正式な着手一致率には使用しない。",
        },
        "invalid_policy_samples": sum(int(result.get("invalid_policy_samples", 0)) for result in results),
        "illegal_label_samples": sum(int(result.get("illegal_label_samples", 0)) for result in results),
        "topn": topn,
        "move_bucket_topn": bucket_rows,
        "raw_hint_samples": raw_hint_samples,
        "worker_summaries": worker_summaries,
        "merged_sources": [result["_source_file"] for result in results],
    }


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge WTHOR blend agreement shard outputs.")
    parser.add_argument("inputs", nargs="*", type=Path, help="Shard output directories or wthor_blend_human_match.json files.")
    parser.add_argument("--input-list", action="append", type=Path, default=[], help="Text file with one shard output path per line.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    inputs = list(args.inputs)
    for input_list in args.input_list:
        inputs.extend(read_input_list(input_list))
    results = load_results(inputs)
    merged = merge_results(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wthor_blend_human_match.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    write_csv(
        args.output_dir / "wthor_blend_human_match_topn.csv",
        merged["topn"],
        [
            "blend_param",
            "top_n",
            "hits",
            "positions",
            "accuracy",
            "symmetric_hits",
            "symmetric_accuracy",
            "exact_hits",
            "exact_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "wthor_blend_human_match_by_move10.csv",
        merged["move_bucket_topn"],
        ["blend_param", "move_bucket", "top_n", "symmetric_hits", "positions", "symmetric_accuracy"],
    )
    print("merged_sources", len(results))
    print("positions", merged["topn"][0]["positions"] if merged["topn"] else 0)
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
