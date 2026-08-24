#!/usr/bin/env python3
"""Recompute a deterministic sample of exact endgame labels with a cold TT."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_result(output: str) -> dict[str, int] | None:
    for line in output.splitlines():
        if not line.startswith("END_PROBCUT_SCORE_V2\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 11:
            continue
        try:
            (
                _, deep, shallow, shallow_value, deep_value, shallow_nodes,
                deep_nodes, shallow_ms, deep_ms, d0, legal,
            ) = fields
            return {
                "deep_depth": int(deep), "shallow_depth": int(shallow),
                "shallow_value": int(shallow_value), "deep_value": int(deep_value),
                "shallow_nodes": int(shallow_nodes), "deep_nodes": int(deep_nodes),
                "shallow_ms": int(shallow_ms), "deep_ms": int(deep_ms),
                "d0_value": int(d0), "legal_count": int(legal),
            }
        except ValueError:
            continue
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.exe = args.exe.resolve()
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.input:
        for row in load_jsonl(path):
            key = (str(row["board"]), int(row["deep_depth"]))
            previous = grouped.get(key)
            if previous is not None and int(previous["deep_value"]) != int(row["deep_value"]):
                raise ValueError(f"inconsistent stored deep labels for {key}")
            grouped.setdefault(key, row)
    ordered = sorted(
        grouped.values(),
        key=lambda row: hashlib.sha256(
            f"{args.seed}|{row['board']}|{row['deep_depth']}".encode()
        ).digest(),
    )[: args.samples]

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        shallow = int(row.get("trace_shallow_depth", row["shallow_depth"]))
        try:
            completed = subprocess.run(
                [
                    str(args.exe), "score-grid", str(row["board"]),
                    str(row["deep_depth"]), str(shallow),
                ],
                capture_output=True, text=True, timeout=args.timeout,
                check=False, cwd=args.exe.parent,
            )
            parsed = parse_result(completed.stdout)
            return {
                "board": row["board"], "root_id": row.get("root_id", ""),
                "stored_deep_value": int(row["deep_value"]),
                "stored_d0_value": int(row["d0_value"]),
                "stored_legal_count": int(row["legal_count"]),
                "return_code": completed.returncode, "timed_out": False,
                "result": parsed, "stderr_tail": completed.stderr[-500:],
            }
        except subprocess.TimeoutExpired as error:
            return {
                "board": row["board"], "root_id": row.get("root_id", ""),
                "stored_deep_value": int(row["deep_value"]),
                "stored_d0_value": int(row["d0_value"]),
                "stored_legal_count": int(row["legal_count"]),
                "return_code": None, "timed_out": True, "result": None,
                "stderr_tail": str(error.stderr or "")[-500:],
            }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(execute, ordered))
    completed_rows = [row for row in rows if row["result"] is not None]
    deep_mismatches = [
        row for row in completed_rows
        if row["stored_deep_value"] != row["result"]["deep_value"]
    ]
    feature_mismatches = [
        row for row in completed_rows
        if (
            row["stored_d0_value"] != row["result"]["d0_value"] or
            row["stored_legal_count"] != row["result"]["legal_count"]
        )
    ]
    report = {
        "requested": len(ordered), "completed": len(completed_rows),
        "timed_out": sum(row["timed_out"] for row in rows),
        "failed": sum(row["return_code"] not in (0, None) for row in rows),
        "deep_mismatch_count": len(deep_mismatches),
        "feature_mismatch_count": len(feature_mismatches),
        "deep_mismatches": deep_mismatches,
        "feature_mismatches": feature_mismatches,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"requested={report['requested']} completed={report['completed']} "
        f"timeouts={report['timed_out']} failures={report['failed']} "
        f"deep_mismatches={report['deep_mismatch_count']} "
        f"feature_mismatches={report['feature_mismatch_count']}"
    )
    return 0 if len(completed_rows) == len(ordered) and not deep_mismatches and not feature_mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
