#!/usr/bin/env python3
"""Add alternative shallow-depth scores to exact-labelled endgame MPC samples."""

from __future__ import annotations

import argparse
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def parse_int_list(text: str) -> list[int]:
    return [int(value) for value in text.split(",")]


def context_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("root_id", "")), str(row["board"]), int(row["deep_depth"]),
        int(row.get("trace_shallow_depth", row["shallow_depth"])),
        int(row["alpha"]), int(row["beta"]), str(row["direction"]),
        int(row["mpc_level"]),
    )


def sample_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (*context_key(row), int(row["shallow_depth"]))


def candidate_depths(row: dict[str, Any], offsets: list[int]) -> list[int]:
    deep = int(row["deep_depth"])
    original = int(row.get("trace_shallow_depth", row["shallow_depth"]))
    result = {0}
    for offset in offsets:
        depth = original + offset
        if 0 < depth < deep and (depth & 1) == (deep & 1):
            result.add(depth)
    return sorted(result)


def parse_grid(output: str) -> dict[int, dict[str, int]]:
    result = {}
    for line in output.splitlines():
        if not line.startswith("END_PROBCUT_SHALLOW_V2\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 8:
            continue
        try:
            _, deep, shallow, value, nodes, elapsed, d0, legal = fields
            result[int(shallow)] = {
                "grid_deep_depth": int(deep),
                "shallow_value": int(value),
                "shallow_nodes": int(nodes),
                "shallow_ms": int(elapsed),
                "d0_value": int(d0),
                "legal_count": int(legal),
            }
        except ValueError:
            continue
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", type=parse_int_list, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.exe = args.exe.resolve()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "samples.jsonl"
    status_path = args.output / "status.jsonl"
    source_rows = load_jsonl(args.input)
    known = {
        sample_key(row) for row in load_jsonl(output_path)
    } if output_path.exists() else set()

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in source_rows:
        group_key = (str(row["board"]), int(row["deep_depth"]))
        if any((*context_key(row), depth) not in known for depth in candidate_depths(row, args.offsets)):
            groups.setdefault(group_key, []).append(row)
    jobs = list(groups.items())
    if args.max_groups is not None:
        jobs = jobs[: args.max_groups]

    def execute(job: tuple[tuple[str, int], list[dict[str, Any]]]) -> tuple[tuple[str, int], list[dict[str, Any]], dict[str, Any]]:
        (board, deep), rows = job
        depths = sorted({depth for row in rows for depth in candidate_depths(row, args.offsets)})
        try:
            completed = subprocess.run(
                [str(args.exe), "shallow-grid", board, str(deep), ",".join(map(str, depths))],
                capture_output=True, text=True, timeout=args.timeout, check=False,
                cwd=args.exe.parent,
            )
            process = {
                "return_code": completed.returncode, "stdout": completed.stdout,
                "stderr": completed.stderr, "timed_out": False,
            }
        except subprocess.TimeoutExpired as error:
            process = {
                "return_code": None, "stdout": error.stdout or "",
                "stderr": error.stderr or "", "timed_out": True,
            }
        return (board, deep), rows, process

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (group_key, rows, process) in enumerate(executor.map(execute, jobs), 1):
            grid = parse_grid(str(process["stdout"]))
            accepted = 0
            if not process["timed_out"] and process["return_code"] == 0:
                for row in rows:
                    original = int(row["shallow_depth"])
                    for depth in candidate_depths(row, args.offsets):
                        key = (*context_key(row), depth)
                        values = grid.get(depth)
                        if key in known or values is None:
                            continue
                        if (
                            values["grid_deep_depth"] != int(row["deep_depth"]) or
                            values["d0_value"] != int(row["d0_value"]) or
                            values["legal_count"] != int(row["legal_count"])
                        ):
                            continue
                        sample = dict(row)
                        sample["trace_shallow_depth"] = original
                        sample["shallow_depth"] = depth
                        sample.update(values)
                        sample.pop("grid_deep_depth")
                        append_jsonl(output_path, sample)
                        known.add(key)
                        accepted += 1
            append_jsonl(status_path, {
                "board": group_key[0], "deep_depth": group_key[1],
                "contexts": len(rows), "accepted": accepted,
                "timed_out": process["timed_out"],
                "return_code": process["return_code"],
                "stderr_tail": str(process["stderr"])[-500:],
            })
            print(
                f"group {index}/{len(jobs)} accepted={accepted} samples={len(known)} "
                f"timeout={process['timed_out']}", flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
