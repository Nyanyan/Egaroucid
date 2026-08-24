#!/usr/bin/env python3
"""Collect reproducible endgame ProbCut contexts and exact labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        file.flush()


def parse_board(text: str) -> tuple[list[str], str]:
    stripped = text.strip()
    if len(stripped) < 65:
        raise ValueError(f"invalid board: {text!r}")
    cells = list(stripped[:64])
    side = stripped[64:].strip()[0]
    if side not in "XO" or any(cell not in "XO-" for cell in cells):
        raise ValueError(f"invalid board: {text!r}")
    return cells, side


def board_text(cells: list[str], side: str) -> str:
    return "".join(cells) + " " + side


def flips_for(cells: list[str], side: str, move: int) -> list[int]:
    if cells[move] != "-":
        return []
    opponent = "O" if side == "X" else "X"
    row, column = divmod(move, 8)
    result: list[int] = []
    for dr, dc in DIRECTIONS:
        r, c = row + dr, column + dc
        line: list[int] = []
        while 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == opponent:
            line.append(r * 8 + c)
            r += dr
            c += dc
        if line and 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == side:
            result.extend(line)
    return result


def legal_moves(cells: list[str], side: str) -> list[tuple[int, list[int]]]:
    result = []
    for move, cell in enumerate(cells):
        if cell == "-":
            flips = flips_for(cells, side, move)
            if flips:
                result.append((move, flips))
    return result


def deterministic_descendant(root: str, target_empty: int, seed: int) -> str | None:
    cells, side = parse_board(root)
    generator = random.Random(seed)
    while cells.count("-") > target_empty:
        moves = legal_moves(cells, side)
        if not moves:
            side = "O" if side == "X" else "X"
            if not legal_moves(cells, side):
                return None
            continue
        move, flips = moves[generator.randrange(len(moves))]
        cells[move] = side
        for square in flips:
            cells[square] = side
        side = "O" if side == "X" else "X"
    return board_text(cells, side)


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


def descendants(roots: list[dict[str, Any]], target_empty: int, count: int, seed: int) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for index, root in enumerate(roots):
        root_board = str(root["board"])
        root_id = stable_id(str(root.get("source", "")), str(root.get("game_id", root.get("source_id", index))), root_board)
        for descendant_index in range(count):
            local_seed = int.from_bytes(hashlib.sha256(f"{seed}|{root_id}|{descendant_index}".encode()).digest()[:8], "big")
            selected = deterministic_descendant(root_board, target_empty, local_seed)
            if selected is None or selected in seen:
                continue
            seen.add(selected)
            result.append({
                "board": selected,
                "trace_id": stable_id(root_id, selected),
                "root_id": root_id,
                "root_board": root_board,
                "descendant_index": descendant_index,
                "source": root.get("source", ""),
                "source_id": root.get("game_id", root.get("source_id", f"line-{index + 1}")),
            })
    return result


def filter_roots(
    roots: list[dict[str, Any]],
    excluded_samples: list[Path],
    max_per_empty: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    excluded = {
        str(row.get("root_board", ""))
        for path in excluded_samples
        for row in load_jsonl(path)
    }
    candidates = [row for row in roots if str(row["board"]) not in excluded]
    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}|{row['board']}|{row.get('source', '')}".encode()
        ).digest()
    )
    counts: dict[int, int] = {}
    selected = []
    for row in candidates:
        empties = int(row.get("empties", str(row["board"])[:64].count("-")))
        if counts.get(empties, 0) >= max_per_empty:
            continue
        counts[empties] = counts.get(empties, 0) + 1
        selected.append(row)
    return selected, len(excluded)


def run_process(command: list[str], timeout: float, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            check=False, cwd=cwd,
        )
        return {"return_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"return_code": None, "stdout": error.stdout or "", "stderr": error.stderr or "", "timed_out": True}


def parse_trace(output: str, item: dict[str, Any], level: int) -> list[dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        if not line.startswith("END_PROBCUT_CONTEXT_V2\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 15:
            continue
        (
            _, board, deep, shallow, n_discs, root_distance, alpha, beta,
            direction, d0, legal, recorded_level, threshold, gate_passed,
            static_probe,
        ) = parts
        if int(recorded_level) != level:
            continue
        rows.append({
            **item,
            "board": board,
            "deep_depth": int(deep),
            "shallow_depth": int(shallow),
            "trace_shallow_depth": int(shallow),
            "n_discs": int(n_discs),
            "root_distance": int(root_distance),
            "alpha": int(alpha),
            "beta": int(beta),
            "direction": direction,
            "boundary": int(beta) if direction == "high" else int(alpha),
            "trace_d0_value": int(d0),
            "trace_legal_count": int(legal),
            "trace_threshold": int(threshold),
            "gate_passed": bool(int(gate_passed)),
            "static_probe": bool(int(static_probe)),
            "mpc_level": level,
            "requested_mpc_level": level,
            "trace_root_board": item["board"],
        })
    return rows


def context_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["board"], row["deep_depth"],
        row.get("trace_shallow_depth", row["shallow_depth"]),
        row["alpha"], row["beta"], row["direction"], row["mpc_level"],
        bool(row.get("static_probe", False)),
    )


def sample_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (*context_key(row), int(row["shallow_depth"]))


def round_robin_contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["root_id"]), int(row["mpc_level"])), []).append(row)
    for key in groups:
        groups[key].sort(key=lambda row: hashlib.sha256(repr(context_key(row)).encode()).digest())
    keys = sorted(groups, key=lambda key: hashlib.sha256(repr(key).encode()).digest())
    result = []
    index = 0
    while True:
        added = False
        for key in keys:
            if index < len(groups[key]):
                result.append(groups[key][index])
                added = True
        if not added:
            return result
        index += 1


def trace_phase(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    runs_path = args.output / "trace_runs.jsonl"
    contexts_path = args.output / "contexts.jsonl"
    completed = {(row["trace_id"], int(row["requested_mpc_level"])) for row in load_jsonl(runs_path)}
    known = {context_key(row) for row in load_jsonl(contexts_path)}
    jobs = [(item, level) for item in items for level in args.mpc_levels if (item["trace_id"], level) not in completed]
    def execute(job: tuple[dict[str, Any], int]) -> tuple[dict[str, Any], int, dict[str, Any]]:
        item, level = job
        result = run_process([
            str(args.exe), "trace", item["board"], str(level),
            str(args.trace_contexts_per_depth), str(args.min_deep), str(args.max_deep),
        ], args.trace_timeout, args.exe.parent)
        return item, level, result
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (item, level, result) in enumerate(executor.map(execute, jobs), 1):
            parsed = parse_trace(str(result["stdout"]) + "\n" + str(result["stderr"]), item, level)
            added = 0
            for row in parsed:
                key = context_key(row)
                if key not in known and args.min_deep <= int(row["deep_depth"]) <= args.max_deep:
                    append_jsonl(contexts_path, row)
                    known.add(key)
                    added += 1
            append_jsonl(runs_path, {"trace_id": item["trace_id"], "root_id": item["root_id"], "requested_mpc_level": level, "contexts_added": added, "timed_out": result["timed_out"], "return_code": result["return_code"]})
            print(f"trace {index}/{len(jobs)} added={added} contexts={len(known)} timeout={result['timed_out']}", flush=True)


def candidate_shallow_depths(row: dict[str, Any], offsets: list[int]) -> list[int]:
    deep_depth = int(row["deep_depth"])
    original = int(row.get("trace_shallow_depth", row["shallow_depth"]))
    if bool(row.get("static_probe", False)):
        return [0]
    depths = {0}
    for offset in offsets:
        depth = original + offset
        if 0 < depth < deep_depth and (depth & 1) == (deep_depth & 1):
            depths.add(depth)
    return sorted(depths)


def parse_score_grid(output: str) -> dict[int, dict[str, int]]:
    result = {}
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
            result[int(shallow)] = {
                "deep_depth": int(deep),
                "shallow_value": int(shallow_value),
                "deep_value": int(deep_value),
                "shallow_nodes": int(shallow_nodes),
                "deep_nodes": int(deep_nodes),
                "shallow_ms": int(shallow_ms),
                "deep_ms": int(deep_ms),
                "d0_value": int(d0),
                "legal_count": int(legal),
            }
        except ValueError:
            continue
    return result


def score_phase(args: argparse.Namespace) -> None:
    contexts = round_robin_contexts(load_jsonl(args.output / "contexts.jsonl"))
    samples_path = args.output / "samples.jsonl"
    status_path = args.output / "score_status.jsonl"
    known = {sample_key(row) for row in load_jsonl(samples_path)}
    remaining = max(0, args.max_samples - len(known))
    if remaining == 0:
        print(f"score complete samples={len(known)}", flush=True)
        return
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    scheduled = 0
    for row in contexts:
        missing = [
            depth for depth in candidate_shallow_depths(row, args.shallow_offsets)
            if (*context_key(row), depth) not in known
        ]
        if not missing:
            continue
        group_key = (str(row["board"]), int(row["deep_depth"]))
        grouped.setdefault(group_key, []).append(row)
        scheduled += len(missing)
        if scheduled >= remaining:
            break
    jobs = list(grouped.items())

    def execute(job: tuple[tuple[str, int], list[dict[str, Any]]]) -> tuple[tuple[str, int], list[dict[str, Any]], dict[str, Any]]:
        (board, deep_depth), rows = job
        depths = sorted({
            depth
            for row in rows
            for depth in candidate_shallow_depths(row, args.shallow_offsets)
            if (*context_key(row), depth) not in known
        })
        result = run_process([
            str(args.exe), "score-grid", board, str(deep_depth),
            ",".join(str(depth) for depth in depths),
        ], args.score_timeout, args.exe.parent)
        return (board, deep_depth), rows, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (group_key, rows, process_result) in enumerate(executor.map(execute, jobs), 1):
            grid = parse_score_grid(str(process_result["stdout"]))
            accepted = 0
            if not process_result["timed_out"] and process_result["return_code"] == 0:
                for row in rows:
                    for shallow_depth in candidate_shallow_depths(row, args.shallow_offsets):
                        key = (*context_key(row), shallow_depth)
                        values = grid.get(shallow_depth)
                        if key in known or values is None or len(known) >= args.max_samples:
                            continue
                        if (
                            values["deep_depth"] != int(row["deep_depth"]) or
                            values["d0_value"] != int(row["trace_d0_value"]) or
                            values["legal_count"] != int(row["trace_legal_count"])
                        ):
                            continue
                        sample = dict(row)
                        sample["trace_shallow_depth"] = int(row.get("trace_shallow_depth", row["shallow_depth"]))
                        sample["shallow_depth"] = shallow_depth
                        sample.update(values)
                        sample.update({"deep_completed": True, "deep_mpc_level": 6})
                        append_jsonl(samples_path, sample)
                        known.add(key)
                        accepted += 1
            append_jsonl(status_path, {
                "group_key": list(group_key), "context_count": len(rows),
                "accepted": accepted, "timed_out": process_result["timed_out"],
                "return_code": process_result["return_code"],
                "stderr_tail": str(process_result["stderr"])[-500:],
            })
            print(f"score {index}/{len(jobs)} accepted={accepted} samples={len(known)} timeout={process_result['timed_out']}", flush=True)


def parse_levels(text: str) -> list[int]:
    return [int(value) for value in text.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("trace", "score", "all"), default="all")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--descendant-empty", type=int, default=18)
    parser.add_argument("--descendants-per-root", type=int, default=2)
    parser.add_argument("--mpc-levels", type=parse_levels, default=[0, 1, 2])
    parser.add_argument("--trace-timeout", type=float, default=5.0)
    parser.add_argument("--score-timeout", type=float, default=20.0)
    parser.add_argument("--min-deep", type=int, default=8)
    parser.add_argument("--max-deep", type=int, default=18)
    parser.add_argument("--trace-contexts-per-depth", type=int, default=8)
    parser.add_argument("--shallow-offsets", type=parse_levels, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exclude-samples", type=Path, action="append", default=[])
    parser.add_argument("--max-roots-per-empty", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.exe = args.exe.resolve()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    input_roots = load_jsonl(args.positions)
    roots, excluded_root_count = filter_roots(
        input_roots, args.exclude_samples, args.max_roots_per_empty, args.seed
    )
    items = descendants(roots, args.descendant_empty, args.descendants_per_root, args.seed)
    metadata_path = args.output / "metadata.json"
    if not metadata_path.exists():
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": str(args.positions.resolve()),
            "input_sha256": sha256_file(args.positions),
            "executable": str(args.exe),
            "executable_sha256": sha256_file(args.exe),
            "seed": args.seed,
            "root_count": len(roots),
            "input_root_count": len(input_roots),
            "excluded_development_root_count": excluded_root_count,
            "exclude_samples": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.exclude_samples
            ],
            "descendant_count": len(items),
            "descendant_empty": args.descendant_empty,
            "descendants_per_root": args.descendants_per_root,
            "mpc_levels": args.mpc_levels,
            "trace_timeout": args.trace_timeout,
            "trace_contexts_per_depth": args.trace_contexts_per_depth,
            "score_timeout": args.score_timeout,
            "shallow_offsets": args.shallow_offsets,
            "workers": args.workers,
            "split_unit": "root_id",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.phase in ("trace", "all"):
        trace_phase(args, items)
    if args.phase in ("score", "all"):
        score_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
