#!/usr/bin/env python3
"""中盤MPCの浅い探索深度、実行条件、上側・下側の係数を決める。

係数を決める際には、評価値の平均二乗誤差ではなく、MPCを実行した場合の
推定探索時間と誤った枝刈りの回数を直接使う。入力TSVは
mid_probcut_dataset_tool.cpp が空の置換表で収集したものを使う。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

Z_VALUES = np.array([1.13, 1.55, 1.81, 2.088, 2.1845, 2.7965])
BOUNDARIES = np.arange(-16, 17, dtype=np.int16)
CURRENT_COEFFICIENTS = np.array([
    0.8335834703936896,
    -4.71778909968251,
    1.1467905781538477,
    -0.5274699259330169,
    6.5091001393587335,
    3.9546352081550378,
    1.8719077939546169,
], dtype=np.float64)
CURRENT_GATE_SLACK = 3


@dataclass(frozen=True)
class Measurement:
    board: str
    n_discs: int
    deep_depth: int
    shallow_depth: int
    static_value: int
    shallow_value: int
    deep_value: int
    shallow_nodes: int
    deep_nodes: int
    shallow_ms: int
    deep_ms: int


@dataclass
class Position:
    board: str
    n_discs: int
    deep_depth: int
    static_value: int
    deep_value: int
    deep_nodes: int
    deep_ms: int
    shallow: dict[int, Measurement]


@dataclass(frozen=True)
class Totals:
    scenarios: int = 0
    attempts: int = 0
    cuts: int = 0
    wrong: int = 0
    wrong_2plus: int = 0
    wrong_4plus: int = 0
    time_ms: int = 0
    nodes: int = 0

    def __add__(self, other: "Totals") -> "Totals":
        return Totals(**{
            field: getattr(self, field) + getattr(other, field)
            for field in self.__dataclass_fields__
        })

    def as_dict(self) -> dict[str, int | float]:
        return {
            **{field: int(getattr(self, field)) for field in self.__dataclass_fields__},
            "cut_rate": self.cuts / self.scenarios if self.scenarios else 0.0,
            "wrong_rate_per_cut": self.wrong / self.cuts if self.cuts else 0.0,
        }


def read_tsv(paths: Iterable[Path]) -> list[Measurement]:
    measurements: dict[tuple[str, int, int], Measurement] = {}
    for path in paths:
        prefix = path.read_bytes()[:2]
        encoding = "utf-16" if prefix in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        with path.open(encoding=encoding, newline="") as source:
            for raw in csv.DictReader(source, delimiter="\t"):
                row = Measurement(
                    board=raw["board"],
                    n_discs=int(raw["n_discs"]),
                    deep_depth=int(raw["deep_depth"]),
                    shallow_depth=int(raw["shallow_depth"]),
                    static_value=int(raw["static_value"]),
                    shallow_value=int(raw["shallow_value"]),
                    deep_value=int(raw["deep_value"]),
                    shallow_nodes=int(raw["shallow_nodes"]),
                    deep_nodes=int(raw["deep_nodes"]),
                    shallow_ms=int(raw["shallow_time_ms"]),
                    deep_ms=int(raw["deep_time_ms"]),
                )
                key = (row.board, row.deep_depth, row.shallow_depth)
                measurements.setdefault(key, row)
    return list(measurements.values())


def group_positions(rows: Iterable[Measurement]) -> list[Position]:
    grouped: dict[tuple[str, int], Position] = {}
    for row in rows:
        key = (row.board, row.deep_depth)
        if key not in grouped:
            grouped[key] = Position(
                board=row.board,
                n_discs=row.n_discs,
                deep_depth=row.deep_depth,
                static_value=row.static_value,
                deep_value=row.deep_value,
                deep_nodes=row.deep_nodes,
                deep_ms=row.deep_ms,
                shallow={},
            )
        grouped[key].shallow[row.shallow_depth] = row
    return list(grouped.values())


def split_number(board: str, parts: int = 5) -> int:
    digest = hashlib.sha256(board.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % parts


def current_shallow_depth(deep_depth: int) -> int:
    return ((deep_depth * 2 // 5) & ~1) + (deep_depth & 1)


def sigma_values(
    n_discs: np.ndarray,
    shallow_depth: int,
    deep_depth: int,
    coefficients: np.ndarray,
) -> np.ndarray:
    x = (
        coefficients[0] * (n_discs / 64.0)
        + coefficients[1] * (shallow_depth / 60.0)
        + coefficients[2] * (deep_depth / 60.0)
    )
    return ((coefficients[3] * x + coefficients[4]) * x + coefficients[5]) * x + coefficients[6]


def simulate(
    positions: list[Position],
    shallow_depth: int,
    direction: str,
    level: int,
    coefficients: np.ndarray,
    gate_slack: int,
) -> Totals:
    selected = [position for position in positions if shallow_depth in position.shallow]
    if not selected:
        return Totals()
    n_discs = np.array([position.n_discs for position in selected], dtype=np.float64)
    static = np.array([position.static_value for position in selected], dtype=np.int16)[:, None]
    shallow = np.array([
        position.shallow[shallow_depth].shallow_value for position in selected
    ], dtype=np.int16)[:, None]
    deep = np.array([position.deep_value for position in selected], dtype=np.int16)[:, None]
    shallow_ms = np.array([
        position.shallow[shallow_depth].shallow_ms for position in selected
    ], dtype=np.int64)[:, None]
    deep_ms = np.array([position.deep_ms for position in selected], dtype=np.int64)[:, None]
    shallow_nodes = np.array([
        position.shallow[shallow_depth].shallow_nodes for position in selected
    ], dtype=np.int64)[:, None]
    deep_nodes = np.array([position.deep_nodes for position in selected], dtype=np.int64)[:, None]
    sigma = sigma_values(n_discs, shallow_depth, selected[0].deep_depth, coefficients)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.10) or np.any(sigma > 40.0):
        return Totals(time_ms=2**62, nodes=2**62)
    margin = np.ceil(Z_VALUES[level] * sigma - 1.0e-12).astype(np.int16)[:, None]
    boundary = BOUNDARIES[None, :]
    if shallow_depth == 0:
        attempt = np.ones((len(selected), len(BOUNDARIES)), dtype=bool)
        if direction == "high":
            cut = static >= boundary + margin
            correct = deep >= boundary
            error = np.maximum(0, boundary - deep)
        else:
            cut = static <= boundary - margin
            correct = deep <= boundary
            error = np.maximum(0, deep - boundary)
        probe_ms = np.zeros_like(deep_ms)
        probe_nodes = np.zeros_like(deep_nodes)
    else:
        if direction == "high":
            attempt = static >= boundary + margin - gate_slack
            cut = attempt & (shallow >= boundary + margin)
            correct = deep >= boundary
            error = np.maximum(0, boundary - deep)
        else:
            attempt = static <= boundary - margin + gate_slack
            cut = attempt & (shallow <= boundary - margin)
            correct = deep <= boundary
            error = np.maximum(0, deep - boundary)
        probe_ms = shallow_ms
        probe_nodes = shallow_nodes
    wrong = cut & ~correct
    total_time = np.where(attempt, probe_ms, 0) + np.where(cut, 0, deep_ms)
    total_nodes = np.where(attempt, probe_nodes, 0) + np.where(cut, 0, deep_nodes)
    return Totals(
        scenarios=int(cut.size),
        attempts=int(np.count_nonzero(attempt)),
        cuts=int(np.count_nonzero(cut)),
        wrong=int(np.count_nonzero(wrong)),
        wrong_2plus=int(np.count_nonzero(wrong & (error >= 2))),
        wrong_4plus=int(np.count_nonzero(wrong & (error >= 4))),
        time_ms=int(np.sum(total_time)),
        nodes=int(np.sum(total_nodes)),
    )


def combined_levels(
    positions: list[Position],
    shallow_depth: int,
    direction: str,
    coefficients: np.ndarray,
    gate_slack: int,
) -> tuple[Totals, list[Totals]]:
    per_level = [
        simulate(
            positions, shallow_depth, direction, level,
            coefficients, gate_slack,
        )
        for level in range(len(Z_VALUES))
    ]
    total = Totals()
    for values in per_level:
        total += values
    return total, per_level


def error_not_greater(candidate: list[Totals], baseline: list[Totals]) -> bool:
    return all(
        new.wrong_2plus <= old.wrong_2plus
        and new.wrong_4plus <= old.wrong_4plus
        for new, old in zip(candidate, baseline)
    )


def choose_depth_and_gate(
    train: list[Position],
    check: list[Position],
    deep_depth: int,
) -> dict[str, object]:
    current_depth = current_shallow_depth(deep_depth)
    available = sorted(set.intersection(*(
        set(position.shallow) for position in train + check
    )))
    available = [depth for depth in available if depth == 0 or (depth & 1) == (deep_depth & 1)]
    baseline: dict[str, dict[str, object]] = {}
    for direction in ("high", "low"):
        train_total, train_levels = combined_levels(
            train, current_depth, direction, CURRENT_COEFFICIENTS,
            CURRENT_GATE_SLACK,
        )
        check_total, check_levels = combined_levels(
            check, current_depth, direction, CURRENT_COEFFICIENTS,
            CURRENT_GATE_SLACK,
        )
        baseline[direction] = {
            "train_total": train_total,
            "train_levels": train_levels,
            "check_total": check_total,
            "check_levels": check_levels,
        }

    candidates = []
    for shallow_depth in available:
        slacks = [0] if shallow_depth == 0 else list(range(0, 9))
        direction_candidates: dict[str, list[dict[str, object]]] = {"high": [], "low": []}
        for direction in direction_candidates:
            for slack in slacks:
                train_total, train_levels = combined_levels(
                    train, shallow_depth, direction, CURRENT_COEFFICIENTS, slack
                )
                check_total, check_levels = combined_levels(
                    check, shallow_depth, direction, CURRENT_COEFFICIENTS, slack
                )
                if not error_not_greater(
                    train_levels, baseline[direction]["train_levels"]
                ):
                    continue
                if not error_not_greater(
                    check_levels, baseline[direction]["check_levels"]
                ):
                    continue
                direction_candidates[direction].append({
                    "slack": slack,
                    "train_total": train_total,
                    "train_levels": train_levels,
                    "check_total": check_total,
                    "check_levels": check_levels,
                })
        if not direction_candidates["high"] or not direction_candidates["low"]:
            continue
        best_high = min(direction_candidates["high"], key=lambda row: row["train_total"].time_ms)
        best_low = min(direction_candidates["low"], key=lambda row: row["train_total"].time_ms)
        candidates.append({
            "shallow_depth": shallow_depth,
            "high": best_high,
            "low": best_low,
            "train_time_ms": best_high["train_total"].time_ms + best_low["train_total"].time_ms,
            "check_time_ms": best_high["check_total"].time_ms + best_low["check_total"].time_ms,
        })
    selected = min(candidates, key=lambda row: row["train_time_ms"])
    return {
        "deep_depth": deep_depth,
        "current_shallow_depth": current_depth,
        "available_shallow_depths": available,
        "selected": selected,
        "baseline": baseline,
        "candidates": candidates,
    }


def serialize_totals(value: object) -> object:
    if isinstance(value, Totals):
        return value.as_dict()
    if isinstance(value, dict):
        return {key: serialize_totals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_totals(item) for item in value]
    return value


def coefficient_score(
    positions_by_depth: dict[int, list[Position]],
    policy: dict[int, dict[str, object]],
    direction: str,
    coefficients: np.ndarray,
) -> tuple[int, list[Totals]]:
    total_time = 0
    totals = [Totals() for _ in Z_VALUES]
    for deep_depth, positions in positions_by_depth.items():
        selected = policy[deep_depth]["selected"]
        shallow_depth = int(selected["shallow_depth"])
        slack = int(selected[direction]["slack"])
        _, per_level = combined_levels(
            positions, shallow_depth, direction, coefficients, slack
        )
        for level, values in enumerate(per_level):
            totals[level] += values
            total_time += values.time_ms
    return total_time, totals


def coefficients_valid(coefficients: np.ndarray) -> bool:
    for deep in range(3, 17):
        shallow = current_shallow_depth(deep)
        n_min = 4
        n_max = 64 - deep
        values = sigma_values(
            np.arange(n_min, n_max + 1, dtype=np.float64),
            shallow, deep, coefficients,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.10) or np.any(values > 40.0):
            return False
    return True


def improve_coefficients(
    train_by_depth: dict[int, list[Position]],
    check_by_depth: dict[int, list[Position]],
    policy: dict[int, dict[str, object]],
    direction: str,
) -> dict[str, object]:
    base_train_time, base_train_levels = coefficient_score(
        train_by_depth, policy, direction, CURRENT_COEFFICIENTS
    )
    base_check_time, base_check_levels = coefficient_score(
        check_by_depth, policy, direction, CURRENT_COEFFICIENTS
    )

    def acceptable(candidate: np.ndarray) -> tuple[bool, int, list[Totals], int, list[Totals]]:
        if not coefficients_valid(candidate):
            return False, 2**62, [], 2**62, []
        train_time, train_levels = coefficient_score(
            train_by_depth, policy, direction, candidate
        )
        if not error_not_greater(train_levels, base_train_levels):
            return False, train_time, train_levels, 2**62, []
        check_time, check_levels = coefficient_score(
            check_by_depth, policy, direction, candidate
        )
        ok = error_not_greater(check_levels, base_check_levels)
        return ok, train_time, train_levels, check_time, check_levels

    best = CURRENT_COEFFICIENTS.copy()
    best_train_time = base_train_time
    best_check_time = base_check_time
    best_train_levels = base_train_levels
    best_check_levels = base_check_levels
    steps = np.array([0.08, 0.30, 0.10, 0.06, 0.35, 0.25, 0.20])
    evaluations = 0
    accepted_changes = 0
    while float(np.max(steps)) > 0.002:
        changed = False
        for index in range(len(best)):
            for sign in (-1.0, 1.0):
                candidate = best.copy()
                candidate[index] += sign * steps[index]
                evaluations += 1
                ok, train_time, train_levels, check_time, check_levels = acceptable(candidate)
                if ok and (
                    train_time < best_train_time
                    or (train_time == best_train_time and check_time < best_check_time)
                ):
                    best = candidate
                    best_train_time = train_time
                    best_check_time = check_time
                    best_train_levels = train_levels
                    best_check_levels = check_levels
                    accepted_changes += 1
                    changed = True
        if not changed:
            steps *= 0.5
    return {
        "direction": direction,
        "current_coefficients": CURRENT_COEFFICIENTS.tolist(),
        "selected_coefficients": best.tolist(),
        "evaluations": evaluations,
        "accepted_changes": accepted_changes,
        "train": {
            "current_time_ms": base_train_time,
            "selected_time_ms": best_train_time,
            "current_levels": [value.as_dict() for value in base_train_levels],
            "selected_levels": [value.as_dict() for value in best_train_levels],
        },
        "check": {
            "current_time_ms": base_check_time,
            "selected_time_ms": best_check_time,
            "current_levels": [value.as_dict() for value in base_check_levels],
            "selected_levels": [value.as_dict() for value in best_check_levels],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "mid_policy.json")
    parser.add_argument("inputs", type=Path, nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.inputs or sorted((ROOT / "benchmark").glob("mid_mpc_model_dev_d*_20260827.tsv")) + sorted(
        (ROOT / "benchmark").glob("mid_probcut_dev_d*_20260827.tsv")
    )
    positions = group_positions(read_tsv(paths))
    by_depth: dict[int, list[Position]] = {}
    for position in positions:
        by_depth.setdefault(position.deep_depth, []).append(position)

    train_by_depth: dict[int, list[Position]] = {}
    check_by_depth: dict[int, list[Position]] = {}
    depth_results: dict[int, dict[str, object]] = {}
    for deep_depth, depth_positions in sorted(by_depth.items()):
        train = [position for position in depth_positions if split_number(position.board) != 0]
        check = [position for position in depth_positions if split_number(position.board) == 0]
        if not train or not check:
            continue
        common_train = set.intersection(*(set(position.shallow) for position in train))
        common_check = set.intersection(*(set(position.shallow) for position in check))
        common = common_train & common_check
        train = [
            Position(**{**position.__dict__, "shallow": {
                depth: position.shallow[depth] for depth in common
            }}) for position in train
        ]
        check = [
            Position(**{**position.__dict__, "shallow": {
                depth: position.shallow[depth] for depth in common
            }}) for position in check
        ]
        train_by_depth[deep_depth] = train
        check_by_depth[deep_depth] = check
        depth_results[deep_depth] = choose_depth_and_gate(train, check, deep_depth)
        selected = depth_results[deep_depth]["selected"]
        print(
            f"deep={deep_depth:2d} shallow={selected['shallow_depth']:2d} "
            f"high_slack={selected['high']['slack']} low_slack={selected['low']['slack']} "
            f"train_ms={selected['train_time_ms']} check_ms={selected['check_time_ms']}",
            flush=True,
        )

    high_coefficients = improve_coefficients(
        train_by_depth, check_by_depth, depth_results, "high"
    )
    print("上側係数の計算完了", flush=True)
    low_coefficients = improve_coefficients(
        train_by_depth, check_by_depth, depth_results, "low"
    )
    print("下側係数の計算完了", flush=True)

    report = {
        "definition": {
            "train": "盤面文字列のSHA-256先頭8バイトを5で割った余りが1～4の局面。係数と浅い探索深度を決めるために使う。",
            "check": "同じ値を5で割った余りが0の局面。候補決定には使わず、誤った枝刈りが増えないことを確認する。",
            "time_ms": "各局面で記録した浅い探索時間と深い探索時間から、MPCを使ったときの合計探索時間を計算した値。",
            "wrong_2plus": "枝刈りした側に真の値がなく、探索窓の境界から2石以上離れていた回数。",
            "wrong_4plus": "枝刈りした側に真の値がなく、探索窓の境界から4石以上離れていた回数。",
        },
        "inputs": [str(path.resolve()) for path in paths],
        "positions": len(positions),
        "depths": serialize_totals(depth_results),
        "high_coefficients": high_coefficients,
        "low_coefficients": low_coefficients,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
