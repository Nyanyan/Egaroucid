#!/usr/bin/env python3
"""実際の零窓探索を使って中盤MPCの条件と係数を反復測定する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIN = ROOT / "bin"
SUMMARY_PATTERN = re.compile(r"(\w+)=([^ ]+)")
CURRENT_COEFFICIENTS = [
    0.8335834703936896, -4.71778909968251, 1.1467905781538477,
    -0.5274699259330169, 6.5091001393587335, 3.9546352081550378,
    1.8719077939546169,
]
LEVELS = (0, 1, 2)


DATASETS = {
    "ggs": {
        "positions": BIN / "problem" / "midgame_ggs_holdout_20260827.txt",
        "reference": ROOT / "benchmark" / "mpc_sigma_sweep_20260829" / "ggs_ref16.tsv",
        "forced": ROOT / "benchmark" / "mpc_depth_aggressiveness_20260830" / "screen_depth_ggs100" / "cache" / "forced_ggs.tsv",
    },
    "final": {
        "positions": BIN / "problem" / "mid_tuning_final_holdout_20260827.txt",
        "reference": ROOT / "benchmark" / "mid_tuning_final_ref16_300.tsv",
        "forced": ROOT / "benchmark" / "mpc_depth_aggressiveness_20260830" / "mid_model_runtime_clean_final300" / "cache" / "forced_final.tsv",
    },
}


@dataclass
class Configuration:
    depths: dict[int, int] = field(default_factory=dict)
    high_slack: dict[int, int] = field(default_factory=dict)
    low_slack: dict[int, int] = field(default_factory=dict)
    high_coefficients: list[float] = field(default_factory=lambda: CURRENT_COEFFICIENTS.copy())
    low_coefficients: list[float] = field(default_factory=lambda: CURRENT_COEFFICIENTS.copy())

    def copy(self) -> "Configuration":
        return Configuration(
            depths=dict(self.depths),
            high_slack=dict(self.high_slack),
            low_slack=dict(self.low_slack),
            high_coefficients=list(self.high_coefficients),
            low_coefficients=list(self.low_coefficients),
        )

    def environment(self) -> dict[str, str]:
        result = {
            "EGAROUCID_MID_MPC_HIGH_COEFFICIENTS": ",".join(
                f"{value:.17g}" for value in self.high_coefficients
            ),
            "EGAROUCID_MID_MPC_LOW_COEFFICIENTS": ",".join(
                f"{value:.17g}" for value in self.low_coefficients
            ),
        }
        for name, values in (
            ("EGAROUCID_MID_MPC_SHALLOW_DEPTHS", self.depths),
            ("EGAROUCID_MID_MPC_HIGH_GATE_SLACK", self.high_slack),
            ("EGAROUCID_MID_MPC_LOW_GATE_SLACK", self.low_slack),
        ):
            if values:
                result[name] = ",".join(
                    f"{depth}:{value}" for depth, value in sorted(values.items())
                )
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "depths": self.depths,
            "high_slack": self.high_slack,
            "low_slack": self.low_slack,
            "high_coefficients": self.high_coefficients,
            "low_coefficients": self.low_coefficients,
        }


def parse_summary(stderr: str) -> dict[str, int | float]:
    line = next((line for line in stderr.splitlines() if line.startswith("SUMMARY ")), None)
    if line is None:
        raise ValueError("探索結果にSUMMARY行がない")
    fields = dict(SUMMARY_PATTERN.findall(line))
    integer_names = (
        "positions", "complete", "regret_ge_2", "regret_ge_4",
        "candidate_nodes", "candidate_time_ms",
    )
    float_names = ("agreement", "mean_regret")
    return {
        **{name: int(fields[name]) for name in integer_names},
        **{name: float(fields[name]) for name in float_names},
    }


class Runner:
    def __init__(self, executable: Path, output: Path, timeout: float) -> None:
        self.executable = executable.resolve()
        self.output = output
        self.timeout = timeout
        self.cache_directory = output / "cache"
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.memory: dict[str, dict[str, Any]] = {}

    def key(self, dataset: str, limit: int, config: Configuration) -> str:
        payload = json.dumps(
            {"dataset": dataset, "limit": limit, "config": config.as_dict()},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def run(self, dataset: str, limit: int, config: Configuration) -> dict[str, Any]:
        key = self.key(dataset, limit, config)
        if key in self.memory:
            return self.memory[key]
        cache_path = self.cache_directory / f"{key}.json"
        if cache_path.exists():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            self.memory[key] = result
            return result
        specification = DATASETS[dataset]
        environment = os.environ.copy()
        environment.update(config.environment())
        per_level = []
        started = time.perf_counter()
        for level in LEVELS:
            command = [
                str(self.executable), str(specification["positions"]), "16",
                str(level), "16", "1", "20", str(limit),
                str(specification["reference"]), str(specification["forced"]), "1",
            ]
            completed = subprocess.run(
                command,
                cwd=BIN,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"探索失敗 level={level} return={completed.returncode}\n{completed.stderr[-1000:]}"
                )
            per_level.append({"level": level, **parse_summary(completed.stderr)})
        nodes = sum(int(row["candidate_nodes"]) for row in per_level)
        elapsed = sum(int(row["candidate_time_ms"]) for row in per_level)
        result = {
            "dataset": dataset,
            "limit": limit,
            "configuration": config.as_dict(),
            "levels": per_level,
            "nodes": nodes,
            "time_ms": elapsed,
            "nps": nodes * 1000.0 / elapsed if elapsed else None,
            "wall_seconds": time.perf_counter() - started,
        }
        cache_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.memory[key] = result
        return result


def errors_not_greater(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return all(
        int(new["regret_ge_2"]) <= int(old["regret_ge_2"])
        and int(new["regret_ge_4"]) <= int(old["regret_ge_4"])
        for new, old in zip(candidate["levels"], baseline["levels"])
    )


def comparison_score(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> float:
    return 0.5 * (
        candidate["nodes"] / baseline["nodes"]
        + candidate["time_ms"] / baseline["time_ms"]
    )


def record_step(
    steps: list[dict[str, Any]],
    name: str,
    accepted: bool,
    config: Configuration,
    result: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    row = {
        "name": name,
        "accepted": accepted,
        "configuration": config.as_dict(),
        "result": result,
        "node_ratio": result["nodes"] / baseline["nodes"],
        "time_ratio": result["time_ms"] / baseline["time_ms"],
        "score": comparison_score(result, baseline),
    }
    steps.append(row)
    print(
        f"{name} accepted={int(accepted)} "
        f"nodes={row['node_ratio']:.5f} time={row['time_ratio']:.5f} "
        f"errors={[ (v['regret_ge_2'], v['regret_ge_4']) for v in result['levels'] ]}",
        flush=True,
    )


def shallow_candidates(deep: int) -> list[int]:
    current = ((deep * 2 // 5) & ~1) + (deep & 1)
    values = {
        max(deep & 1, min(deep - 2, current + offset))
        for offset in (-2, 0, 2)
    }
    return sorted(values)


def tune(
    runner: Runner,
    selection_limit: int,
    validation_limit: int,
) -> dict[str, Any]:
    original = Configuration()
    selection_baseline = runner.run("ggs", selection_limit, original)
    current = original.copy()
    current_result = selection_baseline
    steps: list[dict[str, Any]] = []

    # 深い探索深度ごとの浅い探索深度。
    for deep in range(16, 2, -1):
        best_config = current
        best_result = current_result
        best_score = comparison_score(current_result, selection_baseline)
        for shallow in shallow_candidates(deep):
            candidate = current.copy()
            candidate.depths[deep] = shallow
            result = runner.run("ggs", selection_limit, candidate)
            score = comparison_score(result, selection_baseline)
            safe = errors_not_greater(result, selection_baseline)
            accepted = safe and score < best_score - 0.001
            record_step(
                steps, f"浅い探索深度 深い探索{deep}手 浅い探索{shallow}手",
                accepted, candidate, result, selection_baseline,
            )
            if accepted:
                best_config = candidate
                best_result = result
                best_score = score
        current = best_config
        current_result = best_result

    # 浅い探索を実行する条件。実戦で費用が大きい深度から調べる。
    for deep in range(16, 9, -1):
        for direction in ("high", "low"):
            best_config = current
            best_result = current_result
            best_score = comparison_score(current_result, selection_baseline)
            for slack in (0, 1, 2, 3, 4, 5, 6, 8):
                candidate = current.copy()
                selected = candidate.high_slack if direction == "high" else candidate.low_slack
                selected[deep] = slack
                result = runner.run("ggs", selection_limit, candidate)
                score = comparison_score(result, selection_baseline)
                safe = errors_not_greater(result, selection_baseline)
                accepted = safe and score < best_score - 0.001
                record_step(
                    steps,
                    f"{'上側' if direction == 'high' else '下側'}実行条件 深い探索{deep}手 差{slack}石",
                    accepted, candidate, result, selection_baseline,
                )
                if accepted:
                    best_config = candidate
                    best_result = result
                    best_score = score
            current = best_config
            current_result = best_result

    # 上側と下側の係数を別々に変更する。各係数は実際の探索結果で判定する。
    coefficient_steps = [0.04, 0.15, 0.05, 0.03, 0.15, 0.10, 0.10]
    for direction in ("high", "low"):
        for index, amount in enumerate(coefficient_steps):
            best_config = current
            best_result = current_result
            best_score = comparison_score(current_result, selection_baseline)
            for sign in (-1.0, 1.0):
                candidate = current.copy()
                coefficients = (
                    candidate.high_coefficients
                    if direction == "high" else candidate.low_coefficients
                )
                coefficients[index] += sign * amount
                result = runner.run("ggs", selection_limit, candidate)
                score = comparison_score(result, selection_baseline)
                safe = errors_not_greater(result, selection_baseline)
                accepted = safe and score < best_score - 0.001
                record_step(
                    steps,
                    f"{'上側' if direction == 'high' else '下側'}係数{index + 1} "
                    f"{'加算' if sign > 0 else '減算'}{amount}",
                    accepted, candidate, result, selection_baseline,
                )
                if accepted:
                    best_config = candidate
                    best_result = result
                    best_score = score
            current = best_config
            current_result = best_result

    validation_baseline = runner.run("final", validation_limit, original)
    validation_candidate = runner.run("final", validation_limit, current)
    return {
        "definitions": {
            "selection": "GGS実戦由来の局面。各候補を選ぶために使う。各局面の探索前に置換表を空にする。",
            "validation": "候補選択に使わない別の局面。最後に候補全体を確認する。各局面の探索前に置換表を空にする。",
            "score": "候補の訪問局面数比と探索時間比の平均。2石以上または4石以上の選択誤差を増やす候補は選ばない。",
        },
        "selection_limit": selection_limit,
        "validation_limit": validation_limit,
        "selection_baseline": selection_baseline,
        "selection_final": current_result,
        "selected_configuration": current.as_dict(),
        "steps": steps,
        "validation_baseline": validation_baseline,
        "validation_candidate": validation_candidate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exe", type=Path, default=BIN / "mid_mpc_runtime_tuning.exe"
    )
    parser.add_argument("--output", type=Path, default=HERE / "mid_runtime_tuning")
    parser.add_argument("--selection-limit", type=int, default=40)
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.exe, args.output, args.timeout)
    report = tune(runner, args.selection_limit, args.validation_limit)
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected_configuration": report["selected_configuration"],
        "selection_node_ratio": report["selection_final"]["nodes"] / report["selection_baseline"]["nodes"],
        "selection_time_ratio": report["selection_final"]["time_ms"] / report["selection_baseline"]["time_ms"],
        "validation_node_ratio": report["validation_candidate"]["nodes"] / report["validation_baseline"]["nodes"],
        "validation_time_ratio": report["validation_candidate"]["time_ms"] / report["validation_baseline"]["time_ms"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
