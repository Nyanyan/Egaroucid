"""Aggregate the deterministic Adam retraining experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ROOT = REPO_ROOT / "benchmark" / "eval_optimizer_adam_20260828"
FINAL_STAGES = [
    "deployed_reference",
    "adam_final_float",
    "rounded_nearest",
    "rounded_hillclimb_final",
]
NUMERIC_FIELDS = {
    "phase", "loop", "elapsed_ms", "n", "alpha", "validation_loss_increase",
    "mse", "mae", "bias_e_minus_z", "teacher_mean", "predicted_mean",
    "slope_e_on_z", "intercept_e_on_z", "slope_z_on_e", "intercept_z_on_e",
    "correlation", "ols_r_squared", "prediction_r_squared", "cov_e_with_z_minus_e",
}


def parse_number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def delta_row(
    phase: int,
    split: str,
    comparison: str,
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "phase": phase,
        "split": split,
        "comparison": comparison,
        "before_stage": before["stage"],
        "after_stage": after["stage"],
    }
    for field in [
        "mse", "mae", "bias_e_minus_z", "slope_e_on_z", "slope_z_on_e",
        "cov_e_with_z_minus_e", "prediction_r_squared",
    ]:
        before_value = parse_number(before[field])
        after_value = parse_number(after[field])
        result[f"before_{field}"] = before_value
        result[f"after_{field}"] = after_value
        result[f"delta_{field}"] = after_value - before_value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hash-input-data", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Older runs stored the complete short-status output.  It can contain
    # unrelated user filenames, which are neither needed for reproduction nor
    # appropriate for a focused benchmark artifact.
    if "git_status_short" in manifest:
        manifest["source_worktree_dirty_at_start"] = bool(
            str(manifest.pop("git_status_short")).strip()
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    all_rows: list[dict[str, str]] = []
    run_rows: list[dict[str, object]] = []
    output_hashes: list[dict[str, object]] = []
    fingerprint_pattern = re.compile(
        r"n_train_data (?P<train>\d+) n_val_data (?P<val>\d+) .*?"
        r"validation_mode (?P<mode>\S+) validation_seed (?P<seed>\d+) "
        r"validation_fingerprint (?P<fingerprint>0x[0-9a-fA-F]+) "
        r"validation_fingerprint_records (?P<fingerprint_records>\d+)"
    )

    for phase_info in manifest["phases"]:
        phase = int(phase_info["phase"])
        metrics_path = root / f"phase_{phase}" / "trained" / f"metrics_phase_{phase}.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        rows = read_metrics(metrics_path)
        all_rows.extend(rows)

        stderr_path = root / f"phase_{phase}" / "stderr.txt"
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        match = fingerprint_pattern.search(stderr_text)
        if not match:
            raise ValueError(f"phase {phase}: validation fingerprint not found")
        final_train = next(
            row for row in rows
            if row["stage"] == "adam_final_float" and row["split"] == "train"
        )
        validation_split = "validation_shared_train" if phase <= 11 else "validation"
        final_validation = next(
            row for row in rows
            if row["stage"] == "adam_final_float" and row["split"] == validation_split
        )
        run_rows.append({
            "phase": phase,
            "minutes": phase_info["minutes"],
            "initial_alpha": phase_info["alpha"],
            "records": phase_info["total_records"],
            "train_records": match.group("train"),
            "validation_records": match.group("val"),
            "validation_mode": match.group("mode"),
            "validation_seed": match.group("seed"),
            "validation_fingerprint": match.group("fingerprint"),
            "validation_fingerprint_records": match.group("fingerprint_records"),
            "final_loop": final_train["loop"],
            "adam_elapsed_ms": final_train["elapsed_ms"],
            "adam_budget_ms": int(phase_info["minutes"]) * 60_000,
            "final_alpha": final_train["alpha"],
            "validation_loss_increase": final_train["validation_loss_increase"],
            "stop_reason": (
                "early_stopping"
                if int(final_train["validation_loss_increase"]) > int(phase_info["patience"])
                else "time_limit"
            ),
            "final_train_mse": final_train["mse"],
            "final_validation_mse": final_validation["mse"],
        })
        for filename in [
            f"metrics_phase_{phase}.csv", f"float_{phase}.txt",
            f"{phase}.txt", f"weight_{phase}.txt",
        ]:
            path = root / f"phase_{phase}" / "trained" / filename
            output_hashes.append({
                "phase": phase,
                "file": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    metrics_fields = list(all_rows[0])
    write_rows(root / "all_metrics.csv", all_rows, metrics_fields)
    write_rows(
        root / "run_summary.csv", run_rows,
        [
            "phase", "minutes", "initial_alpha", "records", "train_records",
            "validation_records", "validation_mode", "validation_seed",
            "validation_fingerprint", "validation_fingerprint_records", "final_loop",
            "adam_elapsed_ms", "adam_budget_ms",
            "final_alpha", "validation_loss_increase", "stop_reason", "final_train_mse",
            "final_validation_mse",
        ],
    )

    final_rows = [row for row in all_rows if row["stage"] in FINAL_STAGES]
    for row in final_rows:
        slope_forward = parse_number(row["slope_e_on_z"])
        slope_reverse = parse_number(row["slope_z_on_e"])
        ols_r_squared = parse_number(row["ols_r_squared"])
        mse = parse_number(row["mse"])
        bias = parse_number(row["bias_e_minus_z"])
        prediction_r_squared = parse_number(row["prediction_r_squared"])
        row["forward_minus_ols_r_squared"] = str(slope_forward - ols_r_squared)
        row["reverse_minus_one"] = str(slope_reverse - 1.0)
        row["optimal_affine_scale"] = row["slope_z_on_e"]
        row["optimal_affine_offset"] = row["intercept_z_on_e"]
        if (
            math.isfinite(slope_forward) and math.isfinite(slope_reverse)
            and math.isfinite(prediction_r_squared) and slope_forward * slope_reverse > 0.0
            and prediction_r_squared != 1.0
        ):
            teacher_variance = mse / (1.0 - prediction_r_squared)
            forced_scale = 1.0 / slope_forward
            forced_mse = teacher_variance * (1.0 / (slope_forward * slope_reverse) - 1.0)
            row["scale_to_force_forward_slope_one"] = str(forced_scale)
            row["mse_if_force_forward_slope_one_with_intercept"] = str(forced_mse)
            row["forced_slope_one_mse_change_percent"] = str(100.0 * (forced_mse - mse) / mse)
            row["current_centered_mse"] = str(mse - bias * bias)
        else:
            row["scale_to_force_forward_slope_one"] = "nan"
            row["mse_if_force_forward_slope_one_with_intercept"] = "nan"
            row["forced_slope_one_mse_change_percent"] = "nan"
            row["current_centered_mse"] = "nan"
    final_fields = metrics_fields + [
        "forward_minus_ols_r_squared", "reverse_minus_one",
        "optimal_affine_scale", "optimal_affine_offset",
        "scale_to_force_forward_slope_one",
        "mse_if_force_forward_slope_one_with_intercept",
        "forced_slope_one_mse_change_percent", "current_centered_mse",
    ]
    write_rows(root / "stage_summary.csv", final_rows, final_fields)

    grouped: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in final_rows:
        grouped[(int(row["phase"]), row["split"], row["stage"])] = row

    delta_rows: list[dict[str, object]] = []
    phases = sorted({int(row["phase"]) for row in final_rows})
    for phase in phases:
        splits = sorted({row["split"] for row in final_rows if int(row["phase"]) == phase})
        for split in splits:
            available = {stage: grouped.get((phase, split, stage)) for stage in FINAL_STAGES}
            comparisons = [
                ("deployed_to_new_float", "deployed_reference", "adam_final_float"),
                ("float_to_nearest_integer", "adam_final_float", "rounded_nearest"),
                ("nearest_to_hillclimb", "rounded_nearest", "rounded_hillclimb_final"),
                ("float_to_final_integer", "adam_final_float", "rounded_hillclimb_final"),
            ]
            for comparison, before_stage, after_stage in comparisons:
                before = available[before_stage]
                after = available[after_stage]
                if before is not None and after is not None:
                    delta_rows.append(delta_row(phase, split, comparison, before, after))
    delta_fields = list(delta_rows[0])
    write_rows(root / "stage_deltas.csv", delta_rows, delta_fields)

    trajectory_rows: list[dict[str, object]] = []
    for phase in phases:
        splits = sorted({row["split"] for row in all_rows if int(row["phase"]) == phase})
        for split in splits:
            checkpoints = [
                row for row in all_rows
                if int(row["phase"]) == phase
                and row["split"] == split
                and row["stage"] == "adam_pre_update"
            ]
            checkpoints.sort(key=lambda row: int(row["loop"]))
            final = grouped[(phase, split, "adam_final_float")]
            previous = checkpoints[-1] if checkpoints else final
            prior = checkpoints[-2] if len(checkpoints) >= 2 else previous
            previous_mse = parse_number(previous["mse"])
            final_mse = parse_number(final["mse"])
            prior_mse = parse_number(prior["mse"])
            minimum_checkpoint_mse = min(
                (parse_number(row["mse"]) for row in checkpoints),
                default=final_mse,
            )
            trajectory_rows.append({
                "phase": phase,
                "split": split,
                "first_checkpoint_loop": checkpoints[0]["loop"] if checkpoints else "",
                "last_checkpoint_loop": previous["loop"],
                "final_loop": final["loop"],
                "prior_checkpoint_mse": prior_mse,
                "last_checkpoint_mse": previous_mse,
                "final_mse": final_mse,
                "tail_mse_improvement_percent": (
                    100.0 * (previous_mse - final_mse) / previous_mse
                    if previous_mse else math.nan
                ),
                "minimum_recorded_checkpoint_mse": minimum_checkpoint_mse,
                "final_above_minimum_checkpoint_percent": (
                    100.0 * (final_mse - minimum_checkpoint_mse) / minimum_checkpoint_mse
                    if minimum_checkpoint_mse else math.nan
                ),
                "final_bias_e_minus_z": final["bias_e_minus_z"],
                "final_slope_e_on_z": final["slope_e_on_z"],
                "final_slope_z_on_e": final["slope_z_on_e"],
                "final_reverse_minus_one": parse_number(final["slope_z_on_e"]) - 1.0,
                "final_ols_r_squared": final["ols_r_squared"],
                "final_forward_minus_ols_r_squared": (
                    parse_number(final["slope_e_on_z"]) - parse_number(final["ols_r_squared"])
                ),
                "final_cov_e_with_z_minus_e": final["cov_e_with_z_minus_e"],
                "final_alpha": final["alpha"],
                "validation_loss_increase": final["validation_loss_increase"],
            })
    write_rows(root / "trajectory_summary.csv", trajectory_rows, list(trajectory_rows[0]))
    write_rows(root / "output_hashes.csv", output_hashes, list(output_hashes[0]))
    if args.hash_input_data:
        input_hash_rows: list[dict[str, object]] = []
        for phase_info in manifest["phases"]:
            phase = int(phase_info["phase"])
            print(f"hashing phase {phase} input data", flush=True)
            for input_info in phase_info["input_files"]:
                input_path = Path(input_info["path"])
                input_hash_rows.append({
                    "phase": phase,
                    "data_id": input_info["data_id"],
                    "file": str(input_path),
                    "bytes": input_info["bytes"],
                    "records": input_info["records"],
                    "sha256": sha256_file(input_path),
                })
        write_rows(root / "input_hashes.csv", input_hash_rows, list(input_hash_rows[0]))

    environment_path = root / "environment.json"
    if environment_path.is_file():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment["experiment_started_at"] = manifest["started_at"]
        environment["experiment_finished_at"] = manifest.get("finished_at", "")
        experiment_binary = manifest["binary"]
        assert isinstance(experiment_binary, dict)
        experiment_binary_path = Path(str(experiment_binary["path"]))
        current_binary_hash = sha256_file(experiment_binary_path)
        if current_binary_hash != experiment_binary["sha256"]:
            raise ValueError(
                f"experiment binary changed: {experiment_binary_path} "
                f"manifest={experiment_binary['sha256']} current={current_binary_hash}"
            )
        environment["experiment_binary_sha256"] = experiment_binary["sha256"]
        environment["final_rebuilt_binary_sha256"] = current_binary_hash
        source_files = {
            "eval_optimizer_cuda_cu_sha256": HERE / "eval_optimizer_cuda.cu",
            "eval_optimizer_cuda_vcxproj_sha256": HERE / "eval_optimizer_cuda_12_2_0.vcxproj",
            "runner_sha256": HERE / "run_eval_optimizer_adam_experiment.py",
            "analyzer_sha256": Path(__file__).resolve(),
        }
        manifest_source_files = manifest.get("source_files", {})
        manifest_source_fields = {
            "eval_optimizer_cuda_cu_sha256": "eval_optimizer_cuda.cu",
            "eval_optimizer_cuda_vcxproj_sha256": "eval_optimizer_cuda_12_2_0.vcxproj",
            "runner_sha256": "runner",
        }
        for field, path in source_files.items():
            current_hash = sha256_file(path)
            manifest_field = manifest_source_fields.get(field)
            if manifest_field is not None:
                expected_hash = manifest_source_files.get(manifest_field)
                if current_hash != expected_hash:
                    raise ValueError(
                        f"experiment source changed: {path} "
                        f"manifest={expected_hash} current={current_hash}"
                    )
            environment[field] = current_hash
        environment_path.write_text(
            json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"wrote aggregate results under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
