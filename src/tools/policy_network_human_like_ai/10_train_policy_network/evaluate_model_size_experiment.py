#!/usr/bin/env python3
"""Re-evaluate saved model-size experiments with symmetry-aware agreement."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
HUMAN_LIKE_AI_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[4]
WTHOR_EVALUATOR = (
    HUMAN_LIKE_AI_DIR
    / "20_test_with_wthor"
    / "evaluate_wthor_human_match.py"
)
POLICY_ACCURACY = HUMAN_LIKE_AI_DIR / "policy_accuracy.py"
DEFAULT_CONFIGS = (
    "256x4,384x4,512x4,768x4,1024x4,"
    "512x6,768x6,1024x6"
)
TOP_N = (1, 2, 3, 5, 10)
CONFIG_PATTERN = re.compile(r"^(\d+)x(\d+)$")
SPLIT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}


def default_board_data_dir() -> Path:
    data_root = os.environ.get("EGAROUCID_DATA")
    if not data_root:
        raise ValueError(
            "--board-data-dir is required when EGAROUCID_DATA is not set"
        )
    return Path(data_root) / "train_data" / "board_data" / "records1"


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dat_file_sort_key(path: Path) -> tuple[int, str]:
    if path.stem.isdigit():
        return int(path.stem), path.name
    return 10**9, path.name


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def parse_configs(text: str) -> List[tuple[int, int]]:
    configs: List[tuple[int, int]] = []
    seen = set()
    for raw in text.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        match = CONFIG_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(
                f"invalid config {raw!r}; expected WIDTHxDEPTH"
            )
        config = (int(match.group(1)), int(match.group(2)))
        if config[0] <= 0 or config[1] <= 0:
            raise ValueError(
                f"invalid config {raw!r}; width and depth must be positive"
            )
        if config in seen:
            raise ValueError(f"duplicate config {token}")
        seen.add(config)
        configs.append(config)
    if not configs:
        raise ValueError("no model configurations were provided")
    return configs


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def topn_map(result: dict) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    expected_positions = int(result["positions"])
    for source_row in result["topn"]:
        top_n = int(source_row["top_n"])
        if top_n in rows:
            raise ValueError(f"duplicate top-N result: {top_n}")
        row = {
            "hits": int(source_row["hits"]),
            "positions": int(source_row["positions"]),
            "accuracy": float(source_row["accuracy"]),
        }
        if row["positions"] != expected_positions:
            raise ValueError(
                f"top-{top_n} has {row['positions']} positions; "
                f"expected {expected_positions}"
            )
        if not 0 <= row["hits"] <= row["positions"]:
            raise ValueError(f"invalid top-{top_n} hit count")
        expected_accuracy = (
            row["hits"] / row["positions"]
            if row["positions"]
            else 0.0
        )
        if not math.isclose(
            row["accuracy"],
            expected_accuracy,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"invalid top-{top_n} accuracy")
        rows[top_n] = row

    missing = set(TOP_N) - set(rows)
    if missing:
        raise ValueError(
            f"evaluation is missing top-N values: {sorted(missing)}"
        )
    unexpected = set(rows) - set(TOP_N)
    if unexpected:
        raise ValueError(
            f"evaluation has unexpected top-N values: {sorted(unexpected)}"
        )
    return rows


def assert_single_agreement_schema(result: dict) -> None:
    forbidden = {
        "exact_hits",
        "exact_accuracy",
        "symmetric_hits",
        "symmetric_accuracy",
        "exact_metric_role",
    }

    def visit(value) -> Iterable[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from visit(child)
        elif isinstance(value, list):
            for child in value:
                yield from visit(child)

    found = forbidden.intersection(visit(result))
    if found:
        raise ValueError(
            "evaluation contains removed agreement fields: "
            f"{sorted(found)}"
        )
    if result.get("agreement_definition", {}).get("metric") != (
        "board_symmetry_aware"
    ):
        raise ValueError("evaluation did not use board-symmetry-aware agreement")


def validate_evaluation_result(
    result: dict,
    board_data_dir: Path,
    model_path: Path,
    data_split: str,
    split_seed: int,
) -> None:
    assert_single_agreement_schema(result)
    if result.get("data_split") != data_split:
        raise ValueError(
            f"expected {data_split} result, got {result.get('data_split')}"
        )
    if int(result.get("split_seed", -1)) != split_seed:
        raise ValueError("evaluation used the wrong split seed")
    if result.get("split_ratios") != SPLIT_RATIOS:
        raise ValueError("evaluation used the wrong split ratios")
    if Path(result["board_data_dir"]).resolve() != board_data_dir.resolve():
        raise ValueError("evaluation used the wrong board-data directory")
    if Path(result["model_source"]).resolve() != model_path.resolve():
        raise ValueError("evaluation used the wrong model")

    invalid_policy = int(result.get("invalid_policy_samples", -1))
    illegal_label = int(result.get("illegal_label_samples", -1))
    if invalid_policy or illegal_label:
        raise ValueError(
            "evaluation excluded invalid samples: "
            f"invalid_policy={invalid_policy}, illegal_label={illegal_label}"
        )
    positions = int(result["positions"])
    split_positions = int(result["split_positions"])
    if positions <= 0 or positions != split_positions:
        raise ValueError(
            f"evaluated {positions} positions from a {split_positions}-position "
            "split"
        )
    if int(result["available_positions"]) < split_positions:
        raise ValueError("split is larger than the available dataset")
    split_hash = result.get("selected_position_set_sha256")
    if not isinstance(split_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        split_hash,
    ):
        raise ValueError("evaluation has no valid split SHA-256")
    topn_map(result)


def run_evaluation(
    board_data_dir: Path,
    model_path: Path,
    weights_path: Path,
    data_split: str,
    output_dir: Path,
    split_seed: int,
    batch_size: int,
    predict_batch_size: int,
) -> dict:
    command = [
        sys.executable,
        str(WTHOR_EVALUATOR),
        "--board-data-dir",
        str(board_data_dir),
        "--model",
        str(model_path),
        "--weights",
        str(weights_path),
        "--data-split",
        data_split,
        "--split-seed",
        str(split_seed),
        "--train-ratio",
        "0.8",
        "--val-ratio",
        "0.1",
        "--test-ratio",
        "0.1",
        "--top-n",
        ",".join(str(n) for n in TOP_N),
        "--batch-size",
        str(batch_size),
        "--predict-batch-size",
        str(predict_batch_size),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    result = load_json(output_dir / "wthor_human_match.json")
    validate_evaluation_result(
        result,
        board_data_dir,
        model_path,
        data_split,
        split_seed,
    )
    return result


def resource_summary(
    train_log_dir: Path,
    width: int,
    depth: int,
    epochs: int,
) -> dict:
    path = (
        train_log_dir
        / f"wthor_final_arch_{width}x{depth}_e{epochs}_resource.json"
    )
    resource = load_json(path)
    if int(resource.get("returncode", -1)) != 0:
        raise ValueError(f"training did not complete successfully: {path}")
    return {
        "path": relative_path(path),
        "elapsed_sec": float(resource["elapsed_sec"]),
        "peak_rss_mib": float(resource["peak_rss_mib"]),
        "command": str(resource["command"]),
    }


def model_paths(
    experiment_root: Path,
    width: int,
    depth: int,
    epochs: int,
) -> tuple[Path, Path, Path]:
    run_dir = (
        experiment_root
        / f"wthor_final_arch_{width}x{depth}_e{epochs}"
    )
    model_path = run_dir / "selected_model.h5"
    weights_path = run_dir / "selected_policy_network_weights.bin"
    summary_path = run_dir / "selected_summary.json"
    for path in (model_path, weights_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return model_path, weights_path, summary_path


def validate_training_summary(
    summary: dict,
    summary_path: Path,
    width: int,
    depth: int,
    epochs: int,
) -> None:
    spec = summary.get("spec", {})
    if int(spec.get("width", -1)) != width:
        raise ValueError(f"wrong model width in {summary_path}")
    if int(spec.get("depth", -1)) != depth:
        raise ValueError(f"wrong model depth in {summary_path}")
    if int(summary.get("epochs_ran", -1)) != epochs:
        raise ValueError(f"wrong completed epoch count in {summary_path}")
    if int(summary.get("selected_epoch", -1)) != epochs:
        raise ValueError(f"model is not from the final epoch: {summary_path}")
    if summary.get("selection_rule") != "final_epoch":
        raise ValueError(f"model did not use final-epoch selection: {summary_path}")
    if int(summary.get("params", 0)) <= 0:
        raise ValueError(f"invalid parameter count in {summary_path}")


def selection_key(row: dict) -> tuple[float, float, int]:
    validation = row["validation"]["topn"]
    return (
        float(validation["1"]["accuracy"]),
        float(validation["3"]["accuracy"]),
        -int(row["params"]),
    )


def write_csv_summary(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "model",
        "width",
        "depth",
        "params",
        "validation_top1",
        "validation_top3",
        "test_top1",
        "test_top2",
        "test_top3",
        "test_top5",
        "test_top10",
        "training_elapsed_sec",
        "peak_rss_mib",
        "model_sha256",
        "weights_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            validation = row["validation"]["topn"]
            test = row["test"]["topn"]
            writer.writerow(
                {
                    "model": row["model"],
                    "width": row["width"],
                    "depth": row["depth"],
                    "params": row["params"],
                    "validation_top1": validation["1"]["accuracy"],
                    "validation_top3": validation["3"]["accuracy"],
                    "test_top1": test["1"]["accuracy"],
                    "test_top2": test["2"]["accuracy"],
                    "test_top3": test["3"]["accuracy"],
                    "test_top5": test["5"]["accuracy"],
                    "test_top10": test["10"]["accuracy"],
                    "training_elapsed_sec": row["training"]["elapsed_sec"],
                    "peak_rss_mib": row["training"]["peak_rss_mib"],
                    "model_sha256": row["model_sha256"],
                    "weights_sha256": row["weights_sha256"],
                }
            )


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate saved WTHOR model-size experiments with the sole "
            "board-symmetry-aware agreement metric."
        )
    )
    parser.add_argument("--board-data-dir", type=Path, default=None)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=SCRIPT_DIR / "trained",
    )
    parser.add_argument(
        "--train-log-dir",
        type=Path,
        default=SCRIPT_DIR / "train_log",
    )
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=613)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument(
        "--detail-output-dir",
        type=Path,
        default=(
            HUMAN_LIKE_AI_DIR
            / "20_test_with_wthor"
            / "output"
            / "model_size_symmetry_aware"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "model_size_results.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "model_size_results.csv",
    )
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    board_data_dir = (
        args.board_data_dir
        if args.board_data_dir is not None
        else default_board_data_dir()
    )
    dat_files = sorted(
        board_data_dir.glob("*.dat"),
        key=dat_file_sort_key,
    )
    if not dat_files:
        raise FileNotFoundError(
            f"no board-data files found in {board_data_dir}"
        )

    dataset_files = [
        {
            "path": relative_path(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in dat_files
    ]
    rows = []
    validation_hash = None
    test_hash = None
    available_positions = None
    validation_positions = None
    test_positions = None

    for width, depth in parse_configs(args.configs):
        model_id = f"w{width}_d{depth}"
        model_path, weights_path, summary_path = model_paths(
            args.experiment_root,
            width,
            depth,
            args.epochs,
        )
        training_summary = load_json(summary_path)
        validate_training_summary(
            training_summary,
            summary_path,
            width,
            depth,
            args.epochs,
        )
        model_output_dir = args.detail_output_dir / model_id
        validation_result = run_evaluation(
            board_data_dir,
            model_path,
            weights_path,
            "val",
            model_output_dir / "validation",
            args.split_seed,
            args.batch_size,
            args.predict_batch_size,
        )
        test_result = run_evaluation(
            board_data_dir,
            model_path,
            weights_path,
            "test",
            model_output_dir / "test",
            args.split_seed,
            args.batch_size,
            args.predict_batch_size,
        )

        current_validation_hash = validation_result[
            "selected_position_set_sha256"
        ]
        current_test_hash = test_result["selected_position_set_sha256"]
        validation_hash = validation_hash or current_validation_hash
        test_hash = test_hash or current_test_hash
        if current_validation_hash != validation_hash:
            raise ValueError("validation split differs between models")
        if current_test_hash != test_hash:
            raise ValueError("test split differs between models")

        current_available_positions = int(
            validation_result["available_positions"]
        )
        if (
            int(test_result["available_positions"])
            != current_available_positions
        ):
            raise ValueError("validation and test used different datasets")
        available_positions = (
            available_positions or current_available_positions
        )
        if current_available_positions != available_positions:
            raise ValueError("available position count differs between models")

        current_validation_positions = int(
            validation_result["split_positions"]
        )
        current_test_positions = int(test_result["split_positions"])
        validation_positions = (
            validation_positions or current_validation_positions
        )
        test_positions = test_positions or current_test_positions
        if current_validation_positions != validation_positions:
            raise ValueError("validation size differs between models")
        if current_test_positions != test_positions:
            raise ValueError("test size differs between models")

        rows.append(
            {
                "model": model_id,
                "width": width,
                "depth": depth,
                "params": int(training_summary["params"]),
                "epochs": int(training_summary["epochs_ran"]),
                "model_path": relative_path(model_path),
                "model_sha256": sha256_file(model_path),
                "weights_path": relative_path(weights_path),
                "weights_sha256": sha256_file(weights_path),
                "training_summary_path": relative_path(summary_path),
                "training": resource_summary(
                    args.train_log_dir,
                    width,
                    depth,
                    args.epochs,
                ),
                "validation": {
                    "positions": int(validation_result["positions"]),
                    "invalid_policy_samples": int(
                        validation_result["invalid_policy_samples"]
                    ),
                    "illegal_label_samples": int(
                        validation_result["illegal_label_samples"]
                    ),
                    "split_sha256": current_validation_hash,
                    "topn": {
                        str(n): row
                        for n, row in topn_map(validation_result).items()
                    },
                },
                "test": {
                    "positions": int(test_result["positions"]),
                    "invalid_policy_samples": int(
                        test_result["invalid_policy_samples"]
                    ),
                    "illegal_label_samples": int(
                        test_result["illegal_label_samples"]
                    ),
                    "split_sha256": current_test_hash,
                    "topn": {
                        str(n): row
                        for n, row in topn_map(test_result).items()
                    },
                },
            }
        )

    selected = max(rows, key=selection_key)
    output = {
        "schema_version": 1,
        "issue": "#613",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": "policy_network_model_size_selection",
        "agreement_definition": {
            "metric": "board_symmetry_aware",
            "description": (
                "A prediction matches when any legal move equivalent to the "
                "recorded human move under a board-invariant D4 symmetry is "
                "within the top N."
            ),
            "tie_break": "ascending_policy_index",
        },
        "dataset": {
            "board_data_dir": relative_path(board_data_dir),
            "positions": available_positions,
            "files": dataset_files,
        },
        "split": {
            "seed": args.split_seed,
            "ratios": SPLIT_RATIOS,
            "validation_positions": validation_positions,
            "test_positions": test_positions,
            "validation_sha256": validation_hash,
            "test_sha256": test_hash,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "tensorflow": (
                package_version("tensorflow")
                or package_version("tensorflow-gpu")
                or package_version("tensorflow-cpu")
            ),
        },
        "runner": {
            "path": relative_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "evaluator": {
            "path": relative_path(WTHOR_EVALUATOR),
            "sha256": sha256_file(WTHOR_EVALUATOR),
            "policy_accuracy_path": relative_path(POLICY_ACCURACY),
            "policy_accuracy_sha256": sha256_file(POLICY_ACCURACY),
        },
        "top_n": list(TOP_N),
        "evaluation_parameters": {
            "configs": args.configs,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "predict_batch_size": args.predict_batch_size,
        },
        "models": rows,
        "selection_rule": (
            "highest validation top-1, then validation top-3, then fewer "
            "parameters"
        ),
        "selected_model": selected["model"],
        "reproduction_command": (
            "python "
            "src/tools/policy_network_human_like_ai/"
            "10_train_policy_network/evaluate_model_size_experiment.py"
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
    write_csv_summary(args.output_csv, rows)
    print("selected_model", selected["model"])
    print("output_json", args.output_json)
    print("output_csv", args.output_csv)


if __name__ == "__main__":
    main()
