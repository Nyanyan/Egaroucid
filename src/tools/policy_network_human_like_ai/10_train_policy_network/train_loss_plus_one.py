#!/usr/bin/env python3
"""Prepare disc_loss + 1 sample weights and train the previous Conv8+BN setup."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_data_processing"))
from pipeline_common import (BOARD_DTYPE, SPLIT_NAMES, WEIGHT_DTYPE, NONNEGATIVE_WEIGHT_FORMAT,
                             has_sample_weights, load_manifest, output_directory,
                             source_info, write_manifest)
from resource_monitor import run_monitored_command

SOURCE_FORMULA = "2 ** (disc_loss / 2)"
TARGET_FORMULA = "disc_loss + 1"


@dataclass(frozen=True)
class LossWeighting:
    multiplier: int = 1
    offset: int = 1
    formula: str = TARGET_FORMULA
    label: str = "loss_plus_one"
    power: int = 1


# Retain imports used by the existing linear experiment launchers.
LinearWeighting = LossWeighting
DEFAULT_WEIGHTING = LossWeighting()


def loss_weights(exponential_weights, weighting=DEFAULT_WEIGHTING):
    """Invert the saved, unnormalized exponential weights without losing odd losses."""
    values = np.asarray(exponential_weights, dtype=WEIGHT_DTYPE)
    if np.any(~np.isfinite(values)) or np.any(values < 1) or np.any(values > 2.0 ** 64):
        raise ValueError("expected finite exponential weights for disc losses in [0, 128]")
    losses = 2 * np.log2(values)
    integer_losses = np.rint(losses)
    if not np.allclose(losses, integer_losses, rtol=0, atol=1e-10):
        raise ValueError("source weights do not encode integer disc losses")
    if weighting.multiplier <= 0 or weighting.offset < 0:
        raise ValueError("linear weighting needs a positive multiplier and nonnegative offset")
    if not isinstance(weighting.power, int) or weighting.power < 1:
        raise ValueError("weighting power must be a positive integer")
    weights = np.power(integer_losses * weighting.multiplier + weighting.offset, weighting.power)
    if np.any(~np.isfinite(weights)):
        raise ValueError("sample weight overflow")
    return weights.astype(WEIGHT_DTYPE)


linear_weights = loss_weights


def prepare_dataset(source: Path, destination: Path, weighting=DEFAULT_WEIGHTING) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("source and destination must be separate dataset directories")
    header = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    stage = header.get("stage")
    if stage not in ("board_split", "board_split_increased"):
        raise ValueError("source must be a prepared stage 03/04 dataset")
    print("Checking source dataset checksums...", flush=True)
    original = load_manifest(source, stage)
    if (not has_sample_weights(original)
            or original["sample_weights"].get("formula") != SOURCE_FORMULA
            or original["sample_weights"].get("normalization") != "none in files"
            or original["sample_weights"].get("clipping") is not None):
        raise ValueError("source must contain unclipped, unnormalized 2 ** (disc_loss / 2) weights")
    conversion = ("recover integer disc_loss = round(2 * log2(weight)), then add 1"
                  if weighting == DEFAULT_WEIGHTING else
                  f"recover integer disc_loss = round(2 * log2(weight)), then {weighting.formula}")
    provenance = {**source_info(source), "dataset": original["dataset"],
                  "formula": SOURCE_FORMULA,
                  "conversion": conversion}
    dataset = original["dataset"].removesuffix("_LOSS_WEIGHTED") + "_" + weighting.label.upper()
    if (destination / "manifest.json").exists():
        existing = load_manifest(destination, stage)
        if (existing.get("reweight_source") != provenance or existing.get("dataset") != dataset
                or existing.get("sample_weights", {}).get("formula") != weighting.formula):
            raise ValueError("prepared output belongs to different source data or weighting")
        print("Reusing verified weighted dataset:", destination, flush=True)
        return existing

    metadata = {key: value for key, value in original.items()
                if key not in {"version", "stage", "dataset", "files"}}
    metadata["sample_weights"] = {**original["sample_weights"], "formula": weighting.formula}
    if weighting.offset == 0:
        metadata["sample_weights"]["format"] = NONNEGATIVE_WEIGHT_FORMAT
    metadata["reweight_source"] = provenance
    with output_directory(destination) as output:
        listed = {entry["path"] for entry in original["files"]}
        for name in SPLIT_NAMES:
            paths = sorted((source / name).glob("*.dat"))
            actual = {p.relative_to(source).as_posix() for p in (source / name).iterdir() if p.is_file()}
            expected = {p for p in listed if p.startswith(name + "/")}
            if not paths or actual != expected:
                raise ValueError(f"source files do not match manifest for {name}")
            positions, weight_sum, minimum, maximum = 0, 0.0, None, None
            (output / name).mkdir()
            for path in paths:
                size = path.stat().st_size
                count = size // BOARD_DTYPE.itemsize
                if (not count or size % BOARD_DTYPE.itemsize
                        or path.with_suffix(".game_ids").stat().st_size != count * 8
                        or path.with_suffix(".weights").stat().st_size != count * WEIGHT_DTYPE.itemsize):
                    raise ValueError(f"incomplete boards, game IDs or weights: {path}")
                for suffix in (".dat", ".game_ids"):
                    shutil.copyfile(path.with_suffix(suffix), output / name / path.with_suffix(suffix).name)
                with (path.with_suffix(".weights").open("rb") as src,
                      (output / name / path.with_suffix(".weights").name).open("wb") as dst):
                    while True:
                        raw = np.fromfile(src, dtype=WEIGHT_DTYPE, count=65536)
                        if not len(raw):
                            break
                        weights = loss_weights(raw, weighting)
                        weights.tofile(dst)
                        weight_sum += float(weights.sum())
                        minimum = float(weights.min()) if minimum is None else min(minimum, float(weights.min()))
                        maximum = float(weights.max()) if maximum is None else max(maximum, float(weights.max()))
                positions += count
                print(f"[prepare] {name}: {positions:,}/{original['splits'][name]['positions']:,} positions", flush=True)
            if positions != original["splits"][name]["positions"]:
                raise ValueError(f"source position count mismatch for {name}")
            metadata["splits"][name].update(weight_sum=weight_sum, minimum_weight=minimum,
                                           maximum_weight=maximum)
        # Preserve ancillary records, such as rejected-game reports.
        for relative in listed:
            if relative.split("/")[0] not in SPLIT_NAMES:
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, target)
        result = write_manifest(output, stage, dataset, **metadata)
        old_files = {entry["path"]: entry for entry in original["files"]}
        for entry in result["files"]:
            if not entry["path"].endswith(".weights") and entry != old_files[entry["path"]]:
                raise ValueError(f"copy verification failed: {entry['path']}")
    examples = loss_weights(np.exp2(np.array([0, 1, 2, 4, 6]) / 2), weighting).astype(int)
    print("Prepared weights: loss 0, 1, 2, 4, 6 -> " + ", ".join(map(str, examples)), flush=True)
    return result


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    root = os.environ.get("POLICY_DATA_ROOT")
    if not root and os.environ.get("EGAROUCID_DATA"):
        root = str(Path(os.environ["EGAROUCID_DATA"]) / "20261002_policy_network_human_like_data")
    parser.add_argument("--source-board-split-dir", type=Path,
                        default=Path(root) / "board_split_increased/WTHOR_1977_2024_LOSS_WEIGHTED" if root else None)
    parser.add_argument("--run-dir", type=Path,
                        help="New run directory; contains prepared_data, train and logs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-training", action="store_true")
    for name in SPLIT_NAMES:
        parser.add_argument(f"--max-{name}-samples", type=int, help="Smoke testing only; default: all positions.")
    return parser


def main(argv=None, *, weighting=DEFAULT_WEIGHTING):
    parser = make_argparser()
    args = parser.parse_args(argv)
    if args.source_board_split_dir is None:
        parser.error("set POLICY_DATA_ROOT / EGAROUCID_DATA or pass --source-board-split-dir")
    if args.resume_training and args.run_dir is None:
        parser.error("--resume-training requires --run-dir")
    if min(args.epochs, args.batch_size) <= 0 or any(
            getattr(args, f"max_{name}_samples") is not None and getattr(args, f"max_{name}_samples") <= 0
            for name in SPLIT_NAMES):
        parser.error("epochs, batch size and sample caps must be positive")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (args.run_dir or HERE / "trained" / f"conv8_bn_{weighting.label}_e{args.epochs}_{stamp}").resolve()
    training = run_dir / "train"
    if training.exists() and any(training.iterdir()) and not args.resume_training:
        parser.error("training output exists; choose a new --run-dir or use --resume-training")
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Run directory:", run_dir, flush=True)
    print(f"Sample weighting: {weighting.formula}; training divides by the full train mean.", flush=True)
    print(f"Conv8+BN, epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}, float32", flush=True)
    manifest = prepare_dataset(args.source_board_split_dir, run_dir / "prepared_data", weighting)
    if args.prepare_only:
        return
    command = [sys.executable, "-u", str(HERE / "train_policy_network.py"),
               "--board-split-dir", str(run_dir / "prepared_data"), "--dataset", manifest["dataset"],
               "--configs", "conv8_bn", "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--seed", str(args.seed), "--learning-rate", "0.001", "--l2", "0.0",
               "--predict-batch-size", str(args.batch_size), "--precision", "float32",
               "--steps-per-execution", "1", "--checkpoint", "--progress-interval-sec", "15",
               "--output-dir", str(training)]
    if args.resume_training:
        command.append("--resume-training")
    for name in SPLIT_NAMES:
        cap = getattr(args, f"max_{name}_samples")
        if cap is not None:
            command.extend([f"--max-{name}-samples", str(cap)])
    log_name = f"train_resume_{stamp}" if args.resume_training else "train"
    log = run_dir / "logs" / f"{log_name}.log"
    print("Training log:", log, flush=True)
    record = {"status": "training", "sample_weight_formula": weighting.formula, "command": command,
              "pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
              "log": str(log), "model": str(training / "selected_model.h5")}
    record_path = run_dir / "run.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    summary = run_monitored_command(command, log, cwd=HERE, tee=True)
    (run_dir / "logs" / f"{log_name}_resource.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    record.update(status="completed" if summary["returncode"] == 0 else "failed", returncode=summary["returncode"],
                  ended_at=datetime.now().astimezone().isoformat())
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if summary["returncode"]:
        raise SystemExit(summary["returncode"])
    print("Completed. Model:", training / "selected_model.h5", flush=True)


if __name__ == "__main__":
    main()
