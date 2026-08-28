"""Re-run selected evaluation phases with deterministic validation membership.

This script intentionally reproduces the data-ID filtering in
eval_optimizer_phase.py.  It does not read or modify any test dataset.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from data_range import board_n_moves, use_all_depth_data


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_BINARY = HERE / "eval_optimizer_cuda_12_2_0.exe"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "model" / "20260621_1_afterrand16_used_dev-eval"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "eval_optimizer_adam_20260828"
DISK_RECORD_BYTES = 136

PHASE_SETTINGS = {
    11: {"minutes": 5, "alpha": 300.0, "expected_records": 85_568_632},
    12: {"minutes": 5, "alpha": 400.0, "expected_records": 64_447_874},
    21: {"minutes": 8, "alpha": 450.0, "expected_records": 67_163_942},
    30: {"minutes": 9, "alpha": 500.0, "expected_records": 69_988_462},
    41: {"minutes": 11, "alpha": 550.0, "expected_records": 122_532_802},
}

BASE_DATA_IDS = [
    34, 35,
    37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63,
    67, 68, 69, 70, 71, 72, 73, 74,
    77, 78, 79, 80, 82, 97,
    *range(144, 157),
    *range(158, 166),
    *range(216, 221),
    *range(223, 235),
    311,
    *range(313, 325),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def selected_data_ids(phase: int) -> list[int]:
    candidates = list(BASE_DATA_IDS)
    if phase >= 12:
        candidates.extend([18, 19, 20, 21, 24, 25, 28, 29, 30, 31])
        candidates.extend([65, 66])
        candidates.append(214)
        candidates.extend(range(259, 311))

    n_after_random = 0 if phase < 14 else min(16, phase - 13)
    result = []
    for data_id in candidates:
        minimum_phase, maximum_phase = board_n_moves[str(data_id)]
        if (
            minimum_phase + n_after_random <= phase <= maximum_phase
            or data_id in use_all_depth_data
        ):
            result.append(data_id)
    return sorted(result)


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip()


def powershell_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="11,12,21,30,41")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validation-seed", type=int, default=20260828)
    parser.add_argument("--round-seed", type=int, default=20260828)
    parser.add_argument("--metrics-interval", type=int, default=25)
    parser.add_argument("--round-seconds", type=int, default=60)
    parser.add_argument("--hash-input-data", action="store_true")
    parser.add_argument(
        "--build-command", default="",
        help="exact build command used to create --binary (recorded, not executed)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    phases = [int(value) for value in args.phases.split(",") if value.strip()]
    unsupported = [phase for phase in phases if phase not in PHASE_SETTINGS]
    if unsupported:
        parser.error(f"unsupported phases: {unsupported}")

    data_environment = os.environ.get("EGAROUCID_DATA")
    if not data_environment:
        parser.error("EGAROUCID_DATA is not set")
    data_root = Path(data_environment) / "train_data" / "bin_data" / "20241125_1"
    binary = args.binary.resolve()
    reference_dir = args.reference_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")

    manifest: dict[str, object] = {
        "started_at": iso_now(),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_value("rev-parse", "HEAD"),
        # Record whether the checkout was dirty without copying unrelated file
        # names into the experiment artifact.
        "source_worktree_dirty_at_start": bool(git_value("status", "--short").strip()),
        "platform": platform.platform(),
        "python": sys.version,
        "binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "build_command": args.build_command or "not supplied",
        "source_files": {
            "eval_optimizer_cuda.cu": sha256_file(HERE / "eval_optimizer_cuda.cu"),
            "eval_optimizer_cuda_12_2_0.vcxproj": sha256_file(
                HERE / "eval_optimizer_cuda_12_2_0.vcxproj"
            ),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "validation_seed": args.validation_seed,
        "round_seed": args.round_seed,
        "metrics_interval": args.metrics_interval,
        "round_seconds": args.round_seconds,
        "data_root": str(data_root),
        "disk_record_bytes": DISK_RECORD_BYTES,
        "test_data_touched": False,
        "phases": [],
    }
    commands_path = output_root / "commands.txt"
    commands_path.write_text(
        "# Build command used before this runner\n"
        + (args.build_command or "# not supplied")
        + "\n\n",
        encoding="utf-8",
    )

    for phase in phases:
        settings = PHASE_SETTINGS[phase]
        data_ids = selected_data_ids(phase)
        data_files = [data_root / str(phase) / f"{data_id}.dat" for data_id in data_ids]
        missing = [str(path) for path in data_files if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing training inputs:\n" + "\n".join(missing))

        input_details = []
        total_records = 0
        for data_id, path in zip(data_ids, data_files):
            size = path.stat().st_size
            if size % DISK_RECORD_BYTES:
                raise ValueError(f"unexpected record alignment: {path} has {size} bytes")
            records = size // DISK_RECORD_BYTES
            total_records += records
            detail: dict[str, object] = {
                "data_id": data_id,
                "path": str(path),
                "bytes": size,
                "records": records,
                "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
            }
            if args.hash_input_data:
                detail["sha256"] = sha256_file(path)
            input_details.append(detail)

        expected_records = int(settings["expected_records"])
        if total_records != expected_records:
            raise ValueError(
                f"phase {phase}: selected files contain {total_records:,} records; "
                f"expected {expected_records:,}"
            )

        phase_dir = output_root / f"phase_{phase}"
        trained_dir = phase_dir / "trained"
        trained_dir.mkdir(parents=True, exist_ok=True)
        reference_file = reference_dir / f"{phase}.txt"
        if not reference_file.is_file():
            raise FileNotFoundError(f"missing deployed reference: {reference_file}")
        missing_initial = phase_dir / "missing_zero_initialization" / f"{phase}.txt"

        command = [
            str(binary), str(phase), "0", str(settings["minutes"]), "0",
            str(settings["alpha"]), "100", "30", "0.8", str(missing_initial),
            *map(str, data_files),
        ]
        environment_overrides = {
            "EGAROUCID_EVAL_VALIDATION_SEED": str(args.validation_seed),
            "EGAROUCID_EVAL_ROUND_SEED": str(args.round_seed),
            "EGAROUCID_EVAL_METRICS_INTERVAL": str(args.metrics_interval),
            "EGAROUCID_EVAL_ROUND_SECONDS": str(args.round_seconds),
            "EGAROUCID_EVAL_OUTPUT_DIR": str(trained_dir),
            "EGAROUCID_EVAL_REFERENCE_FILE": str(reference_file),
        }
        command_text = "& " + " ".join(powershell_quote(value) for value in command)
        with commands_path.open("a", encoding="utf-8") as commands:
            commands.write(f"# phase {phase}\n")
            for name, value in environment_overrides.items():
                commands.write(f"$env:{name}={powershell_quote(value)}\n")
            commands.write(command_text + "\n\n")

        phase_manifest: dict[str, object] = {
            "phase": phase,
            "status": "dry_run" if args.dry_run else "running",
            "started_at": iso_now(),
            "minutes": settings["minutes"],
            "alpha": settings["alpha"],
            "patience": 100,
            "reduce_lr_patience": 30,
            "reduce_lr_ratio": 0.8,
            "validation_mode": "shared_all_not_holdout" if phase <= 11 else "fixed_shuffled_suffix_5pct",
            "data_ids": data_ids,
            "total_records": total_records,
            "expected_records": expected_records,
            "input_files": input_details,
            "reference_file": {
                "path": str(reference_file),
                "sha256": sha256_file(reference_file),
            },
            "initialization": "zero (input path intentionally absent)",
            "command": command,
            "environment_overrides": environment_overrides,
        }
        cast_phases = manifest["phases"]
        assert isinstance(cast_phases, list)
        cast_phases.append(phase_manifest)
        write_json(output_root / "run_manifest.json", manifest)

        print(
            f"[{iso_now()}] phase {phase}: {total_records:,} records, "
            f"Adam {settings['minutes']} min, alpha {settings['alpha']}",
            flush=True,
        )
        if args.dry_run:
            continue

        environment = os.environ.copy()
        environment.update(environment_overrides)
        stdout_path = phase_dir / "stdout.txt"
        stderr_path = phase_dir / "stderr.txt"
        with stdout_path.open("w", encoding="utf-8", newline="") as stdout_file, \
                stderr_path.open("w", encoding="utf-8", newline="") as stderr_file:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=stderr_file,
            )
            assert process.stdout is not None
            for line in process.stdout:
                stdout_file.write(line)
                stdout_file.flush()
                print(line, end="", flush=True)
            return_code = process.wait()

        phase_manifest["finished_at"] = iso_now()
        phase_manifest["return_code"] = return_code
        phase_manifest["status"] = "completed" if return_code == 0 else "failed"
        metrics_file = trained_dir / f"metrics_phase_{phase}.csv"
        phase_manifest["outputs"] = {
            "metrics": str(metrics_file),
            "float_weights": str(trained_dir / f"float_{phase}.txt"),
            "integer_weights": str(trained_dir / f"{phase}.txt"),
            "appearance_counts": str(trained_dir / f"weight_{phase}.txt"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        write_json(output_root / "run_manifest.json", manifest)
        if return_code != 0:
            return return_code

    manifest["finished_at"] = iso_now()
    write_json(output_root / "run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
