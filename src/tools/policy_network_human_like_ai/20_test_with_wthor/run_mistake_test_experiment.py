#!/usr/bin/env python3
"""Run mistake-test preparation, all 27 evaluations, and independent verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent


def run(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "run_status.json"
    started = time.time()

    def status(state, stage, **extra):
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(dict(state=state, stage=stage, started_at_unix=started,
                                            updated_at_unix=time.time(), **extra), indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def execute(stage, command):
        status("running", stage, command=command)
        print(f"Starting {stage}", flush=True)
        subprocess.run(command, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    try:
        if args.wait_for_pid is not None:
            import psutil
            try:
                process = psutil.Process(args.wait_for_pid)
                created = process.create_time()
                status("running", "waiting_for_existing_evaluation", pid=args.wait_for_pid,
                       process_created_at=created)
                while process.is_running() and process.create_time() == created:
                    try:
                        process.wait(timeout=30)
                    except psutil.TimeoutExpired:
                        status("running", "waiting_for_existing_evaluation", pid=args.wait_for_pid,
                               process_created_at=created)
            except psutil.NoSuchProcess:
                pass
        datasets = root / "datasets"
        if not (datasets / "dataset_counts.json").exists():
            prepare = [sys.executable, "-u", str(SCRIPT_DIR.parent / "00_data_processing" / "prepare_mistake_test_data.py"),
                       "--dataset", args.dataset, "--output-dir", str(datasets)]
            if args.data_root is not None:
                prepare.extend(["--data-root", str(args.data_root.resolve())])
            execute("prepare", prepare)
        else:
            counts = json.loads((datasets / "dataset_counts.json").read_text(encoding="utf-8"))
            if counts["source_dataset"] != args.dataset:
                raise ValueError("existing output has a different dataset")
        evaluate = [sys.executable, "-u", str(SCRIPT_DIR / "evaluate_mistake_test_models.py"),
                    "--datasets-dir", str(datasets), "--output-dir", str(root / "evaluation"),
                    "--reference-dir", str(args.reference_dir.resolve()),
                    "--normal-model", str(args.normal_model.resolve()),
                    "--weighted-model", str(args.weighted_model.resolve()),
                    "--workers", str(args.workers), "--policy-batch-size", str(args.policy_batch_size),
                    "--random-seed", str(args.random_seed)]
        # Re-evaluate even after a completed run: all hints are reused and the
        # current aggregation code and saved model hashes are checked again.
        execute("evaluate", evaluate)
        execute("verify", [sys.executable, "-u", str(SCRIPT_DIR / "verify_mistake_test_results.py"),
                           "--datasets-dir", str(datasets), "--output-dir", str(root / "evaluation")])
        status("complete", "verified", results=str(root / "evaluation" / "RESULTS.md"),
               elapsed_seconds=time.time() - started)
        print(f"Completed and verified: {root / 'evaluation' / 'RESULTS.md'}", flush=True)
    except BaseException as error:
        status("failed", "error", error=repr(error), elapsed_seconds=time.time() - started)
        raise


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Run root containing datasets/ and evaluation/")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--normal-model", type=Path, required=True)
    parser.add_argument("--weighted-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", default="WTHOR_1977_2024")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--policy-batch-size", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=57)
    parser.add_argument("--wait-for-pid", type=int, help="Wait for an already running evaluator to exit before reusing its caches")
    return parser


if __name__ == "__main__":
    run(make_argparser().parse_args())
