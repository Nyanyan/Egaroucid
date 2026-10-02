#!/usr/bin/env python3
"""Run the game-held-out agreement / XOT strength experiment from Windows BATs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
AI_DIR = SCRIPT_DIR.parent
TRAIN_DIR = AI_DIR / "10_train_policy_network"
HUMAN_DIR = AI_DIR / "20_test_with_wthor"
STRENGTH_DIR = AI_DIR / "40_test_strength"
sys.path[:0] = [str(AI_DIR), str(HUMAN_DIR), str(STRENGTH_DIR)]

from game_dataset import file_sha256, load_manifest_positions
from evaluate_wthor_blend_human_match import discover_dat_files
from battle_blend_strength import OutputRunLock
from strength_elo_confidence import MINIMUM_CLUSTERS
from strength_tournament import atomic_write_json

# Pin the completed runs. Never silently pick a newer model during resume.
DEFAULT_RUNS = {
    "dense512": "wthor_game_split_512x4_e50_20261001_165329_704",
    "conv8_bn": "wthor_game_split_conv8_bn_e50_20261001_170120_867",
}
ALPHAS = "0,0.2,0.4,0.6,0.8,1"
BASELINES = "1,3,5,7,9,11,13,15,17,19"


def default_engine_resource(relative_path):
    repo = AI_DIR.parents[2]
    local = repo / relative_path
    sibling = repo.parent.parent / "Egaroucid" / relative_path
    return local if local.is_file() else sibling


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("both", "dense512", "conv8_bn"), default="both")
    for name, run in DEFAULT_RUNS.items():
        parser.add_argument(f"--{name.replace('_', '-')}-run", type=Path, default=TRAIN_DIR / "trained" / run)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "output" / "game_split_dense_conv8_bn_l21")
    parser.add_argument("--board-data-dir", type=Path)
    parser.add_argument("--egaroucid-exe", type=Path, default=default_engine_resource(
        "bin/versions/Egaroucid_for_Console_7_8_1_Windows_SIMD/Egaroucid_for_Console_7_8_1_SIMD.exe"))
    parser.add_argument("--openings", type=Path, default=default_engine_resource("bin/problem/xot/openingslarge.txt"))
    parser.add_argument("--positions", type=int, default=30000)
    parser.add_argument("--match-sets-per-pair", type=int, default=500)
    parser.add_argument("--human-workers", type=int, default=16)
    parser.add_argument("--human-threads", type=int, default=2)
    parser.add_argument("--parallel-match-sets", type=int, default=20)
    parser.add_argument("--baseline-processes-per-player", type=int, default=2)
    parser.add_argument("--blend-processes-per-player", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-workers", type=int, default=16)
    parser.add_argument("--stage", choices=("all", "human", "strength", "report"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_runs(args, names):
    if args.positions < 1 or args.match_sets_per_pair < MINIMUM_CLUSTERS:
        raise ValueError(f"positions must be positive; at least {MINIMUM_CLUSTERS} XOT sets per pair are needed for confidence intervals")
    if args.bootstrap_replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    if args.board_data_dir is None:
        data_root = os.environ.get("EGAROUCID_DATA")
        if not data_root:
            raise ValueError("Set EGAROUCID_DATA or pass --board-data-dir")
        args.board_data_dir = Path(data_root) / "train_data" / "board_data" / "records1"
    if not args.egaroucid_exe.is_file():
        raise FileNotFoundError(args.egaroucid_exe)
    if not args.openings.is_file():
        raise FileNotFoundError(args.openings)
    paths = discover_dat_files(args.board_data_dir)
    runs = {}
    reference = None
    for name in names:
        run = getattr(args, f"{name}_run").resolve()
        for filename in ("selected_model.h5", "selected_summary.json", "dataset_split.json"):
            if not (run / filename).is_file():
                raise FileNotFoundError(run / filename)
        selected = json.loads((run / "selected_summary.json").read_text(encoding="utf-8"))
        spec = selected["spec"]
        if name == "conv8_bn" and spec.get("architecture") != "conv8_bn":
            raise ValueError(f"Expected Conv8+BN: {run}")
        if name == "dense512" and (spec.get("architecture", "mlp") != "mlp" or spec["width"] != 512 or spec["depth"] != 4):
            raise ValueError(f"Expected Dense 512 x 4: {run}")
        if selected["epochs_ran"] != 50:
            raise ValueError(f"Expected the completed 50-epoch run: {run}")
        indices, manifest = load_manifest_positions(run / "dataset_split.json", paths, "test")
        if manifest["ratios"] != {"train": 0.9, "val": 0.05, "test": 0.05}:
            raise ValueError(f"Expected a 90/5/5 game split: {run}")
        if reference is not None and not np.array_equal(reference, indices):
            raise ValueError("Models must use the identical held-out test positions")
        reference = indices
        if args.positions > len(indices):
            raise ValueError(f"Requested {args.positions} positions, but test has {len(indices)}")
        runs[name] = run
        print(f"{name}: {run}\n  test: {len(indices):,} positions / {manifest['splits']['test']['games']:,} games", flush=True)
    return runs


def experiment_commands(args, run, destination):
    python = [sys.executable, "-u"]
    human = python + [str(HUMAN_DIR / "run_random_wthor_blend_experiment.py"), str(args.positions),
        "--model", str(run / "selected_model.h5"), "--split-manifest", str(run / "dataset_split.json"),
        "--data-split", "test", "--sample-seed", "613", "--board-data-dir", str(args.board_data_dir.resolve()),
        "--egaroucid-exe", str(args.egaroucid_exe.resolve()), "--workers", str(args.human_workers),
        "--egaroucid-threads", str(args.human_threads), "--policy-batch-size", "256",
        "--score-temperature", "1", "--hint-cache-dir", str(args.output_root / "shared_human_hints"),
        "--output-dir", str(destination / "human")]
    strength = python + [str(STRENGTH_DIR / "run_strength_full.py"),
        "--policy-model", str(run / "selected_model.h5"),
        "--weights", str(run / "selected_policy_network_weights.bin"),
        "--policy-backend", "tensorflow", "--logit-blend", "--baseline-levels", BASELINES,
        "--alphas", ALPHAS, "--blend-egaroucid-level", "21", "--score-temperature", "1",
        "--egaroucid-exe", str(args.egaroucid_exe.resolve()), "--seed", "57", "--engine-threads", "1",
        "--openings", str(args.openings.resolve()),
        "--match-sets-per-pair", str(args.match_sets_per_pair),
        "--parallel-match-sets", str(args.parallel_match_sets),
        "--baseline-processes-per-player", str(args.baseline_processes_per_player),
        "--blend-processes-per-player", str(args.blend_processes_per_player),
        "--output-dir", str(destination / "strength"), "--resume"]
    bootstrap = python + [str(STRENGTH_DIR / "reanalyze_strength_elo.py"),
        "--output-dir", str(destination / "strength"), "--bootstrap-seed", "613",
        "--bootstrap-replicates", str(args.bootstrap_replicates), "--workers", str(args.bootstrap_workers)]
    return {"human": human, "strength": strength, "bootstrap": bootstrap}


def launcher_identity(run, commands):
    # Refuse to mix outcomes from different data, weights, settings or code.
    sources = [AI_DIR / "game_dataset.py", Path(__file__), SCRIPT_DIR / "report_model_blend_experiment.py",
               AI_DIR.parents[2] / "bin" / "elo_rating_backcal.py"]
    for folder in (HUMAN_DIR, AI_DIR / "30_blend_with_egaroucid", STRENGTH_DIR):
        sources.extend(path for path in folder.glob("*.py") if not path.name.startswith("test_"))
    inputs = [run / name for name in ("selected_model.h5", "selected_summary.json", "dataset_split.json", "dataset_split.npz")]
    strength = commands["strength"]
    for option in ("--egaroucid-exe", "--openings"):
        inputs.append(Path(strength[strength.index(option) + 1]))
    engine = Path(strength[strength.index("--egaroucid-exe") + 1])
    inputs.extend(sorted(p for p in (engine.parent / "resources").rglob("*") if p.is_file()))
    return {"schema": 1, "commands": commands,
            "inputs_sha256": {str(p): file_sha256(p) for p in inputs},
            "sources_sha256": {str(p): file_sha256(p) for p in sources}}


def ensure_launcher_identity(path, identity, dry_run=False):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Experiment inputs/settings/code changed. Use the original settings or a new --output-root: {path}")
    elif not dry_run:
        atomic_write_json(path, identity)


def run_logged(command, log_path):
    print("RUN " + subprocess.list2cmdline(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nRUN " + subprocess.list2cmdline(command) + "\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", env=env) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def completed_stage(command, products, marker):
    if not marker.is_file() or not all(path.is_file() for path in products):
        return False
    expected = {"command": command, "products": {str(p): file_sha256(p) for p in products}}
    return json.loads(marker.read_text(encoding="utf-8")) == expected


def run_stage(command, products, destination, stage):
    marker = destination / f"{stage}_completed.json"
    if completed_stage(command, products, marker):
        print(f"SKIP completed {destination.name}/{stage}", flush=True)
        return
    run_logged(command, destination / f"{stage}.log")
    if stage == "strength":
        progress = json.loads((destination / "strength" / "strength_progress.json").read_text(encoding="utf-8"))
        if progress["remaining_match_sets"] != 0:
            raise RuntimeError("Strength tournament stopped before completion. Rerun the same BAT to resume.")
    atomic_write_json(marker, {"command": command, "products": {str(p): file_sha256(p) for p in products}})


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = make_argparser().parse_args()
    args.output_root = args.output_root.resolve()
    names = list(DEFAULT_RUNS) if args.models == "both" else [args.models]
    runs = validate_runs(args, names)
    print(f"alpha = {ALPHAS}; engine level = 21; temperature = 1", flush=True)
    print(f"Per model: {args.positions:,} test positions; {120 * args.match_sets_per_pair:,} paired sets / {240 * args.match_sets_per_pair:,} actual games", flush=True)
    plans = {}
    with OutputRunLock(args.output_root):
        for name, run in runs.items():
            destination = args.output_root / name
            commands = experiment_commands(args, run, destination)
            ensure_launcher_identity(destination / "launch_manifest.json", launcher_identity(run, commands), args.dry_run)
            plans[name] = (destination, commands)
        if args.dry_run:
            for name, (destination, commands) in plans.items():
                for stage in ("human", "strength"):
                    run_logged(commands[stage] + ["--dry-run"], destination / f"{stage}_dry_run.log")
            print("Dry run complete. No engine searches or games were started.", flush=True)
            return
        if args.stage == "all":
            # Catch missing runtime inputs and insufficient RAM before spending
            # hours evaluating human agreement.
            for destination, commands in plans.values():
                run_logged(commands["strength"] + ["--dry-run"], destination / "strength_preflight.log")
        if args.stage in ("all", "human"):
            for destination, commands in plans.values():
                run_stage(commands["human"], [destination / "human" / "random_wthor_blend_summary.json"], destination, "human")
        if args.stage in ("all", "strength"):
            for destination, commands in plans.values():
                run_stage(commands["strength"], [destination / "strength" / "strength_results.json",
                          destination / "strength" / "strength_games.jsonl"], destination, "strength")
        if args.stage in ("all", "report"):
            for destination, commands in plans.values():
                # Raw logs are part of the checkpoint: replacing them forces reanalysis.
                run_stage(commands["bootstrap"], [destination / "strength" / "strength_elo_ci95.json",
                          destination / "strength" / "strength_elo_bootstrap_replicates.csv",
                          destination / "strength" / "strength_results.json",
                          destination / "strength" / "strength_manifest.json",
                          destination / "strength" / "strength_games.jsonl"], destination, "bootstrap")
            from report_model_blend_experiment import write_report
            write_report(args.output_root, names)
        print(f"Completed stage '{args.stage}'. Results: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
