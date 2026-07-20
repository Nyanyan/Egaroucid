#!/usr/bin/env python3
"""Fast round-robin strength test for native and blended Egaroucid policies.

One match set consists of two games from the same XOT opening with colors
swapped. The sign of the two-game mean disc difference determines the set
result. The set, not either individual game, is the independent observation
used for the reported score and its 95% confidence interval.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import queue
import random
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
HUMAN_LIKE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BIN_DIR = REPO_ROOT / "bin"
BLEND_DIR = HUMAN_LIKE_DIR / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import (  # noqa: E402
    default_egaroucid_exe,
    default_weights_file,
)
from strength_engine import (  # noqa: E402
    AvailableMemoryLimitError,
    GameRunner,
    ProcessManager,
    available_memory_mib,
    validate_xot_opening,
)
from strength_reporting import (  # noqa: E402
    PerformanceMonitor,
    build_text_report,
    estimate_elos,
    format_elapsed,
    write_outputs,
)
from strength_tournament import (  # noqa: E402
    GameResult,
    MatchSetTask,
    PendingTaskQueue,
    PlayerSpec,
    ResultStore,
    TournamentStats,
    atomic_write_json,
    combine_color_games,
    conservative_score_half_width,
    ensure_manifest,
    limit_tasks,
    make_manifest,
    make_match_set_tasks,
    normalized_duration_weights,
    sha256_file,
    target_match_sets_by_pair,
)


IMPLEMENTATION_REVISION = "clean-strength-tournament-v4"


class OutputRunLock:
    """Operating-system lock preventing two writers in one output directory."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "strength_run.lock"
        self.stream = None

    def __enter__(self) -> "OutputRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    self.stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError(
                f"another tournament is using {self.path.parent}"
            ) from error
        return self

    def __exit__(self, exc_type, exc, traceback_object) -> None:
        if self.stream is None:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


def clear_abandoned_hint_claims(path: Path) -> int:
    """Remove claims left by a killed run after the output lock is held."""

    if not path.exists():
        return 0
    connection = sqlite3.connect(str(path), timeout=60.0)
    try:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'hint_claims'"
        ).fetchone()
        if table_exists is None:
            return 0
        cursor = connection.execute("DELETE FROM hint_claims")
        connection.commit()
        return max(0, int(cursor.rowcount))
    finally:
        connection.close()


def parse_int_list(text: str) -> List[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate integer in {text!r}")
    return values


def parse_float_list(text: str) -> List[float]:
    values = [
        float(token.strip()) for token in text.split(",") if token.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate value in {text!r}")
    return values


def format_alpha(alpha: float) -> str:
    if alpha == 0.0:
        return "0.0"
    if alpha == 1.0:
        return "1.0"
    # repr(float) is the shortest text that round-trips to the same binary
    # value; distinct accepted alpha values therefore cannot collide merely
    # because a display formatter rounded them.
    return repr(alpha)


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def display_command(command: Sequence[str]) -> List[str]:
    displayed: List[str] = []
    for part in command:
        try:
            if Path(part).resolve() == Path(sys.executable).resolve():
                displayed.append("python")
                continue
        except (OSError, RuntimeError):
            pass
        path = Path(part)
        displayed.append(
            repo_relative(path)
            if path.is_absolute() or path.exists()
            else part
        )
    try:
        port_index = displayed.index("--policy-server-port") + 1
    except ValueError:
        return displayed
    if port_index < len(displayed) and displayed[port_index] == "0":
        displayed[port_index] = "<managed-at-runtime>"
    return displayed


def read_openings(path: Path, seed: int) -> List[str]:
    openings = [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not openings:
        raise ValueError(f"no XOT openings found in {path}")
    random.Random(seed).shuffle(openings)
    return openings


def validate_scheduled_openings(
    openings: Sequence[str],
    match_sets_per_pair: int,
) -> None:
    if not openings:
        raise ValueError("at least one XOT opening is required")
    scheduled = [
        openings[index % len(openings)]
        for index in range(match_sets_per_pair)
    ]
    if len(set(scheduled)) != len(scheduled):
        raise ValueError(
            "each match-set index must use a distinct XOT opening for the "
            "reported confidence intervals; provide at least "
            f"{match_sets_per_pair} unique openings"
        )
    for set_index, opening in enumerate(scheduled):
        try:
            validate_xot_opening(opening)
        except ValueError as error:
            raise ValueError(
                f"invalid XOT opening at scheduled set index {set_index}: "
                f"{opening!r}"
            ) from error


def validate_args(args: argparse.Namespace) -> Tuple[List[int], List[float]]:
    baseline_levels = parse_int_list(args.baseline_levels)
    alphas = parse_float_list(args.alphas)
    if any(not 0 <= level <= 60 for level in baseline_levels):
        raise ValueError("baseline levels must be between 0 and 60")
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("all alpha values must be between 0 and 1")
    if len(baseline_levels) + len(alphas) < 2:
        raise ValueError("at least two total participants are required")
    if args.match_sets_per_pair < 1:
        raise ValueError("--match-sets-per-pair must be positive")
    if args.parallel_match_sets < 1:
        raise ValueError("--parallel-match-sets must be positive")
    for option, value in (
        ("--baseline-processes-per-player", args.baseline_processes_per_player),
        ("--blend-processes-per-player", args.blend_processes_per_player),
    ):
        if value < 2 or value % 2:
            raise ValueError(f"{option} must be an even number >= 2")
    if args.engine_threads < 1:
        raise ValueError("--engine-threads must be positive")
    if not 0 <= args.blend_egaroucid_level <= 60:
        raise ValueError("--blend-egaroucid-level must be between 0 and 60")
    if args.score_temperature <= 0.0:
        raise ValueError("--score-temperature must be positive")
    if args.status_every_match_sets < 1:
        raise ValueError("--status-every-match-sets must be positive")
    if args.task_retries < 0:
        raise ValueError("--task-retries must be non-negative")
    if args.time_limit_sec is not None and args.time_limit_sec <= 0.0:
        raise ValueError("--time-limit-sec must be positive")
    if args.gtp_command_timeout_sec <= 0.0:
        raise ValueError("--gtp-command-timeout-sec must be positive")
    if args.egaroucid_hint_timeout_sec <= 0.0:
        raise ValueError("--egaroucid-hint-timeout-sec must be positive")
    if args.policy_server_timeout_sec <= 0.0:
        raise ValueError("--policy-server-timeout-sec must be positive")
    if args.policy_server_startup_timeout_sec <= 0.0:
        raise ValueError(
            "--policy-server-startup-timeout-sec must be positive"
        )
    if args.policy_max_batch_size < 1:
        raise ValueError("--policy-max-batch-size must be positive")
    if args.policy_batch_wait_ms < 0.0:
        raise ValueError("--policy-batch-wait-ms must be non-negative")
    if args.policy_inference_threads < 1:
        raise ValueError("--policy-inference-threads must be positive")
    if args.performance_sample_interval_sec <= 0.0:
        raise ValueError(
            "--performance-sample-interval-sec must be positive"
        )
    if args.minimum_available_memory_mib <= 0.0:
        raise ValueError("--minimum-available-memory-mib must be positive")
    if args.estimated_engine_memory_mib <= 0.0:
        raise ValueError("--estimated-engine-memory-mib must be positive")
    if args.estimated_wrapper_memory_mib <= 0.0:
        raise ValueError("--estimated-wrapper-memory-mib must be positive")
    if args.estimated_policy_server_memory_mib <= 0.0:
        raise ValueError(
            "--estimated-policy-server-memory-mib must be positive"
        )
    return baseline_levels, alphas


def policy_model_path(args: argparse.Namespace) -> Path:
    if args.policy_model is not None:
        return Path(args.policy_model)
    return Path(args.weights).with_name("selected_model.h5")


def validate_input_files(args: argparse.Namespace, alphas: Sequence[float]) -> None:
    required = {
        "Egaroucid executable": Path(args.egaroucid_exe),
        "XOT openings": Path(args.openings),
    }
    if any(alpha > 0.0 for alpha in alphas):
        required["policy weights"] = Path(args.weights)
        if args.policy_backend == "tensorflow":
            required["policy model"] = policy_model_path(args)
            if importlib.util.find_spec("tensorflow") is None:
                raise ModuleNotFoundError(
                    "TensorFlow is required by --policy-backend tensorflow"
                )
    missing = [
        f"{description}: {path}"
        for description, path in required.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "required tournament input is missing:\n" + "\n".join(missing)
        )


@dataclass(frozen=True)
class PolicyServerInfo:
    port: int
    backend: str
    device: str

    @property
    def runtime(self) -> str:
        return f"{self.backend}/{self.device}"


def parse_policy_server_ready(
    line: str,
    requested_backend: str,
    allow_policy_cpu: bool,
) -> PolicyServerInfo:
    parts = line.strip().split()
    if len(parts) != 4 or parts[0] != "READY":
        raise ValueError(f"invalid policy server READY line: {line!r}")
    try:
        port = int(parts[1])
    except ValueError as error:
        raise ValueError(
            f"invalid policy server port in READY line: {line!r}"
        ) from error
    if not 1 <= port <= 65535:
        raise ValueError(f"policy server port is out of range: {port}")
    backend = parts[2]
    device = parts[3]
    if backend != requested_backend:
        raise ValueError(
            "policy inference server selected an unexpected backend: "
            f"requested {requested_backend!r}, got {backend!r}"
        )
    if (
        backend == "tensorflow"
        and device != "GPU"
        and not allow_policy_cpu
    ):
        raise ValueError(
            "TensorFlow policy inference started without a GPU; use "
            "--allow-policy-cpu only when this slower condition is intended"
        )
    return PolicyServerInfo(port=port, backend=backend, device=device)


def start_policy_server(
    args: argparse.Namespace,
    manager: ProcessManager,
    run_id: str,
) -> PolicyServerInfo:
    server_script = BLEND_DIR / "policy_batch_server.py"
    command = [
        sys.executable,
        str(server_script.resolve()),
        "--weights",
        str(Path(args.weights).resolve()),
        "--model",
        str(policy_model_path(args).resolve()),
        "--backend",
        args.policy_backend,
        "--host",
        args.policy_server_host,
        "--port",
        "0",
        "--max-batch-size",
        str(args.policy_max_batch_size),
        "--batch-wait-ms",
        str(args.policy_batch_wait_ms),
        "--stats-path",
        str(
            (
                args.output_dir
                / f"policy_batch_server_stats_{run_id}.json"
            ).resolve()
        ),
    ]
    environment = os.environ.copy()
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    environment["OPENBLAS_NUM_THREADS"] = str(
        args.policy_inference_threads
    )
    environment["OMP_NUM_THREADS"] = str(args.policy_inference_threads)
    process = manager.spawn(command, env=environment)
    ready: queue.Queue[object] = queue.Queue()

    def read_ready_line() -> None:
        try:
            if process.stdout is None:
                ready.put(RuntimeError("policy server stdout is unavailable"))
            else:
                ready.put(process.stdout.readline())
        except BaseException as error:
            ready.put(error)

    threading.Thread(
        target=read_ready_line,
        name="policy-server-ready",
        daemon=True,
    ).start()
    try:
        line = ready.get(
            timeout=args.policy_server_startup_timeout_sec
        )
    except queue.Empty:
        manager.terminate(process, graceful=False)
        raise TimeoutError("policy inference server startup timed out")
    if isinstance(line, BaseException):
        manager.terminate(process, graceful=False)
        raise RuntimeError("policy inference server failed to start") from line
    try:
        info = parse_policy_server_ready(
            str(line),
            requested_backend=args.policy_backend,
            allow_policy_cpu=args.allow_policy_cpu,
        )
    except ValueError as error:
        manager.terminate(process, graceful=False)
        raise RuntimeError(
            f"policy inference server failed to start: {line!r}"
        ) from error
    return info


def build_player_specs(
    args: argparse.Namespace,
    baseline_levels: Sequence[int],
    alphas: Sequence[float],
    policy_server_port: int,
) -> List[PlayerSpec]:
    egaroucid_exe = str(Path(args.egaroucid_exe).resolve())
    specs: List[PlayerSpec] = []
    for level in baseline_levels:
        specs.append(
            PlayerSpec(
                name=f"egaroucid_l{level}",
                command=(
                    egaroucid_exe,
                    "-gtp",
                    "-quiet",
                    "-nobook",
                    "-l",
                    str(level),
                    "-t",
                    str(args.engine_threads),
                ),
                processes_per_player=args.baseline_processes_per_player,
                setboard_before_genmove=False,
            )
        )

    blend_script = BLEND_DIR / "blend_gtp_engine.py"
    hint_cache_db = (
        args.output_dir / "egaroucid_hint_cache_v2.sqlite3"
    ).resolve()
    for alpha in alphas:
        alpha_text = format_alpha(alpha)
        if alpha == 0.0:
            command = (
                egaroucid_exe,
                "-gtp",
                "-quiet",
                "-nobook",
                "-l",
                str(args.blend_egaroucid_level),
                "-t",
                str(args.engine_threads),
            )
            setboard_before_genmove = False
        else:
            command_parts = [
                sys.executable,
                str(blend_script.resolve()),
                "--weights",
                str(Path(args.weights).resolve()),
                "--alpha",
                alpha_text,
                "--egaroucid-exe",
                egaroucid_exe,
                "--egaroucid-level",
                str(args.blend_egaroucid_level),
                "--egaroucid-threads",
                str(args.engine_threads),
                "--egaroucid-timeout-sec",
                str(args.egaroucid_hint_timeout_sec),
                "--minimum-available-memory-mib",
                str(args.minimum_available_memory_mib),
                "--estimated-egaroucid-memory-mib",
                str(args.estimated_engine_memory_mib),
                "--score-temperature",
                str(args.score_temperature),
                "--policy-server-host",
                args.policy_server_host,
                "--policy-server-port",
                str(policy_server_port),
                "--policy-server-timeout-sec",
                str(args.policy_server_timeout_sec),
            ]
            if not args.no_hint_cache:
                command_parts.extend(
                    ["--hint-cache-db", str(hint_cache_db)]
                )
            command = tuple(command_parts)
            setboard_before_genmove = True
        specs.append(
            PlayerSpec(
                name=f"alpha_{alpha_text}",
                command=tuple(command),
                processes_per_player=args.blend_processes_per_player,
                setboard_before_genmove=setboard_before_genmove,
                alpha=alpha,
            )
        )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("participant names must be unique")
    return specs


def task_plan_hash(tasks: Sequence[MatchSetTask]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        digest.update(
            (
                f"{task.task_id}:{task.p0_idx}:{task.p1_idx}:"
                f"{task.set_index}:{task.opening}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    return {
        "path": repo_relative(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def implementation_file_identities() -> Dict[str, dict]:
    """Fingerprint every local source file that can affect this experiment."""

    paths = {
        "battle_orchestrator": SCRIPT_DIR / "battle_blend_strength.py",
        "entry_point": SCRIPT_DIR / "run_strength_full.py",
        "engine_runner": SCRIPT_DIR / "strength_engine.py",
        "reporting": SCRIPT_DIR / "strength_reporting.py",
        "tournament_model": SCRIPT_DIR / "strength_tournament.py",
        "blend_gtp_engine": BLEND_DIR / "blend_gtp_engine.py",
        "blend_policy": BLEND_DIR / "blend_policy_with_egaroucid.py",
        "policy_batch_server": BLEND_DIR / "policy_batch_server.py",
        "elo_fitter": BIN_DIR / "elo_rating_backcal.py",
    }
    return {
        name: file_identity(path)
        for name, path in sorted(paths.items())
    }


def package_runtime_versions(policy_backend: str) -> dict:
    packages = ["numpy", "psutil"]
    if policy_backend == "tensorflow":
        packages.append("tensorflow")
    versions: Dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version,
        "packages": versions,
    }


def gpu_runtime_identity() -> List[str]:
    """Return stable GPU model/driver rows without initializing TensorFlow."""

    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if process.returncode != 0:
        return []
    return sorted(
        line.strip()
        for line in process.stdout.splitlines()
        if line.strip()
    )


def build_manifest_configuration(
    args: argparse.Namespace,
    baseline_levels: Sequence[int],
    alphas: Sequence[float],
    openings: Sequence[str],
    tasks: Sequence[MatchSetTask],
) -> dict:
    inputs: Dict[str, object] = {
        "egaroucid_executable": file_identity(Path(args.egaroucid_exe)),
        "xot_openings": file_identity(Path(args.openings)),
    }
    if any(alpha > 0.0 for alpha in alphas):
        inputs["policy_weights"] = file_identity(Path(args.weights))
        if args.policy_backend == "tensorflow":
            inputs["policy_model"] = file_identity(policy_model_path(args))
    inputs["implementation_sources"] = implementation_file_identities()
    opening_sequence_digest = hashlib.sha256(
        ("\n".join(openings) + "\n").encode("ascii")
    ).hexdigest()
    return {
        "implementation_revision": IMPLEMENTATION_REVISION,
        "participants": {
            "baseline_levels": list(baseline_levels),
            "alphas": list(alphas),
        },
        "schedule": {
            "match_sets_per_pair": args.match_sets_per_pair,
            "max_match_sets": args.max_match_sets,
            "xot_seed": args.seed,
            "same_opening_sequence_for_every_pair": True,
            "max_set_index_lookahead": (
                PendingTaskQueue.MAX_SET_INDEX_LOOKAHEAD
            ),
            "shuffled_opening_sequence_sha256": opening_sequence_digest,
            "task_plan_sha256": task_plan_hash(tasks),
            "total_match_sets": len(tasks),
        },
        "scoring": {
            "actual_games_per_match_set": 2,
            "color_swapped": True,
            "paired_set_result": "sign of mean p0 disc difference",
        },
        "engine": {
            "parallel_match_sets": args.parallel_match_sets,
            "baseline_processes_per_player": (
                args.baseline_processes_per_player
            ),
            "blend_processes_per_player": args.blend_processes_per_player,
            "engine_threads": args.engine_threads,
            "blend_egaroucid_level": args.blend_egaroucid_level,
            "score_temperature": args.score_temperature,
            "hint_cache": not args.no_hint_cache,
            "gtp_command_timeout_sec": args.gtp_command_timeout_sec,
            "egaroucid_hint_timeout_sec": (
                args.egaroucid_hint_timeout_sec
            ),
        },
        "policy_inference": {
            "backend": args.policy_backend,
            "managed_server": True,
            "required_device": (
                "CPU"
                if args.policy_backend == "numpy"
                else ("CPU-or-GPU" if args.allow_policy_cpu else "GPU")
            ),
            "max_batch_size": args.policy_max_batch_size,
            "batch_wait_ms": args.policy_batch_wait_ms,
            "inference_threads": args.policy_inference_threads,
            "request_timeout_sec": args.policy_server_timeout_sec,
            "startup_timeout_sec": args.policy_server_startup_timeout_sec,
        },
        "runtime_versions": package_runtime_versions(args.policy_backend),
        "inputs": inputs,
    }


@dataclass
class InFlightMatchSet:
    task: MatchSetTask
    started_at: float
    futures: Dict[Future, bool] = field(default_factory=dict)
    results: Dict[bool, GameResult] = field(default_factory=dict)
    errors: List[BaseException] = field(default_factory=list)


def print_configuration(
    args: argparse.Namespace,
    specs: Sequence[PlayerSpec],
    total_match_sets: int,
    startup_available_mib: float,
    estimated_max_player_processes: int,
    estimated_heavy_engine_processes: int,
    estimated_max_memory_mib: float,
    policy_runtime: str,
) -> None:
    capacities = [
        spec.concurrent_match_set_capacity for spec in specs
    ]
    capacity_limited_sets = sum(capacities) // 2
    effective_sets = min(args.parallel_match_sets, capacity_limited_sets)
    print("players")
    for spec in specs:
        print(spec.name, " ".join(display_command(spec.command)))
    print("match_sets_per_pair", args.match_sets_per_pair)
    print("actual_games_per_pair", args.match_sets_per_pair * 2)
    if args.max_match_sets is not None:
        print("max_match_sets", args.max_match_sets)
    print("requested_parallel_match_sets", args.parallel_match_sets)
    print("capacity_limited_parallel_match_sets", capacity_limited_sets)
    print("effective_parallel_match_sets", effective_sets)
    print("actual_game_worker_threads", args.parallel_match_sets * 2)
    print("theoretical_concurrent_actual_games", effective_sets * 2)
    print("baseline_processes_per_player", args.baseline_processes_per_player)
    print("blend_processes_per_player", args.blend_processes_per_player)
    print("engine_threads_per_process", args.engine_threads)
    print("xot_openings", repo_relative(Path(args.openings)))
    print("same_opening_sequence_for_every_pair", True)
    print(
        "max_set_index_lookahead",
        PendingTaskQueue.MAX_SET_INDEX_LOOKAHEAD,
    )
    print("xot_shuffle_seed", args.seed)
    print("policy_server_runtime", policy_runtime)
    print("hint_cache", not args.no_hint_cache)
    print(
        "hint_cache_db",
        (
            repo_relative(
                args.output_dir / "egaroucid_hint_cache_v2.sqlite3"
            )
            if not args.no_hint_cache
            else "-"
        ),
    )
    print("gtp_command_timeout_sec", args.gtp_command_timeout_sec)
    print("egaroucid_hint_timeout_sec", args.egaroucid_hint_timeout_sec)
    print("startup_available_memory_mib", f"{startup_available_mib:.0f}")
    print("minimum_available_memory_mib", args.minimum_available_memory_mib)
    print(
        "estimated_max_player_processes",
        estimated_max_player_processes,
    )
    print(
        "estimated_heavy_egaroucid_processes",
        estimated_heavy_engine_processes,
    )
    print(
        "estimated_max_tournament_memory_mib",
        estimated_max_memory_mib,
    )
    print("total_match_sets", total_match_sets)
    print("total_actual_games", total_match_sets * 2)
    print("output_dir", repo_relative(args.output_dir))


def print_status(
    specs: Sequence[PlayerSpec],
    stats: TournamentStats,
    start_time: float,
    completed_match_sets: int,
    completed_at_start: int,
    total_match_sets: int,
    target_matrix: Sequence[Sequence[int]],
) -> None:
    elapsed = time.time() - start_time
    completed_this_run = completed_match_sets - completed_at_start
    speed = completed_this_run / elapsed if elapsed > 0.0 else 0.0
    remaining = max(0, total_match_sets - completed_match_sets)
    eta = remaining / speed if speed > 0.0 else 0.0
    percent = 100.0 * completed_match_sets / max(1, total_match_sets)
    print("\n" + "=" * 80)
    print(
        f"Progress: {completed_match_sets}/{total_match_sets} paired sets "
        f"({completed_match_sets * 2}/{total_match_sets * 2} actual games, "
        f"{percent:.2f}%)"
    )
    print(
        f"Elapsed this run: {format_elapsed(elapsed)}  "
        f"ETA: {format_elapsed(eta)}  "
        f"Speed: {speed:.3f} paired sets/sec"
    )
    ratings = estimate_elos(specs, stats)
    print(build_text_report(specs, stats, target_matrix, ratings))


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a timeout-safe, resumable XOT round robin for Egaroucid "
            "levels and blended policies."
        )
    )
    parser.add_argument(
        "--baseline-levels",
        default="1,3,5,7,9,11,13,15,17,19",
    )
    parser.add_argument(
        "--alphas",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
    )
    parser.add_argument(
        "--match-sets-per-pair",
        type=int,
        default=500,
        help=(
            "Paired XOT observations per player pair. Each set contains two "
            "color-swapped actual games."
        ),
    )
    parser.add_argument(
        "--max-match-sets",
        type=int,
        default=None,
        help="Limit the whole schedule for a benchmark or smoke test.",
    )
    parser.add_argument(
        "--parallel-match-sets",
        type=int,
        default=20,
        help="Maximum admitted color-swapped sets.",
    )
    parser.add_argument(
        "--baseline-processes-per-player",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--blend-processes-per-player",
        type=int,
        default=10,
    )
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument(
        "--status-every-match-sets",
        type=int,
        default=100,
    )
    parser.add_argument("--task-retries", type=int, default=2)
    parser.add_argument("--time-limit-sec", type=float, default=None)
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument(
        "--policy-model",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--egaroucid-exe",
        type=Path,
        default=default_egaroucid_exe(),
    )
    parser.add_argument("--blend-egaroucid-level", type=int, default=21)
    parser.add_argument(
        "--gtp-command-timeout-sec",
        type=float,
        default=1900.0,
        help="Outer timeout for every GTP command, including native genmove.",
    )
    parser.add_argument(
        "--egaroucid-hint-timeout-sec",
        type=float,
        default=1800.0,
        help="Timeout used inside each blended engine for one hint command.",
    )
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument(
        "--openings",
        type=Path,
        default=BIN_DIR / "problem" / "xot" / "openingslarge.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output" / "xot_500sets_16players",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=57,
        help="Deterministic XOT shuffle and pair-order seed.",
    )
    parser.add_argument(
        "--no-hint-cache",
        action="store_true",
        help="Disable Egaroucid hint caching in blended engines.",
    )
    parser.add_argument(
        "--policy-backend",
        choices=("tensorflow", "numpy"),
        default="tensorflow",
    )
    parser.add_argument("--policy-server-host", default="127.0.0.1")
    parser.add_argument(
        "--allow-policy-cpu",
        action="store_true",
        help=(
            "Allow TensorFlow inference without a GPU. By default this is "
            "rejected so a long experiment cannot silently change device."
        ),
    )
    parser.add_argument(
        "--policy-server-timeout-sec",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--policy-server-startup-timeout-sec",
        type=float,
        default=120.0,
    )
    parser.add_argument("--policy-max-batch-size", type=int, default=32)
    parser.add_argument("--policy-batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--policy-inference-threads", type=int, default=4)
    parser.add_argument("--no-performance-monitor", action="store_true")
    parser.add_argument(
        "--performance-sample-interval-sec",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--minimum-available-memory-mib",
        type=float,
        default=16384.0,
        help=(
            "Stop starting engines below this much free RAM. The 16 GiB "
            "reserve is for the default 128 GiB workstation profile."
        ),
    )
    parser.add_argument(
        "--estimated-engine-memory-mib",
        type=float,
        default=1260.0,
        help="Measured resident-memory estimate per Egaroucid process.",
    )
    parser.add_argument(
        "--estimated-wrapper-memory-mib",
        type=float,
        default=40.0,
        help="Memory estimate per Python blended-policy GTP wrapper.",
    )
    parser.add_argument(
        "--estimated-policy-server-memory-mib",
        type=float,
        default=2000.0,
        help="System-memory estimate for the shared policy inference server.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_tournament(args: argparse.Namespace) -> None:
    baseline_levels, alphas = validate_args(args)
    validate_input_files(args, alphas)
    args.output_dir = Path(args.output_dir)

    openings = read_openings(Path(args.openings), args.seed)
    validate_scheduled_openings(openings, args.match_sets_per_pair)
    player_count = len(baseline_levels) + len(alphas)
    all_tasks = make_match_set_tasks(
        player_count,
        openings,
        args.match_sets_per_pair,
        args.seed,
    )
    tasks = limit_tasks(all_tasks, args.max_match_sets)
    target_matrix = target_match_sets_by_pair(tasks, player_count)
    tasks_by_id = {task.task_id: task for task in tasks}
    results_path = args.output_dir / "strength_games.jsonl"
    if results_path.exists() and not args.resume:
        raise FileExistsError(
            f"{results_path} already contains results; use --resume with "
            "the identical experiment or choose a new --output-dir"
        )

    configuration = build_manifest_configuration(
        args,
        baseline_levels,
        alphas,
        openings,
        tasks,
    )
    manifest = make_manifest(configuration)
    manifest_path = args.output_dir / "strength_manifest.json"
    ensure_manifest(
        manifest_path,
        manifest,
        resume=args.resume,
        results_exist=results_path.exists(),
    )
    experiment_id = str(manifest["experiment_id"])
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"_{os.getpid()}"
    )

    estimated_max_player_processes = (
        len(baseline_levels) * args.baseline_processes_per_player
        + len(alphas) * args.blend_processes_per_player
    )
    # alpha=1 uses the policy network only; its Python wrapper never starts
    # the lazily constructed Egaroucid hint subprocess.
    egaroucid_alpha_count = sum(alpha != 1.0 for alpha in alphas)
    estimated_heavy_engine_processes = (
        len(baseline_levels) * args.baseline_processes_per_player
        + egaroucid_alpha_count * args.blend_processes_per_player
    )
    estimated_wrapper_processes = (
        sum(alpha > 0.0 for alpha in alphas)
        * args.blend_processes_per_player
    )
    startup_available_mib = available_memory_mib()
    if startup_available_mib is None:
        raise RuntimeError("psutil is required for the memory safety check")
    estimated_memory_mib = (
        estimated_heavy_engine_processes
        * args.estimated_engine_memory_mib
        + estimated_wrapper_processes
        * args.estimated_wrapper_memory_mib
        + (
            args.estimated_policy_server_memory_mib
            if any(alpha > 0.0 for alpha in alphas)
            else 0.0
        )
    )
    if (
        estimated_memory_mib + args.minimum_available_memory_mib
        > startup_available_mib
    ):
        raise MemoryError(
            "memory capacity check failed: "
            f"{estimated_heavy_engine_processes} Egaroucid processes and "
            f"{estimated_wrapper_processes} wrappers are estimated to use "
            f"{estimated_memory_mib:.0f} MiB, only "
            f"{startup_available_mib:.0f} MiB is currently available, and "
            f"{args.minimum_available_memory_mib:.0f} MiB must remain free"
        )

    store = ResultStore(args.output_dir)
    completed_task_ids: set[int] = set()
    loaded_results = []
    if args.resume:
        completed_task_ids, loaded_results = store.load(tasks_by_id)
    stats = TournamentStats(player_count)
    for result in loaded_results:
        stats.record(result)
    completed_match_sets = len(loaded_results)
    completed_at_start = completed_match_sets
    remaining_tasks = [
        task for task in tasks if task.task_id not in completed_task_ids
    ]

    placeholder_specs = build_player_specs(
        args,
        baseline_levels,
        alphas,
        policy_server_port=0,
    )
    if args.dry_run:
        print_configuration(
            args,
            placeholder_specs,
            len(tasks),
            startup_available_mib,
            estimated_max_player_processes,
            estimated_heavy_engine_processes,
            estimated_memory_mib,
            "dry-run",
        )
        atomic_write_json(
            args.output_dir / "strength_dry_run.json",
            {
                "experiment_id": experiment_id,
                "players": [
                    {
                        "name": spec.name,
                        "command": display_command(spec.command),
                        "processes_per_player": spec.processes_per_player,
                        "concurrent_match_set_capacity": (
                            spec.concurrent_match_set_capacity
                        ),
                    }
                    for spec in placeholder_specs
                ],
                "match_sets_per_pair": args.match_sets_per_pair,
                "actual_games_per_pair": args.match_sets_per_pair * 2,
                "total_match_sets": len(tasks),
                "total_actual_games": len(tasks) * 2,
                "parallel_match_sets": args.parallel_match_sets,
                "capacity_limited_parallel_match_sets": (
                    sum(
                        spec.concurrent_match_set_capacity
                        for spec in placeholder_specs
                    )
                    // 2
                ),
                "estimated_max_player_processes": (
                    estimated_max_player_processes
                ),
                "estimated_heavy_egaroucid_processes": (
                    estimated_heavy_engine_processes
                ),
                "estimated_max_tournament_memory_mib": (
                    estimated_memory_mib
                ),
                "startup_available_memory_mib": startup_available_mib,
                "minimum_available_memory_mib": (
                    args.minimum_available_memory_mib
                ),
                "planning_ci95_half_width_at_50_percent": {
                    str(match_sets): conservative_score_half_width(match_sets)
                    for match_sets in (50, 100, 200, 300, 500)
                },
                "resume_completed_match_sets": completed_match_sets,
                "remaining_match_sets": len(remaining_tasks),
            },
        )
        return

    if not remaining_tasks:
        print("The manifest-matched tournament is already complete.")
        write_outputs(
            placeholder_specs,
            stats,
            args.output_dir,
            completed_match_sets,
            len(tasks),
            target_matrix,
            experiment_id,
        )
        previous_runtime = "not-started"
        progress_path = args.output_dir / "strength_progress.json"
        if progress_path.exists():
            try:
                previous_progress = json.loads(
                    progress_path.read_text(encoding="utf-8")
                )
                if previous_progress.get("experiment_id") == experiment_id:
                    previous_runtime = str(
                        previous_progress.get(
                            "policy_server_runtime",
                            previous_runtime,
                        )
                    )
            except (OSError, ValueError):
                pass
        atomic_write_json(
            progress_path,
            {
                "schema_version": 3,
                "experiment_id": experiment_id,
                "completed_match_sets": completed_match_sets,
                "total_match_sets": len(tasks),
                "completed_actual_games": completed_match_sets * 2,
                "total_actual_games": len(tasks) * 2,
                "remaining_match_sets": 0,
                "remaining_actual_games": 0,
                "stop_reason": "already_complete",
                "run_id": run_id,
                "policy_server_runtime": previous_runtime,
                "elapsed_sec_this_run": 0.0,
            },
        )
        return

    if not args.no_hint_cache:
        cleared_claims = clear_abandoned_hint_claims(
            args.output_dir / "egaroucid_hint_cache_v2.sqlite3"
        )
        if cleared_claims:
            print("cleared_abandoned_hint_claims", cleared_claims)

    manager = ProcessManager(args.minimum_available_memory_mib)
    monitor: Optional[PerformanceMonitor] = None
    store.open()
    start_time = time.time()
    stop_reason = "finished"
    fatal_error: Optional[BaseException] = None
    executor: Optional[ThreadPoolExecutor] = None
    last_reported_match_sets: Optional[int] = None
    try:
        if any(alpha > 0.0 for alpha in alphas):
            policy_server = start_policy_server(args, manager, run_id)
        else:
            policy_server = PolicyServerInfo(0, "none", "not-needed")
        runtime_manifest = make_manifest(
            {
                "kind": "strength-tournament-runtime",
                "experiment_id": experiment_id,
                "policy_server": {
                    "backend": policy_server.backend,
                    "device": policy_server.device,
                },
                "gpu_model_and_driver": (
                    gpu_runtime_identity()
                    if policy_server.device == "GPU"
                    else []
                ),
                "runtime_versions": package_runtime_versions(
                    args.policy_backend
                ),
            }
        )
        ensure_manifest(
            args.output_dir / "strength_runtime_manifest.json",
            runtime_manifest,
            resume=args.resume,
            results_exist=results_path.exists(),
        )
        specs = build_player_specs(
            args,
            baseline_levels,
            alphas,
            policy_server.port,
        )
        print_configuration(
            args,
            specs,
            len(tasks),
            startup_available_mib,
            estimated_max_player_processes,
            estimated_heavy_engine_processes,
            estimated_memory_mib,
            policy_server.runtime,
        )
        if args.resume:
            print("resume_completed_match_sets", completed_match_sets)
            print("remaining_match_sets", len(remaining_tasks))

        game_runner = GameRunner(
            specs,
            manager,
            args.gtp_command_timeout_sec,
        )
        if not args.no_performance_monitor:
            monitor = PerformanceMonitor(
                args.output_dir,
                args.performance_sample_interval_sec,
                manager,
                run_id=run_id,
            )
            monitor.start()

        capacities = [
            spec.concurrent_match_set_capacity for spec in specs
        ]
        pending = PendingTaskQueue(
            remaining_tasks,
            player_count=len(specs),
        )
        active_counts = [0 for _ in specs]
        duration_weights = [1.0 for _ in specs]
        duration_observations = [0 for _ in specs]
        retry_counts: Dict[int, int] = {}
        active_sets: Dict[int, InFlightMatchSet] = {}
        future_to_task: Dict[Future, int] = {}
        executor = ThreadPoolExecutor(
            max_workers=args.parallel_match_sets * 2,
            thread_name_prefix="strength-game",
        )
        launch_new_tasks = True
        next_status = (
            (completed_match_sets // args.status_every_match_sets + 1)
            * args.status_every_match_sets
        )

        def admit_tasks() -> None:
            schedule_duration_weights = normalized_duration_weights(
                duration_weights,
                duration_observations,
            )
            while (
                launch_new_tasks
                and len(active_sets) < args.parallel_match_sets
                and pending
            ):
                task = pending.pop_schedulable(
                    active_counts,
                    capacities,
                    schedule_duration_weights,
                )
                if task is None:
                    return
                active_counts[task.p0_idx] += 1
                active_counts[task.p1_idx] += 1
                in_flight = InFlightMatchSet(
                    task=task,
                    started_at=time.monotonic(),
                )
                for p0_is_black in (True, False):
                    future = executor.submit(
                        game_runner.play_single_game,
                        task,
                        p0_is_black,
                    )
                    in_flight.futures[future] = p0_is_black
                    future_to_task[future] = task.task_id
                active_sets[task.task_id] = in_flight

        admit_tasks()
        while active_sets or (pending and launch_new_tasks):
            elapsed = time.time() - start_time
            if (
                launch_new_tasks
                and args.time_limit_sec is not None
                and elapsed >= args.time_limit_sec
            ):
                launch_new_tasks = False
                stop_reason = "time_limit"
                print(
                    "Time limit reached; draining already admitted games.",
                    flush=True,
                )
            if launch_new_tasks and manager.low_memory_event.is_set():
                launch_new_tasks = False
                stop_reason = "low_available_memory"
                print(
                    "Available-memory limit reached; draining admitted games.",
                    flush=True,
                )

            if not active_sets:
                if pending and launch_new_tasks:
                    raise RuntimeError(
                        "pending tasks remain but none can be scheduled"
                    )
                break
            done, _ = wait(
                tuple(future_to_task),
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            finalized_ids: set[int] = set()
            for future in done:
                task_id = future_to_task.pop(future)
                in_flight = active_sets[task_id]
                p0_is_black = in_flight.futures.pop(future)
                try:
                    in_flight.results[p0_is_black] = future.result()
                except BaseException as error:
                    in_flight.errors.append(error)
                if not in_flight.futures:
                    finalized_ids.add(task_id)

            for task_id in finalized_ids:
                in_flight = active_sets.pop(task_id)
                task = in_flight.task
                active_counts[task.p0_idx] -= 1
                active_counts[task.p1_idx] -= 1
                duration = max(
                    0.001,
                    time.monotonic() - in_flight.started_at,
                )
                for player_idx in (task.p0_idx, task.p1_idx):
                    if duration_observations[player_idx] == 0:
                        duration_weights[player_idx] = duration
                    else:
                        duration_weights[player_idx] = (
                            0.8 * duration_weights[player_idx]
                            + 0.2 * duration
                        )
                    duration_observations[player_idx] += 1

                if in_flight.errors:
                    error = in_flight.errors[0]
                    attempt = retry_counts.get(task.task_id, 0) + 1
                    store.append_failure(
                        task,
                        attempt,
                        error,
                        "".join(
                            traceback.format_exception(
                                type(error),
                                error,
                                error.__traceback__,
                            )
                        ),
                    )
                    if (
                        isinstance(error, AvailableMemoryLimitError)
                        or "available memory is too low" in str(error).lower()
                    ):
                        manager.low_memory_event.set()
                        launch_new_tasks = False
                        stop_reason = "low_available_memory"
                    elif attempt <= args.task_retries:
                        retry_counts[task.task_id] = attempt
                        pending.push_front(task)
                        print(
                            f"retry_match_set {task.task_id} "
                            f"{attempt}/{args.task_retries}",
                            flush=True,
                        )
                    else:
                        launch_new_tasks = False
                        stop_reason = "task_failed"
                        fatal_error = error
                    continue

                result = combine_color_games(
                    task,
                    list(in_flight.results.values()),
                )
                # Commit order is deliberate: durable log first, aggregate
                # second. Resume can always reconstruct exactly this state.
                store.append_result(task, result)
                stats.record(result)
                completed_match_sets += 1

                if (
                    completed_match_sets >= next_status
                    or completed_match_sets == len(tasks)
                ):
                    store.checkpoint()
                    print_status(
                        specs,
                        stats,
                        start_time,
                        completed_match_sets,
                        completed_at_start,
                        len(tasks),
                        target_matrix,
                    )
                    write_outputs(
                        specs,
                        stats,
                        args.output_dir,
                        completed_match_sets,
                        len(tasks),
                        target_matrix,
                        experiment_id,
                    )
                    last_reported_match_sets = completed_match_sets
                    while next_status <= completed_match_sets:
                        next_status += args.status_every_match_sets

            if fatal_error is not None:
                launch_new_tasks = False
            admit_tasks()

        if completed_match_sets == len(tasks):
            stop_reason = "finished"
        elif stop_reason == "finished":
            stop_reason = "interrupted"
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        manager.close_all()
    finally:
        # Closing the process trees first also releases workers promptly when
        # an unexpected coordinator exception occurs.
        manager.close_all()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if monitor is not None:
            monitor.stop()
        store.checkpoint()
        store.close()

    final_specs = build_player_specs(
        args,
        baseline_levels,
        alphas,
        policy_server_port=(
            policy_server.port
            if "policy_server" in locals()
            else 0
        ),
    )
    if last_reported_match_sets != completed_match_sets:
        print_status(
            final_specs,
            stats,
            start_time,
            completed_match_sets,
            completed_at_start,
            len(tasks),
            target_matrix,
        )
        write_outputs(
            final_specs,
            stats,
            args.output_dir,
            completed_match_sets,
            len(tasks),
            target_matrix,
            experiment_id,
        )
    atomic_write_json(
        args.output_dir / "strength_progress.json",
        {
            "schema_version": 3,
            "experiment_id": experiment_id,
            "completed_match_sets": completed_match_sets,
            "total_match_sets": len(tasks),
            "completed_actual_games": completed_match_sets * 2,
            "total_actual_games": len(tasks) * 2,
            "remaining_match_sets": len(tasks) - completed_match_sets,
            "remaining_actual_games": (
                len(tasks) - completed_match_sets
            )
            * 2,
            "stop_reason": stop_reason,
            "run_id": run_id,
            "policy_server_runtime": (
                policy_server.runtime
                if "policy_server" in locals()
                else "not-started"
            ),
            "elapsed_sec_this_run": time.time() - start_time,
        },
    )
    if fatal_error is not None:
        raise RuntimeError(
            "a paired match set exhausted all retries"
        ) from fatal_error


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = make_argparser().parse_args(argv)
    args.output_dir = Path(args.output_dir)
    with OutputRunLock(args.output_dir):
        run_tournament(args)


if __name__ == "__main__":
    main()
