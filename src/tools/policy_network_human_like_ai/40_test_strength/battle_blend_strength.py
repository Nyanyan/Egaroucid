#!/usr/bin/env python3
"""
Round-robin strength test for Egaroucid 7.8.1 levels and blended policy engines.

Default settings follow the research request:
  - uniformly random legal-move player
  - Egaroucid levels: 1, 3, 5, ..., 19
  - alpha: 0.0, 0.2, ..., 1.0
  - XOT color-swapped match sets per pair: 100
  - parallel matches: 16
  - at most two engine processes per player

Each task plays two color-swapped games from the same XOT opening. The requested
games-per-pair value is counted as paired match sets. One paired match set
contains two actual games and is scored by the average disc difference.

For alpha-series players, generated moves after the XOT opening are also
measured against alpha=0.0. The per-move loss is the best legal Egaroucid level
21 score minus the level 21 score of the selected move.
"""

from __future__ import annotations

import argparse
import atexit
from collections import deque
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
HUMAN_LIKE_DIR = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[3]
BIN_DIR = REPO_ROOT / "bin"
BLEND_DIR = HUMAN_LIKE_DIR / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))
sys.path.insert(0, str(BIN_DIR))

from blend_policy_with_egaroucid import BLACK, WHITE, BoardState, coord_to_policy, default_egaroucid_exe, default_weights_file, policy_to_coord, side_to_gtp_color  # noqa: E402
from elo_rating_backcal import fit_elo_from_winrates_with_interval  # noqa: E402


QUIT_TIMEOUT_SEC = 2.0
KILL_TIMEOUT_SEC = 5.0
PROCESS_POOL_GET_TIMEOUT_SEC = 0.2

process_registry = set()
process_registry_lock = threading.Lock()
shutdown_event = threading.Event()
results_lock = threading.Lock()
low_memory_event = threading.Event()
minimum_available_memory_mib = 0.0


class AvailableMemoryLimitError(RuntimeError):
    pass


def available_memory_mib() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.virtual_memory().available / (1024.0 * 1024.0))


class PerformanceMonitor:
    def __init__(self, output_dir: Path, interval_sec: float, minimum_available_mib: float):
        self.output_dir = output_dir
        self.interval_sec = float(interval_sec)
        self.minimum_available_mib = float(minimum_available_mib)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.samples: List[dict] = []
        self.started_at = 0.0

    def start(self) -> None:
        try:
            import psutil  # noqa: F401
        except ImportError:
            return
        self.started_at = time.time()
        self.thread = threading.Thread(target=self._loop, name="performance-monitor", daemon=True)
        self.thread.start()

    @staticmethod
    def _gpu_sample() -> Tuple[Optional[float], Optional[float]]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5.0,
            )
            if proc.returncode != 0:
                return None, None
            first_line = proc.stdout.strip().splitlines()[0]
            utilization, memory_mib = [float(part.strip()) for part in first_line.split(",")[:2]]
            return utilization, memory_mib
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None, None

    def _loop(self) -> None:
        import psutil

        psutil.cpu_percent(interval=None)
        sample_idx = 0
        last_gpu = (None, None)
        while not self.stop_event.wait(self.interval_sec):
            if sample_idx % max(1, round(5.0 / self.interval_sec)) == 0:
                last_gpu = self._gpu_sample()
            memory = psutil.virtual_memory()
            try:
                children = psutil.Process(os.getpid()).children(recursive=True)
            except psutil.Error:
                children = []
            self.samples.append(
                {
                    "elapsed_sec": time.time() - self.started_at,
                    "cpu_percent": float(psutil.cpu_percent(interval=None)),
                    "system_memory_used_mib": float(memory.used / (1024.0 * 1024.0)),
                    "available_memory_mib": float(memory.available / (1024.0 * 1024.0)),
                    "system_memory_percent": float(memory.percent),
                    "gpu_percent": last_gpu[0],
                    "gpu_memory_used_mib": last_gpu[1],
                    "child_processes": len(children),
                }
            )
            sample = self.samples[-1]
            if sample["available_memory_mib"] < self.minimum_available_mib:
                low_memory_event.set()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "performance_live.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        **sample,
                        "minimum_available_memory_mib": self.minimum_available_mib,
                        "low_memory_limit_reached": low_memory_event.is_set(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            sample_idx += 1

    @staticmethod
    def _average(samples: Sequence[dict], key: str) -> Optional[float]:
        values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _maximum(samples: Sequence[dict], key: str) -> Optional[float]:
        values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
        return max(values) if values else None

    @staticmethod
    def _minimum(samples: Sequence[dict], key: str) -> Optional[float]:
        values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
        return min(values) if values else None

    def stop(self) -> None:
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=max(10.0, self.interval_sec * 2.0))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fields = [
            "elapsed_sec",
            "cpu_percent",
            "system_memory_used_mib",
            "available_memory_mib",
            "system_memory_percent",
            "gpu_percent",
            "gpu_memory_used_mib",
            "child_processes",
        ]
        with (self.output_dir / "performance_samples.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)
        summary = {
            "sample_interval_sec": self.interval_sec,
            "samples": len(self.samples),
            "average_cpu_percent": self._average(self.samples, "cpu_percent"),
            "maximum_cpu_percent": self._maximum(self.samples, "cpu_percent"),
            "average_system_memory_used_mib": self._average(self.samples, "system_memory_used_mib"),
            "maximum_system_memory_used_mib": self._maximum(self.samples, "system_memory_used_mib"),
            "minimum_available_memory_mib": self._minimum(self.samples, "available_memory_mib"),
            "configured_minimum_available_memory_mib": self.minimum_available_mib,
            "low_memory_limit_reached": low_memory_event.is_set(),
            "average_gpu_percent": self._average(self.samples, "gpu_percent"),
            "maximum_gpu_percent": self._maximum(self.samples, "gpu_percent"),
            "maximum_gpu_memory_used_mib": self._maximum(self.samples, "gpu_memory_used_mib"),
            "maximum_child_processes": self._maximum(self.samples, "child_processes"),
        }
        with (self.output_dir / "performance_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class Task:
    task_id: int
    p0_idx: int
    p1_idx: int
    opening: str
    actual_games: int


def parse_int_list(text: str) -> List[int]:
    if text.strip() == "":
        return []
    return [int(token) for token in text.split(",") if token.strip()]


def parse_float_list(text: str) -> List[float]:
    if text.strip() == "":
        return []
    return [float(token) for token in text.split(",") if token.strip()]


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def display_command(command: Sequence[str]) -> List[str]:
    result = []
    for part in command:
        try:
            if Path(part).resolve() == Path(sys.executable).resolve():
                result.append("python")
                continue
        except (OSError, RuntimeError):
            pass
        path = Path(part)
        if path.is_absolute() or path.exists():
            result.append(repo_relative(path))
        else:
            result.append(part)
    return result


def default_output_dir() -> Path:
    return SCRIPT_DIR / "output" / time.strftime("strength_%Y%m%d_%H%M%S")


def display_path(path: Path) -> str:
    return repo_relative(Path(path))


class Player:
    def __init__(
        self,
        name: str,
        command: List[str],
        processes_per_player: int,
        close_after_game: bool,
        setboard_before_genmove: bool,
        alpha: Optional[float] = None,
        random_seed: Optional[int] = None,
        reports_move_stone_loss: bool = False,
    ):
        self.name = name
        self.command = command
        self.processes: List[Optional[subprocess.Popen]] = []
        self.proc_pool = [queue.Queue(), queue.Queue()]
        self.proc_color_started = [0, 0]
        self.lock = threading.Lock()
        self.results: List[List[int]] = []
        self.disc_diff: List[float] = []
        self.n_played: List[int] = []
        self.move_stone_loss: List[float] = []
        self.move_stone_loss_count: List[int] = []
        self.processes_per_player = processes_per_player
        self.close_after_game = close_after_game
        self.setboard_before_genmove = setboard_before_genmove
        self.alpha = alpha
        self.random_seed = random_seed
        self.reports_move_stone_loss = reports_move_stone_loss

    def start_processes(self) -> None:
        # Processes are started lazily by acquire_process_idx(). Prestarting one
        # process per player slot would be too memory-heavy for 32 parallel
        # matches with blended engines.
        return

    def try_start_process_for_color(self, color_pool: int) -> Optional[int]:
        if self.random_seed is not None:
            raise RuntimeError("the internal random player does not use an engine process")
        half = self.processes_per_player // 2
        with self.lock:
            if self.proc_color_started[color_pool] >= half:
                return None
            proc_idx = len(self.processes)
            self.processes.append(start_engine(self.command))
            self.proc_color_started[color_pool] += 1
            return proc_idx


def format_elapsed(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds %= 60
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def start_engine(command: List[str], env: Optional[dict] = None) -> subprocess.Popen:
    popen_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "text": True,
        "bufsize": 1,
    }
    if env is not None:
        popen_kwargs["env"] = env
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    with process_registry_lock:
        if shutdown_event.is_set():
            raise RuntimeError("shutdown in progress")
        available_mib = available_memory_mib()
        if available_mib is not None and available_mib < minimum_available_memory_mib:
            low_memory_event.set()
            raise AvailableMemoryLimitError(
                f"available memory {available_mib:.0f} MiB is below "
                f"the configured minimum {minimum_available_memory_mib:.0f} MiB"
            )
        proc = subprocess.Popen(command, **popen_kwargs)
        process_registry.add(proc)
        return proc


def needs_policy_batch_server(args: argparse.Namespace) -> bool:
    if args.no_policy_batch_server:
        return False
    return any(alpha > 1.0e-12 for alpha in parse_float_list(args.blend_params))


def start_policy_batch_server(args: argparse.Namespace) -> Tuple[subprocess.Popen, int, str, str]:
    server_script = (HUMAN_LIKE_DIR / "30_blend_with_egaroucid" / "policy_batch_server.py").resolve()
    model_path = args.policy_model
    if model_path is None:
        model_path = Path(args.weights).with_name("selected_model.h5")
    stats_path = args.output_dir / "policy_batch_server_stats.json"
    command = [
        sys.executable,
        str(server_script),
        "--weights",
        str(Path(args.weights).resolve()),
        "--model",
        str(Path(model_path).resolve()),
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
        str(stats_path.resolve()),
    ]
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["OPENBLAS_NUM_THREADS"] = str(args.policy_inference_threads)
    env["OMP_NUM_THREADS"] = str(args.policy_inference_threads)
    proc = start_engine(command, env=env)
    ready_queue: queue.Queue[str] = queue.Queue()

    def read_ready() -> None:
        try:
            ready_queue.put(proc.stdout.readline())
        except BaseException:
            ready_queue.put("")

    threading.Thread(target=read_ready, daemon=True).start()
    try:
        line = ready_queue.get(timeout=args.policy_server_startup_timeout_sec).strip()
    except queue.Empty:
        close_process(proc, send_quit=False)
        raise TimeoutError("policy batch server startup timed out")
    parts = line.split()
    if len(parts) != 4 or parts[0] != "READY":
        close_process(proc, send_quit=False)
        raise RuntimeError(f"policy batch server failed to start: {line!r}")
    return proc, int(parts[1]), parts[2], parts[3]


def unregister_process(proc: subprocess.Popen) -> None:
    with process_registry_lock:
        process_registry.discard(proc)


def kill_process_tree(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=KILL_TIMEOUT_SEC)
            if proc.poll() is not None:
                return
        except Exception:
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def close_process(proc: subprocess.Popen, send_quit: bool = True) -> None:
    try:
        if send_quit and proc.poll() is None and proc.stdin is not None:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
        if send_quit and proc.poll() is None:
            try:
                proc.wait(timeout=QUIT_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                kill_process_tree(proc)
        elif not send_quit:
            kill_process_tree(proc)
    finally:
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
        unregister_process(proc)


def shutdown_all_processes() -> None:
    shutdown_event.set()
    with process_registry_lock:
        procs = list(process_registry)
    for proc in procs:
        close_process(proc)


atexit.register(shutdown_all_processes)


def restart_process(players: List[Player], player_idx: int, proc_idx: int) -> subprocess.Popen:
    old_proc = players[player_idx].processes[proc_idx]
    if old_proc is not None:
        close_process(old_proc, send_quit=False)
    new_proc = start_engine(players[player_idx].command)
    players[player_idx].processes[proc_idx] = new_proc
    print("restart", players[player_idx].name, "proc", proc_idx, flush=True)
    return new_proc


def send_command(players: List[Player], player_idx: int, proc_idx: int, command: str) -> str:
    proc = players[player_idx].processes[proc_idx]
    if proc is None:
        proc = restart_process(players, player_idx, proc_idx)
    for _ in range(2):
        if proc.poll() is not None:
            proc = restart_process(players, player_idx, proc_idx)
        try:
            proc.stdin.write(command)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc = restart_process(players, player_idx, proc_idx)
            continue
        while True:
            raw = proc.stdout.readline()
            if raw == "":
                proc = restart_process(players, player_idx, proc_idx)
                break
            line = raw.replace("\r", "").replace("\n", "")
            if line:
                return line
    raise RuntimeError(f"failed to communicate with {players[player_idx].name}: {command.strip()}")


def acquire_process_idx(player: Player, color_pool: int) -> int:
    while True:
        if shutdown_event.is_set():
            raise RuntimeError("shutdown in progress")
        try:
            return player.proc_pool[color_pool].get_nowait()
        except queue.Empty:
            proc_idx = player.try_start_process_for_color(color_pool)
            if proc_idx is not None:
                return proc_idx
        try:
            return player.proc_pool[color_pool].get(timeout=PROCESS_POOL_GET_TIMEOUT_SEC)
        except queue.Empty:
            pass


def release_process_idx(player: Player, color_pool: int, proc_idx: int) -> None:
    if not player.close_after_game:
        player.proc_pool[color_pool].put(proc_idx)
        return
    with player.lock:
        proc = player.processes[proc_idx]
        player.processes[proc_idx] = None
        player.proc_color_started[color_pool] = max(0, player.proc_color_started[color_pool] - 1)
    if proc is not None:
        close_process(proc)


def parse_gtp_move(line: str) -> str:
    line = line.strip()
    if line.startswith("?"):
        raise RuntimeError(f"engine error: {line}")
    if line.startswith("="):
        line = line[1:].strip()
    if not line:
        return "pass"
    return line.split()[-1].lower()


def parse_gtp_float(line: str) -> float:
    line = line.strip()
    if line.startswith("?"):
        raise RuntimeError(f"engine error: {line}")
    if line.startswith("="):
        line = line[1:].strip()
    return float(line)


def count_bits(x: int) -> int:
    return bin(int(x)).count("1")


def disc_diff_for_p0(state: BoardState, p0_is_black: bool) -> int:
    black_count = count_bits(state.black)
    white_count = count_bits(state.white)
    empty = 64 - black_count - white_count
    diff_black = black_count - white_count
    if diff_black > 0:
        diff_black += empty
    elif diff_black < 0:
        diff_black -= empty
    return diff_black if p0_is_black else -diff_black


def play_command_to_stateful(
    players: List[Player],
    p0_idx: int,
    p0_proc: Optional[int],
    p1_idx: int,
    p1_proc: Optional[int],
    side: int,
    move: str,
) -> None:
    cmd = f"play {side_to_gtp_color(side)} {move}\n"
    if p0_proc is not None:
        send_command(players, p0_idx, p0_proc, cmd)
    if p1_proc is not None:
        send_command(players, p1_idx, p1_proc, cmd)


def generate_move(
    players: List[Player],
    player_idx: int,
    held_proc: Optional[int],
    color_pool: int,
    state: BoardState,
    side: int,
    random_generator: Optional[random.Random] = None,
) -> Tuple[str, Optional[float]]:
    player = players[player_idx]
    if player.random_seed is not None:
        if random_generator is None:
            raise RuntimeError("the internal random player requires a game-local random generator")
        legal = state.legal_policies(side)
        if not legal:
            return "pass", None
        policy = random_generator.choice(legal)
        return policy_to_coord(policy), None

    proc_idx = held_proc
    borrowed_for_move = False
    if player.setboard_before_genmove:
        proc_idx = acquire_process_idx(player, color_pool)
        borrowed_for_move = True
    if proc_idx is None:
        raise RuntimeError(f"no process available for {player.name}")
    try:
        if player.setboard_before_genmove:
            send_command(players, player_idx, proc_idx, f"setboard {state.to_egaroucid_board(side)}\n")
        line = send_command(players, player_idx, proc_idx, f"genmove {side_to_gtp_color(side)}\n")
        move = parse_gtp_move(line)
        move_stone_loss = None
        if move != "pass":
            if player.reports_move_stone_loss:
                loss_line = send_command(players, player_idx, proc_idx, "last_move_stone_loss\n")
                move_stone_loss = parse_gtp_float(loss_line)
            elif player.alpha is not None and abs(player.alpha) < 1.0e-12:
                move_stone_loss = 0.0
        return move, move_stone_loss
    finally:
        if borrowed_for_move:
            release_process_idx(player, color_pool, proc_idx)


def play_single_game(
    players: List[Player],
    p0_idx: int,
    p1_idx: int,
    opening: str,
    p0_is_black: bool,
    task_id: int,
) -> dict:
    black_idx = p0_idx if p0_is_black else p1_idx
    white_idx = p1_idx if p0_is_black else p0_idx
    p0_color_pool = 0 if p0_is_black else 1
    p1_color_pool = 1 if p0_is_black else 0
    p0_proc = None
    p1_proc = None
    state = BoardState.initial()
    transcript = ""
    alpha_move_stone_loss = {}
    random_generators = {
        player_idx: random.Random(f"{players[player_idx].random_seed}:{task_id}:{int(p0_is_black)}:{player_idx}")
        for player_idx in (p0_idx, p1_idx)
        if players[player_idx].random_seed is not None
    }

    def record_move_stone_loss(player_idx: int, loss: Optional[float]) -> None:
        if players[player_idx].alpha is None:
            return
        if loss is None:
            raise RuntimeError(f"missing move stone loss for {players[player_idx].name}")
        row = alpha_move_stone_loss.setdefault(
            str(player_idx),
            {"moves": 0, "total_stone_loss": 0.0},
        )
        row["moves"] += 1
        row["total_stone_loss"] += float(loss)

    try:
        process_requests = []
        if not players[p0_idx].setboard_before_genmove and players[p0_idx].random_seed is None:
            process_requests.append((p0_idx, p0_color_pool, 0))
        if not players[p1_idx].setboard_before_genmove and players[p1_idx].random_seed is None:
            process_requests.append((p1_idx, p1_color_pool, 1))
        for player_idx, color_pool, owner in sorted(process_requests):
            proc_idx = acquire_process_idx(players[player_idx], color_pool)
            if owner == 0:
                p0_proc = proc_idx
            else:
                p1_proc = proc_idx
            send_command(players, player_idx, proc_idx, "clear_board\n")

        for i in range(0, len(opening), 2):
            if not state.legal_policies(state.side):
                play_command_to_stateful(players, p0_idx, p0_proc, p1_idx, p1_proc, state.side, "pass")
                state.side ^= 1
                if not state.legal_policies(state.side):
                    break
            move = opening[i : i + 2].lower()
            play_command_to_stateful(players, p0_idx, p0_proc, p1_idx, p1_proc, state.side, move)
            state.apply_move(state.side, coord_to_policy(move))
            transcript += move

        while True:
            if not state.legal_policies(state.side):
                play_command_to_stateful(players, p0_idx, p0_proc, p1_idx, p1_proc, state.side, "pass")
                state.side ^= 1
                if not state.legal_policies(state.side):
                    break

            mover = black_idx if state.side == BLACK else white_idx
            watcher = white_idx if state.side == BLACK else black_idx
            mover_proc = p0_proc if mover == p0_idx else p1_proc
            watcher_proc = p0_proc if watcher == p0_idx else p1_proc
            mover_color_pool = p0_color_pool if mover == p0_idx else p1_color_pool
            side = state.side
            move, move_stone_loss = generate_move(
                players,
                mover,
                mover_proc,
                mover_color_pool,
                state,
                side,
                random_generator=random_generators.get(mover),
            )
            if move == "pass":
                state.side ^= 1
                if watcher_proc is not None:
                    send_command(players, watcher, watcher_proc, f"play {side_to_gtp_color(side)} pass\n")
                continue
            record_move_stone_loss(mover, move_stone_loss)
            state.apply_move(side, coord_to_policy(move))
            transcript += move
            if watcher_proc is not None:
                send_command(players, watcher, watcher_proc, f"play {side_to_gtp_color(side)} {move}\n")

        diff = disc_diff_for_p0(state, p0_is_black)
        return {
            "p0_idx": p0_idx,
            "p1_idx": p1_idx,
            "black_idx": black_idx,
            "white_idx": white_idx,
            "p0_disc_diff": diff,
            "black_stones": count_bits(state.black),
            "white_stones": count_bits(state.white),
            "transcript": transcript,
            "alpha_move_stone_loss": alpha_move_stone_loss,
        }
    finally:
        if p0_proc is not None:
            release_process_idx(players[p0_idx], p0_color_pool, p0_proc)
        if p1_proc is not None:
            release_process_idx(players[p1_idx], p1_color_pool, p1_proc)


def update_result(players: List[Player], result: dict) -> None:
    p0_idx = result["p0_idx"]
    p1_idx = result["p1_idx"]
    diff = result["p0_disc_diff"]
    with results_lock:
        if diff > 0:
            players[p0_idx].results[p1_idx][0] += 1
            players[p1_idx].results[p0_idx][2] += 1
        elif diff < 0:
            players[p1_idx].results[p0_idx][0] += 1
            players[p0_idx].results[p1_idx][2] += 1
        else:
            players[p0_idx].results[p1_idx][1] += 1
            players[p1_idx].results[p0_idx][1] += 1
        players[p0_idx].disc_diff[p1_idx] += diff
        players[p1_idx].disc_diff[p0_idx] -= diff
        players[p0_idx].n_played[p1_idx] += 1
        players[p1_idx].n_played[p0_idx] += 1
        for game in result.get("color_games", []):
            for player_idx_text, metric in game.get("alpha_move_stone_loss", {}).items():
                player_idx = int(player_idx_text)
                opponent_idx = p1_idx if player_idx == p0_idx else p0_idx
                players[player_idx].move_stone_loss[opponent_idx] += float(metric["total_stone_loss"])
                players[player_idx].move_stone_loss_count[opponent_idx] += int(metric["moves"])


def play_task(players: List[Player], task: Task) -> List[dict]:
    if task.actual_games != 2:
        raise ValueError("each XOT match set must contain exactly two color-swapped games")
    with ThreadPoolExecutor(max_workers=2) as executor:
        black_future = executor.submit(play_single_game, players, task.p0_idx, task.p1_idx, task.opening, True, task.task_id)
        white_future = executor.submit(play_single_game, players, task.p0_idx, task.p1_idx, task.opening, False, task.task_id)
        black_result = black_future.result()
        white_result = white_future.result()
    average_disc_diff = (float(black_result["p0_disc_diff"]) + float(white_result["p0_disc_diff"])) / 2.0
    result = {
        "p0_idx": task.p0_idx,
        "p1_idx": task.p1_idx,
        "opening": task.opening,
        "p0_disc_diff": average_disc_diff,
        "p0_black_disc_diff": black_result["p0_disc_diff"],
        "p0_white_disc_diff": white_result["p0_disc_diff"],
        "actual_games": 2,
        "color_games": [black_result, white_result],
    }
    update_result(players, result)
    return [result]


def estimate_elos(players: List[Player]) -> Dict[str, Tuple[float, float]]:
    names = [player.name for player in players]
    n_players = len(players)
    win_rates = np.full((n_players, n_players), np.nan, dtype=float)
    games = np.zeros((n_players, n_players), dtype=float)
    for i, player in enumerate(players):
        for j in range(n_players):
            if i == j:
                continue
            w, d, l = player.results[j]
            n = player.n_played[j]
            games[i, j] = float(n)
            if n:
                win_rates[i, j] = (w + 0.5 * d) / n
    try:
        ratings, intervals = fit_elo_from_winrates_with_interval(win_rates, games=games, names=names, confidence=0.95)
    except ValueError:
        return {}
    return {name: (float(ratings[name]), float(intervals[name])) for name in names}


def build_matrix_report(players: List[Player], target_per_pair: int) -> str:
    ratings = estimate_elos(players)
    names = [player.name for player in players]
    lines = []

    lines.append("Win Rate")
    lines.append("\t".join(["vs >"] + names + ["all", "e_rate95"]))
    for i, player in enumerate(players):
        cells = [player.name]
        for j in range(len(players)):
            if i == j:
                cells.append("-")
                continue
            w, d, l = player.results[j]
            n = w + d + l
            cells.append(f"{(w + 0.5 * d) / n:.4f}" if n else "0.0000")
        total_w = sum(result[0] for result in player.results)
        total_d = sum(result[1] for result in player.results)
        total_l = sum(result[2] for result in player.results)
        total_n = total_w + total_d + total_l
        cells.append(f"{(total_w + 0.5 * total_d) / total_n:.4f}" if total_n else "0.0000")
        estimated = ratings.get(player.name)
        cells.append("-" if estimated is None else f"{estimated[0]:.1f}+-{estimated[1]:.1f}")
        lines.append("\t".join(cells))

    lines.append("Average Disc Difference")
    lines.append("\t".join(["vs >"] + names + ["all", "e_rate95"]))
    for i, player in enumerate(players):
        cells = [player.name]
        for j in range(len(players)):
            if i == j:
                cells.append("-")
                continue
            n = player.n_played[j]
            average = player.disc_diff[j] / n if n else 0.0
            cells.append(f"{average:+.2f}")
        total_n = sum(player.n_played)
        total_average = sum(player.disc_diff) / total_n if total_n else 0.0
        cells.append(f"{total_average:+.2f}")
        estimated = ratings.get(player.name)
        cells.append("-" if estimated is None else f"{estimated[0]:.1f}+-{estimated[1]:.1f}")
        lines.append("\t".join(cells))

    lines.append("Average Estimated Stone Loss per Move vs alpha=0.0 (Egaroucid level 21)")
    lines.append("\t".join(["vs >"] + names + ["all"]))
    for i, player in enumerate(players):
        cells = [player.name]
        for j in range(len(players)):
            if i == j or player.alpha is None:
                cells.append("-")
                continue
            count = player.move_stone_loss_count[j]
            cells.append(f"{player.move_stone_loss[j] / count:.4f}" if count else "-")
        total_count = sum(player.move_stone_loss_count)
        cells.append(f"{sum(player.move_stone_loss) / total_count:.4f}" if total_count else "-")
        lines.append("\t".join(cells))

    lines.append("Games Progress (played/target)")
    lines.append("\t".join(["vs >"] + names + ["all"]))
    target_per_player = target_per_pair * (len(players) - 1)
    for i, player in enumerate(players):
        cells = [player.name]
        for j in range(len(players)):
            if i == j:
                cells.append("-")
            else:
                cells.append(f"{player.n_played[j]}/{target_per_pair}")
        cells.append(f"{sum(player.n_played)}/{target_per_player}")
        lines.append("\t".join(cells))
    return "\n".join(lines)


def write_matrix_tables(players: List[Player], output_dir: Path, target_per_pair: int) -> None:
    ratings = estimate_elos(players)
    names = [player.name for player in players]
    header = ["player"] + names + ["all", "elo", "elo_ci95"]

    with (output_dir / "strength_win_rate_matrix.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for i, player in enumerate(players):
            row = [player.name]
            for j in range(len(players)):
                if i == j:
                    row.append("-")
                else:
                    w, d, l = player.results[j]
                    n = w + d + l
                    row.append((w + 0.5 * d) / n if n else 0.0)
            total_w = sum(result[0] for result in player.results)
            total_d = sum(result[1] for result in player.results)
            total_l = sum(result[2] for result in player.results)
            total_n = total_w + total_d + total_l
            row.append((total_w + 0.5 * total_d) / total_n if total_n else 0.0)
            elo, ci = ratings.get(player.name, (None, None))
            row.extend([elo, ci])
            writer.writerow(row)

    with (output_dir / "strength_disc_diff_matrix.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for i, player in enumerate(players):
            row = [player.name]
            for j in range(len(players)):
                if i == j:
                    row.append("-")
                else:
                    n = player.n_played[j]
                    row.append(player.disc_diff[j] / n if n else 0.0)
            total_n = sum(player.n_played)
            row.append(sum(player.disc_diff) / total_n if total_n else 0.0)
            elo, ci = ratings.get(player.name, (None, None))
            row.extend([elo, ci])
            writer.writerow(row)

    with (output_dir / "strength_move_stone_loss_matrix.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["player"] + names + ["all"])
        for i, player in enumerate(players):
            row = [player.name]
            for j in range(len(players)):
                if i == j or player.alpha is None:
                    row.append("-")
                    continue
                count = player.move_stone_loss_count[j]
                row.append(player.move_stone_loss[j] / count if count else "-")
            total_count = sum(player.move_stone_loss_count)
            row.append(sum(player.move_stone_loss) / total_count if total_count else "-")
            writer.writerow(row)

    with (output_dir / "strength_progress_matrix.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["player"] + names + ["all"])
        target_per_player = target_per_pair * (len(players) - 1)
        for i, player in enumerate(players):
            row = [player.name]
            for j in range(len(players)):
                row.append("-" if i == j else f"{player.n_played[j]}/{target_per_pair}")
            row.append(f"{sum(player.n_played)}/{target_per_player}")
            writer.writerow(row)

    with (output_dir / "strength_report.txt").open("w", encoding="utf-8") as f:
        f.write(build_matrix_report(players, target_per_pair) + "\n")


def write_outputs(
    players: List[Player],
    output_dir: Path,
    completed_match_sets: int,
    total_match_sets: int,
    target_per_pair: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ratings = estimate_elos(players)
    names = [player.name for player in players]
    summary_rows = []
    for i, player in enumerate(players):
        total_w = total_d = total_l = total_n = 0
        total_disc = 0.0
        for j in range(len(players)):
            w, d, l = player.results[j]
            n = player.n_played[j]
            total_w += w
            total_d += d
            total_l += l
            total_n += n
            total_disc += player.disc_diff[j]
        elo, ci = ratings.get(player.name, (None, None))
        total_move_stone_loss_count = sum(player.move_stone_loss_count)
        summary_rows.append(
            {
                "name": player.name,
                "match_sets": total_n,
                "actual_games": total_n * 2,
                "win": total_w,
                "draw": total_d,
                "loss": total_l,
                "win_rate": (total_w + 0.5 * total_d) / total_n if total_n else 0.0,
                "avg_disc_diff": total_disc / total_n if total_n else 0.0,
                "measured_moves": total_move_stone_loss_count if player.alpha is not None else None,
                "avg_stone_loss_per_move_vs_alpha0": (
                    sum(player.move_stone_loss) / total_move_stone_loss_count
                    if player.alpha is not None and total_move_stone_loss_count
                    else None
                ),
                "elo": elo,
                "elo_ci95": ci,
            }
        )
    with (output_dir / "strength_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["name"])
        writer.writeheader()
        writer.writerows(summary_rows)

    pair_rows = []
    for i, player in enumerate(players):
        for j, opponent in enumerate(players):
            if i == j:
                continue
            w, d, l = player.results[j]
            n = player.n_played[j]
            measured_moves = player.move_stone_loss_count[j]
            pair_rows.append(
                {
                    "player": player.name,
                    "opponent": opponent.name,
                    "match_sets": n,
                    "actual_games": n * 2,
                    "win": w,
                    "draw": d,
                    "loss": l,
                    "win_rate": (w + 0.5 * d) / n if n else 0.0,
                    "avg_disc_diff": player.disc_diff[j] / n if n else 0.0,
                    "measured_moves": measured_moves if player.alpha is not None else None,
                    "avg_stone_loss_per_move_vs_alpha0": (
                        player.move_stone_loss[j] / measured_moves
                        if player.alpha is not None and measured_moves
                        else None
                    ),
                }
            )
    with (output_dir / "strength_pair_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows else ["player"])
        writer.writeheader()
        writer.writerows(pair_rows)
    write_matrix_tables(players, output_dir, target_per_pair)

    data = {
        "completed_match_sets": completed_match_sets,
        "total_match_sets": total_match_sets,
        "completed_actual_games": completed_match_sets * 2,
        "total_actual_games": total_match_sets * 2,
        "target_match_sets_per_pair": target_per_pair,
        "move_stone_loss_metric": {
            "players": "alpha series only",
            "reference": "alpha=0.0 (Egaroucid level 21)",
            "formula": "best legal move score at level 21 minus selected move score at level 21",
            "unit": "estimated final disc difference",
            "xot_opening_moves_included": False,
        },
        "players": [
            {
                "name": player.name,
                "command": display_command(player.command),
            }
            for player in players
        ],
        "summary": summary_rows,
        "pair_results": pair_rows,
    }
    with (output_dir / "strength_results.json").open("w") as f:
        json.dump(data, f, indent=2)


def print_status(
    players: List[Player],
    output_dir: Path,
    start_time: float,
    completed_match_sets: int,
    total_match_sets: int,
    target_per_pair: int,
) -> None:
    elapsed = time.time() - start_time
    pct = 100.0 * completed_match_sets / max(1, total_match_sets)
    speed = completed_match_sets / elapsed if elapsed > 0 else 0.0
    eta = (total_match_sets - completed_match_sets) / speed if speed > 0 else 0.0
    print("\n" + "=" * 80)
    print(f"Progress: {completed_match_sets}/{total_match_sets} match sets ({completed_match_sets * 2}/{total_match_sets * 2} actual games, {pct:.2f}%)")
    print(f"Elapsed: {format_elapsed(elapsed)}  ETA: {format_elapsed(eta)}  Speed: {speed:.2f} match sets/sec")
    print(build_matrix_report(players, target_per_pair))


def read_openings(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        openings = [line.strip() for line in f if line.strip()]
    if not openings:
        raise FileNotFoundError(f"no openings in {path}")
    return openings


def build_players(args: argparse.Namespace) -> List[Player]:
    players: List[Player] = []
    egaroucid_exe = Path(args.egaroucid_exe).resolve()
    for level in parse_int_list(args.baseline_levels):
        players.append(
            Player(
                f"egaroucid_l{level}",
                [
                    str(egaroucid_exe),
                    "-gtp",
                    "-quiet",
                    "-nobook",
                    "-l",
                    str(level),
                    "-t",
                    str(args.engine_threads),
                ],
                args.processes_per_player,
                args.close_processes_after_game,
                False,
            )
        )

    blend_script = (HUMAN_LIKE_DIR / "30_blend_with_egaroucid" / "blend_gtp_engine.py").resolve()
    for alpha in parse_float_list(args.blend_params):
        native_alpha_zero = args.native_alpha_zero and abs(alpha) < 1.0e-12
        if native_alpha_zero:
            command = [
                str(egaroucid_exe),
                "-gtp",
                "-quiet",
                "-nobook",
                "-l",
                str(args.blend_egaroucid_level),
                "-t",
                str(args.engine_threads),
            ]
        else:
            command = [
                sys.executable,
                str(blend_script),
                "--weights",
                str(Path(args.weights).resolve()),
                "--alpha",
                f"{alpha:.1f}",
                "--egaroucid-exe",
                str(egaroucid_exe),
                "--egaroucid-level",
                str(args.blend_egaroucid_level),
                "--egaroucid-threads",
                str(args.engine_threads),
                "--egaroucid-timeout-sec",
                str(args.egaroucid_timeout_sec),
                "--score-temperature",
                str(args.score_temperature),
                "--measure-move-stone-loss",
            ]
            if not args.no_blend_cache:
                command.append("--cache-egaroucid")
            if args.hint_cache_db is not None:
                command.extend(["--hint-cache-db", str(Path(args.hint_cache_db).resolve())])
            if args.policy_server_port is not None:
                command.extend(
                    [
                        "--policy-server-host",
                        args.policy_server_host,
                        "--policy-server-port",
                        str(args.policy_server_port),
                        "--policy-server-timeout-sec",
                        str(args.policy_server_timeout_sec),
                    ]
                )
        players.append(
            Player(
                f"alpha_{alpha:.1f}",
                command,
                args.processes_per_player,
                args.close_processes_after_game,
                not native_alpha_zero,
                alpha=alpha,
                reports_move_stone_loss=not native_alpha_zero,
            )
        )
    if not args.no_random_player:
        players.append(
            Player(
                "random_legal",
                ["internal:uniform_random_legal", "--seed", str(args.random_seed)],
                args.processes_per_player,
                args.close_processes_after_game,
                False,
                random_seed=args.random_seed,
            )
        )
    n = len(players)
    for player in players:
        player.results = [[0, 0, 0] for _ in range(n)]
        player.disc_diff = [0.0 for _ in range(n)]
        player.n_played = [0 for _ in range(n)]
        player.move_stone_loss = [0.0 for _ in range(n)]
        player.move_stone_loss_count = [0 for _ in range(n)]
    return players


def make_tasks(players: List[Player], openings: Sequence[str], games_per_pair: int, seed: int, same_openings_for_all_pairs: bool) -> List[Task]:
    rng = random.Random(seed)
    pairs = [(i, j) for i in range(len(players)) for j in range(i + 1, len(players))]
    raw_tasks = []
    opening_idx = 0
    for set_idx in range(games_per_pair):
        round_pairs = list(pairs)
        rng.shuffle(round_pairs)
        for p0, p1 in round_pairs:
            if same_openings_for_all_pairs:
                opening = openings[set_idx % len(openings)]
            else:
                opening = openings[opening_idx]
                opening_idx = (opening_idx + 1) % len(openings)
            raw_tasks.append((p0, p1, opening, 2))
    return [Task(task_id, p0, p1, opening, actual_games) for task_id, (p0, p1, opening, actual_games) in enumerate(raw_tasks)]


def limit_tasks(tasks: Sequence[Task], max_match_sets: Optional[int]) -> List[Task]:
    if max_match_sets is None:
        return list(tasks)
    if max_match_sets <= 0:
        raise ValueError("--max-match-sets must be positive when set")
    return list(tasks[:max_match_sets])


def game_results_path(output_dir: Path) -> Path:
    return output_dir / "strength_games.jsonl"


def append_task_results(output_dir: Path, task: Task, results: Sequence[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "task_id": task.task_id,
        "p0_idx": task.p0_idx,
        "p1_idx": task.p1_idx,
        "opening": task.opening,
        "actual_games": task.actual_games,
        "results": list(results),
    }
    with game_results_path(output_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def failed_tasks_path(output_dir: Path) -> Path:
    return output_dir / "strength_failed_tasks.jsonl"


def append_task_failure(output_dir: Path, task: Task, attempt: int, exc: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "task_id": task.task_id,
        "p0_idx": task.p0_idx,
        "p1_idx": task.p1_idx,
        "opening": task.opening,
        "actual_games": task.actual_games,
        "attempt": attempt,
        "error": repr(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    with failed_tasks_path(output_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_task_results(players: List[Player], output_dir: Path) -> Tuple[set, int]:
    path = game_results_path(output_dir)
    if not path.exists():
        return set(), 0
    completed_task_ids = set()
    completed_match_sets = 0
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = int(row["task_id"])
            if task_id in completed_task_ids:
                continue
            completed_task_ids.add(task_id)
            results = row.get("results", [])
            for result in results:
                update_result(players, result)
            completed_match_sets += len(results)
    return completed_task_ids, completed_match_sets


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run round-robin strength tests for blended policy engines.")
    parser.add_argument("--no-random-player", action="store_true", help="Exclude the uniformly random legal-move player.")
    parser.add_argument("--random-seed", type=int, default=613, help="Seed for the uniformly random legal-move player.")
    parser.add_argument("--baseline-levels", default="1,3,5,7,9,11,13,15,17,19")
    parser.add_argument("--blend-params", "--alphas", dest="blend_params", default="0.0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--games-per-pair", type=int, default=100, help="Number of XOT color-swapped match sets per pair. One set contains two actual games.")
    parser.add_argument("--max-match-sets", "--max-games", dest="max_match_sets", type=int, default=None, help="Optional benchmark cap in XOT match sets; default runs the full requested schedule.")
    parser.add_argument("--parallel-matches", type=int, default=16)
    parser.add_argument("--processes-per-player", type=int, default=2)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--status-every-match-sets", "--status-every-games", dest="status_every_match_sets", type=int, default=200)
    parser.add_argument("--time-limit-sec", type=float, default=None, help="Stop launching new tasks after this many seconds.")
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--blend-egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=300.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--openings", type=Path, default=BIN_DIR / "problem" / "xot" / "openingslarge.txt")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--seed", type=int, default=57, help="Seed used to shuffle XOT openings and pair order.")
    parser.add_argument("--no-shuffle-openings", action="store_true", help="Keep the XOT opening file order instead of shuffling it.")
    parser.add_argument("--no-blend-cache", action="store_true", help="Disable per-process Egaroucid hint caching in blended engines.")
    parser.add_argument(
        "--hint-cache-db",
        type=Path,
        default=None,
        help="Shared SQLite cache for Egaroucid hint scores and alpha-series move-loss measurements. Defaults to output_dir/egaroucid_hint_cache.sqlite3.",
    )
    parser.add_argument("--no-policy-batch-server", action="store_true", help="Run one local policy model in each blended GTP process.")
    parser.add_argument("--policy-model", type=Path, default=None, help="Keras model used by the shared policy inference server.")
    parser.add_argument("--policy-backend", choices=("auto", "tensorflow", "numpy"), default="auto")
    parser.add_argument("--policy-server-host", default="127.0.0.1")
    parser.add_argument("--policy-server-port", type=int, default=None, help="Reuse an existing policy server instead of starting one.")
    parser.add_argument("--policy-server-timeout-sec", type=float, default=30.0)
    parser.add_argument("--policy-server-startup-timeout-sec", type=float, default=120.0)
    parser.add_argument("--policy-max-batch-size", type=int, default=128)
    parser.add_argument("--policy-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--policy-inference-threads", type=int, default=4)
    parser.add_argument("--no-performance-monitor", action="store_true")
    parser.add_argument("--performance-sample-interval-sec", type=float, default=2.0)
    parser.add_argument(
        "--minimum-available-memory-mib",
        type=float,
        default=24576.0,
        help="Stop before available system memory falls below this many MiB.",
    )
    parser.add_argument(
        "--estimated-engine-memory-mib",
        type=float,
        default=1400.0,
        help="Conservative per-engine memory estimate used by the startup capacity check.",
    )
    parser.add_argument("--no-native-alpha-zero", dest="native_alpha_zero", action="store_false", help="Use the Python blend engine even for alpha=0.0.")
    parser.set_defaults(native_alpha_zero=True)
    parser.add_argument("--same-openings-for-all-pairs", action="store_true", help="Use the same opening sequence for every pair.")
    parser.add_argument("--close-processes-after-game", action="store_true", help="Close engines after each game instead of keeping them in per-player pools.")
    parser.add_argument("--task-retries", type=int, default=2, help="Retry a failed task this many times before aborting.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    global minimum_available_memory_mib

    args = make_argparser().parse_args()
    if args.processes_per_player < 2 or args.processes_per_player % 2 != 0:
        raise ValueError("--processes-per-player must be an even number >= 2")
    if args.parallel_matches < 1:
        raise ValueError("--parallel-matches must be positive")
    if args.games_per_pair < 1:
        raise ValueError("--games-per-pair must be positive")
    if args.time_limit_sec is not None and args.time_limit_sec <= 0.0:
        raise ValueError("--time-limit-sec must be positive when set")
    if args.task_retries < 0:
        raise ValueError("--task-retries must be non-negative")
    if args.policy_max_batch_size < 1:
        raise ValueError("--policy-max-batch-size must be positive")
    if args.policy_batch_wait_ms < 0.0:
        raise ValueError("--policy-batch-wait-ms must be non-negative")
    if args.policy_inference_threads < 1:
        raise ValueError("--policy-inference-threads must be positive")
    if args.performance_sample_interval_sec <= 0.0:
        raise ValueError("--performance-sample-interval-sec must be positive")
    if args.minimum_available_memory_mib <= 0.0:
        raise ValueError("--minimum-available-memory-mib must be positive")
    if args.estimated_engine_memory_mib <= 0.0:
        raise ValueError("--estimated-engine-memory-mib must be positive")
    minimum_available_memory_mib = args.minimum_available_memory_mib
    low_memory_event.clear()

    if args.hint_cache_db is None:
        args.hint_cache_db = args.output_dir / "egaroucid_hint_cache.sqlite3"

    engine_player_count = len(parse_int_list(args.baseline_levels)) + len(parse_float_list(args.blend_params))
    estimated_max_engine_processes = engine_player_count * args.processes_per_player
    estimated_max_engine_memory_mib = estimated_max_engine_processes * args.estimated_engine_memory_mib
    startup_available_memory_mib = available_memory_mib()
    if startup_available_memory_mib is None:
        raise RuntimeError("psutil is required for the available-memory safety check")
    if (
        estimated_max_engine_memory_mib + args.minimum_available_memory_mib
        > startup_available_memory_mib
    ):
        raise MemoryError(
            "memory capacity check failed: "
            f"up to {estimated_max_engine_processes} engine processes are estimated to use "
            f"{estimated_max_engine_memory_mib:.0f} MiB, while only "
            f"{startup_available_memory_mib:.0f} MiB is available and "
            f"{args.minimum_available_memory_mib:.0f} MiB must remain free; "
            "reduce --processes-per-player"
        )

    openings = read_openings(args.openings)
    if not args.no_shuffle_openings:
        random.Random(args.seed).shuffle(openings)
    policy_server_proc = None
    policy_server_runtime = "-"
    if needs_policy_batch_server(args):
        if args.policy_server_port is None:
            if args.dry_run:
                args.policy_server_port = 0
                policy_server_runtime = "auto"
            else:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                policy_server_proc, args.policy_server_port, backend, device = start_policy_batch_server(args)
                policy_server_runtime = f"{backend}/{device}"
        else:
            policy_server_runtime = "external"
    players = build_players(args)
    if len(players) < 2:
        raise ValueError("at least two players are required")
    full_tasks = make_tasks(players, openings, args.games_per_pair, args.seed, args.same_openings_for_all_pairs)
    tasks = limit_tasks(full_tasks, args.max_match_sets)
    total_match_sets = len(tasks)
    completed_task_ids = set()
    completed_match_sets = 0
    if args.resume:
        completed_task_ids, completed_match_sets = load_completed_task_results(players, args.output_dir)
        tasks = [task for task in tasks if task.task_id not in completed_task_ids]

    print("players")
    for player in players:
        print(player.name, " ".join(display_command(player.command)))
    print("match_sets_per_pair", args.games_per_pair)
    print("actual_games_per_pair", args.games_per_pair * 2)
    if args.max_match_sets is not None:
        print("max_match_sets", args.max_match_sets)
    print("parallel_matches", args.parallel_matches)
    print("max_processes_per_player", args.processes_per_player)
    print("max_concurrent_match_sets_per_engine_player", args.processes_per_player // 2)
    print("xot_openings", display_path(args.openings))
    print("shuffle_xot_openings", not args.no_shuffle_openings)
    print("seed", args.seed)
    print("random_legal_player", not args.no_random_player)
    print("random_legal_seed", args.random_seed if not args.no_random_player else "-")
    print("blend_cache_egaroucid", not args.no_blend_cache)
    print("shared_hint_cache_db", display_path(args.hint_cache_db) if args.hint_cache_db is not None else "-")
    print("native_alpha_zero", args.native_alpha_zero)
    print("policy_batch_server", needs_policy_batch_server(args))
    print("policy_batch_server_runtime", policy_server_runtime)
    print("policy_batch_server_endpoint", f"{args.policy_server_host}:{args.policy_server_port}" if args.policy_server_port is not None else "-")
    print("policy_max_batch_size", args.policy_max_batch_size)
    print("policy_batch_wait_ms", args.policy_batch_wait_ms)
    print("performance_monitor", not args.no_performance_monitor)
    print("performance_sample_interval_sec", args.performance_sample_interval_sec)
    print("startup_available_memory_mib", f"{startup_available_memory_mib:.0f}" if startup_available_memory_mib is not None else "-")
    print("minimum_available_memory_mib", args.minimum_available_memory_mib)
    print("estimated_max_engine_processes", estimated_max_engine_processes)
    print("estimated_engine_memory_mib", args.estimated_engine_memory_mib)
    print("estimated_max_engine_memory_mib", estimated_max_engine_memory_mib)
    print("same_openings_for_all_pairs", args.same_openings_for_all_pairs)
    print("close_processes_after_game", args.close_processes_after_game)
    print("task_retries", args.task_retries)
    if args.time_limit_sec is not None:
        print("time_limit_sec", args.time_limit_sec)
    print("total_match_sets", total_match_sets)
    print("total_actual_games", total_match_sets * 2)
    if args.resume:
        print("resume_completed_tasks", len(completed_task_ids))
        print("resume_completed_match_sets", completed_match_sets)
        print("resume_completed_actual_games", completed_match_sets * 2)
        print("remaining_match_sets", len(tasks))
        print("remaining_actual_games", len(tasks) * 2)
    print("output_dir", display_path(args.output_dir))

    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "strength_dry_run.json").open("w") as f:
            json.dump(
                {
                    "players": [{"name": p.name, "command": display_command(p.command)} for p in players],
                    "match_sets_per_pair": args.games_per_pair,
                    "actual_games_per_pair": args.games_per_pair * 2,
                    "max_match_sets": args.max_match_sets,
                    "parallel_matches": args.parallel_matches,
                    "max_processes_per_player": args.processes_per_player,
                    "max_concurrent_match_sets_per_engine_player": args.processes_per_player // 2,
                    "xot_openings": display_path(args.openings),
                    "shuffle_xot_openings": not args.no_shuffle_openings,
                    "seed": args.seed,
                    "random_legal_player": not args.no_random_player,
                    "random_legal_seed": args.random_seed if not args.no_random_player else None,
                    "blend_cache_egaroucid": not args.no_blend_cache,
                    "shared_hint_cache_db": display_path(args.hint_cache_db) if args.hint_cache_db is not None else None,
                    "native_alpha_zero": args.native_alpha_zero,
                    "policy_batch_server": needs_policy_batch_server(args),
                    "policy_batch_server_runtime": policy_server_runtime,
                    "policy_batch_server_endpoint": f"{args.policy_server_host}:{args.policy_server_port}" if args.policy_server_port is not None else None,
                    "policy_max_batch_size": args.policy_max_batch_size,
                    "policy_batch_wait_ms": args.policy_batch_wait_ms,
                    "performance_monitor": not args.no_performance_monitor,
                    "performance_sample_interval_sec": args.performance_sample_interval_sec,
                    "startup_available_memory_mib": startup_available_memory_mib,
                    "minimum_available_memory_mib": args.minimum_available_memory_mib,
                    "estimated_max_engine_processes": estimated_max_engine_processes,
                    "estimated_engine_memory_mib": args.estimated_engine_memory_mib,
                    "estimated_max_engine_memory_mib": estimated_max_engine_memory_mib,
                    "same_openings_for_all_pairs": args.same_openings_for_all_pairs,
                    "close_processes_after_game": args.close_processes_after_game,
                    "task_retries": args.task_retries,
                    "time_limit_sec": args.time_limit_sec,
                    "total_match_sets": total_match_sets,
                    "total_actual_games": total_match_sets * 2,
                    "n_tasks": len(tasks),
                    "output_dir": display_path(args.output_dir),
                    "resume": args.resume,
                    "resume_completed_tasks": len(completed_task_ids),
                    "resume_completed_match_sets": completed_match_sets,
                    "resume_completed_actual_games": completed_match_sets * 2,
                },
                f,
                indent=2,
            )
        return

    for player in players:
        player.start_processes()

    start_time = time.time()
    performance_monitor = None
    if not args.no_performance_monitor:
        performance_monitor = PerformanceMonitor(
            args.output_dir,
            args.performance_sample_interval_sec,
            args.minimum_available_memory_mib,
        )
        performance_monitor.start()
    stop_reason = "finished"
    executor = ThreadPoolExecutor(max_workers=args.parallel_matches)
    retry_counts: Dict[int, int] = {}
    pending_tasks = deque(tasks)
    active_task_counts = [0] * len(players)
    task_capacities = [
        args.parallel_matches if player.random_seed is not None else args.processes_per_player // 2
        for player in players
    ]
    futures = {}

    def submit_available_tasks() -> None:
        scan_count = len(pending_tasks)
        for _ in range(scan_count):
            if len(futures) >= args.parallel_matches:
                return
            task = pending_tasks.popleft()
            if (
                active_task_counts[task.p0_idx] >= task_capacities[task.p0_idx]
                or active_task_counts[task.p1_idx] >= task_capacities[task.p1_idx]
            ):
                pending_tasks.append(task)
                continue
            active_task_counts[task.p0_idx] += 1
            active_task_counts[task.p1_idx] += 1
            futures[executor.submit(play_task, players, task)] = task

    try:
        submit_available_tasks()
        while futures or pending_tasks:
            if low_memory_event.is_set():
                stop_reason = "low_available_memory"
                print("Stopping: available memory reached the configured lower limit.", flush=True)
                break
            if not futures:
                raise RuntimeError("no schedulable task remains despite pending work")
            done, _ = wait(tuple(futures), timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                task = futures.pop(future)
                active_task_counts[task.p0_idx] -= 1
                active_task_counts[task.p1_idx] -= 1
                try:
                    results = future.result()
                except AvailableMemoryLimitError as exc:
                    stop_reason = "low_available_memory"
                    print(f"Stopping: {exc}", flush=True)
                    break
                except Exception as exc:
                    attempt = retry_counts.get(task.task_id, 0) + 1
                    append_task_failure(args.output_dir, task, attempt, exc)
                    if attempt <= args.task_retries:
                        retry_counts[task.task_id] = attempt
                        print(f"retry_task {task.task_id} attempt {attempt}/{args.task_retries}", flush=True)
                        pending_tasks.appendleft(task)
                    else:
                        write_outputs(players, args.output_dir, completed_match_sets, total_match_sets, args.games_per_pair)
                        raise
                    break
                append_task_results(args.output_dir, task, results)
                completed_match_sets += len(results)
                if completed_match_sets % args.status_every_match_sets < len(results) or completed_match_sets == total_match_sets:
                    print_status(players, args.output_dir, start_time, completed_match_sets, total_match_sets, args.games_per_pair)
                    write_outputs(players, args.output_dir, completed_match_sets, total_match_sets, args.games_per_pair)
                if args.time_limit_sec is not None and time.time() - start_time >= args.time_limit_sec:
                    stop_reason = "time_limit"
                break
            if stop_reason != "finished":
                break
            submit_available_tasks()
    finally:
        shutdown_all_processes()
        executor.shutdown(wait=True, cancel_futures=True)
        if performance_monitor is not None:
            performance_monitor.stop()
    if completed_match_sets >= total_match_sets:
        stop_reason = "finished"
        print("\nAll games finished.")
    else:
        print(f"\nStopped before full schedule: {stop_reason}.")
    print_status(players, args.output_dir, start_time, completed_match_sets, total_match_sets, args.games_per_pair)
    write_outputs(players, args.output_dir, completed_match_sets, total_match_sets, args.games_per_pair)
    with (args.output_dir / "strength_progress.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_match_sets": completed_match_sets,
                "total_match_sets": total_match_sets,
                "completed_actual_games": completed_match_sets * 2,
                "total_actual_games": total_match_sets * 2,
                "stop_reason": stop_reason,
                "elapsed_sec": time.time() - start_time,
                "remaining_match_sets": max(0, total_match_sets - completed_match_sets),
                "remaining_actual_games": max(0, total_match_sets - completed_match_sets) * 2,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
