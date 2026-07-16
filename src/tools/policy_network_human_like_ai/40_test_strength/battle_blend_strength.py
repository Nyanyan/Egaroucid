#!/usr/bin/env python3
"""
Round-robin strength test for Egaroucid 7.8.1 levels and blended policy engines.

Default settings follow the research request:
  - Egaroucid levels: 1, 5, 10, 15, 21
  - blend_param: 0.0, 0.1, ..., 1.0
  - games per pair: 1000
  - parallel matches: 32

Each task plays two color-swapped games from the same opening. The requested
games-per-pair value is counted as actual games, not paired tasks.
"""

from __future__ import annotations

import argparse
import atexit
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        path = Path(part)
        if path.is_absolute() or path.exists():
            result.append(repo_relative(path))
        else:
            result.append(part)
    return result


def default_output_dir() -> Path:
    return SCRIPT_DIR / "output" / time.strftime("strength_%Y%m%d_%H%M%S")


class Player:
    def __init__(self, name: str, command: List[str], processes_per_player: int):
        self.name = name
        self.command = command
        self.processes: List[subprocess.Popen] = []
        self.proc_pool = [queue.Queue(), queue.Queue()]
        self.results: List[List[int]] = []
        self.disc_diff: List[float] = []
        self.n_played: List[int] = []
        self.processes_per_player = processes_per_player

    def start_processes(self) -> None:
        half = self.processes_per_player // 2
        self.processes = [start_engine(self.command) for _ in range(self.processes_per_player)]
        for i in range(half):
            self.proc_pool[0].put(i)
            self.proc_pool[1].put(i + half)


def format_elapsed(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds %= 60
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def start_engine(command: List[str]) -> subprocess.Popen:
    popen_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    with process_registry_lock:
        if shutdown_event.is_set():
            raise RuntimeError("shutdown in progress")
        proc = subprocess.Popen(command, **popen_kwargs)
        process_registry.add(proc)
        return proc


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
    close_process(old_proc, send_quit=False)
    new_proc = start_engine(players[player_idx].command)
    players[player_idx].processes[proc_idx] = new_proc
    print("restart", players[player_idx].name, "proc", proc_idx, flush=True)
    return new_proc


def send_command(players: List[Player], player_idx: int, proc_idx: int, command: str) -> str:
    proc = players[player_idx].processes[proc_idx]
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
            return player.proc_pool[color_pool].get(timeout=PROCESS_POOL_GET_TIMEOUT_SEC)
        except queue.Empty:
            pass


def parse_gtp_move(line: str) -> str:
    line = line.strip()
    if line.startswith("?"):
        raise RuntimeError(f"engine error: {line}")
    if line.startswith("="):
        line = line[1:].strip()
    if not line:
        return "pass"
    return line.split()[-1].lower()


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


def play_command_to_all(players: List[Player], p0_idx: int, p0_proc: int, p1_idx: int, p1_proc: int, side: int, move: str) -> None:
    cmd = f"play {side_to_gtp_color(side)} {move}\n"
    send_command(players, p0_idx, p0_proc, cmd)
    send_command(players, p1_idx, p1_proc, cmd)


def play_single_game(players: List[Player], p0_idx: int, p1_idx: int, opening: str, p0_is_black: bool) -> dict:
    black_idx = p0_idx if p0_is_black else p1_idx
    white_idx = p1_idx if p0_is_black else p0_idx
    p0_color_pool = 0 if p0_is_black else 1
    p1_color_pool = 1 if p0_is_black else 0
    p0_proc = None
    p1_proc = None
    state = BoardState.initial()
    record = ""
    try:
        p0_proc = acquire_process_idx(players[p0_idx], p0_color_pool)
        p1_proc = acquire_process_idx(players[p1_idx], p1_color_pool)
        send_command(players, p0_idx, p0_proc, "clear_board\n")
        send_command(players, p1_idx, p1_proc, "clear_board\n")

        for i in range(0, len(opening), 2):
            if not state.legal_policies(state.side):
                play_command_to_all(players, p0_idx, p0_proc, p1_idx, p1_proc, state.side, "pass")
                state.side ^= 1
                if not state.legal_policies(state.side):
                    break
            move = opening[i : i + 2].lower()
            play_command_to_all(players, p0_idx, p0_proc, p1_idx, p1_proc, state.side, move)
            state.apply_move(state.side, coord_to_policy(move))
            record += move

        while True:
            if not state.legal_policies(state.side):
                play_command_to_all(players, p0_idx, p0_proc, p1_idx, p1_proc, state.side, "pass")
                state.side ^= 1
                if not state.legal_policies(state.side):
                    break

            mover = black_idx if state.side == BLACK else white_idx
            watcher = white_idx if state.side == BLACK else black_idx
            mover_proc = p0_proc if mover == p0_idx else p1_proc
            watcher_proc = p0_proc if watcher == p0_idx else p1_proc
            side = state.side
            line = send_command(players, mover, mover_proc, f"genmove {side_to_gtp_color(side)}\n")
            move = parse_gtp_move(line)
            if move == "pass":
                state.side ^= 1
                send_command(players, watcher, watcher_proc, f"play {side_to_gtp_color(side)} pass\n")
                continue
            state.apply_move(side, coord_to_policy(move))
            record += move
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
            "record": record,
        }
    finally:
        if p0_proc is not None:
            players[p0_idx].proc_pool[p0_color_pool].put(p0_proc)
        if p1_proc is not None:
            players[p1_idx].proc_pool[p1_color_pool].put(p1_proc)


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


def play_task(players: List[Player], p0_idx: int, p1_idx: int, opening: str, games_to_use: int) -> List[dict]:
    results = []
    if games_to_use >= 1:
        results.append(play_single_game(players, p0_idx, p1_idx, opening, True))
    if games_to_use >= 2:
        results.append(play_single_game(players, p0_idx, p1_idx, opening, False))
    for result in results:
        update_result(players, result)
    return results


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


def write_outputs(players: List[Player], output_dir: Path, completed_games: int, total_games: int) -> None:
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
        summary_rows.append(
            {
                "name": player.name,
                "games": total_n,
                "win": total_w,
                "draw": total_d,
                "loss": total_l,
                "win_rate": (total_w + 0.5 * total_d) / total_n if total_n else 0.0,
                "avg_disc_diff": total_disc / total_n if total_n else 0.0,
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
            pair_rows.append(
                {
                    "player": player.name,
                    "opponent": opponent.name,
                    "games": n,
                    "win": w,
                    "draw": d,
                    "loss": l,
                    "win_rate": (w + 0.5 * d) / n if n else 0.0,
                    "avg_disc_diff": player.disc_diff[j] / n if n else 0.0,
                }
            )
    with (output_dir / "strength_pair_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows else ["player"])
        writer.writeheader()
        writer.writerows(pair_rows)

    data = {
        "completed_games": completed_games,
        "total_games": total_games,
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


def print_status(players: List[Player], output_dir: Path, start_time: float, completed_games: int, total_games: int) -> None:
    elapsed = time.time() - start_time
    pct = 100.0 * completed_games / max(1, total_games)
    speed = completed_games / elapsed if elapsed > 0 else 0.0
    eta = (total_games - completed_games) / speed if speed > 0 else 0.0
    print("\n" + "=" * 80)
    print(f"Progress: {completed_games}/{total_games} games ({pct:.2f}%)")
    print(f"Elapsed: {format_elapsed(elapsed)}  ETA: {format_elapsed(eta)}  Speed: {speed:.2f} games/sec")
    ratings = estimate_elos(players)
    for player in players:
        n = sum(player.n_played)
        w = sum(row[0] for row in player.results)
        d = sum(row[1] for row in player.results)
        l = sum(row[2] for row in player.results)
        wr = (w + 0.5 * d) / n if n else 0.0
        elo, ci = ratings.get(player.name, (None, None))
        elo_text = "-" if elo is None else f"{elo:.1f}+-{ci:.1f}"
        print(f"{player.name}\tgames={n}\tWDL={w}/{d}/{l}\tWR={wr:.4f}\tElo={elo_text}")
    write_outputs(players, output_dir, completed_games, total_games)


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
            )
        )

    blend_script = (HUMAN_LIKE_DIR / "30_blend_with_egaroucid" / "blend_gtp_engine.py").resolve()
    for blend in parse_float_list(args.blend_params):
        players.append(
            Player(
                f"blend_{blend:.1f}",
                [
                    sys.executable,
                    str(blend_script),
                    "--weights",
                    str(Path(args.weights).resolve()),
                    "--blend-param",
                    f"{blend:.1f}",
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
                ],
                args.processes_per_player,
            )
        )
    n = len(players)
    for player in players:
        player.results = [[0, 0, 0] for _ in range(n)]
        player.disc_diff = [0.0 for _ in range(n)]
        player.n_played = [0 for _ in range(n)]
    return players


def make_tasks(players: List[Player], openings: Sequence[str], games_per_pair: int, seed: int) -> List[Tuple[int, int, str, int]]:
    rng = random.Random(seed)
    pairs = [(i, j) for i in range(len(players)) for j in range(i + 1, len(players))]
    tasks = []
    opening_idx = 0
    n_sets = (games_per_pair + 1) // 2
    for p0, p1 in pairs:
        games_left = games_per_pair
        for _ in range(n_sets):
            games_to_use = min(2, games_left)
            tasks.append((p0, p1, openings[opening_idx], games_to_use))
            games_left -= games_to_use
            opening_idx = (opening_idx + 1) % len(openings)
    rng.shuffle(tasks)
    return tasks


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run round-robin strength tests for blended policy engines.")
    parser.add_argument("--baseline-levels", default="1,5,10,15,21")
    parser.add_argument("--blend-params", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--games-per-pair", type=int, default=1000)
    parser.add_argument("--parallel-matches", type=int, default=32)
    parser.add_argument("--processes-per-player", type=int, default=32)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--status-every-games", type=int, default=200)
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--blend-egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=60.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--openings", type=Path, default=BIN_DIR / "problem" / "xot" / "openingslarge.txt")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    if args.processes_per_player < 2 or args.processes_per_player % 2 != 0:
        raise ValueError("--processes-per-player must be an even number >= 2")
    if args.parallel_matches < 1:
        raise ValueError("--parallel-matches must be positive")
    if args.games_per_pair < 1:
        raise ValueError("--games-per-pair must be positive")

    openings = read_openings(args.openings)
    players = build_players(args)
    if len(players) < 2:
        raise ValueError("at least two players are required")
    tasks = make_tasks(players, openings, args.games_per_pair, args.seed)
    total_games = sum(task[3] for task in tasks)

    print("players")
    for player in players:
        print(player.name, " ".join(display_command(player.command)))
    print("games_per_pair", args.games_per_pair)
    print("parallel_matches", args.parallel_matches)
    print("processes_per_player", args.processes_per_player)
    print("total_games", total_games)
    print("output_dir", args.output_dir)

    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "strength_dry_run.json").open("w") as f:
            json.dump(
                {
                    "players": [{"name": p.name, "command": p.command} for p in players],
                    "players_for_log": [{"name": p.name, "command": display_command(p.command)} for p in players],
                    "games_per_pair": args.games_per_pair,
                    "parallel_matches": args.parallel_matches,
                    "processes_per_player": args.processes_per_player,
                    "total_games": total_games,
                    "n_tasks": len(tasks),
                },
                f,
                indent=2,
            )
        return

    for player in players:
        player.start_processes()

    start_time = time.time()
    completed_games = 0
    executor = ThreadPoolExecutor(max_workers=args.parallel_matches)
    try:
        iterator = iter(tasks)
        futures = {}
        for _ in range(min(args.parallel_matches, len(tasks))):
            task = next(iterator)
            futures[executor.submit(play_task, players, *task)] = task
        while futures:
            for future in as_completed(list(futures.keys())):
                task = futures.pop(future)
                results = future.result()
                completed_games += len(results)
                if completed_games % args.status_every_games < len(results) or completed_games == total_games:
                    print_status(players, args.output_dir, start_time, completed_games, total_games)
                try:
                    next_task = next(iterator)
                except StopIteration:
                    pass
                else:
                    futures[executor.submit(play_task, players, *next_task)] = next_task
                break
    finally:
        shutdown_all_processes()
        executor.shutdown(wait=True, cancel_futures=True)
    print("\nAll games finished.")
    print_status(players, args.output_dir, start_time, completed_games, total_games)


if __name__ == "__main__":
    main()
