#!/usr/bin/env python3
"""Measure the GPW30 proposal players' own moves using Egaroucid level 21."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for directory in ("00_data_processing", "40_test_strength"):
    sys.path.insert(0, str(HERE.parent / directory))

from analysis_pipeline import Progress, validate_positive
from battle_blend_strength import OutputRunLock, task_plan_hash
from board_replay import replay_games
from miss_analysis import AnalysisCache, AnalyzeEngine, engine_identity, parse_analysis, transcript_key
from raw_formats import normalize_transcript
from strength_tournament import MatchSetTask, atomic_write_json
from strength_engine import _create_windows_kill_job, _close_windows_job
from wthor_dataset import file_sha256

LEVEL = 21
ALPHAS = (.2, .4, .6, .8, 1.0)
LEVELS = (1, 3, 5, 9, 13, 17, 21)
FAMILIES = (("normal_egaroucid", "normal", "egaroucid_l21"),
            ("loss_a_normal", "loss_a", "normal"),
            ("loss_b_loss_a", "loss_b", "loss_a"))
PROPOSALS = {f"{family}_alpha_{alpha:.1f}": {"family": family, "alpha": alpha}
             for family, _, _ in FAMILIES for alpha in ALPHAS}
PLAYER_NAMES = (tuple(f"egaroucid_l{level}" for level in LEVELS)
                + tuple(list(PROPOSALS)[:5])
                + tuple(f"edax_l{level}" for level in LEVELS)
                + tuple(list(PROPOSALS)[5:]) + ("random",))
DEFAULT_TOURNAMENT = HERE.parent / "40_test_strength/output/gpw30_independent_xot_50sets"


@dataclass(frozen=True)
class Game:
    task_id: int
    color_game: int
    black: str
    white: str
    opening_plies: int
    transcript: str


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_tournament(directory):
    """Read only; reject incomplete logs, another roster, and duplicate matches."""
    manifest = read_json(directory / "strength_manifest.json")
    result = read_json(directory / "strength_results.json")
    config = manifest["configuration"]
    roster = config["participants"]["paper_roster"]
    names = tuple(player["name"] for player in result["players"])
    families = [{"pi1": pi1, "pi2": pi2, "alphas": list(ALPHAS)} for _, pi1, pi2 in FAMILIES]
    sets_per_pair = config["schedule"]["match_sets_per_pair"]
    expected_sets = 435 * sets_per_pair
    if (names != PLAYER_NAMES or roster["total_players"] != 30
            or roster["policy_families"] != families
            or manifest["experiment_id"] != result["experiment_id"]
            or sets_per_pair <= 0
            or result["completed_match_sets"] != expected_sets
            or result["total_match_sets"] != expected_sets
            or result["completed_actual_games"] != 2 * expected_sets
            or result["total_actual_games"] != 2 * expected_sets):
        raise ValueError("Expected the completed GPW 30-player round robin and its three proposal families")
    tasks, pair_sets, games = {}, set(), []
    with (directory / "strength_games.jsonl").open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task, match = row["task"], row["result"]
            task_id, p0, p1 = task["task_id"], task["p0_idx"], task["p1_idx"]
            pair_set = (p0, p1, task["set_index"])
            if (row["schema_version"] != 2 or not 0 <= p0 < p1 < 30
                    or not 0 <= task["set_index"] < sets_per_pair
                    or task_id in tasks or pair_set in pair_sets
                    or any(match[key] != value for key, value in task.items())
                    or len(match["color_games"]) != 2):
                raise ValueError(f"Invalid or duplicate match at line {line_number}")
            tasks[task_id] = MatchSetTask(**task)
            pair_sets.add(pair_set)
            opening = normalize_transcript(task["opening"])
            colors_seen = set()
            for color_game, game in enumerate(match["color_games"]):
                black, white = game["black_idx"], game["white_idx"]
                p0_black = game["p0_is_black"]
                if (type(p0_black) is not bool or p0_black in colors_seen
                        or (game["p0_idx"], game["p1_idx"]) != (p0, p1)
                        or (black, white) != ((p0, p1) if p0_black else (p1, p0))):
                    raise ValueError(f"Invalid player/color assignment at line {line_number}")
                colors_seen.add(p0_black)
                transcript = normalize_transcript(game["transcript"])
                if not transcript.startswith(opening) or len(transcript) <= len(opening):
                    raise ValueError(f"Transcript does not continue its opening at line {line_number}")
                if names[black] in PROPOSALS or names[white] in PROPOSALS:
                    games.append(Game(task_id, color_game, names[black], names[white],
                                      len(opening) // 2, transcript))
    if (len(tasks) != expected_sets or set(tasks) != set(range(expected_sets))
            or task_plan_hash([tasks[i] for i in sorted(tasks)]) != config["schedule"]["task_plan_sha256"]):
        raise ValueError("Result log is incomplete or differs from the manifest's task plan")
    return manifest, sorted(games, key=lambda game: (game.task_id, game.color_game))


def validate_transcripts(games, interval):
    """Replay passes explicitly; ply parity alone cannot identify a move's owner."""
    transcripts = list(dict.fromkeys(game.transcript for game in games))
    unique = {}
    progress = Progress("validate transcripts", len(transcripts), interval)
    for start in range(0, len(transcripts), 1024):
        batch = transcripts[start:start + 1024]
        boards, _, rejected = replay_games(list(enumerate(batch)))
        if rejected:
            raise ValueError(f"Invalid tournament transcript: {rejected[0]}")
        offset = 0
        for transcript in batch:
            length = len(transcript) // 2
            unique[transcript] = boards["color"][offset:offset + length].tobytes()
            offset += length
        progress.show(len(unique))
    progress.show(len(unique), force=True)
    return unique


class Level21Engine(AnalyzeEngine):
    """L21 console with explicit thread count and parent-owned process lifetime."""

    def __init__(self, exe, hash_level, timeout, cancel, threads=1):
        self.timeout, self.cancel = timeout, cancel
        self.reader, self._job = None, None
        self.proc = subprocess.Popen(
            [str(exe), "-l", str(LEVEL), "-t", str(threads), "-hash", str(hash_level),
             "-quiet", "-nobook", "-noboard", "-m", "3"],
            cwd=exe.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            self._job = _create_windows_kill_job(self.proc)
            if os.name == "nt" and self._job is None:
                raise RuntimeError("Could not attach Egaroucid to a kill-on-close Windows job")
            self.output = queue.Queue()
            self.reader = threading.Thread(target=self._pump, daemon=True)
            self.reader.start()
            self._read_prompt()
        except BaseException:
            self.close()
            raise

    def analyze(self, transcript):
        # Each game is independent of worker scheduling and earlier cache hits.
        self.command("clearcache")
        return super().analyze(transcript)

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            if self.reader is not None:
                self.reader.join(timeout=2)
            self.proc.stdin.close()
            self.proc.stdout.close()
        finally:
            _close_windows_job(self._job)
            self._job = None


def reference_identity(exe, hash_level, threads=1):
    return {**engine_identity(exe, hash_level), "level": LEVEL,
            "threads": threads, "clear_cache_per_game": True, "protocol": "console_analyze_v1"}


class ThreadedAnalysisCache:
    """Retain actual search thread counts while reusing either permitted mode."""

    def __init__(self, directory, identity, tail_threads):
        self.caches = {}
        try:
            for threads in dict.fromkeys((1, tail_threads)):
                self.caches[threads] = AnalysisCache(directory, {**identity, "threads": threads})
            self.available = {threads: cache.keys() for threads, cache in self.caches.items()}
        except BaseException:
            self.close()
            raise

    @property
    def configurations(self):
        return {str(threads): cache.configuration for threads, cache in self.caches.items()}

    def keys(self):
        return set().union(*self.available.values())

    def put(self, transcript, raw, elapsed, threads):
        self.caches[threads].put(transcript, raw, elapsed)
        self.available[threads].add(transcript_key(transcript))

    def get(self, transcript):
        for threads, cache in self.caches.items():
            if transcript_key(transcript) in self.available[threads]:
                return cache.get(transcript)
        raise ValueError("Missing analysis cache entry")

    def close(self):
        for cache in self.caches.values():
            cache.close()


def analyze_games(unique, cache, args, on_progress=None):
    cached = cache.keys()
    remaining = {transcript: colors for transcript, colors in unique.items()
                 if transcript_key(transcript) not in cached}
    hits = len(unique) - len(remaining)
    counts = {"parallel": 0, "tail": 0}
    progress = Progress("analyze L21", len(remaining), args.progress_seconds)

    def run_phase(phase, workers, threads, stop_remaining=0):
        cancel, local = threading.Event(), threading.local()
        engines, lock = [], threading.Lock()
        pending, todo = {}, iter(list(remaining.items()))
        last_status = -float("inf")

        def publish(force=False):
            nonlocal last_status
            now = time.monotonic()
            if not force and now - last_status < args.progress_seconds:
                return
            with lock:
                pids = [engine.proc.pid for engine in engines]
            progress.show(sum(counts.values()), f"cached {hits:,}; {phase} "
                          f"{workers} processes x {threads} threads; remaining {len(remaining):,}", force=True)
            if on_progress:
                on_progress({"phase": phase, "engine_processes": len(pids), "engine_pids": pids,
                             "process_limit": workers, "threads_per_engine": threads,
                             "cache_hits": hits, "completed_transcripts": len(unique) - len(remaining),
                             "total_transcripts": len(unique), "remaining_transcripts": len(remaining),
                             "parallel_analyzed_transcripts": counts["parallel"],
                             "tail_analyzed_transcripts": counts["tail"]})
            last_status = now

        def work(transcript, colors):
            if cancel.is_set():
                raise InterruptedError("Analysis cancelled")
            if not hasattr(local, "engine"):
                local.engine = Level21Engine(args.egaroucid_exe, args.hash_level, args.timeout_sec,
                                            cancel, threads=threads)
                with lock:
                    engines.append(local.engine)
            started = time.monotonic()
            raw = local.engine.analyze(transcript)
            parse_analysis(raw, transcript, colors)
            return raw, time.monotonic() - started

        def save(future, transcript):
            raw, elapsed = future.result()
            cache.put(transcript, raw, elapsed, threads=threads)
            remaining.pop(transcript)
            counts[phase] += 1

        executor = ThreadPoolExecutor(max_workers=workers)
        publish(force=True)
        transitioning = False
        try:
            while remaining:
                while len(pending) < workers:
                    job = next(todo, None)
                    if job is None:
                        break
                    pending[executor.submit(work, *job)] = job[0]
                if not pending:
                    break
                ready, _ = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
                for future in ready:
                    transcript = pending.pop(future)
                    try:
                        save(future, transcript)
                    except Exception as error:
                        raise RuntimeError(f"Analysis failed for {transcript_key(transcript)}: {error}") from error
                publish()
                if stop_remaining and len(remaining) <= stop_remaining:
                    # Do not drain the last batch at 1 thread each. Cancel its
                    # unfinished searches and redo them with the full tail budget.
                    transitioning = True
                    break
        finally:
            cancel.set()
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            try:
                for future, transcript in pending.items():
                    if not future.cancelled() and future.exception() is None:
                        save(future, transcript)
                    elif (transitioning and not future.cancelled()
                          and not isinstance(future.exception(), InterruptedError)):
                        raise RuntimeError(f"Analysis failed during tail transition: {transcript_key(transcript)}") from future.exception()
            finally:
                for engine in engines:
                    engine.close()
                engines.clear()
        publish(force=True)

    if remaining and (not args.tail_games or len(remaining) > args.tail_games):
        run_phase("parallel", min(args.workers, len(remaining)), 1, args.tail_games)
    if remaining:
        print(f"Tail: {len(remaining):,} unfinished transcripts -> 1 process x {args.tail_threads} threads", flush=True)
        run_phase("tail", 1, args.tail_threads)
    return {"cache_hits": hits, "analyzed_transcripts": sum(counts.values()),
            "parallel_analyzed_transcripts": counts["parallel"], "tail_analyzed_transcripts": counts["tail"]}


def game_statistics(game, rows):
    for color, player, opponent in ((0, game.black, game.white), (1, game.white, game.black)):
        if player not in PROPOSALS:
            continue
        losses = [row["loss"] for row in rows
                  if row["ply"] > game.opening_plies and row["color"] == color]
        yield {"task_id": game.task_id, "color_game": game.color_game,
               "player": player, **PROPOSALS[player], "opponent": opponent,
               "color": "black" if color == 0 else "white", "opening_plies": game.opening_plies,
               "moves": len(losses), "total_disc_loss": sum(losses),
               "avg_disc_loss": sum(losses) / len(losses) if losses else None,
               "nonzero_loss_moves": sum(loss > 0 for loss in losses),
               "transcript_sha256": transcript_key(game.transcript)}


def summarize(records):
    games = len(records)
    moves = sum(row["moves"] for row in records)
    total = sum(row["total_disc_loss"] for row in records)
    return {"games": games, "moves": moves, "total_disc_loss": total,
            "avg_disc_loss": total / moves if moves else None,
            "avg_total_disc_loss_per_game": total / games if games else None,
            "nonzero_loss_moves": sum(row["nonzero_loss_moves"] for row in records)}


def write_csv(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def aggregate(games, unique, cache, output, interval):
    records, by_player, by_opponent = [], defaultdict(list), defaultdict(list)
    progress = Progress("aggregate", len(games), interval)
    for index, game in enumerate(games, 1):
        rows = parse_analysis(cache.get(game.transcript), game.transcript, unique[game.transcript])
        for row in game_statistics(game, rows):
            records.append(row)
            by_player[row["player"]].append(row)
            by_opponent[row["player"], row["opponent"]].append(row)
        progress.show(index)
    summary = [{"player": player, **metadata, **summarize(by_player[player])}
               for player, metadata in PROPOSALS.items()]
    opponents = [{"player": player, **PROPOSALS[player], "opponent": opponent, **summarize(rows)}
                 for (player, opponent), rows in sorted(by_opponent.items())]
    write_csv(output / "avg_disc_loss.csv", summary)
    write_csv(output / "avg_disc_loss_by_opponent.csv", opponents)
    write_csv(output / "avg_disc_loss_by_game.csv", records)
    return summary


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-dir", type=Path, default=DEFAULT_TOURNAMENT)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: this script's output/<tournament-name>_l21[_smokeN]")
    parser.add_argument("--egaroucid-exe", type=Path,
                        help="Default: the executable recorded in the tournament manifest (hash checked)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel engine processes, one search thread each (default: 32)")
    parser.add_argument("--tail-games", type=int, default=32,
                        help="Switch to one engine when unfinished transcripts reach this count (default: 32; 0 disables)")
    parser.add_argument("--tail-threads", type=int, default=32,
                        help="Threads for the single tail engine (default: 32)")
    parser.add_argument("--hash-level", type=int, default=25)
    parser.add_argument("--timeout-sec", type=float, default=1800)
    parser.add_argument("--progress-seconds", type=float, default=5)
    parser.add_argument("--max-games", type=int,
                        help="Smoke test only: first N selected games in task order, marked as partial")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without searching or writing output")
    return parser


def run(args):
    validate_positive(args, "workers", "tail_threads", "timeout_sec", "progress_seconds")
    if args.tail_games < 0:
        raise ValueError("--tail-games must be nonnegative")
    if not 0 <= args.hash_level <= 29 or (args.max_games is not None and args.max_games <= 0):
        raise ValueError("--hash-level must be 0..29 and --max-games must be positive")
    directory = args.tournament_dir.expanduser().resolve()
    source_paths = [directory / name for name in
                    ("strength_manifest.json", "strength_results.json", "strength_games.jsonl")]
    sources = {path.name: {"path": str(path), "sha256": file_sha256(path)} for path in source_paths}
    manifest, games = load_tournament(directory)
    total_games = len(games)
    if args.max_games is not None:
        games = games[:args.max_games]
    unique = validate_transcripts(games, args.progress_seconds)
    recorded = manifest["configuration"]["inputs"]["egaroucid_executable"]
    exe = args.egaroucid_exe or Path(recorded["path"])
    args.egaroucid_exe = (exe if exe.is_absolute() else REPO / exe).resolve()
    if file_sha256(args.egaroucid_exe) != recorded["sha256"]:
        raise ValueError("Egaroucid executable differs from the tournament manifest")
    identity = reference_identity(args.egaroucid_exe, args.hash_level)
    # Detect source replacement while validation was in progress.
    if any(file_sha256(path) != sources[path.name]["sha256"] for path in source_paths):
        raise ValueError("Tournament files changed during validation; retry after they are stable")
    counts = Counter(player for game in games for player in (game.black, game.white) if player in PROPOSALS)
    print(f"GPW30 | proposal players {len(PROPOSALS)} | selected games {len(games):,}/{total_games:,} "
          f"| unique transcripts {len(unique):,} | reference L{LEVEL}", flush=True)
    for player in PROPOSALS:
        print(f"  {player}: {counts[player]:,} games", flush=True)
    print(f"Execution: {args.workers} processes x 1 thread; tail <= {args.tail_games} transcripts: "
          f"1 process x {args.tail_threads} threads ({'enabled' if args.tail_games else 'disabled'})", flush=True)
    if args.dry_run:
        print("Dry-run passed. XOT moves and passes are excluded; only each proposal's own moves count.", flush=True)
        return
    suffix = f"_smoke{args.max_games}" if args.max_games is not None else ""
    output = (args.output_dir or HERE / "output" / f"{directory.name}_l21{suffix}").expanduser().resolve()
    report = {"schema_version": 1, "experiment_id": manifest["experiment_id"],
              "sources": sources, "reference": identity,
              "tail_reference": {**identity, "threads": args.tail_threads},
              "execution": {"workers": args.workers, "threads_per_worker": 1,
                            "tail_games": args.tail_games, "tail_threads": args.tail_threads},
              "loss_formula": "max(played_score, alternative_score) - played_score; no alternative => 0",
              "average_formula": "sum of own disc losses / own played moves after XOT (zero losses included)",
              "include_opening": False, "passes_counted": False,
              "total_eligible_games": total_games, "selected_games": len(games),
              "unique_transcripts": len(unique), "partial": len(games) < total_games}
    with OutputRunLock(output):
        status_path = output / "status.json"
        state = {"status": "running", "partial": report["partial"], "pid": os.getpid(),
                 "started_at": datetime.now(timezone.utc).isoformat(), "execution": report["execution"]}

        def update_status(values):
            state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            atomic_write_json(status_path, state)

        update_status({"phase": "prepare"})
        try:
            with closing(ThreadedAnalysisCache(output / "cache", identity, args.tail_threads)) as cache:
                stats = analyze_games(unique, cache, args, on_progress=update_status)
                update_status({"phase": "aggregate", "completed_transcripts": len(unique),
                               "remaining_transcripts": 0, "engine_processes": 0, "engine_pids": []})
                summary = aggregate(games, unique, cache, output, args.progress_seconds)
                report.update(status="completed", analysis=stats, cache_configurations=cache.configurations,
                              summary=summary, completed_at=datetime.now(timezone.utc).isoformat())
                atomic_write_json(output / "avg_disc_loss.json", report)
            update_status({"status": "completed", "phase": "completed"})
        except BaseException as error:
            update_status({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           "error": str(error), "engine_processes": 0, "engine_pids": []})
            raise
    print(f"Saved: {output / 'avg_disc_loss.csv'}", flush=True)
    return report


def main():
    args = make_argparser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted. Run the same command to resume from completed analyses.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
