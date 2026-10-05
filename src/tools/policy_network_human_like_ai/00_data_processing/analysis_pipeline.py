"""Shared transcript validation, bounded scheduling and ETA for analyze workflows."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from board_replay import replay_games
from miss_analysis import AnalyzeEngine, default_engine, parse_analysis, transcript_key
from pipeline_common import SPLIT_NAMES, iter_transcripts
from raw_formats import normalize_transcript


def add_analysis_arguments(parser):
    parser.add_argument("--egaroucid-exe", type=Path,
                        default=os.environ.get("EGAROUCID_EXE") or default_engine())
    parser.add_argument("--workers", type=int, default=32,
                        help="Persistent engines; one search thread per engine.")
    parser.add_argument("--hash-level", type=int, default=25)
    parser.add_argument("--timeout-sec", type=float, default=600)
    add_replay_arguments(parser)


def add_replay_arguments(parser):
    parser.add_argument("--progress-seconds", type=float, default=5)
    parser.add_argument("--games-per-batch", type=int, default=4096)


def validate_positive(args, *names):
    for name in names:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")


def duration(seconds):
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def transcript_multiplicities(accepted):
    return Counter(transcript for games in accepted.values() for _, transcript in games)


def loss_histogram(rows):
    return np.bincount([row["loss"] for row in rows], minlength=129)


def format_loss_percentages(counts):
    """Report inclusive, disjoint loss ranges against all completed positions."""
    total = int(sum(counts))
    groups = (("1-2", 1, 3), ("3-4", 3, 5), ("5-6", 5, 7), ("7+", 7, 129))
    percentages = [f"{label}={100 * int(sum(counts[start:end])) / total:.2f}%" if total else f"{label}=--"
                   for label, start, end in groups]
    return f"loss[positions={total:,}]: " + ", ".join(percentages)


class Progress:
    def __init__(self, label, total, interval):
        self.label, self.total, self.interval = label, total, interval
        self.started = time.monotonic()
        self.last = -float("inf")

    def show(self, done, extra="", force=False):
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        elapsed = now - self.started
        eta = elapsed * (self.total - done) / done if done else None
        if done == self.total:
            eta = 0
        print(f"{self.label}: {done:,}/{self.total:,} | elapsed {duration(elapsed)} "
              f"| ETA {duration(eta)} | {extra}", flush=True)
        self.last = now


def collect_games(source, original, args):
    """Validate before paying for search; retain only small transcripts in RAM."""
    accepted = {name: [] for name in SPLIT_NAMES}
    unique, rejected, seen, membership = {}, [], set(), {}
    counts = {}
    progress = Progress("validate", sum(original["splits"][s]["games"] for s in SPLIT_NAMES),
                        args.progress_seconds)
    processed = 0
    for name in SPLIT_NAMES:
        games = iter_transcripts(source / name)
        count = rejected_count = 0
        while True:
            batch = list(islice(games, args.games_per_batch))
            if not batch:
                break
            for game_id, transcript, _, _ in batch:
                if game_id in seen:
                    raise ValueError(f"repeated game ID across transcript splits: {game_id}")
                seen.add(game_id)
            boards, ids, errors = replay_games([(g[0], g[1]) for g in batch])
            rejected.extend({"split": name, **entry} for entry in errors)
            bad = {entry["game_id"] for entry in errors}
            offset = 0
            for game_id, transcript, _, _ in batch:
                if game_id in bad:
                    continue
                transcript = normalize_transcript(transcript)
                if membership.setdefault(transcript, name) != name:
                    raise ValueError("identical transcript appears in multiple splits")
                length = len(transcript) // 2
                colors = boards["color"][offset:offset + length].tobytes()
                unique.setdefault(transcript, colors)
                accepted[name].append((game_id, transcript))
                offset += length
            if offset != len(boards):
                raise ValueError("replayed board count mismatch")
            count += len(batch)
            processed += len(batch)
            rejected_count += len(errors)
            progress.show(processed, name)
        if count != original["splits"][name]["games"]:
            raise ValueError(f"transcript count mismatch for {name}")
        counts[name] = {"input_games": count, "valid_games": count - rejected_count,
                        "rejected_games": rejected_count}
    progress.show(processed, force=True)
    return accepted, unique, rejected, counts


def analyze_missing(unique, cache, args, *, game_counts=None):
    cached = cache.keys()
    missing = [(transcript, colors) for transcript, colors in unique.items()
               if transcript_key(transcript) not in cached]
    hits = len(unique) - len(missing)
    multiplicities = game_counts if game_counts is not None else {}
    loss_counts = np.zeros(129, dtype=np.int64)
    if hits:
        cached_progress = Progress("cache loss statistics", hits, args.progress_seconds)
        loaded = 0
        for transcript, colors in unique.items():
            if transcript_key(transcript) in cached:
                rows = parse_analysis(cache.get(transcript), transcript, colors)
                loss_counts += loss_histogram(rows) * multiplicities.get(transcript, 1)
                loaded += 1
                cached_progress.show(loaded, format_loss_percentages(loss_counts))
        cached_progress.show(loaded, format_loss_percentages(loss_counts), force=True)
    progress = Progress("analyze", len(missing), args.progress_seconds)
    progress.show(0, f"cache {hits:,}/{len(unique):,} | workers {min(args.workers, len(missing))} "
                  f"| {format_loss_percentages(loss_counts)}", force=True)
    if not missing:
        return {"cache_hits": hits, "analyzed_games": 0}

    cancel, local = threading.Event(), threading.local()
    engines, engine_lock = [], threading.Lock()

    def work(transcript, colors):
        if cancel.is_set():
            raise InterruptedError("analysis cancelled")
        if not hasattr(local, "engine"):
            local.engine = AnalyzeEngine(args.egaroucid_exe, args.hash_level, args.timeout_sec, cancel)
            with engine_lock:
                engines.append(local.engine)
        started = time.monotonic()
        raw = local.engine.analyze(transcript)
        rows = parse_analysis(raw, transcript, colors)
        return raw, time.monotonic() - started, loss_histogram(rows)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    pending, completed = {}, 0
    todo = iter(missing)

    def fill():
        # Keep workers busy without scheduling the entire corpus as Futures.
        while len(pending) < 2 * args.workers:
            job = next(todo, None)
            if job is None:
                break
            pending[executor.submit(work, *job)] = job[0]

    try:
        fill()
        while pending:
            ready, _ = wait(pending, timeout=min(args.progress_seconds, .5), return_when=FIRST_COMPLETED)
            for future in ready:
                transcript = pending.pop(future)
                try:
                    raw, elapsed, histogram = future.result()
                except Exception as error:
                    raise RuntimeError(f"analysis failed for transcript {transcript_key(transcript)}: {error}") from error
                cache.put(transcript, raw, elapsed)
                loss_counts += histogram * multiplicities.get(transcript, 1)
                completed += 1
            fill()
            progress.show(completed, f"cache {hits:,} | queued/running {len(pending)} "
                          f"| {format_loss_percentages(loss_counts)}")
        progress.show(completed, f"cache {hits:,} | {format_loss_percentages(loss_counts)}", force=True)
    finally:
        cancel.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        # Preserve successes completed alongside a failed/interrupting task too.
        try:
            for future, transcript in pending.items():
                if not future.cancelled() and future.exception() is None:
                    raw, elapsed, _ = future.result()
                    cache.put(transcript, raw, elapsed)
        finally:
            for engine in engines:
                engine.close()
    return {"cache_hits": hits, "analyzed_games": completed}


