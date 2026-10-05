"""Egaroucid console analyze protocol and threshold-independent durable cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import re
import sqlite3
import subprocess
import threading
import time
import zlib

from wthor_dataset import file_sha256


LEVEL = 10
THREADS = 1


def default_engine() -> Path:
    return (Path(__file__).resolve().parents[4] / "bin" / "versions"
            / "Egaroucid_for_Console_7_8_1_Windows_SIMD"
            / "Egaroucid_for_Console_7_8_1_SIMD.exe")


def engine_identity(exe: Path, hash_level: int) -> dict:
    # Include both evaluation files; relocating an identical binary is harmless.
    resources = exe.parent / "resources"
    files = [exe, resources / "eval.egev2", resources / "eval_move_ordering_end.egev",
             resources / "hash" / f"hash{hash_level}.eghs"]
    return {"cache_version": 1, "level": LEVEL, "threads": THREADS,
            "hash_level": hash_level, "book": False, "clear_cache_per_game": False,
            "files": {p.name: file_sha256(p) for p in files}}


def transcript_key(transcript: str) -> str:
    return hashlib.sha256(transcript.encode("ascii")).hexdigest()


def coord_policy(coord: str) -> int:
    if not re.fullmatch(r"[a-h][1-8]", coord):
        raise ValueError(f"invalid analyze coordinate: {coord!r}")
    return 63 - ((ord(coord[1]) - ord("1")) * 8 + ord(coord[0]) - ord("a"))


def parse_analysis(raw: str, transcript: str, colors: bytes) -> list:
    """Validate every row and return it in forward ply order (passes excluded)."""
    rows = {}
    if "[ERROR]" in raw.upper():
        raise ValueError(f"Egaroucid reported an error: {raw[-1000:]}")
    for line in raw.splitlines():
        cells = [cell.strip() for cell in line.strip().split("|")]
        if len(cells) < 2 or not cells[1].isdigit():
            continue
        if len(cells) != 11 or cells[0] or cells[-1]:
            raise ValueError(f"malformed analyze row: {line}")
        _, ply, color, played, depth, score, alternative, alt_depth, alt_score, judge, _ = cells
        ply = int(ply)
        if ply in rows or not 1 <= ply <= len(colors):
            raise ValueError(f"duplicate or out-of-range analyze ply: {ply}")
        if color != ("Black" if colors[ply - 1] == 0 else "White"):
            raise ValueError(f"analyze player mismatch at ply {ply}")
        played = played.lower()
        if played != transcript[2 * (ply - 1):2 * ply]:
            raise ValueError(f"analyze move mismatch at ply {ply}")
        score = int(score)
        if alternative.lower() == "none":
            if alt_depth or alt_score:
                raise ValueError("unexpected score for missing alternative")
            alternative, alt_score = None, None
        else:
            alternative = coord_policy(alternative.lower())
            alt_score = int(alt_score)
        if not -64 <= score <= 64 or (alt_score is not None and not -64 <= alt_score <= 64):
            raise ValueError("analyze score outside [-64, 64]")
        # Alternative is the best OTHER move and can be worse than Played.
        best = max(score, alt_score) if alt_score is not None else score
        rows[ply] = {"ply": ply, "color": colors[ply - 1], "played": coord_policy(played),
                     "played_depth": depth, "played_score": score,
                     "alternative": alternative, "alternative_depth": alt_depth,
                     "alternative_score": alt_score, "best_score": best,
                     "loss": best - score, "judge": judge}
    if len(rows) != len(colors) or len(transcript) != len(colors) * 2:
        raise ValueError(f"incomplete analyze result: {len(rows)}/{len(colors)} plies")
    return [rows[ply] for ply in range(1, len(colors) + 1)]


class AnalysisCache:
    """Only the coordinator writes SQLite; each completed game is committed."""

    def __init__(self, directory: Path, identity: dict):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "analyze.sqlite3"
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS configurations (id TEXT PRIMARY KEY, json TEXT NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS analyses (
            configuration TEXT NOT NULL, key TEXT NOT NULL, transcript TEXT NOT NULL,
            raw_zlib BLOB NOT NULL, elapsed_seconds REAL NOT NULL,
            completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (configuration, key))""")
        encoded = json.dumps(identity, sort_keys=True)
        self.configuration = hashlib.sha256(encoded.encode()).hexdigest()
        self.db.execute("INSERT OR IGNORE INTO configurations VALUES (?, ?)", (self.configuration, encoded))
        self.db.commit()
        self.identity = identity
        self._set_compatible_configurations()

    def _set_compatible_configurations(self):
        """Reuse old cold-search results with the same engine running warm.

        Preserve their original configuration IDs and raw output. All other
        settings, including binary/evaluation hashes, must match exactly.
        """
        self.compatible_configurations = [self.configuration]
        if self.identity.get("clear_cache_per_game") is not False:
            return
        legacy_identity = {**self.identity, "clear_cache_per_game": True}
        for configuration, encoded in self.db.execute("SELECT id, json FROM configurations ORDER BY id"):
            if json.loads(encoded) == legacy_identity:
                self.compatible_configurations.append(configuration)

    @classmethod
    def open_existing(cls, directory: Path, configuration=None):
        """Read cached scores without needing an executable or changing the DB."""
        instance = cls.__new__(cls)
        instance.path = directory / "analyze.sqlite3"
        instance.db = sqlite3.connect(instance.path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            configurations = dict(instance.db.execute("SELECT id, json FROM configurations"))
            if configuration is None:
                if len(configurations) != 1:
                    raise ValueError("cache has multiple/no configurations; pass --configuration with an ID from configurations")
                configuration = next(iter(configurations))
            if configuration not in configurations:
                raise ValueError("configuration does not exist in analyze cache")
            instance.configuration = configuration
            instance.identity = json.loads(configurations[configuration])
            instance._set_compatible_configurations()
            return instance
        except BaseException:
            instance.db.close()
            raise

    def keys(self) -> set:
        return {row[0] for configuration in self.compatible_configurations
                for row in self.db.execute("SELECT key FROM analyses WHERE configuration = ?", (configuration,))}

    def put(self, transcript: str, raw: str, elapsed: float):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO analyses "
                            "(configuration, key, transcript, raw_zlib, elapsed_seconds) VALUES (?, ?, ?, ?, ?)",
                            (self.configuration, transcript_key(transcript), transcript,
                             zlib.compress(raw.encode("utf-8")), elapsed))

    def get(self, transcript: str) -> str:
        key = transcript_key(transcript)
        # Prefer the selected configuration if both it and a legacy run exist.
        for configuration in self.compatible_configurations:
            row = self.db.execute("SELECT transcript, raw_zlib FROM analyses WHERE configuration = ? AND key = ?",
                                  (configuration, key)).fetchone()
            if row is not None:
                if row[0] != transcript:
                    raise ValueError("mismatched analyze cache entry")
                return zlib.decompress(row[1]).decode("utf-8")
        raise ValueError("missing analyze cache entry")

    def close(self):
        self.db.close()


class AnalyzeEngine:
    """One persistent, single-search-thread console owned by one Python worker."""

    def __init__(self, exe: Path, hash_level: int, timeout: float, cancel: threading.Event):
        self.timeout, self.cancel = timeout, cancel
        self.proc = subprocess.Popen(
            [str(exe), "-l", str(LEVEL), "-t", str(THREADS), "-hash", str(hash_level),
             "-quiet", "-nobook", "-noboard", "-m", "3"],
            cwd=exe.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.output = queue.Queue()
        self.reader = threading.Thread(target=self._pump, daemon=True)
        self.reader.start()
        try:
            self._read_prompt()
        except BaseException:
            self.close()
            raise

    def _pump(self):
        chunk = ""
        try:
            while True:
                char = self.proc.stdout.read(1)
                if not char:
                    if chunk:
                        self.output.put(chunk)
                    break
                chunk += char
                if char == "\n" or chunk.endswith("> "):
                    self.output.put(chunk)
                    chunk = ""
        finally:
            self.output.put(None)

    def _read_prompt(self) -> str:
        deadline = time.monotonic() + self.timeout
        chunks = []
        while True:
            if self.cancel.is_set():
                raise InterruptedError("analysis cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Egaroucid timeout ({self.timeout}s): {''.join(chunks)[-1000:]}")
            try:
                chunk = self.output.get(timeout=min(.25, max(.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if chunk is None:
                raise RuntimeError(f"Egaroucid exited: {''.join(chunks)[-1000:]}")
            chunks.append(chunk)
            if chunk.endswith("> "):
                result = "".join(chunks)
                if "[ERROR]" in result.upper():
                    raise RuntimeError(f"Egaroucid error: {result[-1000:]}")
                return result

    def command(self, command: str) -> str:
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        return self._read_prompt()

    def analyze(self, transcript: str) -> str:
        # init resets board/history only. Keep the transposition table alive
        # across games so shared positions/subtrees can reuse scores and bounds.
        # Leave the engine's normal replacement/aging policy enabled for long runs.
        self.command("init")
        self.command("play " + transcript)
        return self.command("analyze")

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.reader.join(timeout=2)
        self.proc.stdin.close()
        self.proc.stdout.close()
