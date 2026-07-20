#!/usr/bin/env python3
"""
Blend a policy-network distribution with Egaroucid for Console hint scores.

Formula:
  blended = policy^alpha * softmax(egaroucid_scores)^(1 - alpha)

The policy input is side-relative:
  [player-to-move bits, opponent bits]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import re
import socket
import sqlite3
import struct
import subprocess
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


HW = 8
HW2 = 64
HW2_M1 = 63
BLACK = 0
WHITE = 1
INPUT_SIZE = 128
POLICY_SIZE = 64
ALL_LEGAL_HINT_COUNT = POLICY_SIZE
BIT_MASKS = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)).reshape(1, HW2)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_weights_file() -> Path:
    return (
        repo_root()
        / "src"
        / "tools"
        / "policy_network_human_like_ai"
        / "10_train_policy_network"
        / "trained"
        / "wthor_final_arch_512x4_e50"
        / "selected_policy_network_weights.bin"
    )


def default_egaroucid_exe() -> Path:
    return (
        repo_root()
        / "bin"
        / "versions"
        / "Egaroucid_for_Console_7_8_1_Windows_SIMD"
        / "Egaroucid_for_Console_7_8_1_SIMD.exe"
    )


def policy_to_xy(policy: int) -> Tuple[int, int]:
    pos = HW2_M1 - policy
    return pos % HW, pos // HW


def xy_to_policy(x: int, y: int) -> int:
    return HW2_M1 - (y * HW + x)


def coord_to_policy(coord: str) -> int:
    coord = coord.strip().lower()
    if len(coord) != 2 or not ("a" <= coord[0] <= "h") or not ("1" <= coord[1] <= "8"):
        raise ValueError(f"invalid coordinate: {coord}")
    return xy_to_policy(ord(coord[0]) - ord("a"), int(coord[1]) - 1)


def policy_to_coord(policy: int) -> str:
    x, y = policy_to_xy(policy)
    return chr(ord("a") + x) + str(y + 1)


def bit_from_policy(policy: int) -> int:
    return 1 << policy


def parse_side(text: str) -> int:
    text = text.strip().lower()
    if text in ("x", "b", "black", "0"):
        return BLACK
    if text in ("o", "w", "white", "1"):
        return WHITE
    raise ValueError(f"invalid side: {text}")


def side_to_egaroucid_char(side: int) -> str:
    return "X" if side == BLACK else "O"


def side_to_gtp_color(side: int) -> str:
    return "b" if side == BLACK else "w"


class BoardState:
    def __init__(self, black: int, white: int, side: int = BLACK):
        self.black = int(black)
        self.white = int(white)
        self.side = int(side)

    @classmethod
    def initial(cls) -> "BoardState":
        black = bit_from_policy(coord_to_policy("e4")) | bit_from_policy(coord_to_policy("d5"))
        white = bit_from_policy(coord_to_policy("d4")) | bit_from_policy(coord_to_policy("e5"))
        return cls(black, white, BLACK)

    @classmethod
    def from_board_string(cls, board: str, side: int) -> "BoardState":
        board = board.strip()
        if len(board) != HW2:
            raise ValueError("board string must contain exactly 64 squares")
        black = 0
        white = 0
        for pos, ch in enumerate(board):
            policy = HW2_M1 - pos
            if ch in ("X", "x", "B", "b", "*"):
                black |= bit_from_policy(policy)
            elif ch in ("O", "o", "W", "w", "1"):
                white |= bit_from_policy(policy)
            elif ch in ("-", "."):
                pass
            else:
                raise ValueError(f"invalid board character: {ch}")
        return cls(black, white, side)

    @classmethod
    def from_transcript(cls, transcript: str) -> "BoardState":
        state = cls.initial()
        transcript = transcript.strip()
        if len(transcript) % 2 != 0:
            raise ValueError("transcript length must be even")
        for i in range(0, len(transcript), 2):
            if not state.legal_policies(state.side):
                state.side ^= 1
                if not state.legal_policies(state.side):
                    raise ValueError("transcript continues after game end")
            state.apply_move(state.side, coord_to_policy(transcript[i : i + 2]))
        if not state.legal_policies(state.side):
            state.side ^= 1
        return state

    def copy(self) -> "BoardState":
        return BoardState(self.black, self.white, self.side)

    def player_opponent_bits(self, side: Optional[int] = None) -> Tuple[int, int]:
        if side is None:
            side = self.side
        if side == BLACK:
            return self.black, self.white
        return self.white, self.black

    def to_egaroucid_board(self, side: Optional[int] = None) -> str:
        if side is None:
            side = self.side
        chars = []
        for pos in range(HW2):
            policy = HW2_M1 - pos
            bit = bit_from_policy(policy)
            if self.black & bit:
                chars.append("X")
            elif self.white & bit:
                chars.append("O")
            else:
                chars.append("-")
        return "".join(chars) + " " + side_to_egaroucid_char(side)

    @staticmethod
    def calc_flips(player: int, opponent: int, policy: int) -> int:
        if policy < 0 or policy >= HW2:
            return 0
        move_bit = bit_from_policy(policy)
        if (player | opponent) & move_bit:
            return 0
        x0, y0 = policy_to_xy(policy)
        flips = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            x = x0 + dx
            y = y0 + dy
            line = 0
            while 0 <= x < HW and 0 <= y < HW:
                p = xy_to_policy(x, y)
                bit = bit_from_policy(p)
                if opponent & bit:
                    line |= bit
                else:
                    if (player & bit) and line:
                        flips |= line
                    break
                x += dx
                y += dy
        return flips

    def legal_policies(self, side: Optional[int] = None) -> List[int]:
        player, opponent = self.player_opponent_bits(side)
        occupied = player | opponent
        return [
            policy
            for policy in range(HW2)
            if (occupied & bit_from_policy(policy)) == 0 and self.calc_flips(player, opponent, policy)
        ]

    def apply_move(self, side: int, policy: int) -> None:
        player, opponent = self.player_opponent_bits(side)
        flips = self.calc_flips(player, opponent, policy)
        if not flips:
            raise ValueError(f"illegal move {policy_to_coord(policy)} for {side_to_gtp_color(side)}")
        player ^= flips | bit_from_policy(policy)
        opponent ^= flips
        if side == BLACK:
            self.black, self.white = player, opponent
        else:
            self.white, self.black = player, opponent
        self.side = side ^ 1


def softmax_values(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    values = values.astype(np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values, dtype=np.float32)
    total = np.sum(exp_values)
    if total <= 0.0:
        return np.full_like(values, 1.0 / len(values), dtype=np.float32)
    return exp_values / total


class BinaryPolicyNetwork:
    def __init__(self, path: Path):
        self.layers = []
        with path.open("rb") as f:
            magic = f.read(16)
            if magic != b"EGR_POLICY_V1\0\0\0":
                raise ValueError(f"invalid policy weights magic in {path}")
            version, n_layers, input_size, policy_size = struct.unpack("<IIII", f.read(16))
            if version != 1 or input_size != INPUT_SIZE or policy_size != POLICY_SIZE or n_layers <= 0:
                raise ValueError(f"unsupported policy weights header in {path}")
            expected_input = input_size
            for _ in range(n_layers):
                layer, expected_input = self._read_layer(f, expected_input)
                self.layers.append(layer)

    @staticmethod
    def _read_layer(f, expected_input: int):
        in_dim, out_dim, activation, alpha = struct.unpack("<IIIf", f.read(16))
        if in_dim != expected_input or out_dim <= 0:
            raise ValueError(f"invalid layer shape {in_dim}x{out_dim}")
        weights = np.frombuffer(f.read(in_dim * out_dim * 4), dtype="<f4").copy().reshape(in_dim, out_dim)
        bias = np.frombuffer(f.read(out_dim * 4), dtype="<f4").copy()
        return (weights, bias, activation, alpha), out_dim

    @staticmethod
    def _forward_layer(x: np.ndarray, layer) -> np.ndarray:
        weights, bias, activation, alpha = layer
        y = x @ weights + bias
        if activation == 1:
            y = np.where(y >= 0.0, y, y * alpha)
        return y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return softmax_rows(self.predict_logits(x))

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.layers:
            y = self._forward_layer(y, layer)
        return y


class PolicyBatchClient:
    REQUEST_STRUCT = struct.Struct("<QQB")
    RESPONSE_SIZE = POLICY_SIZE * 4

    def __init__(self, host: str, port: int, timeout_sec: float = 30.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.sock: Optional[socket.socket] = None

    def _connect(self) -> socket.socket:
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        return self.sock

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise EOFError("policy inference server closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def predict(self, state: BoardState, side: int) -> np.ndarray:
        payload = self.REQUEST_STRUCT.pack(int(state.black), int(state.white), int(side))
        for attempt in range(2):
            sock = self.sock
            if sock is None:
                sock = self._connect()
            try:
                sock.sendall(payload)
                response = self._recv_exact(sock, self.RESPONSE_SIZE)
                return np.frombuffer(response, dtype="<f4").copy().reshape(1, POLICY_SIZE)
            except (EOFError, OSError):
                self.close()
                if attempt:
                    raise
        raise RuntimeError("policy inference request failed")

    def close(self) -> None:
        sock = self.sock
        self.sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def __del__(self):
        self.close()


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def board_to_features(state: BoardState, side: int) -> np.ndarray:
    player, opponent = state.player_opponent_bits(side)
    x = np.empty((1, INPUT_SIZE), dtype=np.float32)
    player_bits = np.array([[player]], dtype=np.uint64)
    opponent_bits = np.array([[opponent]], dtype=np.uint64)
    x[:, :HW2] = ((player_bits & BIT_MASKS) != 0).astype(np.float32)
    x[:, HW2:] = ((opponent_bits & BIT_MASKS) != 0).astype(np.float32)
    return x


def normalize_policy_on_legal(policy: np.ndarray, legal_policies: Sequence[int]) -> np.ndarray:
    result = np.zeros(POLICY_SIZE, dtype=np.float32)
    if not legal_policies:
        return result
    legal_values = np.maximum(policy[np.array(legal_policies, dtype=np.int64)], 0.0)
    total = float(np.sum(legal_values))
    if total <= 0.0:
        result[np.array(legal_policies, dtype=np.int64)] = 1.0 / len(legal_policies)
    else:
        result[np.array(legal_policies, dtype=np.int64)] = legal_values / total
    return result


def geometric_blend_distribution(
    policy_dist: np.ndarray,
    egaroucid_dist: np.ndarray,
    legal_policies: Sequence[int],
    alpha: float,
) -> np.ndarray:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not legal_policies:
        return np.zeros(POLICY_SIZE, dtype=np.float32)
    if alpha <= 0.0:
        return normalize_policy_on_legal(egaroucid_dist, legal_policies)
    if alpha >= 1.0:
        return normalize_policy_on_legal(policy_dist, legal_policies)

    legal = np.array(legal_policies, dtype=np.int64)
    policy_values = np.maximum(policy_dist[legal], 0.0)
    egaroucid_values = np.maximum(egaroucid_dist[legal], 0.0)
    blended_values = (
        np.power(policy_values, alpha).astype(np.float32, copy=False)
        * np.power(egaroucid_values, 1.0 - alpha).astype(np.float32, copy=False)
    )
    result = np.zeros(POLICY_SIZE, dtype=np.float32)
    result[legal] = blended_values
    return normalize_policy_on_legal(result, legal_policies)


def parse_score(text: str) -> float:
    text = text.strip()
    if text.lower() in ("inf", "+inf", "win"):
        return 1000000.0
    if text.lower() in ("-inf", "loss"):
        return -1000000.0
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"cannot parse score: {text}")
    return float(match.group(0))


def parse_hint_rows(output: str) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    for line in output.splitlines():
        if "|" not in line or "Move" in line or "Score" in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5:
            continue
        move = parts[3].lower()
        if not re.fullmatch(r"[a-h][1-8]", move):
            continue
        try:
            rows.append((coord_to_policy(move), parse_score(parts[4])))
        except ValueError:
            continue
    return rows


def parse_hint_output(output: str) -> Dict[int, float]:
    return dict(parse_hint_rows(output))


def parse_hint_move_order(output: str) -> List[int]:
    return [policy for policy, _ in parse_hint_rows(output)]


class EgaroucidHintRunner:
    def __init__(
        self,
        exe: Path,
        level: int = 21,
        threads: int = 1,
        timeout_sec: float = 30.0,
        persistent: bool = True,
        command_stagger_sec: float = 0.0,
        command_stagger_lock=None,
        command_stagger_state=None,
        command_semaphore=None,
        cancel_event=None,
    ):
        self.exe = Path(exe)
        self.level = int(level)
        self.threads = int(threads)
        self.timeout_sec = float(timeout_sec)
        self.persistent = bool(persistent)
        self.command_stagger_sec = float(command_stagger_sec)
        if self.command_stagger_sec < 0.0:
            raise ValueError("command_stagger_sec must not be negative")
        self.command_stagger_lock = command_stagger_lock
        self.command_stagger_state = command_stagger_state
        self.command_semaphore = command_semaphore
        self.cancel_event = cancel_event
        self.local_stagger_lock = threading.Lock()
        self.local_next_command_time = time.monotonic()
        self.proc: Optional[subprocess.Popen] = None
        self.stdout_queue = None
        self.stdout_thread: Optional[threading.Thread] = None

    def command(self) -> List[str]:
        return [
            str(self.exe),
            "-l",
            str(self.level),
            "-quiet",
            "-nobook",
            "-t",
            str(self.threads),
        ]

    def _ensure_process(self) -> subprocess.Popen:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc
        self.proc = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
        )
        self.stdout_queue = queue.Queue()
        self.stdout_thread = threading.Thread(
            target=self._pump_stdout,
            args=(self.proc, self.stdout_queue),
            name=f"egaroucid-stdout-{self.proc.pid}",
            daemon=True,
        )
        self.stdout_thread.start()
        self._read_until_prompt()
        return self.proc

    @staticmethod
    def _pump_stdout(
        proc: subprocess.Popen,
        output_queue,
    ) -> None:
        try:
            if proc.stdout is None:
                output_queue.put(
                    RuntimeError("Egaroucid stdout is not available")
                )
                return
            while True:
                ch = proc.stdout.read(1)
                output_queue.put(ch)
                if ch == "":
                    return
        except BaseException as exc:
            output_queue.put(exc)

    def _read_until_prompt(self) -> str:
        if self.proc is None or self.stdout_queue is None:
            raise RuntimeError("Egaroucid process is not running")
        chars: List[str] = []
        previous = ""
        deadline = time.monotonic() + self.timeout_sec
        while True:
            if (
                self.cancel_event is not None
                and self.cancel_event.is_set()
            ):
                self.proc.kill()
                raise InterruptedError("Egaroucid command cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                partial = "".join(chars[-500:])
                self.proc.kill()
                raise TimeoutError(
                    f"Egaroucid prompt timeout after {self.timeout_sec} sec. "
                    f"partial output: {partial}"
                )
            try:
                ch = self.stdout_queue.get(
                    timeout=(
                        min(remaining, 0.25)
                        if self.cancel_event is not None
                        else remaining
                    )
                )
            except queue.Empty:
                if self.cancel_event is not None:
                    continue
                partial = "".join(chars[-500:])
                self.proc.kill()
                raise TimeoutError(
                    f"Egaroucid prompt timeout after {self.timeout_sec} sec. "
                    f"partial output: {partial}"
                )
            if isinstance(ch, BaseException):
                raise RuntimeError(
                    "failed to read Egaroucid stdout"
                ) from ch
            if ch == "":
                partial = "".join(chars[-500:])
                raise RuntimeError(
                    f"Egaroucid process ended. partial output: {partial}"
                )
            chars.append(ch)
            if previous == ">" and ch == " ":
                return "".join(chars)
            previous = ch

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        stdout_thread = self.stdout_thread
        self.stdout_thread = None
        self.stdout_queue = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if stdout_thread is not None:
            stdout_thread.join(timeout=2.0)

    def _wait_for_hint_command_slot(self) -> bool:
        if self.command_stagger_lock is None or self.command_stagger_state is None:
            if self.command_stagger_sec <= 0.0:
                return False
            with self.local_stagger_lock:
                now = time.monotonic()
                delay = max(0.0, self.local_next_command_time - now)
                if delay > 0.0:
                    time.sleep(delay)
                self.local_next_command_time = time.monotonic() + self.command_stagger_sec
            return False

        with self.command_stagger_lock:
            now = time.time()
            delay = (
                max(0.0, float(self.command_stagger_state.next_time) - now)
                if self.command_stagger_sec > 0.0
                else 0.0
            )
            if delay > 0.0:
                time.sleep(delay)
            dispatched_at = time.time()
            count = int(self.command_stagger_state.count)
            if count > 0:
                spacing = dispatched_at - float(self.command_stagger_state.last_time)
                self.command_stagger_state.minimum_spacing = min(
                    float(self.command_stagger_state.minimum_spacing),
                    spacing,
                )
            self.command_stagger_state.count = count + 1
            self.command_stagger_state.last_time = dispatched_at
            self.command_stagger_state.next_time = dispatched_at + self.command_stagger_sec
            active_count = int(self.command_stagger_state.active_count) + 1
            self.command_stagger_state.active_count = active_count
            self.command_stagger_state.maximum_active_count = max(
                int(self.command_stagger_state.maximum_active_count),
                active_count,
            )
        return True

    def _acquire_hint_execution_slot(self) -> None:
        if self.command_semaphore is not None:
            self.command_semaphore.acquire()

    def _release_hint_execution_slot(self, active_recorded: bool) -> None:
        try:
            if (
                active_recorded
                and self.command_stagger_lock is not None
                and self.command_stagger_state is not None
            ):
                with self.command_stagger_lock:
                    self.command_stagger_state.active_count = max(
                        0,
                        int(self.command_stagger_state.active_count) - 1,
                    )
        finally:
            if self.command_semaphore is not None:
                self.command_semaphore.release()

    def raw_hint(
        self,
        state: BoardState,
        side: int,
        hint_count: int = ALL_LEGAL_HINT_COUNT,
    ) -> str:
        hint_count = int(hint_count)
        if hint_count <= 0:
            raise ValueError("hint_count must be positive")
        if self.persistent:
            proc = self._ensure_process()
            if proc.stdin is None:
                raise RuntimeError("Egaroucid stdin is not available")
            proc.stdin.write(f"setboard {state.to_egaroucid_board(side)}\n")
            proc.stdin.flush()
            self._read_until_prompt()
            self._acquire_hint_execution_slot()
            active_recorded = False
            try:
                active_recorded = self._wait_for_hint_command_slot()
                proc.stdin.write(f"hint {hint_count}\n")
                proc.stdin.flush()
                return self._read_until_prompt()
            finally:
                self._release_hint_execution_slot(active_recorded)
        input_text = f"setboard {state.to_egaroucid_board(side)}\nhint {hint_count}\nquit\n"
        self._acquire_hint_execution_slot()
        active_recorded = False
        try:
            active_recorded = self._wait_for_hint_command_slot()
            proc = subprocess.run(
                self.command(),
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_sec,
            )
        finally:
            self._release_hint_execution_slot(active_recorded)
        if proc.returncode != 0:
            raise RuntimeError(f"Egaroucid exited with code {proc.returncode}: {proc.stdout}")
        return proc.stdout

    def hint_scores(
        self,
        state: BoardState,
        side: int,
        hint_count: int = ALL_LEGAL_HINT_COUNT,
    ) -> Tuple[Dict[int, float], str]:
        raw = self.raw_hint(state, side, hint_count)
        return parse_hint_output(raw), raw

    def __del__(self):
        self.close()


def hint_cache_key(
    level: int,
    black: int,
    white: int,
    side: int,
    hint_count: int = ALL_LEGAL_HINT_COUNT,
) -> str:
    return (
        f"level={int(level)}:hint={int(hint_count)}:"
        f"{int(black):016x}:{int(white):016x}:{int(side)}"
    )


def require_complete_egaroucid_scores(
    scores: Dict[int, float],
    legal_policies: Sequence[int],
) -> None:
    legal = {int(policy) for policy in legal_policies}
    scored = {int(policy) for policy in scores}
    missing = sorted(legal - scored)
    unexpected = sorted(scored - legal)
    if missing or unexpected:
        raise ValueError(
            "Egaroucid hint output does not cover exactly all legal moves: "
            f"missing={missing}, unexpected={unexpected}"
        )


class SharedHintCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60.0, isolation_level=None)
        self.conn.execute("PRAGMA busy_timeout = 60000")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hint_scores (
                key TEXT PRIMARY KEY,
                scores_json TEXT NOT NULL,
                raw_hint TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hint_claims (
                key TEXT PRIMARY KEY,
                claimed_at REAL NOT NULL
            )
            """
        )

    def get(self, key: str) -> Optional[Tuple[Dict[int, float], str]]:
        row = self.conn.execute("SELECT scores_json, raw_hint FROM hint_scores WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        scores = {int(k): float(v) for k, v in json.loads(row[0]).items()}
        return scores, str(row[1])

    def put(self, key: str, scores: Dict[int, float], raw_hint: str) -> None:
        scores_json = json.dumps({str(k): float(v) for k, v in scores.items()}, sort_keys=True)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO hint_scores(key, scores_json, raw_hint, created_at) VALUES (?, ?, ?, ?)",
                (key, scores_json, raw_hint, time.time()),
            )
            self.conn.execute("DELETE FROM hint_claims WHERE key = ?", (key,))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def try_claim(self, key: str, stale_after_sec: float) -> bool:
        stale_before = time.time() - stale_after_sec
        self.conn.execute("DELETE FROM hint_claims WHERE key = ? AND claimed_at < ?", (key, stale_before))
        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO hint_claims(key, claimed_at) VALUES (?, ?)",
            (key, time.time()),
        )
        return cursor.rowcount == 1

    def wait_for(self, key: str, timeout_sec: float) -> Optional[Tuple[Dict[int, float], str]]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            cached = self.get(key)
            if cached is not None:
                return cached
            claim = self.conn.execute("SELECT 1 FROM hint_claims WHERE key = ?", (key,)).fetchone()
            if claim is None:
                return None
            time.sleep(0.01)
        return None

    def release_claim(self, key: str) -> None:
        self.conn.execute("DELETE FROM hint_claims WHERE key = ?", (key,))

    def close(self) -> None:
        conn = self.conn
        self.conn = None
        if conn is not None:
            conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class BlendedPolicy:
    def __init__(
        self,
        weights: Path,
        egaroucid_exe: Path = default_egaroucid_exe(),
        egaroucid_level: int = 21,
        egaroucid_threads: int = 1,
        egaroucid_timeout_sec: float = 30.0,
        persistent_egaroucid: bool = True,
        cache_egaroucid: bool = False,
        hint_cache_db: Optional[Path] = None,
        policy_server_host: Optional[str] = None,
        policy_server_port: Optional[int] = None,
        policy_server_timeout_sec: float = 30.0,
        score_temperature: float = 1.0,
        legal_mask_policy: bool = True,
        hint_command_stagger_sec: float = 0.0,
        hint_command_stagger_lock=None,
        hint_command_stagger_state=None,
        hint_command_semaphore=None,
    ):
        self.policy_client = None
        if policy_server_port is not None:
            self.policy_network = None
            self.policy_client = PolicyBatchClient(
                policy_server_host or "127.0.0.1",
                policy_server_port,
                timeout_sec=policy_server_timeout_sec,
            )
        else:
            self.policy_network = BinaryPolicyNetwork(Path(weights))
        self.hint_runner = EgaroucidHintRunner(
            egaroucid_exe,
            level=egaroucid_level,
            threads=egaroucid_threads,
            timeout_sec=egaroucid_timeout_sec,
            persistent=persistent_egaroucid,
            command_stagger_sec=hint_command_stagger_sec,
            command_stagger_lock=hint_command_stagger_lock,
            command_stagger_state=hint_command_stagger_state,
            command_semaphore=hint_command_semaphore,
        )
        self.score_temperature = float(score_temperature)
        if self.score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")
        self.legal_mask_policy = bool(legal_mask_policy)
        self.cache_egaroucid = bool(cache_egaroucid)
        self.egaroucid_cache: Dict[Tuple[int, int, int], Tuple[Dict[int, float], str]] = {}
        self.shared_hint_cache = SharedHintCache(Path(hint_cache_db)) if hint_cache_db is not None else None

    def policy_distribution(self, state: BoardState, side: int, legal_policies: Sequence[int]) -> np.ndarray:
        if self.policy_client is not None:
            policy = self.policy_client.predict(state, side)[0]
        else:
            policy = self.policy_network.predict(board_to_features(state, side))[0]
        if self.legal_mask_policy:
            return normalize_policy_on_legal(policy, legal_policies)
        return policy.astype(np.float32)

    def egaroucid_distribution(self, scores: Dict[int, float], legal_policies: Sequence[int]) -> np.ndarray:
        result = np.zeros(POLICY_SIZE, dtype=np.float32)
        require_complete_egaroucid_scores(scores, legal_policies)
        if not legal_policies:
            return result
        values = np.array(
            [
                scores[policy] / self.score_temperature
                for policy in legal_policies
            ],
            dtype=np.float32,
        )
        result[np.array(legal_policies, dtype=np.int64)] = softmax_values(
            values
        )
        return result

    def cached_hint_scores(self, state: BoardState, side: int) -> Tuple[Dict[int, float], str]:
        memory_key = (state.black, state.white, int(side))
        if self.cache_egaroucid:
            cached = self.egaroucid_cache.get(memory_key)
            if cached is not None:
                require_complete_egaroucid_scores(
                    cached[0],
                    state.legal_policies(side),
                )
                return cached
        shared_key = hint_cache_key(
            self.hint_runner.level,
            state.black,
            state.white,
            int(side),
            ALL_LEGAL_HINT_COUNT,
        )
        if self.shared_hint_cache is not None:
            while True:
                cached = self.shared_hint_cache.get(shared_key)
                if cached is not None:
                    require_complete_egaroucid_scores(
                        cached[0],
                        state.legal_policies(side),
                    )
                    if self.cache_egaroucid:
                        self.egaroucid_cache[memory_key] = cached
                    return cached
                stale_after_sec = max(60.0, self.hint_runner.timeout_sec * 2.0)
                if self.shared_hint_cache.try_claim(shared_key, stale_after_sec):
                    break
                cached = self.shared_hint_cache.wait_for(shared_key, self.hint_runner.timeout_sec)
                if cached is not None:
                    require_complete_egaroucid_scores(
                        cached[0],
                        state.legal_policies(side),
                    )
                    if self.cache_egaroucid:
                        self.egaroucid_cache[memory_key] = cached
                    return cached
        try:
            scores, raw_hint = self.hint_runner.hint_scores(
                state,
                side,
                ALL_LEGAL_HINT_COUNT,
            )
            require_complete_egaroucid_scores(
                scores,
                state.legal_policies(side),
            )
            if self.shared_hint_cache is not None:
                self.shared_hint_cache.put(shared_key, scores, raw_hint)
        except BaseException:
            if self.shared_hint_cache is not None:
                self.shared_hint_cache.release_claim(shared_key)
            raise
        if self.cache_egaroucid:
            self.egaroucid_cache[memory_key] = (scores, raw_hint)
        return scores, raw_hint

    def close(self) -> None:
        self.hint_runner.close()
        if self.policy_client is not None:
            self.policy_client.close()
        if self.shared_hint_cache is not None:
            self.shared_hint_cache.close()
            self.shared_hint_cache = None

    def blended_distribution(self, state: BoardState, side: int, blend_param: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, float], str]:
        legal_policies = state.legal_policies(side)
        alpha = float(blend_param)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        policy_dist = self.policy_distribution(state, side, legal_policies)
        if alpha >= 1.0:
            egaroucid_dist = np.zeros(POLICY_SIZE, dtype=np.float32)
            return normalize_policy_on_legal(policy_dist, legal_policies), policy_dist, egaroucid_dist, {}, ""
        scores, raw_hint = self.cached_hint_scores(state, side)
        egaroucid_dist = self.egaroucid_distribution(scores, legal_policies)
        blended = geometric_blend_distribution(policy_dist, egaroucid_dist, legal_policies, alpha)
        return blended, policy_dist, egaroucid_dist, scores, raw_hint


def distribution_rows(
    blended: np.ndarray,
    policy_dist: np.ndarray,
    egaroucid_dist: np.ndarray,
    scores: Dict[int, float],
    legal_policies: Sequence[int],
    top: int,
) -> List[dict]:
    order = sorted(legal_policies, key=lambda policy: float(blended[policy]), reverse=True)
    rows = []
    for rank, policy in enumerate(order[:top], start=1):
        rows.append(
            {
                "rank": rank,
                "move": policy_to_coord(policy),
                "policy": float(policy_dist[policy]),
                "egaroucid": float(egaroucid_dist[policy]),
                "blended": float(blended[policy]),
                "score": scores.get(policy),
            }
        )
    return rows


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blend policy-network output with Egaroucid hint scores.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--board", help="64-character board string using X/O/- in a1..h8 order.")
    source.add_argument("--transcript", help="Transcript such as f5d6c5.")
    parser.add_argument("--side", default="black", help="Side to move for --board: black/white/X/O.")
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-threads", type=int, default=1)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=30.0)
    parser.add_argument("--no-persistent-egaroucid", action="store_true")
    parser.add_argument("--cache-egaroucid", action="store_true")
    parser.add_argument("--hint-cache-db", type=Path, default=None)
    parser.add_argument("--blend-param", "--alpha", dest="blend_param", type=float, default=0.5)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--no-legal-mask-policy", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-raw-hint", action="store_true")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    if args.board is not None:
        side = parse_side(args.side)
        state = BoardState.from_board_string(args.board, side)
    else:
        state = BoardState.from_transcript(args.transcript)
        side = state.side

    blender = BlendedPolicy(
        weights=args.weights,
        egaroucid_exe=args.egaroucid_exe,
        egaroucid_level=args.egaroucid_level,
        egaroucid_threads=args.egaroucid_threads,
        egaroucid_timeout_sec=args.egaroucid_timeout_sec,
        persistent_egaroucid=not args.no_persistent_egaroucid,
        cache_egaroucid=args.cache_egaroucid,
        hint_cache_db=args.hint_cache_db,
        score_temperature=args.score_temperature,
        legal_mask_policy=not args.no_legal_mask_policy,
    )
    try:
        blended, policy_dist, egaroucid_dist, scores, raw_hint = blender.blended_distribution(state, side, args.blend_param)
    finally:
        blender.close()
    legal_policies = state.legal_policies(side)
    rows = distribution_rows(blended, policy_dist, egaroucid_dist, scores, legal_policies, args.top)
    result = {
        "alpha": args.blend_param,
        "blend_param": args.blend_param,
        "side": side_to_gtp_color(side),
        "legal_moves": [policy_to_coord(policy) for policy in legal_policies],
        "top": rows,
    }
    if args.show_raw_hint:
        result["raw_hint"] = raw_hint

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"alpha {args.blend_param}")
        print(f"side {side_to_gtp_color(side)}")
        print("rank move blended policy egaroucid score")
        for row in rows:
            score_text = "-" if row["score"] is None else f"{row['score']:.3f}"
            print(
                f"{row['rank']:>4} {row['move']:>4} {row['blended']:.8f} "
                f"{row['policy']:.8f} {row['egaroucid']:.8f} {score_text}"
            )
        if args.show_raw_hint:
            print("\nraw_hint")
            print(raw_hint)


if __name__ == "__main__":
    main()
