#!/usr/bin/env python3
"""Timeout-safe GTP process pools and one-game execution."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from typing import Dict, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
HUMAN_LIKE_DIR = SCRIPT_DIR.parent
BLEND_DIR = HUMAN_LIKE_DIR / "30_blend_with_egaroucid"

import sys

sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import (  # noqa: E402
    BLACK,
    BoardState,
    coord_to_policy,
    side_to_gtp_color,
)
from strength_tournament import GameResult, MatchSetTask, PlayerSpec  # noqa: E402


PROCESS_POOL_WAIT_SEC = 0.2
GRACEFUL_SHUTDOWN_SEC = 2.0
FORCED_SHUTDOWN_SEC = 5.0
_EOF = object()


def _create_windows_kill_job(
    process: subprocess.Popen,
) -> Optional[int]:
    """Put one process tree in a kill-on-close Windows Job Object."""

    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    information = ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(
        job,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        kernel32.CloseHandle(job)
        return None
    if not kernel32.AssignProcessToJobObject(
        job,
        wintypes.HANDLE(process._handle),
    ):
        kernel32.CloseHandle(job)
        return None
    return int(job)


def _close_windows_job(job: Optional[int]) -> None:
    if os.name != "nt" or job is None:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(wintypes.HANDLE(job))


class AvailableMemoryLimitError(RuntimeError):
    pass


class GtpProcessError(RuntimeError):
    pass


def available_memory_mib() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.virtual_memory().available / (1024.0 * 1024.0))


class ProcessManager:
    """Own every subprocess created by a tournament run."""

    def __init__(self, minimum_available_memory_mib: float):
        self.minimum_available_memory_mib = float(minimum_available_memory_mib)
        self.stop_event = threading.Event()
        self.low_memory_event = threading.Event()
        self._processes: set[subprocess.Popen] = set()
        self._windows_jobs: Dict[subprocess.Popen, Optional[int]] = {}
        self._lock = threading.Lock()

    def spawn(
        self,
        command: Sequence[str],
        env: Optional[dict] = None,
        stderr=None,
    ) -> subprocess.Popen:
        with self._lock:
            if self.stop_event.is_set():
                raise RuntimeError("process shutdown is in progress")
            available_mib = available_memory_mib()
            if (
                available_mib is not None
                and available_mib < self.minimum_available_memory_mib
            ):
                self.low_memory_event.set()
                raise AvailableMemoryLimitError(
                    f"available memory {available_mib:.0f} MiB is below "
                    f"the configured minimum "
                    f"{self.minimum_available_memory_mib:.0f} MiB"
                )
            popen_kwargs = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.DEVNULL if stderr is None else stderr,
                "text": True,
                "bufsize": 1,
            }
            if env is not None:
                popen_kwargs["env"] = env
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(list(command), **popen_kwargs)
            job = _create_windows_kill_job(process)
            if os.name == "nt" and job is None:
                try:
                    process.kill()
                    process.wait(timeout=FORCED_SHUTDOWN_SEC)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                for pipe in (process.stdin, process.stdout, process.stderr):
                    try:
                        if pipe is not None:
                            pipe.close()
                    except OSError:
                        pass
                raise RuntimeError(
                    "failed to assign subprocess to a kill-on-close "
                    "Windows Job Object"
                )
            self._processes.add(process)
            self._windows_jobs[process] = job
            return process

    def unregister(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)
            job = self._windows_jobs.pop(process, None)
        _close_windows_job(job)

    @staticmethod
    def _kill_tree(process: subprocess.Popen) -> None:
        if os.name == "nt":
            if process.poll() is not None:
                return
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=FORCED_SHUTDOWN_SEC,
                )
                if process.poll() is not None:
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            # The process-group leader may already have exited while one of
            # its children remains. killpg still reaches that orphan.
            try:
                os.killpg(process.pid, signal.SIGKILL)
                return
            except OSError:
                if process.poll() is not None:
                    return
        try:
            process.kill()
        except OSError:
            pass

    def terminate(
        self,
        process: subprocess.Popen,
        graceful: bool,
    ) -> None:
        try:
            if graceful and process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    process.wait(timeout=GRACEFUL_SHUTDOWN_SEC)
                except subprocess.TimeoutExpired:
                    self._kill_tree(process)
            elif process.poll() is None:
                self._kill_tree(process)
            if process.poll() is None:
                try:
                    process.wait(timeout=FORCED_SHUTDOWN_SEC)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        process.wait(timeout=FORCED_SHUTDOWN_SEC)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
        finally:
            for pipe in (process.stdin, process.stdout, process.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except OSError:
                    pass
            self.unregister(process)

    def close_all(self) -> None:
        """Send quit to all engines first, then use one shared deadline."""

        self.stop_event.set()
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
        deadline = time.monotonic() + GRACEFUL_SHUTDOWN_SEC
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
        for process in processes:
            if process.poll() is None:
                self._kill_tree(process)
        for process in processes:
            for pipe in (process.stdin, process.stdout, process.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except OSError:
                    pass
            self.unregister(process)


class GtpProcess:
    """One GTP subprocess with a deadline for every command.

    A failed stateful process is never restarted in the middle of a game. The
    exception escapes to the match-set scheduler, which retries both colors
    from the opening with fresh processes.
    """

    def __init__(
        self,
        manager: ProcessManager,
        command: Sequence[str],
        timeout_sec: float,
    ):
        self.manager = manager
        self.command_line = tuple(command)
        self.timeout_sec = float(timeout_sec)
        self.process = manager.spawn(command)
        self._output: queue.Queue[object] = queue.Queue()
        self._request_lock = threading.Lock()
        self._reader = threading.Thread(
            target=self._pump_stdout,
            name=f"gtp-stdout-{self.process.pid}",
            daemon=True,
        )
        self._reader.start()

    def _pump_stdout(self) -> None:
        try:
            if self.process.stdout is None:
                self._output.put(
                    GtpProcessError("GTP subprocess stdout is unavailable")
                )
                return
            for line in self.process.stdout:
                self._output.put(line)
            self._output.put(_EOF)
        except BaseException as error:
            self._output.put(error)

    @property
    def usable(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _fail(self) -> None:
        process = self.process
        if process is not None:
            self.manager.terminate(process, graceful=False)

    def request(self, command: str) -> str:
        with self._request_lock:
            if not self.usable:
                raise GtpProcessError("GTP subprocess is not running")
            process = self.process
            assert process.stdin is not None
            try:
                process.stdin.write(command.rstrip("\r\n") + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                self._fail()
                raise GtpProcessError(
                    f"failed to send GTP command {command!r}"
                ) from error

            deadline = time.monotonic() + self.timeout_sec
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._fail()
                    raise TimeoutError(
                        f"GTP command timed out after {self.timeout_sec:g}s: "
                        f"{command.rstrip()}"
                    )
                try:
                    response = self._output.get(timeout=remaining)
                except queue.Empty:
                    self._fail()
                    raise TimeoutError(
                        f"GTP command timed out after {self.timeout_sec:g}s: "
                        f"{command.rstrip()}"
                    )
                if response is _EOF:
                    self._fail()
                    raise GtpProcessError(
                        f"GTP subprocess ended while handling "
                        f"{command.rstrip()!r}"
                    )
                if isinstance(response, BaseException):
                    self._fail()
                    raise GtpProcessError(
                        f"failed to read the response to {command.rstrip()!r}"
                    ) from response
                line = str(response).strip()
                if line:
                    return line

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        self.process = None  # type: ignore[assignment]
        self.manager.terminate(process, graceful=True)
        self._reader.join(timeout=1.0)


class PlayerProcessPool:
    """Lazy process pool with separate black and white leases."""

    def __init__(
        self,
        spec: PlayerSpec,
        manager: ProcessManager,
        command_timeout_sec: float,
    ):
        self.spec = spec
        self.manager = manager
        self.command_timeout_sec = float(command_timeout_sec)
        self._limit_per_color = spec.processes_per_player // 2
        self._available = [queue.Queue(), queue.Queue()]
        self._created = [0, 0]
        self._lock = threading.Lock()

    def acquire(self, color_pool: int) -> GtpProcess:
        while True:
            if self.manager.stop_event.is_set():
                raise RuntimeError("process shutdown is in progress")
            try:
                engine = self._available[color_pool].get_nowait()
            except queue.Empty:
                engine = None
            if engine is not None:
                if engine.usable:
                    return engine
                engine.close()
                with self._lock:
                    self._created[color_pool] -= 1
                continue

            create_new = False
            with self._lock:
                if self._created[color_pool] < self._limit_per_color:
                    self._created[color_pool] += 1
                    create_new = True
            if create_new:
                try:
                    return GtpProcess(
                        self.manager,
                        self.spec.command,
                        self.command_timeout_sec,
                    )
                except BaseException:
                    with self._lock:
                        self._created[color_pool] -= 1
                    raise
            try:
                engine = self._available[color_pool].get(
                    timeout=PROCESS_POOL_WAIT_SEC
                )
            except queue.Empty:
                continue
            if engine.usable:
                return engine
            engine.close()
            with self._lock:
                self._created[color_pool] -= 1

    def release(self, color_pool: int, engine: GtpProcess) -> None:
        if engine.usable and not self.manager.stop_event.is_set():
            self._available[color_pool].put(engine)
            return
        engine.close()
        with self._lock:
            self._created[color_pool] = max(
                0,
                self._created[color_pool] - 1,
            )


def parse_gtp_response(line: str) -> str:
    line = line.strip()
    if line.startswith("?"):
        raise GtpProcessError(f"engine rejected a GTP command: {line}")
    if line.startswith("="):
        line = line[1:].strip()
    return line


def parse_gtp_move(line: str) -> str:
    response = parse_gtp_response(line)
    if not response:
        return "pass"
    return response.split()[-1].lower()


def count_bits(value: int) -> int:
    # Python 3.9 is still used by the experiment machine.
    return bin(int(value)).count("1")


def disc_diff_for_p0(state: BoardState, p0_is_black: bool) -> int:
    black_count = count_bits(state.black)
    white_count = count_bits(state.white)
    empty_count = 64 - black_count - white_count
    black_diff = black_count - white_count
    if black_diff > 0:
        black_diff += empty_count
    elif black_diff < 0:
        black_diff -= empty_count
    return black_diff if p0_is_black else -black_diff


def validate_xot_opening(opening: str) -> None:
    """Replay one opening and reject malformed or illegal transcripts."""

    if len(opening) % 2:
        raise ValueError(f"odd-length XOT opening: {opening!r}")
    state = BoardState.initial()
    for offset in range(0, len(opening), 2):
        if not state.legal_policies(state.side):
            state.side ^= 1
            if not state.legal_policies(state.side):
                raise ValueError(
                    "XOT opening continues after the game has ended"
                )
        move = opening[offset : offset + 2].lower()
        policy = coord_to_policy(move)
        if policy not in state.legal_policies(state.side):
            raise ValueError(
                f"illegal XOT move {move!r} in {opening!r}"
            )
        state.apply_move(state.side, policy)


class GameRunner:
    def __init__(
        self,
        specs: Sequence[PlayerSpec],
        manager: ProcessManager,
        command_timeout_sec: float,
    ):
        self.specs = list(specs)
        self.pools = [
            PlayerProcessPool(spec, manager, command_timeout_sec)
            for spec in specs
        ]

    @staticmethod
    def _send(engine: GtpProcess, command: str) -> str:
        return parse_gtp_response(engine.request(command))

    def _generate_move(
        self,
        player_idx: int,
        held_engine: Optional[GtpProcess],
        color_pool: int,
        state: BoardState,
        side: int,
    ) -> str:
        spec = self.specs[player_idx]
        if not spec.setboard_before_genmove:
            if held_engine is None:
                raise RuntimeError(f"no stateful process held for {spec.name}")
            return parse_gtp_move(
                held_engine.request(f"genmove {side_to_gtp_color(side)}")
            )

        engine = self.pools[player_idx].acquire(color_pool)
        try:
            self._send(
                engine,
                f"setboard {state.to_egaroucid_board(side)}",
            )
            return parse_gtp_move(
                engine.request(f"genmove {side_to_gtp_color(side)}")
            )
        finally:
            self.pools[player_idx].release(color_pool, engine)

    def play_single_game(
        self,
        task: MatchSetTask,
        p0_is_black: bool,
    ) -> GameResult:
        p0_idx = task.p0_idx
        p1_idx = task.p1_idx
        black_idx = p0_idx if p0_is_black else p1_idx
        white_idx = p1_idx if p0_is_black else p0_idx
        color_pools = {
            p0_idx: 0 if p0_is_black else 1,
            p1_idx: 1 if p0_is_black else 0,
        }
        held: Dict[int, GtpProcess] = {}
        state = BoardState.initial()
        transcript = ""

        def send_play_to_stateful(side: int, move: str) -> None:
            command = f"play {side_to_gtp_color(side)} {move}"
            for engine in held.values():
                self._send(engine, command)

        try:
            # A global participant-index order prevents resource cycles when
            # several stateful games begin at the same time.
            for player_idx in sorted((p0_idx, p1_idx)):
                if self.specs[player_idx].setboard_before_genmove:
                    continue
                engine = self.pools[player_idx].acquire(
                    color_pools[player_idx]
                )
                held[player_idx] = engine
                self._send(engine, "clear_board")

            if len(task.opening) % 2:
                raise ValueError(f"odd-length XOT opening: {task.opening!r}")
            for offset in range(0, len(task.opening), 2):
                if not state.legal_policies(state.side):
                    send_play_to_stateful(state.side, "pass")
                    state.side ^= 1
                    if not state.legal_policies(state.side):
                        raise ValueError(
                            "XOT opening continues after the game has ended"
                        )
                move = task.opening[offset : offset + 2].lower()
                policy = coord_to_policy(move)
                if policy not in state.legal_policies(state.side):
                    raise ValueError(
                        f"illegal XOT move {move!r} in {task.opening!r}"
                    )
                send_play_to_stateful(state.side, move)
                state.apply_move(state.side, policy)
                transcript += move

            while True:
                legal = state.legal_policies(state.side)
                if not legal:
                    send_play_to_stateful(state.side, "pass")
                    state.side ^= 1
                    legal = state.legal_policies(state.side)
                    if not legal:
                        break

                side = state.side
                mover = black_idx if side == BLACK else white_idx
                watcher = white_idx if side == BLACK else black_idx
                move = self._generate_move(
                    mover,
                    held.get(mover),
                    color_pools[mover],
                    state,
                    side,
                )
                if move == "pass":
                    if legal:
                        raise GtpProcessError(
                            f"{self.specs[mover].name} passed with legal moves"
                        )
                    state.side ^= 1
                    watcher_engine = held.get(watcher)
                    if watcher_engine is not None:
                        self._send(
                            watcher_engine,
                            f"play {side_to_gtp_color(side)} pass",
                        )
                    continue

                policy = coord_to_policy(move)
                if policy not in legal:
                    raise GtpProcessError(
                        f"{self.specs[mover].name} generated illegal move "
                        f"{move!r}"
                    )
                state.apply_move(side, policy)
                transcript += move
                watcher_engine = held.get(watcher)
                if watcher_engine is not None:
                    self._send(
                        watcher_engine,
                        f"play {side_to_gtp_color(side)} {move}",
                    )

            return GameResult(
                p0_idx=p0_idx,
                p1_idx=p1_idx,
                p0_is_black=p0_is_black,
                black_idx=black_idx,
                white_idx=white_idx,
                p0_disc_diff=disc_diff_for_p0(state, p0_is_black),
                black_stones=count_bits(state.black),
                white_stones=count_bits(state.white),
                transcript=transcript,
            )
        finally:
            for player_idx, engine in held.items():
                self.pools[player_idx].release(
                    color_pools[player_idx],
                    engine,
                )
