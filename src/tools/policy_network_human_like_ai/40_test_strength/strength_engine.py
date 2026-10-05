#!/usr/bin/env python3
"""Timeout-safe GTP process pools and one-game execution."""

from __future__ import annotations

import os
import hashlib
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
                    # A live wrapper can report that its nested search timed
                    # out. Retire that entire process tree before returning
                    # the error, so the next lease cannot reuse broken state.
                    if line.startswith("?"):
                        self._fail()
                    return line

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        self.process = None  # type: ignore[assignment]
        self.manager.terminate(process, graceful=True)
        self._reader.join(timeout=1.0)


class EdaxXBoardProcess(GtpProcess):
    """GTP-facing adapter for Edax's TT-preserving XBoard `cores` command."""

    def __init__(self, manager, command, timeout_sec):
        command = list(command)
        command[command.index("-gtp")] = "-xboard"
        self.level = int(command[command.index("-l") + 1])
        self._ping_id = 0
        super().__init__(manager, command, timeout_sec)
        try:
            self._exchange(["force", "easy", "nopost", f"sd {self.level}"])
        except BaseException:
            self.close()
            raise

    def _exchange(self, commands):
        # Ping is processed after the preceding commands, including a search.
        # Keep all lines through pong so terminal results cannot leak into the
        # next request. Never send a stop/force command while search is running.
        with self._request_lock:
            if not self.usable:
                raise GtpProcessError("Edax subprocess is not running")
            self._ping_id += 1
            marker = f"pong {self._ping_id}"
            try:
                self.process.stdin.write("\n".join([*commands, f"ping {self._ping_id}", ""]))
                self.process.stdin.flush()
                deadline = time.monotonic() + self.timeout_sec
                lines = []
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    response = self._output.get(timeout=remaining)
                    if response is _EOF or isinstance(response, BaseException):
                        raise GtpProcessError("Edax subprocess ended during XBoard command")
                    line = str(response).strip()
                    if line == marker:
                        return lines
                    if line.startswith(("Illegal move", "Error")):
                        raise GtpProcessError(line)
                    if line:
                        lines.append(line)
            except queue.Empty as error:
                self._fail()
                raise TimeoutError(f"Edax XBoard command timed out: {commands!r}") from error
            except BaseException:
                self._fail()
                raise

    def set_search_threads(self, threads):
        if not 1 <= threads <= 64:
            raise ValueError("Edax runtime threads must be between 1 and 64")
        # Edax 4.6 search_set_task_number resizes worker tasks, not hash tables.
        # No new/setboard/memory command is sent here.
        self._exchange([f"cores {threads}"])

    def request(self, command):
        if command == "clear_board":
            # XBoard 'new' resets the level to 60; restore the tournament level.
            self._exchange(["new", "force", "easy", "nopost", f"sd {self.level}"])
            return "="
        action, color, *moves = command.split()
        if action == "play":
            move = "@@@@" if moves[0] == "pass" else "P@" + moves[0]
            self._exchange([f"usermove {move}"])
            return "="
        if action == "genmove":
            lines = self._exchange(["go"])
            self._exchange(["force"])
            moves = [line.split()[1] for line in lines if line.startswith("move ")]
            if len(moves) != 1:
                self._fail()
                raise GtpProcessError(f"Expected one Edax move, received {lines!r}")
            return "= " + ("pass" if moves[0] == "@@@@" else moves[0].removeprefix("P@").lower())
        raise GtpProcessError(f"Unsupported Edax adapter command: {command}")


class PlayerProcessPool:
    """Lazy process pool with separate black and white leases."""

    def __init__(
        self,
        spec: PlayerSpec,
        manager: ProcessManager,
        command_timeout_sec: float,
        *,
        live_edax_threads: bool = False,
    ):
        self.spec = spec
        self.manager = manager
        self.command_timeout_sec = float(command_timeout_sec)
        self._limit_per_color = spec.processes_per_player // 2
        self._available = [queue.Queue(), queue.Queue()]
        self._created = [0, 0]
        self._lock = threading.Lock()
        self._retired = False
        self.live_edax_threads = live_edax_threads and spec.name.startswith("edax_")
        self._command = spec.command
        self._revision = 0
        self._live_threads: Dict[GtpProcess, int] = {}
        self._leased_engines: set[GtpProcess] = set()
        self._thread_option = (
            None if spec.alpha == 1.0 else
            "--egaroucid-threads" if spec.setboard_before_genmove else
            "-n" if spec.name.startswith("edax_") else "-t"
        )
        if self._thread_option not in self._command:
            self._thread_option = None
        self.search_threads = (
            int(self._command[self._command.index(self._thread_option) + 1])
            if self._thread_option is not None else 1
        )

    @property
    def reserved_search_threads(self) -> int:
        with self._lock:
            return max([self.search_threads, *self._live_threads.values()])

    @property
    def pending_thread_change(self) -> bool:
        with self._lock:
            return any(self._live_threads[engine] != self.search_threads
                       and getattr(engine, "game_thread_override", False) is not True
                       for engine in self._leased_engines)

    def search_thread_status(self):
        with self._lock:
            counts = {}
            for threads in self._live_threads.values():
                if self.spec.alpha != 1.0:
                    counts[str(threads)] = counts.get(str(threads), 0) + 1
            pending = [self._live_threads[engine] for engine in self._leased_engines
                       if self._live_threads[engine] != self.search_threads
                       and getattr(engine, "game_thread_override", False) is not True]
            return {"actual_search_threads": counts,
                    "active_search_threads": (sum(self._live_threads[e] for e in self._leased_engines)
                                              if self.spec.alpha != 1.0 else 0),
                    "idle_extra_search_threads": sum(max(0, threads - self.search_threads)
                        for engine, threads in self._live_threads.items() if engine not in self._leased_engines),
                    "pending_thread_processes": len(pending),
                    "pending_extra_search_threads": sum(max(0, self.search_threads - threads)
                                                        for threads in pending)}

    def set_search_threads(self, threads: int) -> bool:
        """Request new search concurrency for engines leased by the next game."""
        if threads < 1:
            raise ValueError("search threads must be positive")
        with self._lock:
            if self._retired or self._thread_option is None or threads == self.search_threads:
                return False
            command = list(self._command)
            command[command.index(self._thread_option) + 1] = str(threads)
            self._command = tuple(command)
            self.search_threads = threads
            self._revision += 1
            return True

    def _is_current(self, engine: GtpProcess) -> bool:
        return engine.usable and engine.pool_revision == self._revision

    def resize_leased_engine(self, engine, threads):
        """Called by the owning game worker, only between completed moves."""
        with self._lock:
            if not self.live_edax_threads or engine not in self._leased_engines:
                raise RuntimeError("engine does not support a live thread change")
        engine.set_search_threads(threads)
        with self._lock:
            self._live_threads[engine] = threads

    def _lease(self, engine: GtpProcess) -> GtpProcess:
        with self._lock:
            self._leased_engines.add(engine)
        return engine

    def _discard(self, color_pool: int, engine: GtpProcess) -> None:
        engine.close()
        with self._lock:
            self._live_threads.pop(engine, None)
            self._leased_engines.discard(engine)
            self._created[color_pool] = max(0, self._created[color_pool] - 1)

    @property
    def created_processes(self) -> int:
        with self._lock:
            return sum(self._created)

    def process_reservation(self, games_by_color):
        """Count retained engines and admitted color games not yet started."""
        with self._lock:
            return (sum(max(count, games) for count, games in zip(self._created, games_by_color)),
                    sum(max(0, games - count) for count, games in zip(self._created, games_by_color)))

    def close_idle(self) -> int:
        """Reclaim only queued engines; never touch a game's leased engines."""
        closed = 0
        for color, items in enumerate(self._available):
            while True:
                try:
                    engine = items.get_nowait()
                except queue.Empty:
                    break
                self._discard(color, engine)
                closed += 1
        return closed

    def grow(self, set_capacity: int) -> None:
        with self._lock:
            if self._retired or set_capacity < self._limit_per_color:
                raise ValueError("cannot grow a retired pool or reduce its capacity")
            self._limit_per_color = set_capacity

    def retire(self) -> int:
        """Close an idle participant's engines, including wrapper children."""
        with self._lock:
            if self._retired:
                return 0
            if sum(self._created) != sum(items.qsize() for items in self._available):
                raise RuntimeError("cannot retire a player with leased engines")
            self._retired = True
            self._limit_per_color = 0
            engines = []
            for items in self._available:
                while not items.empty():
                    engines.append(items.get_nowait())
            self._created = [0, 0]
            self._live_threads.clear()
            self._leased_engines.clear()
        for engine in engines:
            engine.close()
        return len(engines)

    def acquire(self, color_pool: int, search_threads: Optional[int] = None) -> GtpProcess:
        if search_threads is not None and (search_threads < 1 or self._thread_option is None):
            raise ValueError("invalid search thread allocation")

        def matches(engine):
            if search_threads is None:
                return self._is_current(engine)
            return engine.usable and self._live_threads[engine] == search_threads

        while True:
            if self._retired:
                raise RuntimeError("player process pool is retired")
            if self.manager.stop_event.is_set():
                raise RuntimeError("process shutdown is in progress")
            try:
                engine = self._available[color_pool].get_nowait()
            except queue.Empty:
                engine = None
            if engine is not None:
                if matches(engine):
                    engine.game_thread_override = search_threads is not None
                    return self._lease(engine)
                self._discard(color_pool, engine)
                continue

            create_new = False
            with self._lock:
                if self._created[color_pool] < self._limit_per_color:
                    self._created[color_pool] += 1
                    create_new = True
                    command, revision, threads = self._command, self._revision, self.search_threads
                    if search_threads is not None:
                        command = list(command)
                        command[command.index(self._thread_option) + 1] = str(search_threads)
                        command, threads = tuple(command), search_threads
            if create_new:
                try:
                    process_type = EdaxXBoardProcess if self.live_edax_threads else GtpProcess
                    engine = process_type(
                        self.manager,
                        command,
                        self.command_timeout_sec,
                    )
                    engine.pool_revision = revision
                    engine.game_thread_override = search_threads is not None
                    with self._lock:
                        self._live_threads[engine] = threads
                    return self._lease(engine)
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
            if matches(engine):
                engine.game_thread_override = search_threads is not None
                return self._lease(engine)
            self._discard(color_pool, engine)

    def release(self, color_pool: int, engine: GtpProcess) -> None:
        with self._lock:
            self._leased_engines.discard(engine)
        if self._is_current(engine) and not self.manager.stop_event.is_set():
            self._available[color_pool].put(engine)
            return
        self._discard(color_pool, engine)


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
        *,
        game_boundary_threads: bool = False,
        sequential_fresh_tt: bool = False,
    ):
        self.specs = list(specs)
        self.game_boundary_threads = game_boundary_threads
        self.sequential_fresh_tt = sequential_fresh_tt
        self.thread_budget = None
        self.pools = [
            PlayerProcessPool(spec, manager, command_timeout_sec,
                              live_edax_threads=game_boundary_threads)
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

        engine = held_engine if held_engine is not None else self.pools[player_idx].acquire(color_pool)
        try:
            self._send(
                engine,
                f"setboard {state.to_egaroucid_board(side)}",
            )
            return parse_gtp_move(
                engine.request(f"genmove {side_to_gtp_color(side)}")
            )
        except Exception as error:
            raise GtpProcessError(
                f"{spec.name} failed on board "
                f"{state.to_egaroucid_board(side)!r}: {error}"
            ) from error
        finally:
            if held_engine is None:
                self.pools[player_idx].release(color_pool, engine)

    def play_single_game(
        self,
        task: MatchSetTask,
        p0_is_black: bool,
        search_threads: Optional[Dict[int, int]] = None,
    ) -> GameResult:
        p0_idx = task.p0_idx
        p1_idx = task.p1_idx
        black_idx = p0_idx if p0_is_black else p1_idx
        white_idx = p1_idx if p0_is_black else p0_idx
        color_pools = {
            p0_idx: 0 if p0_is_black else 1,
            p1_idx: 1 if p0_is_black else 0,
        }
        if self.sequential_fresh_tt:
            # Only one game can lease these pools. Reuse the same process even
            # when its color changes, after the acknowledged TT reset below.
            color_pools = {p0_idx: 0, p1_idx: 0}
        held: Dict[int, GtpProcess] = {}
        leased: Dict[int, GtpProcess] = {}
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
                stateful = not self.specs[player_idx].setboard_before_genmove
                if not stateful and not self.game_boundary_threads:
                    continue
                if (self.sequential_fresh_tt and stateful and self.specs[player_idx].alpha != 1.0
                        and not self.pools[player_idx].live_edax_threads):
                    # The native Egaroucid GTP interface has no TT-reset command
                    # (clear_board only changes the board). Only this interface
                    # needs a fresh process; hint wrappers expose clearcache.
                    self.pools[player_idx].close_idle()
                engine = self.pools[player_idx].acquire(
                    color_pools[player_idx],
                    search_threads=(search_threads or {}).get(player_idx),
                )
                # Pin blend wrappers as well when adapting threads. Returning
                # them after each move would reset their nested engine/TT midgame.
                leased[player_idx] = engine
                if self.sequential_fresh_tt and not stateful and self.specs[player_idx].alpha != 1.0:
                    self._send(engine, "clear_cache")
                    print(f"game_tt_reset player={self.specs[player_idx].name} "
                          f"pid={engine.process.pid} method=clearcache", flush=True)
                if stateful:
                    held[player_idx] = engine
                    if self.specs[player_idx].random_seed is not None:
                        # Pair/set identity survives appended participants and task-ID remapping.
                        identity = (self.specs[player_idx].random_seed, p0_idx, p1_idx,
                                    task.set_index, p0_is_black)
                        seed = int.from_bytes(hashlib.sha256(repr(identity).encode("ascii")).digest()[:8], "big")
                        self._send(engine, f"set_game_seed {seed}")
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
                if self.thread_budget is not None:
                    for player, engine in leased.items():
                        if self.pools[player].live_edax_threads:
                            self.thread_budget.resize_at_move_boundary(
                                task, p0_is_black, player,
                                lambda threads, p=player, e=engine:
                                    self.pools[p].resize_leased_engine(e, threads))
                legal = state.legal_policies(state.side)
                if not legal:
                    # Edax rejects a pass after both players have no moves.
                    # Detect game over before sending any further commands.
                    if not state.legal_policies(state.side ^ 1):
                        break
                    send_play_to_stateful(state.side, "pass")
                    state.side ^= 1
                    legal = state.legal_policies(state.side)

                side = state.side
                mover = black_idx if side == BLACK else white_idx
                watcher = white_idx if side == BLACK else black_idx
                move = self._generate_move(
                    mover,
                    leased.get(mover),
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
            for player_idx, engine in leased.items():
                self.pools[player_idx].release(
                    color_pools[player_idx],
                    engine,
                )
