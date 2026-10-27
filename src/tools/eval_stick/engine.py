"""Strictly sequential helper protocol and Console -solve execution."""
from pathlib import Path
import json
import ctypes
import os
import queue
import re
import subprocess
import threading
import time
import uuid

from common import EngineLock, append_json, digest, kill_and_reap, stamp, write_json


class OwnedJob:
    """Windows kills the owned child even if its Python parent is terminated."""
    def __init__(self, process):
        self.handle = None
        if os.name != "nt":
            return
        size = ctypes.c_size_t

        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", ctypes.c_ulong), ("min_working", size), ("max_working", size),
                        ("active_limit", ctypes.c_ulong), ("affinity", size),
                        ("priority", ctypes.c_ulong), ("scheduling", ctypes.c_ulong)]

        class Counters(ctypes.Structure):
            _fields_ = [("counter" + str(i), ctypes.c_ulonglong) for i in range(6)]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", Counters), ("process_memory", size),
                        ("job_memory", size), ("peak_process_memory", size), ("peak_job_memory", size)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.api.CreateJobObjectW.restype = ctypes.c_void_p
        self.api.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        self.api.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.api.CloseHandle.argtypes = [ctypes.c_void_p]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended(); info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            code = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(code)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def deadline_watchdog(process):
    """Cover allocation/startup too, independently of any protocol wait."""
    deadline = os.environ.get("EGAROUCID_EVAL_STICK_DEADLINE")
    if deadline is None:
        return None

    def stop():
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass  # the owned process may have just exited

    timer = threading.Timer(max(0, float(deadline) - time.time()), stop)
    timer.daemon = True
    timer.start()
    return timer


class Helper:
    def __init__(self, engine, resources, data, threads=28):
        self.engine, self.resources, self.data = Path(engine), Path(resources), Path(data)
        self.threads = threads
        self.tag = 0
        self.process = None

    def __enter__(self):
        self.lock = EngineLock().__enter__()
        raw = self.data / "raw" / ("helper_" + uuid.uuid4().hex + ".log")
        raw.parent.mkdir(parents=True, exist_ok=True)
        self.log = raw.open("w", encoding="utf-8")
        self.raw = str(raw)
        self.command = [str(self.engine), str(self.threads), str(self.resources)]
        self.sha256 = digest(self.engine)
        self.process = subprocess.Popen(self.command, cwd=self.resources.parent, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        try:
            self.owned_job = OwnedJob(self.process)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.watchdog = deadline_watchdog(self.process)
        self.queue = queue.Queue()

        def reader():
            for line in self.process.stdout:
                self.queue.put(line.strip())
            self.queue.put(None)

        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        try:
            if self.queue.get(timeout=120) != "READY":
                raise RuntimeError("helper failed to initialize: " + self.raw)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def request(self, operation, timeout=3600, metadata=None):
        self.tag += 1
        rec = {"tag": self.tag, "operation": operation, "started": stamp(),
               "pid": self.process.pid, "parent_pid": os.getpid(), "engine_sha256": self.sha256,
               "raw": self.raw, **(metadata or {})}
        write_json(self.data / "active.json", rec)
        self.log.write("\n# request " + json.dumps(rec) + "\n"); self.log.flush()
        start = time.monotonic()
        next_progress = start + 60
        next_note = start + 300
        self.process.stdin.write(str(self.tag) + " " + operation + "\n")
        self.process.stdin.flush()
        try:
            while True:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("helper search time limit")
                if (self.data / "stop_requested.txt").exists():
                    raise InterruptedError("stop requested")
                try:
                    line = self.queue.get(timeout=min(1, remaining))
                except queue.Empty:
                    now = time.monotonic()
                    if now >= next_progress:
                        elapsed = now - start
                        write_json(self.data / "active.json", {**rec, "elapsed_seconds": elapsed, "updated": stamp()})
                        print(stamp(), "ACTIVE", rec.get("phase", "search"), rec.get("position"),
                              rec.get("requested_move"), f"{elapsed:.0f}s", flush=True)
                        next_progress = now + 60
                        if now >= next_note:
                            with (self.data / "progress.md").open("a", encoding="utf-8") as progress:
                                progress.write(f"\n- {stamp()}: 探索継続。工程{rec.get('phase', 'search')}、局面{rec.get('position')}、手{rec.get('requested_move')}、探索経過{elapsed:.0f}秒、エンジンPID {self.process.pid}。\n")
                            next_note = now + 300
                    continue
                if line is None:
                    raise RuntimeError("helper exited: " + self.raw)
                result = json.loads(line)
                if result["tag"] != self.tag:
                    raise RuntimeError("helper protocol tag mismatch")
                rec.update(result, status="ok", wall_seconds=time.monotonic() - start, finished=stamp())
                append_json(self.data / "operations.jsonl", rec)
                write_json(self.data / "active.json", {"idle": True, "pid": self.process.pid, "last": rec})
                return rec
        except BaseException as exc:
            rec.update(status="timeout" if isinstance(exc, TimeoutError) else "interrupted" if isinstance(exc, InterruptedError) else "failed",
                       error=str(exc), wall_seconds=time.monotonic() - start, finished=stamp())
            append_json(self.data / "operations.jsonl", rec)
            kill_and_reap(self.process)
            raise

    def clear(self):
        return self.request("clear", timeout=120)

    def root(self, pos, move, depth, probability, timeout=3600, **metadata):
        if self.process.poll() is not None:
            raise RuntimeError("helper has exited")
        result = self.request(f"root {depth} {probability} {pos['board']} {pos['turn']} {move}", timeout,
                              {"position": pos.get("id"), "requested_move": move, **metadata})
        if result["depth"] != depth or result["probability"] != probability or (move != "all" and result["move"] != move):
            raise RuntimeError("wrong helper result")
        return result

    def __exit__(self, *args):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    kill_and_reap(self.process)
            self.process.stdout.close()
        if getattr(self, "watchdog", None) is not None:
            self.watchdog.cancel()
        if hasattr(self, "owned_job"):
            self.owned_job.close()
        if hasattr(self, "log"):
            self.log.close()
        if hasattr(self, "lock"):
            self.lock.__exit__(*args)


def parse_solve(text):
    rows = []
    for line in text.splitlines():
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) != 7 or not re.fullmatch(r"[a-h][1-8]|pa", cols[2].lower()):
            continue
        hh, mm, ss = cols[4].split(":")
        depth = re.fullmatch(r"(\d+)(?:@([\d.]+)%)?", cols[1])
        if not depth:
            raise ValueError("unrecognized depth " + cols[1])
        rows.append({"depth_text": cols[1], "depth": int(depth[1]),
                     "probability": float(depth[2]) if depth[2] else 100,
                     "move": cols[2].lower(), "score": int(cols[3]),
                     "time_seconds": int(hh) * 3600 + int(mm) * 60 + float(ss),
                     "nodes": int(cols[5])})
    return rows


def solve(engine, options, positions, out, timeout=3600, active=None):
    """Each input is cold: production solve_problems calls TT.init per line."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    problem = out / "problems.txt"
    expected = "".join(p["board"] + " " + p["turn"] + "\n" for p in positions)
    if problem.exists() and problem.read_text(encoding="utf-8") != expected:
        raise RuntimeError("refusing to overwrite a different problem list")
    problem.write_text(expected, encoding="utf-8")
    command = [str(Path(engine).resolve()), "-nobook", "-quiet"] + list(options) + ["-solve", str(problem.resolve())]
    previous = json.loads((out / "run.json").read_text(encoding="utf-8")) if (out / "run.json").exists() else None
    if previous and previous.get("status") == "ok":
        if previous["command"] != command or previous["engine_sha256"] != digest(engine) or previous["problem_sha256"] != digest(problem):
            raise RuntimeError("completed solve batch has different inputs")
        return previous
    raw = out / ("raw_" + uuid.uuid4().hex + ".txt")
    rec = {"command": command, "engine_sha256": digest(engine), "problem_sha256": digest(problem),
           "started": stamp(), "raw": str(raw), "positions": [p["id"] for p in positions]}
    with EngineLock(), raw.open("wb") as log:
        process = subprocess.Popen(command, cwd=Path(engine).resolve().parent, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            owned_job = OwnedJob(process)
        except BaseException:
            kill_and_reap(process)
            raise
        watchdog = deadline_watchdog(process)
        rec["pid"] = process.pid
        if active:
            write_json(active, rec)
        begin = time.monotonic()
        try:
            process.wait(timeout=timeout)
            rec["status"] = "ok" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            rec["status"] = "timeout"
        finally:
            kill_and_reap(process)
            owned_job.close()
            if watchdog is not None:
                watchdog.cancel()
        rec.update(exit=process.returncode, wall_seconds=time.monotonic() - begin, finished=stamp())
    output = raw.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    rec["rows"] = parse_solve(output)
    if "[ERROR]" in output or "[FATAL]" in output:
        rec["status"] = "engine_error"  # also reject an existing silent TT-size downgrade
    if len(rec["rows"]) != len(positions):
        rec["status"] = "incomplete"
    write_json(out / "run.json", rec)
    if rec["status"] != "ok":
        raise RuntimeError("solve failed; raw output: " + str(raw))
    return rec
