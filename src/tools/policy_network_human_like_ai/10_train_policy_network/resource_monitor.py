#!/usr/bin/env python3
"""Small subprocess resource monitor for experiment logs."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import datetime as dt
from pathlib import Path
import subprocess
import time
from typing import Optional, Sequence


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _rss_bytes_psutil(pid: int) -> Optional[int]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        proc = psutil.Process(pid)
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        return int(rss)
    except psutil.Error:
        return None


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _rss_bytes_windows(pid: int) -> Optional[int]:
    if not hasattr(ctypes, "windll"):
        return None
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid))
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return int(counters.WorkingSetSize)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def rss_bytes(pid: int) -> Optional[int]:
    value = _rss_bytes_psutil(pid)
    if value is not None:
        return value
    return _rss_bytes_windows(pid)


def mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def command_text(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def run_monitored_command(
    command: Sequence[str],
    log_path: Path,
    poll_interval_sec: float = 1.0,
    cwd: Optional[Path] = None,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    start_iso = now_iso()
    max_main_memory: Optional[int] = None
    n_samples = 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + command_text(command) + "\n")
        log.write(f"resource_monitor_start {start_iso}\n\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=str(cwd) if cwd is not None else None, stdout=log, stderr=subprocess.STDOUT)
        while True:
            current = rss_bytes(proc.pid)
            if current is not None:
                max_main_memory = current if max_main_memory is None else max(max_main_memory, current)
                n_samples += 1
            if proc.poll() is not None:
                break
            time.sleep(max(0.1, poll_interval_sec))
        returncode = proc.wait()
        elapsed = time.time() - start
        end_iso = now_iso()
        log.write("\n")
        log.write(f"resource_monitor_end {end_iso}\n")
        log.write(f"resource_elapsed_sec {elapsed:.3f}\n")
        if max_main_memory is None:
            log.write("resource_max_main_memory_mib unknown\n")
        else:
            log.write(f"resource_max_main_memory_mib {mib(max_main_memory):.3f}\n")
        log.write(f"resource_samples {n_samples}\n")
        log.write(f"resource_returncode {returncode}\n")
    return {
        "command": command_text(command),
        "returncode": returncode,
        "start_time": start_iso,
        "end_time": end_iso,
        "elapsed_sec": elapsed,
        "max_main_memory_bytes": max_main_memory,
        "max_main_memory_mib": mib(max_main_memory),
        "resource_samples": n_samples,
        "log": str(log_path),
    }
