"""Run both hint pipelines under one shared limit on engine processes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import sys
import threading
import time

from wthor_hint_pipeline import collect_hints, format_duration


class ComparisonProgressReporter:
    """Print one block after every running engine has supplied a new snapshot.

    Format metrics on their collector's thread so row updates cannot race a
    metric snapshot. Completed engines keep their final rows in later blocks.
    """

    def __init__(self, engines, metric_formatters=None):
        self.engines = tuple(engines)
        self.metric_formatters = metric_formatters or {}
        self.snapshots = {}
        self.metric_lines = {}
        self.updated = set()
        self.lock = threading.Lock()

    def report(self, snapshot):
        engine = snapshot["engine"]
        formatter = self.metric_formatters.get(engine)
        metric_lines = formatter(snapshot) if formatter else []
        with self.lock:
            self.snapshots[engine] = snapshot
            self.metric_lines[engine] = metric_lines
            self.updated.add(engine)
            if any(name not in self.snapshots for name in self.engines):
                return
            if any(name not in self.updated and not self.snapshots[name]["final"]
                   for name in self.engines):
                return
            sys.stderr.write("\n".join(self.format_lines()) + "\n")
            sys.stderr.flush()
            self.updated.clear()

    def format_lines(self):
        snapshots = [self.snapshots[name] for name in self.engines]
        completed = sum(row["completed_states"] for row in snapshots)
        total = sum(row["total_states"] for row in snapshots)
        elapsed = max(row["elapsed_sec"] for row in snapshots)
        rate = completed / elapsed if elapsed else 0.0
        estimates = [row["eta_sec"] for row in snapshots]
        eta = (format_duration(max(estimates))
               if all(value is not None for value in estimates) else "算出待ち")
        final = all(row["final"] for row in snapshots)
        kind = "hint進捗・完了" if final else "hint進捗"
        lines = [
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][{kind}] "
            f"{completed:,}/{total:,}件のhint計算 "
            f"({completed / total if total else 1.0:.1%})"
            f" | 経過 {format_duration(elapsed)} | 概算残り {eta}"
            f" | {rate:.2f}件/秒"
        ]
        for engine in self.engines:
            row = self.snapshots[engine]
            label = "Edax" if engine == "edax" else "Egaroucid for Console"
            levels = ", ".join(
                f"level {level}:{count}"
                for level, count in sorted(row["active_tasks_by_level"].items(), reverse=True)
            )
            since = row["seconds_since_last_hint"]
            status = ("完了" if row["final"] else
                      f"最終結果 {format_duration(since)}前" if since is not None else
                      "初回結果待ち")
            lines.append(
                f"  [{label}] {row['completed_states']:,}/{row['total_states']:,}件"
                f" | プロセス完了 {row['completed_tasks']}"
                f" | 起動 {row['submitted_tasks']} | 稼働 {row['active_tasks']}"
                + (f" | {levels}" if levels else "")
                + f" | cache検出 {row['cache_hits']:,} | 再試行 {row['retry_count']} | {status}"
            )
        # Both monitors sample the same machine and process tree; show it once.
        cpu = next((row["cpu"] for row in sorted(
            snapshots, key=lambda row: row["elapsed_sec"], reverse=True,
        ) if row["cpu"].get("available")), None)
        if cpu is not None:
            lines.append(
                f"  [資源] CPU {cpu['cpu_percent']:.0f}%"
                f" | 空きRAM {cpu['available_memory_mib']:.0f} MiB"
                f" | 関連メモリ {cpu['process_tree_memory_mib'] / 1024.0:.1f} GiB"
                f" | Egaroucid {cpu['egaroucid_processes']} | Edax {cpu['edax_processes']}"
            )
        for engine in self.engines:
            lines.extend(self.metric_lines[engine])
        return lines


class SharedEngineSlots:
    """Reserve startup slots for both engines, then reuse freed capacity."""

    def __init__(self, workers, limits, startup_slots):
        self.workers = workers
        self.limits = dict(limits)
        self.reserved = dict(startup_slots)
        if sum(self.reserved.values()) > workers:
            raise ValueError("not enough workers for the engine startup reservations")
        self.active = {engine: 0 for engine in limits}
        self.lock = threading.Lock()

    def acquire(self, engine):
        with self.lock:
            free = self.workers - sum(self.active.values())
            other_reservations = sum(
                reserved
                for other, reserved in self.reserved.items() if other != engine
            )
            if self.active[engine] >= self.limits[engine] or free <= other_reservations:
                return False
            self.active[engine] += 1
            self.reserved[engine] = max(0, self.reserved[engine] - 1)
            return True

    def release(self, engine):
        with self.lock:
            if self.active[engine] <= 0:
                raise RuntimeError(f"no active slot to release for {engine}")
            self.active[engine] -= 1

    def finish(self, engine):
        with self.lock:
            self.reserved[engine] = 0


def collect_comparison_hints(groups, workers, engine_options, metric_formatters=None):
    """Collect both engines concurrently; errors/interrupts cancel both pools."""
    cancellation = multiprocessing.get_context("spawn").Event()
    progress_lock = threading.Lock()
    reporter = ComparisonProgressReporter(engine_options, metric_formatters)
    slots = SharedEngineSlots(
        workers,
        {name: options["workers"] for name, options in engine_options.items()},
        {name: min(len(options["levels"]), options["workers"])
         for name, options in engine_options.items()},
    )

    def run(engine, options):
        start = time.perf_counter()
        try:
            result = collect_hints(
                groups, **options, engine=engine,
                cancellation_event=cancellation, progress_lock=progress_lock,
                shared_slots=slots,
                progress_callback=reporter.report, print_progress=False,
            )
            return result, time.perf_counter() - start
        except BaseException:
            cancellation.set()
            raise
        finally:
            slots.finish(engine)

    executor = ThreadPoolExecutor(max_workers=len(engine_options))
    try:
        futures = {
            executor.submit(run, name, options): name
            for name, options in engine_options.items()
        }
        return {futures[future]: future.result() for future in as_completed(futures)}
    except BaseException:
        cancellation.set()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
