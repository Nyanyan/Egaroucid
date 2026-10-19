"""Run both hint pipelines under one shared limit on engine processes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import threading
import time

from wthor_hint_pipeline import collect_hints, format_duration


def quest_progress_context(engine_options):
    """Read reporting scope from the saved run without changing evaluation identity.

    Quest writes caches to <run>/<dataset>/<engine>. Keep the context lookup
    here so a display-only change does not invalidate its resumable run hashes.
    """
    directories = [Path(options["output_dir"]).resolve().parent
                   for options in engine_options.values() if "output_dir" in options]
    if not directories or len(set(directories)) != 1:
        return None
    directory = directories[0]
    try:
        identity = json.loads((directory.parent / "experiment_identity.json").read_text(encoding="utf-8"))
        datasets = identity.get("datasets", [])
        index = next(i for i, item in enumerate(datasets) if item["name"] == directory.name)
        completed = 0
        for item in datasets:
            summary = directory.parent / item["name"] / "summary.json"
            if summary.exists() and json.loads(summary.read_text(encoding="utf-8")).get("status") == "passed":
                completed += 1
        return dict(dataset=directory.name, index=index + 1, total=len(datasets), completed=completed)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None


class ComparisonProgressReporter:
    """Print one block after every running engine has supplied a new snapshot.

    Format metrics on their collector's thread so row updates cannot race a
    metric snapshot. Completed engines keep their final rows in later blocks.
    """

    def __init__(self, engines, metric_formatters=None, *, progress_context=None):
        self.engines = tuple(engines)
        self.metric_formatters = metric_formatters or {}
        self.progress_context = progress_context
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
        kind = "エンジン探索完了" if final else "エンジン探索進捗"
        context = self.progress_context
        scope = (f"データセット{context['index']}/{context['total']}: {context['dataset']}"
                 if context else "今回呼び出したエンジン探索")
        lines = [
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][{kind}][{scope}] "
            f"今回実行分 {completed:,}/{total:,}件 "
            f"({completed / total if total else 1.0:.1%})"
            f" | この探索の経過 {format_duration(elapsed)} | この探索工程の推定残り {eta}"
            f" | 探索処理速度 {rate:.2f}件/秒",
            "  [件数の単位] 1件 = 1つの盤面・手番 × 1エンジン × 1レベル。"
            "キャッシュ再利用分は今回実行分に含めない。",
        ]
        if context:
            lines.append(
                f"  [全体] 評価完了 {context['completed']}/{context['total']}データセット"
                " | 全体ETA: 未算出"
                " | この探索後: NN3モデルの推論 → 融合・30プレイヤの一致率集計"
            )
        else:
            lines.append("  [範囲] 上記の進捗・推定残りはこの探索工程のみ。実験全体のETAは未算出。")
        lines.append("  [ETA算出] 各エンジンの残件数÷今回の平均処理速度の大きい方。"
                     "レベル・局面ごとの探索時間の違いは補正していない。")
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
    reporter = ComparisonProgressReporter(engine_options, metric_formatters,
                                          progress_context=quest_progress_context(engine_options))
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
