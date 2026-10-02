"""Run both hint pipelines under one shared limit on engine processes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading
import time

from wthor_hint_pipeline import collect_hints


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


def collect_comparison_hints(groups, workers, engine_options):
    """Collect both engines concurrently; errors/interrupts cancel both pools."""
    cancellation = multiprocessing.get_context("spawn").Event()
    progress_lock = threading.Lock()
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
