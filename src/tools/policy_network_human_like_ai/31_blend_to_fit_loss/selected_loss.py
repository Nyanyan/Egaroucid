"""Warm-TT analyze for only human moves and possible geometric Top-1 moves."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import queue
import threading
import time
import traceback

import numpy as np
import calculate_quest_model_disc_loss as common
from calculate_quest_selected_move_loss import SelectedRunner
from strength_engine import _create_windows_kill_job, _close_windows_job


def possible_top1(stronger, weaker, legal):
    """All moves on the upper envelope of s + alpha*(w-s), alpha in [0,1].

    Include boundary ties conservatively, so arbitrary fine alpha sweeps cannot
    request an unmeasured move. Intersect each move's pairwise dominance bounds.
    """
    result = np.zeros((len(legal), 64), dtype=bool)
    started = last = time.monotonic()
    for index, mask in enumerate(legal):
        policies = np.flatnonzero(common.BITS & mask)
        intercept = stronger[index, policies].astype(np.float64)
        slope = weaker[index, policies].astype(np.float64)-intercept
        a = intercept[:, None]-intercept[None, :]
        b = slope[:, None]-slope[None, :]
        threshold = np.divide(-a, b, out=np.zeros_like(a), where=b != 0)
        lower = np.maximum(0, np.max(np.where(b > 0, threshold, -np.inf), axis=1))
        upper = np.minimum(1, np.min(np.where(b < 0, threshold, np.inf), axis=1))
        parallel_ok = np.all((b != 0) | (a >= 0), axis=1)
        candidates = policies[parallel_ok & (lower <= upper+1e-12)]
        common.require(len(candidates) > 0, 'Empty Top-1 envelope')
        result[index, candidates] = True
        if index+1 == len(legal) or time.monotonic()-last >= 5:
            elapsed = time.monotonic()-started
            print(f'Find possible Top-1: {index+1:,}/{len(legal):,} | elapsed {elapsed:.1f}s | '
                  f'ETA {elapsed*(len(legal)-index-1)/(index+1):.1f}s', flush=True)
            last = time.monotonic()
    return result


def analyze(args, directory, states, legal, requested, identity):
    boards, moves = np.nonzero(requested)
    pairs = np.empty(len(boards), dtype=[('player', '<u8'), ('opponent', '<u8'), ('policy', 'i1')])
    pairs['player'], pairs['opponent'], pairs['policy'] = states['player'][boards], states['opponent'][boards], moves
    identity = dict(identity, unique_board_move_pairs=len(pairs),
                    board_move_sha256=hashlib.sha256(pairs.tobytes()).hexdigest())
    proof = directory / 'analyze_identity.json'
    fresh = not proof.exists()
    if not fresh:
        common.require(common.read(proof) == identity, 'Analyze resume identity differs')
        values = np.load(directory / 'analyzed_moves.npy', mmap_mode='r+', allow_pickle=False)
    else:
        values = np.lib.format.open_memmap(directory / 'analyzed_moves.npy', mode='w+', dtype='<i2', shape=(len(pairs), 3))
        values[:] = -32768
        values.flush()
        np.save(directory / 'board_move_pairs.npy', pairs, allow_pickle=False)
        common.save(proof, identity)
    common.require(values.shape == (len(pairs), 3), 'Analyze cache shape differs')
    if fresh and getattr(args, 'reuse_caches', None):
        reused = []
        for source in args.reuse_caches:
            source = source.resolve()
            if not (source / 'analyze_identity.json').exists():
                continue
            old = common.read(source / 'analyze_identity.json')
            common.require(old['engine'] == identity['engine'] and old['protocol'] == identity['protocol'] and
                           old['loss_formula'] == identity['loss_formula'] and old['forced_move_rule'] == identity['forced_move_rule'],
                           'Imported analyze conditions differ: ' + str(source))
            old_pairs = np.load(source / 'board_move_pairs.npy', allow_pickle=False)
            common.require(old_pairs.dtype == pairs.dtype and
                           hashlib.sha256(old_pairs.tobytes()).hexdigest() == old['board_move_sha256'],
                           'Imported board/move identity differs')
            common.require(np.array_equal(np.unique(old_pairs), old_pairs), 'Imported pairs are not sorted and unique')
            old_values = np.load(source / 'analyzed_moves.npy', mmap_mode='r', allow_pickle=False)
            common.require(old_values.shape == (len(old_pairs), 3), 'Imported cache shape differs')
            indexes = np.searchsorted(old_pairs, pairs)
            rows = np.flatnonzero((indexes < len(old_pairs)) & (values[:, 2] < 0))
            rows = rows[old_pairs[indexes[rows]] == pairs[rows]]
            rows = rows[old_values[indexes[rows], 2] >= 0]
            imported = old_values[indexes[rows]]
            common.require(np.all(imported[:, 2] == np.maximum(imported[:, 0], imported[:, 1])-imported[:, 0]),
                           'Imported loss formula differs')
            values[rows] = imported
            reused.append(dict(source=str(source), imported_pairs=len(rows), board_move_sha256=old['board_move_sha256']))
            print(f'Reused {len(rows):,} exact board/move analyses from {source}', flush=True)
        values.flush()
        common.save(directory / 'cache_reuse.json', dict(sources=reused))
    complete = values[:, 2] >= 0
    common.require(np.all(values[complete, 2] == np.maximum(values[complete, 0], values[complete, 1])-values[complete, 0]),
                   'Cached loss formula differs')
    # A forced move has loss zero without any search. No played-score claim.
    forced = (legal[boards] & (legal[boards]-np.uint64(1))) == 0
    values[forced] = 0
    complete |= forced
    todo, results, stop = queue.Queue(), queue.Queue(), threading.Event()
    pending = np.flatnonzero(~complete)
    priority = getattr(args, 'priority_pairs', None)
    if priority is not None:
        priority = np.unique(np.asarray(priority, dtype=np.int64))
        common.require(np.all((priority >= 0) & (priority < len(pairs))), 'Invalid priority pair indexes')
        pending = np.concatenate((priority[~complete[priority]], np.setdiff1d(pending, priority, assume_unique=True)))
    for index in pending:
        todo.put(int(index))

    def worker():
        runner = SelectedRunner(common.default_engine(), level=args.level, threads=1, timeout_sec=86400,
                                persistent=True, cancel_event=stop)
        job = None
        try:
            runner._ensure_process()
            job = _create_windows_kill_job(runner.proc)
            common.require(os.name != 'nt' or job is not None, 'Engine job attachment failed')
            while not stop.is_set():
                try:
                    index = todo.get_nowait()
                except queue.Empty:
                    return
                pair = pairs[index]
                result = runner.analyze_move(int(pair['player']), int(pair['opponent']), int(pair['policy']))
                results.put((index, result, None))
        except BaseException:
            results.put((None, None, traceback.format_exc()))
            stop.set()
        finally:
            runner.close()
            _close_windows_job(job)

    initial = done = int(complete.sum())
    started = last = time.monotonic()

    def progress(status='running'):
        elapsed = time.monotonic()-started
        rate = (done-initial)/elapsed if elapsed else 0
        eta = (len(pairs)-done)/rate if rate else None
        common.save(directory / 'status.json', dict(status=status, pid=os.getpid(), level=args.level,
                    unique_board_move_pairs=len(pairs), completed_pairs=done, workers=args.workers,
                    elapsed_sec=elapsed, pairs_per_sec=rate, eta_sec=eta, updated_unix=time.time()))
        eta_text = 'estimating' if eta is None else f'{int(eta)//60:02d}:{int(eta)%60:02d}'
        print(f'{status} | selected-move analyze L{args.level} | {done:,}/{len(pairs):,} '
              f'({100*done/len(pairs):.2f}%) | elapsed {elapsed:.0f}s | {rate:.1f} pairs/sec | ETA {eta_text}', flush=True)
        callback = getattr(args, 'progress_callback', None)
        if callback is not None:
            callback(values)

    progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker) for _ in range(min(args.workers, todo.qsize()))]
        try:
            while done < len(pairs):
                try:
                    index, result, error = results.get(timeout=1)
                except queue.Empty:
                    if all(f.done() for f in futures):
                        raise RuntimeError('Workers ended before all results arrived')
                else:
                    if error:
                        raise RuntimeError(error)
                    values[index, :2] = result[:2]
                    values[index, 2] = result[2]  # commit marker last
                    done += 1
                if time.monotonic()-last >= 5:
                    values.flush()
                    progress()
                    last = time.monotonic()
            for future in futures:
                future.result()
        finally:
            stop.set()
            values.flush()
    progress('complete')
    loss = np.full((len(states), 64), np.nan, dtype=np.float32)
    loss[boards, moves] = values[:, 2]
    common.require(np.all(np.isfinite(loss) == requested), 'Requested move coverage differs')
    return loss
