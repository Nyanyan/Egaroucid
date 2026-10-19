"""Evaluate frozen Quest fine-tuned Top-1 moves on the common [800,2200) test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parent / '00_data_processing'))
sys.path.insert(0, str(HERE.parent / '30_blend_with_egaroucid'))
sys.path.insert(0, str(HERE.parent / '40_test_strength'))
from miss_analysis import default_engine, engine_identity
from wthor_dataset import file_sha256
from blend_policy_with_egaroucid import (EgaroucidHintRunner, BoardState, BLACK,
                                       require_complete_egaroucid_scores)
from strength_engine import _create_windows_kill_job, _close_windows_job

BANDS = tuple(f'{n}-{n+200}' for n in range(800, 2200, 200))
MODEL_BANDS = ('under_800',) + BANDS + ('2200_plus',)
BITS = np.uint64(1) << np.arange(64, dtype=np.uint64)
KEY_DTYPE = np.dtype([('player', '<u8'), ('opponent', '<u8')])


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers can briefly prevent a rename. Keep the atomic write and
    # retry sharing/access conflicts instead of losing a completed search run.
    deadline = time.monotonic() + 15
    while True:
        try:
            tmp.replace(path)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32) or time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Runner(EgaroucidHintRunner):
    def command(self):
        return super().command() + ['-hash', '25', '-noboard', '-m', '3']

    @staticmethod
    def _pump_stdout(proc, output_queue):
        # One queue item per pipe chunk avoids a Python queue/lock per character.
        try:
            while True:
                chunk = os.read(proc.stdout.fileno(), 65536)
                output_queue.put(chunk.decode('utf-8', errors='replace'))
                if not chunk:
                    return
        except BaseException as error:
            output_queue.put(error)

    def _read_until_prompt(self):
        chunks = []
        deadline = time.monotonic() + self.timeout_sec
        while True:
            if self.cancel_event.is_set():
                self.proc.kill()
                raise InterruptedError('Engine search cancelled')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Engine prompt timeout')
            try:
                chunk = self.stdout_queue.get(timeout=min(remaining, .25))
            except queue.Empty:
                continue
            if isinstance(chunk, BaseException):
                raise RuntimeError('Engine pipe read failed') from chunk
            if not chunk:
                raise RuntimeError('Engine ended before prompt')
            chunks.append(chunk)
            output = ''.join(chunks)
            if output.endswith('> '):
                return output


def prepare(args):
    evaluation = args.campaign / 'evaluation'
    plan = read(args.campaign / 'campaign_plan.json')
    records, legals, predictions, sources, models, lengths = [], [], [], {}, [], []
    for model_band in MODEL_BANDS:
        model = read(evaluation / 'models' / ('finetune_' + model_band) / 'quest_800-1000_test' / 'summary.json')['identity']['model']
        require(file_sha256(Path(model['path'])) == model['sha256'], 'Selected model changed')
        models.append(model)
    test_bands = MODEL_BANDS if args.own_band else BANDS
    for band in test_bands:
        name = 'quest_' + band + '_test'
        directory = evaluation / 'datasets' / name
        summary = read(directory / 'summary.json')
        require(summary['status'] == 'passed', 'Unverified test dataset')
        info = summary['identity']['dataset']
        frozen = Path(info['directory'])
        require(file_sha256(frozen / 'manifest.json') == info['manifest_sha256'], 'Source manifest changed')
        manifest = read(frozen / 'manifest.json')
        for entry in summary['identity']['source_files']:
            require(file_sha256(frozen / entry['path']) == entry['sha256'], 'Frozen source test file changed')
        arrays = {}
        for filename, sha in summary['files'].items():
            require(file_sha256(directory / filename) == sha, 'Dataset array changed')
            sources[str((directory / filename).resolve())] = sha
            arrays[filename[:-4]] = np.load(directory / filename, mmap_mode='r', allow_pickle=False)
        n = info['positions']
        require(n == manifest['splits']['test']['positions'] == len(arrays['records']), 'Full test count differs')
        require(np.all(arrays['legal'] != 0), 'Test position has no move')
        move_arrays = []
        for model in models:
            if args.own_band and model['band'] != band:
                continue
            md = evaluation / 'models' / model['name'] / name
            proof = read(md / 'summary.json')
            require(proof['status'] == 'passed' and proof['identity']['model'] == model,
                    'Prediction model identity differs')
            require(proof['identity']['dataset'] == info and proof['identity']['dataset_cache_files'] == summary['files'],
                    'Prediction board identity differs')
            path = md / 'predictions.npy'
            require(file_sha256(path) == proof['predictions_sha256'], 'Saved predictions changed')
            sources[str(path.resolve())] = proof['predictions_sha256']
            moves = np.load(path, allow_pickle=False)
            require(moves.shape == (n,) and np.all((moves >= 0) & (moves < 64)), 'Prediction count/range differs')
            require(np.all(BITS[moves] & arrays['legal']), 'Illegal saved model move')
            move_arrays.append(moves)
        take = min(n, args.max_positions) if args.max_positions else n
        records.append(arrays['records'][:take])
        legals.append(arrays['legal'][:take])
        predictions.append(np.stack(move_arrays, axis=1)[:take])
        lengths.append(take)
    all_records = np.concatenate(records)
    legal = np.concatenate(legals)
    moves = np.concatenate(predictions)
    keys = np.empty(len(all_records), dtype=KEY_DTYPE)
    for field in KEY_DTYPE.names:
        keys[field] = all_records[field]
    states, inverse, first = None, None, None
    states, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    # Finish the small rating bands first, making complete band results visible.
    if args.own_band:
        first_band = np.searchsorted(np.cumsum(lengths), first, side='right')
        band_priority = np.argsort(np.argsort(lengths, kind='stable'), kind='stable')
        args.search_order = np.lexsort((first, band_priority[first_band]))
    unique_legal = legal[first]
    require(np.array_equal(unique_legal[inverse], legal), 'Duplicate board legal masks differ')
    identity = dict(level=args.level, engine={**engine_identity(default_engine(), 25), 'level': args.level,
                    'clear_cache_per_position': True},
                    protocol='setboard relative player as black; clearcache; hint 100; all legal moves',
                    rating_min_inclusive=None if args.own_band else 800,
                    rating_max_exclusive=None if args.own_band else 2200, bands=list(test_bands),
                    models=models, source_files_sha256=sources, positions=len(keys), unique_positions=len(states),
                    board_sha256=hashlib.sha256(states.tobytes()).hexdigest(), augmentation=False,
                    max_positions_per_band=args.max_positions, partial=bool(args.max_positions),
                    loss_formula='max(all legal L10 hint scores) - selected model move score' if args.level == 10 else
                                 'max(all legal hint scores) - selected model move score',
                    denominator='all original test positions, including duplicates, zero loss and forced moves; passes excluded')
    if args.own_band:
        identity['model_test_pairing'] = 'each fine-tuned model evaluated only on its own training rating band'
    return states, unique_legal, inverse, moves, lengths, identity


def import_scores(directory, states, legal, scores, identity):
    source = read(directory / 'identity.json')
    require(source['engine'] == identity['engine'] and source['protocol'] == identity['protocol'],
            'Imported engine conditions differ')
    arrays = []
    for band in source['bands']:
        matching = [Path(p) for p in source['source_files_sha256']
                    if Path(p).name == 'records.npy' and Path(p).parent.name == 'quest_' + band + '_test']
        require(len(matching) == 1, 'Imported board source missing')
        path = matching[0]
        require(file_sha256(path) == source['source_files_sha256'][str(path)], 'Imported source boards changed')
        records = np.load(path, allow_pickle=False)
        limit = source['max_positions_per_band']
        if limit:
            records = records[:limit]
        keys = np.empty(len(records), dtype=KEY_DTYPE)
        for field in KEY_DTYPE.names:
            keys[field] = records[field]
        arrays.append(keys)
    old_states = np.unique(np.concatenate(arrays))
    require(hashlib.sha256(old_states.tobytes()).hexdigest() == source['board_sha256'],
            'Imported board order differs')
    old_scores = np.load(directory / 'scores.npy', mmap_mode='r', allow_pickle=False)
    require(old_scores.shape == (len(old_states), 64), 'Imported score shape differs')
    imported = 0
    for start in range(0, len(states), 32768):
        end = min(start + 32768, len(states))
        positions = np.searchsorted(old_states, states[start:end])
        present = positions < len(old_states)
        rows = np.flatnonzero(present)
        rows = rows[old_states[positions[rows]] == states[start:end][rows]]
        if not len(rows):
            continue
        values = old_scores[positions[rows]]
        masks = (legal[start:end][rows, None] & BITS) != 0
        usable = np.all(np.isfinite(values) == masks, axis=1)
        target = start + rows[usable]
        scores[target] = values[usable]
        imported += len(target)
    scores.flush()
    print(f'Reused {imported:,} exact-board scores from {directory}', flush=True)
    return imported


def own_band_rows(scores, inverse, moves, lengths, identity, *, partial=False):
    rows, offset = [], 0
    for model, band, count in zip(identity['models'], identity['bands'], lengths):
        require(model['band'] == band, 'Own-band model/test pairing differs')
        idx = inverse[offset:offset+count]
        chosen = moves[offset:offset+count, 0]
        values = scores[idx]
        if partial and not np.all(np.any(np.isfinite(values), axis=1)):
            offset += count
            continue
        best = np.nanmax(values, axis=1)
        loss = best - values[np.arange(count), chosen]
        require(np.all(np.isfinite(loss)) and np.all(loss >= 0), 'Own-band loss missing/negative')
        total = float(loss.sum(dtype=np.float64))
        scalar = sum(float(b) - float(scores[int(i), int(m)]) for b, i, m in zip(best, idx, chosen))
        require(total == scalar, 'Own-band independent scalar sum differs')
        rows.append(dict(model=model['name'], training_band=band, test_band=band,
                         adopted_epoch=model['adopted_epoch'], positions=count,
                         total_disc_loss=total, mean_disc_loss=total/count,
                         zero_loss_positions=int(np.count_nonzero(loss == 0))))
        offset += count
    require(offset == identity['positions'], 'Own-band denominator differs')
    return rows


def search(args, states, legal, identity):
    proof = args.output / 'identity.json'
    if proof.exists():
        require(read(proof) == identity, 'Resume identity differs')
        scores = np.load(args.output / 'scores.npy', mmap_mode='r+', allow_pickle=False)
    else:
        scores = np.lib.format.open_memmap(args.output / 'scores.npy', mode='w+', dtype='<f4', shape=(len(states), 64))
        scores[:] = np.nan
        scores.flush()
        save(proof, identity)
        if args.reuse_scores:
            import_scores(args.reuse_scores, states, legal, scores, identity)
    require(scores.shape == (len(states), 64), 'Score cache shape differs')
    complete = np.any(np.isfinite(scores), axis=1)
    for start in range(0, len(states), 32768):
        end = min(start + 32768, len(states))
        legal_mask = (legal[start:end, None] & BITS) != 0
        finite = np.isfinite(scores[start:end])
        require(np.all(finite[complete[start:end]] == legal_mask[complete[start:end]]), 'Incomplete legal score cache row')
    todo = queue.Queue()
    order = getattr(args, 'search_order', np.arange(len(states)))
    for index in order[~complete[order]]:
        todo.put(int(index))
    results = queue.Queue()
    stop = threading.Event()
    def worker():
        runner = Runner(default_engine(), level=args.level, threads=1, timeout_sec=86400, persistent=True,
                        cancel_event=stop)
        job = None
        try:
            runner._ensure_process()
            job = _create_windows_kill_job(runner.proc)
            require(os.name != 'nt' or job is not None, 'Engine job attachment failed')
            while not stop.is_set():
                try:
                    index = todo.get_nowait()
                except queue.Empty:
                    return
                runner.proc.stdin.write('clearcache\n')
                runner.proc.stdin.flush()
                runner._read_until_prompt()
                state = BoardState(int(states[index]['player']), int(states[index]['opponent']), BLACK)
                values, raw = runner.hint_scores(state, BLACK, hint_count=100)
                policies = [j for j in range(64) if int(legal[index]) & (1 << j)]
                require_complete_egaroucid_scores(values, policies)
                require(all(np.isfinite(v) and -64 <= v <= 64 for v in values.values()), 'Invalid engine score')
                results.put((index, values, None))
        except BaseException:
            results.put((None, None, traceback.format_exc()))
            stop.set()
        finally:
            runner.close()
            _close_windows_job(job)
    initial = done = int(np.count_nonzero(complete))
    started = last = time.monotonic()
    published = set()
    def progress(status='running'):
        elapsed = time.monotonic() - started
        rate = (done-initial) / elapsed if elapsed else 0
        value = dict(status=status, pid=os.getpid(), level=args.level, positions=identity['positions'],
                     unique_positions=len(states), completed_unique_positions=done, workers=args.workers,
                     elapsed_sec=elapsed, unique_positions_per_sec=rate,
                     eta_sec=(len(states)-done)/rate if rate else None, updated_unix=time.time())
        save(args.output / 'status.json', value)
        eta = value['eta_sec']
        if eta is None:
            eta_text = 'estimating'
        else:
            hours, remainder = divmod(int(eta), 3600)
            minutes, seconds = divmod(remainder, 60)
            eta_text = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        print(f'{status} | L{args.level} | {done:,}/{len(states):,} unique boards '
              f'({100*done/len(states):.2f}%) | elapsed {elapsed:.0f}s '
              f'| {rate:.2f} boards/sec | ETA {eta_text}', flush=True)
        if args.own_band:
            inverse, moves, lengths = args.own_data
            rows = own_band_rows(scores, inverse, moves, lengths, identity, partial=True)
            save(args.output / 'completed_band_results.json', dict(status='partial', rows=rows,
                 note='Only fully evaluated bands are shown; remaining bands are still pending'))
            for row in rows:
                if row['test_band'] not in published:
                    print(f"BAND COMPLETE: {row['test_band']} | n={row['positions']:,} | "
                          f"mean disc loss={row['mean_disc_loss']:.6f}", flush=True)
                    published.add(row['test_band'])
    progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker) for _ in range(min(args.workers, todo.qsize()))]
        try:
            while done < len(states):
                try:
                    index, values, error = results.get(timeout=1)
                except queue.Empty:
                    if all(f.done() for f in futures):
                        raise RuntimeError('Workers ended before all scores were received')
                else:
                    if error:
                        raise RuntimeError(error)
                    row = np.full(64, np.nan, dtype='<f4')
                    for move, value in values.items():
                        row[move] = value
                    scores[index] = row
                    done += 1
                if time.monotonic() - last >= 5:
                    scores.flush()
                    progress()
                    last = time.monotonic()
            for future in futures:
                future.result()
        finally:
            stop.set()
            scores.flush()
    progress('search_complete')
    return scores


def aggregate(args, scores, inverse, moves, lengths, identity):
    if args.own_band:
        rows = own_band_rows(scores, inverse, moves, lengths, identity)
        with (args.output / 'model_avg_disc_loss.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save(args.output / 'model_avg_disc_loss.json', dict(status='complete', identity=identity, rows=rows,
             scores_sha256=file_sha256(args.output / 'scores.npy'),
             verification='source/model/prediction hashes and legal moves verified; independent scalar sums agree'))
        save(args.output / 'status.json', dict(status='complete', positions=identity['positions'], models=len(rows), level=args.level))
        print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
        return
    best = np.nanmax(scores, axis=1)
    totals = np.zeros(len(MODEL_BANDS), dtype=np.float64)
    detailed = []
    offset = 0
    for band, count in zip(BANDS, lengths):
        idx = inverse[offset:offset+count]
        selected = moves[offset:offset+count]
        loss = best[idx, None] - scores[idx[:, None], selected]
        require(np.isfinite(loss).all() and np.all(loss >= 0), 'Missing or negative loss')
        summed = loss.sum(axis=0, dtype=np.float64)
        for j, model in enumerate(identity['models']):
            # Independent scalar indexing catches broadcasting/index errors.
            scalar = sum(float(best[int(i)]) - float(scores[int(i), int(m)]) for i, m in zip(idx, selected[:, j]))
            require(scalar == float(summed[j]), 'Scalar loss recount differs')
            detailed.append(dict(model=model['name'], training_band=model['band'], test_band=band,
                                 positions=count, total_disc_loss=float(summed[j]), mean_disc_loss=float(summed[j]/count),
                                 zero_loss_positions=int(np.count_nonzero(loss[:, j] == 0))))
        totals += summed
        offset += count
    require(offset == identity['positions'], 'Aggregation denominator differs')
    rows = [dict(model=m['name'], training_band=m['band'], adopted_epoch=m['adopted_epoch'],
                 positions=offset, total_disc_loss=float(totals[j]), mean_disc_loss=float(totals[j]/offset))
            for j, m in enumerate(identity['models'])]
    for filename, values in [('model_avg_disc_loss.csv', rows), ('model_avg_disc_loss_by_test_band.csv', detailed)]:
        with (args.output / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    report = dict(status='complete', identity=identity, rows=rows, by_test_band=detailed,
                  scores_sha256=file_sha256(args.output / 'scores.npy'),
                  verification='all source/model/prediction SHA256 checked; all chosen moves legal; all legal scores present; independent scalar loss recount agrees')
    save(args.output / 'model_avg_disc_loss.json', report)
    save(args.output / 'status.json', dict(status='complete', positions=offset, models=len(rows), level=args.level))
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'experiment_bat/20261010_quest_rating_bands')
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--max-positions', type=int, default=0, help='per-band smoke limit; 0 evaluates every position')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--own-band', action='store_true', help='evaluate each model only on its own rating band, including under 800 and 2200+')
    parser.add_argument('--reuse-scores', type=Path, help='reuse exact-board scores from a stopped run with identical engine conditions')
    args = parser.parse_args()
    require(args.workers > 0 and 1 <= args.level <= 60 and args.max_positions >= 0, 'Invalid options')
    stem = 'quest_finetune_models_own_band' if args.own_band else 'quest_finetune_models_800_2200'
    args.output = args.output or HERE / 'output' / f'{stem}_l{args.level}'
    require(not args.max_positions or args.output.name != f'{stem}_l{args.level}', 'Smoke requires a separate output')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        print('Verifying all frozen source test files, models and saved predictions...', flush=True)
        states, legal, inverse, moves, lengths, identity = prepare(args)
        if args.own_band:
            args.own_data = (inverse, moves, lengths)
        scores = search(args, states, legal, identity)
        aggregate(args, scores, inverse, moves, lengths, identity)
    except BaseException as error:
        save(args.output / 'status.json', dict(status='failed', error=str(error), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
