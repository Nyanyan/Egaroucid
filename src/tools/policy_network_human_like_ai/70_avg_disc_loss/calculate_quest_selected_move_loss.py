"""Analyze only each fine-tuned model's saved move on its own Quest test band."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np
import calculate_quest_model_disc_loss as common
from miss_analysis import coord_policy
from blend_policy_with_egaroucid import BoardState, BLACK
from strength_engine import _create_windows_kill_job, _close_windows_job


class SelectedRunner(common.Runner):
    def command_read(self, command):
        self.proc.stdin.write(command + '\n')
        self.proc.stdin.flush()
        raw = self._read_until_prompt()
        common.require('[ERROR]' not in raw.upper(), 'Engine rejected command: ' + raw)
        return raw

    def analyze_move(self, player, opponent, policy):
        board = BoardState(int(player), int(opponent), BLACK)
        self.command_read('setboard ' + board.to_egaroucid_board(BLACK))
        index = 63 - int(policy)
        coord = chr(ord('a') + index % 8) + str(index // 8 + 1)
        self.command_read('play ' + coord)
        raw = self.command_read('analyze')
        rows = []
        for line in raw.splitlines():
            cells = [cell.strip() for cell in line.strip().split('|')]
            if len(cells) < 2 or not cells[1].isdigit():
                continue
            common.require(len(cells) == 11 and cells[2] == 'Black' and coord_policy(cells[3].lower()) == policy,
                           'Analyze returned wrong move or color: ' + line)
            expected_ply = bin(int(player) | int(opponent)).count('1') - 3
            common.require(int(cells[1]) == expected_ply, 'Analyze returned wrong board ply')
            played = int(cells[5])
            alternative = played if cells[6].lower() == 'none' else int(cells[8])
            common.require(-64 <= played <= 64 and -64 <= alternative <= 64, 'Invalid analyze score')
            rows.append((played, alternative, max(played, alternative) - played))
        common.require(len(rows) == 1, 'Expected exactly one analyzed model move: ' + raw)
        return rows[0]


def run(args):
    states, legal, board_inverse, moves, lengths, identity = common.prepare(args)
    dtype = np.dtype([('player', '<u8'), ('opponent', '<u8'), ('policy', 'i1')])
    records = np.empty(len(board_inverse), dtype=dtype)
    for field in ('player', 'opponent'):
        records[field] = states[board_inverse][field]
    records['policy'] = moves[:, 0]
    keys, first, inverse = np.unique(records, return_index=True, return_inverse=True)
    identity['engine']['clear_cache_per_position'] = False
    identity['protocol'] = 'setboard; play saved model move; analyze; persistent warm TT, identical to Quest human analyze settings'
    identity['loss_formula'] = 'max(played_score, best_other_move_score) - played_score; no alternative => 0'
    identity['unique_board_move_pairs'] = len(keys)
    identity['board_move_sha256'] = hashlib.sha256(keys.tobytes()).hexdigest()
    proof = args.output / 'identity.json'
    if proof.exists():
        common.require(common.read(proof) == identity, 'Selected-move resume identity differs')
        values = np.load(args.output / 'analyzed_moves.npy', mmap_mode='r+', allow_pickle=False)
    else:
        values = np.lib.format.open_memmap(args.output / 'analyzed_moves.npy', mode='w+', dtype='<i2', shape=(len(keys), 3))
        values[:] = -32768
        values.flush()
        common.save(proof, identity)
    common.require(values.shape == (len(keys), 3), 'Selected-move cache shape differs')
    complete = values[:, 2] >= 0
    common.require(np.all(values[complete, 2] == np.maximum(values[complete, 0], values[complete, 1]) - values[complete, 0]),
                   'Cached loss formula differs')
    first_band = np.searchsorted(np.cumsum(lengths), first, side='right')
    priority = np.argsort(np.argsort(lengths, kind='stable'), kind='stable')
    order = np.lexsort((first, priority[first_band]))
    todo, results, stop = queue.Queue(), queue.Queue(), threading.Event()
    for index in order[~complete[order]]:
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
                key = keys[index]
                result = runner.analyze_move(int(key['player']), int(key['opponent']), int(key['policy']))
                results.put((index, result, None))
        except BaseException:
            results.put((None, None, traceback.format_exc()))
            stop.set()
        finally:
            runner.close()
            _close_windows_job(job)
    initial = done = int(complete.sum())
    started = last = time.monotonic()
    published = set()
    def band_rows():
        rows, offset = [], 0
        for model, band, n in zip(identity['models'], identity['bands'], lengths):
            idx = inverse[offset:offset+n]
            losses = values[idx, 2]
            if np.all(losses >= 0):
                total = int(losses.sum(dtype=np.int64))
                common.require(total == sum(int(values[int(i), 2]) for i in idx), 'Independent loss sum differs')
                rows.append(dict(model=model['name'], training_band=band, test_band=band,
                                 adopted_epoch=model['adopted_epoch'], positions=n, total_disc_loss=total,
                                 mean_disc_loss=total/n, zero_loss_positions=int(np.count_nonzero(losses == 0))))
            offset += n
        common.require(offset == len(inverse), 'Band denominator mismatch')
        return rows
    def progress(status='running'):
        elapsed = time.monotonic() - started
        rate = (done-initial)/elapsed if elapsed else 0
        eta = (len(keys)-done)/rate if rate else None
        common.save(args.output / 'status.json', dict(status=status, pid=os.getpid(), level=args.level,
                    positions=len(records), unique_board_move_pairs=len(keys), completed_pairs=done,
                    workers=args.workers, elapsed_sec=elapsed, pairs_per_sec=rate, eta_sec=eta, updated_unix=time.time()))
        eta_text = 'estimating' if eta is None else f'{int(eta)//3600:02d}:{int(eta)%3600//60:02d}:{int(eta)%60:02d}'
        print(f'{status} | selected moves L{args.level} | {done:,}/{len(keys):,} '
              f'({100*done/len(keys):.2f}%) | elapsed {elapsed:.0f}s | {rate:.1f} moves/sec | ETA {eta_text}', flush=True)
        rows = band_rows()
        common.save(args.output / 'completed_band_results.json', dict(status='partial', rows=rows))
        for row in rows:
            if row['test_band'] not in published:
                print(f"BAND COMPLETE: {row['test_band']} | n={row['positions']:,} | mean disc loss={row['mean_disc_loss']:.6f}", flush=True)
                published.add(row['test_band'])
    progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker) for _ in range(min(args.workers, todo.qsize()))]
        try:
            while done < len(keys):
                try:
                    index, result, error = results.get(timeout=1)
                except queue.Empty:
                    if all(f.done() for f in futures):
                        raise RuntimeError('Workers ended before all results arrived')
                else:
                    if error:
                        raise RuntimeError(error)
                    # Completion marker is stored last for safe interruption/resume.
                    values[index, :2] = result[:2]
                    values[index, 2] = result[2]
                    done += 1
                if time.monotonic() - last >= 5:
                    values.flush()
                    progress()
                    last = time.monotonic()
            for future in futures:
                future.result()
        finally:
            stop.set()
            values.flush()
    rows = band_rows()
    common.require(len(rows) == 9, 'Expected all nine own-band results')
    with (args.output / 'model_avg_disc_loss.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common.save(args.output / 'model_avg_disc_loss.json', dict(status='complete', identity=identity, rows=rows,
                analyzed_moves_sha256=common.file_sha256(args.output / 'analyzed_moves.npy'),
                verification='source/model/prediction hashes and legal saved moves checked; analyze row board/ply/color/move checked; scalar sums agree'))
    progress('complete')
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=common.ROOT / 'experiment_bat/20261010_quest_rating_bands')
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--max-positions', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.own_band = True
    args.output = args.output or common.HERE / 'output' / f'quest_finetune_selected_own_band_l{args.level}'
    common.require(args.workers > 0 and 1 <= args.level <= 60 and args.max_positions >= 0, 'Invalid arguments')
    common.require(not args.max_positions or args.output.name != f'quest_finetune_selected_own_band_l{args.level}', 'Smoke needs separate output')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        print('Verifying saved own-band model moves; selected-move analyze, warm TT; no full-legal hint search.', flush=True)
        run(args)
    except BaseException as error:
        common.save(args.output / 'status.json', dict(status='failed', error=str(error), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
