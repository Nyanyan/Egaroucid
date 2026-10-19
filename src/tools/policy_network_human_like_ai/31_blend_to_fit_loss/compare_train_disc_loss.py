"""Human and two adjacent FT policies on the same unaugmented Quest train data."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import fit_blend_to_loss as fit
from policy_data import BOARD_DTYPE

common = fit.common


def dataset(args):
    plan = common.read(args.campaign / 'campaign_plan.json')
    band_info = next(row for row in plan['bands'] if row['band'] == args.band)
    source = args.campaign / 'data_snapshot/quest/board_split' / args.band
    manifest_path = source / 'manifest.json'
    common.require(common.file_sha256(manifest_path) == band_info['manifest_sha256'], 'Training manifest changed')
    manifest = common.read(manifest_path)
    common.require(manifest['stage'] == 'board_split', 'Expected unaugmented board split')
    files = [entry for entry in manifest['files'] if entry['path'].startswith('train/')]
    common.require(sorted(entry['path'] for entry in files) ==
                   sorted(path.relative_to(source).as_posix() for path in (source / 'train').iterdir() if path.is_file()),
                   'Unexpected training files')
    for entry in files:
        common.require(common.file_sha256(source / entry['path']) == entry['sha256'], 'Training source changed')
    paths = sorted((source / 'train').glob('*.dat'))
    records = np.concatenate([np.fromfile(path, dtype=BOARD_DTYPE) for path in paths])
    ids = np.concatenate([np.fromfile(path.with_suffix('.game_ids'), dtype='<u8') for path in paths])
    common.require(len(records) == len(ids) == manifest['splits']['train']['positions'], 'Training count differs')
    take = min(len(records), args.max_positions) if args.max_positions else len(records)
    records, ids = records[:take], ids[:take]
    legal = np.empty(take, dtype=np.uint64)
    started = last = time.monotonic()
    for start in range(0, take, 32768):
        end = min(start+32768, take)
        block = records[start:end]
        legal[start:end] = fit.calc_legal_batch(block['player'], block['opponent'])
        policies = block['policy'].astype(np.int64)
        common.require(np.all((policies >= 0) & (policies < 64)) and
                       np.all(legal[start:end] & common.BITS[policies]) and
                       not np.any(block['player'] & block['opponent']), 'Invalid training board/human move')
        if end == take or time.monotonic()-last >= 5:
            elapsed = time.monotonic()-started
            print(f'Verify train boards: {end:,}/{take:,} | elapsed {elapsed:.1f}s | ETA {elapsed*(take-end)/end:.1f}s', flush=True)
            last = time.monotonic()
    identity = dict(split='train', band=args.band, source=str(source), manifest_sha256=band_info['manifest_sha256'],
                    source_files=files, original_positions=len(ids) if not args.max_positions else manifest['splits']['train']['positions'],
                    positions=take, augmentation=False, partial=bool(args.max_positions),
                    ordered_records_sha256=hashlib.sha256(records.tobytes()).hexdigest(),
                    ordered_game_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
    common.save(args.output / 'dataset_identity.json', identity)
    return records, legal, ids, identity


def rows_from_losses(losses, names):
    rows = []
    common.require(losses.shape[1] == len(names) and np.all(losses >= 0), 'Invalid comparison losses')
    for index, name in enumerate(names):
        column = losses[:, index]
        total = int(column.sum(dtype=np.int64))
        rows.append(dict(policy=name, positions=len(column), total_disc_loss=total,
                         mean_disc_loss=total/len(column), zero_loss_positions=int(np.count_nonzero(column == 0))))
    return rows


def run(args):
    print(f'{args.band} train: human + target FT + lower-band FT; L{args.level}; warm TT', flush=True)
    common.save(args.output / 'run_status.json', dict(status='running', phase='prepare', pid=os.getpid(), started_unix=time.time()))
    records, legal, ids, data_identity = dataset(args)
    roster = fit.model_roster(args)
    keys = np.empty(len(records), dtype=common.KEY_DTYPE)
    for field in keys.dtype.names:
        keys[field] = records[field]
    states, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    identity = dict(dataset=data_identity, models=roster, level=args.level,
                    engine={**common.engine_identity(common.default_engine(), 25), 'level': args.level,
                            'clear_cache_per_position': False},
                    board_sha256=hashlib.sha256(states.tobytes()).hexdigest(),
                    unique_positions=len(states), positions=len(records),
                    protocol='setboard relative player as black; play selected move; analyze; persistent warm TT',
                    loss_formula='max(played_score, best_other_move_score) - played_score; no alternative => 0',
                    forced_move_rule='loss zero without search; cached scores are placeholders',
                    requested_moves='actual human move and legal Top-1 of both FT models on identical train positions',
                    tie_rule='smallest policy index',
                    denominator='all original train positions including duplicates, zero losses and forced moves; passes excluded')
    logits = fit.infer_logits(records, roster, args.output, identity)
    predictions = np.column_stack((records['policy'], fit.select_moves(logits, legal, 0), fit.select_moves(logits, legal, 1))).astype('i1')
    del logits
    np.save(args.output / 'predictions.npy', predictions, allow_pickle=False)
    common.require(np.all(legal[:, None] & common.BITS[predictions]), 'Illegal comparison prediction')
    requested = np.zeros((len(states), 64), dtype=bool)
    requested[inverse[:, None], predictions] = True
    encoded = np.flatnonzero(requested.ravel())
    record_pairs = inverse[:, None]*64 + predictions.astype(np.int64)
    pair_indexes = np.searchsorted(encoded, record_pairs)
    common.require(np.array_equal(encoded[pair_indexes], record_pairs), 'Board/move pairing differs')
    names = ['human', roster[0]['name'], roster[1]['name']]
    sample_size = min(len(records), args.preview_positions)
    sample = np.sort(np.random.default_rng(args.seed).choice(len(records), sample_size, replace=False))
    sample_pairs = pair_indexes[sample]
    args.priority_pairs = np.unique(sample_pairs)
    published = False

    def preview(values):
        nonlocal published
        if published or not sample_size:
            return
        losses = values[sample_pairs, 2]
        if np.any(losses < 0):
            return
        rows = rows_from_losses(losses, names)
        common.save(args.output / 'preview_result.json', dict(status='sample_complete', dataset=data_identity,
                    sample_method='uniform random original positions without replacement; same sample for all policies',
                    sample_positions=sample_size, total_positions=len(records), seed=args.seed, rows=rows,
                    sample_indices_sha256=hashlib.sha256(sample.tobytes()).hexdigest(),
                    note='Full training evaluation is still running; these are sampled estimates'))
        np.save(args.output / 'preview_indices.npy', sample, allow_pickle=False)
        fit.write_csv(args.output / 'preview_result.csv', rows)
        print(f'PREVIEW COMPLETE: common random sample {sample_size:,}/{len(records):,} positions', flush=True)
        for row in rows:
            print(f'  {row["policy"]}: mean disc loss={row["mean_disc_loss"]:.8f}', flush=True)
        published = True

    args.progress_callback = preview
    loss_table = fit.analyze(args, args.output, states, legal[first], requested, identity)
    losses = loss_table[inverse[:, None], predictions]
    common.require(np.isfinite(losses).all(), 'Missing comparison losses')
    np.save(args.output / 'losses.npy', losses.astype('<i2'), allow_pickle=False)
    rows = rows_from_losses(losses, names)
    for index, row in enumerate(rows):
        common.require(row['total_disc_loss'] == sum(int(value) for value in losses[:, index]), 'Independent scalar sum differs')
    for model in roster:
        common.require(common.file_sha256(Path(model['path'])) == model['sha256'], 'Model changed during evaluation')
    result = dict(status='complete', identity=identity, rows=rows, partial=bool(args.max_positions),
                  predictions_sha256=common.file_sha256(args.output / 'predictions.npy'),
                  losses_sha256=common.file_sha256(args.output / 'losses.npy'),
                  verification='source/model hashes, legal Top-1, board/move pairing and independent integer sums passed')
    common.save(args.output / 'result.json', result)
    fit.write_csv(args.output / 'result.csv', rows)
    common.save(args.output / 'run_status.json', dict(status='complete', band=args.band, positions=len(records), rows=rows))
    print('FULL TRAIN COMPARISON COMPLETE:', rows, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=fit.ROOT / 'experiment_bat/20261010_quest_rating_bands')
    parser.add_argument('--band', default='1400-1600', choices=common.MODEL_BANDS[1:])
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--preview-positions', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=613)
    parser.add_argument('--max-positions', type=int, default=0, help='Prefix smoke test only; default full train')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    common.require(args.workers > 0 and args.preview_positions >= 0 and args.max_positions >= 0 and 1 <= args.level <= 60,
                   'Invalid arguments')
    suffix = f'train_{args.band}_l{args.level}' + (f'_smoke{args.max_positions}' if args.max_positions else '')
    args.output = args.output or fit.HERE / 'output' / suffix
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(fit.HERE.parent / '11_fine_tuning'))
    from fine_tune_policy import run_lock
    with run_lock(args.output):
        try:
            run(args)
        except BaseException as error:
            common.save(args.output / 'run_status.json', dict(status='failed', error=str(error), pid=os.getpid()))
            raise


if __name__ == '__main__':
    main()
