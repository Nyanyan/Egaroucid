"""Fit two FT model pairs against human loss on a common validation rating band."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import fit_blend_to_loss as fit

common = fit.common
MODEL_BANDS = ('1000-1200', '1200-1400', '1400-1600')
PAIRS = (('1200-1400', '1000-1200'), ('1400-1600', '1000-1200'))


def models(args):
    result = {}
    for band in MODEL_BANDS:
        path = args.campaign / 'evaluation/models' / ('finetune_' + band) / 'quest_800-1000_test/summary.json'
        model = common.read(path)['identity']['model']
        common.require(model['method'] == 'finetune' and model['band'] == band and
                       common.file_sha256(Path(model['path'])) == model['sha256'], 'Selected FT model differs')
        result[band] = model
    return result


def load_logits(args, records, roster, dataset, board_sha):
    result = {}
    for band, model in roster.items():
        for directory in args.reuse_caches:
            proof_path = directory / 'logits_identity.json'
            if not proof_path.exists():
                continue
            proof = common.read(proof_path)
            info = proof['identity']
            if info['dataset'] != dataset or info['board_sha256'] != board_sha or model not in info['models']:
                continue
            index = info['models'].index(model)
            path = directory / f'logits_{index}.npy'
            common.require(common.file_sha256(path) == proof['files'][path.name], 'Reused logits changed')
            result[band] = np.load(path, allow_pickle=False)
            print(f'Reused saved policy logits: {band}', flush=True)
            break
        if band not in result:
            directory = args.output / 'inference' / band
            directory.mkdir(parents=True, exist_ok=True)
            identity = dict(dataset=dataset, models=[model], board_sha256=board_sha,
                            output='pre-softmax logits, legal Top-1; same inference as original calibration')
            result[band] = fit.infer_logits(records, [model], directory, identity)[0]
        common.require(result[band].shape == (len(records), 64) and np.isfinite(result[band]).all(), 'Invalid logits')
    return result


def compact_metric(logits, possible, losses):
    """Preserve full-legal Top-1 using the complete upper-envelope candidates."""
    counts = possible.sum(axis=1)
    common.require(np.all(counts > 0), 'Empty candidate row')
    rows, policies = np.nonzero(possible)
    offsets = np.cumsum(counts)-counts
    columns = np.arange(len(rows))-offsets[rows]
    shape = (len(possible), int(counts.max()))
    indexes = np.full(shape, -1, dtype=np.int64)
    strong = np.full(shape, -np.inf, dtype=np.float64)
    weak = strong.copy()
    candidate_losses = np.full(shape, np.nan, dtype=np.float64)
    indexes[rows, columns] = policies
    strong[rows, columns] = logits[0][rows, policies]
    weak[rows, columns] = logits[1][rows, policies]
    candidate_losses[rows, columns] = losses[rows, policies]
    common.require(np.isfinite(candidate_losses[rows, columns]).all(), 'Missing envelope candidate loss')
    print(f'Compact alpha sweep: {shape[0]:,} positions x at most {shape[1]} possible moves', flush=True)
    row_indexes = np.arange(shape[0])

    def moves(alpha):
        common.require(0 <= alpha <= 1, 'Invalid alpha')
        values = strong if alpha == 0 else weak if alpha == 1 else (1-alpha)*strong + alpha*weak
        chosen = values.argmax(axis=1)
        return indexes[row_indexes, chosen], candidate_losses[row_indexes, chosen]

    def metric(alpha):
        _, selected = moves(alpha)
        total = float(selected.sum(dtype=np.float64))
        return dict(alpha=float(alpha), positions=len(selected), total_disc_loss=total,
                    mean_disc_loss=total/len(selected), zero_loss_positions=int(np.count_nonzero(selected == 0)))

    return metric, moves


def run(args):
    common.save(args.output / 'run_status.json', dict(status='running', pid=os.getpid(), band=args.band,
                started_unix=time.time(), mode='two model pairs, validation only'))
    records, legal, ids, dataset = fit.load_dataset(args, 'val')
    roster = models(args)
    keys = np.empty(len(records), dtype=common.KEY_DTYPE)
    for field in keys.dtype.names:
        keys[field] = records[field]
    states, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    board_sha = hashlib.sha256(states.tobytes()).hexdigest()
    logits = load_logits(args, records, roster, dataset, board_sha)
    identity = dict(dataset=dataset, models=list(roster.values()), level=args.level,
                    engine={**common.engine_identity(common.default_engine(), 25), 'level': args.level,
                            'clear_cache_per_position': False},
                    board_sha256=board_sha, positions=len(records), unique_positions=len(states),
                    protocol='setboard relative player as black; play selected move; analyze; persistent warm TT',
                    loss_formula='max(played_score, best_other_move_score) - played_score; no alternative => 0',
                    forced_move_rule='loss zero without search; cached scores are placeholders',
                    pairs=[list(pair) for pair in PAIRS],
                    formula='P_higher^(1-alpha) * P_1000-1200^alpha; legal Top-1',
                    tie_rule='smallest policy index', augmentation=False,
                    denominator='all original validation positions including duplicates, zero loss and forced moves; passes excluded')
    possible = {}
    requested_bits = np.zeros(len(records), dtype=np.uint64)
    for higher, lower in PAIRS:
        print(f'Collect possible moves: {higher} -> {lower}', flush=True)
        possible[higher] = fit.possible_top1(logits[higher], logits[lower], legal)
        requested_bits |= np.sum(possible[higher].astype(np.uint64)*common.BITS, axis=1, dtype=np.uint64)
    requested_bits |= common.BITS[records['policy']]
    unique_bits = np.zeros(len(states), dtype=np.uint64)
    np.bitwise_or.at(unique_bits, inverse, requested_bits)
    requested = (unique_bits[:, None] & common.BITS) != 0
    analysis = args.output / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    losses = fit.analyze(args, analysis, states, legal[first], requested, identity)[inverse]
    human = losses[np.arange(len(records)), records['policy']]
    target = float(human.mean(dtype=np.float64))
    np.save(args.output / 'human_losses.npy', human.astype('<i2'), allow_pickle=False)
    print(f'HUMAN TARGET: {target:.8f} | {len(records):,} validation positions', flush=True)
    rows = []
    for higher, lower in PAIRS:
        pair = (logits[higher], logits[lower])
        directory = args.output / f'{higher}_to_{lower}'
        directory.mkdir(parents=True, exist_ok=True)
        calculate, compact_moves = compact_metric(pair, possible[higher], losses)
        # Check the reduced search against the original full-legal distribution.
        for alpha in (0, .123, .5, .876, 1):
            compact_chosen, _ = compact_moves(alpha)
            common.require(np.array_equal(compact_chosen, fit.select_moves(pair, legal, alpha)),
                           'Compact and full-legal Top-1 differ')
        best = fit.fit_alpha(pair, legal, losses, target, args.alpha_intervals, directory,
                             metric_function=calculate, refinement_levels=3)
        moves = fit.select_moves(pair, legal, best['alpha'])
        selected = losses[np.arange(len(records)), moves]
        scalar = sum(int(value) for value in selected)
        common.require(scalar == best['total_disc_loss'], 'Independent selected loss sum differs')
        common.require(np.array_equal(moves, compact_moves(best['alpha'])[0]), 'Selected compact move differs')
        np.save(directory / 'selected_moves.npy', moves.astype('i1'), allow_pickle=False)
        np.save(directory / 'selected_losses.npy', selected.astype('<i2'), allow_pickle=False)
        endpoints = [fit.metric(pair, legal, losses, alpha) for alpha in (0, 1)]
        result = dict(status='complete', validation_band=args.band, positions=len(records),
                      higher=roster[higher], lower=roster[lower], human_mean_disc_loss=target,
                      alpha_direction='0=higher-rated training band; 1=1000-1200', best=best, endpoints=endpoints,
                      partial=bool(args.max_positions), identity=identity,
                      verification='hash-checked models/logits, all-alpha envelope coverage, compact/full-legal Top-1 checks and scalar sums passed')
        common.save(directory / 'result.json', result)
        rows.append(dict(higher_band=higher, lower_band=lower, validation_band=args.band, positions=len(records),
                         human_mean_disc_loss=target, higher_mean_disc_loss=endpoints[0]['mean_disc_loss'],
                         lower_mean_disc_loss=endpoints[1]['mean_disc_loss'], alpha=best['alpha'],
                         blend_mean_disc_loss=best['mean_disc_loss'], absolute_error=best['absolute_error']))
    for model in roster.values():
        common.require(common.file_sha256(Path(model['path'])) == model['sha256'], 'Model changed during run')
    fit.write_csv(args.output / 'results.csv', rows)
    common.save(args.output / 'results.json', dict(status='complete', identity=identity, rows=rows,
                note='Validation alpha search only; no test evaluation was requested'))
    common.save(args.output / 'run_status.json', dict(status='complete', rows=rows))
    print('VALIDATION MODEL PAIRS COMPLETE:', rows, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=fit.ROOT / 'experiment_bat/20261010_quest_rating_bands')
    parser.add_argument('--band', default='1400-1600', choices=common.MODEL_BANDS)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--alpha-intervals', type=int, default=1000)
    parser.add_argument('--max-positions', type=int, default=0)
    parser.add_argument('--reuse-cache', type=Path, action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    common.require(args.workers > 0 and args.alpha_intervals > 0 and args.max_positions >= 0 and 1 <= args.level <= 60,
                   'Invalid arguments')
    args.reuse_caches = args.reuse_cache or [fit.HERE / 'output/1400-1600_l10/val',
                                           fit.HERE / 'output/train_1400-1600_l10']
    suffix = f'validation_{args.band}_pairs_1000_l{args.level}' + (f'_smoke{args.max_positions}' if args.max_positions else '')
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
