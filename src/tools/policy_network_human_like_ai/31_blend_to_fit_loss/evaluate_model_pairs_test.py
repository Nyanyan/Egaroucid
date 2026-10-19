"""Evaluate frozen validation-selected blends on common Quest test positions."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import fit_validation_model_pairs as pairs_script

fit = pairs_script.fit
common = fit.common
sys.path.insert(0, str(fit.HERE.parent))
from policy_accuracy import equivalent_policy_mask


def agreement(moves, labels, targets):
    hits = int(np.count_nonzero(common.BITS[moves] & targets))
    exact = int(np.count_nonzero(moves == labels))
    common.require(hits == sum(bool(int(target) & (1 << int(move))) for target, move in zip(targets, moves)),
                   'Scalar symmetry-aware agreement differs')
    common.require(exact == sum(int(move) == int(label) for move, label in zip(moves, labels)),
                   'Scalar exact agreement differs')
    return dict(top1_hits=hits, top1_accuracy=hits/len(moves), top1_accuracy_percent=100*hits/len(moves),
                exact_top1_hits=exact, exact_top1_accuracy=exact/len(moves))


def run(args):
    started = time.monotonic()
    common.save(args.output / 'run_status.json', dict(status='running', pid=os.getpid(), started_unix=time.time(),
                mode='test loss and symmetry-aware Top-1; fixed validation alphas'))
    calibration = common.read(args.calibration / 'results.json')
    common.require(calibration['status'] == 'complete' and
                   common.read(args.calibration / 'verification.json')['status'] == 'passed', 'Unverified calibration')
    common.require(calibration['identity']['level'] == args.level and not calibration['identity']['dataset']['partial'],
                   'Calibration level differs or calibration is partial')
    fixed = calibration['rows']
    common.require([(row['higher_band'], row['lower_band']) for row in fixed] == list(pairs_script.PAIRS) and
                   all(row['validation_band'] == args.band for row in fixed), 'Calibration pairs or dataset band differ')
    roster = pairs_script.models(args)
    common.require(calibration['identity']['models'] == list(roster.values()), 'Calibration model identities changed')
    records, legal, ids, dataset = fit.load_dataset(args, 'test')
    val_args = SimpleNamespace(campaign=args.campaign, band=args.band, max_positions=0)
    _, _, validation_ids, validation_dataset = fit.load_dataset(val_args, 'val')
    common.require(validation_dataset == calibration['identity']['dataset'], 'Calibration validation data changed')
    common.require(not np.intersect1d(ids, validation_ids).size, 'Test and validation game IDs overlap')
    directory = args.campaign / 'evaluation/datasets' / f'quest_{args.band}_test'
    target_path = directory / 'symmetry_targets.npy'
    common.require(common.file_sha256(target_path) == dataset['files']['symmetry_targets.npy'], 'Target cache changed')
    targets = np.load(target_path, allow_pickle=False)[:len(records)]
    # Rebuild board-preserving equivalents from the source boards/actual move,
    # rather than trusting only the cached bit mask.
    rebuilt = np.empty(len(records), dtype=np.uint64)
    for start in range(0, len(records), 32768):
        end = min(start+32768, len(records))
        block = records[start:end]
        mask = equivalent_policy_mask(block['player'], block['opponent'], block['policy'].astype(np.int64))
        rebuilt[start:end] = np.sum(mask.astype(np.uint64)*common.BITS, axis=1, dtype=np.uint64)
    common.require(np.array_equal(rebuilt, targets) and not np.any(targets & ~legal), 'Symmetry target definition differs')
    keys = np.empty(len(records), dtype=common.KEY_DTYPE)
    for field in keys.dtype.names:
        keys[field] = records[field]
    states, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    board_sha = hashlib.sha256(states.tobytes()).hexdigest()
    logits = pairs_script.load_logits(args, records, roster, dataset, board_sha)
    predictions = np.column_stack([fit.select_moves((logits[row['higher_band']], logits[row['lower_band']]), legal, row['alpha'])
                                   for row in fixed]).astype('i1')
    common.require(np.all(legal[:, None] & common.BITS[predictions]), 'Illegal blended test move')
    np.save(args.output / 'predictions.npy', predictions, allow_pickle=False)
    agreements = [agreement(predictions[:, index], records['policy'], targets) for index in range(len(fixed))]
    for row, metric in zip(fixed, agreements):
        print(f'TEST AGREEMENT {row["higher_band"]} -> {row["lower_band"]} | alpha={row["alpha"]:.12g} | '
              f'Top-1={metric["top1_accuracy_percent"]:.6f}% ({metric["top1_hits"]:,}/{len(records):,})', flush=True)
    identity = dict(dataset=dataset, models=list(roster.values()), level=args.level,
                    engine={**common.engine_identity(common.default_engine(), 25), 'level': args.level,
                            'clear_cache_per_position': False},
                    board_sha256=board_sha, positions=len(records), unique_positions=len(states),
                    protocol='setboard relative player as black; play selected move; analyze; persistent warm TT',
                    loss_formula='max(played_score, best_other_move_score) - played_score; no alternative => 0',
                    forced_move_rule='loss zero without search; cached scores are placeholders',
                    formula='P_higher^(1-alpha) * P_1000-1200^alpha; legal Top-1',
                    tie_rule='smallest policy index', calibration_directory=str(args.calibration.resolve()),
                    calibration_results_sha256=common.file_sha256(args.calibration / 'results.json'),
                    fixed_pairs=[dict(higher=row['higher_band'], lower=row['lower_band'], alpha=row['alpha']) for row in fixed],
                    agreement_definition='Top-1 including actual human move and board-preserving rotation/reflection equivalents',
                    predictions_sha256=common.file_sha256(args.output / 'predictions.npy'),
                    validation_test_game_overlap=0,
                    denominator='all original test positions including duplicates, zero loss and forced moves; passes excluded')
    common.save(args.output / 'identity.json', identity)
    requested = np.zeros((len(states), 64), dtype=bool)
    requested[inverse[:, None], predictions] = True
    analysis = args.output / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    loss_table = fit.analyze(args, analysis, states, legal[first], requested, identity)
    losses = loss_table[inverse[:, None], predictions]
    common.require(np.isfinite(losses).all() and np.all(losses >= 0), 'Incomplete test loss')
    np.save(args.output / 'losses.npy', losses.astype('<i2'), allow_pickle=False)
    rows = []
    for index, row in enumerate(fixed):
        total = int(losses[:, index].sum(dtype=np.int64))
        common.require(total == sum(int(value) for value in losses[:, index]), 'Scalar test loss sum differs')
        rows.append(dict(higher_band=row['higher_band'], lower_band=row['lower_band'], alpha=row['alpha'],
                         test_band=args.band, positions=len(records), total_disc_loss=total,
                         mean_disc_loss=total/len(records), zero_loss_positions=int(np.count_nonzero(losses[:, index] == 0)),
                         **agreements[index]))
    for model in roster.values():
        common.require(common.file_sha256(Path(model['path'])) == model['sha256'], 'Model changed during test evaluation')
    common.require(identity['calibration_results_sha256'] == common.file_sha256(args.calibration / 'results.json'),
                   'Validation alpha file changed during test evaluation')
    result = dict(status='complete', identity=identity, rows=rows, partial=bool(args.max_positions),
                  elapsed_sec=time.monotonic()-started, losses_sha256=common.file_sha256(args.output / 'losses.npy'),
                  verification='fixed validation alphas, source/model/logit hashes, legal moves, rebuilt symmetry targets, '
                               'disjoint game IDs and independent scalar loss/agreement counts passed')
    common.save(args.output / 'results.json', result)
    fit.write_csv(args.output / 'results.csv', rows)
    common.save(args.output / 'run_status.json', dict(status='complete', rows=rows))
    print('TEST MODEL PAIRS COMPLETE:', rows, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=fit.ROOT / 'experiment_bat/20261010_quest_rating_bands')
    parser.add_argument('--calibration', type=Path, default=fit.HERE / 'output/validation_1400-1600_pairs_1000_l10')
    parser.add_argument('--band', default='1400-1600')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--max-positions', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    common.require(args.workers > 0 and args.max_positions >= 0 and 1 <= args.level <= 60, 'Invalid arguments')
    suffix = f'test_{args.band}_pairs_1000_l{args.level}' + (f'_smoke{args.max_positions}' if args.max_positions else '')
    args.output = args.output or fit.HERE / 'output' / suffix
    args.output.mkdir(parents=True, exist_ok=True)
    args.reuse_caches = [fit.HERE / 'output/1400-1600_l10/test', args.calibration / 'analysis',
                        fit.HERE / 'output/1400-1600_l10/val', fit.HERE / 'output/train_1400-1600_l10']
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
