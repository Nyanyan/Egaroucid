"""Recheck frozen data, saved moves, analyze losses and exact mean arithmetic."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import fit_blend_to_loss as fit

common = fit.common


def run(output, campaign):
    result = common.read(output / 'result.json')
    common.require(result['status'] == 'complete', 'Calculation is not complete')
    args = SimpleNamespace(campaign=campaign, band=result['band'],
                           max_positions=result['test_identity']['dataset']['evaluated_positions'] if result['partial'] else 0)
    for model in (result['stronger'], result['weaker']):
        common.require(common.file_sha256(Path(model['path'])) == model['sha256'], 'Model changed')
    metrics, ids, humans = {}, {}, None
    for split in ('val', 'test'):
        directory = output / split
        records, legal, ids[split], dataset = fit.load_dataset(args, split)
        proof = common.read(directory / 'analyze_identity.json')
        common.require(proof['dataset'] == dataset and not proof['engine']['clear_cache_per_position'],
                       'Wrong dataset or engine conditions')
        pairs = np.load(directory / 'board_move_pairs.npy', allow_pickle=False)
        import hashlib
        common.require(hashlib.sha256(pairs.tobytes()).hexdigest() == proof['board_move_sha256'], 'Pair order changed')
        values = np.load(directory / 'analyzed_moves.npy', allow_pickle=False)
        common.require(values.shape == (len(pairs), 3) and np.all(values[:, 2] >= 0), 'Incomplete analyze cache')
        common.require(np.array_equal(np.maximum(values[:, 0], values[:, 1])-values[:, 0], values[:, 2]),
                       'Analyze loss formula differs')
        common.require(np.array_equal(np.unique(pairs), pairs), 'Pairs are not sorted and unique')
        logits_proof = common.read(directory / 'logits_identity.json')
        logits = []
        for index in (0, 1):
            path = directory / f'logits_{index}.npy'
            common.require(common.file_sha256(path) == logits_proof['files'][path.name], 'Logits changed')
            logits.append(np.load(path, allow_pickle=False))
        moves = fit.select_moves(logits, legal, result['alpha'])

        def losses_for(policies):
            queries = np.empty(len(records), dtype=pairs.dtype)
            queries['player'], queries['opponent'], queries['policy'] = records['player'], records['opponent'], policies
            indexes = np.searchsorted(pairs, queries)
            common.require(np.all(indexes < len(pairs)) and np.array_equal(pairs[indexes], queries),
                           'Missing saved board/move analysis')
            return values[indexes, 2]

        loss = losses_for(moves)
        total = sum(int(value) for value in loss)
        common.require(total == int(loss.sum(dtype=np.int64)), 'Independent scalar sum differs')
        metrics[split] = dict(alpha=result['alpha'], positions=len(records), total_disc_loss=float(total),
                             mean_disc_loss=total/len(records), zero_loss_positions=int(np.count_nonzero(loss == 0)))
        if split == 'val':
            humans = losses_for(records['policy'])
            common.require(np.array_equal(humans, np.load(directory / 'human_losses.npy', allow_pickle=False)),
                           'Human losses differ from selected-move analysis')
        else:
            common.require(np.array_equal(moves, np.load(directory / 'blend_moves.npy', allow_pickle=False)) and
                           np.array_equal(loss, np.load(directory / 'blend_losses.npy', allow_pickle=False)),
                           'Saved test moves/losses differ')
    common.require(not np.intersect1d(ids['val'], ids['test']).size, 'Validation/test game IDs overlap')
    # Convert any earlier float32 averages to exact totals / original counts.
    # Do not change alpha: verify the same alpha still minimizes the recorded curve.
    import csv
    with (output / 'validation_alpha_curve.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    target = sum(int(value) for value in humans)/len(humans)
    for row in rows:
        row['alpha'] = float(row['alpha'])
        row['positions'] = int(row['positions'])
        row['total_disc_loss'] = float(row['total_disc_loss'])
        row['zero_loss_positions'] = int(row['zero_loss_positions'])
        row['mean_disc_loss'] = row['total_disc_loss']/row['positions']
        row['absolute_error'] = abs(row['mean_disc_loss']-target)
    best = min(rows, key=lambda row: (row['absolute_error'], row['alpha']))
    common.require(best['alpha'] == result['alpha'] and best['total_disc_loss'] == metrics['val']['total_disc_loss'],
                   'Selected alpha differs from independent curve/move check')
    selection = common.read(output / 'selected_alpha.json')
    selection.update(target_human_mean_disc_loss=target, best=best,
                     observed_loss_min=min(row['mean_disc_loss'] for row in rows),
                     observed_loss_max=max(row['mean_disc_loss'] for row in rows))
    selection['target_in_observed_range'] = selection['observed_loss_min'] <= target <= selection['observed_loss_max']
    fit.write_csv(output / 'validation_alpha_curve.csv', rows)
    common.save(output / 'selected_alpha.json', selection)
    identity = common.read(output / 'selection_identity.json')
    identity['selected'] = best
    common.save(output / 'selection_identity.json', identity)
    result.update(validation=identity, test_blend=metrics['test'])
    common.save(output / 'result.json', result)
    fit.write_csv(output / 'test_result.csv', [metrics['test']])
    status = common.read(output / 'status.json')
    status['test_mean_disc_loss'] = metrics['test']['mean_disc_loss']
    common.save(output / 'status.json', status)
    verification = dict(status='passed', alpha=result['alpha'], human_validation_mean_disc_loss=target,
                        validation_blend=metrics['val'], test_blend=metrics['test'],
                        validation_absolute_error=best['absolute_error'],
                        target_in_observed_range=selection['target_in_observed_range'],
                        checks='source/model/logit hashes, legal geometric Top-1, board/move lookup, analyze formula, '
                               'saved human/test losses, independent scalar sums, curve optimum and disjoint game IDs')
    common.save(output / 'verification.json', verification)
    print(verification)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=fit.HERE / 'output/1400-1600_l10')
    parser.add_argument('--campaign', type=Path, default=fit.ROOT / 'experiment_bat/20261010_quest_rating_bands')
    args = parser.parse_args()
    run(args.output, args.campaign)
