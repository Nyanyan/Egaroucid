"""Independent cell-array D4 audit of validation/test symmetry-aware Top-1."""
import argparse
from pathlib import Path
import numpy as np
import fit_blend_to_loss as fit

common = fit.common
CELL_BITS = common.BITS[::-1]  # a1 first: policy 63, then b1: policy 62, ...


def cell_targets(records):
    """Rotate/reflect 8x8 cell arrays; do not use policy_accuracy mappings."""
    player = ((records['player'][:, None] & CELL_BITS) != 0).reshape(-1, 8, 8)
    opponent = ((records['opponent'][:, None] & CELL_BITS) != 0).reshape(-1, 8, 8)
    labels = np.zeros((len(records), 64), dtype=bool)
    labels[np.arange(len(records)), 63-records['policy'].astype(np.int64)] = True
    labels = labels.reshape(-1, 8, 8)
    equivalents = np.zeros_like(labels)
    for rotations in range(4):
        rp, ro, rl = [np.rot90(array, rotations, axes=(1, 2)) for array in (player, opponent, labels)]
        for mirrored in (False, True):
            tp, to, tl = [np.flip(array, axis=2) if mirrored else array for array in (rp, ro, rl)]
            invariant = np.all(tp == player, axis=(1, 2)) & np.all(to == opponent, axis=(1, 2))
            equivalents[invariant] |= tl[invariant]
    return np.sum(equivalents.reshape(-1, 64).astype(np.uint64)*CELL_BITS, axis=1, dtype=np.uint64)


def coord(policy):
    index = 63-int(policy)
    return chr(97+index % 8)+str(index//8+1)


def run(campaign, output):
    root = fit.HERE / 'output'
    rows, examples, checked_positions = [], [], 0
    for split in ('val', 'test'):
        args = argparse.Namespace(campaign=campaign, band='1400-1600', max_positions=0)
        records, legal, _, dataset = fit.load_dataset(args, split)
        checked_positions += len(records)
        stage = 'validation_evaluation' if split == 'val' else 'evaluation'
        cache = campaign / stage / 'datasets' / f'quest_1400-1600_{split}'
        targets = np.load(cache / 'symmetry_targets.npy', allow_pickle=False)
        for start in range(0, len(records), 4096):
            end = min(start+4096, len(records))
            common.require(np.array_equal(cell_targets(records[start:end]), targets[start:end]),
                           f'Independent cell-array symmetry differs: {split} at {start}')
        common.require(not np.any(targets & ~legal), 'Equivalent targets include an illegal move')
        if split == 'val':
            directory = root / 'validation_1400-1600_pairs_1000_l10'
            source_rows = common.read(directory / 'validation_metrics.json')['rows']
            predictions = np.column_stack([np.load(directory / f'{row["higher_band"]}_to_{row["lower_band"]}' /
                                                   'selected_moves.npy', allow_pickle=False) for row in source_rows])
        else:
            directory = root / 'test_1400-1600_pairs_1000_l10'
            source_rows = common.read(directory / 'results.json')['rows']
            predictions = np.load(directory / 'predictions.npy', allow_pickle=False)
        for index, expected in enumerate(source_rows):
            moves = predictions[:, index]
            common.require(np.all(legal & common.BITS[moves]), 'Illegal selected move')
            exact = moves == records['policy']
            correct = (targets & common.BITS[moves]) != 0
            hits = sum(bool(int(mask) & (1 << int(move))) for mask, move in zip(targets, moves))
            common.require(hits == int(correct.sum()) == expected['top1_hits'] and
                           int(exact.sum()) == expected['exact_top1_hits'], 'Reported agreement counts differ')
            rows.append(dict(split=split, higher_band=expected['higher_band'], lower_band=expected['lower_band'],
                             positions=len(records), exact_hits=int(exact.sum()), symmetry_aware_hits=hits,
                             additional_symmetry_hits=int(np.count_nonzero(correct & ~exact)),
                             exact_accuracy_percent=100*int(exact.sum())/len(records),
                             symmetry_aware_accuracy_percent=100*hits/len(records),
                             multiple_equivalent_target_positions=int(np.count_nonzero(targets & (targets-np.uint64(1))))))
            additional = np.flatnonzero(correct & ~exact)
            if len(additional):
                i = int(additional[0])
                record = records[i]
                examples.append(dict(split=split, higher_band=expected['higher_band'], position_index=i,
                                     player_hex=f'{int(record["player"]):016x}',
                                     opponent_hex=f'{int(record["opponent"]):016x}',
                                     human_move=coord(record['policy']), selected_move=coord(moves[i]),
                                     accepted_moves=[coord(p) for p in range(64) if int(targets[i]) & (1 << p)],
                                     occupied_squares=bin(int(record['player'] | record['opponent'])).count('1')))
        print(f'{split}: all {len(records):,} target sets match independent cell-array rotations/reflections')
    result = dict(status='passed', checked_positions=checked_positions,
                  definition='preserve player and opponent separately; transform the actual human move only under these symmetries',
                  rows=rows, examples=examples,
                  audit='all cached correct-move sets matched an independent 8x8 cell-array implementation; reported hit counts reproduced')
    common.save(output, result)
    print(rows)
    print(examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=fit.ROOT / 'experiment_bat/20261010_quest_rating_bands')
    parser.add_argument('--output', type=Path, default=fit.HERE / 'output/symmetry_accuracy_audit.json')
    args = parser.parse_args()
    run(args.campaign, args.output)
