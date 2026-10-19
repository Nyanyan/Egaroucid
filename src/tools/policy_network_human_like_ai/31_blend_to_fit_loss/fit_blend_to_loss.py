"""Fit adjacent fine-tuned Quest policies to validation human L10 disc loss."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
for directory in ('70_avg_disc_loss', '00_data_processing'):
    sys.path.insert(0, str(HERE.parent / directory))
import calculate_quest_model_disc_loss as common
from policy_data import position_samples_to_targets, calc_legal_batch
from selected_loss import possible_top1, analyze


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(args, split):
    stage = 'validation_evaluation' if split == 'val' else 'evaluation'
    directory = args.campaign / stage / 'datasets' / f'quest_{args.band}_{split}'
    proof = common.read(directory / 'summary.json')
    common.require(proof['status'] == 'passed', 'Dataset was not verified')
    info = proof['identity']['dataset']
    source = Path(info['directory'])
    common.require(common.file_sha256(source / 'manifest.json') == info['manifest_sha256'],
                   'Source manifest changed')
    manifest = common.read(source / 'manifest.json')
    common.require(manifest['stage'] == 'board_split', 'Expected unaugmented data')
    entries = proof['identity']['source_files']
    common.require(all(e['path'].startswith(split + '/') for e in entries), 'Wrong split')
    for entry in entries:
        common.require(common.file_sha256(source / entry['path']) == entry['sha256'],
                       'Source file changed: ' + entry['path'])
    arrays = {}
    for filename, digest in proof['files'].items():
        common.require(common.file_sha256(directory / filename) == digest, 'Dataset cache changed')
        arrays[filename[:-4]] = np.load(directory / filename, allow_pickle=False)
    records, legal = arrays['records'], arrays['legal']
    common.require(len(records) == info['positions'] == manifest['splits'][split]['positions'],
                   'Full split count differs')
    common.require(np.array_equal(calc_legal_batch(records['player'], records['opponent']), legal),
                   'Legal masks differ from boards')
    labels = records['policy'].astype(np.int64)
    common.require(np.all((labels >= 0) & (labels < 64)) and np.all(legal & common.BITS[labels]),
                   'Invalid human labels')
    take = min(len(records), args.max_positions) if args.max_positions else len(records)
    identity = dict(dataset=info, files=proof['files'], source_files=entries,
                    evaluated_positions=take, partial=bool(args.max_positions))
    return records[:take], legal[:take], arrays['game_ids'][:take], identity


def model_roster(args):
    bands = list(common.MODEL_BANDS)
    common.require(args.band in bands and bands.index(args.band) > 0,
                   'Target band needs a lower neighboring band; under_800 has no lower model')
    weaker = bands[bands.index(args.band)-1]
    roster = []
    for band in (args.band, weaker):
        path = args.campaign / 'evaluation/models' / ('finetune_' + band) / 'quest_800-1000_test/summary.json'
        model = common.read(path)['identity']['model']
        common.require(model['method'] == 'finetune' and model['band'] == band and
                       common.file_sha256(Path(model['path'])) == model['sha256'], 'FT model changed')
        roster.append(model)
    return roster


def infer_logits(records, roster, directory, identity):
    proof_path = directory / 'logits_identity.json'
    if proof_path.exists():
        proof = common.read(proof_path)
        common.require(all(proof['identity'][key] == identity[key] for key in ('dataset', 'models', 'board_sha256')),
                       'Inference dataset/model identity differs')
        arrays = []
        for index in range(len(roster)):
            path = directory / f'logits_{index}.npy'
            common.require(common.file_sha256(path) == proof['files'][path.name], 'Logits changed')
            value = np.load(path, allow_pickle=False)
            common.require(value.shape == (len(records), 64) and np.isfinite(value).all(), 'Bad logits cache')
            arrays.append(value)
        return arrays
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    arrays, files = [], {}
    for index, info in enumerate(roster):
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(info['path'], compile=False)
        common.require(model.input_shape == (None, 128) and model.output_shape == (None, 64), 'Model shape differs')
        # Paper Conv8 models scatter 60 outputs into 64 squares; central
        # squares are occupied forever. Use pre-softmax logits in both formats.
        if any(layer.name == 'policy_logits60' for layer in model.layers):
            output_layer = model.get_layer('policy_logits60').output
            output_policies = [p for p in range(64) if p not in (27, 28, 35, 36)]
        else:
            common.require(isinstance(model.layers[-1], tf.keras.layers.Softmax), 'Expected final Softmax')
            output_layer = model.layers[-1].input
            output_policies = list(range(64))
        logits_model = tf.keras.Model(model.inputs, output_layer)
        path = directory / f'logits_{index}.npy'
        logits = np.lib.format.open_memmap(path, mode='w+', dtype='<f4', shape=(len(records), 64))
        started = last = time.monotonic()
        for start in range(0, len(records), 512):
            end = min(start + 512, len(records))
            features, _ = position_samples_to_targets(records[start:end])
            raw_values = np.asarray(logits_model(features, training=False), dtype=np.float32)
            values = np.zeros((end-start, 64), dtype=np.float32)
            values[:, output_policies] = raw_values
            common.require(np.isfinite(values).all(), 'Non-finite logits')
            if start == 0:
                probabilities = np.asarray(model(features, training=False), dtype=np.float64)
                expected = np.exp(raw_values.astype(np.float64)-raw_values.max(axis=1, keepdims=True))
                expected /= expected.sum(axis=1, keepdims=True)
                np.testing.assert_allclose(probabilities[:, output_policies], expected, rtol=1e-4, atol=1e-6)
            logits[start:end] = values
            if end == len(records) or time.monotonic()-last >= 5:
                elapsed = time.monotonic()-started
                print(f'Inference {info["name"]}: {end:,}/{len(records):,} | elapsed {elapsed:.1f}s | '
                      f'ETA {elapsed*(len(records)-end)/end:.1f}s', flush=True)
                last = time.monotonic()
        logits.flush()
        common.require(common.file_sha256(Path(info['path'])) == info['sha256'], 'Model changed during inference')
        files[path.name] = common.file_sha256(path)
        arrays.append(np.asarray(logits).copy())
    common.save(proof_path, dict(identity=identity, files=files))
    return arrays


def evaluate_split(args, split, roster, test_alpha=None):
    print(f'Preparing {args.band}/{split}: unaugmented, all original positions', flush=True)
    records, legal, ids, dataset = load_dataset(args, split)
    directory = args.output / split
    directory.mkdir(parents=True, exist_ok=True)
    keys = np.empty(len(records), dtype=common.KEY_DTYPE)
    for field in keys.dtype.names:
        keys[field] = records[field]
    states, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    identity = dict(dataset=dataset, models=roster, level=args.level,
                    engine={**common.engine_identity(common.default_engine(), 25),
                            'level': args.level, 'clear_cache_per_position': False},
                    protocol='setboard relative player as black; play selected move; analyze; persistent warm TT',
                    positions=len(records), unique_positions=len(states),
                    board_sha256=hashlib.sha256(states.tobytes()).hexdigest(),
                    formula='P_stronger^(1-alpha) * P_weaker^alpha; legal Top-1',
                    tie_rule='smallest policy index', augmentation=False,
                    loss_formula='max(played_score, best_other_move_score) - played_score; no alternative => 0',
                    forced_move_rule='loss zero without search; cached scores are placeholders',
                    denominator='all original positions including duplicate boards, forced moves and zero loss; passes excluded')
    logits = infer_logits(records, roster, directory, identity)
    if split == 'val':
        # Union original-record candidates after inference, so repeated boards
        # remain covered even if different GPU batch shapes round logits slightly.
        candidates = possible_top1(logits[0], logits[1], legal)
        candidate_bits = np.sum(candidates.astype(np.uint64)*common.BITS, axis=1, dtype=np.uint64)
        unique_bits = np.zeros(len(states), dtype=np.uint64)
        np.bitwise_or.at(unique_bits, inverse, candidate_bits)
        requested = (unique_bits[:, None] & common.BITS) != 0
        requested[inverse, records['policy']] = True
        identity['requested_moves'] = 'human labels plus every possible Top-1 over continuous alpha [0,1]'
    else:
        common.require(test_alpha is not None, 'Test alpha must already be selected on validation')
        requested = np.zeros((len(states), 64), dtype=bool)
        chosen = select_moves(logits, legal, test_alpha)
        requested[inverse, chosen] = True
        identity.update(requested_moves='fixed validation-alpha Top-1 only', fixed_alpha=test_alpha)
    losses = analyze(args, directory, states, legal[first], requested, identity)[inverse]
    human = None
    if split == 'val':
        human = losses[np.arange(len(records)), records['policy']]
        common.require(np.isfinite(human).all() and np.all(human >= 0), 'Invalid human losses')
        np.save(directory / 'human_losses.npy', human, allow_pickle=False)
        print(f'{split} HUMAN: n={len(records):,}, mean disc loss={human.mean(dtype=np.float64):.8f}', flush=True)
    return logits, legal, losses, human, ids, identity


def select_moves(logits, legal, alpha):
    common.require(0 <= alpha <= 1, 'Alpha outside [0,1]')
    stronger, weaker = logits
    scores = stronger.astype(np.float64) if alpha == 0 else weaker.astype(np.float64) if alpha == 1 else (
        (1-alpha) * stronger.astype(np.float64) + alpha * weaker.astype(np.float64))
    scores[(legal[:, None] & common.BITS) == 0] = -np.inf
    return scores.argmax(axis=1)


def metric(logits, legal, losses, alpha):
    moves = select_moves(logits, legal, alpha)
    selected = losses[np.arange(len(moves)), moves]
    common.require(np.isfinite(selected).all() and np.all(selected >= 0), 'Invalid selected losses')
    return dict(alpha=float(alpha), positions=len(moves), total_disc_loss=float(selected.sum(dtype=np.float64)),
                mean_disc_loss=float(selected.mean(dtype=np.float64)), zero_loss_positions=int(np.count_nonzero(selected == 0)))


def fit_alpha(logits, legal, losses, target, intervals, output, metric_function=None, refinement_levels=1):
    rows = []
    started = last = time.monotonic()
    # Top-1 can change discontinuously and mean loss need not be monotonic.
    # Evaluate the entire interval, then refine around the closest grid value.
    coarse = np.linspace(0, 1, intervals+1)
    calculate = metric_function or (lambda alpha: metric(logits, legal, losses, alpha))
    for index, alpha in enumerate(coarse):
        row = calculate(alpha)
        row['absolute_error'] = abs(row['mean_disc_loss']-target)
        rows.append(row)
        if index == intervals or time.monotonic()-last >= 5:
            elapsed = time.monotonic()-started
            print(f'Alpha sweep: {index+1}/{intervals+1} | elapsed {elapsed:.1f}s | '
                  f'ETA {elapsed*(intervals-index)/(index+1):.1f}s', flush=True)
            last = time.monotonic()
    best = min(rows, key=lambda r: (r['absolute_error'], r['alpha']))
    step = 1/intervals
    for level in range(refinement_levels):
        for alpha in np.linspace(max(0, best['alpha']-step), min(1, best['alpha']+step), 201):
            row = calculate(alpha)
            row['absolute_error'] = abs(row['mean_disc_loss']-target)
            rows.append(row)
        best = min(rows, key=lambda r: (r['absolute_error'], r['alpha']))
        step /= 100
        if refinement_levels > 1:
            print(f'Alpha refinement {level+1}/{refinement_levels}: alpha={best["alpha"]:.10f}, '
                  f'error={best["absolute_error"]:.10f}', flush=True)
    rows.sort(key=lambda r: r['alpha'])
    write_csv(output / 'validation_alpha_curve.csv', rows)
    best = min(rows, key=lambda r: (r['absolute_error'], r['alpha']))
    observed = [r['mean_disc_loss'] for r in rows]
    common.save(output / 'selected_alpha.json', dict(status='selected_on_validation', target_human_mean_disc_loss=target,
                best=best, coarse_step=1/intervals, refinement_points=201,
                refinement_levels=refinement_levels, finest_nominal_step=step,
                observed_loss_min=min(observed), observed_loss_max=max(observed),
                target_in_observed_range=min(observed) <= target <= max(observed),
                selection='minimum absolute mean loss error; tie => smallest alpha; finite grid, no monotonicity assumption'))
    print(f'ALPHA SELECTED: {best["alpha"]:.8f} | target={target:.8f} | '
          f'blend={best["mean_disc_loss"]:.8f} | error={best["absolute_error"]:.8f}', flush=True)
    return best


def run(args):
    common.save(args.output / 'status.json', dict(status='running', band=args.band, pid=os.getpid(),
                method='selected-move analyze, persistent warm TT', started_unix=time.time()))
    roster = model_roster(args)
    val = evaluate_split(args, 'val', roster)
    selected = fit_alpha(*val[:3], float(val[3].mean(dtype=np.float64)), args.alpha_intervals, args.output)
    selection_identity = dict(validation=val[5], alpha_intervals=args.alpha_intervals,
                              selected=selected)
    common.save(args.output / 'selection_identity.json', selection_identity)
    val_ids = val[4]
    del val
    test = evaluate_split(args, 'test', roster, test_alpha=selected['alpha'])
    common.require(not np.intersect1d(val_ids, test[4]).size, 'Game IDs overlap between val and test')
    alpha = selected['alpha']
    result = metric(*test[:3], alpha)
    moves = select_moves(test[0], test[1], alpha)
    np.save(args.output / 'test/blend_moves.npy', moves.astype('i1'), allow_pickle=False)
    chosen = test[2][np.arange(len(moves)), moves]
    np.save(args.output / 'test/blend_losses.npy', chosen, allow_pickle=False)
    common.require(float(chosen.sum()) == sum(float(v) for v in chosen), 'Scalar loss sum differs')
    summary = dict(status='complete', band=args.band, stronger=roster[0], weaker=roster[1],
                   alpha=alpha, validation=selection_identity, test_identity=test[5], test_blend=result,
                   partial=bool(args.max_positions), validation_test_game_overlap=0)
    common.save(args.output / 'result.json', summary)
    write_csv(args.output / 'test_result.csv', [result])
    common.save(args.output / 'status.json', dict(status='complete', band=args.band, alpha=alpha,
                test_mean_disc_loss=result['mean_disc_loss'], positions=result['positions']))
    print(f'COMPLETE {args.band}: alpha={alpha:.8f}, test mean disc loss={result["mean_disc_loss"]:.8f} '
          f'(n={result["positions"]:,})', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'experiment_bat/20261010_quest_rating_bands')
    parser.add_argument('--band', default='1400-1600', choices=common.MODEL_BANDS[1:])
    parser.add_argument('--all-bands', action='store_true', help='Run every band with a lower neighboring model')
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--alpha-intervals', type=int, default=1000)
    parser.add_argument('--max-positions', type=int, default=0, help='Prefix smoke test; default full val and test')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    common.require(args.workers > 0 and args.alpha_intervals > 0 and args.max_positions >= 0 and 1 <= args.level <= 60,
                   'Invalid arguments')
    base = args.output or HERE / 'output'
    for band in common.MODEL_BANDS[1:] if args.all_bands else [args.band]:
        args.band = band
        suffix = f'{band}_l{args.level}' + (f'_smoke{args.max_positions}' if args.max_positions else '')
        args.output = base / suffix
        args.output.mkdir(parents=True, exist_ok=True)
        # OS releases the lock after a crash or Ctrl+C. Refuse concurrent runs.
        from fine_tune_policy import run_lock
        with run_lock(args.output):
            try:
                run(args)
            except BaseException as error:
                common.save(args.output / 'status.json', dict(status='failed', error=str(error), pid=os.getpid()))
                raise


if __name__ == '__main__':
    sys.path.insert(0, str(HERE.parent / '11_fine_tuning'))
    main()
