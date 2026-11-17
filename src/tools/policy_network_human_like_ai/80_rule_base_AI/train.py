"""Fit a linear softmax of hand-written rules on Quest [800,1000)."""
import argparse
import csv
import os
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize
from common import (HERE, BASE, WORKSPACE, default_data, load_split, save, read,
                    file_sha256, metrics, bootstrap_difference)
from rule_policy import (FEATURE_NAMES, PHASE_NAMES, extract, phase_weights,
                         legal_moves, popcount, manual_weights, RulePolicy)


@njit(cache=True)
def feature_batch(player, opponent):
    offsets = np.zeros(len(player) + 1, dtype=np.int64)
    phases = np.empty((len(player), 3), dtype=np.float32)
    for i in range(len(player)):
        offsets[i+1] = offsets[i] + popcount(legal_moves(player[i], opponent[i]))
        phases[i] = phase_weights(player[i], opponent[i])
    moves = np.empty(offsets[-1], dtype=np.int8)
    features = np.empty((offsets[-1], len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(len(player)):
        m, f = extract(player[i], opponent[i])
        moves[offsets[i]:offsets[i+1]] = m
        features[offsets[i]:offsets[i+1]] = f
    return offsets, moves, features, phases


def prepare(records, directory, split):
    directory.mkdir(parents=True, exist_ok=True)
    # A source/code identity prevents accidental cache reuse after feature changes.
    import hashlib
    identity = dict(records_sha256=hashlib.sha256(records.tobytes()).hexdigest(),
                    feature_code_sha256=file_sha256(HERE / 'rule_policy.py'),
                    batch_code_sha256=file_sha256(Path(__file__)))
    proof = directory / 'identity.json'
    if proof.exists():
        if read(proof) != identity:
            raise ValueError('Feature cache differs; use a fresh output directory')
        return tuple(np.load(directory / (name + '.npy'), mmap_mode='r')
                     for name in ('offsets', 'moves', 'features', 'phases'))
    started = time.monotonic()
    chunks = []
    for start in range(0, len(records), 20000):
        block = records[start:start+20000]
        chunks.append(feature_batch(block['player'].copy(), block['opponent'].copy()))
        done = min(start+20000, len(records))
        elapsed = time.monotonic() - started
        print(f'features {split}: {done:,}/{len(records):,} | elapsed {elapsed:.1f}s | '
              f'ETA {elapsed*(len(records)-done)/done:.1f}s', flush=True)
    offsets = np.concatenate([np.array([0]), np.cumsum(np.concatenate([np.diff(c[0]) for c in chunks]))])
    arrays = (offsets,) + tuple(np.concatenate([c[i] for c in chunks]) for i in range(1, 4))
    for name, array in zip(('offsets', 'moves', 'features', 'phases'), arrays):
        np.save(directory / (name + '.npy'), array)
    save(proof, identity)
    return arrays


def predict_cache(cache, weights, records):
    offsets, moves, features, phases = cache
    predictions = np.empty(len(records), dtype=np.int8)
    probabilities = np.zeros((len(records), 64), dtype=np.float64)
    for start in range(0, len(records), 20000):
        stop = min(start+20000, len(records))
        groups = np.repeat(np.arange(stop-start), np.diff(offsets[start:stop+1]))
        a, b = offsets[start], offsets[stop]
        logits = np.sum((features[a:b].astype(np.float64) @ weights) * phases[start:stop][groups], axis=1)
        local = offsets[start:stop+1] - a
        maxima = np.maximum.reduceat(logits, local[:-1])
        exp = np.exp(logits - maxima[groups])
        prob = exp / np.add.reduceat(exp, local[:-1])[groups]
        probabilities[start+groups, moves[a:b]] = prob
        predictions[start:stop] = np.argmax(probabilities[start:stop], axis=1)
    return predictions, probabilities


class Objective:
    def __init__(self, cache, records, device):
        import torch
        self.torch = torch
        offsets, moves, features, phases = cache
        self.n = len(records)
        self.x = torch.tensor(np.asarray(features), device=device)
        self.phase = torch.tensor(np.asarray(phases), device=device)
        self.groups = torch.tensor(np.repeat(np.arange(self.n), np.diff(offsets)), dtype=torch.int64, device=device)
        targets = np.empty(self.n, dtype=np.int64)
        for i in range(self.n):
            positions = np.flatnonzero(moves[offsets[i]:offsets[i+1]] == records['policy'][i])
            if len(positions) != 1:
                raise ValueError('Feature candidate label missing')
            targets[i] = offsets[i] + positions[0]
        self.targets = torch.tensor(targets, device=device)
        self.device = device

    def __call__(self, flat, penalty):
        t = self.torch
        weights = t.tensor(flat.reshape(-1, 3), dtype=t.float32, device=self.device)
        with t.no_grad():
            logits = (self.x @ weights * self.phase[self.groups]).sum(1)
            maxima = t.full((self.n,), -float('inf'), device=self.device)
            maxima.scatter_reduce_(0, self.groups, logits, reduce='amax', include_self=False)
            exps = t.exp(logits - maxima[self.groups])
            sums = t.zeros(self.n, device=self.device).scatter_add_(0, self.groups, exps)
            losses = maxima.double() + sums.double().log() - logits[self.targets].double()
            loss = losses.mean().item() + .5 * penalty * float(flat @ flat)
            residual = exps / sums[self.groups]
            residual[self.targets] -= 1
            gradient = (self.x.T @ (residual[:, None] * self.phase[self.groups]) / self.n).cpu().numpy().astype(np.float64)
            gradient += penalty * flat.reshape(-1, 3)
        return loss, gradient.ravel()


def model_json(weights, **metadata):
    return dict(kind='linear_softmax_hand_written_rules', features=list(FEATURE_NAMES),
                phases=list(PHASE_NAMES), weights=weights.tolist(),
                phase_formula='t=(occupied-4)/60; [max(1-2t,0), 1-abs(2t-1), max(2t-1,0)]',
                policy_formula='softmax over every legal move of features @ weights @ phase',
                tie_rule='ascending policy bit index, h8=0, a1=63', **metadata)


def run(args):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive handle is released automatically on crashes; stale lock files are harmless.
    import msvcrt
    lock = (output / 'run.lock').open('a+b')
    lock.seek(0)
    lock.write(b'0')
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    started = time.monotonic()
    splits = {s: load_split(args.data, s) for s in ('train', 'val', 'test')}
    for a, b in (('train','val'), ('train','test'), ('val','test')):
        if np.intersect1d(splits[a][1], splits[b][1]).size:
            raise ValueError('Game leakage between ' + a + ' and ' + b)
    identity = dict(data=str(args.data.resolve()), manifest_sha256=file_sha256(args.data / 'manifest.json'),
                    seed=args.seed, counts={s: len(v[0]) for s,v in splits.items()},
                    source_hashes={p.name: file_sha256(p) for p in (HERE/'rule_policy.py', HERE/'train.py', HERE/'common.py')},
                    augmentation='D4-invariant rules: eight transformed copies have identical loss and gradient; train computed once per orbit',
                    selection='highest validation symmetry-aware top1; break ties by lower human-move NLL',
                    optimizer='L-BFGS-B; uniform per-position cross entropy plus L2',
                    l2=args.l2, max_iterations=args.iterations, device=args.device,
                    parameters=3*len(FEATURE_NAMES), test_used_for_selection=False,
                    split_game_ids_disjoint=True)
    save(output / 'identity.json', identity)
    caches = {s: prepare(v[0], output/'features'/s, s) for s,v in splits.items()}
    objective = Objective(caches['train'], splits['train'][0], args.device)
    best = None
    history = []
    total = len(args.l2) * args.iterations
    steps = 0
    for penalty in args.l2:
        iteration = 0
        def callback(flat):
            nonlocal best, iteration, steps
            iteration += 1
            steps += 1
            weights = flat.reshape(-1, 3)
            moves, prob = predict_cache(caches['val'], weights, splits['val'][0])
            stats, _ = metrics(splits['val'][0], moves, prob)
            elapsed = time.monotonic() - started
            row = dict(l2=penalty, iteration=iteration, elapsed_sec=elapsed, **stats)
            history.append(row)
            rank = (stats['top1_hits'], -stats['human_move_nll'])
            if best is None or rank > best[0]:
                best = (rank, weights.copy(), row)
                save(output/'selected_model.json', model_json(weights, selection=row, identity=identity))
            print(f'fit L2={penalty:g} iter={iteration}/{args.iterations} | '
                  f'val top1={100*stats["top1_accuracy"]:.3f}% NLL={stats["human_move_nll"]:.5f} | '
                  f'elapsed {elapsed:.0f}s | ETA (upper estimate) {elapsed*(total-steps)/max(steps,1):.0f}s', flush=True)
            save(output/'training_history.json', history)
        initial = np.zeros(len(FEATURE_NAMES)*3, dtype=np.float64)
        result = minimize(lambda flat: objective(flat, penalty), initial, jac=True, method='L-BFGS-B',
                          callback=callback, options=dict(maxiter=args.iterations, ftol=1e-10, gtol=1e-6, maxls=30))
        if iteration == 0:
            callback(result.x)
        print(f'optimizer L2={penalty:g}: {result.message}', flush=True)
    weights = best[1]
    # Reload the deliverable model, then evaluate test exactly once after selection.
    selected = RulePolicy.load(output/'selected_model.json')
    if not np.array_equal(selected.weights, weights):
        raise ValueError('Saved model weights differ')
    records, game_ids, legal = splits['test']
    results, hits = {}, {}
    predictions = {}
    for name, w in (('trained_rules', weights), ('manual_rules', manual_weights())):
        moves, prob = predict_cache(caches['test'], w, records)
        results[name], hits[name] = metrics(records, moves, prob)
        predictions[name] = moves
        np.save(output/(name+'_predictions.npy'), moves)
        if name == 'trained_rules':
            np.save(output/'test_probabilities.npy', prob.astype(np.float32))
    offsets, all_moves, features, _ = caches['test']
    greedy = np.array([all_moves[offsets[i] + np.argmax(features[offsets[i]:offsets[i+1],24])]
                       for i in range(len(records))], dtype=np.int8)
    results['most_flips'], hits['most_flips'] = metrics(records, greedy)
    np.save(output/'most_flips_predictions.npy', greedy)
    targets = __import__('policy_accuracy').equivalent_policy_mask(records['player'],records['opponent'],records['policy'].astype(int))
    counts = np.diff(offsets)
    results['uniform_random_expected'] = dict(positions=len(records), top1_accuracy=float((targets.sum(1)/counts).mean()),
                                             exact_top1_accuracy=float((1/counts).mean()))
    baseline_sources = {}
    for method in ('finetune', 'scratch'):
        path = args.baselines / 'models' / (method+'_800-1000') / 'quest_800-1000_test'
        if not path.exists():
            continue
        proof = read(path/'summary.json')
        dataset = args.baselines / 'datasets/quest_800-1000_test'
        saved_records = np.load(dataset/'records.npy', allow_pickle=False)
        saved_ids = np.load(dataset/'game_ids.npy', allow_pickle=False)
        if not np.array_equal(saved_records,records) or not np.array_equal(saved_ids,game_ids):
            raise ValueError('NN baseline test differs')
        if file_sha256(path/'predictions.npy') != proof['predictions_sha256']:
            raise ValueError('NN baseline prediction hash differs')
        moves = np.load(path/'predictions.npy', allow_pickle=False)
        if np.any((legal & (np.uint64(1)<<moves.astype(np.uint64))) == 0):
            raise ValueError('Illegal NN baseline move')
        name = method+'_conv8_bn'
        results[name],hits[name] = metrics(records,moves)
        if results[name]['top1_hits'] != proof['metrics']['top1_hits']:
            raise ValueError('NN baseline agreement does not reproduce')
        baseline_sources[name] = dict(summary=str(path/'summary.json'), prediction_sha256=proof['predictions_sha256'])
    comparisons = {name: bootstrap_difference(game_ids, hits['trained_rules'], h, args.seed)
                   for name,h in hits.items() if name != 'trained_rules'}
    np.save(output/'test_records.npy', records)
    np.save(output/'test_game_ids.npy', game_ids)
    np.save(output/'test_legal.npy', legal)
    saved_moves = np.load(output/'trained_rules_predictions.npy', allow_pickle=False)
    if np.any((legal & (np.uint64(1)<<saved_moves.astype(np.uint64))) == 0):
        raise ValueError('Illegal saved trained prediction')
    for i in np.linspace(0,len(records)-1,128,dtype=int):
        p = selected.predict(int(records['player'][i]),int(records['opponent'][i]))
        if int(p.argmax()) != int(saved_moves[i]):
            raise ValueError('Standalone policy differs from cached evaluation')
    summary = dict(status='complete', identity=identity, selected_validation=best[2], test=results,
                   paired_game_bootstrap=comparisons, baseline_sources=baseline_sources,
                   model_sha256=file_sha256(output/'selected_model.json'),
                   predictions_sha256=file_sha256(output/'trained_rules_predictions.npy'),
                   records_sha256=file_sha256(output/'test_records.npy'),
                   elapsed_sec=time.monotonic()-started)
    save(output/'agreement.json',summary)
    with (output/'weights.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream)
        writer.writerow(('feature',)+PHASE_NAMES)
        writer.writerows((name, *w) for name,w in zip(FEATURE_NAMES,weights))
    print('COMPLETE agreement: '+str(output/'agreement.json'),flush=True)
    for name,row in results.items():
        print(f'{name}: {100*row["top1_accuracy"]:.4f}%',flush=True)
    lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=default_data())
    parser.add_argument('--output',type=Path,default=HERE/'output/quest_800-1000')
    parser.add_argument('--baselines',type=Path,default=WORKSPACE/'experiment_bat/20261010_quest_rating_bands/evaluation')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--iterations',type=int,default=120)
    parser.add_argument('--l2',nargs='+',type=float,default=[1e-4,1e-5])
    parser.add_argument('--seed',type=int,default=613)
    args=parser.parse_args()
    if args.iterations<1 or any(p<=0 for p in args.l2):
        parser.error('iterations and L2 must be positive')
    run(args)


if __name__=='__main__':
    main()
