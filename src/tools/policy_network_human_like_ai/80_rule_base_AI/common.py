"""Verified data access and durable, reproducible experiment outputs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[2]
WORKSPACE = REPO.parent
for folder in (BASE, BASE / '00_data_processing', BASE / '70_avg_disc_loss'):
    sys.path.insert(0, str(folder))
from wthor_dataset import BOARD_DTYPE, file_sha256
from policy_data import calc_legal_batch
from policy_accuracy import equivalent_policy_mask


def default_data():
    if os.environ.get('POLICY_DATA_ROOT'):
        root = Path(os.environ['POLICY_DATA_ROOT'])
    elif os.environ.get('EGAROUCID_DATA'):
        root = Path(os.environ['EGAROUCID_DATA']) / '20261002_policy_network_human_like_data'
    else:
        raise ValueError('Set POLICY_DATA_ROOT, EGAROUCID_DATA, or --data')
    return root / 'othello_quest_split/board_split/800-1000'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(100):
        try:
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(.05)
    raise RuntimeError('Atomic result write blocked: ' + str(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_split(directory, split):
    directory = Path(directory)
    manifest = read(directory / 'manifest.json')
    if (manifest['dataset'] != '800-1000' or manifest['stage'] != 'board_split'
        or manifest['rating_range'] != dict(min_inclusive=800, max_exclusive=1000)
        or manifest['split_unit'] != 'identical_transcript_group_across_all_rating_bands'
        or manifest.get('sample_weights')):
        raise ValueError('Expected unaugmented, unweighted, game-disjoint Quest [800,1000)')
    records, ids = [], []
    for entry in manifest['files']:
        if not entry['path'].startswith(split + '/'):
            continue
        path = directory / entry['path']
        if path.stat().st_size != entry['size'] or file_sha256(path) != entry['sha256']:
            raise ValueError('Input hash differs: ' + str(path))
    for path in sorted((directory / split).glob('*.dat')):
        records.append(np.fromfile(path, dtype=BOARD_DTYPE))
        ids.append(np.fromfile(path.with_suffix('.game_ids'), dtype='<u8'))
    records, ids = np.concatenate(records), np.concatenate(ids)
    if len(records) != len(ids) or len(records) != manifest['splits'][split]['positions']:
        raise ValueError('Record count differs')
    legal = calc_legal_batch(records['player'], records['opponent'])
    labels = records['policy'].astype(np.int64)
    if np.any(records['player'] & records['opponent']) or np.any((labels < 0) | (labels >= 64)):
        raise ValueError('Invalid board or label')
    if np.any((legal & (np.uint64(1) << labels.astype(np.uint64))) == 0):
        raise ValueError('Illegal human label')
    return records, ids, legal


def metrics(records, moves, probabilities=None):
    targets = equivalent_policy_mask(records['player'], records['opponent'], records['policy'].astype(int))
    hits = targets[np.arange(len(records)), moves]
    exact = moves == records['policy']
    result = dict(positions=len(records), top1_hits=int(hits.sum()), top1_accuracy=float(hits.mean()),
                  exact_top1_hits=int(exact.sum()), exact_top1_accuracy=float(exact.mean()))
    if probabilities is not None:
        result['human_move_nll'] = float(-np.log(np.maximum(probabilities[np.arange(len(records)), records['policy']], 1e-300)).mean())
    return result, hits


def bootstrap_difference(ids, hits_a, hits_b, seed=613):
    """Paired game-cluster bootstrap, retaining the position-weighted denominator."""
    _, inverse = np.unique(ids, return_inverse=True)
    sizes = np.bincount(inverse)
    delta = np.bincount(inverse, weights=hits_a.astype(float) - hits_b.astype(float))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sizes), size=(2000, len(sizes)))
    values = delta[draws].sum(1) / sizes[draws].sum(1)
    return dict(difference=float(np.mean(hits_a) - np.mean(hits_b)),
                game_bootstrap_95_ci=np.quantile(values, [.025, .975]).tolist(), replicates=2000, seed=seed)
