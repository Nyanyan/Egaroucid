"""Snapshot evaluation files and prepare deterministic hash data for a repair."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys


def prepare(source, destination, level, seed):
    if not 25 <= level <= 29:
        raise ValueError('Hash level must be 25..29')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'hash').mkdir(exist_ok=True)
    manifest = dict(hash_level=level, seed=seed, files={})
    for name in ['eval.egev2', 'eval_move_ordering_end.egev']:
        target = destination / name
        if target.exists() and target.read_bytes() != (source / name).read_bytes():
            raise RuntimeError('Existing evaluation snapshot differs: ' + str(target))
        shutil.copyfile(source / name, target)
        manifest['files'][name] = hashlib.sha256(target.read_bytes()).hexdigest()
    name = f'hash{level}.eghs'
    if (source / 'hash' / name).exists():
        data = (source / 'hash' / name).read_bytes()
        manifest['generation'] = 'copied existing hash data'
    else:
        # Same contract as engine hash_init_rand: independent masked words with
        # at least level//6 set bits. This is not an offline hash optimization.
        rng = random.Random(seed)
        words = array('I')
        for _ in range(8 * 65536):
            word = rng.getrandbits(level)
            while bin(word).count('1') < level // 6:
                word = rng.getrandbits(level)
            words.append(word)
        if sys.byteorder != 'little':
            words.byteswap()
        data = words.tobytes()
        manifest['generation'] = 'deterministic random fallback, engine hash_init_rand contract'
    if len(data) != 2097152:
        raise RuntimeError('Incorrect hash file length')
    words = array('I'); words.frombytes(data)
    if sys.byteorder != 'little':
        words.byteswap()
    if max(words) >= (1 << level) or not any(word & (1 << (level - 1)) for word in words):
        raise RuntimeError('Hash data does not address the requested table size')
    target = destination / 'hash' / name
    if target.exists() and target.read_bytes() != data:
        raise RuntimeError('Existing hash snapshot differs')
    target.write_bytes(data)
    manifest['files']['hash/' + name] = hashlib.sha256(data).hexdigest()
    (destination / 'resources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hash-level', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    print(json.dumps(prepare(args.resources, args.output, args.hash_level, args.seed), indent=2))
