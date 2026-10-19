"""Shuffle a training file (records of convert_data.cpp and their P values) on disk.

Transcript files hold the positions of a game next to each other, and all of them share the game's
result as the label. A streaming trainer that reads contiguous blocks then sees few games at a time.
This writes every record to a random bucket file, then shuffles each bucket in memory and appends it to
the output, so contiguous blocks of the output are uniform samples.

Usage: python src/tools/nnue/shuffle_records.py <in dir> <out dir> [--name train_tr] [--buckets 32]
Reads <in dir>/<name>.bin and <name>.p.f32; writes <out dir>/<name>.bin and <name>.p.f32.
"""
import argparse
from pathlib import Path

import numpy as np

REC = np.dtype([("p", "<u8"), ("o", "<u8"), ("s", "i1"), ("e", "u1"), ("src", "u1"), ("r", "u1")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--name", default="train_tr")
    ap.add_argument("--buckets", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=50_000_000)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    tmp = dst / "shuffle_tmp"
    tmp.mkdir(exist_ok=True)
    a = np.memmap(src / f"{args.name}.bin", dtype=REC, mode="r")
    p = np.memmap(src / f"{args.name}.p.f32", dtype=np.float32, mode="r")
    assert len(a) == len(p)
    rng = np.random.default_rng(args.seed)
    rec_files = [open(tmp / f"b{k:03d}.bin", "wb") for k in range(args.buckets)]
    p_files = [open(tmp / f"b{k:03d}.p.f32", "wb") for k in range(args.buckets)]
    for start in range(0, len(a), args.chunk):
        ra = np.asarray(a[start:start + args.chunk])
        rp = np.asarray(p[start:start + args.chunk])
        b = rng.integers(0, args.buckets, len(ra))
        order = np.argsort(b, kind="stable")
        cnt = np.bincount(b, minlength=args.buckets)
        pos = 0
        for k in range(args.buckets):
            sel = order[pos:pos + cnt[k]]
            rec_files[k].write(ra[sel].tobytes())
            p_files[k].write(rp[sel].tobytes())
            pos += cnt[k]
        print(f"split {min(start + args.chunk, len(a))}/{len(a)}", flush=True)
    for f in rec_files + p_files:
        f.close()
    with open(dst / f"{args.name}.bin", "wb") as wa, open(dst / f"{args.name}.p.f32", "wb") as wp:
        for k in range(args.buckets):
            ra = np.fromfile(tmp / f"b{k:03d}.bin", dtype=REC)
            rp = np.fromfile(tmp / f"b{k:03d}.p.f32", dtype=np.float32)
            perm = rng.permutation(len(ra))
            wa.write(ra[perm].tobytes())
            wp.write(rp[perm].tobytes())
            (tmp / f"b{k:03d}.bin").unlink()
            (tmp / f"b{k:03d}.p.f32").unlink()
            print(f"bucket {k + 1}/{args.buckets}", flush=True)
    tmp.rmdir()


if __name__ == "__main__":
    main()
