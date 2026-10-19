"""Label training positions with search values of an Egaroucid build.

Samples records from a converted record file (convert_data.cpp format), solves each position with
the console's -solve at a fixed depth (-dpr 1 60 <depth> <probability>, one thread per process,
several processes), and writes the same records with the score replaced by the search value
(side to move, clamped to [-64, 64]), plus the original score in a side file.

Usage:
    python src/tools/nnue/label_positions.py --input ignored/nnue_data/merged/train_tr.bin \
        --engine bin/Egaroucid_nnue_rw64f_lazy6_v4.exe --n 20000000 --depth 8 --procs 24 \
        --out ignored/nnue_data/labels/d8_nnue
    -> <out>.bin (records with search values), <out>.orig.i8 (original scores), <out>.idx.u64 (source rows),
       <out>.p.f32 (pattern values of the positions, from <input>.p.f32 when it exists)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REC = np.dtype([("p", "<u8"), ("o", "<u8"), ("s", "i1"), ("e", "u1"), ("src", "u1"), ("r", "u1")])
BIT = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64))  # char k <-> bit 63 - k


def board_lines(recs):
    p = recs["p"][:, None] & BIT
    o = recs["o"][:, None] & BIT
    grid = np.full(p.shape, ord("-"), dtype=np.uint8)
    grid[p != 0] = ord("X")
    grid[o != 0] = ord("O")
    return [row.tobytes().decode() + " X\n" for row in grid]


def parse_values(text):
    vals = []
    for line in text.splitlines():
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) == 7 and cols[0] != "Level":
            vals.append(int(cols[3]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--prob", type=int, default=100)
    ap.add_argument("--min-empties", type=int, default=10)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=50000, help="positions per engine run")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--collect", action="store_true", help="only gather the chunks finished so far (same arguments as the run)")
    args = ap.parse_args()
    a = np.memmap(args.input, dtype=REC, mode="r")
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero(np.asarray(a["e"]) >= args.min_empties)
    rows = np.sort(rng.choice(cand, size=min(args.n, len(cand)), replace=False))
    recs = np.array(a[rows])
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / (out.name + "_work")
    work.mkdir(exist_ok=True)
    exe = Path(args.engine).resolve()
    values = np.zeros(len(recs), dtype=np.int16)
    chunks = [(i, min(i + args.chunk, len(recs))) for i in range(0, len(recs), args.chunk)]
    if args.collect:
        keep = np.zeros(len(recs), dtype=bool)
        for c, (lo, hi) in enumerate(chunks):
            res = work / f"chunk_{c:05d}.out"
            if res.exists():
                vals = parse_values(res.read_text(encoding="utf-8"))
                if len(vals) == hi - lo:
                    values[lo:hi] = vals
                    keep[lo:hi] = True
        recs, rows, values = recs[keep], rows[keep], values[keep]
        print(f"collected {len(recs)} finished positions")
        chunks, pending = [], []
    pending = [] if args.collect else list(range(len(chunks)))
    running = {}
    done = 0
    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.procs:
            c = pending.pop(0)
            lo, hi = chunks[c]
            prob = work / f"chunk_{c:05d}.txt"
            res = work / f"chunk_{c:05d}.out"
            if not (res.exists() and len(parse_values(res.read_text(encoding="utf-8"))) == hi - lo):
                prob.write_text("".join(board_lines(recs[lo:hi])), encoding="utf-8")
                f = open(res, "w", encoding="utf-8")
                proc = subprocess.Popen([str(exe), "-nobook", "-quiet", "-thread", "1", "-hash", "20",
                                         "-dpr", "1", "60", str(args.depth), str(args.prob), "-solve", str(prob)],
                                        cwd=exe.parent, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.DEVNULL,
                                        creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))  # leave room for training
                running[c] = (proc, f)
            else:
                running[c] = (None, None)
        for c in list(running):
            proc, f = running[c]
            if proc is not None and proc.poll() is None:
                continue
            if f is not None:
                f.close()
            lo, hi = chunks[c]
            vals = parse_values((work / f"chunk_{c:05d}.out").read_text(encoding="utf-8"))
            if len(vals) != hi - lo:
                print(f"chunk {c}: {len(vals)} values for {hi - lo} positions, retrying", flush=True)
                (work / f"chunk_{c:05d}.out").unlink()
                pending.append(c)
            else:
                values[lo:hi] = vals
                done += hi - lo
                (work / f"chunk_{c:05d}.txt").unlink(missing_ok=True)
            del running[c]
        el = time.time() - t0
        sys.stdout.write(f"\r{done}/{len(recs)} positions, {el:.0f} s, {done / max(el, 1):.0f}/s   ")
        sys.stdout.flush()
        time.sleep(0.5)
    print()
    if args.collect:
        out = out.with_name(out.name + "_partial")  # keep the full run's outputs apart
    labeled = recs.copy()
    labeled["s"] = np.clip(values, -64, 64).astype(np.int8)
    labeled.tofile(out.with_suffix(".bin"))
    recs["s"].astype(np.int8).tofile(out.with_suffix(".orig.i8"))
    rows.astype(np.uint64).tofile(out.with_suffix(".idx.u64"))
    pfile = Path(args.input).with_suffix(".p.f32")
    if pfile.exists():
        np.asarray(np.memmap(pfile, dtype=np.float32, mode="r")[rows]).tofile(out.with_suffix(".p.f32"))
    diff = values.astype(np.float64) - recs["s"]
    print(f"wrote {len(recs)} records; mean (search - result) {diff.mean():.3f}, mean squared {np.mean(diff ** 2):.2f}")


if __name__ == "__main__":
    main()
