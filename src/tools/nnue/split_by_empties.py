"""Split a shuffled training file into one file per empties count (for train_nnue.py --stream-by-empties).

Usage: python src/tools/nnue/split_by_empties.py <dir with shuffled train_tr.bin/.p.f32> <out dir>
Writes <out dir>/e<NN>.bin and e<NN>.p.f32 (records keep the shuffled order).
"""
import sys
from pathlib import Path

import numpy as np

REC = np.dtype([("p", "<u8"), ("o", "<u8"), ("s", "i1"), ("e", "u1"), ("src", "u1"), ("r", "u1")])


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    # files of an earlier split would be read as training data too
    for f in list(dst.glob("e*.bin")) + list(dst.glob("e*.p.f32")):
        f.unlink()
    a = np.memmap(src / "train_tr.bin", dtype=REC, mode="r")
    p = np.memmap(src / "train_tr.p.f32", dtype=np.float32, mode="r")
    files = {}
    chunk = 50_000_000
    for start in range(0, len(a), chunk):
        ra = np.asarray(a[start:start + chunk])
        rp = np.asarray(p[start:start + chunk])
        order = np.argsort(ra["e"], kind="stable")
        e_sorted = ra["e"][order]
        for e in np.unique(e_sorted):
            sel = order[np.searchsorted(e_sorted, e, "left"):np.searchsorted(e_sorted, e, "right")]
            if e not in files:
                files[e] = (open(dst / f"e{e:02d}.bin", "wb"), open(dst / f"e{e:02d}.p.f32", "wb"))
            files[e][0].write(ra[sel].tobytes())
            files[e][1].write(rp[sel].tobytes())
        print(f"{min(start + chunk, len(a))}/{len(a)}", flush=True)
    for fa, fp in files.values():
        fa.close()
        fp.close()


if __name__ == "__main__":
    main()
