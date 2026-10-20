"""Choose the z of a candidate MPC error model so that it makes as many wrong cuts as the engine's current model.

Both runs must hold predictions for the same rows (train_mpc_error.py on the same data; the reference with
--apply-head <the engine's head.json> so that its coefficients are the engine's). For each group and MPC
level that the engine uses, the reference margins (high max(1, ceil(z sigma - mu)), low max(1, ceil(z sigma + mu))
with the engine's z) give a wrong-cut rate P(e < -high) + P(e > low) on the validation boards; the candidate's z
is solved to give the same rate. Endgame static cuts exist only on the recalibrated path (10-18 empties at
<= 93%, Z_END_STATIC_CUT). Prints both tails and the mean margins; writes <run>/z_match.json.

Usage: python src/tools/nnue/calibrate_mpc_match.py --ref ignored/nnue_runs/mpcerr_net3_on_new --ref-z ignored/nnue_runs/mpcerr_b1/z_final.json \
           --run ignored/nnue_runs/mpcerr_dual_new
"""
import argparse
import json
from pathlib import Path

import numpy as np

LEVELS = ["74", "88", "93", "98", "99", "99.9"]
GROUPS = ["mid_static", "mid_shallow", "end_static", "end_shallow"]


def margins(z, sig, mu):
    return np.maximum(1, np.ceil(z * sig - mu)), np.maximum(1, np.ceil(z * sig + mu))


def tails(e, mh, ml):
    return float(np.mean(e < -mh)), float(np.mean(e > ml))


def solve_z(e, sig, mu, rate):
    lo, hi = 0.05, 12.0
    for _ in range(60):
        z = 0.5 * (lo + hi)
        if sum(tails(e, *margins(z, sig, mu))) > rate:
            lo = z
        else:
            hi = z
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-z", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--all-rows", action="store_true")
    args = ap.parse_args()
    r = np.load(Path(args.ref) / "predictions.npz")
    c = np.load(Path(args.run) / "predictions.npz")
    for k in ("e", "g", "n", "D", "s"):
        assert np.array_equal(r[k], c[k]), f"the runs have different rows ({k})"
    zr = json.loads(Path(args.ref_z).read_text(encoding="utf-8"))
    sel = np.ones(len(r["e"]), bool) if args.all_rows else r["val"].astype(bool)
    e, g, D = r["e"][sel], r["g"][sel], r["D"][sel].astype(int)
    sr, mr = np.exp(r["ls"][sel]), r["mu"][sel]
    sc, mc = np.exp(c["ls"][sel]), c["mu"][sel]
    Z = [[9.99] * 7 for _ in range(4)]
    Z_SC = [9.99] * 7
    for gi, name in enumerate(GROUPS):
        for li, lv in enumerate(LEVELS):
            m = g == gi
            if gi == 2:
                zref = zr["Z_END_STATIC_CUT"][li]
                m &= (D >= 10) & (D <= 18)
            else:
                zref = zr["Z"][gi][li]
            if zref >= 9.0 or m.sum() == 0:
                continue
            hr, lr = margins(zref, sr[m], mr[m])
            th, tl = tails(e[m], hr, lr)
            rate = th + tl if th + tl > 0 else 0.5 / m.sum()
            z = solve_z(e[m], sc[m], mc[m], rate)
            hc, lc = margins(z, sc[m], mc[m])
            ch, cl = tails(e[m], hc, lc)
            if gi == 2:
                Z_SC[li] = z
            else:
                Z[gi][li] = z
            print(f"{name:11s} {lv:>4s}%: rows {m.sum():6d} | reference z {zref:.3f} wrong high {th:.4f} low {tl:.4f} margin {(hr.mean() + lr.mean()) / 2:5.2f}"
                  f" | candidate z {z:.3f} wrong high {ch:.4f} low {cl:.4f} margin {(hc.mean() + lc.mean()) / 2:5.2f}")
    for zz in Z + [Z_SC]:
        for li in range(1, 6):
            zz[li] = max(zz[li], zz[li - 1])
    (Path(args.run) / "z_match.json").write_text(json.dumps({"Z": Z, "Z_END_STATIC_CUT": Z_SC, "ref": str(args.ref), "ref_z": str(args.ref_z)}, indent=1), encoding="utf-8")
    print("wrote", Path(args.run) / "z_match.json")


if __name__ == "__main__":
    main()
