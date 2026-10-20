"""Calibrate the z of the MPC error model (train_mpc_error.py) to the current NNUE build's wrong-cut rates.

For each group and MPC level, z is chosen on the validation boards so that the model margins
(high ceil(z sigma - mu), low ceil(z sigma + mu), at least 1) give the same tail rate
P(e < -m_high) + P(e > m_low) as the production margins of the current NNUE build:
  midgame shallow / static: ceil(z_mid * k * sigma_dir(discs, s, D)), k = MID_MPC_SIGMA_EXTRA_SCALE (0.9)
  endgame shallow, <= 93% and 10-18 empties: END_MPC_SHALLOW_ERROR tables; else ceil(z_end * sigma_end_dir)
  endgame static cut (<= 93%, 10-18 empties): END_MPC_STATIC_HIGH/LOW_OFFSET
  endgame margins are multiplied by u03 (clamp(exp(0.3 (u - mean u)), 0.75, 1.33)) as the build does.
The constants mirror src/engine/multi_probcut.hpp and level.hpp. Writes <run>/z.json and prints the mean
margins of the model and of the production rule at the same tail rate.

Usage: python src/tools/nnue/calibrate_mpc_net.py ignored/nnue_runs/mpcerr_b1 [--k 0.9] [--no-u03]
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
Z_MID = [1.13, 1.55, 1.81, 2.088, 2.1845, 2.7965]
Z_END = [1.13, 1.55, 1.81, 2.32, 2.57, 3.29]
MID_HIGH = (0.7935834703936896, -4.86778909968251, 1.0967905781538476, -0.5574699259330169, 6.5091001393587335, 3.9546352081550378, 1.7719077939546168)
MID_LOW = (0.8335834703936896, -4.71778909968251, 1.1467905781538477, -0.5274699259330169, 6.5091001393587335, 3.9546352081550378, 1.871907793954617)
END_HIGH = (-1.4382333120273683, -7.29290557735024, -0.058806541462447556, 0.48284187178125065, 5.289589936037036, 11.940601436361513)
END_LOW = (-1.3982333120273682, -7.29290557735024, -0.06480654146244756, 0.3828418717812507, 5.289589936037036, 11.940601436361513)
REC_ERR_HIGH = [[6, 6, 7, 5, 6, 5, 6, 5, 5], [8, 8, 9, 7, 9, 7, 8, 6, 7], [10, 9, 11, 9, 11, 8, 9, 8, 8]]
REC_ERR_LOW = [[6, 6, 7, 6, 7, 5, 6, 5, 5], [9, 8, 10, 8, 9, 7, 8, 7, 7], [10, 9, 11, 9, 11, 8, 10, 8, 9]]
STATIC_HIGH = [18, 19, 20, 21, 21, 19, 20, 19, 19]
STATIC_LOW = [-27, -28, -29, -30, -30, -29, -29, -29, -28]
STATIC_BIAS = 3.2015961138098543
LEVELS = ["74", "88", "93", "98", "99", "99.9"]


def sigma_mid(coef, n, d1, d2):
    a, b, c, d, e, f, g = coef
    r = a * n / 64.0 + b * d1 / 60.0 + c * d2 / 60.0
    return d * r ** 3 + e * r ** 2 + f * r + g


def sigma_end(coef, n, d):
    a, b, c, dd, e, f = coef
    r = a * n / 64.0 + b * d / 60.0
    return c * r ** 3 + dd * r ** 2 + e * r + f


def mean_u_table():
    txt = (ROOT / "src" / "engine" / "nnue" / "nnue_uncertainty_table.hpp").read_text(encoding="utf-8")
    return [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", txt[txt.index("{"):txt.index("}")])]


def tail(e, mh, ml):
    return float(np.mean(e < -mh) + np.mean(e > ml))


def model_margins(z, sig, mu):
    return np.maximum(1, np.ceil(z * sig - mu)), np.maximum(1, np.ceil(z * sig + mu))


def solve_z(e, sig, mu, rate):
    lo, hi = 0.05, 12.0
    for _ in range(60):
        z = 0.5 * (lo + hi)
        if tail(e, *model_margins(z, sig, mu)) > rate:
            lo = z
        else:
            hi = z
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=float, default=0.9)
    ap.add_argument("--no-u03", action="store_true")
    ap.add_argument("--all-rows", action="store_true", help="calibrate on all boards instead of the validation boards")
    args = ap.parse_args()
    d = np.load(Path(args.run) / "predictions.npz")
    sel = np.ones(len(d["e"]), bool) if args.all_rows else d["val"].astype(bool)
    e, g, n, D, s, u = d["e"][sel], d["g"][sel], d["n"][sel].astype(int), d["D"][sel].astype(int), d["s"][sel].astype(int), d["u"][sel]
    sig, mu = np.exp(d["ls"][sel]), d["mu"][sel]
    mean_u = mean_u_table()
    u03 = np.ones_like(u) if args.no_u03 else np.clip(np.exp(0.3 * (u - np.array([mean_u[x] for x in n]))), 0.75, 1.33)
    Z = [[9.99] * 7 for _ in range(4)]
    Z_SC = [9.99] * 7
    report = []
    for gi, name in enumerate(["mid_static", "mid_shallow", "end_static", "end_shallow"]):
        m = g == gi
        if m.sum() == 0:
            continue
        for li, lv in enumerate(LEVELS):
            mm = m.copy()
            if gi < 2:
                mh = np.ceil(Z_MID[li] * args.k * np.array([sigma_mid(MID_HIGH, a, b, c) for a, b, c in zip(n[mm], s[mm], D[mm])]))
                ml = np.ceil(Z_MID[li] * args.k * np.array([sigma_mid(MID_LOW, a, b, c) for a, b, c in zip(n[mm], s[mm], D[mm])]))
            elif gi == 3:
                rec = (li <= 2) & (D[mm] >= 10) & (D[mm] <= 18)
                mh = np.where(rec, [REC_ERR_HIGH[min(li, 2)][min(max(x - 10, 0), 8)] for x in D[mm]],
                              np.ceil(Z_END[li] * np.array([sigma_end(END_HIGH, a, b) for a, b in zip(n[mm], s[mm])])))
                ml = np.where(rec, [REC_ERR_LOW[min(li, 2)][min(max(x - 10, 0), 8)] for x in D[mm]],
                              np.ceil(Z_END[li] * np.array([sigma_end(END_LOW, a, b) for a, b in zip(n[mm], s[mm])])))
                mh = np.maximum(1, np.ceil(mh * u03[mm]))
                ml = np.maximum(1, np.ceil(ml * u03[mm]))
            else:
                # the static cut exists only on the recalibrated path (<= 93%, 10-18 empties)
                if li > 2:
                    continue
                mm &= (D >= 10) & (D <= 18)
                idx = np.clip(D[mm] - 10, 0, 8)
                oh = np.array([STATIC_HIGH[i] for i in idx], float)
                ol = np.array([STATIC_LOW[i] for i in idx], float)
                mh = np.ceil(-STATIC_BIAS + (oh + STATIC_BIAS) * u03[mm])
                ml = -np.floor(-STATIC_BIAS + (ol + STATIC_BIAS) * u03[mm])
            rate = tail(e[mm], mh, ml)
            if rate <= 0:
                rate = 0.5 / mm.sum()  # nothing in the tail: aim at half a sample
            z = solve_z(e[mm], sig[mm], mu[mm], rate)
            if gi == 2:
                Z_SC[li] = z
            else:
                Z[gi][li] = z
            nh, nl = model_margins(z, sig[mm], mu[mm])
            report.append(f"{name:11s} {lv:>4s}%: production tail {rate:.4f}, z {z:.3f}, mean margin production {(mh.mean() + ml.mean()) / 2:5.2f} model {(nh.mean() + nl.mean()) / 2:5.2f}")
    print("\n".join(report))
    # level 74-93 static cuts share the production offsets; keep the z non-decreasing with the level
    for zz in Z + [Z_SC]:
        for li in range(1, 6):
            zz[li] = max(zz[li], zz[li - 1])
    (Path(args.run) / "z.json").write_text(json.dumps({"Z": Z, "Z_END_STATIC_CUT": Z_SC, "k": args.k, "u03": not args.no_u03}, indent=1), encoding="utf-8")
    print("wrote", Path(args.run) / "z.json")


if __name__ == "__main__":
    main()
