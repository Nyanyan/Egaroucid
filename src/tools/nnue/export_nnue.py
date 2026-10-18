"""Quantize a trained pattern NNUE, write the engine weight file, and check the integer model.

Integer model (the engine computes exactly this, src/engine/nnue/nnue_eval.hpp):
  acc   = sum of 64 int16 rows + bias          (1.0 = 512)
  a     = clamp(acc, 0, 512)
  x     = min((a[k] * a[k + W/2]) >> 11, 127)  (1.0 = 128, uint8)
  zacc  = int8 l1 weights (1.0 = 64) . x + int32 bias      (1.0 = 8192)
  z     = clamp(zacc >> 6, 0, 127)              (1.0 = 128)
  oacc  = int16 out weights (1.0 = 1024) . z + int32 bias  (1 disc = 8192)
  lin   = sum of 64 int16 linear rows + int16 disc-count term (1 disc = 256)
  total = lin * 32 + oacc   (direct)            total = P_raw * 256 + oacc (residual, P_raw: 1 disc = 32)
  score = round half away from zero (total / 8192), clamped to [-64, 64]

File (little endian): "EGNNUE01", int32 [version=1, mode, rows, instances, W, H, buckets, 0],
  int16 ft[rows][W], int16 ft_bias[W], int8 l1_w[buckets][H][W/2], int32 l1_b[buckets][H],
  int16 out_w[buckets][H], int32 out_b[buckets], int16 lin[buckets][rows], int16 num[buckets][65]
  (mode: 0 direct, 1 residual, 2 linear; absent parts are written as zeros)

Usage: python src/tools/nnue/export_nnue.py --run <run dir> [--model best.pt] --data <merged dir> --out <file.egnn>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import nnue_patterns as pat
from train_nnue import N_BUCKETS, Features, Net, load_split, to_tensors, bucket_of

MODES = {"direct": 0, "residual": 1, "linear": 2}


def quantize(model, mode):
    rows = pat.N_ROWS
    W, H = (model.width, model.hidden) if mode != "linear" else (0, 0)
    q = {}
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    if mode != "linear":
        ft = np.round(sd["ft.weight"] * 512)
        q["ft"] = np.clip(ft, -32768, 32767).astype(np.int16)
        q["ft_bias"] = np.clip(np.round(sd["ft_bias"] * 512), -32768, 32767).astype(np.int16)
        l1w = sd["l1.weight"].reshape(N_BUCKETS, H, W // 2)
        q["l1_w"] = np.clip(np.round(l1w * 64), -127, 127).astype(np.int8)
        q["l1_b"] = np.round(sd["l1.bias"].reshape(N_BUCKETS, H) * 8192).astype(np.int32)
        q["out_w"] = np.clip(np.round(sd["out.weight"] * 1024), -32768, 32767).astype(np.int16)  # (buckets, H)
        q["out_b"] = np.round(sd["out.bias"] * 131072).astype(np.int32)
    else:
        q["ft"] = np.zeros((rows, 0), np.int16)
        q["ft_bias"] = np.zeros(0, np.int16)
        q["l1_w"] = np.zeros((N_BUCKETS, 0, 0), np.int8)
        q["l1_b"] = np.zeros((N_BUCKETS, 0), np.int32)
        q["out_w"] = np.zeros((N_BUCKETS, 0), np.int16)
        q["out_b"] = np.zeros(N_BUCKETS, np.int32)
    if mode != "residual":
        q["lin"] = np.clip(np.round(sd["lin.weight"].reshape(N_BUCKETS, rows) * 256), -32768, 32767).astype(np.int16)
        q["num"] = np.clip(np.round(sd["num.weight"].reshape(N_BUCKETS, 65) * 256), -32768, 32767).astype(np.int16)
    else:
        q["lin"] = np.zeros((N_BUCKETS, rows), np.int16)
        q["num"] = np.zeros((N_BUCKETS, 65), np.int16)
    q["W"], q["H"] = W, H
    return q


def write_file(q, mode, path):
    with open(path, "wb") as f:
        f.write(b"EGNNUE01")
        f.write(np.array([1, MODES[mode], pat.N_ROWS, len(pat.INSTANCES), q["W"], q["H"], N_BUCKETS, 0], np.int32).tobytes())
        for k in ("ft", "ft_bias", "l1_w", "l1_b", "out_w", "out_b", "lin", "num"):
            f.write(np.ascontiguousarray(q[k]).tobytes())


def q_forward(q, mode, idx, bucket, n_player, p_raw=None):
    """idx (N,64) int64, bucket (N,), n_player (N,); p_raw: production eval in 1/32 discs (residual)."""
    n = idx.shape[0]
    W = q["W"]
    total = np.zeros(n, np.int64)
    overflow = 0
    if mode != "linear":
        acc = q["ft"][idx].astype(np.int64).sum(1) + q["ft_bias"].astype(np.int64)
        overflow = int((np.abs(acc) > 32767).any(1).sum())
        a = np.clip(acc, 0, 512)
        x = np.minimum((a[:, : W // 2] * a[:, W // 2:]) >> 11, 127)
        zacc = np.einsum("nk,nhk->nh", x, q["l1_w"][bucket].astype(np.int64)) + q["l1_b"][bucket]
        z = np.clip(zacc >> 6, 0, 127)
        oacc = (z * q["out_w"][bucket].astype(np.int64)).sum(1) + q["out_b"][bucket]
        total += oacc
    if mode == "residual":
        total += p_raw.astype(np.int64) * 256
    else:
        lin = q["lin"][bucket[:, None], idx].astype(np.int64).sum(1) + q["num"][bucket, n_player]
        total += lin * 32
    score = np.where(total >= 0, (total + 4096) // 8192, -((-total + 4096) // 8192))
    return np.clip(score, -64, 64), total, overflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-check", type=int, default=400000)
    ap.add_argument("--n-vectors", type=int, default=2000)
    args = ap.parse_args()
    run = Path(args.run)
    targs = json.loads((run / "args.json").read_text(encoding="utf-8"))
    mode = targs["mode"]
    model = Net(targs["width"], targs["hidden"], mode)
    model.load_state_dict(torch.load(run / args.model, map_location="cpu"))
    model.eval()
    if targs.get("fold"):
        # copy the canonical rows back to every row they stand for
        fold = torch.tensor(pat.fold_rows(), dtype=torch.long)
        with torch.no_grad():
            if mode != "linear":
                model.ft.weight.copy_(model.ft.weight[fold])
            if mode != "residual":
                lw = model.lin.weight.view(N_BUCKETS, pat.N_ROWS)
                model.lin.weight.copy_(lw[:, fold].reshape(-1, 1))
    q = quantize(model, mode)
    write_file(q, mode, args.out)

    # check: float model vs integer model on held-out transcripts
    va_a, va_p = load_split(args.data, "val_tr")
    rng = np.random.default_rng(1)
    sel = np.sort(rng.permutation(len(va_a))[: args.n_check])
    val = to_tensors(va_a, va_p, sel)
    feat = Features(torch.device("cpu"))
    st = feat.states(val["p"], val["o"])
    idx = feat.indices(st)
    bucket = bucket_of(val["e"].long())
    n_player = (st == 1).sum(1)
    P = val["P"]
    with torch.no_grad():
        out_f = []
        for i in range(0, len(sel), 65536):
            o, _ = model(idx[i:i + 65536], bucket[i:i + 65536], n_player[i:i + 65536], P[i:i + 65536])
            out_f.append(o)
        out_f = torch.cat(out_f).numpy()
    p_raw = np.round(P.numpy() * 32).astype(np.int64)
    score_q, total_q, overflow = q_forward(q, mode, idx.numpy(), bucket.numpy(), n_player.numpy(), p_raw)
    y = val["s"].numpy().astype(np.float64)
    e = val["e"].numpy()
    rep = {"mode": mode, "overflow_positions": overflow, "n": int(len(sel))}
    for name, v in (("float", out_f), ("int_raw", total_q / 8192.0), ("int_rounded", score_q), ("P_rounded", np.clip(np.where(P.numpy() >= 0, np.floor(P.numpy() + 0.5), -np.floor(-P.numpy() + 0.5)), -64, 64))):
        err = (v - y) ** 2
        rep[name] = {"all": float(err.mean()), "30-59": float(err[e >= 30].mean())}
    rep["float_vs_int_mean_abs"] = float(np.abs(out_f - total_q / 8192.0).mean())
    print(json.dumps(rep, indent=1))
    Path(args.out).with_suffix(".check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")

    # test vectors for the engine: boards, expected indices, expected integer total
    nv = min(args.n_vectors, len(sel))
    with open(Path(args.out).with_suffix(".vectors.txt"), "w", encoding="utf-8", newline="\n") as f:
        for i in range(nv):
            pv = int(val["p"][i]) & ((1 << 64) - 1)
            ov = int(val["o"][i]) & ((1 << 64) - 1)
            f.write(f"{pv:016x} {ov:016x} {int(p_raw[i])} {int(total_q[i])} {int(score_q[i])} " + " ".join(str(int(v)) for v in idx[i].tolist()) + "\n")


if __name__ == "__main__":
    main()
