"""Write the engine file of an uncertainty head (train_uncertainty_head.py or train_dual_finetune.py), its mean u
by disc count, and test vectors for the engine.

The head belongs to one main weight file: the file stores that file's FNV-1a 64 hash and the engine refuses a
head whose hash does not match (src/engine/nnue/nnue_eval.hpp init_uhead). The mean u per disc count (used by
MPC nodes that skip the head) is measured with the engine's integer first layer on held-out transcripts and
written to <run>/mean_u.json. Test vectors (board, pattern value, expected total and u) go to <out>.vectors.txt
(src/tools/nnue/uhead_check.cpp).

Usage: python src/tools/nnue/export_uncertainty_head.py --run ignored/nnue_runs/uhead_d3_axz_h16_int \
           --main-file bin/resources/nnue.egnn --data ignored/nnue_data2/merged --out bin/resources/unused/nnue_uhead.egnh
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nnue import N_BUCKETS, batch_inputs, load_split, to_tensors  # noqa: E402
from train_uncertainty_head import N_IN, IntTrunk, UHead, head_inputs  # noqa: E402

INPUTS = {"x": 0, "xz": 1, "axz": 2, "z": 3}


def fnv1a64(path):
    """FNV-1a 64 over the file's bytes (a few seconds for a 15 MB file)"""
    h = 1469598103934665603
    mask = (1 << 64) - 1
    for b in Path(path).read_bytes():
        h = ((h ^ b) * 1099511628211) & mask
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="head run (best.pt + args.json with inputs and hidden)")
    ap.add_argument("--main-file", required=True, help="the engine weight file (.egnn, int8) the head belongs to")
    ap.add_argument("--data", required=True, help="merged data dir (val_tr: mean u per disc count, test vectors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-vectors", type=int, default=2000)
    ap.add_argument("--p-suffix", default=".p.f32", help="suffix of the P files (as train_nnue.py --p-suffix)")
    args = ap.parse_args()
    import train_nnue
    train_nnue.P_SUFFIX = args.p_suffix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hargs = json.loads((Path(args.run) / "args.json").read_text(encoding="utf-8"))
    it = IntTrunk(args.main_file, device)

    class Dims:
        width, hidden = it.W, it.H
    n_private = hargs.get("private", 0)
    head = UHead(N_IN[hargs["inputs"]](Dims), hargs["hidden"], n_private).to(device)
    head.load_state_dict(torch.load(Path(args.run) / "best.pt", map_location=device))
    head.eval()
    n_in, hidden = head.n_in + n_private, head.hidden
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in head.state_dict().items()}
    main_hash = fnv1a64(args.main_file)
    with open(args.out, "wb") as f:
        f.write(b"EGUHEAD1")
        f.write(np.array([2, INPUTS[hargs["inputs"]], n_in, hidden, it.W, it.H, N_BUCKETS, n_private], np.int32).tobytes())
        f.write(np.array([main_hash], np.uint64).tobytes())
        if hidden:
            f.write(np.ascontiguousarray(sd["l1.weight"].reshape(N_BUCKETS, hidden, n_in)).tobytes())
            f.write(np.ascontiguousarray(sd["l1.bias"].reshape(N_BUCKETS, hidden)).tobytes())
        # hidden 0: one linear output per bucket on the inputs, out_w[buckets][n_in]
        f.write(np.ascontiguousarray(sd["out.weight"].reshape(N_BUCKETS, hidden or n_in)).tobytes())
        f.write(np.ascontiguousarray(sd["out.bias"].reshape(N_BUCKETS)).tobytes())
        if n_private:
            # private rows in 1/64 (int8), unfolded: every row holds its canonical row; bias in 1/64 (int16)
            w = np.round(sd["private.weight"].astype(np.float64) * 64)
            assert np.abs(w).max() <= 127
            fold = head.fold.cpu().numpy()
            f.write(np.ascontiguousarray(w[fold].astype(np.int8)).tobytes())
            f.write(np.round(sd["private_bias"].astype(np.float64) * 64).astype(np.int16).tobytes())
    print("wrote", args.out, "hash of the head file", fnv1a64(args.out))

    va_a, va_p = load_split(args.data, "val_tr")
    sel = np.sort(np.random.default_rng(20261020).permutation(len(va_a))[:1000000])
    val = to_tensors(va_a, va_p, sel)
    sums, cnts = np.zeros(65), np.zeros(65)
    vec_rows = []
    with torch.no_grad():
        for i in range(0, val["s"].shape[0], 65536):
            vb = {k: v[i:i + 65536] for k, v in val.items()}
            idx, bucket, _, e = batch_inputs(it.feat, vb, device, False)
            oacc, t = it(idx, bucket)
            u = head(head_inputs(t, hargs["inputs"]), bucket, idx).cpu().numpy().astype(np.float64)
            n = 64 - e.cpu().numpy()
            np.add.at(sums, n, u)
            np.add.at(cnts, n, 1)
            if len(vec_rows) < args.n_vectors:
                p_raw = np.round(vb["P"].numpy().astype(np.float64) * 32).astype(np.int64)
                total = p_raw * 256 + oacc.cpu().numpy()
                for k in range(min(args.n_vectors - len(vec_rows), len(u))):
                    pv = int(vb["p"][k]) & ((1 << 64) - 1)
                    ov = int(vb["o"][k]) & ((1 << 64) - 1)
                    vec_rows.append(f"{pv:016x} {ov:016x} {int(p_raw[k])} {int(total[k])} {u[k]:.6f}")
    mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    have = np.flatnonzero(cnts > 0)
    for k in range(65):
        if cnts[k] == 0:
            mean[k] = mean[have[np.argmin(np.abs(have - k))]]
    (Path(args.run) / "mean_u.json").write_text(json.dumps({"mean_u": mean.tolist(), "count": cnts.tolist()}, indent=1), encoding="utf-8")
    Path(args.out).with_suffix(".vectors.txt").write_text("\n".join(vec_rows) + "\n", encoding="utf-8")
    print("mean u by disc count:", np.round(mean, 3).tolist())


if __name__ == "__main__":
    main()
