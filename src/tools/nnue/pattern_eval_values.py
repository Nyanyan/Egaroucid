"""Pattern-evaluation values of NNUE training records for a run of train_pattern_eval.py (the P input of
residual NNUE training), exactly as the engine computes them: quantized table rows + disc-count value, in discs.

Usage:
    python src/tools/nnue/pattern_eval_values.py --run ignored/pattern_runs/nnue12_main --suffix .p12.f32 a.bin b.bin ...
writes a.p12.f32 next to a.bin (one float32 per record).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import train_pattern_eval as tpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--suffix", default=".p12.f32")
    ap.add_argument("--chunk", type=int, default=1 << 18)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    run_args = __import__("json").loads((Path(args.run) / "args.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    ps = tpe.PatternSet(run_args["patterns"])
    feat = tpe.GpuFeatures(ps, device, fold=False)  # weights.npz already holds every row's folded weight
    w = np.load(Path(args.run) / "weights.npz")
    qt = torch.clamp(torch.round(torch.from_numpy(w["table"]).to(device) * tpe.STEP), -tpe.EVAL_MAX, tpe.EVAL_MAX).to(torch.int32)
    qn = torch.round(torch.from_numpy(w["num"]).to(device) * tpe.STEP).to(torch.int32)
    qt_flat, qn_flat = qt.view(-1), qn.view(-1)
    for f in args.files:
        a = np.memmap(f, dtype=tpe.REC, mode="r")
        out = Path(str(Path(f).with_suffix("")) + args.suffix)
        res = np.empty(len(a), dtype=np.float32)
        for i in range(0, len(a), args.chunk):
            b = a[i:i + args.chunk]
            p = torch.from_numpy(np.ascontiguousarray(b["p"]).view(np.int64)).to(device)
            o = torch.from_numpy(np.ascontiguousarray(b["o"]).view(np.int64)).to(device)
            phase = (60 - torch.from_numpy(np.ascontiguousarray(b["e"]).astype(np.int64)).to(device)).clamp(0, tpe.N_PHASES - 1)
            idx, n_player = feat(p, o)
            v = qt_flat[idx + (phase * ps.n_rows).view(-1, 1)].sum(1) + qn_flat[phase * 65 + n_player]
            res[i:i + len(b)] = (v.float() / tpe.STEP).cpu().numpy()
        res.tofile(out)
        print(f"{f}: {len(a)} records -> {out}", flush=True)


if __name__ == "__main__":
    main()
