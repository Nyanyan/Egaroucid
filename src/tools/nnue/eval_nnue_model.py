"""Score trained NNUE models on the same held-out data (all of val_tr and the 1% board hold-out).

The trainer's validation uses a sample of val_tr picked by --val-seed, so runs with different seeds
are not directly comparable; this scores every model on the whole held-out set.

Usage: python src/tools/nnue/eval_nnue_model.py --data ignored/nnue_data/merged ignored/nnue_runs/a ignored/nnue_runs/b[:last.pt] ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nnue import Features, Net, evaluate, load_split, to_tensors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directory, optionally :model file (default best.pt)")
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    va_a, va_p = load_split(args.data, "val_tr")
    val_tr = to_tensors(va_a, va_p)
    bd_a, bd_p = load_split(args.data, "train_boards")
    ids = np.arange(len(bd_a))
    val_bd = to_tensors(bd_a, bd_p, ids[ids % 100 == 37])
    for spec in args.runs:
        run, _, model_file = spec.partition(":")
        run = Path(run)
        targs = json.loads((run / "args.json").read_text(encoding="utf-8"))
        model = Net(targs["width"], targs["hidden"], targs["mode"], bool(targs.get("bucket_bias"))).to(device)
        model.load_state_dict(torch.load(run / (model_file or "best.pt"), map_location=device))
        feat = Features(device, bool(targs.get("fold")))
        vt = evaluate(model, feat, val_tr, device)
        vb = evaluate(model, feat, val_bd, device)
        bands = " ".join(f"{k} {v:.2f}" for k, v in vt["model"].items() if k not in ("all", "30-59"))
        print(f"{spec}: val_tr all {vt['model']['all']:.3f} 30-59 {vt['model']['30-59']:.3f} (P {vt['P']['all']:.3f} / {vt['P']['30-59']:.3f}) | "
              f"boards {vb['model']['all']:.3f} (P {vb['P']['all']:.3f}) | {bands}", flush=True)


if __name__ == "__main__":
    main()
