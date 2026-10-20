"""Turn the second output of a dual network (train_nnue.py --u-weight) into an uncertainty-head run, so that
export_uncertainty_head.py can write it for the engine: inputs z (the last hidden layer), hidden 0 (linear).

Usage: python src/tools/nnue/joint_to_uhead.py --run ignored/nnue_runs/joint60k_l3 --int-trunk bin/resources/<exported>.egnn \
           --out ignored/nnue_runs/joint60k_l3_uhead
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_uncertainty_head import UHead  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--int-trunk", required=True, help="the engine weight file exported from --run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    targs = json.loads((Path(args.run) / "args.json").read_text(encoding="utf-8"))
    sd = torch.load(Path(args.run) / "best.pt", map_location="cpu")
    head = UHead(targs["hidden"], 0)
    with torch.no_grad():
        head.out.weight.copy_(sd["uout.weight"])
        head.out.bias.copy_(sd["uout.bias"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "best.pt")
    (out / "args.json").write_text(json.dumps({"main": str(args.run), "inputs": "z", "hidden": 0, "int_trunk": str(args.int_trunk)}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
