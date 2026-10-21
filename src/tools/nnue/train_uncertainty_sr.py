"""Train an uncertainty estimate on search residuals instead of game results.

Rows of mid_probcut_dataset_tool with shallow depth 0 give, for a position, e = deep search value - static NNUE
value (the error of the MPC static test at that depth). The model outputs u = log sigma^2 of e, trained with the
Gaussian negative log likelihood u + e^2 exp(-u):
  - default: an uncertainty head on the main network's first layer (dual network, train_uncertainty_head.py's
    UHead; --int-trunk takes the features from the engine's integer weights). The run is read by
    train_mpc_error.py --dual and export_uncertainty_head.py.
  - --own-width W: a separate network with its own first layer of width W (like train_uncertainty.py), for
    comparison; read by train_mpc_error.py --unc.
Boards are split into training and validation sets by a hash of the board (all positions of the MPC benchmark
data are held-out transcripts, these are training transcripts). best.pt follows the validation NLL.

Usage: python src/tools/nnue/train_uncertainty_sr.py --tsv "ignored/mpc_sr_data/sr_d8_*.tsv" --main ignored/nnue_runs/residual_w64_fold_d3 \
           --int-trunk bin/resources/nnue.egnn --out ignored/nnue_runs/uhead_sr_axz16 [--inputs axz --hidden 16 | --own-width 32]
"""
import argparse
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nnue import Features, Net, bucket_of  # noqa: E402
from train_uncertainty_head import N_IN, IntTrunk, UHead, head_inputs, load_run, trunk  # noqa: E402


def read_rows(pattern):
    boards, errs = [], []
    for f in sorted(glob.glob(pattern)):
        if not Path(f).with_suffix(".done").exists():
            continue
        for line in Path(f).read_text(encoding="utf-8").splitlines()[1:]:
            c = line.split("\t")
            if len(c) < 15 or c[4] != "0":
                continue
            boards.append(c[1])
            errs.append(int(c[7]) - int(c[5]))
    return boards, np.array(errs, np.float32)


def board_bits(board):
    cells, turn = board.split()
    p = o = 0
    for k, ch in enumerate(cells):
        if ch == "-":
            continue
        if ch == turn:
            p |= 1 << (63 - k)
        else:
            o |= 1 << (63 - k)
    return p, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--main", required=True, help="run directory of the main NNUE")
    ap.add_argument("--int-trunk", default="", help="the engine weight file (.egnn, int8) exported from --main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default="axz", choices=list(N_IN))
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--own-width", type=int, default=0, help="a separate network with its own first layer instead of a head")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-final", type=float, default=2e-5)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-mod", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    boards, errs = read_rows(args.tsv)
    print(f"rows {len(boards)}", flush=True)
    bits = np.array([board_bits(b) for b in boards], dtype=np.uint64)
    is_val = np.array([int(hashlib.md5(b.encode()).hexdigest(), 16) % args.val_mod == 0 for b in boards])
    p_all = torch.from_numpy(bits[:, 0].view(np.int64)).to(device)
    o_all = torch.from_numpy(bits[:, 1].view(np.int64)).to(device)
    e_all = torch.from_numpy(errs).to(device)
    tr_idx = torch.from_numpy(np.flatnonzero(~is_val)).to(device)
    va_idx = torch.from_numpy(np.flatnonzero(is_val)).to(device)

    if args.own_width:
        model = Net(args.own_width, 32, "residual").to(device)
        feat = Features(device, True)
        params = list(model.parameters())
        (out / "args.json").write_text(json.dumps(dict(vars(args), width=args.own_width, hidden=32, mode="residual", fold=True), indent=1), encoding="utf-8")

        def predict(sel):
            st = feat.states(p_all[sel], o_all[sel])
            idx = feat.indices(st)
            return model.correction(idx, bucket_of(64 - (st != 0).sum(1)))
    else:
        main_model, mfeat, _ = load_run(args.main, device)
        it = IntTrunk(args.int_trunk, device) if args.int_trunk else None
        feat = it.feat if it is not None else mfeat
        model = UHead(N_IN[args.inputs](main_model), args.hidden).to(device)
        params = list(model.parameters())
        (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")

        def predict(sel):
            st = feat.states(p_all[sel], o_all[sel])
            idx = feat.indices(st)
            bucket = bucket_of(64 - (st != 0).sum(1))
            with torch.no_grad():
                t = it(idx, bucket)[1] if it is not None else trunk(main_model, idx, bucket)
            return model(head_inputs(t, args.inputs), bucket)

    opt = torch.optim.Adam(params, lr=args.lr)
    const = float(torch.log((e_all[va_idx] ** 2).mean()) + 1.0)
    best = math.inf
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = (step - 1) / max(args.steps - 1, 1)
        for g in opt.param_groups:
            g["lr"] = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * frac))
        sel = tr_idx[torch.randint(0, tr_idx.shape[0], (args.batch,), device=device)]
        u = predict(sel)
        loss = (u + e_all[sel] ** 2 * torch.exp(-u)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if args.own_width:
            model.clip_for_quantization()
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                tot = 0.0
                for i in range(0, va_idx.shape[0], 65536):
                    vs = va_idx[i:i + 65536]
                    uu = predict(vs)
                    tot += float((uu + e_all[vs] ** 2 * torch.exp(-uu)).sum())
            model.train()
            nll = tot / va_idx.shape[0]
            rec = {"step": step, "loss": loss.item(), "val_nll": nll, "val_nll_const": const, "elapsed": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step} loss {loss.item():.4f} | val nll {nll:.4f} (constant {const:.4f}) | {time.time() - t0:.0f}s", flush=True)
            torch.save(model.state_dict(), out / "last.pt")
            if nll < best:
                best = nll
                torch.save(model.state_dict(), out / "best.pt")
    log.close()


if __name__ == "__main__":
    main()
