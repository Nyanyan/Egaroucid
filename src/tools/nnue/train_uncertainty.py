"""Train a per-position uncertainty network: how far is the final result from the NNUE's prediction?

A trained NNUE (frozen) predicts y_hat; a second network U on the same pattern features outputs
u = log sigma^2 of e = y - y_hat, trained with the Gaussian negative log likelihood
    loss = u + e^2 exp(-u)
on the transcript records (final results). The aim is a cheap estimate of how volatile a
position is, as a candidate input for per-position MPC margins.

Usage:
    python src/tools/nnue/train_uncertainty.py --data ignored/nnue_data2/merged --main ignored/nnue_runs/residual_w64_fold_d2b \
        --out ignored/nnue_runs/uncertainty_w32 --width 32 --steps 50000 [--empties-ref <record file>]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nnue import REC, Features, Net, batch_inputs, load_split, to_tensors  # noqa: E402


def load_main(run, device):
    targs = json.loads((Path(run) / "args.json").read_text(encoding="utf-8"))
    model = Net(targs["width"], targs["hidden"], targs["mode"], bool(targs.get("bucket_bias"))).to(device)
    model.load_state_dict(torch.load(Path(run) / "best.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, Features(device, bool(targs.get("fold")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--main", required=True, help="run directory of the trained NNUE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-final", type=float, default=2e-5)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--empties-ref", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(dict(vars(args), mode="residual", fold=True), indent=1), encoding="utf-8")
    main_model, main_feat = load_main(args.main, device)
    tr_a, tr_p = load_split(args.data, "train_tr")
    va_a, va_p = load_split(args.data, "val_tr")
    train = to_tensors(tr_a, tr_p)
    vsel = np.sort(np.random.default_rng(20261019).permutation(len(va_a))[:500000])
    val = to_tensors(va_a, va_p, vsel)
    n_tr = train["s"].shape[0]
    order = start = count = ref_p = None
    if args.empties_ref:
        e_tr = train["e"].numpy()
        order = torch.from_numpy(np.argsort(e_tr, kind="stable"))
        cnt = np.bincount(e_tr, minlength=61)
        count = torch.from_numpy(cnt)
        start = torch.from_numpy(np.concatenate([[0], np.cumsum(cnt)[:-1]]))
        ref = np.bincount(np.asarray(np.memmap(args.empties_ref, dtype=REC, mode="r")["e"]), minlength=61).astype(np.float64)
        ref[cnt == 0] = 0
        ref_p = torch.from_numpy(ref / ref.sum())
    feat = Features(device, True)
    unc = Net(args.width, args.hidden, "residual").to(device)
    opt = torch.optim.Adam(unc.parameters(), lr=args.lr)
    zero = None

    def residual_and_u(b, augment):
        nonlocal zero
        idx_m, bucket_m, np_m, e = batch_inputs(main_feat, b, device, augment)
        P = b["P"].to(device, non_blocking=True)
        with torch.no_grad():
            y_hat, _ = main_model(idx_m, bucket_m, np_m, P)
        y = b["s"].to(device, non_blocking=True).float()
        idx, bucket, n_player, _ = batch_inputs(feat, b, device, augment)
        if zero is None or zero.shape[0] != idx.shape[0]:
            zero = torch.zeros(idx.shape[0], device=device)
        u, _ = unc(idx, bucket, n_player, zero)
        return y - y_hat, u, e

    log = open(out / "log.jsonl", "a", encoding="utf-8")
    best = math.inf
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = (step - 1) / max(args.steps - 1, 1)
        for g in opt.param_groups:
            g["lr"] = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * frac))
        if ref_p is not None:
            ee = torch.multinomial(ref_p, args.batch, replacement=True)
            sel = order[start[ee] + (torch.rand(args.batch, dtype=torch.float64) * count[ee]).long()]
        else:
            sel = torch.randint(0, n_tr, (args.batch,))
        b = {k: v[sel] for k, v in train.items()}
        # the main network uses the same symmetry augmentation as the uncertainty network only when both see the same board;
        # keep both unaugmented here so that the residual belongs to the board the uncertainty network sees
        res, u, _ = residual_and_u(b, False)
        loss = (u + res ** 2 * torch.exp(-u)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        unc.clip_for_quantization()
        if step % args.val_every == 0 or step == args.steps:
            unc.eval()
            with torch.no_grad():
                tot, sq, n, rows = 0.0, 0.0, 0, []
                for i in range(0, val["s"].shape[0], 65536):
                    vb = {k: v[i:i + 65536] for k, v in val.items()}
                    r, uu, e = residual_and_u(vb, False)
                    tot += float((uu + r ** 2 * torch.exp(-uu)).sum())
                    n += r.shape[0]
                    rows.append((r.cpu().numpy(), uu.cpu().numpy(), e.cpu().numpy()))
            unc.train()
            r = np.concatenate([x[0] for x in rows]); uu = np.concatenate([x[1] for x in rows])
            const = float(np.log(np.mean(r ** 2)) + 1.0)  # NLL of a constant variance
            q = np.quantile(uu, [0.2, 0.8])
            lo, hi = r[uu <= q[0]], r[uu >= q[1]]
            rec = {"step": step, "loss": loss.item(), "val_nll": tot / n, "val_nll_const": const,
                   "sd_low20": float(lo.std()), "sd_high20": float(hi.std()), "elapsed": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step} loss {loss.item():.3f} | val nll {tot / n:.4f} (constant {const:.4f}) | residual sd low20 {lo.std():.2f} high20 {hi.std():.2f} | {time.time() - t0:.0f}s", flush=True)
            torch.save(unc.state_dict(), out / "last.pt")
            if tot / n < best:
                best = tot / n
                torch.save(unc.state_dict(), out / "best.pt")
    log.close()


if __name__ == "__main__":
    main()
