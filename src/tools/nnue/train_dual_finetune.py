"""Fine-tune the main NNUE and an uncertainty head together (dual network, shared first layer).

Starting from a trained main network (frozen copy kept as the reference) and optionally a head from
train_uncertainty_head.py, train
    loss = mean (y - y_ref)^2 + lambda * mean (u + r^2 exp(-u)),   r = result - y (no gradient through r)
on the transcript records: the evaluation y is held to the reference network's by distillation, and the
first layer may change so that the uncertainty head u (on the shared first-layer features) predicts the
residual's size better. Validation reports how far the evaluation moved (mean |y - y_ref|, and the squared
error against the final results for both) and the uncertainty NLL.

Output: <out>/best.pt + args.json (a main-network run that export_nnue.py reads) and <out>/uhead/
(best.pt + args.json, a head run that load_uhead / train_mpc_error.py --dual read).

Usage:
    python src/tools/nnue/train_dual_finetune.py --data ignored/nnue_data2/merged --main ignored/nnue_runs/residual_w64_fold_d3 \
        --uhead ignored/nnue_runs/uhead_d3_x_h16 --out ignored/nnue_runs/dual_d3_l03 --lam 0.3 --steps 20000
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
from train_nnue import REC, batch_inputs, load_split, to_tensors  # noqa: E402
from train_uncertainty_head import N_IN, UHead, head_inputs, load_run, trunk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--uhead", default="", help="head run to start from (its inputs and size are used)")
    ap.add_argument("--inputs", default="x", choices=list(N_IN), help="without --uhead")
    ap.add_argument("--hidden", type=int, default=16, help="without --uhead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr-trunk", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-final-mult", type=float, default=0.01)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--train-size", type=int, default=100000000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--empties-ref", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    (out / "uhead").mkdir(parents=True, exist_ok=True)
    model, feat, targs = load_run(args.main, device)
    ref, _, _ = load_run(args.main, device)
    for p in model.parameters():
        p.requires_grad_(True)
    inputs, hidden = args.inputs, args.hidden
    if args.uhead:
        uargs = json.loads((Path(args.uhead) / "args.json").read_text(encoding="utf-8"))
        inputs, hidden = uargs["inputs"], uargs["hidden"]
    head = UHead(N_IN[inputs](model), hidden).to(device)
    if args.uhead:
        head.load_state_dict(torch.load(Path(args.uhead) / "best.pt", map_location=device))
    main_args = {k: targs[k] for k in ("width", "hidden", "mode", "fold", "bucket_bias") if k in targs}
    (out / "args.json").write_text(json.dumps(dict(main_args, finetune=vars(args)), indent=1), encoding="utf-8")
    (out / "uhead" / "args.json").write_text(json.dumps({"main": str(out), "inputs": inputs, "hidden": hidden}, indent=1), encoding="utf-8")

    tr_a, tr_p = load_split(args.data, "train_tr")
    va_a, va_p = load_split(args.data, "val_tr")
    rng = np.random.default_rng(args.seed)
    block = 1000000
    n_blocks = min(len(tr_a) // block, max(1, args.train_size // block))
    starts = np.sort(rng.choice(len(tr_a) // block, n_blocks, replace=False)) * block
    parts = [to_tensors(tr_a[s:s + block], tr_p[s:s + block]) for s in starts]
    train = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    del parts
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
        rf = np.bincount(np.asarray(np.memmap(args.empties_ref, dtype=REC, mode="r")["e"]), minlength=61).astype(np.float64)
        rf[cnt == 0] = 0
        ref_p = torch.from_numpy(rf / rf.sum())

    trunk_params = [model.ft.weight, model.ft_bias] + ([model.ft_bias_b] if model.ft_bias_b is not None else [])
    eval_head_params = list(model.l1.parameters()) + list(model.out.parameters())
    opt = torch.optim.Adam([{"params": trunk_params + eval_head_params, "lr": args.lr_trunk, "base": args.lr_trunk},
                            {"params": list(head.parameters()), "lr": args.lr_head, "base": args.lr_head}])

    def run(b):
        idx, bucket, n_player, e = batch_inputs(feat, b, device, False)
        P = b["P"].to(device, non_blocking=True)
        s = b["s"].to(device, non_blocking=True).float()
        with torch.no_grad():
            y_ref, _ = ref(idx, bucket, n_player, P)
        y, _ = model(idx, bucket, n_player, P)
        u = head(head_inputs(trunk(model, idx, bucket), inputs), bucket)
        r = s - y.detach()
        return y, y_ref, u, r, s, e

    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = (step - 1) / max(args.steps - 1, 1)
        for g in opt.param_groups:
            lo = g["base"] * args.lr_final_mult
            g["lr"] = lo + 0.5 * (g["base"] - lo) * (1 + math.cos(math.pi * frac))
        if ref_p is not None:
            ee = torch.multinomial(ref_p, args.batch, replacement=True)
            sel = order[start[ee] + (torch.rand(args.batch, dtype=torch.float64) * count[ee]).long()]
        else:
            sel = torch.randint(0, n_tr, (args.batch,))
        b = {k: v[sel] for k, v in train.items()}
        y, y_ref, u, r, _, _ = run(b)
        distill = ((y - y_ref) ** 2).mean()
        unc = (u + r ** 2 * torch.exp(-u)).mean()
        loss = distill + args.lam * unc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.clip_for_quantization()
        if step % args.val_every == 0 or step == args.steps:
            model.eval(); head.eval()
            with torch.no_grad():
                sums = {"drift": 0.0, "mse": 0.0, "mse_ref": 0.0, "nll": 0.0, "mse30": 0.0, "mse30_ref": 0.0}
                n = n30 = 0
                for i in range(0, val["s"].shape[0], 65536):
                    vb = {k: v[i:i + 65536] for k, v in val.items()}
                    y, y_ref, u, r, s, e = run(vb)
                    sums["drift"] += float((y - y_ref).abs().sum())
                    sums["mse"] += float(((y - s) ** 2).sum())
                    sums["mse_ref"] += float(((y_ref - s) ** 2).sum())
                    sums["nll"] += float((u + r ** 2 * torch.exp(-u)).sum())
                    m = e >= 30
                    sums["mse30"] += float(((y - s) ** 2)[m].sum())
                    sums["mse30_ref"] += float(((y_ref - s) ** 2)[m].sum())
                    n += s.shape[0]
                    n30 += int(m.sum())
            model.train(); head.train()
            rec = {"step": step, "loss": loss.item(), "distill": distill.item(), "elapsed": time.time() - t0,
                   "val_drift_mae": sums["drift"] / n, "val_mse": sums["mse"] / n, "val_mse_ref": sums["mse_ref"] / n,
                   "val_mse_30_59": sums["mse30"] / n30, "val_mse_30_59_ref": sums["mse30_ref"] / n30, "val_nll": sums["nll"] / n}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step} | drift {rec['val_drift_mae']:.3f} | mse {rec['val_mse']:.3f} (ref {rec['val_mse_ref']:.3f}) "
                  f"30-59 {rec['val_mse_30_59']:.3f} (ref {rec['val_mse_30_59_ref']:.3f}) | u nll {rec['val_nll']:.4f} | {time.time() - t0:.0f}s", flush=True)
            torch.save(model.state_dict(), out / "last.pt")
            torch.save(head.state_dict(), out / "uhead" / "last.pt")
            # the last step is the result (the learning rate has decayed); best.pt follows the last validation
            torch.save(model.state_dict(), out / "best.pt")
            torch.save(head.state_dict(), out / "uhead" / "best.pt")
    log.close()


if __name__ == "__main__":
    main()
