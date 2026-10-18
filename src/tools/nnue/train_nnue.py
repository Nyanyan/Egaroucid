"""Train the pattern NNUE on records made by convert_data.cpp (and P values from eval_positions.cpp).

Model (all evaluations in discs for the side to move):
  idx    : 64 pattern instances -> rows of one shared table (src/tools/nnue/nnue_patterns.py)
  L      : linear evaluation, one scalar per row and phase bucket + a disc-count term  (light eval)
  acc    : sum of the 64 first-layer rows (width W) + bias, clipped to [0, 1]
  x      : acc[:W/2] * acc[W/2:]                                   (pairwise product)
  R      : per-bucket head  W/2 -> H -> 1  (clipped ReLU)          (non-linear correction)
modes:
  direct   : output = L + R, loss = mse(L + R) + lambda_l * mse(L)
  residual : output = P + R, P = production pattern evaluation (no L)
  linear   : output = L
Phase bucket = (60 - empties) // 4, 15 buckets.

Usage:
  python src/tools/nnue/train_nnue.py --data <merged dir> --out <run dir> [--mode direct] [--width 64]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import nnue_patterns as pat

N_BUCKETS = 15
REC = np.dtype([("p", "<u8"), ("o", "<u8"), ("s", "i1"), ("e", "u1"), ("src", "u1"), ("r", "u1")])


def bucket_of(empties):
    return torch.clamp((60 - empties) // 4, 0, N_BUCKETS - 1)


class Features:
    """Bitboards -> 64 instance row indices, on the GPU."""

    def __init__(self, device):
        n_inst = len(pat.INSTANCES)
        cells = torch.full((n_inst, 9), 64, dtype=torch.long)
        pow3 = torch.zeros((n_inst, 9), dtype=torch.long)
        offs = torch.zeros(n_inst, dtype=torch.long)
        for i, inst in enumerate(pat.INSTANCES):
            for k, (x, y) in enumerate(inst["cells"]):
                cells[i, k] = 8 * y + x
                pow3[i, k] = 3 ** k
            offs[i] = pat.PATTERNS[inst["pattern"]]["offset"]
        self.cells = cells.to(device).view(-1)
        self.pow3 = pow3.to(device)
        self.offs = offs.to(device)
        self.shifts = (63 - torch.arange(64, device=device)).view(1, 64)
        perms = torch.zeros((8, 64), dtype=torch.long)
        for s, f in enumerate(pat.SYMMETRIES):
            for y in range(8):
                for x in range(8):
                    nx, ny = f(x, y)
                    perms[s, 8 * ny + nx] = 8 * y + x  # new board at s(c) holds the old cell c
        self.perms = perms.to(device)
        self.device = device

    def states(self, p, o, sym=None):
        st = ((p.view(-1, 1) >> self.shifts) & 1) + 2 * ((o.view(-1, 1) >> self.shifts) & 1)
        if sym is not None:
            st = st.gather(1, self.perms[sym])
        return st

    def indices(self, st):
        ext = torch.cat([st, torch.zeros((st.shape[0], 1), dtype=st.dtype, device=st.device)], 1)
        g = ext[:, self.cells].view(st.shape[0], -1, 9)
        return (g * self.pow3).sum(-1) + self.offs


class Net(nn.Module):
    def __init__(self, width, hidden, mode):
        super().__init__()
        self.mode, self.width, self.hidden = mode, width, hidden
        rows = pat.N_ROWS
        if mode != "linear":
            self.ft = nn.Embedding(rows, width)
            nn.init.uniform_(self.ft.weight, -0.02, 0.02)
            self.ft_bias = nn.Parameter(torch.full((width,), 0.25))
            self.l1 = nn.Linear(width // 2, hidden * N_BUCKETS)
            self.out = nn.Linear(hidden, N_BUCKETS)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        if mode != "residual":
            self.lin = nn.Embedding(rows * N_BUCKETS, 1)
            nn.init.zeros_(self.lin.weight)
            self.num = nn.Embedding(65 * N_BUCKETS, 1)
            nn.init.zeros_(self.num.weight)

    def linear(self, idx, bucket, n_player):
        rows = pat.N_ROWS
        return self.lin(idx + (bucket * rows).view(-1, 1)).sum((1, 2)) + self.num(bucket * 65 + n_player).view(-1)

    def correction(self, idx, bucket):
        n = idx.shape[0]
        acc = self.ft(idx).sum(1) + self.ft_bias
        a = torch.clamp(acc, 0.0, 1.0)
        h = self.width // 2
        x = a[:, :h] * a[:, h:]
        z = self.l1(x).view(n, N_BUCKETS, self.hidden)
        z = z[torch.arange(n, device=idx.device), bucket]
        z = torch.clamp(z, 0.0, 1.0)
        r = self.out(z)
        return r[torch.arange(n, device=idx.device), bucket] * 16.0

    def forward(self, idx, bucket, n_player, p_eval):
        lin = self.linear(idx, bucket, n_player) if self.mode != "residual" else None
        if self.mode == "linear":
            return lin, lin
        r = self.correction(idx, bucket)
        base = p_eval if self.mode == "residual" else lin
        return base + r, lin

    def clip_for_quantization(self):
        if self.mode == "linear":
            return
        with torch.no_grad():
            self.ft.weight.clamp_(-63.0, 63.0)
            self.l1.weight.clamp_(-127 / 64, 127 / 64)
            self.out.weight.clamp_(-127 / 64, 127 / 64)


def load_split(data_dir, name):
    a = np.memmap(Path(data_dir) / f"{name}.bin", dtype=REC, mode="r")
    p = np.memmap(Path(data_dir) / f"{name}.p.f32", dtype=np.float32, mode="r")
    return a, p


def to_tensors(a, p, sel=None):
    """Contiguous per-field tensors; sel picks rows (sorted indices keep the reads sequential)."""
    if sel is None:
        sel = slice(None)
    return {
        "p": torch.from_numpy(np.ascontiguousarray(a["p"][sel]).view(np.int64)),
        "o": torch.from_numpy(np.ascontiguousarray(a["o"][sel]).view(np.int64)),
        "s": torch.from_numpy(np.ascontiguousarray(a["s"][sel])),
        "e": torch.from_numpy(np.ascontiguousarray(a["e"][sel])),
        "P": torch.from_numpy(np.ascontiguousarray(p[sel])),
    }


def batch_inputs(feat, b, device, augment):
    p = b["p"].to(device, non_blocking=True)
    o = b["o"].to(device, non_blocking=True)
    e = b["e"].to(device, non_blocking=True).long()
    sym = torch.randint(0, 8, (p.shape[0],), device=device) if augment else None
    st = feat.states(p, o, sym)
    idx = feat.indices(st)
    n_player = (st == 1).sum(1)
    return idx, bucket_of(e), n_player, e


@torch.no_grad()
def evaluate(model, feat, val, device, bs=65536):
    model.eval()
    n = val["s"].shape[0]
    sums = {}
    for i in range(0, n, bs):
        b = {k: v[i:i + bs] for k, v in val.items()}
        idx, bucket, n_player, e = batch_inputs(feat, b, device, False)
        y = b["s"].to(device).float()
        P = b["P"].to(device)
        out, lin = model(idx, bucket, n_player, P)
        band = (e // 10).clamp(0, 5)
        for name, v in (("model", out), ("linear", lin), ("P", P)):
            if v is None:
                continue
            err = (v - y) ** 2
            for k in range(6):
                m = band == k
                key = (name, k)
                s0, c0 = sums.get(key, (0.0, 0))
                sums[key] = (s0 + float(err[m].sum()), c0 + int(m.sum()))
    model.train()
    res = {}
    for (name, k), (s0, c0) in sums.items():
        res.setdefault(name, {})[f"{10 * k:02d}-{10 * k + 9:02d}"] = s0 / max(c0, 1)
    for name in res:
        tot = sum(sums[(name, k)][0] for k in range(6) if (name, k) in sums)
        cnt = sum(sums[(name, k)][1] for k in range(6) if (name, k) in sums)
        res[name]["all"] = tot / max(cnt, 1)
        mid = [k for k in (3, 4, 5) if (name, k) in sums]
        res[name]["30-59"] = sum(sums[(name, k)][0] for k in mid) / max(sum(sums[(name, k)][1] for k in mid), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="direct", choices=["direct", "residual", "linear"])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-final", type=float, default=2e-5)
    ap.add_argument("--lambda-l", type=float, default=0.5)
    ap.add_argument("--lin-lr-mult", type=float, default=25.0, help="learning-rate multiplier of the linear tables")
    ap.add_argument("--boards-weight", type=float, default=1.0, help="sampling weight of a board-file record vs a transcript record")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--init", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")

    t0 = time.time()
    tr_a, tr_p = load_split(args.data, "train_tr")
    bd_a, bd_p = load_split(args.data, "train_boards")
    va_a, va_p = load_split(args.data, "val_tr")
    # hold out 1% of the board-file records (that file has no game boundaries)
    ids = np.arange(len(bd_a))
    hold = ids % 100 == 37
    train_tr = to_tensors(tr_a, tr_p)
    train_bd = to_tensors(bd_a, bd_p, ids[~hold])
    val_bd = to_tensors(bd_a, bd_p, ids[hold])
    vsel = np.sort(rng.permutation(len(va_a))[: args.val_size])
    val_tr = to_tensors(va_a, va_p, vsel)
    n_tr, n_bd = train_tr["s"].shape[0], train_bd["s"].shape[0]
    frac_bd = args.boards_weight * n_bd / (n_tr + args.boards_weight * n_bd)
    print(f"loaded train {n_tr} + {n_bd} (board share {frac_bd:.3f}), val {len(vsel)} + {int(hold.sum())} in {time.time() - t0:.0f}s", flush=True)

    feat = Features(device)
    model = Net(args.width, args.hidden, args.mode).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    lin_params = [p for n, p in model.named_parameters() if n.startswith(("lin.", "num."))]
    other_params = [p for n, p in model.named_parameters() if not n.startswith(("lin.", "num."))]
    groups = [g for g in ({"params": other_params, "mult": 1.0}, {"params": lin_params, "mult": args.lin_lr_mult}) if g["params"]]
    opt = torch.optim.Adam(groups, lr=args.lr)
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    best = math.inf
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = (step - 1) / max(args.steps - 1, 1)
        lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr * g["mult"]
        nb = int(np.random.binomial(args.batch, frac_bd))
        sel_tr = torch.randint(0, n_tr, (args.batch - nb,))
        sel_bd = torch.randint(0, n_bd, (nb,))
        b = {k: torch.cat([train_tr[k][sel_tr], train_bd[k][sel_bd]]) for k in train_tr}
        idx, bucket, n_player, _ = batch_inputs(feat, b, device, True)
        y = b["s"].to(device, non_blocking=True).float()
        P = b["P"].to(device, non_blocking=True)
        pred, lin = model(idx, bucket, n_player, P)
        loss = ((pred - y) ** 2).mean()
        if args.mode == "direct" and args.lambda_l > 0:
            loss = loss + args.lambda_l * ((lin - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.clip_for_quantization()
        if step % args.val_every == 0 or step == args.steps:
            vt = evaluate(model, feat, val_tr, device)
            vb = evaluate(model, feat, val_bd, device)
            rec = {"step": step, "lr": lr, "loss": loss.item(), "elapsed": time.time() - t0, "val_tr": vt, "val_boards": vb}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            key = vt["model"]["30-59"]
            msg = (f"step {step} lr {lr:.2e} loss {loss.item():.2f} | val_tr model {vt['model']['all']:.2f} "
                   f"(30-59 {key:.2f}) P {vt['P']['all']:.2f} (30-59 {vt['P']['30-59']:.2f})")
            if "linear" in vt:
                msg += f" linear {vt['linear']['all']:.2f}"
            msg += f" | boards model {vb['model']['all']:.2f} P {vb['P']['all']:.2f} | {time.time() - t0:.0f}s"
            print(msg, flush=True)
            torch.save(model.state_dict(), out / "last.pt")
            if key < best:
                best = key
                torch.save(model.state_dict(), out / "best.pt")
    log.close()


if __name__ == "__main__":
    main()
