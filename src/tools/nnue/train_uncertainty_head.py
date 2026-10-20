"""Train an uncertainty head on the main NNUE's first layer (dual network).

The main NNUE (frozen) gives, for each position, its prediction y_hat and its first-layer features:
a = the clamped accumulator (W values), x = its pairwise products (W/2 values) and z = the clamped hidden
layer of the position's bucket (H values). A small head on these features outputs
    u = log sigma^2 of e = y - y_hat
trained with the Gaussian negative log likelihood u + e^2 exp(-u) on the transcript records (final
results), like train_uncertainty.py, but without a first layer of its own: the engine computes the main
accumulator once and gets both the evaluation and u.

Head: inputs (x; x and z; or a, x and z) -> hidden (one set per bucket) -> clamp [0, 1] -> one output per bucket (x 16).
--private P adds P private first-layer lanes read only by the head (their own small table over the same pattern
rows, sum + bias, clamp [0, 1]), as inputs beside the shared features. Their weights and bias are used on the
1/64 grid (straight-through rounding), so the engine's int8 rows give the same values.

Usage:
    python src/tools/nnue/train_uncertainty_head.py --data ignored/nnue_data2/merged --main ignored/nnue_runs/residual_w64_fold_d3 \
        --out ignored/nnue_runs/uhead_d3 [--inputs xz] [--hidden 16] [--steps 20000] [--empties-ref <record file>] [--compare <uncertainty run>]
--compare evaluates a separate uncertainty network (train_uncertainty.py) on the same validation positions
and residuals. --int-trunk <file.egnn> takes the features and the prediction from the engine's integer weights
(the exported --main run) instead of the float model, so the head is trained on what the engine computes.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnue_patterns as pat  # noqa: E402
from train_nnue import N_BUCKETS, REC, Features, Net, batch_inputs, load_split, make_features, make_net, to_tensors  # noqa: E402


def load_run(run, device, model_file="best.pt"):
    targs = json.loads((Path(run) / "args.json").read_text(encoding="utf-8"))
    model = make_net(targs).to(device)
    model.load_state_dict(torch.load(Path(run) / model_file, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, make_features(device, targs), targs


def trunk(model, idx, bucket):
    """first-layer features of the main network: x (pairwise products), z (clamped hidden layer of the bucket)
    and a (clamped accumulator)"""
    n = idx.shape[0]
    acc = model.ft(idx).sum(1) + model.ft_bias
    if model.ft_bias_b is not None:
        acc = acc + model.ft_bias_b[bucket]
    a = torch.clamp(acc, 0.0, 1.0)
    h = model.width // 2
    x = a[:, :h] * a[:, h:]
    z = model.l1(x).view(n, N_BUCKETS, model.hidden)[torch.arange(n, device=idx.device), bucket]
    return x, torch.clamp(z, 0.0, 1.0), a


class IntTrunk:
    """the engine's integer first layer and hidden layer (src/engine/nnue/nnue_eval.hpp) from a version-2 (int8) .egnn file"""

    def __init__(self, path, device):
        raw = Path(path).read_bytes()
        assert raw[:8] == b"EGNNUE01"
        head = np.frombuffer(raw[8:40], np.int32)
        version, mode, rows, inst, W, H, buckets, shift = (int(v) for v in head)
        assert version == 2 and mode == 1 and rows == pat.N_ROWS and inst == len(pat.INSTANCES) and buckets == N_BUCKETS
        off = 40
        def take(dtype, count):
            nonlocal off
            a = np.frombuffer(raw, dtype, count, off)
            off += a.nbytes
            return a
        self.W, self.H, self.shift = W, H, shift
        t = lambda a, shape: torch.from_numpy(a.astype(np.int64).reshape(shape)).to(device)
        self.ft = torch.from_numpy(take(np.int8, rows * W).reshape(rows, W).astype(np.int16)).to(device)
        self.ft_bias = t(take(np.int16, W), (W,))
        self.l1_w = t(take(np.int8, N_BUCKETS * H * (W // 2)), (N_BUCKETS, H, W // 2))
        self.l1_b = t(take(np.int32, N_BUCKETS * H), (N_BUCKETS, H))
        self.out_w = t(take(np.int16, N_BUCKETS * H), (N_BUCKETS, H))
        self.out_b = t(take(np.int32, N_BUCKETS), (N_BUCKETS,))
        self.feat = Features(device, False)  # exported rows are unfolded

    def __call__(self, idx_raw, bucket):
        """(oacc in 1/8192 discs, x, z, a) with x, z, a scaled to [0, 1] like the float trunk"""
        acc8 = self.ft[idx_raw].to(torch.int64).sum(1)
        acc = torch.clamp(torch.clamp(acc8, -4095, 4095) * (1 << self.shift) + self.ft_bias, -32768, 32767)
        a = torch.clamp(acc, 0, 512)
        h = self.W // 2
        x = torch.clamp((a[:, :h] * a[:, h:]) >> 11, max=127)
        # float64 is exact here (|zacc| < 2^53) and CUDA has no integer matmul
        zacc = torch.einsum("nk,nhk->nh", x.double(), self.l1_w[bucket].double()).round().long() + self.l1_b[bucket]
        z = torch.clamp(zacc >> 6, 0, 127)
        oacc = (z * self.out_w[bucket]).sum(1) + self.out_b[bucket]
        return oacc, (x.float() / 128.0, z.float() / 128.0, a.float() / 512.0)


class UHead(nn.Module):
    def __init__(self, n_in, hidden, private=0):
        super().__init__()
        self.n_in, self.hidden, self.n_private = n_in, hidden, private
        if private:
            self.private = nn.Embedding(pat.N_ROWS, private)
            nn.init.uniform_(self.private.weight, -0.02, 0.02)
            self.private_bias = nn.Parameter(torch.full((private,), 0.25))
            self.register_buffer("fold", torch.tensor(pat.fold_rows(), dtype=torch.long))
        # hidden 0: one linear output per bucket on the inputs (with inputs z: the last-layer split of a dual network)
        self.l1 = nn.Linear(n_in + private, hidden * N_BUCKETS) if hidden else None
        self.out = nn.Linear(hidden or (n_in + private), N_BUCKETS)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f, bucket, idx=None):
        """idx: the 64 row indices (raw or folded) of the positions, for the private lanes"""
        n = f.shape[0]
        if self.n_private:
            q = lambda w: w + (torch.round(w * 64.0) / 64.0 - w).detach()
            rows = torch.nn.functional.embedding(self.fold[idx], q(self.private.weight))
            pa = torch.clamp(rows.sum(1) + q(self.private_bias), 0.0, 1.0)
            f = torch.cat([f, pa], 1)
        ar = torch.arange(n, device=f.device)
        h = torch.clamp(self.l1(f).view(n, N_BUCKETS, self.hidden)[ar, bucket], 0.0, 1.0) if self.l1 is not None else f
        return self.out(h)[ar, bucket] * 16.0

    def clip_for_quantization(self):
        if self.n_private:
            with torch.no_grad():
                self.private.weight.clamp_(-127 / 64, 127 / 64)


N_IN = {"x": lambda m: m.width // 2, "xz": lambda m: m.width // 2 + m.hidden, "axz": lambda m: m.width + m.width // 2 + m.hidden,
        "z": lambda m: m.hidden}


def head_inputs(t, inputs):
    """t = trunk(...)"""
    x, z, a = t
    return {"x": lambda: x, "xz": lambda: torch.cat([x, z], 1), "axz": lambda: torch.cat([a, x, z], 1), "z": lambda: z}[inputs]()


def load_uhead(run, device):
    """(trunk function (idx, bucket) -> trunk tuple, features for idx, head, inputs) of a run of this script"""
    args = json.loads((Path(run) / "args.json").read_text(encoding="utf-8"))
    main_model, feat, _ = load_run(args["main"], device, args.get("main_model", "best.pt"))
    if args.get("int_trunk"):
        it = IntTrunk(args["int_trunk"], device)
        trunk_fn, feat = (lambda idx, bucket: it(idx, bucket)[1]), it.feat
    else:
        trunk_fn = lambda idx, bucket: trunk(main_model, idx, bucket)
    head = UHead(N_IN[args["inputs"]](main_model), args["hidden"], args.get("private", 0)).to(device)
    head.load_state_dict(torch.load(Path(run) / "best.pt", map_location=device))
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return trunk_fn, feat, head, args["inputs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--main", required=True, help="run directory of the trained NNUE (the engine's weights)")
    ap.add_argument("--main-model", default="best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default="x", choices=list(N_IN))
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--private", type=int, default=0, help="private first-layer lanes of the head")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-final", type=float, default=2e-5)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--train-size", type=int, default=100000000, help="training records loaded (a random contiguous-block sample)")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--empties-ref", default="")
    ap.add_argument("--compare", default="", help="run directory of a separate uncertainty network to evaluate alongside")
    ap.add_argument("--distill", type=float, default=0.0, help="weight a of (u - u_compare)^2 (needs --compare): loss = (1 - a) NLL + a distillation")
    ap.add_argument("--int-trunk", default="", help="the engine weight file (.egnn, int8) exported from --main")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    main_model, feat, _ = load_run(args.main, device, args.main_model)
    int_trunk = IntTrunk(args.int_trunk, device) if args.int_trunk else None
    tr_a, tr_p = load_split(args.data, "train_tr")
    va_a, va_p = load_split(args.data, "val_tr")
    # a random sample of 1M-record blocks keeps the load sequential
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
    print(f"train records {n_tr}, val {val['s'].shape[0]}", flush=True)
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
    head = UHead(N_IN[args.inputs](main_model), args.hidden, args.private).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    def residual_and_u(b):
        P = b["P"].to(device, non_blocking=True)
        with torch.no_grad():
            if int_trunk is not None:
                idx, bucket, n_player, e = batch_inputs(int_trunk.feat, b, device, False)
                oacc, t = int_trunk(idx, bucket)
                y_hat = P + oacc.float() / 8192.0
            else:
                idx, bucket, n_player, e = batch_inputs(feat, b, device, False)
                y_hat, _ = main_model(idx, bucket, n_player, P)
                t = trunk(main_model, idx, bucket)
        y = b["s"].to(device, non_blocking=True).float()
        return y - y_hat, head(head_inputs(t, args.inputs), bucket, idx), e, idx, bucket

    cmp_net = None
    if args.compare:
        ctargs = json.loads((Path(args.compare) / "args.json").read_text(encoding="utf-8"))
        cmp_net = Net(ctargs["width"], ctargs["hidden"], "residual").to(device)
        cmp_net.load_state_dict(torch.load(Path(args.compare) / "best.pt", map_location=device))
        cmp_net.eval()
        cmp_feat = Features(device, True)

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
        res, u, _, idx_b, bucket_b = residual_and_u(b)
        loss = (u + res ** 2 * torch.exp(-u)).mean()
        if args.distill > 0:
            with torch.no_grad():
                cidx, cb, _, _ = batch_inputs(cmp_feat, b, device, False)
                cu = cmp_net.correction(cidx, cb)
            loss = (1 - args.distill) * loss + args.distill * ((u - cu) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        head.clip_for_quantization()
        if step % args.val_every == 0 or step == args.steps:
            head.eval()
            with torch.no_grad():
                tot, tot_c, n, rows = 0.0, 0.0, 0, []
                for i in range(0, val["s"].shape[0], 65536):
                    vb = {k: v[i:i + 65536] for k, v in val.items()}
                    r, uu, e, _, bucket = residual_and_u(vb)
                    tot += float((uu + r ** 2 * torch.exp(-uu)).sum())
                    if cmp_net is not None:
                        cidx, cb, _, _ = batch_inputs(cmp_feat, vb, device, False)
                        cu = cmp_net.correction(cidx, cb)
                        tot_c += float((cu + r ** 2 * torch.exp(-cu)).sum())
                    n += r.shape[0]
                    rows.append((r.cpu().numpy(), uu.cpu().numpy()))
            head.train()
            r = np.concatenate([x[0] for x in rows]); uu = np.concatenate([x[1] for x in rows])
            const = float(np.log(np.mean(r ** 2)) + 1.0)  # NLL of a constant variance
            q = np.quantile(uu, [0.2, 0.8])
            lo, hi = r[uu <= q[0]], r[uu >= q[1]]
            rec = {"step": step, "loss": loss.item(), "val_nll": tot / n, "val_nll_const": const,
                   "sd_low20": float(lo.std()), "sd_high20": float(hi.std()), "elapsed": time.time() - t0}
            if cmp_net is not None:
                rec["val_nll_compare"] = tot_c / n
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step} loss {loss.item():.3f} | val nll {tot / n:.4f} (constant {const:.4f}"
                  + (f", separate net {tot_c / n:.4f}" if cmp_net is not None else "")
                  + f") | residual sd low20 {lo.std():.2f} high20 {hi.std():.2f} | {time.time() - t0:.0f}s", flush=True)
            torch.save(head.state_dict(), out / "last.pt")
            if tot / n < best:
                best = tot / n
                torch.save(head.state_dict(), out / "best.pt")
    log.close()


if __name__ == "__main__":
    main()
