"""Train an MPC error model: for a position x and an MPC test (deep depth D, shallow depth s; s = 0 is
the static evaluation; midgame or endgame), predict the mean mu and the scale sigma of
    e = deep value - shallow value
so that MPC margins come from the model alone (no depth / disc-count sigma formula):
    high cut: shallow >= beta + ceil(z * sigma - mu),   low cut: shallow <= alpha - ceil(z * sigma + mu).

Model: u(x) is the output of the pattern network trained by train_uncertainty.py (frozen, or with its
heads fine-tuned with --finetune), or with --dual the output of the uncertainty head on the main NNUE's
first layer (train_uncertainty_head.py; the dual network). --x-terms adds linear terms in the main
network's first-layer features f (the head's inputs) to mu and log sigma: + a_g . f, with an L2 penalty.
--mean-u-below K follows the engine: MPC nodes shallower than K use the mean u for the disc count instead of
the network (EGAROUCID_NNUE_MPC_NET_MIN_DEPTH). --apply-head <head.json> fits nothing and writes the predictions
of a given model (e.g. the engine's current coefficients) for these rows. Four groups (mid static, mid shallow, end static, end shallow) each have
    mu        = w_mu . [1, parity, n', u, v/32]
    log sigma = w_sigma . [1, n', s', gap', u, u^2, |v|/32]
(n' = (discs - 28) / 14, s' = min(s, S_MAX) / 13, gap' = min(D - s, GAP_MAX) / 13 with the group's largest
values in the data, so deeper searches keep the edge value instead of extrapolating; parity = D mod 2;
v = the static NNUE value).
Loss: Gaussian negative log likelihood. A baseline without u and v (f = [1, n', s', gap', parity]) is
trained alongside to show what the network adds. Boards are split into training and validation sets
by a hash of the board, so all depths of a board stay on one side.

Data: TSVs of mid_probcut_dataset_tool (midgame: ignored/mpc_data/<tag>_d*.tsv; endgame:
ignored/mpc_end_data/end_<tag>_e*.tsv), only files with a .done marker.

Usage: python src/tools/nnue/train_mpc_error.py --mid ignored/mpc_data/d3_d*.tsv ignored/mpc_data/d3x_d*.tsv \
           --end ignored/mpc_end_data/end_d3_e*.tsv --unc ignored/nnue_runs/uncertainty_w32 --out ignored/nnue_runs/mpcerr_b1
"""
import argparse
import glob
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nnue import Features, Net, bucket_of  # noqa: E402
from train_uncertainty_head import head_inputs, load_uhead  # noqa: E402

GROUPS = ["mid_static", "mid_shallow", "end_static", "end_shallow"]


def read_rows(patterns, is_end):
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            if not Path(f).with_suffix(".done").exists():
                continue
            for line in Path(f).read_text(encoding="utf-8").splitlines()[1:]:
                c = line.split("\t")
                if len(c) < 15:
                    continue
                board, n_discs, D, s, static, shallow, deep = c[1], int(c[2]), int(c[3]), int(c[4]), int(c[5]), int(c[6]), int(c[7])
                rows.append((board, n_discs, D, s, static, deep - shallow, is_end, int(c[13]), int(c[14])))
    return rows


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


def features(u, n, D, s, v, parity, s_max, gap_max, full=True, u2=None):
    """(mu features, log sigma features); s_max / gap_max are per-row clamps (the group's data range);
    u2: a second u (diagnostic: does it add to u?)"""
    one = torch.ones_like(n)
    nn_ = (n - 28.0) / 14.0
    ss = torch.minimum(s, s_max) / 13.0
    gg = torch.minimum(D - s, gap_max) / 13.0
    if not full:
        return torch.stack([one, parity, nn_], 1), torch.stack([one, nn_, ss, gg], 1)
    fm = [one, parity, nn_, u, v / 32.0]
    fs = [one, nn_, ss, gg, u, u * u, v.abs() / 32.0]
    if u2 is not None:
        fm += [u2]
        fs += [u2, u2 * u2, u * u2]
    return torch.stack(fm, 1), torch.stack(fs, 1)


class Head(torch.nn.Module):
    def __init__(self, n_mu, n_ls, n_x=0):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(len(GROUPS), n_mu))
        self.ls = torch.nn.Parameter(torch.zeros(len(GROUPS), n_ls))
        self.n_x = n_x
        if n_x:
            self.mx = torch.nn.Parameter(torch.zeros(len(GROUPS), n_x))
            self.lx = torch.nn.Parameter(torch.zeros(len(GROUPS), n_x))
        with torch.no_grad():
            self.ls[:, 0] = math.log(3.0)

    def forward(self, f, g, fx=None):
        fm, fs = f
        mu, ls = (fm * self.mu[g]).sum(1), (fs * self.ls[g]).sum(1)
        if self.n_x:
            mu = mu + (fx * self.mx[g]).sum(1)
            ls = ls + (fx * self.lx[g]).sum(1)
        return mu, ls

    def penalty(self):
        return (self.mx ** 2).sum() + (self.lx ** 2).sum() if self.n_x else 0.0


def nll(e, mu, ls):
    return (ls + 0.5 * ((e - mu) * torch.exp(-ls)) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid", nargs="*", default=[])
    ap.add_argument("--end", nargs="*", default=[])
    ap.add_argument("--unc", default="", help="run directory of the uncertainty network")
    ap.add_argument("--dual", default="", help="run directory of train_uncertainty_head.py (u from the main network's first layer)")
    ap.add_argument("--x-terms", action="store_true", help="linear terms in the main network's first-layer features (needs --dual)")
    ap.add_argument("--x-l2", type=float, default=1e-3, help="L2 penalty of the x terms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--finetune", action="store_true", help="also train the uncertainty network's heads (first layer frozen)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr-net", type=float, default=2e-4)
    ap.add_argument("--val-mod", type=int, default=5)
    ap.add_argument("--no-u", action="store_true", help="ablation: replace u by 0 (cheap features only)")
    ap.add_argument("--mean-u-below", type=int, default=0, help="rows with a deep depth below this use the mean u for the disc count")
    ap.add_argument("--extra", default="", help="comma-separated extra features of mu and log sigma: dpv (|static - pattern value| / 32), legal (legal moves / 10)")
    ap.add_argument("--apply-head", default="", help="head.json of a fitted model: no fitting, its predictions for these rows")
    args = ap.parse_args()
    both_u = bool(args.unc) and bool(args.dual)
    if not (args.unc or args.dual) or (args.x_terms and not args.dual) or (args.finetune and args.dual):
        ap.error("give --unc and / or --dual; --x-terms needs --dual; --finetune needs --unc only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.mid, False) + read_rows(args.end, True)
    print(f"rows {len(rows)} (mid {sum(1 for r in rows if not r[6])}, end {sum(1 for r in rows if r[6])})", flush=True)
    boards = sorted({r[0] for r in rows})
    bidx = {b: i for i, b in enumerate(boards)}
    bits = np.array([board_bits(b) for b in boards], dtype=np.uint64)
    is_val_board = np.array([int(hashlib.md5(b.encode()).hexdigest(), 16) % args.val_mod == 0 for b in boards])
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    rb = t([bidx[r[0]] for r in rows], torch.long)
    n = t([r[1] for r in rows]); D = t([r[2] for r in rows]); s = t([r[3] for r in rows])
    v = t([r[4] for r in rows]); e = t([r[5] for r in rows])
    extra_cols = []
    for name in [x for x in args.extra.split(",") if x]:
        if name == "dpv":
            extra_cols.append(t([abs(r[4] - r[8]) / 32.0 for r in rows]))
        elif name == "legal":
            extra_cols.append(t([r[7] / 10.0 for r in rows]))
        else:
            ap.error(f"unknown extra feature {name}")
    fx_extra = torch.stack(extra_cols, 1) if extra_cols else None
    parity = D.remainder(2)
    g = t([(2 if r[6] else 0) + (0 if r[3] == 0 else 1) for r in rows], torch.long)
    val = t(is_val_board[[bidx[r[0]] for r in rows]], torch.bool)
    net = None
    if args.unc:
        targs = json.loads((Path(args.unc) / "args.json").read_text(encoding="utf-8"))
        net = Net(targs["width"], targs["hidden"], "residual").to(device)
        net.load_state_dict(torch.load(Path(args.unc) / "best.pt", map_location=device))
        feat = Features(device, True)
    if args.dual:
        trunk_fn, dfeat, uhead, inputs = load_uhead(args.dual, device)
    p_all = torch.from_numpy(bits[:, 0].view(np.int64)).to(device)
    o_all = torch.from_numpy(bits[:, 1].view(np.int64)).to(device)
    base_feat = feat if args.unc else dfeat
    st = base_feat.states(p_all, o_all)
    idx_all = base_feat.indices(st)
    empt_all = 64 - (st != 0).sum(1)
    bucket_all = bucket_of(empt_all)
    fx_all = None
    if args.dual:
        with torch.no_grad():
            didx_all = dfeat.indices(st)
            f_all = head_inputs(trunk_fn(didx_all, bucket_all), inputs)
            u_dual = uhead(f_all, bucket_all, didx_all)
            if args.x_terms:
                fx_all = f_all
    if net is not None:
        for prm in net.parameters():
            prm.requires_grad_(False)
    net_params = []
    if args.finetune:
        for prm in list(net.l1.parameters()) + list(net.out.parameters()):
            prm.requires_grad_(True)
            net_params.append(prm)

    mean_u = None
    if args.mean_u_below > 0:
        if args.dual:
            table = json.loads((Path(args.dual) / "mean_u.json").read_text(encoding="utf-8"))["mean_u"]
        else:
            txt = (Path(__file__).resolve().parents[3] / "src" / "engine" / "nnue" / "nnue_uncertainty_table.hpp").read_text(encoding="utf-8")
            table = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", txt[txt.index("{"):txt.index("}")])]
        mean_u = torch.tensor(table, dtype=torch.float32, device=device)[n.long()]
    shallow_row = D < args.mean_u_below

    def row_u():
        # u per row: the network output for the board (log sigma^2 of the outcome residual, the engine's total / 8192),
        # or the disc-count mean where the engine skips the network
        u = (net.correction(idx_all, bucket_all) if args.unc else u_dual)[rb]
        if mean_u is not None:
            u = torch.where(shallow_row, mean_u, u)
        return u * 0.0 if args.no_u else u

    def row_u2():
        # with both: the head's u as extra features (no disc-count mean policy; a diagnostic)
        return u_dual[rb] if both_u else None

    # clamps: the largest shallow depth and depth gap of each group in the data
    s_max_g = torch.zeros(len(GROUPS), device=device)
    gap_max_g = torch.zeros(len(GROUPS), device=device)
    for gi in range(len(GROUPS)):
        m = g == gi
        if m.any():
            s_max_g[gi] = s[m].max()
            gap_max_g[gi] = (D[m] - s[m]).max()
    applied = json.loads(Path(args.apply_head).read_text(encoding="utf-8")) if args.apply_head else None
    if applied is not None:
        s_max_g = torch.tensor(applied["s_max"], device=device)
        gap_max_g = torch.tensor(applied["gap_max"], device=device)
        args.steps = 0
    s_max, gap_max = s_max_g[g], gap_max_g[g]
    fx = fx_all[rb] if fx_all is not None else None
    if fx_extra is not None:
        # extra features enter like the x terms (linear in mu and log sigma per group)
        fx = fx_extra if fx is None else torch.cat([fx, fx_extra], 1)
    head = Head(5 + (1 if both_u else 0), 7 + (3 if both_u else 0), fx.shape[1] if fx is not None else 0).to(device)
    if applied is not None:
        with torch.no_grad():
            head.mu.copy_(torch.tensor(applied["mu"], device=device))
            head.ls.copy_(torch.tensor(applied["log_sigma"], device=device))
    base_head = Head(3, 4).to(device)
    groups = [{"params": list(head.parameters()) + list(base_head.parameters()), "lr": args.lr}]
    if net_params:
        groups.append({"params": net_params, "lr": args.lr_net})
    opt = torch.optim.Adam(groups)
    tr = ~val
    t0 = time.time()
    for step in range(1, args.steps + 1):
        u = row_u()
        mu, ls = head(features(u, n, D, s, v, parity, s_max, gap_max, u2=row_u2()), g, fx)
        mu0, ls0 = base_head(features(u, n, D, s, v, parity, s_max, gap_max, full=False), g)
        loss = nll(e[tr], mu[tr], ls[tr]) + nll(e[tr], mu0[tr], ls0[tr]) + args.x_l2 * head.penalty()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if net_params:
            net.clip_for_quantization()
        if step % 500 == 0 or step == args.steps:
            with torch.no_grad():
                line = f"step {step}"
                for gi, name in enumerate(GROUPS):
                    m = val & (g == gi)
                    if m.sum() == 0:
                        continue
                    line += f" | {name} val nll {nll(e[m], mu[m], ls[m]).item():.4f} (base {nll(e[m], mu0[m], ls0[m]).item():.4f})"
            print(line + f" | {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        u = row_u()
        mu, ls = head(features(u, n, D, s, v, parity, s_max, gap_max, u2=row_u2()), g, fx)
        mu0, ls0 = base_head(features(u, n, D, s, v, parity, s_max, gap_max, full=False), g)
    res = {"groups": GROUPS, "mu_features": ["1", "parity", "n'", "u", "v/32"],
           "log_sigma_features": ["1", "n'", "s'", "gap'", "u", "u^2", "|v|/32"],
           "s_max": s_max_g.cpu().tolist(), "gap_max": gap_max_g.cpu().tolist(),
           "mu": head.mu.detach().cpu().tolist(), "log_sigma": head.ls.detach().cpu().tolist(),
           "base_mu_features": ["1", "parity", "n'"], "base_log_sigma_features": ["1", "n'", "s'", "gap'"],
           "base_mu": base_head.mu.detach().cpu().tolist(), "base_log_sigma": base_head.ls.detach().cpu().tolist(),
           "unc": str(args.unc), "dual": str(args.dual), "finetune": args.finetune}
    if head.n_x:
        res["x_mu"] = head.mx.detach().cpu().tolist()
        res["x_log_sigma"] = head.lx.detach().cpu().tolist()
    # standardized residual quantiles on validation (tail shape per group)
    for gi, name in enumerate(GROUPS):
        m = (val & (g == gi)).cpu().numpy()
        if m.sum() == 0:
            continue
        r = ((e - mu) * torch.exp(-ls)).cpu().numpy()[m]
        r0 = ((e - mu0) * torch.exp(-ls0)).cpu().numpy()[m]
        qs = [0.005, 0.02, 0.13, 0.87, 0.98, 0.995]
        res[f"val_quantiles_{name}"] = {"model": np.quantile(r, qs).tolist(), "base": np.quantile(r0, qs).tolist(), "q": qs}
        print(f"{name}: standardized residual quantiles {qs}: model {np.round(np.quantile(r, qs), 2).tolist()} base {np.round(np.quantile(r0, qs), 2).tolist()}")
    (out / "head.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(out / "predictions.npz", mu=mu.cpu().numpy(), ls=ls.cpu().numpy(), mu0=mu0.cpu().numpy(), ls0=ls0.cpu().numpy(),
             e=e.cpu().numpy(), g=g.cpu().numpy(), val=val.cpu().numpy(), n=n.cpu().numpy(), D=D.cpu().numpy(), s=s.cpu().numpy(),
             u=u.cpu().numpy(), v=v.cpu().numpy())
    with open(out / "rows.tsv", "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    if args.finetune:
        torch.save(net.state_dict(), out / "best.pt")
        (out / "args.json").write_text(json.dumps(dict(targs), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
