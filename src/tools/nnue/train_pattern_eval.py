"""Train a 60-phase pattern evaluation (eval.egev2 format) on the NNUE training records.

Model, one independent table set per phase (phase = 60 - empties, as in the engine):
    value = sum over the instances of table[row(instance)] + num[number of discs of the side to move]
Pattern sets:
    nnue12 : the 12 NNUE shapes, 64 instances (src/tools/nnue/nnue_patterns.py); rows are the NNUE rows
             (digit 0 = empty, 1 = side to move, 2 = opponent, digit k of an instance weighs 3^k)
    prod16 : the 16 production shapes, 64 features (feature_to_coord in src/engine/evaluate_simd.hpp)
Rows that a pattern's own symmetry exchanges share one weight (--no-fold to disable), so the evaluation is
exactly symmetric. Phases are trained from the last one (1 empty) backwards, each starting from the weights
of the phase after it. Empties counts without transcript data (> --max-data-empties) copy the last trained
phase.

Output (in --out): eval.egev2 (the engine format: per phase the pattern rows then 65 disc-count values, 1 disc
= 32, run-length coded zeros), weights.npz (float weights in training rows), summary.json (validation MSE).

Usage:
    python src/tools/nnue/train_pattern_eval.py --patterns nnue12 --data ignored/nnue_data3 --out <dir>
    --subset after_random: keep a transcript position only when at least min(16, phase - 13) moves were
    played after the random opening (the rule of src/tools/evaluation/eval_optimizer_phase.py)
    --mo-end a,b,c,d: instead train the endgame move-ordering evaluation (eval_move_ordering_end.egev
    format, one phase, no disc-count term) on positions with 10-12 empties over the named nnue12 patterns
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

import nnue_patterns as pat

ROOT = Path(__file__).resolve().parents[3]
REC = np.dtype([("p", "<u8"), ("o", "<u8"), ("s", "i1"), ("e", "u1"), ("src", "u1"), ("r", "u1")])
N_PHASES = 60
STEP = 32
EVAL_MAX = 4091
N_ZEROS_PLUS = 1 << 12
N_MAX_ZEROS = 28600


def cell_of_name(name):
    return (ord(name[0]) - ord("A"), int(name[1]) - 1)


def prod16_instances():
    """(cells per feature, pattern id per feature) from the engine's feature_to_coord table"""
    src = (ROOT / "src/engine/evaluate_simd.hpp").read_text(encoding="utf-8")
    block = src[src.index("constexpr Feature_to_coord feature_to_coord"):]
    block = block[:block.index("};")]
    feats = []
    for m in re.finditer(r"\{(\d+), \{([^}]*)\}\}", block):
        n = int(m.group(1))
        names = [t.strip()[len("COORD_"):] for t in m.group(2).split(",")]
        feats.append([cell_of_name(c) for c in names[:n]])
    assert len(feats) == 64, len(feats)
    return feats, [i // 4 for i in range(64)]


class PatternSet:
    """instances (cell lists), their pattern ids and the row layout; rows use the NNUE digit convention"""

    def __init__(self, name, only=None):
        self.name = name
        if name == "nnue12":
            inst = [(i["cells"], i["pattern"]) for i in pat.INSTANCES]
            self.pattern_names = [p["name"] for p in pat.PATTERNS]
        else:
            cells, pids = prod16_instances()
            inst = list(zip(cells, pids))
            self.pattern_names = [f"p{k}" for k in range(16)]
        if only is not None:
            keep = [self.pattern_names.index(n) for n in only]
            inst = [(c, keep.index(p)) for c, p in inst if p in keep]
            self.pattern_names = [self.pattern_names[k] for k in keep]
        self.inst_cells = [c for c, _ in inst]
        self.inst_pattern = [p for _, p in inst]
        n_pat = len(self.pattern_names)
        self.sizes = []
        for k in range(n_pat):
            first = next(c for c, p in inst if p == k)
            self.sizes.append(len(first))
        self.offsets = [int(v) for v in np.concatenate([[0], np.cumsum([3 ** s for s in self.sizes])[:-1]])]
        self.n_rows = int(sum(3 ** s for s in self.sizes))
        self.max_cells = max(self.sizes)

    def base_cells(self, k):
        return next(c for c, p in zip(self.inst_cells, self.inst_pattern) if p == k)

    def fold_rows(self):
        """row -> smallest row that one of the pattern's own symmetries maps it to"""
        fold = np.arange(self.n_rows)
        for k, n in enumerate(self.sizes):
            cs = self.base_cells(k)
            perms = []
            for f in pat.SYMMETRIES:
                img = [f(*c) for c in cs]
                if set(img) == set(cs):
                    perms.append([cs.index(c) for c in img])
            local = np.arange(3 ** n)
            digits = [(local // 3 ** j) % 3 for j in range(n)]
            best = local.copy()
            for perm in perms:
                v = np.zeros_like(local)
                for j in range(n):
                    v += digits[j] * 3 ** perm[j]
                best = np.minimum(best, v)
            fold[self.offsets[k]:self.offsets[k] + 3 ** n] = self.offsets[k] + best
        return fold

    def engine_order(self):
        """engine row of each training row: nnue12 keeps the NNUE rows; prod16 rows follow the production
        index (digit 0 = side to move, 1 = opponent, 2 = empty, the first cell is the highest digit)"""
        if self.name == "nnue12":
            return np.arange(self.n_rows)
        order = np.zeros(self.n_rows, dtype=np.int64)
        for k, n in enumerate(self.sizes):
            local = np.arange(3 ** n)
            prod = np.zeros_like(local)
            for j in range(n):
                dn = (local // 3 ** j) % 3          # NNUE digit of cell j
                dp = (dn + 2) % 3                   # 0 empty -> 2, 1 own -> 0, 2 opponent -> 1
                prod += dp * 3 ** (n - 1 - j)
            order[self.offsets[k] + local] = self.offsets[k] + prod
        return order


class GpuFeatures:
    def __init__(self, ps, device, fold=True):
        n_inst = len(ps.inst_cells)
        m = ps.max_cells
        cells = torch.full((n_inst, m), 64, dtype=torch.long)
        pow3 = torch.zeros((n_inst, m), dtype=torch.long)
        offs = torch.zeros(n_inst, dtype=torch.long)
        for i, cs in enumerate(ps.inst_cells):
            for j, (x, y) in enumerate(cs):
                cells[i, j] = 8 * y + x
                pow3[i, j] = 3 ** j
            offs[i] = ps.offsets[ps.inst_pattern[i]]
        self.cells = cells.to(device).view(-1)
        self.pow3 = pow3.to(device)
        self.offs = offs.to(device)
        self.m = m
        self.shifts = (63 - torch.arange(64, device=device)).view(1, 64)
        self.fold = torch.from_numpy(ps.fold_rows()).to(device) if fold else None

    def __call__(self, p, o):
        st = ((p.view(-1, 1) >> self.shifts) & 1) + 2 * ((o.view(-1, 1) >> self.shifts) & 1)
        ext = torch.cat([st, torch.zeros((st.shape[0], 1), dtype=st.dtype, device=st.device)], 1)
        g = ext[:, self.cells].view(st.shape[0], -1, self.m)
        idx = (g * self.pow3).sum(-1) + self.offs
        if self.fold is not None:
            idx = self.fold[idx]
        n_player = (st == 1).sum(1)
        return idx, n_player


def to_gpu(a, sel, device):
    return {
        "p": torch.from_numpy(np.ascontiguousarray(a["p"][sel]).view(np.int64)).to(device),
        "o": torch.from_numpy(np.ascontiguousarray(a["o"][sel]).view(np.int64)).to(device),
        "y": torch.from_numpy(np.ascontiguousarray(a["s"][sel]).astype(np.float32)).to(device),
    }


def cat(parts):
    parts = [q for q in parts if q is not None and q["y"].shape[0]]
    if not parts:
        return None
    return {k: torch.cat([q[k] for q in parts]) for k in parts[0]}


def keep_after_random(a, e):
    """--subset after_random: moves after the random opening >= min(16, phase - 13)"""
    phase = 60 - e
    need = min(16, max(0, phase - 13))
    played = 60 - e - np.asarray(a["r"]).astype(np.int64)
    return played >= need


class Data:
    def __init__(self, args, device):
        d = Path(args.data)
        self.device = device
        self.by_e = d / "by_e"
        bd = np.memmap(d / "shuffled/train_boards.bin", dtype=REC, mode="r")
        ids = np.arange(len(bd))
        hold = ids % 100 == 37  # the board holdout of train_nnue.py
        e_bd = np.asarray(bd["e"])
        self.bd, self.bd_train_e, self.bd_val_e = bd, e_bd * ~hold, e_bd * hold
        self.bd_hold = hold
        self.bd_e = e_bd
        va = np.memmap(d / "shuffled/val_tr.bin", dtype=REC, mode="r")
        vsel = np.sort(np.random.default_rng(args.val_seed).permutation(len(va))[: args.val_size])
        self.va = va
        self.vsel = vsel
        self.va_e = np.asarray(va["e"][vsel])
        self.va_p = np.asarray(np.memmap(d / "shuffled/val_tr.p.f32", dtype=np.float32, mode="r")[vsel])
        self.bd_p = np.asarray(np.memmap(d / "shuffled/train_boards.p.f32", dtype=np.float32, mode="r"))
        self.subset = args.subset

    def train(self, e, max_records=0, rng=None):
        parts = []
        f = self.by_e / f"e{e:02d}.bin"
        if f.exists():
            a = np.memmap(f, dtype=REC, mode="r")
            n = len(a) if not max_records else min(len(a), max_records)
            sel = np.arange(n)
            if self.subset == "after_random":
                sel = sel[keep_after_random(a[:n], e)]
            parts.append(to_gpu(a, sel, self.device))
        bsel = np.nonzero((self.bd_e == e) & ~self.bd_hold)[0]
        if len(bsel):
            parts.append(to_gpu(self.bd, bsel, self.device))
        return cat(parts)

    def val(self, e):
        vs = self.vsel[self.va_e == e]
        bsel = np.nonzero((self.bd_e == e) & self.bd_hold)[0]
        tr = to_gpu(self.va, vs, self.device) if len(vs) else None
        if tr is not None:
            tr["P"] = torch.from_numpy(self.va_p[self.va_e == e]).to(self.device)
        bd = to_gpu(self.bd, bsel, self.device) if len(bsel) else None
        if bd is not None:
            bd["P"] = torch.from_numpy(self.bd_p[bsel]).to(self.device)
        return tr, bd


def predict(table, num, feat, b, bs=262144, quantized=False):
    out = []
    for i in range(0, b["y"].shape[0], bs):
        idx, n_player = feat(b["p"][i:i + bs], b["o"][i:i + bs])
        if quantized:
            v = table[idx].sum(1) + num[n_player]
            out.append(v.float() / STEP)
        else:
            out.append(table[idx].sum(1) + (num[n_player] if num is not None else 0.0))
    return torch.cat(out)


def mse(pred, y):
    return float(((pred - y) ** 2).mean())


def train_phase(table, num, feat, tr, va, args, n_rows, log):
    """Adam over minibatches, cosine learning rate; returns the weights with the best validation MSE"""
    n = tr["y"].shape[0]
    steps = max(args.min_steps, int(math.ceil(n * args.epochs / args.batch)))
    w = torch.nn.Parameter(table.clone())
    params = [w]
    nm = None
    if num is not None:
        nm = torch.nn.Parameter(num.clone())
        params.append(nm)
    opt = torch.optim.Adam(params, lr=args.lr)
    best = (math.inf, w.detach().clone(), nm.detach().clone() if nm is not None else None)
    val_every = max(1, steps // args.n_val)
    for step in range(1, steps + 1):
        frac = (step - 1) / max(steps - 1, 1)
        lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        sel = torch.randint(0, n, (args.batch,), device=tr["y"].device)
        idx, n_player = feat(tr["p"][sel], tr["o"][sel])
        pred = w[idx].sum(1) + (nm[n_player] if nm is not None else 0.0)
        loss = ((pred - tr["y"][sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if va is not None and (step % val_every == 0 or step == steps):
            with torch.no_grad():
                v = mse(predict(w.detach(), nm.detach() if nm is not None else None, feat, va), va["y"])
            if v < best[0]:
                best = (v, w.detach().clone(), nm.detach().clone() if nm is not None else None)
    if va is None:
        best = (math.nan, w.detach().clone(), nm.detach().clone() if nm is not None else None)
    log["steps"] = steps
    return best


def quantize(table, num):
    qt = torch.clamp(torch.round(table * STEP), -EVAL_MAX, EVAL_MAX).to(torch.int32)
    qn = torch.round(num * STEP).to(torch.int32) if num is not None else None
    return qt, qn


def write_egev2(path, phase_rows, phase_num):
    """phase_rows: per phase int16 rows in engine order; phase_num: per phase 65 int16 values"""
    flat = np.concatenate([np.concatenate([np.asarray(rows), np.asarray(num)]) for rows, num in zip(phase_rows, phase_num)])
    # values >= N_ZEROS_PLUS would read as runs of zeros, and int16 must not wrap
    if flat.min() < -32768 or flat.max() >= N_ZEROS_PLUS:
        raise ValueError(f"egev2 values out of range: {flat.min()} .. {flat.max()} (allowed -32768 .. {N_ZEROS_PLUS - 1})")
    flat = flat.astype(np.int16)
    out = []
    zeros = 0
    for v in flat.tolist():
        if v == 0:
            zeros += 1
            if zeros == N_MAX_ZEROS:
                out.append(zeros + N_ZEROS_PLUS)
                zeros = 0
        else:
            if zeros:
                out.append(zeros + N_ZEROS_PLUS)
                zeros = 0
            out.append(v)
    if zeros:
        out.append(zeros + N_ZEROS_PLUS)
    arr = np.array(out, dtype=np.int16)
    with open(path, "wb") as f:
        f.write(np.array([len(arr)], dtype=np.int32).tobytes())
        f.write(arr.tobytes())


def run_main(args, device):
    ps = PatternSet(args.patterns)
    feat = GpuFeatures(ps, device, not args.no_fold)
    data = Data(args, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    table = torch.zeros(ps.n_rows, device=device)
    num = torch.zeros(65, device=device)
    tables, nums = {}, {}
    summary = {"phases": {}, "patterns": args.patterns, "rows": ps.n_rows}
    t0 = time.time()
    only = [int(x) for x in args.only_empties.split(",")] if args.only_empties else None
    for e in (only or range(1, 61)):
        phase = 60 - e
        if only:
            # training-budget check: each listed phase from zero weights, no egev2 written
            table = torch.zeros(ps.n_rows, device=device)
            num = torch.zeros(65, device=device)
        if e > args.max_data_empties:
            tables[phase], nums[phase] = table.clone(), num.clone()
            continue
        tr = data.train(e)
        va_tr, va_bd = data.val(e)
        va = cat([va_tr, va_bd])
        log = {"n_train": int(tr["y"].shape[0]) if tr is not None else 0}
        if tr is None:
            tables[phase], nums[phase] = table.clone(), num.clone()
            continue
        v, table, num = train_phase(table, num, feat, tr, va, args, ps.n_rows, log)
        tables[phase], nums[phase] = table.clone(), num.clone()
        qt, qn = quantize(table, num)
        with torch.no_grad():
            for name, vb in (("val_tr", va_tr), ("val_boards", va_bd)):
                if vb is None:
                    continue
                pq = predict(qt, qn, feat, vb, quantized=True)
                log[name] = {"n": int(vb["y"].shape[0]), "model": mse(pq, vb["y"]), "P": mse(vb["P"], vb["y"])}
        summary["phases"][phase] = log
        msg = f"phase {phase} (empties {e}): train {log['n_train']} steps {log['steps']}"
        for name in ("val_tr", "val_boards"):
            if name in log:
                msg += f" | {name} model {log[name]['model']:.2f} P {log[name]['P']:.2f}"
        print(msg + f" | {time.time() - t0:.0f}s", flush=True)
        del tr, va, va_tr, va_bd
        torch.cuda.empty_cache()
    if only:
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return
    # totals over the validation transcripts and boards
    for name in ("val_tr", "val_boards"):
        tot_m = tot_p = cnt = 0.0
        band = {}
        for phase, log in summary["phases"].items():
            if name not in log:
                continue
            n = log[name]["n"]
            tot_m += log[name]["model"] * n
            tot_p += log[name]["P"] * n
            cnt += n
            k = (60 - int(phase)) // 10
            b = band.setdefault(k, [0.0, 0.0, 0])
            b[0] += log[name]["model"] * n
            b[1] += log[name]["P"] * n
            b[2] += n
        if cnt:
            summary[name] = {"model": tot_m / cnt, "P": tot_p / cnt, "n": cnt,
                             "bands": {f"{10 * k:02d}-{10 * k + 9:02d}": {"model": b[0] / b[2], "P": b[1] / b[2], "n": b[2]}
                                       for k, b in sorted(band.items())}}
            print(f"{name}: model {tot_m / cnt:.3f} P {tot_p / cnt:.3f} ({int(cnt)} positions)", flush=True)
    order = ps.engine_order()
    rows, nm = [], []
    w_all = np.zeros((N_PHASES, ps.n_rows), dtype=np.float32)
    n_all = np.zeros((N_PHASES, 65), dtype=np.float32)
    fold = ps.fold_rows() if not args.no_fold else np.arange(ps.n_rows)
    for phase in range(N_PHASES):
        qt, qn = quantize(tables[phase], nums[phase])
        q = qt.cpu().numpy()[fold]  # every row gets its folded weight
        eng = np.zeros(ps.n_rows, dtype=np.int32)
        eng[order] = q
        rows.append(eng)
        nm.append(qn.cpu().numpy())
        w_all[phase] = tables[phase].cpu().numpy()[fold]
        n_all[phase] = nums[phase].cpu().numpy()
    write_egev2(out / "eval.egev2", rows, nm)
    np.savez_compressed(out / "weights.npz", table=w_all, num=n_all)
    summary["elapsed"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {out / 'eval.egev2'} in {time.time() - t0:.0f}s", flush=True)


def run_mo_end(args, device):
    names = args.mo_end.split(",")
    ps = PatternSet("nnue12", only=names)
    feat = GpuFeatures(ps, device, not args.no_fold)
    data = Data(args, device)
    parts, vparts = [], []
    for e in (10, 11, 12):
        parts.append(data.train(e, args.mo_end_records))
        vt, vb = data.val(e)
        vparts += [vt, vb]
    tr = cat(parts)
    va = cat(vparts)
    log = {"n_train": int(tr["y"].shape[0])}
    table = torch.zeros(ps.n_rows, device=device)
    v, table, _ = train_phase(table, None, feat, tr, va, args, ps.n_rows, log)
    qt = torch.clamp(torch.round(table * STEP), -16380, 16380).to(torch.int32)
    with torch.no_grad():
        pq = predict(qt, torch.zeros(65, dtype=torch.int32, device=device), feat, va, quantized=True)
    res = {"patterns": names, "n_train": log["n_train"], "steps": log["steps"], "val_mse": mse(pq, va["y"]),
           "val_mse_float": v, "n_val": int(va["y"].shape[0])}
    print(json.dumps(res), flush=True)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        fold = ps.fold_rows() if not args.no_fold else np.arange(ps.n_rows)
        q = qt.cpu().numpy()[fold].astype(np.int16)
        q.tofile(out / "eval_move_ordering_end.egev")
        (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="nnue12", choices=["nnue12", "prod16"])
    ap.add_argument("--data", default="ignored/nnue_data3")
    ap.add_argument("--out", default="")
    ap.add_argument("--subset", default="", choices=["", "after_random"])
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--min-steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr-final", type=float, default=0.0002)
    ap.add_argument("--n-val", type=int, default=8, help="validations per phase (the best one is kept)")
    ap.add_argument("--max-data-empties", type=int, default=52, help="larger empties counts copy this phase")
    ap.add_argument("--val-size", type=int, default=1000000)
    ap.add_argument("--val-seed", type=int, default=20261019)
    ap.add_argument("--no-fold", action="store_true")
    ap.add_argument("--only-empties", default="", help="comma-separated empties counts: train only these phases from zero (no egev2)")
    ap.add_argument("--mo-end", default="", help="comma-separated nnue12 pattern names: train the endgame ordering table")
    ap.add_argument("--mo-end-records", type=int, default=0, help="records per empties count for --mo-end (0: all)")
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if args.mo_end:
        run_mo_end(args, device)
    else:
        run_main(args, device)


if __name__ == "__main__":
    main()
