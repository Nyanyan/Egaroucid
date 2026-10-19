"""Build the GGS 読み切り (endgame search with MPC) mistake benchmark.

Steps (each caches its engine results under ignored/yomikiri, so a step can be re-run or resumed):
  sample  pick a pool of GGS 読み切り decisions from ignored/yomikiri/candidates.jsonl
  screen  solve the pool with a high MPC probability (cheap screen); flag positions where the GGS move
          differs from the screen move or the GGS value is off by --value-flag discs or more
  exact   solve flagged positions (plus a random control share) exactly (MPC 100%), and the child
          positions of the GGS move and the screen move when they differ from the exact best move
  write   write bin/problem/yomikiri_ggs.txt and bin/problem/yomikiri_ggs.json (answers: exact
          value and known exact move values)
  extend  solve the moves that bin/yomikiritest.py reported as UNKNOWN (its out= JSON files) exactly
          and add them to bin/problem/yomikiri_ggs.json
The candidates come from src/tools/yomikiri/collect_decisions.py.

Usage:
  python src/tools/yomikiri/build_benchmark.py sample --min-empties 28 --max-empties 32 --per-empties 200 --per-empties-99 100
  python src/tools/yomikiri/build_benchmark.py screen --exe bin/Egaroucid_nnue_rw64f_d2b_i8_lazy6.exe --prob 99.9 --procs 26 --value-flag 6
  python src/tools/yomikiri/build_benchmark.py exact --exe bin/Egaroucid_nnue_base.exe --procs 7 --threads 4 --control 0.05 --timeout 900 --max-exact-empties 31
  python src/tools/yomikiri/build_benchmark.py write
  python src/tools/yomikiri/build_benchmark.py extend --results ignored/yomikiri/run_a.json ...
(bin/problem/yomikiri_ggs.json of 2026-10-20 was made with these commands; the engine runs are cached in
ignored/yomikiri, so a step can be resumed.)
"""
import argparse
import json
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "ignored" / "yomikiri"


# ---------------------------------------------------------------- board helpers
DIRS = (1, -1, 8, -8, 9, -9, 7, -7)


def parse_board(s):
    """'<64 chars> X' -> (player bits, opponent bits, side char)"""
    cells, side = s.split()
    me, op = 0, 0
    for i, c in enumerate(cells):
        if c == side:
            me |= 1 << i
        elif c != "-":
            op |= 1 << i
    return me, op, side


def to_str(me, op, side):
    other = "O" if side == "X" else "X"
    return "".join(side if me >> i & 1 else other if op >> i & 1 else "-" for i in range(64)) + " " + side


def flips(me, op, pos):
    if (me | op) >> pos & 1:
        return 0
    x0, y0 = pos % 8, pos // 8
    total = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        x, y, f = x0 + dx, y0 + dy, 0
        while 0 <= x < 8 and 0 <= y < 8 and op >> (y * 8 + x) & 1:
            f |= 1 << (y * 8 + x)
            x += dx
            y += dy
        if f and 0 <= x < 8 and 0 <= y < 8 and me >> (y * 8 + x) & 1:
            total |= f
    return total


def legal_moves(me, op):
    return [p for p in range(64) if flips(me, op, p)]


def coord(pos):
    return "abcdefgh"[pos % 8] + str(pos // 8 + 1)


def pos_of(c):
    return (int(c[1]) - 1) * 8 + "abcdefgh".index(c[0])


def child(board, move):
    """position after <move> from the mover's view: (board string to solve, sign, terminal value or None).
    The move value for the mover is sign * value(child) (or the terminal value)."""
    me, op, side = parse_board(board)
    p = pos_of(move)
    f = flips(me, op, p)
    assert f, (board, move)
    me |= f | 1 << p
    op &= ~f
    other = "O" if side == "X" else "X"
    if legal_moves(op, me):
        return to_str(op, me, other), -1, None
    if legal_moves(me, op):
        return to_str(me, op, side), 1, None  # the opponent passes
    a, b = bin(me).count("1"), bin(op).count("1")
    e = 64 - a - b
    return None, 1, (a - b + e if a > b else a - b - e if a < b else 0)


# ---------------------------------------------------------------- engine runs with a cache
def parse_solve(text):
    rows = []
    for line in text.splitlines():
        c = [x.strip() for x in line.split("|") if x.strip()]
        if len(c) == 7 and c[0] != "Level":
            h, m, s = c[4].split(":")
            rows.append({"depth": c[1], "move": c[2].lower(), "value": int(c[3]), "time": int(h) * 3600 + int(m) * 60 + float(s), "nodes": int(c[5])})
    return rows


def load_cache(tag):
    f = WORK / f"cache_{tag}.jsonl"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["board"]] = r
    return out


def solve(boards, exe, prob, threads, procs, tag, per_process=0, hash_level=23, timeout=0):
    """solve boards with -dpr 1 60 60 <prob>; results cached in ignored/yomikiri/cache_<tag>.jsonl.
    per_process > 0: that many boards per engine process (shards otherwise). timeout > 0: kill a process
    after that many seconds (its boards stay unsolved and are listed in cache_<tag>.timeout)."""
    cache = load_cache(tag)
    todo = [b for b in dict.fromkeys(boards) if b not in cache]
    if not todo:
        return cache
    exe = (ROOT / exe).resolve()
    tmp = WORK / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    n_chunks = procs if per_process <= 0 else (len(todo) + per_process - 1) // per_process
    chunks = [todo[k::n_chunks] for k in range(n_chunks)]
    jobs = list(enumerate(chunks))
    running = []
    cache_file = open(WORK / f"cache_{tag}.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    done = 0
    while jobs or running:
        while jobs and len(running) < procs:
            k, chunk = jobs.pop(0)
            if not chunk:
                continue
            pf = tmp / f"{tag}_{k:04d}.txt"
            pf.write_text("".join(b + "\n" for b in chunk), encoding="utf-8")
            of = open(tmp / f"{tag}_{k:04d}.out", "w", encoding="utf-8")
            p = subprocess.Popen([str(exe), "-nobook", "-quiet", "-thread", str(threads), "-hash", str(hash_level), "-dpr", "1", "60", "60", str(prob), "-solve", str(pf)],
                                 cwd=exe.parent, stdin=subprocess.DEVNULL, stdout=of, stderr=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
            running.append((p, of, k, chunk, time.time()))
        for item in list(running):
            p, of, k, chunk, started = item
            if p.poll() is None:
                if timeout > 0 and time.time() - started > timeout:
                    p.kill()
                    p.wait()
                    with open(WORK / f"cache_{tag}.timeout", "a", encoding="utf-8") as tf:
                        tf.write("".join(b + "\n" for b in chunk))
                    print(f"  {tag}: timeout after {timeout} s for chunk {k}", flush=True)
                else:
                    continue
            of.close()
            running.remove(item)
            rows = parse_solve((tmp / f"{tag}_{k:04d}.out").read_text(encoding="utf-8"))
            for b, r in zip(chunk, rows):
                r["board"] = b
                cache[b] = r
                cache_file.write(json.dumps(r) + "\n")
            cache_file.flush()
            done += len(rows)
            if len(rows) != len(chunk):
                print(f"  warning: chunk {k} returned {len(rows)} of {len(chunk)}", flush=True)
            print(f"  {tag}: {done}/{len(todo)} solved, {time.time() - t0:.0f} s", flush=True)
        time.sleep(0.5)
    cache_file.close()
    return cache


# ---------------------------------------------------------------- steps
def step_sample(args):
    rows = [json.loads(l) for l in open(WORK / "candidates.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    pool = []
    for e in range(args.min_empties, args.max_empties + 1):
        # positions decided at a low probability (<= 98%) first, then a share of 99% decisions
        low = [r for r in rows if r["empties"] == e and min(float(d["prob"]) for d in r["decisions"]) <= 98]
        p99 = [r for r in rows if r["empties"] == e and min(float(d["prob"]) for d in r["decisions"]) == 99]
        rng.shuffle(low)
        rng.shuffle(p99)
        pool += low[:args.per_empties] + p99[:args.per_empties_99]
    with open(WORK / "pool.jsonl", "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    print(f"pool {len(pool)}")


def load_pool():
    return [json.loads(l) for l in open(WORK / "pool.jsonl", encoding="utf-8")]


def step_screen(args):
    pool = load_pool()
    cache = solve([r["board"] for r in pool], args.exe, args.prob, args.threads, args.procs, f"screen_p{args.prob}", hash_level=args.hash)
    flagged = []
    for r in pool:
        s = cache.get(r["board"])
        if s is None:
            continue
        bad = [d for d in r["decisions"] if d["move"] != s["move"] or abs(d["value"] - s["value"]) >= args.value_flag]
        if bad:
            flagged.append(r["board"])
    (WORK / "flagged.txt").write_text("".join(b + "\n" for b in flagged), encoding="utf-8")
    print(f"screened {len(cache)}, flagged {len(flagged)}")


def step_exact(args):
    pool = {r["board"]: r for r in load_pool()}
    flagged = [l for l in (WORK / "flagged.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(args.seed)
    rest = [b for b in pool if b not in set(flagged)]
    control = rng.sample(rest, min(len(rest), int(round(args.control * len(rest)))))
    targets = [b for b in flagged + control if pool[b]["empties"] <= args.max_exact_empties]
    (WORK / "exact_targets.txt").write_text("".join(b + "\n" for b in targets), encoding="utf-8")
    screen = load_cache(f"screen_p{args.screen_prob}")
    # root exact values
    roots = solve(targets, args.exe, 100, args.threads, args.procs, "exact", per_process=1, hash_level=args.hash, timeout=args.timeout)
    # child positions of the GGS moves and the screen move that differ from the exact best move
    kids = []
    for b in targets:
        if b not in roots:
            continue
        moves = {d["move"] for d in pool[b]["decisions"]}
        if b in screen:
            moves.add(screen[b]["move"])
        for m in moves - {roots[b]["move"]}:
            cb, _, term = child(b, m)
            if cb is not None:
                kids.append(cb)
    solve(kids, args.exe, 100, args.threads, args.procs, "exact", per_process=1, hash_level=args.hash, timeout=args.timeout)
    print(f"exact roots {len(targets)}, children {len(kids)}")


def move_values(b, exact):
    """known exact move values of position b (from the exact cache)"""
    me, op, _ = parse_board(b)
    out = {}
    if b in exact:
        out[exact[b]["move"]] = exact[b]["value"]
    for p in legal_moves(me, op):
        m = coord(p)
        cb, sign, term = child(b, m)
        if term is not None:
            out[m] = term
        elif cb in exact:
            out[m] = sign * exact[cb]["value"]
    return out


def step_write(args):
    pool = {r["board"]: r for r in load_pool()}
    exact = load_cache("exact")
    screen = load_cache(f"screen_p{args.screen_prob}")
    targets = [l for l in (WORK / "exact_targets.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    flagged = set(l for l in (WORK / "flagged.txt").read_text(encoding="utf-8").splitlines() if l.strip())
    problems = []
    n_mistake = 0
    for b in targets:
        if b not in exact:
            continue
        best = exact[b]["value"]
        mv = move_values(b, exact)
        decisions = []
        mistake = False
        for d in pool[b]["decisions"]:
            q = mv.get(d["move"])
            loss = None if q is None else best - q
            mistake |= bool(loss)
            decisions.append({**d, "exact_move_value": q, "loss": loss})
        n_mistake += mistake
        problems.append({"board": b, "empties": pool[b]["empties"], "best_value": best, "move_values": mv,
                         "ggs": decisions, "ggs_mistake": mistake, "flagged": b in flagged,
                         "screen": {k: screen[b][k] for k in ("move", "value")} if b in screen else None})
    problems.sort(key=lambda p: (-p["ggs_mistake"], p["empties"], p["board"]))
    out_txt = ROOT / "bin" / "problem" / "yomikiri_ggs.txt"
    out_json = ROOT / "bin" / "problem" / "yomikiri_ggs.json"
    out_txt.write_text("".join(p["board"] + "\n" for p in problems), encoding="utf-8")
    out_json.write_text(json.dumps(problems, indent=0), encoding="utf-8")
    print(f"problems {len(problems)}, GGS mistakes {n_mistake}")


def step_extend(args):
    out_json = ROOT / "bin" / "problem" / "yomikiri_ggs.json"
    problems = json.loads(out_json.read_text(encoding="utf-8"))
    by_board = {p["board"]: p for p in problems}
    todo = []
    for f in args.results:
        for r in json.loads(Path(f).read_text(encoding="utf-8"))["results"]:
            if r["loss"] is None and r["move"] not in by_board[r["board"]]["move_values"]:
                todo.append((r["board"], r["move"]))
    if args.from_sweep:  # moves listed by reports/20261019_3_yomikiri_sweep.py score
        for f in sorted(WORK.glob("sweep_unknown_p*.txt")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    b, m = line.split("	")
                    if m not in by_board[b]["move_values"]:
                        todo.append((b, m))
    todo = list(dict.fromkeys(todo))
    kids = [child(b, m)[0] for b, m in todo]
    exact = solve([k for k in kids if k is not None], args.exe, 100, args.threads, args.procs, "exact", per_process=1, hash_level=args.hash)
    for (b, m), k in zip(todo, kids):
        _, sign, term = child(b, m)
        by_board[b]["move_values"][m] = term if k is None else sign * exact[k]["value"]
    out_json.write_text(json.dumps(problems, indent=0), encoding="utf-8")
    print(f"added {len(todo)} move values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "screen", "exact", "write", "extend"])
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--from-sweep", action="store_true")
    ap.add_argument("--min-empties", type=int, default=28)
    ap.add_argument("--max-empties", type=int, default=32)
    ap.add_argument("--per-empties", type=int, default=300)
    ap.add_argument("--per-empties-99", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--exe", default="bin/Egaroucid_nnue_base.exe")
    ap.add_argument("--prob", default="99.9")
    ap.add_argument("--screen-prob", default="99.9")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--procs", type=int, default=26)
    ap.add_argument("--hash", type=int, default=23)
    ap.add_argument("--value-flag", type=int, default=4)
    ap.add_argument("--control", type=float, default=0.1)
    ap.add_argument("--timeout", type=float, default=0, help="seconds per exact solve (0: no limit)")
    ap.add_argument("--max-exact-empties", type=int, default=60, help="solve only targets with at most this many empties")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"sample": step_sample, "screen": step_screen, "exact": step_exact, "write": step_write, "extend": step_extend}[args.step](args)


if __name__ == "__main__":
    main()
