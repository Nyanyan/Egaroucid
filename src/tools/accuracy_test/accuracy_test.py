"""Score an engine build on bin/problem/accuracy_test/accuracy_test.jsonl.

The engine solves every selected position with `-solve` (the transposition table is cleared per
position) and the chosen move is judged against the evals stored for that position:
  - best match : the eval that searched every move (ggs_analyze) names the same best move
  - known loss : best value - chosen move value, when the chosen move has a value in the same eval
  - unknown    : the chosen move was never searched by a stored eval
Positions without a full eval (divergence / schedule references) only count pairwise: the chosen move
is scored against the best of the listed moves when it is one of them.

Usage:
  python src/tools/accuracy_test/accuracy_test.py --engine bin/Egaroucid_for_Console.exe \
      --min-empties 30 --max-empties 59 --out reports/data_x/acc_base -- -l 11 -thread 1
Everything after "--" is passed to the engine (level, threads, -dpr, ...).
"""
import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCH = ROOT / "bin/problem/accuracy_test/accuracy_test.jsonl"


def load(bench, lo, hi, limit, stride):
    rows = [json.loads(l) for l in open(bench, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if lo <= r["empties"] <= hi]
    rows = rows[::stride]
    return rows[:limit] if limit else rows


def parse_time(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def run_engine(engine, extra, rows, workdir):
    prob = workdir / "problems.txt"
    with open(prob, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(f"{r['board']} {r['turn']}\n")
    cmd = [str(engine), "-nobook", "-quiet"] + extra + ["-solve", str(prob)]
    p = subprocess.run(cmd, cwd=Path(engine).parent, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    out = []
    for line in p.stdout.splitlines():
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) == 7 and cols[0].isdigit():
            out.append({"depth": cols[1], "move": cols[2].lower(), "score": int(cols[3]),
                        "time": parse_time(cols[4]), "nodes": int(cols[5])})
    if len(out) != len(rows):
        sys.exit(f"engine returned {len(out)} results for {len(rows)} positions\n{p.stderr[-2000:]}")
    return out


def judge(row, move):
    full = [e for e in row["evals"] if e["best"]]
    for e in full:
        if move in e["values"]:
            loss = max(e["values"].values()) - e["values"][move]
            return {"kind": "full", "best": loss == 0, "loss": loss, "depth": e["depth"]}
    if full:
        return {"kind": "full", "best": False, "loss": None, "depth": full[0]["depth"]}
    for e in row["evals"]:
        if move in e["values"]:
            top = max(e["values"].values())
            return {"kind": "pair", "best": e["values"][move] == top, "loss": top - e["values"][move], "depth": e["depth"]}
    return {"kind": "pair", "best": None, "loss": None, "depth": None}


def summarize(rows, results, judged):
    groups = defaultdict(list)
    for r, res, j in zip(rows, results, judged):
        b = (r["empties"] // 5) * 5
        groups[f"{b:02d}-{b + 4:02d}"].append((r, res, j))
        groups["all"].append((r, res, j))
    lines = ["| empties | positions | best match (full) | known loss mean | unknown | pairwise best | time (s) | nodes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for g in sorted(groups, key=lambda k: (k == "all", k), reverse=False):
        items = groups[g]
        full = [j for _, _, j in items if j["kind"] == "full"]
        n_best = sum(1 for j in full if j["best"])
        known = [j["loss"] for j in full if j["loss"] is not None]
        unknown = sum(1 for j in full if j["loss"] is None)
        pair = [j for _, _, j in items if j["kind"] == "pair" and j["best"] is not None]
        t = sum(res["time"] for _, res, _ in items)
        n = sum(res["nodes"] for _, res, _ in items)
        rate = n_best / len(full) if full else float("nan")
        se = math.sqrt(rate * (1 - rate) / len(full)) if full else float("nan")
        mean_loss = sum(known) / len(known) if known else float("nan")
        pr = sum(1 for j in pair if j["best"]) / len(pair) if pair else float("nan")
        summary[g] = {"positions": len(items), "full": len(full), "best_match": rate, "best_match_se": se,
                      "known_loss_mean": mean_loss, "known": len(known), "unknown": unknown,
                      "pairwise_best": pr, "pairwise": len(pair), "time": t, "nodes": n}
        lines.append(f"| {g} | {len(items)} | {rate:.3f} ± {se:.3f} ({len(full)}) | {mean_loss:.3f} ({len(known)}) | "
                     f"{unknown} | {pr:.3f} ({len(pair)}) | {t:.1f} | {n} |")
    return summary, "\n".join(lines)


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--bench", default=str(DEFAULT_BENCH))
    ap.add_argument("--min-empties", type=int, default=0)
    ap.add_argument("--max-empties", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.bench, args.min_empties, args.max_empties, args.limit, args.stride)
    results = run_engine(Path(args.engine).resolve(), extra, rows, out)
    judged = [judge(r, res["move"]) for r, res in zip(rows, results)]
    with open(out / "results.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r, res, j in zip(rows, results, judged):
            f.write(json.dumps({"id": r["id"], "empties": r["empties"], **res, "judge": j}) + "\n")
    summary, table = summarize(rows, results, judged)
    meta = {"engine": str(args.engine), "engine_args": extra, "bench": str(args.bench),
            "min_empties": args.min_empties, "max_empties": args.max_empties, "limit": args.limit, "stride": args.stride}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1), encoding="utf-8")
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
