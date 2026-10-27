"""Score -solve/movetime choices, retaining unknown moves rather than dropping them.

python src/tools/eval_stick/score_eval_stick.py --engine ENGINE --out OUT -- -thread 8 -hash 28 -movetime 3000
"""
import argparse
import json
from pathlib import Path
import sys

from common import ROOT, json_lines, legal, stamp, write_json
from engine import solve


def judge(position, result):
    move = result["move"]
    if move not in legal(position["board"], position["turn"]):
        raise ValueError("engine returned illegal move: " + move)
    values = {c["move"]: c["value"] for c in position["candidates"]}
    known = move in values
    best = max(values.values())
    return {"id": position["id"], "empties": position["empties"], "game": position["game"],
            **result, "loss": best - values[move] if known else None,
            "best_match": values[move] == best if known else None,
            "reference_value": values.get(move), "best_value": best,
            "unscored_move": not known}


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--bench", type=Path, default=ROOT / "bin/problem/eval_stick/eval_stick.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    rows = json_lines(args.bench)[:args.limit]
    if not rows:
        raise ValueError("empty benchmark")
    run = solve(args.engine, extra, rows, args.out)
    results = [judge(p, r) for p, r in zip(rows, run["rows"])]
    args.out.joinpath("results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    known = [r for r in results if r["loss"] is not None]
    summary = {"created": stamp(), "positions": len(results), "known": len(known),
               "unscored_moves": [r for r in results if r["unscored_move"]],
               "mean_loss": sum(r["loss"] for r in known) / len(known) if known else None,
               "best_matches": sum(r["best_match"] for r in known)}
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
