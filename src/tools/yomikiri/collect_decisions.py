"""Collect GGS endgame-search-with-MPC (読み切り) decisions from the GGS client logs.

Reads every "search result game ..." line under bin/ggs/log (recursively). A decision is a 読み切り when
the reported depth equals the number of empties and the MPC probability is below 100%. Positions are
deduplicated by board and side; every decision on the same position is kept.

Usage: python src/tools/yomikiri/collect_decisions.py [--min-empties 24] [--max-empties 40]
Output: ignored/yomikiri/candidates.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "bin" / "ggs" / "log"
OUT_DIR = ROOT / "ignored" / "yomikiri"

PAT = re.compile(r"search result game (\S+) discs (\d+) move (\S+) value (-?\d+) depth (\d+)@([\d.]+)% time (\d+) .*? board ([-XO]{64}) ([XO])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-empties", type=int, default=24)
    ap.add_argument("--max-empties", type=int, default=40)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    positions = {}
    n_lines = 0
    for log in sorted(LOG_DIR.rglob("*_ggs.log")):
        rel = log.relative_to(ROOT).as_posix()
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "search result game" not in line:
                    continue
                m = PAT.search(line)
                if not m:
                    continue
                game, discs, move, value, depth, prob, tm, board, side = m.groups()
                empties = 64 - int(discs)
                if int(depth) != empties or float(prob) >= 100 or not (args.min_empties <= empties <= args.max_empties):
                    continue
                n_lines += 1
                key = board + " " + side
                p = positions.setdefault(key, {"board": key, "empties": empties, "decisions": []})
                p["decisions"].append({"move": move.lower(), "value": int(value), "prob": prob, "time_ms": int(tm), "log": rel, "game": game})
    with open(OUT_DIR / "candidates.jsonl", "w", encoding="utf-8") as f:
        for p in positions.values():
            f.write(json.dumps(p) + "\n")
    by_e = Counter(p["empties"] for p in positions.values())
    print(f"decisions {n_lines}, unique positions {len(positions)}")
    print(" ".join(f"{e}:{by_e[e]}" for e in sorted(by_e)))


if __name__ == "__main__":
    main()
