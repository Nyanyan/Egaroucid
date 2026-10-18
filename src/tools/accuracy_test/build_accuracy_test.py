"""Collect positions whose move values are known from earlier deep analyses into one benchmark.

Sources (all optional, searched relative to the repository root):
  - ggs_analyze caches  (bin/ggs/log/**/positions_*.jsonl, reports/data_*/cache/positions_*.jsonl)
      each row: the best move and, when the played move differs, the played move, searched with
      the same depth and MPC probability
  - GGS divergence references (reports/data_20261010_1_divergence/positions.csv + searches.csv)
      two candidate moves per position, searched to the end with MPC
  - 2000-position time-schedule references (reports/data_20260929_2/reference_results.jsonl)
      values of the moves on which schedules disagreed

Output: bin/problem/accuracy_test/accuracy_test.jsonl, one position per line:
  {"id", "board" (64 chars, X = black, O = white), "turn" ("X" or "O", side to move), "empties",
   "evals": [{"depth": "36@88%", "values": {move: score for the side to move}, "best": move or null,
              "kind": "ggs_analyze" | "divergence_ref" | "schedule_ref", "source": ...}]}
"best" is set only when the eval searched every move (an unrestricted search found it); otherwise the
values only compare the listed moves.

Usage: python src/tools/accuracy_test/build_accuracy_test.py [--out DIR]
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def norm_board(board64, turn):
    assert len(board64) == 64 and turn in "XO", (board64, turn)
    return board64, turn


def parse_board_turn(s):
    s = s.strip()
    return norm_board(s[:64], s[-1])


def reliability(depth):
    # "36@88%" -> (88, 36); exact searches ("@100%") rank first, then deeper
    try:
        d, p = depth.rstrip("%").split("@")
        return (float(p), int(d))
    except ValueError:
        return (0.0, 0)


def collect_ggs_analyze(positions):
    files = sorted(set(glob.glob(str(ROOT / "bin/ggs/log/**/positions_*.jsonl"), recursive=True))
                   | set(glob.glob(str(ROOT / "reports/data_*/cache/positions_*.jsonl"))))
    n = 0
    for f in files:
        setting = Path(f).stem  # positions_<hash of the depth schedule>
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            board, turn = parse_board_turn(r["board"])
            values = {r["played"]: r["played_score"]}
            if r.get("alt_move"):
                if r.get("alt_depth") and r["alt_depth"] != r["played_depth"]:
                    continue
                values[r["alt_move"]] = r["alt_score"]
                # the unrestricted search's move often scores below the played move (search noise),
                # so the best is the higher of the two, as ggs_analyze itself decides
                best = r["alt_move"] if r["alt_score"] > r["played_score"] else r["played"]
            else:
                best = r["played"]
            positions[(board, turn)].append({
                "depth": r["played_depth"], "values": values, "best": best, "kind": "ggs_analyze",
                "source": f"{setting} {r.get('source', '')}".strip()})
            n += 1
    return n, len(files)


def collect_divergence(positions):
    base = ROOT / "reports/data_20261010_1_divergence"
    if not (base / "positions.csv").exists():
        return 0
    pos = {}
    with open(base / "positions.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pos[r["position"]] = norm_board(r["board"], r["turn"])
    groups = defaultdict(dict)
    with open(base / "searches.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stage"] != "reference" or r["position"] not in pos or r["value"] == "":
                continue
            depth = f"{r['depth']}@{float(r['probability']):g}%"
            groups[(r["position"], depth)][r["move"]] = int(float(r["value"]))
    n = 0
    for (p, depth), values in groups.items():
        if len(values) < 2:
            continue
        positions[pos[p]].append({"depth": depth, "values": values, "best": None, "kind": "divergence_ref",
                                  "source": f"20261010_1 position {p}"})
        n += 1
    return n


def collect_schedule_refs(positions):
    f = ROOT / "reports/data_20260929_2/reference_results.jsonl"
    if not f.exists():
        return 0
    n = 0
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if len(r.get("reference_values") or {}) < 2:
            continue
        board, turn = parse_board_turn(r["board"])
        positions[(board, turn)].append({
            "depth": f"ref@{r['reference_probability']}%", "values": r["reference_values"], "best": None,
            "kind": "schedule_ref", "source": f"20260929 position {r['position_id']}"})
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "bin/problem/accuracy_test"))
    args = ap.parse_args()
    positions = defaultdict(list)
    n_ggs, n_files = collect_ggs_analyze(positions)
    n_div = collect_divergence(positions)
    n_sched = collect_schedule_refs(positions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (board, turn), evals in positions.items():
        # drop exact duplicates, keep the most reliable first
        seen = set()
        uniq = []
        for e in sorted(evals, key=lambda e: reliability(e["depth"]), reverse=True):
            key = (e["depth"], tuple(sorted(e["values"].items())), e["best"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        rows.append({"board": board, "turn": turn, "empties": board.count("-"), "evals": uniq})
    rows.sort(key=lambda r: (-r["empties"], r["board"], r["turn"]))
    for i, r in enumerate(rows):
        r["id"] = i
    with open(out / "accuracy_test.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "board": r["board"], "turn": r["turn"], "empties": r["empties"],
                                "evals": r["evals"]}, ensure_ascii=False) + "\n")
    by_e = defaultdict(int)
    with_best = defaultdict(int)
    with_loss = defaultdict(int)
    for r in rows:
        b = r["empties"] // 10 * 10
        by_e[b] += 1
        if any(e["best"] for e in r["evals"]):
            with_best[b] += 1
        if any(e["best"] and len(e["values"]) >= 2 for e in r["evals"]):
            with_loss[b] += 1
    print(f"ggs_analyze rows {n_ggs} from {n_files} files, divergence evals {n_div}, schedule evals {n_sched}")
    print(f"positions {len(rows)}")
    for b in sorted(by_e, reverse=True):
        print(f"  empties {b:2d}-{b + 9:2d}: {by_e[b]:5d} positions, {with_best[b]:5d} with best move, "
              f"{with_loss[b]:5d} where a known worse move was played")


if __name__ == "__main__":
    main()
