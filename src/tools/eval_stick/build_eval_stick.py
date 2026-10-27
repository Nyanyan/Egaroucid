"""Extract a reproducible pool; publish only positions with complete references.

The engine runner is in reports/20261026_3_run.py. This module contains the
reusable board/source selection and final JSONL construction steps.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
import re

from common import (ROOT, canonical, coord, digest, json_lines, legal, play,
                    read_json, stamp, transform_move, write_json)

DATA = ROOT / "reports/data_20261026_3"
SEED = 202610263
PROCESSING_SEED = 202610264


def processing_order(chosen):
    """Randomize the selected corpus, independently of selection priorities."""
    rng = random.Random(PROCESSING_SEED)
    buckets = defaultdict(list)
    for row in chosen:
        row.setdefault("selection_rank", row["selection_order"])
        buckets[row["empties"]].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)
    ordered = []
    while any(buckets.values()):
        counts = list(range(41, 51)); rng.shuffle(counts)
        for empty in counts:
            if buckets[empty]:
                row = buckets[empty].pop()
                row["selection_order"] = len(ordered)
                ordered.append(row)
    return ordered


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def extract_pool(data=DATA, count=200):
    dest = data / "selection.json"
    if dest.exists():
        return read_json(dest)
    pool = {}
    source_manifest = {}
    rejected = []

    def add(board, turn, moves, source, priority=0):
        if not 41 <= board.count("-") <= 50:
            return
        assert len(board) == 64 and set(board) <= set("XO-") and turn in ("X", "O")
        cb, ct, symmetry = canonical(board, turn)
        key = cb + " " + ct
        row = pool.setdefault(key, {"board": cb, "turn": ct, "empties": cb.count("-"),
                                   "source": [], "recorded_moves": {}, "priority": 0})
        src = dict(source, symmetry_to_canonical=symmetry)
        if src not in row["source"]:
            row["source"].append(src)
        row["priority"] = max(priority, row["priority"])
        for name, move in moves.items():
            cm = transform_move(move.lower(), symmetry)
            if cm not in legal(cb, ct):
                raise ValueError(("source move illegal", source, cm))
            row["recorded_moves"].setdefault(cm, []).append(name)

    old_dir = ROOT / "reports/data_20261010_1_divergence"
    legacy = csv_rows(old_dir / "positions.csv")
    for row in legacy:
        add(row["board"], row["turn"], {"egrcd": row["a"], "opponent": row["b"]},
            {"kind": "divergence_1010", "file": "reports/data_20261010_1_divergence/positions.csv",
             "position": int(row["position"]), "session": row["session"],
             "game": row["session"] + "/" + row["game"],
             "pair": row["session"] + "/" + row["pair"], "record": row["source_a"]}, 2)
    source_manifest["reports/data_20261010_1_divergence/positions.csv"] = digest(old_dir / "positions.csv")

    # Existing parsed game inventories bind saved transcripts to exact sessions.
    inventories = []
    for dirname in ("data_20261009_0", "data_20261016_0", "data_20261017_0", "data_20261017_1", "data_20261023_2"):
        base = ROOT / "reports" / dirname
        if base.exists():
            inventories.extend(base.rglob("games.csv"))
    metadata = {}
    for path in sorted(inventories):
        rows = csv_rows(path)
        if rows and "saved_record" in rows[0]:
            source_manifest[str(path.relative_to(ROOT))] = digest(path)
        for row in rows:
            record = row.get("saved_record", "").replace("\\", "/")
            if not record:
                continue
            session = row.get("session") or path.parent.name
            game = row.get("gid") or row.get("game") or row.get("game_id") or row.get("key")
            order = row.get("match_order") or row.get("pair") or row.get("match_id") or game.rsplit(".", 1)[0]
            metadata.setdefault(record, {"session": session, "game": session + "/" + game,
                                         "pair": session + "/" + order,
                                         "match_order": row.get("match_order", "")})

    divergence23 = {int(r["match_order"]): r for r in csv_rows(ROOT / "reports/data_20261023_2/checks/divergence.csv")
                    if r["condition"] == "10-23 8T N" and r["divergence_empties"] != "0"}
    targets = {12: (45, "a6"), 51: (49, "g6"), 91: (49, "d7"),
               14: (46, "d7"), 96: (41, "c7"), 107: (44, "h6"), 89: (43, "b5")}
    seen_records = set()
    replayed = {}
    for path in sorted((ROOT / "bin/ggs/log").rglob("*.txt")):
        if not re.match(r"2026-10-(09|10|16|17|18|23|24)-", path.name):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if "initial board:" not in text or "transcript:" not in text:
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen_records:
            continue
        seen_records.add(h)
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        source_manifest[relative] = digest(path)
        try:
            fields = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
            if "egrcd" not in (fields.get("black"), fields.get("white")):
                continue
            board, turn = fields["initial board"].split()
            meta = metadata.get(relative)
            if meta is None:
                # A saved-record identity is conservative when an inventory is absent:
                # do not merge unrelated games merely because GGS reused an ID.
                gid = text.splitlines()[0].strip()
                session = path.parent.parent.name if path.parent.name == "game" and path.parent.parent.name != "log" else path.name[:10]
                meta = {"session": session, "game": relative, "pair": relative,
                        "match_order": "", "session_inferred": True, "gid": gid}
            moves = re.findall(r"[a-h][1-8]|pa", fields["transcript"].lower())
            history = []
            for move in moves:
                if not legal(board, turn):
                    turn = "O" if turn == "X" else "X"
                    if move == "pa":
                        continue
                if move == "pa":
                    raise ValueError("unnecessary pass")
                history.append((board, turn, move))
                board, turn = play(board, turn, move)
            score = board.count("X") - board.count("O")
            score += board.count("-") * ((score > 0) - (score < 0))
            final = int(fields["black's score"])
            if not legal(board, turn) and not legal(board, "O" if turn == "X" else "X") and score != final:
                raise ValueError(("final score", score, final))
            replayed[relative] = history
            for board, turn, move in history:
                source = {"kind": "record", "file": relative, **meta,
                          "actor": fields["black"] if turn == "X" else fields["white"]}
                priority = 0
                names = {source["actor"]: move}
                if "2026-10-23" in meta["session"] and meta["match_order"]:
                    order = int(meta["match_order"])
                    div = divergence23.get(order)
                    if div and board.count("-") == int(div["divergence_empties"]):
                        # Verify both recorded moves on this board. The first differing
                        # board is shared by the two games, later boards need not be.
                        if div["our_move"] in legal(board, turn) and div["their_move"] in legal(board, turn):
                            names.update(egrcd=div["our_move"], opponent=div["their_move"])
                            priority = 3
                            source["kind"] = "divergence_1023"
                    if order in targets and (board.count("-"), move) == targets[order]:
                        priority = 4
                        source["kind"] = "lost_class_1_2"
                add(board, turn, names, source, priority)
        except (KeyError, ValueError, AssertionError) as exc:
            rejected.append({"file": relative, "reason": str(exc)})

    # Round-robin across randomly ordered empty counts keeps every prefix balanced.
    # Within each count: mandatory class 1/2 positions, divergences, ordinary play.
    rng = random.Random(SEED)
    by_empty = defaultdict(list)
    for row in pool.values():
        row["id"] = hashlib.sha256((row["board"] + row["turn"]).encode()).hexdigest()[:16]
        row["source"].sort(key=lambda x: (-int(x["kind"] == "lost_class_1_2"),
                                           -int(x["kind"].startswith("divergence")), x["file"]))
        row["game"] = row["source"][0]["game"]
        row["pair"] = row["source"][0]["pair"]
        row["legal_moves"] = legal(row["board"], row["turn"])
        by_empty[row["empties"]].append(row)
    for rows in by_empty.values():
        rng.shuffle(rows)
        # Reserve roughly one quarter for ordinary play instead of filling entirely
        # from disagreements. Forced loss examples stay first within their stratum.
        special = sorted([r for r in rows if r["priority"]], key=lambda r: -r["priority"])
        ordinary = [r for r in rows if not r["priority"]]
        mixed = []
        while special or ordinary:
            mixed.extend(special[:3]); del special[:3]
            if ordinary:
                mixed.append(ordinary.pop())
            elif special:
                continue
        rows[:] = mixed
    chosen = []
    used = set()
    per_empty = Counter()
    while len(chosen) < count:
        counts = list(range(41, 51)); rng.shuffle(counts)
        changed = False
        for empty in counts:
            rows = by_empty[empty]
            while rows:
                row = rows.pop(0)
                band = (50 - empty) // 2
                aliases = {(s["game"], band) for s in row["source"]}
                if aliases & used:
                    continue
                row["selection_order"] = len(chosen)
                chosen.append(row); used.update(aliases)
                per_empty[empty] += 1; changed = True
                break
            if len(chosen) >= count:
                break
        if not changed:
            break
    chosen = processing_order(chosen)
    result = {"seed": SEED, "processing_seed": PROCESSING_SEED, "created": stamp(), "requested": count, "positions": chosen,
              "pool_size": len(pool), "replayed_records": len(replayed), "rejected_records": rejected,
              "counts": dict(per_empty), "source_manifest": source_manifest,
              "selection": "prioritized corpus selection; independent seeded random processing within empty strata; max one per source game/two-empty band; all eight symmetries deduplicated"}
    write_json(data / "pool.json", list(pool.values()))
    write_json(dest, result)
    return result


def publish(data=DATA, output=None):
    output = Path(output or ROOT / "bin/problem/eval_stick/eval_stick.jsonl")
    selection = read_json(data / "selection.json")
    candidates = read_json(data / "candidates.json", {})
    settings = read_json(data / "reference_settings.json")
    cache = {r["key"]: r for r in json_lines(data / "references.jsonl") if r.get("status") == "ok"}
    noise = read_json(data / "noise.json", {})
    rows = []
    workload = read_json(data / "workload.json")
    selected_positions = selection["positions"][:workload["target"]] if workload else selection["positions"]
    for pos in selected_positions:
        if pos["id"] not in candidates:
            continue
        record = candidates[pos["id"]]
        values = []
        for move in record["moves"]:
            key = reference_key(pos, move, settings)
            if key not in cache:
                break
            ref = cache[key]
            values.append({"move": move, "value": ref["value"], "settings": ref["settings"],
                           "cache_key": key, "origin": ref.get("origin", "new"),
                           "reason": record["reasons"].get(move, [])})
        else:
            if not values:
                continue
            best = max(v["value"] for v in values)
            rows.append({"id": pos["id"], "board": pos["board"], "turn": pos["turn"],
                         "empties": pos["empties"], "source": pos["source"], "game": pos["game"],
                         "pair": pos["pair"], "selection_order": pos["selection_order"],
                         "candidate_selection": record["method"], "candidates": values,
                         "best_move": min(v["move"] for v in values if v["value"] == best),
                         "best_moves": sorted(v["move"] for v in values if v["value"] == best),
                         "best_value": best, "reference_noise_flag": noise.get(pos["id"], {}).get("flag", "unchecked"),
                         "reference_noise": noise.get(pos["id"])})
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(output)
    return rows


def probability(empties):
    return 88 if empties >= 48 else 93 if empties >= 44 else 98


def reference_key(pos, move, settings, higher=False):
    prob = probability(pos["empties"])
    if higher:
        prob = {88: 93, 93: 98, 98: 99}[prob]
    payload = {"board": pos["board"], "turn": pos["turn"], "move": move,
               "depth": pos["empties"], "probability": prob, **settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pool", "publish"])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--count", type=int, default=200)
    args = ap.parse_args()
    if args.action == "pool":
        result = extract_pool(args.data, args.count)
        print(json.dumps({k: result[k] for k in ("pool_size", "replayed_records", "counts", "rejected_records")}, ensure_ascii=False))
    else:
        print("published", len(publish(args.data)))
