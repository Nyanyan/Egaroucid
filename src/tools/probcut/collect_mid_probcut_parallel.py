"""Run mid_probcut_dataset_tool over many positions in parallel (one thread per process).

For each deep depth D, the shallow depths are 0 (static evaluation) and the midgame MPC policy's
shallow depth for D (MID_MPC_POLICY_SHALLOW_DEPTH in src/engine/multi_probcut.hpp). Positions are
split into shards; every (D, shard) is one process writing <out>/<tag>_d<D>_<shard>.tsv (plus a .done
marker when it exits normally), and finished jobs are skipped on a rerun. Positions without a legal
move are skipped by the tool. Processes run at below-normal priority.

Usage:
    python src/tools/probcut/collect_mid_probcut_parallel.py --tool bin/mid_probcut_dataset_nnue.exe --tag nnue \
        --positions ignored/mpc_data/positions_6000.txt --out ignored/mpc_data --depths 3-14:3000 15-16:1000:20 --procs 24
A third field in a depth range is that range's positions per process (deep searches: smaller shards keep
the last processes from running alone for hours).
"""
import argparse
import subprocess
import time
from pathlib import Path

POLICY_SHALLOW = {3: 1, 4: 0, 5: 1, 6: 2, 7: 3, 8: 2, 9: 3, 10: 4, 11: 5, 12: 4, 13: 5, 14: 4, 15: 7, 16: 6, 17: 7, 18: 6}  # 17-18: the formula


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", nargs="+", required=True, help="ranges with position counts and optional shard sizes, e.g. 3-14:3000 15-16:1000:20")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--shard", type=int, default=100, help="positions per process")
    ap.add_argument("--hash", type=int, default=20)
    args = ap.parse_args()
    tool = Path(args.tool).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lines = [l for l in Path(args.positions).read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []
    for spec in args.depths:
        rng, _, rest = spec.partition(":")
        count, _, shard = rest.partition(":")
        shard = int(shard or args.shard)
        lo, _, hi = rng.partition("-")
        for d in range(int(lo), int(hi or lo) + 1):
            n = min(int(count), len(lines))
            for k in range(0, n, shard):
                jobs.append((d, k, min(k + shard, n)))
    jobs.sort(key=lambda j: -j[0])  # deep ones first: they are the long ones
    pending, running = list(jobs), {}
    t0 = time.time()
    done = 0
    while pending or running:
        while pending and len(running) < args.procs:
            d, lo, hi = pending.pop(0)
            res = out / f"{args.tag}_d{d}_{lo:05d}.tsv"
            if res.with_suffix(".done").exists():
                done += 1
                continue
            pos = out / f"{args.tag}_d{d}_{lo:05d}.pos"
            pos.write_text("\n".join(lines[lo:hi]) + "\n", encoding="utf-8")
            f = open(res, "w", encoding="utf-8")
            depths = f"0,{POLICY_SHALLOW[d]}" if POLICY_SHALLOW[d] else "0"
            proc = subprocess.Popen([str(tool), str(pos), str(d), depths, "1", str(args.hash)], cwd=tool.parent,
                                    stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.DEVNULL,
                                    creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
            running[(d, lo)] = (proc, f, pos, res)
        for key in list(running):
            proc, f, pos, res = running[key]
            if proc.poll() is None:
                continue
            f.close()
            pos.unlink(missing_ok=True)
            if proc.returncode == 0:
                res.with_suffix(".done").write_text("", encoding="utf-8")
            else:
                print(f"\njob {key} exited with {proc.returncode}", flush=True)
            done += 1
            del running[key]
        print(f"\r{done}/{len(jobs)} jobs, {time.time() - t0:.0f} s   ", end="", flush=True)
        time.sleep(1)
    print()


if __name__ == "__main__":
    main()
