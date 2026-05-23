import argparse
import os
import re
import statistics
import subprocess
import tempfile


def run_once(exe, problem_file, seconds, threads, hash_level):
    cmd = [
        exe,
        "-time", str(seconds),
        "-nobook",
        "-thread", str(threads),
        "-hash", str(hash_level),
        "-solve", problem_file,
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    total_match = re.search(r"total\s+(\d+)\s+nodes\s+in\s+([0-9.]+)s\s+NPS\s+(\d+)", completed.stdout)
    if not total_match:
        raise RuntimeError("total line was not found in benchmark output")
    return {
        "nodes": int(total_match.group(1)),
        "time_sec": float(total_match.group(2)),
        "nps": int(total_match.group(3)),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def make_problem_subset(source, count):
    with open(source, "r", encoding="utf-8") as f:
        boards = [line for line in f if line.strip()]
    boards = boards[:count]
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix="_egaroucid_tl_bench.txt", encoding="utf-8")
    with tmp:
        tmp.writelines(boards)
    return tmp.name


def summarize(label, results):
    times = [r["time_sec"] for r in results]
    nodes = [r["nodes"] for r in results]
    nps = [r["nps"] for r in results]
    summary = {
        "time_avg": statistics.mean(times),
        "time_median": statistics.median(times),
        "nodes_avg": statistics.mean(nodes),
        "nps_avg": statistics.mean(nps),
    }
    print(
        f"{label}: runs={len(results)} "
        f"time_avg={summary['time_avg']:.3f}s "
        f"time_median={summary['time_median']:.3f}s "
        f"nodes_avg={summary['nodes_avg']:.0f} "
        f"nps_avg={summary['nps_avg']:.0f}"
    )
    return summary


def print_delta(baseline, current):
    def pct(new, old):
        if old == 0:
            return 0.0
        return (new - old) * 100.0 / old

    print(
        "delta: "
        f"time_avg={pct(current['time_avg'], baseline['time_avg']):+.2f}% "
        f"nodes_avg={pct(current['nodes_avg'], baseline['nodes_avg']):+.2f}% "
        f"nps_avg={pct(current['nps_avg'], baseline['nps_avg']):+.2f}%"
    )


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_exe = os.path.join(repo_root, "bin", "Egaroucid_for_Console.exe")
    default_problems = os.path.join(repo_root, "bin", "problem", "midgame_test.txt")
    parser = argparse.ArgumentParser(description="Small time-limit search benchmark for Egaroucid.")
    parser.add_argument("--exe", default=default_exe)
    parser.add_argument("--baseline-exe")
    parser.add_argument("--problems", default=default_problems)
    parser.add_argument("--positions", type=int, default=6)
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hash", type=int, default=21)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    problem_subset = make_problem_subset(args.problems, args.positions)
    try:
        baseline_summary = None
        if args.baseline_exe:
            baseline = [run_once(args.baseline_exe, problem_subset, args.seconds, args.threads, args.hash) for _ in range(args.repeat)]
            baseline_summary = summarize("baseline", baseline)
        current = [run_once(args.exe, problem_subset, args.seconds, args.threads, args.hash) for _ in range(args.repeat)]
        current_summary = summarize("current", current)
        if baseline_summary:
            print_delta(baseline_summary, current_summary)
    finally:
        os.remove(problem_subset)


if __name__ == "__main__":
    main()
