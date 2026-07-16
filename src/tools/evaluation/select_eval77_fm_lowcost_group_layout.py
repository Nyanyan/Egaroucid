import argparse
import math
import re
import sys


N_PHASES = 60
PHASE_RE = re.compile(r"^phase\s+(\d+)\s+n=(\d+).*?\srmse=([0-9.]+)")
MODE_RE = re.compile(r"^grouped(\d+)(copyfirst|copylast)$")


def read_phase_sse(path):
    stats = {}
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    match = PHASE_RE.search(line)
                    if match is None:
                        continue
                    phase = int(match.group(1))
                    n = int(match.group(2))
                    rmse = float(match.group(3))
                    stats[phase] = n * rmse * rmse
            break
        except UnicodeDecodeError:
            stats.clear()
            continue
    if not stats:
        raise ValueError("no phase stats found in {}".format(path))
    return stats


def parse_candidate_arg(arg):
    if "=" not in arg:
        raise ValueError("candidate must be MODE=PATH: {}".format(arg))
    mode, path = arg.split("=", 1)
    match = MODE_RE.match(mode)
    if match is None:
        raise ValueError("candidate mode must look like groupedNcopyfirst/copylast: {}".format(mode))
    return mode, int(match.group(1)), match.group(2), path


def add_cost(costs, start, sse, source, representative):
    current = costs.get(start)
    if current is None or sse < current["sse"]:
        costs[start] = {
            "sse": sse,
            "source": source,
            "representative": representative,
        }


def collect_pair_costs(candidate_args):
    costs = {}
    for arg in candidate_args:
        mode, group_count, copy_mode, path = parse_candidate_arg(arg)
        phase_sse = read_phase_sse(path)
        for group in range(group_count):
            start = group * N_PHASES // group_count
            end = (group + 1) * N_PHASES // group_count
            if end - start != 2:
                continue
            if copy_mode == "copyfirst":
                error_phase = end - 1
                representative = start
            else:
                error_phase = start
                representative = end - 1
            if error_phase not in phase_sse:
                continue
            add_cost(costs, start, phase_sse[error_phase], mode, representative)
    return costs


def solve_layout(target_groups, pair_costs):
    pairs_needed = N_PHASES - target_groups
    if pairs_needed < 0 or N_PHASES // 2 < pairs_needed:
        raise ValueError("target_groups must be in [{}, {}]".format(N_PHASES // 2, N_PHASES))

    inf = math.inf
    dp = [[inf] * (pairs_needed + 1) for _ in range(N_PHASES + 2)]
    take = [[False] * (pairs_needed + 1) for _ in range(N_PHASES + 2)]
    dp[N_PHASES][0] = 0.0
    dp[N_PHASES + 1][0] = 0.0
    for i in range(N_PHASES - 1, -1, -1):
        for k in range(pairs_needed + 1):
            best = dp[i + 1][k]
            use_pair = inf
            if k > 0 and i <= N_PHASES - 2 and i in pair_costs:
                use_pair = pair_costs[i]["sse"] + dp[i + 2][k - 1]
            if use_pair < best:
                dp[i][k] = use_pair
                take[i][k] = True
            else:
                dp[i][k] = best

    if not math.isfinite(dp[0][pairs_needed]):
        raise ValueError("cannot build a {}-group layout from the supplied pair costs".format(target_groups))

    pairs = []
    i = 0
    k = pairs_needed
    while i < N_PHASES and k > 0:
        if take[i][k]:
            pairs.append(i)
            i += 2
            k -= 1
        else:
            i += 1

    sizes = []
    representatives = []
    phase = 0
    for pair_start in pairs:
        while phase < pair_start:
            sizes.append(1)
            representatives.append(phase)
            phase += 1
        sizes.append(2)
        representatives.append(pair_costs[pair_start]["representative"])
        phase += 2
    while phase < N_PHASES:
        sizes.append(1)
        representatives.append(phase)
        phase += 1

    return dp[0][pairs_needed], pairs, sizes, representatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_groups", type=int)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="MODE=PATH",
        help="phase-error log from a groupedNcopyfirst/copylast model; repeatable",
    )
    args = parser.parse_args()

    try:
        pair_costs = collect_pair_costs(args.candidate)
        total_sse, pairs, sizes, representatives = solve_layout(args.target_groups, pair_costs)
    except ValueError as e:
        print("[ERROR] {}".format(e), file=sys.stderr)
        return 1

    print("target_groups={}".format(args.target_groups))
    print("pairs_needed={}".format(N_PHASES - args.target_groups))
    print("total_sse={:.0f}".format(total_sse))
    print("pairs")
    for pair_start in pairs:
        info = pair_costs[pair_start]
        print(
            "{}-{} source={} representative={} sse={:.0f}".format(
                pair_start,
                pair_start + 1,
                info["source"],
                info["representative"],
                info["sse"],
            )
        )
    print("sizes={}".format(",".join(str(v) for v in sizes)))
    print("representatives={}".format(",".join(str(v) for v in representatives)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
