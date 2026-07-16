import argparse
import math
import re
import sys


N_PHASES = 60
MODE_RE = re.compile(r"^grouped(\d+)(copyfirst|copylast)$")
SIGN_COST_WEIGHT = 1.0e9


def parse_phase_line(line):
    parts = line.strip().split()
    if len(parts) < 3 or parts[0] != "phase":
        return None
    try:
        phase = int(parts[1])
    except ValueError:
        return None
    fields = {}
    for part in parts[2:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    fields["phase"] = phase
    return fields


def get_required_float(fields, key, path, phase):
    if key not in fields:
        raise ValueError("missing {} in {} phase {}".format(key, path, phase))
    return float(fields[key])


def get_required_int(fields, key, path, phase):
    if key not in fields:
        raise ValueError("missing {} in {} phase {}".format(key, path, phase))
    return int(fields[key])


def compute_phase_cost(fields, metric, path):
    phase = fields["phase"]
    n = get_required_int(fields, "n", path, phase)
    rmse = get_required_float(fields, "rmse", path, phase)
    if metric == "rmse":
        return n * rmse * rmse
    if metric == "mae":
        mae = get_required_float(fields, "mae", path, phase)
        return n * mae
    if metric == "sign":
        sign_disagree = get_required_int(fields, "sign_disagree", path, phase)
        return sign_disagree * SIGN_COST_WEIGHT + n * rmse * rmse
    raise ValueError("unsupported metric: {}".format(metric))


def read_phase_costs(path, metric):
    stats = {}
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    fields = parse_phase_line(line)
                    if fields is None:
                        continue
                    stats[fields["phase"]] = compute_phase_cost(fields, metric, path)
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


def collect_pair_costs(candidate_args, metric):
    costs = {}
    for arg in candidate_args:
        mode, group_count, copy_mode, path = parse_candidate_arg(arg)
        phase_costs = read_phase_costs(path, metric)
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
            if error_phase not in phase_costs:
                continue
            add_cost(costs, start, phase_costs[error_phase], mode, representative)
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
    parser.add_argument(
        "--metric",
        choices=("rmse", "mae", "sign"),
        default="rmse",
        help="pair cost metric; sign minimizes sign disagreements first and breaks ties by RMSE",
    )
    args = parser.parse_args()

    try:
        pair_costs = collect_pair_costs(args.candidate, args.metric)
        total_cost, pairs, sizes, representatives = solve_layout(args.target_groups, pair_costs)
    except ValueError as e:
        print("[ERROR] {}".format(e), file=sys.stderr)
        return 1

    print("target_groups={}".format(args.target_groups))
    print("metric={}".format(args.metric))
    print("pairs_needed={}".format(N_PHASES - args.target_groups))
    print("total_cost={:.0f}".format(total_cost))
    print("pairs")
    for pair_start in pairs:
        info = pair_costs[pair_start]
        print(
            "{}-{} source={} representative={} cost={:.0f}".format(
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
