import argparse
from pathlib import Path


def read_phase(path):
    values = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                values.append(int(stripped))
    return values


def write_phase(path, values):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(f"{value}\n")


def parse_phases(value, n_phases):
    if value is None or value.strip() == "":
        return set(range(n_phases))
    phases = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if start <= end else -1
            phases.update(range(start, end + step, step))
        else:
            phases.add(int(part))
    invalid = [phase for phase in phases if phase < 0 or phase >= n_phases]
    if invalid:
        raise SystemExit(f"phase out of range: {sorted(invalid)}")
    return phases


def blend_values(base, candidate, ratio):
    return [
        int(round((1.0 - ratio) * base_value + ratio * candidate_value))
        for base_value, candidate_value in zip(base, candidate)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--phases", type=int, default=60)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--blend-phases", default=None)
    args = parser.parse_args()

    if not (0.0 <= args.ratio <= 1.0):
        raise SystemExit("--ratio must be in [0.0, 1.0]")
    blend_phases = parse_phases(args.blend_phases, args.phases)

    base_dir = Path(args.base_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    total_params = 0
    total_changed = 0
    total_abs_delta = 0
    max_abs_delta = 0

    for phase in range(args.phases):
        base_path = base_dir / f"{phase}.txt"
        candidate_path = candidate_dir / f"{phase}.txt"
        if not base_path.exists():
            raise SystemExit(f"missing base phase file: {base_path}")
        if not candidate_path.exists():
            raise SystemExit(f"missing candidate phase file: {candidate_path}")

        base = read_phase(base_path)
        candidate = read_phase(candidate_path)
        if len(base) != len(candidate):
            raise SystemExit(
                f"phase {phase}: parameter count mismatch "
                f"base={len(base)} candidate={len(candidate)}"
            )

        if phase in blend_phases:
            blended = blend_values(base, candidate, args.ratio)
        else:
            blended = list(base)
        write_phase(output_dir / f"{phase}.txt", blended)

        deltas = [value - base_value for value, base_value in zip(blended, base)]
        changed = sum(1 for delta in deltas if delta != 0)
        abs_delta_sum = sum(abs(delta) for delta in deltas)
        phase_max_abs_delta = max((abs(delta) for delta in deltas), default=0)

        total_params += len(blended)
        total_changed += changed
        total_abs_delta += abs_delta_sum
        max_abs_delta = max(max_abs_delta, phase_max_abs_delta)
        summary_rows.append(
            (
                phase,
                len(blended),
                changed,
                abs_delta_sum / max(1, len(blended)),
                phase_max_abs_delta,
            )
        )

    with (output_dir / "blend_summary.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("phase\tn_params\tn_changed\tavg_abs_delta_from_base\tmax_abs_delta_from_base\n")
        for row in summary_rows:
            f.write("{}\t{}\t{}\t{:.6f}\t{}\n".format(*row))

    with (output_dir / "blend_config.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"base_dir={base_dir.resolve()}\n")
        f.write(f"candidate_dir={candidate_dir.resolve()}\n")
        f.write(f"output_dir={output_dir.resolve()}\n")
        f.write(f"phases={args.phases}\n")
        f.write(f"ratio={args.ratio}\n")
        f.write(f"blend_phases={args.blend_phases or '0-' + str(args.phases - 1)}\n")
        f.write(f"total_params={total_params}\n")
        f.write(f"total_changed={total_changed}\n")
        f.write(f"avg_abs_delta_from_base={total_abs_delta / max(1, total_params):.6f}\n")
        f.write(f"max_abs_delta_from_base={max_abs_delta}\n")

    print(
        "ratio={} total_params={} total_changed={} "
        "avg_abs_delta_from_base={:.6f} max_abs_delta_from_base={}".format(
            args.ratio,
            total_params,
            total_changed,
            total_abs_delta / max(1, total_params),
            max_abs_delta,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
