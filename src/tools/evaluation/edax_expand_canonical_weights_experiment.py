import argparse
from pathlib import Path


PATTERN_SIZES = [9, 10, 10, 10, 8, 8, 8, 8, 7, 6, 5, 4, 0]
EVAL_SIZES = [3**n for n in PATTERN_SIZES]
SYM_S10 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
SYM_C10 = [9, 8, 7, 6, 4, 5, 3, 2, 1, 0]
SYM_C9 = [0, 2, 1, 4, 3, 5, 7, 6, 8]


def player_feature_lsd(sym, idx):
    res = 0
    for out_lsd, src_lsd in enumerate(sym):
        res += ((idx // (3**src_lsd)) % 3) * (3**out_lsd)
    return res


def sym_for_eval(eval_idx):
    if eval_idx == 0:
        return SYM_C9
    if eval_idx == 1:
        return SYM_C10
    if eval_idx in (2, 3):
        return SYM_S10
    if eval_idx in (4, 5, 6, 7):
        return SYM_S10[2:]
    if eval_idx == 8:
        return SYM_S10[3:]
    if eval_idx == 9:
        return SYM_S10[4:]
    if eval_idx == 10:
        return SYM_S10[5:]
    if eval_idx == 11:
        return SYM_S10[6:]
    return None


def build_canonical_maps():
    maps = []
    for eval_idx, size in enumerate(EVAL_SIZES):
        sym = sym_for_eval(eval_idx)
        if sym is None:
            maps.append(list(range(size)))
            continue
        maps.append([min(idx, player_feature_lsd(sym, idx)) for idx in range(size)])
    return maps


def parse_phases(value):
    phases = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if start <= end else -1
            phases.extend(range(start, end + step, step))
        else:
            phases.append(int(part))
    return phases


def read_weights(path, expected_size):
    values = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))
    if len(values) != expected_size:
        raise ValueError(f"{path} has {len(values)} weights; expected {expected_size}")
    return values


def expand_values(values, canonical_maps):
    expanded = []
    offset = 0
    for eval_idx, mapping in enumerate(canonical_maps):
        group = values[offset : offset + EVAL_SIZES[eval_idx]]
        expanded.extend(group[canonical_idx] for canonical_idx in mapping)
        offset += EVAL_SIZES[eval_idx]
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phases", default="0-59")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_size = sum(EVAL_SIZES)
    canonical_maps = build_canonical_maps()

    for phase in parse_phases(args.phases):
        input_path = input_dir / f"{phase}.txt"
        output_path = output_dir / f"{phase}.txt"
        values = read_weights(input_path, expected_size)
        expanded = expand_values(values, canonical_maps)
        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            for value in expanded:
                f.write(f"{value}\n")
        n_changed = sum(1 for a, b in zip(values, expanded) if a != b)
        print(f"phase {phase:02d} expanded {expected_size} weights changed={n_changed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
