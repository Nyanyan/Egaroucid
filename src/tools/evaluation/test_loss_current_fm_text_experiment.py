import argparse
import math
import struct
from pathlib import Path


N_PATTERN_PARAMS_RAW = 612360
MAX_STONE_NUM = 65
N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM
N_FEATURES = 65

# Current 7.5 evaluator layout from evaluation_definition_20241125_1_7_5.hpp.
EVAL_SIZES = [
    3 ** 8,
    3 ** 9,
    3 ** 8,
    3 ** 9,
    3 ** 8,
    3 ** 9,
    3 ** 7,
    3 ** 10,
    3 ** 10,
    3 ** 10,
    3 ** 10,
    3 ** 10,
    3 ** 10,
    3 ** 10,
    3 ** 10,
    3 ** 10,
    MAX_STONE_NUM,
]

FEATURE_TO_EVAL = [idx // 4 for idx in range(64)] + [16]


def make_feature_offsets():
    offsets = []
    start = 0
    last_eval = None
    for eval_idx in FEATURE_TO_EVAL:
        if last_eval is not None and eval_idx > last_eval:
            start += EVAL_SIZES[last_eval]
        offsets.append(start)
        last_eval = eval_idx
    return offsets


FEATURE_OFFSETS = make_feature_offsets()


def read_fm_text(path, dim):
    values = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            values.append(float(line) if line else 0.0)
    expected = N_PARAMS_PER_PHASE * (dim + 1)
    if len(values) != expected:
        raise ValueError(f"{path} has {len(values)} values; expected {expected}")
    linear = values[:N_PARAMS_PER_PHASE]
    vectors = [
        values[N_PARAMS_PER_PHASE * (d + 1):N_PARAMS_PER_PHASE * (d + 2)]
        for d in range(dim)
    ]
    return linear, vectors


def iter_records(paths):
    record_struct = struct.Struct("<hh65Hh")
    for path in paths:
        path = Path(path)
        with path.open("rb") as f:
            while True:
                raw = f.read(record_struct.size)
                if not raw:
                    break
                if len(raw) != record_struct.size:
                    raise ValueError(f"{path} ended with a partial record")
                unpacked = record_struct.unpack(raw)
                features = unpacked[2:2 + N_FEATURES]
                score = float(unpacked[-1])
                active_ids = [
                    FEATURE_OFFSETS[i] + int(features[i])
                    for i in range(N_FEATURES)
                ]
                yield active_ids, score


def predict(linear, vectors, active_ids, dim):
    value = 0.0
    sums = [0.0] * dim
    sums_sq = [0.0] * dim
    for param_id in active_ids:
        value += linear[param_id]
        for d in range(dim):
            vec_value = vectors[d][param_id]
            sums[d] += vec_value
            sums_sq[d] += vec_value * vec_value
    interaction = 0.0
    for d in range(dim):
        interaction += sums[d] * sums[d] - sums_sq[d]
    return value + 0.5 * interaction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm-text", required=True)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("data_files", nargs="+")
    args = parser.parse_args()

    linear, vectors = read_fm_text(args.fm_text, args.dim)
    n_data = 0
    mse = 0.0
    mae = 0.0
    max_abs_error = 0.0
    for active_ids, score in iter_records(args.data_files):
        pred = predict(linear, vectors, active_ids, args.dim)
        err = score - pred
        abs_err = abs(err)
        mse += err * err
        mae += abs_err
        max_abs_error = max(max_abs_error, abs_err)
        n_data += 1
        if args.max_records > 0 and n_data >= args.max_records:
            break

    if n_data == 0:
        raise RuntimeError("no records read")
    mse /= n_data
    mae /= n_data
    rmse = math.sqrt(mse)
    print(
        f"fm_text {args.fm_text} n_data {n_data} "
        f"mse {mse:.8g} rmse {rmse:.8g} mae {mae:.8g} max_abs_error {max_abs_error:.8g}"
    )


if __name__ == "__main__":
    main()
