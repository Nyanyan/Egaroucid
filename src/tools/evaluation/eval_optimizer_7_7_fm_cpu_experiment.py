import argparse
from array import array
import datetime as _datetime
import math
import random
import struct
import time
from pathlib import Path


N_PHASES = 60
N_PATTERN_FEATURES = 64
N_FEATURES = 65
N_PATTERN_PARAMS_RAW = 944784
N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + 65
STEP = 32
VERSION_LINEAR_FM_INT16_INT8 = 8
N_ZEROS_PLUS = 1 << 12

PATTERN_OFFSETS = [
    0, 59049, 118098, 177147,
    236196, 295245, 354294, 413343,
    472392, 531441, 590490, 649539,
    708588, 767637, 826686, 885735,
]

FEATURE_TO_PATTERN = [i // 4 for i in range(N_PATTERN_FEATURES)]
if len(PATTERN_OFFSETS) != 16 or len(FEATURE_TO_PATTERN) != N_PATTERN_FEATURES:
    raise RuntimeError("7.7-FM optimizer constants are inconsistent")

RECORD = struct.Struct("<hh65Hh")


def parse_phase_list(text):
    phases = []
    for elem in text.split(","):
        elem = elem.strip()
        if not elem:
            continue
        if "-" in elem:
            start_text, end_text = elem.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid phase range: {elem}")
            phases.extend(range(start, end + 1))
        else:
            phases.append(int(elem))
    for phase in phases:
        if phase < 0 or N_PHASES <= phase:
            raise ValueError(f"phase out of range: {phase}")
    return sorted(set(phases))


def load_unzip_egev2(path):
    data = Path(path).read_bytes()
    if len(data) < 4:
        raise ValueError(f"{path} is too short")
    n_compressed = struct.unpack_from("<i", data, 0)[0]
    expected_size = 4 + n_compressed * 2
    if len(data) < expected_size:
        raise ValueError(f"{path} is broken: {len(data)} bytes, expected at least {expected_size}")

    out = array("h")
    pos = 4
    for _ in range(n_compressed):
        value = struct.unpack_from("<h", data, pos)[0]
        pos += 2
        if value >= N_ZEROS_PLUS:
            out.extend([0] * (value - N_ZEROS_PLUS))
        else:
            out.append(value)
    return out


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def read_phase_records(train_root, phase, max_records):
    phase_dir = Path(train_root) / str(phase)
    if not phase_dir.exists():
        return []
    records = []
    for path in sorted(phase_dir.glob("*.dat")):
        data = path.read_bytes()
        usable = len(data) - (len(data) % RECORD.size)
        for pos in range(0, usable, RECORD.size):
            unpacked = RECORD.unpack_from(data, pos)
            idxes = unpacked[2:2 + N_FEATURES]
            score = float(unpacked[-1])
            pattern_ids = tuple(
                PATTERN_OFFSETS[FEATURE_TO_PATTERN[i]] + int(idxes[i])
                for i in range(N_PATTERN_FEATURES)
            )
            active_ids = pattern_ids + (N_PATTERN_PARAMS_RAW + int(idxes[N_PATTERN_FEATURES]),)
            records.append((active_ids, score))
            if max_records > 0 and len(records) >= max_records:
                return records
    return records


def split_records(records, validation_ratio):
    if len(records) < 10 or validation_ratio <= 0.0:
        return records, records
    n_val = max(1, int(round(len(records) * validation_ratio)))
    n_val = min(n_val, len(records) - 1)
    return records[:-n_val], records[-n_val:]


def predict(linear_phase, vectors, active_ids, dim):
    linear = 0.0
    sums = [0.0] * dim
    sums_sq = [0.0] * dim
    for param_id in active_ids:
        linear += linear_phase[param_id]
        vec = vectors.get(param_id)
        if vec is None:
            continue
        for d in range(dim):
            value = vec[d]
            sums[d] += value
            sums_sq[d] += value * value
    interaction = 0.0
    for d in range(dim):
        interaction += sums[d] * sums[d] - sums_sq[d]
    return linear + 0.5 * interaction, sums


def metrics(linear_phase, vectors, records, dim):
    if not records:
        return 0.0, 0.0
    mse = 0.0
    mae = 0.0
    for active_ids, score in records:
        pred, _ = predict(linear_phase, vectors, active_ids, dim)
        err = score - pred
        mse += err * err
        mae += abs(err)
    return mse / len(records), mae / len(records)


def ensure_vectors(active_ids, vectors, m, v, dim, rng, init_scale):
    for param_id in active_ids:
        if param_id not in vectors:
            vectors[param_id] = [rng.uniform(-init_scale, init_scale) for _ in range(dim)]
            m[param_id] = [0.0] * dim
            v[param_id] = [0.0] * dim


def train_phase(linear_phase, records, dim, epochs, lr, init_scale, residual_clip, l2, seed):
    rng = random.Random(seed)
    train_records, val_records = split_records(records, 0.1)
    vectors = {}
    m = {}
    v = {}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    step = 0

    initial_train_mse, initial_train_mae = metrics(linear_phase, vectors, train_records, dim)
    initial_val_mse, initial_val_mae = metrics(linear_phase, vectors, val_records, dim)

    for _ in range(epochs):
        order = list(range(len(train_records)))
        rng.shuffle(order)
        for rec_idx in order:
            active_ids, score = train_records[rec_idx]
            ensure_vectors(active_ids, vectors, m, v, dim, rng, init_scale)
            pred, sums = predict(linear_phase, vectors, active_ids, dim)
            residual = score - pred
            if residual_clip > 0.0:
                residual = clamp(residual, -residual_clip, residual_clip)
            step += 1
            lr_t = lr * math.sqrt(1.0 - beta2 ** step) / (1.0 - beta1 ** step)
            for param_id in active_ids:
                vec = vectors[param_id]
                mom = m[param_id]
                vel = v[param_id]
                for d in range(dim):
                    grad = -residual * (sums[d] - vec[d]) + l2 * vec[d]
                    mom[d] = beta1 * mom[d] + (1.0 - beta1) * grad
                    vel[d] = beta2 * vel[d] + (1.0 - beta2) * grad * grad
                    vec[d] -= lr_t * mom[d] / (math.sqrt(vel[d]) + eps)

    final_train_mse, final_train_mae = metrics(linear_phase, vectors, train_records, dim)
    final_val_mse, final_val_mae = metrics(linear_phase, vectors, val_records, dim)
    return {
        "vectors": vectors,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "initial_train_mse": initial_train_mse,
        "initial_train_mae": initial_train_mae,
        "initial_val_mse": initial_val_mse,
        "initial_val_mae": initial_val_mae,
        "final_train_mse": final_train_mse,
        "final_train_mae": final_train_mae,
        "final_val_mse": final_val_mse,
        "final_val_mae": final_val_mae,
    }


def vector_scale_for(vectors):
    max_abs = 0.0
    for vec in vectors.values():
        for value in vec:
            max_abs = max(max_abs, abs(value))
    return max_abs / 127.0 if max_abs > 0.0 else 1.0, max_abs


def write_egev4(output, timestamp, linear_values, trained_by_phase, dim):
    linear_scales = [1.0 / STEP for _ in range(N_PHASES)]
    vector_scales = []
    phase_vector_stats = {}
    for phase in range(N_PHASES):
        vectors = trained_by_phase.get(phase, {}).get("vectors", {})
        scale, max_abs = vector_scale_for(vectors)
        vector_scales.append(scale)
        phase_vector_stats[phase] = {
            "vector_scale": scale,
            "vector_max_abs": max_abs,
            "nonzero_vector_values": 0,
        }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(timestamp.encode("ascii"))
        f.write(b"EGEV")
        f.write(struct.pack("<4i", VERSION_LINEAR_FM_INT16_INT8, N_PHASES, N_PARAMS_PER_PHASE, dim))
        f.write(struct.pack("<60f", *linear_scales))
        f.write(struct.pack("<60f", *vector_scales))
        for phase in range(N_PHASES):
            base = phase * N_PARAMS_PER_PHASE
            vectors = trained_by_phase.get(phase, {}).get("vectors", {})
            vector_scale = vector_scales[phase]
            nz_vector = 0
            for param_id in range(N_PARAMS_PER_PHASE):
                q_linear = clamp(int(linear_values[base + param_id]), -32767, 32767)
                f.write(struct.pack("<h", q_linear))
                vec = vectors.get(param_id)
                for d in range(dim):
                    value = 0.0 if vec is None else vec[d]
                    q_vec = int(round(value / vector_scale)) if vector_scale != 0.0 else 0
                    q_vec = clamp(q_vec, -127, 127)
                    if q_vec != 0:
                        nz_vector += 1
                    f.write(struct.pack("b", q_vec))
            phase_vector_stats[phase]["nonzero_vector_values"] = nz_vector
    return phase_vector_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-egev2", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output-egev4", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--phases", default="2-10")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--residual-clip", type=float, default=16.0)
    parser.add_argument("--l2", type=float, default=0.00001)
    parser.add_argument("--max-records-per-phase", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    if args.dim <= 0:
        raise ValueError("--dim must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    phases = parse_phase_list(args.phases)
    linear_values = load_unzip_egev2(args.input_egev2)
    expected = N_PHASES * N_PARAMS_PER_PHASE
    if len(linear_values) != expected:
        raise ValueError(f"{args.input_egev2} expands to {len(linear_values)} values; expected {expected}")
    timestamp = args.timestamp or _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(timestamp) != 14 or not timestamp.isdigit():
        raise ValueError("--timestamp must be 14 digits")

    trained_by_phase = {}
    rows = []
    start_time = time.time()
    for phase in phases:
        phase_start = time.time()
        records = read_phase_records(args.train_root, phase, args.max_records_per_phase)
        if not records:
            rows.append([phase, 0, 0, "no-data", 0, 0, 0, 0, 0, 0, 1.0, 0, 0])
            print(f"phase {phase} no data")
            continue
        result = train_phase(
            [
                linear_values[phase * N_PARAMS_PER_PHASE + param_id] / STEP
                for param_id in range(N_PARAMS_PER_PHASE)
            ],
            records,
            args.dim,
            args.epochs,
            args.lr,
            args.init_scale,
            args.residual_clip,
            args.l2,
            args.seed + phase,
        )
        trained_by_phase[phase] = result
        elapsed_ms = int((time.time() - phase_start) * 1000)
        print(
            "phase {phase} time {time_ms} ms n_train {n_train} n_val {n_val} "
            "initial_val_MAE {initial_val_mae:.6g} final_val_MAE {final_val_mae:.6g}".format(
                phase=phase,
                time_ms=elapsed_ms,
                **result,
            )
        )

    vector_stats = write_egev4(args.output_egev4, timestamp, linear_values, trained_by_phase, args.dim)
    total_elapsed_ms = int((time.time() - start_time) * 1000)

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8", newline="\n") as f:
        f.write(
            "phase\tn_train\tn_val\tinitial_train_mse\tinitial_train_mae\tinitial_val_mse\tinitial_val_mae\t"
            "final_train_mse\tfinal_train_mae\tfinal_val_mse\tfinal_val_mae\tvector_scale\t"
            "vector_max_abs\tnonzero_vector_values\n"
        )
        for phase in phases:
            result = trained_by_phase.get(phase)
            stats = vector_stats[phase]
            if result is None:
                f.write(f"{phase}\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\n")
                continue
            f.write(
                f"{phase}\t{result['n_train']}\t{result['n_val']}\t"
                f"{result['initial_train_mse']:.10g}\t{result['initial_train_mae']:.10g}\t"
                f"{result['initial_val_mse']:.10g}\t{result['initial_val_mae']:.10g}\t"
                f"{result['final_train_mse']:.10g}\t{result['final_train_mae']:.10g}\t"
                f"{result['final_val_mse']:.10g}\t{result['final_val_mae']:.10g}\t"
                f"{stats['vector_scale']:.10g}\t{stats['vector_max_abs']:.10g}\t"
                f"{stats['nonzero_vector_values']}\n"
            )

    print(f"wrote {args.output_egev4}")
    print(f"summary {summary}")
    print(f"total_time_ms {total_elapsed_ms}")


if __name__ == "__main__":
    main()
