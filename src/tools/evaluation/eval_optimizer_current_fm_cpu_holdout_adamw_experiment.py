import argparse
from array import array
import copy
import datetime as _datetime
import math
import random
import struct
import time
from pathlib import Path


N_PHASES = 60
N_PATTERN_FEATURES = 64
N_FEATURES = 65
N_PATTERN_PARAMS_RAW = 612360
MAX_STONE_NUM = 65
N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM
STEP = 32
N_ZEROS_PLUS = 1 << 12
VERSION_LINEAR_FM_INT16_INT8 = 8

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

FEATURE_TO_EVAL = [idx // 4 for idx in range(N_PATTERN_FEATURES)] + [16]
RECORD = struct.Struct("<hh65Hh")


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


def parse_ids(text):
    if not text:
        return []
    return [int(elem.strip()) for elem in text.split(",") if elem.strip()]


def parse_phases(text):
    phases = parse_ids(text)
    if not phases:
        raise ValueError("phase list is empty")
    return phases


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


def read_records(data_root, phase, file_ids, max_records_per_file):
    records = []
    data_root = Path(data_root)
    for file_id in file_ids:
        path = data_root / str(phase) / f"{file_id}.dat"
        if not path.exists():
            raise FileNotFoundError(path)
        n_file_records = 0
        with path.open("rb") as f:
            while True:
                raw = f.read(RECORD.size)
                if not raw:
                    break
                if len(raw) != RECORD.size:
                    raise ValueError(f"{path} ended with a partial record")
                unpacked = RECORD.unpack(raw)
                idxes = unpacked[2:2 + N_FEATURES]
                score = float(unpacked[-1])
                active_ids = tuple(FEATURE_OFFSETS[i] + int(idxes[i]) for i in range(N_FEATURES))
                records.append((active_ids, score))
                n_file_records += 1
                if max_records_per_file > 0 and n_file_records >= max_records_per_file:
                    break
    return records


def predict(linear_phase, vectors, active_ids, dim):
    value = 0.0
    sums = [0.0] * dim
    sums_sq = [0.0] * dim
    for param_id in active_ids:
        value += linear_phase[param_id]
        vec = vectors.get(param_id)
        if vec is None:
            continue
        for d in range(dim):
            vec_value = vec[d]
            sums[d] += vec_value
            sums_sq[d] += vec_value * vec_value
    interaction = 0.0
    for d in range(dim):
        interaction += sums[d] * sums[d] - sums_sq[d]
    return value + 0.5 * interaction, sums


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


def train_vectors(linear_phase, train_records, holdout_records, args):
    rng = random.Random(args.seed + args.phase)
    vectors = {}
    m = {}
    v = {}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    step = 0
    best_vectors = {}
    best_epoch = 0
    best_holdout_mse, best_holdout_mae = metrics(linear_phase, vectors, holdout_records, args.dim)
    initial_train_mse, initial_train_mae = metrics(linear_phase, vectors, train_records, args.dim)
    history = [
        {
            "epoch": 0,
            "train_mse": initial_train_mse,
            "train_mae": initial_train_mae,
            "holdout_mse": best_holdout_mse,
            "holdout_mae": best_holdout_mae,
            "n_vectors": 0,
        }
    ]

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_records)))
        rng.shuffle(order)
        for rec_idx in order:
            active_ids, score = train_records[rec_idx]
            ensure_vectors(active_ids, vectors, m, v, args.dim, rng, args.init_scale)
            pred, sums = predict(linear_phase, vectors, active_ids, args.dim)
            residual = score - pred
            if args.residual_clip > 0.0:
                residual = clamp(residual, -args.residual_clip, args.residual_clip)
            step += 1
            lr_t = args.lr * math.sqrt(1.0 - beta2 ** step) / (1.0 - beta1 ** step)
            for param_id in active_ids:
                vec = vectors[param_id]
                mom = m[param_id]
                vel = v[param_id]
                for d in range(args.dim):
                    grad = -residual * (sums[d] - vec[d]) + args.l2 * vec[d]
                    mom[d] = beta1 * mom[d] + (1.0 - beta1) * grad
                    vel[d] = beta2 * vel[d] + (1.0 - beta2) * grad * grad
                    vec[d] -= lr_t * mom[d] / (math.sqrt(vel[d]) + eps)
                    if args.weight_decay > 0.0:
                        decay = 1.0 - args.lr * args.weight_decay
                        vec[d] *= decay if decay > 0.0 else 0.0

        train_mse, train_mae = metrics(linear_phase, vectors, train_records, args.dim)
        holdout_mse, holdout_mae = metrics(linear_phase, vectors, holdout_records, args.dim)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "holdout_mse": holdout_mse,
                "holdout_mae": holdout_mae,
                "n_vectors": len(vectors),
            }
        )
        if holdout_mae < best_holdout_mae:
            best_holdout_mse = holdout_mse
            best_holdout_mae = holdout_mae
            best_epoch = epoch
            best_vectors = copy.deepcopy(vectors)
        elif args.early_stop_patience > 0 and epoch - best_epoch >= args.early_stop_patience:
            break

    return best_vectors, best_epoch, history


def vector_scale_for(vectors):
    max_abs = 0.0
    for vec in vectors.values():
        for value in vec:
            max_abs = max(max_abs, abs(value))
    return max_abs / 127.0 if max_abs > 0.0 else 1.0, max_abs


def write_egev4(output, timestamp, linear_values, phase_vectors, dim):
    vector_scales = [1.0 for _ in range(N_PHASES)]
    phase_stats = {}
    for phase, vectors in phase_vectors.items():
        vector_scale, vector_max_abs = vector_scale_for(vectors)
        vector_scales[phase] = vector_scale
        phase_stats[phase] = {
            "vector_scale": vector_scale,
            "vector_max_abs": vector_max_abs,
            "nonzero_vector_values": 0,
        }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(timestamp.encode("ascii"))
        f.write(b"EGEV")
        f.write(struct.pack("<4i", VERSION_LINEAR_FM_INT16_INT8, N_PHASES, N_PARAMS_PER_PHASE, dim))
        f.write(struct.pack("<60f", *([1.0 / STEP] * N_PHASES)))
        f.write(struct.pack("<60f", *vector_scales))
        for phase_idx in range(N_PHASES):
            base = phase_idx * N_PARAMS_PER_PHASE
            vectors = phase_vectors.get(phase_idx, {})
            phase_vector_scale = vector_scales[phase_idx]
            for param_id in range(N_PARAMS_PER_PHASE):
                q_linear = clamp(int(linear_values[base + param_id]), -32767, 32767)
                f.write(struct.pack("<h", q_linear))
                vec = vectors.get(param_id)
                for d in range(dim):
                    value = 0.0 if vec is None else vec[d]
                    q_vec = int(round(value / phase_vector_scale)) if phase_vector_scale != 0.0 else 0
                    q_vec = clamp(q_vec, -127, 127)
                    if phase_idx in phase_stats and q_vec != 0:
                        phase_stats[phase_idx]["nonzero_vector_values"] += 1
                    f.write(struct.pack("b", q_vec))
    return phase_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-egev2", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--phases", default=None)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--holdout-ids", required=True)
    parser.add_argument("--output-egev4", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--init-scale", type=float, default=0.005)
    parser.add_argument("--residual-clip", type=float, default=16.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="AdamW-style decoupled weight decay applied after each active-vector update",
    )
    parser.add_argument("--max-records-per-file", type=int, default=20000)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    if args.dim <= 0:
        raise ValueError("--dim must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.phase is None and args.phases is None:
        raise ValueError("either --phase or --phases is required")
    if args.phase is not None and args.phases is not None:
        raise ValueError("use only one of --phase or --phases")

    phases = parse_phases(args.phases) if args.phases is not None else [args.phase]
    if len(set(phases)) != len(phases):
        raise ValueError("duplicate phase in phase list")
    for phase in phases:
        if not (0 <= phase < N_PHASES):
            raise ValueError(f"phase {phase} out of range")

    linear_values = load_unzip_egev2(args.input_egev2)
    expected = N_PHASES * N_PARAMS_PER_PHASE
    if len(linear_values) != expected:
        raise ValueError(f"{args.input_egev2} expands to {len(linear_values)} values; expected {expected}")

    train_ids = parse_ids(args.train_ids)
    holdout_ids = parse_ids(args.holdout_ids)
    if not train_ids or not holdout_ids:
        raise ValueError("--train-ids and --holdout-ids are required")

    start = time.time()
    phase_results = {}
    phase_vectors = {}
    for phase in phases:
        phase_args = copy.copy(args)
        phase_args.phase = phase
        linear_phase = [
            linear_values[phase * N_PARAMS_PER_PHASE + param_id] / STEP
            for param_id in range(N_PARAMS_PER_PHASE)
        ]
        train_records = read_records(args.data_root, phase, train_ids, args.max_records_per_file)
        holdout_records = read_records(args.data_root, phase, holdout_ids, args.max_records_per_file)
        vectors, best_epoch, history = train_vectors(linear_phase, train_records, holdout_records, phase_args)
        phase_vectors[phase] = vectors
        phase_results[phase] = {
            "best_epoch": best_epoch,
            "history": history,
            "n_train": len(train_records),
            "n_holdout": len(holdout_records),
        }

    timestamp = args.timestamp or _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(timestamp) != 14 or not timestamp.isdigit():
        raise ValueError("--timestamp must be 14 digits")
    vector_stats = write_egev4(args.output_egev4, timestamp, linear_values, phase_vectors, args.dim)
    elapsed_ms = int((time.time() - start) * 1000)

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8", newline="\n") as f:
        if len(phases) == 1:
            f.write(f"phase={phases[0]}\n")
        f.write(f"phases={','.join(str(elem) for elem in phases)}\n")
        f.write(f"train_ids={','.join(str(elem) for elem in train_ids)}\n")
        f.write(f"holdout_ids={','.join(str(elem) for elem in holdout_ids)}\n")
        f.write(f"max_records_per_file={args.max_records_per_file}\n")
        f.write(f"dim={args.dim}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"init_scale={args.init_scale}\n")
        f.write(f"residual_clip={args.residual_clip}\n")
        f.write(f"l2={args.l2}\n")
        f.write(f"weight_decay={args.weight_decay}\n")
        f.write(f"early_stop_patience={args.early_stop_patience}\n")
        f.write(f"total_n_train={sum(phase_results[phase]['n_train'] for phase in phases)}\n")
        f.write(f"total_n_holdout={sum(phase_results[phase]['n_holdout'] for phase in phases)}\n")
        f.write(f"elapsed_ms={elapsed_ms}\n")
        for phase in phases:
            result = phase_results[phase]
            stats = vector_stats[phase]
            f.write(f"\n[phase {phase}]\n")
            f.write(f"n_train={result['n_train']}\n")
            f.write(f"n_holdout={result['n_holdout']}\n")
            f.write(f"best_epoch={result['best_epoch']}\n")
            f.write(f"vector_scale={stats['vector_scale']:.10g}\n")
            f.write(f"vector_max_abs={stats['vector_max_abs']:.10g}\n")
            f.write(f"nonzero_vector_values={stats['nonzero_vector_values']}\n")
            f.write("epoch\ttrain_mse\ttrain_mae\tholdout_mse\tholdout_mae\tn_vectors\n")
            for row in result["history"]:
                f.write(
                    f"{row['epoch']}\t{row['train_mse']:.10g}\t{row['train_mae']:.10g}\t"
                    f"{row['holdout_mse']:.10g}\t{row['holdout_mae']:.10g}\t{row['n_vectors']}\n"
                )

    for phase in phases:
        result = phase_results[phase]
        best = result["history"][result["best_epoch"]]
        stats = vector_stats[phase]
        print(
            f"phase {phase} n_train {result['n_train']} n_holdout {result['n_holdout']} "
            f"best_epoch {result['best_epoch']} best_holdout_MAE {best['holdout_mae']:.8g} "
            f"nonzero_vector_values {stats['nonzero_vector_values']}"
        )
    print(f"elapsed_ms {elapsed_ms}")
    print(f"wrote {args.output_egev4}")
    print(f"summary {summary}")


if __name__ == "__main__":
    main()
