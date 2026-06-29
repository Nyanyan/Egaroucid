import argparse
import datetime as _datetime
import struct
from pathlib import Path


N_PHASES = 60
N_PATTERN_PARAMS_RAW = 944784
MAX_STONE_NUM = 65
N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM
VERSION_LINEAR_FM_INT16_INT8 = 8
STEP = 32
N_ZEROS_PLUS = 1 << 12


def iter_unzip_egev2(path):
    data = Path(path).read_bytes()
    if len(data) < 4:
        raise ValueError(f"{path} is too short")
    n_compressed = struct.unpack_from("<i", data, 0)[0]
    expected_size = 4 + n_compressed * 2
    if len(data) < expected_size:
        raise ValueError(f"{path} is broken: {len(data)} bytes, expected at least {expected_size}")

    pos = 4
    for _ in range(n_compressed):
        value = struct.unpack_from("<h", data, pos)[0]
        pos += 2
        if value >= N_ZEROS_PLUS:
            for _ in range(value - N_ZEROS_PLUS):
                yield 0
        else:
            yield value


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="7.7 beta EGEV2 file")
    parser.add_argument("--output", required=True, help="output 7.7 beta + FM EGEV4 file")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    if args.dim <= 0:
        raise ValueError("--dim must be positive")

    expected = N_PHASES * N_PARAMS_PER_PHASE

    timestamp = args.timestamp or _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(timestamp) != 14 or not timestamp.isdigit():
        raise ValueError("--timestamp must be 14 digits")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    linear_scales = [1.0 / STEP for _ in range(N_PHASES)]
    vector_scales = [1.0 for _ in range(N_PHASES)]
    n_nonzero = 0
    n_clamped = 0
    n_values = 0
    min_value = None
    max_value = None

    with output.open("wb") as f:
        f.write(timestamp.encode("ascii"))
        f.write(b"EGEV")
        f.write(struct.pack("<4i", VERSION_LINEAR_FM_INT16_INT8, N_PHASES, N_PARAMS_PER_PHASE, args.dim))
        f.write(struct.pack("<60f", *linear_scales))
        f.write(struct.pack("<60f", *vector_scales))
        zero_vector = bytes(args.dim)
        for value in iter_unzip_egev2(args.input):
            q_value = int(value)
            min_value = q_value if min_value is None else min(min_value, q_value)
            max_value = q_value if max_value is None else max(max_value, q_value)
            clamped = clamp(q_value, -32767, 32767)
            if clamped != q_value:
                n_clamped += 1
            if clamped != 0:
                n_nonzero += 1
            f.write(struct.pack("<h", clamped))
            f.write(zero_vector)
            n_values += 1

    if n_values != expected:
        output.unlink(missing_ok=True)
        raise ValueError(f"{args.input} expands to {n_values} values; expected {expected}")

    print(f"wrote {output}")
    print(f"version {VERSION_LINEAR_FM_INT16_INT8} phases {N_PHASES} params_per_phase {N_PARAMS_PER_PHASE} dim {args.dim}")
    print(f"linear_scale {1.0 / STEP}")
    print("vector_scale 1")
    print(f"expanded_values {n_values} nonzero {n_nonzero} max {max_value} min {min_value} clamped {n_clamped}")


if __name__ == "__main__":
    main()
