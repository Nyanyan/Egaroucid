import argparse
import datetime as _datetime
import struct
from pathlib import Path


N_PHASES = 60
N_PATTERN_PARAMS_RAW = 612360
MAX_STONE_NUM = 65
N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM
N_ZEROS_PLUS = 1 << 12
STEP = 32
VERSION_LINEAR_FM_INT16_INT8 = 8


def load_unzip_egev2(path):
    data = Path(path).read_bytes()
    if len(data) < 4:
        raise ValueError(f"{path} is too short")
    n_compressed = struct.unpack_from("<i", data, 0)[0]
    expected_size = 4 + n_compressed * 2
    if len(data) < expected_size:
        raise ValueError(f"{path} is broken: {len(data)} bytes, expected at least {expected_size}")

    values = []
    offset = 4
    for _ in range(n_compressed):
        elem = struct.unpack_from("<h", data, offset)[0]
        offset += 2
        if elem >= N_ZEROS_PLUS:
            values.extend([0] * (elem - N_ZEROS_PLUS))
        else:
            values.append(elem)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="current-model EGEV2 file")
    parser.add_argument("--output", required=True, help="output EGEV4 file for the current-model + FM experiment")
    parser.add_argument("--dim", type=int, default=2, help="FM vector dimension; vectors are initialized to zero")
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    if args.dim <= 0:
        raise ValueError("--dim must be positive")

    values = load_unzip_egev2(args.input)
    expected = N_PHASES * N_PARAMS_PER_PHASE
    if len(values) != expected:
        raise ValueError(f"{args.input} expands to {len(values)} values; expected {expected}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = args.timestamp or _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(timestamp) != 14 or not timestamp.isdigit():
        raise ValueError("--timestamp must be 14 digits")

    linear_scales = [1.0 / STEP for _ in range(N_PHASES)]
    vector_scales = [1.0 for _ in range(N_PHASES)]
    zero_vector = bytes(args.dim)
    max_value = max(values)
    min_value = min(values)
    n_nonzero = sum(1 for value in values if value != 0)

    with output.open("wb") as f:
        f.write(timestamp.encode("ascii"))
        f.write(b"EGEV")
        f.write(struct.pack("<4i", VERSION_LINEAR_FM_INT16_INT8, N_PHASES, N_PARAMS_PER_PHASE, args.dim))
        f.write(struct.pack("<60f", *linear_scales))
        f.write(struct.pack("<60f", *vector_scales))
        for value in values:
            f.write(struct.pack("<h", value))
            f.write(zero_vector)

    print(f"wrote {output}")
    print(f"version {VERSION_LINEAR_FM_INT16_INT8} phases {N_PHASES} params_per_phase {N_PARAMS_PER_PHASE} dim {args.dim}")
    print(f"linear_scale {1.0 / STEP:.10g} vector_scale 1")
    print(f"expanded_values {len(values)} nonzero {n_nonzero} max {max_value} min {min_value}")


if __name__ == "__main__":
    main()
