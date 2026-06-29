import argparse
import datetime as _datetime
import struct
from pathlib import Path


N_PHASES = 60
N_PATTERN_PARAMS_RAW = 612360
MAX_STONE_NUM = 65
N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM
VERSION_LINEAR_FM_INT16_INT8 = 8


def quant_scale(max_abs, quant_max):
    return max_abs / quant_max if max_abs > 0.0 else 1.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def read_phase_values(path, dim):
    values = []
    with path.open("r", encoding="utf-8") as f:
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


def make_phase_payload(path, dim):
    linear, vectors = read_phase_values(path, dim)
    linear_max = max((abs(v) for v in linear), default=0.0)
    vector_max = max((abs(v) for vec in vectors for v in vec), default=0.0)
    linear_scale = quant_scale(linear_max, 32767.0)
    vector_scale = quant_scale(vector_max, 127.0)
    payload = bytearray()
    nz_linear = 0
    nz_vector = 0
    for idx in range(N_PARAMS_PER_PHASE):
        q_linear = 0 if linear_scale == 0.0 else int(round(linear[idx] / linear_scale))
        q_linear = clamp(q_linear, -32767, 32767)
        if q_linear != 0:
            nz_linear += 1
        payload += struct.pack("<h", q_linear)
        for d in range(dim):
            q_vec = 0 if vector_scale == 0.0 else int(round(vectors[d][idx] / vector_scale))
            q_vec = clamp(q_vec, -127, 127)
            if q_vec != 0:
                nz_vector += 1
            payload += struct.pack("b", q_vec)
    return bytes(payload), float(linear_scale), float(vector_scale), nz_linear, nz_vector


def parse_phase_from_name(path):
    stem = path.name
    if not stem.endswith("_fm.txt"):
        return None
    phase_text = stem[:-len("_fm.txt")]
    if not phase_text.isdigit():
        return None
    phase = int(phase_text)
    return phase if 0 <= phase < N_PHASES else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        dest="input_dirs",
        action="append",
        required=True,
        help="directory containing *_fm.txt files; repeat to merge phases, with later directories overriding earlier ones"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument(
        "--fill-missing",
        choices=["zero", "hold-last"],
        default="zero",
        help="how to fill phases without *_fm.txt"
    )
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    input_dirs = [Path(input_dir) for input_dir in args.input_dirs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    phase_files = {}
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*_fm.txt")):
            phase = parse_phase_from_name(path)
            if phase is not None:
                phase_files[phase] = path
    if not phase_files:
        raise RuntimeError(f"no *_fm.txt files found in {', '.join(str(path) for path in input_dirs)}")

    zero_payload = bytes(N_PARAMS_PER_PHASE * (2 + args.dim))
    payloads = [zero_payload for _ in range(N_PHASES)]
    linear_scales = [1.0 for _ in range(N_PHASES)]
    vector_scales = [1.0 for _ in range(N_PHASES)]
    summaries = []
    last_payload = zero_payload
    last_linear_scale = 1.0
    last_vector_scale = 1.0

    for phase in range(N_PHASES):
        if phase in phase_files:
            payload, ls, vs, nz_l, nz_v = make_phase_payload(phase_files[phase], args.dim)
            payloads[phase] = payload
            linear_scales[phase] = ls
            vector_scales[phase] = vs
            last_payload = payload
            last_linear_scale = ls
            last_vector_scale = vs
            summaries.append((phase, "text", nz_l, nz_v, ls, vs))
        elif args.fill_missing == "hold-last":
            payloads[phase] = last_payload
            linear_scales[phase] = last_linear_scale
            vector_scales[phase] = last_vector_scale
            summaries.append((phase, "hold-last", None, None, last_linear_scale, last_vector_scale))
        else:
            summaries.append((phase, "zero", 0, 0, 1.0, 1.0))

    timestamp = args.timestamp or _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(timestamp) != 14 or not timestamp.isdigit():
        raise ValueError("--timestamp must be 14 digits")

    with output.open("wb") as f:
        f.write(timestamp.encode("ascii"))
        f.write(b"EGEV")
        f.write(struct.pack("<4i", VERSION_LINEAR_FM_INT16_INT8, N_PHASES, N_PARAMS_PER_PHASE, args.dim))
        f.write(struct.pack("<60f", *linear_scales))
        f.write(struct.pack("<60f", *vector_scales))
        for payload in payloads:
            f.write(payload)

    print(f"wrote {output}")
    print(f"version {VERSION_LINEAR_FM_INT16_INT8} phases {N_PHASES} params_per_phase {N_PARAMS_PER_PHASE} dim {args.dim}")
    print("input_dirs " + " ".join(str(path) for path in input_dirs))
    print(f"fill_missing {args.fill_missing}")
    for phase, source, nz_l, nz_v, ls, vs in summaries:
        if phase in phase_files or source == "hold-last":
            print(f"phase {phase} source {source} nz_linear {nz_l} nz_vector {nz_v} linear_scale {ls:.10g} vector_scale {vs:.10g}")


if __name__ == "__main__":
    main()
