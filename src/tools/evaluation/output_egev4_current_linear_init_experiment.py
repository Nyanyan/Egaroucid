import argparse
import datetime as _datetime
import random
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


def parse_phase_list(text):
    if not text:
        return []
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
                raise ValueError(f"invalid phase range {elem}")
            phases.extend(range(start, end + 1))
        else:
            phases.append(int(elem))
    for phase in phases:
        if phase < 0 or N_PHASES <= phase:
            raise ValueError(f"phase out of range: {phase}")
    return sorted(set(phases))


def parse_phase_from_name(path):
    stem = path.name
    if not stem.endswith("_fm.txt"):
        return None
    phase_text = stem[:-len("_fm.txt")]
    if not phase_text.isdigit():
        return None
    phase = int(phase_text)
    return phase if 0 <= phase < N_PHASES else None


def read_fm_text(path, dim):
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


def quant_scale(max_abs, quant_max):
    return max_abs / quant_max if max_abs > 0.0 else 1.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_text_override_payload(path, dim):
    linear, vectors = read_fm_text(path, dim)
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


def collect_override_files(input_dirs):
    phase_files = {}
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*_fm.txt")):
            phase = parse_phase_from_name(path)
            if phase is not None:
                phase_files[phase] = path
    return phase_files


def write_phase_text(values, phase, dim, text_dir, vector_init, vector_scale, vector_seed):
    start = phase * N_PARAMS_PER_PHASE
    end = start + N_PARAMS_PER_PHASE
    text_dir.mkdir(parents=True, exist_ok=True)
    out_path = text_dir / f"{phase}_fm.txt"
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values[start:end]:
            f.write(f"{value / STEP:.17g}\n")
        if vector_init == "zero":
            zero_block = "0\n" * N_PARAMS_PER_PHASE
            for _ in range(dim):
                f.write(zero_block)
        else:
            rng = random.Random(vector_seed + phase)
            for _ in range(dim):
                for _ in range(N_PARAMS_PER_PHASE):
                    f.write(f"{rng.uniform(-vector_scale, vector_scale):.17g}\n")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="current-model EGEV2 file")
    parser.add_argument("--output", default="", help="output EGEV4 file for the current-model + FM experiment")
    parser.add_argument("--dim", type=int, default=2, help="FM vector dimension; vectors are initialized to zero")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument(
        "--text-dir",
        default="",
        help="optional directory where optimizer input *_fm.txt files are written"
    )
    parser.add_argument(
        "--text-phases",
        default="",
        help="comma-separated phases or ranges to write to --text-dir, for example 2,5-8"
    )
    parser.add_argument(
        "--text-vector-init",
        choices=["zero", "random"],
        default="zero",
        help="how to initialize FM vectors in optimizer input text"
    )
    parser.add_argument(
        "--text-vector-scale",
        type=float,
        default=0.01,
        help="random vector values are sampled uniformly from [-scale, scale]"
    )
    parser.add_argument("--text-vector-seed", type=int, default=20260629)
    parser.add_argument(
        "--override-input-dir",
        dest="override_input_dirs",
        action="append",
        default=[],
        help="optional *_fm.txt directory whose phases override the current-linear baseline in the output EGEV4"
    )
    args = parser.parse_args()

    if args.dim <= 0:
        raise ValueError("--dim must be positive")
    if not args.output and not args.text_dir:
        raise ValueError("at least one of --output or --text-dir is required")
    if args.text_dir and not args.text_phases:
        raise ValueError("--text-phases is required when --text-dir is used")

    values = load_unzip_egev2(args.input)
    expected = N_PHASES * N_PARAMS_PER_PHASE
    if len(values) != expected:
        raise ValueError(f"{args.input} expands to {len(values)} values; expected {expected}")

    timestamp = args.timestamp or _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(timestamp) != 14 or not timestamp.isdigit():
        raise ValueError("--timestamp must be 14 digits")

    linear_scales = [1.0 / STEP for _ in range(N_PHASES)]
    vector_scales = [1.0 for _ in range(N_PHASES)]
    max_value = max(values)
    min_value = min(values)
    n_nonzero = sum(1 for value in values if value != 0)

    if args.text_dir:
        text_dir = Path(args.text_dir)
        phases = parse_phase_list(args.text_phases)
        for phase in phases:
            out_path = write_phase_text(
                values,
                phase,
                args.dim,
                text_dir,
                args.text_vector_init,
                args.text_vector_scale,
                args.text_vector_seed,
            )
            print(f"wrote_text {out_path} vector_init {args.text_vector_init} vector_scale {args.text_vector_scale:.10g}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        phase_payloads = [None for _ in range(N_PHASES)]
        override_dirs = [Path(input_dir) for input_dir in args.override_input_dirs]
        phase_files = collect_override_files(override_dirs)
        override_summaries = []
        for phase, path in sorted(phase_files.items()):
            payload, ls, vs, nz_l, nz_v = make_text_override_payload(path, args.dim)
            phase_payloads[phase] = payload
            linear_scales[phase] = ls
            vector_scales[phase] = vs
            override_summaries.append((phase, path, nz_l, nz_v, ls, vs))

        zero_vector = bytes(args.dim)
        with output.open("wb") as f:
            f.write(timestamp.encode("ascii"))
            f.write(b"EGEV")
            f.write(struct.pack("<4i", VERSION_LINEAR_FM_INT16_INT8, N_PHASES, N_PARAMS_PER_PHASE, args.dim))
            f.write(struct.pack("<60f", *linear_scales))
            f.write(struct.pack("<60f", *vector_scales))
            for phase in range(N_PHASES):
                if phase_payloads[phase] is not None:
                    f.write(phase_payloads[phase])
                    continue
                start = phase * N_PARAMS_PER_PHASE
                end = start + N_PARAMS_PER_PHASE
                for value in values[start:end]:
                    f.write(struct.pack("<h", value))
                    f.write(zero_vector)

        print(f"wrote {output}")
        print(f"version {VERSION_LINEAR_FM_INT16_INT8} phases {N_PHASES} params_per_phase {N_PARAMS_PER_PHASE} dim {args.dim}")
        print(f"linear_scale {1.0 / STEP:.10g} vector_scale 1")
        if override_dirs:
            print("override_input_dirs " + " ".join(str(path) for path in override_dirs))
            for phase, path, nz_l, nz_v, ls, vs in override_summaries:
                print(f"phase {phase} override {path} nz_linear {nz_l} nz_vector {nz_v} linear_scale {ls:.10g} vector_scale {vs:.10g}")
    print(f"expanded_values {len(values)} nonzero {n_nonzero} max {max_value} min {min_value}")


if __name__ == "__main__":
    main()
