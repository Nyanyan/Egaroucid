import argparse
import datetime as _datetime
import shutil
import struct
from pathlib import Path


TIMESTAMP_SIZE = 14
MAGIC = b"EGEV"
N_PHASES = 60
VERSION_LINEAR_FM_INT16_INT8 = 8


def parse_phase_list(text):
    if not text:
        return list(range(N_PHASES))
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
    phases = sorted(set(phases))
    for phase in phases:
        if phase < 0 or N_PHASES <= phase:
            raise ValueError(f"phase out of range: {phase}")
    return phases


def parse_timestamp(text):
    if text == "now":
        return _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(text) != TIMESTAMP_SIZE or not text.isdigit():
        raise ValueError("--timestamp must be 14 digits or 'now'")
    return text


def read_egev4(path):
    min_size = TIMESTAMP_SIZE + len(MAGIC) + 16 + N_PHASES * 4 * 2
    input_path = Path(path)
    with input_path.open("rb") as f:
        data = f.read(min_size)
    if len(data) != min_size:
        raise ValueError(f"{path} is too short for an EGEV4 header")

    timestamp = data[:TIMESTAMP_SIZE].decode("ascii")
    magic = data[TIMESTAMP_SIZE:TIMESTAMP_SIZE + len(MAGIC)]
    if magic != MAGIC:
        raise ValueError(f"{path} has unexpected magic: {magic!r}")

    header_offset = TIMESTAMP_SIZE + len(MAGIC)
    version, n_phases, params_per_phase, dim = struct.unpack_from("<4i", data, header_offset)
    if version != VERSION_LINEAR_FM_INT16_INT8:
        raise ValueError(
            f"{path} has unsupported version {version}; expected {VERSION_LINEAR_FM_INT16_INT8}"
        )
    if n_phases != N_PHASES or params_per_phase <= 0:
        raise ValueError(
            f"{path} has unsupported shape: phases={n_phases}, params_per_phase={params_per_phase}"
        )
    if dim <= 0:
        raise ValueError(f"{path} has invalid FM dimension: {dim}")

    scales_offset = header_offset + 16
    linear_scales_offset = scales_offset
    vector_scales_offset = linear_scales_offset + N_PHASES * 4
    payload_offset = vector_scales_offset + N_PHASES * 4
    expected_size = payload_offset + n_phases * params_per_phase * (2 + dim)
    actual_size = input_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path} has {actual_size} bytes; expected {expected_size}")

    linear_scales = list(struct.unpack_from("<60f", data, linear_scales_offset))
    vector_scales = list(struct.unpack_from("<60f", data, vector_scales_offset))
    return {
        "path": input_path,
        "timestamp": timestamp,
        "version": version,
        "n_phases": n_phases,
        "params_per_phase": params_per_phase,
        "dim": dim,
        "linear_scales": linear_scales,
        "vector_scales": vector_scales,
        "vector_scales_offset": vector_scales_offset,
    }


def write_scaled_egev4(input_info, output_path, phases, vector_multiplier, timestamp):
    vector_scales = list(input_info["vector_scales"])
    before_after = []
    for phase in phases:
        before = vector_scales[phase]
        after = before * vector_multiplier
        vector_scales[phase] = after
        before_after.append((phase, before, after))

    offset = input_info["vector_scales_offset"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == input_info["path"].resolve():
        raise ValueError("input and output paths must differ")
    shutil.copyfile(input_info["path"], output_path)
    with output_path.open("r+b") as f:
        if timestamp is not None:
            f.write(timestamp.encode("ascii"))
        f.seek(offset)
        f.write(struct.pack("<60f", *vector_scales))
    return before_after


def write_summary(path, input_path, output_path, input_info, phases, vector_multiplier, timestamp, before_after):
    if not path:
        return
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("EGEV4 FM vector scale experiment\n")
        f.write(f"input={input_path}\n")
        f.write(f"output={output_path}\n")
        f.write(f"input_timestamp={input_info['timestamp']}\n")
        f.write(f"output_timestamp={timestamp or input_info['timestamp']}\n")
        f.write(f"version={input_info['version']}\n")
        f.write(f"n_phases={input_info['n_phases']}\n")
        f.write(f"params_per_phase={input_info['params_per_phase']}\n")
        f.write(f"dim={input_info['dim']}\n")
        f.write(f"phases={','.join(str(phase) for phase in phases)}\n")
        f.write(f"vector_multiplier={vector_multiplier:.10g}\n")
        f.write("note=Only the per-phase FM vector_scale values are multiplied; quantized vectors and linear weights are copied unchanged.\n")
        f.write("phase\tvector_scale_before\tvector_scale_after\n")
        for phase, before, after in before_after:
            f.write(f"{phase}\t{before:.10g}\t{after:.10g}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input EGEV4 file")
    parser.add_argument("--output", required=True, help="scaled output EGEV4 file")
    parser.add_argument(
        "--phases",
        default="",
        help="comma-separated zero-based phases or ranges to scale; empty means all phases",
    )
    parser.add_argument(
        "--vector-multiplier",
        type=float,
        required=True,
        help="multiplier applied to selected per-phase FM vector scales",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="14-digit output timestamp, 'now', or empty to preserve the input timestamp",
    )
    parser.add_argument("--summary", default="", help="optional text summary path")
    args = parser.parse_args()

    if args.vector_multiplier < 0.0:
        raise ValueError("--vector-multiplier must be non-negative")

    phases = parse_phase_list(args.phases)
    timestamp = parse_timestamp(args.timestamp) if args.timestamp else None
    input_info = read_egev4(args.input)
    before_after = write_scaled_egev4(
        input_info,
        args.output,
        phases,
        args.vector_multiplier,
        timestamp,
    )
    write_summary(
        args.summary,
        args.input,
        args.output,
        input_info,
        phases,
        args.vector_multiplier,
        timestamp,
        before_after,
    )

    print(f"wrote {args.output}")
    print(
        f"version {input_info['version']} phases {input_info['n_phases']} "
        f"params_per_phase {input_info['params_per_phase']} dim {input_info['dim']}"
    )
    print(f"scaled_phases {','.join(str(phase) for phase in phases)}")
    print(f"vector_multiplier {args.vector_multiplier:.10g}")
    for phase, before, after in before_after:
        print(f"phase {phase} vector_scale_before {before:.10g} vector_scale_after {after:.10g}")


if __name__ == "__main__":
    main()
