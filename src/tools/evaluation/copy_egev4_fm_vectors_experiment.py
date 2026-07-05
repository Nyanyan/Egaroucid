import argparse
import datetime as _datetime
import struct
from pathlib import Path


TIMESTAMP_SIZE = 14
MAGIC = b"EGEV"
N_PHASES = 60
N_PARAMS_PER_PHASE = 612425
VERSION_LINEAR_FM_INT16_INT8 = 8


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
    phases = sorted(set(phases))
    if not phases:
        raise ValueError("at least one target phase is required")
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
    data = Path(path).read_bytes()
    min_size = TIMESTAMP_SIZE + len(MAGIC) + 16 + N_PHASES * 4 * 2
    if len(data) < min_size:
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
    if n_phases != N_PHASES or params_per_phase != N_PARAMS_PER_PHASE:
        raise ValueError(
            f"{path} has unsupported shape: phases={n_phases}, params_per_phase={params_per_phase}"
        )
    if dim <= 0:
        raise ValueError(f"{path} has invalid FM dimension: {dim}")

    scales_offset = header_offset + 16
    linear_scales_offset = scales_offset
    vector_scales_offset = linear_scales_offset + N_PHASES * 4
    payload_offset = vector_scales_offset + N_PHASES * 4
    phase_stride = params_per_phase * (2 + dim)
    expected_size = payload_offset + n_phases * phase_stride
    if len(data) != expected_size:
        raise ValueError(f"{path} has {len(data)} bytes; expected {expected_size}")

    vector_scales = list(struct.unpack_from("<60f", data, vector_scales_offset))
    return {
        "data": data,
        "timestamp": timestamp,
        "version": version,
        "n_phases": n_phases,
        "params_per_phase": params_per_phase,
        "dim": dim,
        "vector_scales": vector_scales,
        "vector_scales_offset": vector_scales_offset,
        "payload_offset": payload_offset,
        "phase_stride": phase_stride,
        "record_stride": 2 + dim,
    }


def count_nonzero_vector_bytes(data, phase_offset, params_per_phase, dim, record_stride):
    total = 0
    phase_end = phase_offset + params_per_phase * record_stride
    phase_data = data[phase_offset:phase_end]
    for vector_dim in range(dim):
        total += sum(1 for value in phase_data[2 + vector_dim::record_stride] if value != 0)
    return total


def copy_fm_vectors(input_info, output_path, source_phase, target_phases, timestamp):
    if source_phase < 0 or N_PHASES <= source_phase:
        raise ValueError(f"source phase out of range: {source_phase}")

    output = bytearray(input_info["data"])
    if timestamp is not None:
        output[:TIMESTAMP_SIZE] = timestamp.encode("ascii")

    dim = input_info["dim"]
    record_stride = input_info["record_stride"]
    params_per_phase = input_info["params_per_phase"]
    phase_stride = input_info["phase_stride"]
    payload_offset = input_info["payload_offset"]
    source_offset = payload_offset + source_phase * phase_stride
    source_phase_data = input_info["data"][source_offset:source_offset + phase_stride]
    source_nz = count_nonzero_vector_bytes(
        input_info["data"], source_offset, params_per_phase, dim, record_stride
    )

    vector_scales = list(input_info["vector_scales"])
    source_scale = vector_scales[source_phase]
    summaries = []
    for phase in target_phases:
        target_offset = payload_offset + phase * phase_stride
        before_scale = vector_scales[phase]
        before_nz = count_nonzero_vector_bytes(
            output, target_offset, params_per_phase, dim, record_stride
        )
        target_phase_data = output[target_offset:target_offset + phase_stride]
        for vector_dim in range(dim):
            start = 2 + vector_dim
            target_phase_data[start::record_stride] = source_phase_data[start::record_stride]
        output[target_offset:target_offset + phase_stride] = target_phase_data
        vector_scales[phase] = source_scale
        summaries.append((phase, before_scale, source_scale, before_nz, source_nz))

    offset = input_info["vector_scales_offset"]
    output[offset:offset + N_PHASES * 4] = struct.pack("<60f", *vector_scales)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    return source_scale, source_nz, summaries


def write_summary(path, input_path, output_path, input_info, source_phase, target_phases, timestamp, source_scale, source_nz, summaries):
    if not path:
        return
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("EGEV4 FM vector copy experiment\n")
        f.write(f"input={input_path}\n")
        f.write(f"output={output_path}\n")
        f.write(f"input_timestamp={input_info['timestamp']}\n")
        f.write(f"output_timestamp={timestamp or input_info['timestamp']}\n")
        f.write(f"version={input_info['version']}\n")
        f.write(f"n_phases={input_info['n_phases']}\n")
        f.write(f"params_per_phase={input_info['params_per_phase']}\n")
        f.write(f"dim={input_info['dim']}\n")
        f.write(f"source_phase={source_phase}\n")
        f.write(f"source_vector_scale={source_scale:.10g}\n")
        f.write(f"source_nonzero_vector_bytes={source_nz}\n")
        f.write(f"target_phases={','.join(str(phase) for phase in target_phases)}\n")
        f.write("note=Only FM vector bytes and vector_scale are copied. Linear weights and linear_scale stay unchanged in every phase.\n")
        f.write("phase\tvector_scale_before\tvector_scale_after\tnonzero_vector_bytes_before\tnonzero_vector_bytes_after\n")
        for phase, before_scale, after_scale, before_nz, after_nz in summaries:
            f.write(f"{phase}\t{before_scale:.10g}\t{after_scale:.10g}\t{before_nz}\t{after_nz}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input EGEV4 file")
    parser.add_argument("--output", required=True, help="output EGEV4 file")
    parser.add_argument("--source-phase", type=int, required=True, help="phase whose FM vectors are copied")
    parser.add_argument(
        "--target-phases",
        required=True,
        help="comma-separated zero-based target phases or ranges, for example 11-59",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="14-digit output timestamp, 'now', or empty to preserve the input timestamp",
    )
    parser.add_argument("--summary", default="", help="optional text summary path")
    args = parser.parse_args()

    target_phases = parse_phase_list(args.target_phases)
    timestamp = parse_timestamp(args.timestamp) if args.timestamp else None
    input_info = read_egev4(args.input)
    source_scale, source_nz, summaries = copy_fm_vectors(
        input_info,
        args.output,
        args.source_phase,
        target_phases,
        timestamp,
    )
    write_summary(
        args.summary,
        args.input,
        args.output,
        input_info,
        args.source_phase,
        target_phases,
        timestamp,
        source_scale,
        source_nz,
        summaries,
    )

    print(f"wrote {args.output}")
    print(
        f"version {input_info['version']} phases {input_info['n_phases']} "
        f"params_per_phase {input_info['params_per_phase']} dim {input_info['dim']}"
    )
    print(f"source_phase {args.source_phase}")
    print(f"source_vector_scale {source_scale:.10g}")
    print(f"source_nonzero_vector_bytes {source_nz}")
    print(f"target_phases {','.join(str(phase) for phase in target_phases)}")
    for phase, before_scale, after_scale, before_nz, after_nz in summaries:
        print(
            f"phase {phase} vector_scale_before {before_scale:.10g} "
            f"vector_scale_after {after_scale:.10g} "
            f"nonzero_vector_bytes_before {before_nz} "
            f"nonzero_vector_bytes_after {after_nz}"
        )


if __name__ == "__main__":
    main()
