import argparse
import datetime as _datetime
import struct
from pathlib import Path


TIMESTAMP_SIZE = 14
MAGIC = b"EGEV"
N_PHASES = 60
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
        raise ValueError("at least one phase is required")
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
        raise ValueError(f"{path} is too short")
    if data[TIMESTAMP_SIZE:TIMESTAMP_SIZE + len(MAGIC)] != MAGIC:
        raise ValueError(f"{path} has unexpected magic")
    header_offset = TIMESTAMP_SIZE + len(MAGIC)
    version, n_phases, params_per_phase, dim = struct.unpack_from("<4i", data, header_offset)
    if version != VERSION_LINEAR_FM_INT16_INT8 or n_phases != N_PHASES or params_per_phase <= 0 or dim <= 0:
        raise ValueError(f"{path} has unsupported EGEV4 header")
    scales_offset = header_offset + 16
    linear_scales_offset = scales_offset
    vector_scales_offset = linear_scales_offset + N_PHASES * 4
    payload_offset = vector_scales_offset + N_PHASES * 4
    record_stride = 2 + dim
    phase_stride = params_per_phase * record_stride
    expected_size = payload_offset + n_phases * phase_stride
    if len(data) != expected_size:
        raise ValueError(f"{path} has {len(data)} bytes; expected {expected_size}")
    vector_scales = list(struct.unpack_from("<60f", data, vector_scales_offset))
    return {
        "path": str(path),
        "data": data,
        "version": version,
        "n_phases": n_phases,
        "params_per_phase": params_per_phase,
        "dim": dim,
        "vector_scales": vector_scales,
        "vector_scales_offset": vector_scales_offset,
        "payload_offset": payload_offset,
        "record_stride": record_stride,
        "phase_stride": phase_stride,
    }


def count_nonzero_vectors(data, phase_offset, params_per_phase, dim, record_stride):
    phase_data = data[phase_offset:phase_offset + params_per_phase * record_stride]
    total = 0
    for d in range(dim):
        total += sum(1 for value in phase_data[2 + d::record_stride] if value != 0)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base EGEV4 file")
    parser.add_argument("--overlay", required=True, help="EGEV4 file whose selected phase FM vectors are copied")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    phases = parse_phase_list(args.phases)
    timestamp = parse_timestamp(args.timestamp) if args.timestamp else None
    base = read_egev4(args.base)
    overlay = read_egev4(args.overlay)
    for key in ("version", "n_phases", "params_per_phase", "dim"):
        if base[key] != overlay[key]:
            raise ValueError(f"shape mismatch for {key}: base={base[key]} overlay={overlay[key]}")

    output = bytearray(base["data"])
    if timestamp is not None:
        output[:TIMESTAMP_SIZE] = timestamp.encode("ascii")
    vector_scales = list(base["vector_scales"])
    rows = []
    for phase in phases:
        base_offset = base["payload_offset"] + phase * base["phase_stride"]
        overlay_offset = overlay["payload_offset"] + phase * overlay["phase_stride"]
        before_nz = count_nonzero_vectors(output, base_offset, base["params_per_phase"], base["dim"], base["record_stride"])
        after_nz = count_nonzero_vectors(overlay["data"], overlay_offset, overlay["params_per_phase"], overlay["dim"], overlay["record_stride"])
        phase_data = bytearray(output[base_offset:base_offset + base["phase_stride"]])
        overlay_phase = overlay["data"][overlay_offset:overlay_offset + overlay["phase_stride"]]
        for d in range(base["dim"]):
            start = 2 + d
            phase_data[start::base["record_stride"]] = overlay_phase[start::overlay["record_stride"]]
        output[base_offset:base_offset + base["phase_stride"]] = phase_data
        before_scale = vector_scales[phase]
        vector_scales[phase] = overlay["vector_scales"][phase]
        rows.append((phase, before_scale, vector_scales[phase], before_nz, after_nz))

    offset = base["vector_scales_offset"]
    output[offset:offset + N_PHASES * 4] = struct.pack("<60f", *vector_scales)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)

    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="\n") as f:
            f.write("7.7 beta EGEV4 FM phase merge\n")
            f.write(f"base={args.base}\n")
            f.write(f"overlay={args.overlay}\n")
            f.write(f"output={args.output}\n")
            f.write(f"phases={','.join(str(p) for p in phases)}\n")
            f.write(f"params_per_phase={base['params_per_phase']}\n")
            f.write(f"dim={base['dim']}\n")
            f.write("note=Linear weights are copied from base; selected FM vector bytes and vector_scale values are copied from overlay.\n")
            f.write("phase\tvector_scale_before\tvector_scale_after\tnonzero_vectors_before\tnonzero_vectors_after\n")
            for row in rows:
                f.write(f"{row[0]}\t{row[1]:.10g}\t{row[2]:.10g}\t{row[3]}\t{row[4]}\n")

    print(f"wrote {args.output}")
    print(f"phases {','.join(str(p) for p in phases)}")
    for phase, before_scale, after_scale, before_nz, after_nz in rows:
        print(
            f"phase {phase} vector_scale_before {before_scale:.10g} "
            f"vector_scale_after {after_scale:.10g} "
            f"nonzero_vectors_before {before_nz} nonzero_vectors_after {after_nz}"
        )


if __name__ == "__main__":
    main()
