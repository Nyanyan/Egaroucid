import argparse
import datetime as _datetime
import struct
from pathlib import Path


N_PHASES = 60
N_EDAX_PLIES = 61
N_PACKED_PARAMS = 114364
N_PARAMS_PER_PHASE = 226315
VERSION_LINEAR_FM_INT16_INT8 = 8
SCORE_STEP = 128

PATTERN_SIZES = [9, 10, 10, 10, 8, 8, 8, 8, 7, 6, 5, 4, 0]
EVAL_SIZES = [3**n for n in PATTERN_SIZES]
PACKED_SIZES = [10206, 29889, 29646, 29646, 3321, 3321, 3321, 3321, 1134, 378, 135, 45, 1]
SYM_S10 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
SYM_C10 = [9, 8, 7, 6, 4, 5, 3, 2, 1, 0]
SYM_C9 = [0, 2, 1, 4, 3, 5, 7, 6, 8]
POW3 = [3**i for i in range(11)]


def opponent_feature(idx, n_digits):
    opponent_digit = [1, 0, 2]
    res = opponent_digit[idx % 3]
    if n_digits > 1:
        res += opponent_feature(idx // 3, n_digits - 1) * 3
    return res


def player_feature(sym, n_digits, idx):
    res = 0
    for i in range(n_digits):
        res += ((idx // POW3[sym[i]]) % 3) * POW3[i]
    return res


def unpack_map(n_digits, size, sym):
    player = [0] * size
    opponent = [0] * size
    n_packed = 0
    for idx in range(size):
        symmetric_idx = player_feature(sym, n_digits, idx)
        if symmetric_idx < idx:
            player[idx] = player[symmetric_idx]
        else:
            player[idx] = n_packed
            n_packed += 1
        opponent[opponent_feature(idx, n_digits)] = player[idx]
    return player, opponent


def build_maps():
    return [
        unpack_map(9, 19683, SYM_C9)[0],
        unpack_map(10, 59049, SYM_C10)[0],
        unpack_map(10, 59049, SYM_S10)[0],
        unpack_map(10, 59049, SYM_S10)[0],
        unpack_map(8, 6561, SYM_S10[2:])[0],
        unpack_map(8, 6561, SYM_S10[2:])[0],
        unpack_map(8, 6561, SYM_S10[2:])[0],
        unpack_map(8, 6561, SYM_S10[2:])[0],
        unpack_map(7, 2187, SYM_S10[3:])[0],
        unpack_map(6, 729, SYM_S10[4:])[0],
        unpack_map(5, 243, SYM_S10[5:])[0],
        unpack_map(4, 81, SYM_S10[6:])[0],
        [0],
    ]


def read_eval_dat(path):
    data = Path(path).read_bytes()
    header_size = 4 * 5 + 8
    expected = header_size + N_EDAX_PLIES * N_PACKED_PARAMS * 2
    if len(data) < expected:
        raise ValueError(f"{path} is too short: {len(data)} bytes, expected at least {expected}")
    version, release, build = struct.unpack_from("<3I", data, 8)
    packed_by_ply = []
    pos = header_size
    for _ply in range(N_EDAX_PLIES):
        packed = list(struct.unpack_from(f"<{N_PACKED_PARAMS}h", data, pos))
        pos += N_PACKED_PARAMS * 2
        packed_by_ply.append(packed)
    return version, release, build, packed_by_ply


def expand_player0(packed, maps):
    raw = []
    packed_offset = 0
    for eval_idx, mapping in enumerate(maps):
        for raw_idx in range(EVAL_SIZES[eval_idx]):
            raw.append(packed[packed_offset + mapping[raw_idx]])
        packed_offset += PACKED_SIZES[eval_idx]
    if len(raw) != N_PARAMS_PER_PHASE:
        raise RuntimeError(f"expanded to {len(raw)} params, expected {N_PARAMS_PER_PHASE}")
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="official Edax eval.dat")
    parser.add_argument("--output", required=True, help="output official-Edax + FM EGEV4 file")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    if args.dim <= 0 or args.dim > 16:
        raise ValueError("--dim must be in 1..16")
    timestamp = args.timestamp or _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(timestamp) != 14 or not timestamp.isdigit():
        raise ValueError("--timestamp must be 14 digits")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    version, release, build, packed_by_ply = read_eval_dat(args.input)
    maps = build_maps()
    zero_vec = bytes(args.dim)
    linear_scales = [1.0 / SCORE_STEP for _ in range(N_PHASES)]
    vector_scales = [1.0 for _ in range(N_PHASES)]
    n_nonzero = 0

    with output.open("wb") as f:
        f.write(timestamp.encode("ascii"))
        f.write(b"EGEV")
        f.write(struct.pack("<4i", VERSION_LINEAR_FM_INT16_INT8, N_PHASES, N_PARAMS_PER_PHASE, args.dim))
        f.write(struct.pack("<60f", *linear_scales))
        f.write(struct.pack("<60f", *vector_scales))
        for phase in range(N_PHASES):
            raw = expand_player0(packed_by_ply[phase], maps)
            payload = bytearray()
            for value in raw:
                n_nonzero += value != 0
                payload += struct.pack("<h", value)
                payload += zero_vec
            f.write(payload)

    print(f"wrote {output}")
    print(f"edax_version {version}.{release}.{build}")
    print(f"version {VERSION_LINEAR_FM_INT16_INT8} phases {N_PHASES} params_per_phase {N_PARAMS_PER_PHASE} dim {args.dim}")
    print(f"linear_scale {1.0 / SCORE_STEP}")
    print(f"nonzero_linear_values {n_nonzero}")


if __name__ == "__main__":
    main()
