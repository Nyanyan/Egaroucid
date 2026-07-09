import argparse
import datetime as _datetime
import random
import struct
from pathlib import Path


N_PHASES = 60
N_PARAMS_PER_PHASE = 226315
VERSION_LINEAR_FM_INT16_INT8 = 8


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="output EGEV4 vector file")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--mode", choices=["zero", "pattern", "random"], default="zero")
    parser.add_argument("--random-density", type=float, default=1.0)
    parser.add_argument("--random-abs", type=int, default=4)
    parser.add_argument("--vector-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    if args.dim <= 0:
        raise ValueError("--dim must be positive")
    if not (0.0 <= args.random_density <= 1.0):
        raise ValueError("--random-density must be in [0, 1]")
    if args.random_abs < 0 or args.random_abs > 127:
        raise ValueError("--random-abs must be in [0, 127]")
    if args.vector_scale <= 0.0:
        raise ValueError("--vector-scale must be positive")

    timestamp = args.timestamp or _datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if len(timestamp) != 14 or not timestamp.isdigit():
        raise ValueError("--timestamp must be 14 digits")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    linear_scales = [0.0 for _ in range(N_PHASES)]
    vector_scales = [float(args.vector_scale) for _ in range(N_PHASES)]
    nonzero = 0

    with output.open("wb") as f:
        f.write(timestamp.encode("ascii"))
        f.write(b"EGEV")
        f.write(struct.pack("<4i", VERSION_LINEAR_FM_INT16_INT8, N_PHASES, N_PARAMS_PER_PHASE, args.dim))
        f.write(struct.pack("<60f", *linear_scales))
        f.write(struct.pack("<60f", *vector_scales))
        if args.mode == "zero":
            phase_payload = (struct.pack("<h", 0) + bytes(args.dim)) * N_PARAMS_PER_PHASE
            for _phase in range(N_PHASES):
                f.write(phase_payload)
        else:
            for phase in range(N_PHASES):
                phase_payload = bytearray()
                for param_id in range(N_PARAMS_PER_PHASE):
                    phase_payload += struct.pack("<h", 0)
                    for dim in range(args.dim):
                        q = 0
                        if args.mode == "pattern":
                            if args.random_abs > 0:
                                span = args.random_abs * 2 + 1
                                q = ((phase * 17 + param_id * 5 + dim * 3) % span) - args.random_abs
                                if q == 0:
                                    q = 1
                        elif rng.random() < args.random_density:
                            q = rng.randint(-args.random_abs, args.random_abs)
                            if q == 0 and args.random_abs > 0:
                                q = 1 if rng.random() < 0.5 else -1
                        q = clamp(q, -127, 127)
                        nonzero += q != 0
                        phase_payload += struct.pack("b", q)
                f.write(phase_payload)

    print(f"wrote {output}")
    print(f"version {VERSION_LINEAR_FM_INT16_INT8} phases {N_PHASES} params_per_phase {N_PARAMS_PER_PHASE} dim {args.dim}")
    print(f"mode {args.mode} vector_scale {args.vector_scale} nonzero_vector_values {nonzero}")


if __name__ == "__main__":
    main()
