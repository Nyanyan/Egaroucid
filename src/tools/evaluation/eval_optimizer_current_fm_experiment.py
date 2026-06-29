import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_phase", type=int)
    parser.add_argument("end_phase", type=int)
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--train-ids", required=True, help="comma-separated train data IDs used for every phase")
    parser.add_argument("--initial-dir", default="")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--n-patience", type=int, default=100)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-ratio", type=float, default=0.7)
    parser.add_argument("--exe", default="src/tools/evaluation/eval_optimizer_fm_cuda_12_2_0.exe")
    args = parser.parse_args()

    if args.start_phase < 0 or args.end_phase < args.start_phase or args.end_phase >= 60:
        raise ValueError("phase range must be within [0, 59]")

    script = Path(__file__).resolve().parent / "eval_optimizer_phase_current_fm_experiment.py"
    repo = Path(__file__).resolve().parents[3]
    for phase in range(args.start_phase, args.end_phase + 1):
        print(f"optimizing FM phase {phase}", flush=True)
        cmd = [
            sys.executable,
            str(script),
            str(phase),
            "--seconds", str(args.seconds),
            "--alpha", str(args.alpha),
            "--train-root", args.train_root,
            "--train-ids", args.train_ids,
            "--work-dir", args.work_dir,
            "--n-patience", str(args.n_patience),
            "--reduce-lr-patience", str(args.reduce_lr_patience),
            "--reduce-lr-ratio", str(args.reduce_lr_ratio),
            "--exe", args.exe,
        ]
        if args.initial_dir:
            cmd.extend(["--initial-dir", args.initial_dir])
        proc = subprocess.run(cmd, cwd=repo)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
