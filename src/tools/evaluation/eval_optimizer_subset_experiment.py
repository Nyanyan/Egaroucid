import argparse
import os
import subprocess
import sys
from pathlib import Path


VARIANT_EXE = {
    "stable": "eval_optimizer_cuda_stable_adam_12_2_0.exe",
    "robust": "eval_optimizer_cuda_robust_adam_12_2_0.exe",
}


def parse_phases(value):
    phases = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if start <= end else -1
            phases.extend(range(start, end + step, step))
        else:
            phases.append(int(part))
    return phases


def pick_train_files(phase_dir, max_files):
    files = [p for p in phase_dir.glob("*.dat") if p.stat().st_size > 0]
    files.sort(key=lambda p: (p.stat().st_size, p.name))
    return files[:max_files]


def read_last_line(path):
    if not path.exists():
        return ""
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                last = line.strip()
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANT_EXE))
    parser.add_argument("output_dir")
    parser.add_argument("--phases", default="0-59")
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=300.0)
    parser.add_argument("--n-patience", type=int, default=2)
    parser.add_argument("--reduce-lr-patience", type=int, default=1)
    parser.add_argument("--reduce-lr-ratio", type=float, default=0.8)
    parser.add_argument("--round-ms", type=int, default=200)
    parser.add_argument("--max-files-per-phase", type=int, default=1)
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--initial-dir", default=None)
    parser.add_argument("--timeout-sec", type=int, default=0)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    exe = script_dir / VARIANT_EXE[args.variant]
    if not exe.exists():
        print(f"missing executable: {exe}", file=sys.stderr)
        return 1

    train_root = Path(args.train_root or (os.environ["EGAROUCID_DATA"] + "/train_data/bin_data/20241125_1"))
    output_dir = Path(args.output_dir).resolve()
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    phases = parse_phases(args.phases)
    env = os.environ.copy()
    env["EGAROUCID_EVAL_TRAINED_DIR"] = str(output_dir)
    env["EGAROUCID_EVAL_ROUND_TL_MS"] = str(args.round_ms)

    config_path = output_dir / "run_config.txt"
    with config_path.open("w", encoding="utf-8") as f:
        f.write(f"variant={args.variant}\n")
        f.write(f"executable={exe}\n")
        f.write(f"train_root={train_root}\n")
        f.write(f"phases={args.phases}\n")
        f.write(f"seconds={args.seconds}\n")
        f.write(f"alpha={args.alpha}\n")
        f.write(f"n_patience={args.n_patience}\n")
        f.write(f"reduce_lr_patience={args.reduce_lr_patience}\n")
        f.write(f"reduce_lr_ratio={args.reduce_lr_ratio}\n")
        f.write(f"round_ms={args.round_ms}\n")
        f.write(f"max_files_per_phase={args.max_files_per_phase}\n")
        f.write(f"initial_dir={args.initial_dir or ''}\n")

    summary_path = output_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="\n") as summary_file:
        summary_file.write("phase\treturn_code\tfiles\tsummary\n")

        for phase in phases:
            phase_dir = train_root / str(phase)
            train_files = pick_train_files(phase_dir, args.max_files_per_phase)
            if not train_files:
                print(f"phase {phase}: no train data in {phase_dir}", file=sys.stderr)
                return 1

            in_file = "none.txt"
            if args.initial_dir is not None:
                candidate = Path(args.initial_dir) / f"{phase}.txt"
                if candidate.exists():
                    in_file = str(candidate)

            cmd = [
                str(exe),
                str(phase),
                "0",
                "0",
                str(args.seconds),
                str(args.alpha),
                str(args.n_patience),
                str(args.reduce_lr_patience),
                str(args.reduce_lr_ratio),
                in_file,
            ]
            cmd.extend(str(path) for path in train_files)

            stdout_path = log_dir / f"phase_{phase:02d}.stdout.log"
            stderr_path = log_dir / f"phase_{phase:02d}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    cmd,
                    cwd=script_dir,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=args.timeout_sec if args.timeout_sec > 0 else None,
                )
            summary = read_last_line(stdout_path) or read_last_line(stderr_path)
            file_names = ",".join(p.name for p in train_files)
            summary_file.write(f"{phase}\t{completed.returncode}\t{file_names}\t{summary}\n")
            summary_file.flush()
            print(
                f"phase {phase:02d} rc={completed.returncode} "
                f"files={[p.name for p in train_files]} summary={summary}"
            )
            if completed.returncode != 0:
                return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
