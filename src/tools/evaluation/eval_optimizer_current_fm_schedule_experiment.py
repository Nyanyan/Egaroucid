import argparse
import re
import subprocess
import sys
from pathlib import Path


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


def parse_ids(value):
    if not value:
        return []
    return [int(elem.strip()) for elem in value.split(",") if elem.strip()]


def parse_alpha_by_phase(value):
    alpha_by_phase = {}
    if not value:
        return alpha_by_phase
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        phases_s, alpha_s = part.split(":", 1)
        alpha = float(alpha_s)
        for phase in parse_phases(phases_s):
            alpha_by_phase[phase] = alpha
    return alpha_by_phase


def parse_schedule_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    file_ids_by_phase = {}
    alpha_by_phase = {}
    pattern = re.compile(
        r"phase\s+(\d+).*?alpha\s+([0-9.]+).*?train_data_nums=\[(.*?)\]",
        re.DOTALL,
    )
    for phase_s, alpha_s, ids_s in pattern.findall(text):
        phase = int(phase_s)
        alpha_by_phase[phase] = float(alpha_s)
        file_ids_by_phase[phase] = [int(value) for value in re.findall(r"\d+", ids_s)]
    return file_ids_by_phase, alpha_by_phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="0-59")
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--alpha-by-phase", default="")
    parser.add_argument("--schedule-log", default="")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--train-ids", default="", help="fallback comma-separated train data IDs")
    parser.add_argument("--file-id-limit", type=int, default=0)
    parser.add_argument("--exclude-train-ids", default="", help="comma-separated train data IDs to skip")
    parser.add_argument("--max-train-file-bytes", type=int, default=0, help="skip train files larger than this size; 0 disables the filter")
    parser.add_argument("--initial-dir", default="")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--n-patience", type=int, default=100)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-ratio", type=float, default=0.7)
    parser.add_argument("--exe", default="src/tools/evaluation/eval_optimizer_fm_cuda_12_2_0.exe")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    script = Path(__file__).resolve().parent / "eval_optimizer_phase_current_fm_experiment.py"
    phases = parse_phases(args.phases)
    file_ids_by_phase = {}
    log_alpha_by_phase = {}
    if args.schedule_log:
        file_ids_by_phase, log_alpha_by_phase = parse_schedule_log(args.schedule_log)
    alpha_by_phase = log_alpha_by_phase
    alpha_by_phase.update(parse_alpha_by_phase(args.alpha_by_phase))
    fallback_ids = parse_ids(args.train_ids)

    work_dir = (repo / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    opt_log = work_dir / "opt_log.txt"
    if opt_log.exists():
        opt_log.unlink()
    with (work_dir / "schedule_config.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"phases={args.phases}\n")
        f.write(f"seconds={args.seconds}\n")
        f.write(f"alpha={args.alpha}\n")
        f.write(f"alpha_scale={args.alpha_scale}\n")
        f.write(f"alpha_by_phase={args.alpha_by_phase}\n")
        f.write(f"schedule_log={args.schedule_log}\n")
        f.write(f"train_root={args.train_root}\n")
        f.write(f"train_ids={args.train_ids}\n")
        f.write(f"file_id_limit={args.file_id_limit}\n")
        f.write(f"exclude_train_ids={args.exclude_train_ids}\n")
        f.write(f"max_train_file_bytes={args.max_train_file_bytes}\n")
        f.write(f"initial_dir={args.initial_dir}\n")
        f.write(f"n_patience={args.n_patience}\n")
        f.write(f"reduce_lr_patience={args.reduce_lr_patience}\n")
        f.write(f"reduce_lr_ratio={args.reduce_lr_ratio}\n")
        f.write(f"exe={args.exe}\n")

    for phase in phases:
        ids = file_ids_by_phase.get(phase, fallback_ids)
        if args.file_id_limit > 0:
            ids = ids[:args.file_id_limit]
        if not ids:
            raise RuntimeError(f"phase {phase}: no train data IDs")
        phase_alpha = alpha_by_phase.get(phase, args.alpha) * args.alpha_scale
        print(
            f"optimizing FM phase {phase} seconds={args.seconds} "
            f"alpha={phase_alpha} train_ids={ids}",
            flush=True,
        )
        cmd = [
            sys.executable,
            str(script),
            str(phase),
            "--seconds", str(args.seconds),
            "--alpha", str(phase_alpha),
            "--train-root", args.train_root,
            "--train-ids", ",".join(str(elem) for elem in ids),
            "--exclude-train-ids", args.exclude_train_ids,
            "--max-train-file-bytes", str(args.max_train_file_bytes),
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
