import argparse
import re
import shutil
import subprocess
from pathlib import Path


def parse_ids(text):
    if not text:
        return []
    return [int(elem.strip()) for elem in text.split(",") if elem.strip()]


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


def collect_train_files(train_root, phase, train_ids, excluded_ids, max_train_file_bytes):
    files = []
    for train_id in train_ids:
        if train_id in excluded_ids:
            continue
        path = train_root / str(phase) / f"{train_id}.dat"
        if not path.exists():
            continue
        size = path.stat().st_size
        if size <= 0:
            continue
        if max_train_file_bytes > 0 and size > max_train_file_bytes:
            continue
        files.append((train_id, path, size))
    return files


def make_rotating_subset(files, start_idx, target_bytes):
    if not files:
        raise RuntimeError("no training files available")
    if target_bytes <= 0:
        return files[:], 0

    subset = []
    total = 0
    idx = start_idx
    visited = 0
    while visited < len(files):
        item = files[idx]
        if subset and total + item[2] > target_bytes:
            break
        subset.append(item)
        total += item[2]
        idx = (idx + 1) % len(files)
        visited += 1
        if total >= target_bytes:
            break
    return subset, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=int)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--schedule-log", default="")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--train-ids", default="", help="fallback comma-separated train data IDs")
    parser.add_argument("--exclude-train-ids", default="")
    parser.add_argument("--max-train-file-bytes", type=int, default=0)
    parser.add_argument(
        "--target-subset-bytes",
        type=int,
        default=0,
        help="target bytes per optimizer call; 0 means use all eligible files each round",
    )
    parser.add_argument("--initial-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--n-patience", type=int, default=100)
    parser.add_argument("--reduce-lr-patience", type=int, default=30)
    parser.add_argument("--reduce-lr-ratio", type=float, default=0.8)
    parser.add_argument("--exe", default="src/tools/evaluation/eval_optimizer_fm_cuda_12_2_0.exe")
    args = parser.parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    repo = Path(__file__).resolve().parents[3]
    work_dir = (repo / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    final_trained_dir = work_dir / "trained"
    final_trained_dir.mkdir(parents=True, exist_ok=True)

    train_root = Path(args.train_root)
    if not train_root.is_absolute():
        train_root = (repo / train_root).resolve()
    initial_dir = Path(args.initial_dir)
    if not initial_dir.is_absolute():
        initial_dir = (repo / initial_dir).resolve()
    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = (repo / exe).resolve()

    file_ids_by_phase = {}
    alpha_by_phase = {}
    if args.schedule_log:
        file_ids_by_phase, alpha_by_phase = parse_schedule_log(args.schedule_log)
    train_ids = file_ids_by_phase.get(args.phase, parse_ids(args.train_ids))
    if not train_ids:
        raise RuntimeError(f"phase {args.phase}: no train data IDs")

    excluded_ids = set(parse_ids(args.exclude_train_ids))
    files = collect_train_files(
        train_root,
        args.phase,
        train_ids,
        excluded_ids,
        args.max_train_file_bytes,
    )
    if not files:
        raise RuntimeError("no non-empty training files found")

    phase_alpha = alpha_by_phase.get(args.phase, args.alpha) * args.alpha_scale
    with (work_dir / "rotating_subset_config.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"phase={args.phase}\n")
        f.write(f"rounds={args.rounds}\n")
        f.write(f"seconds={args.seconds}\n")
        f.write(f"alpha={phase_alpha}\n")
        f.write(f"alpha_scale={args.alpha_scale}\n")
        f.write(f"schedule_log={args.schedule_log}\n")
        f.write(f"train_root={args.train_root}\n")
        f.write(f"train_ids={','.join(str(elem) for elem in train_ids)}\n")
        f.write(f"exclude_train_ids={args.exclude_train_ids}\n")
        f.write(f"max_train_file_bytes={args.max_train_file_bytes}\n")
        f.write(f"target_subset_bytes={args.target_subset_bytes}\n")
        f.write(f"initial_dir={args.initial_dir}\n")
        f.write(f"n_patience={args.n_patience}\n")
        f.write(f"reduce_lr_patience={args.reduce_lr_patience}\n")
        f.write(f"reduce_lr_ratio={args.reduce_lr_ratio}\n")
        f.write(f"exe={args.exe}\n")
        f.write("eligible_files=" + ",".join(f"{train_id}:{size}" for train_id, _, size in files) + "\n")

    opt_log_path = work_dir / "opt_log.txt"
    opt_log_path.write_text("", encoding="utf-8")
    in_file = initial_dir / f"{args.phase}_fm.txt"
    start_idx = 0
    last_trained = None
    for round_idx in range(args.rounds):
        subset, start_idx = make_rotating_subset(files, start_idx, args.target_subset_bytes)
        round_dir = work_dir / f"round{round_idx + 1:02d}"
        round_trained_dir = round_dir / "trained"
        round_trained_dir.mkdir(parents=True, exist_ok=True)
        subset_ids = [train_id for train_id, _, _ in subset]
        subset_bytes = sum(size for _, _, size in subset)
        print(
            f"round {round_idx + 1}/{args.rounds} phase={args.phase} "
            f"alpha={phase_alpha} subset_ids={subset_ids} subset_bytes={subset_bytes}",
            flush=True,
        )
        cmd = [
            str(exe),
            str(args.phase),
            "0",
            "0",
            str(args.seconds),
            str(phase_alpha),
            str(args.n_patience),
            str(args.reduce_lr_patience),
            str(args.reduce_lr_ratio),
            str(in_file),
        ] + [str(path) for _, path, _ in subset]

        stdout_path = round_dir / f"phase{args.phase}_stdout.log"
        stderr_path = round_dir / f"phase{args.phase}_stderr.log"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.run(cmd, cwd=round_dir, stdout=stdout, stderr=stderr)

        summary = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
        summary_with_data = (
            summary
            + f" round={round_idx + 1}"
            + f" subset_bytes={subset_bytes}"
            + " train_data_nums="
            + str(subset_ids)
        )
        with opt_log_path.open("a", encoding="utf-8") as f:
            f.write(summary_with_data + "\n")
        print(summary_with_data)
        print("returncode", proc.returncode)
        print("stderr", stderr_path)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        last_trained = round_trained_dir / f"{args.phase}_fm.txt"
        if not last_trained.exists():
            raise RuntimeError(f"optimizer did not write {last_trained}")
        in_file = last_trained

    shutil.copyfile(last_trained, final_trained_dir / f"{args.phase}_fm.txt")
    print("final_trained", final_trained_dir / f"{args.phase}_fm.txt")


if __name__ == "__main__":
    main()
