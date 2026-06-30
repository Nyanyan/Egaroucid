import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_ids(text):
    if not text:
        return []
    return [int(elem.strip()) for elem in text.split(",") if elem.strip()]


def collect_files(root, phase, ids):
    files = []
    for file_id in ids:
        path = root / str(phase) / f"{file_id}.dat"
        if not path.exists():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"{path} is empty")
        files.append((file_id, path, size))
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


def parse_loss_line(text):
    pattern = re.compile(
        r"n_data\s+(\d+)\s+mse\s+([0-9.eE+-]+)\s+rmse\s+([0-9.eE+-]+)\s+mae\s+([0-9.eE+-]+)"
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"could not parse loss output: {text}")
    return {
        "n_data": int(match.group(1)),
        "mse": float(match.group(2)),
        "rmse": float(match.group(3)),
        "mae": float(match.group(4)),
    }


def run_holdout_loss(repo, loss_script, fm_text, holdout_paths, max_records, dim, cwd, stdout_name):
    cmd = [
        sys.executable,
        str(loss_script),
        "--fm-text",
        str(fm_text),
        "--dim",
        str(dim),
    ]
    if max_records > 0:
        cmd += ["--max-records", str(max_records)]
    cmd += [str(path) for path in holdout_paths]
    stdout_path = cwd / stdout_name
    stderr_path = cwd / stdout_name.replace(".log", "_stderr.log")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.run(cmd, cwd=repo, stdout=stdout, stderr=stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    text = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
    return text, parse_loss_line(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=int)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--holdout-root", required=True)
    parser.add_argument("--holdout-ids", required=True)
    parser.add_argument("--target-subset-bytes", type=int, default=0)
    parser.add_argument("--initial-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--holdout-max-records", type=int, default=50000)
    parser.add_argument("--n-patience", type=int, default=100)
    parser.add_argument("--reduce-lr-patience", type=int, default=30)
    parser.add_argument("--reduce-lr-ratio", type=float, default=0.8)
    parser.add_argument("--exe", default="src/tools/evaluation/eval_optimizer_fm_cuda_12_2_0.exe")
    parser.add_argument(
        "--loss-script",
        default="src/tools/evaluation/test_loss_current_fm_text_experiment.py",
    )
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
    holdout_root = Path(args.holdout_root)
    if not holdout_root.is_absolute():
        holdout_root = (repo / holdout_root).resolve()
    initial_dir = Path(args.initial_dir)
    if not initial_dir.is_absolute():
        initial_dir = (repo / initial_dir).resolve()
    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = (repo / exe).resolve()
    loss_script = Path(args.loss_script)
    if not loss_script.is_absolute():
        loss_script = (repo / loss_script).resolve()

    train_ids = parse_ids(args.train_ids)
    holdout_ids = parse_ids(args.holdout_ids)
    train_files = collect_files(train_root, args.phase, train_ids)
    holdout_files = collect_files(holdout_root, args.phase, holdout_ids)
    holdout_paths = [path for _, path, _ in holdout_files]

    config_path = work_dir / "holdout_config.txt"
    with config_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"phase={args.phase}\n")
        f.write(f"rounds={args.rounds}\n")
        f.write(f"seconds={args.seconds}\n")
        f.write(f"alpha={args.alpha}\n")
        f.write(f"train_root={args.train_root}\n")
        f.write(f"train_ids={','.join(str(elem) for elem in train_ids)}\n")
        f.write(f"holdout_root={args.holdout_root}\n")
        f.write(f"holdout_ids={','.join(str(elem) for elem in holdout_ids)}\n")
        f.write(f"target_subset_bytes={args.target_subset_bytes}\n")
        f.write(f"initial_dir={args.initial_dir}\n")
        f.write(f"dim={args.dim}\n")
        f.write(f"holdout_max_records={args.holdout_max_records}\n")
        f.write(f"n_patience={args.n_patience}\n")
        f.write(f"reduce_lr_patience={args.reduce_lr_patience}\n")
        f.write(f"reduce_lr_ratio={args.reduce_lr_ratio}\n")
        f.write(f"exe={args.exe}\n")
        f.write(f"loss_script={args.loss_script}\n")
        f.write("train_files=" + ",".join(f"{tid}:{size}" for tid, _, size in train_files) + "\n")
        f.write("holdout_files=" + ",".join(f"{tid}:{size}" for tid, _, size in holdout_files) + "\n")

    opt_log_path = work_dir / "opt_log.txt"
    holdout_log_path = work_dir / "holdout_loss_log.txt"
    opt_log_path.write_text("", encoding="utf-8")
    holdout_log_path.write_text("", encoding="utf-8")

    in_file = initial_dir / f"{args.phase}_fm.txt"
    initial_loss_text, initial_loss = run_holdout_loss(
        repo,
        loss_script,
        in_file,
        holdout_paths,
        args.holdout_max_records,
        args.dim,
        work_dir,
        f"phase{args.phase}_holdout_initial.log",
    )
    with holdout_log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(
            f"round=0 n_data={initial_loss['n_data']} mse={initial_loss['mse']:.8g} "
            f"rmse={initial_loss['rmse']:.8g} mae={initial_loss['mae']:.8g} "
            f"fm_text={in_file}\n"
        )
    print(initial_loss_text, flush=True)

    start_idx = 0
    last_trained = None
    for round_idx in range(args.rounds):
        subset, start_idx = make_rotating_subset(train_files, start_idx, args.target_subset_bytes)
        round_dir = work_dir / f"round{round_idx + 1:02d}"
        round_trained_dir = round_dir / "trained"
        round_trained_dir.mkdir(parents=True, exist_ok=True)
        subset_ids = [train_id for train_id, _, _ in subset]
        subset_bytes = sum(size for _, _, size in subset)
        print(
            f"round {round_idx + 1}/{args.rounds} phase={args.phase} "
            f"alpha={args.alpha} subset_ids={subset_ids} subset_bytes={subset_bytes}",
            flush=True,
        )
        cmd = [
            str(exe),
            str(args.phase),
            "0",
            "0",
            str(args.seconds),
            str(args.alpha),
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
        with opt_log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(summary_with_data + "\n")
        print(summary_with_data, flush=True)
        print("returncode", proc.returncode, flush=True)
        print("stderr", stderr_path, flush=True)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        last_trained = round_trained_dir / f"{args.phase}_fm.txt"
        if not last_trained.exists():
            raise RuntimeError(f"optimizer did not write {last_trained}")

        loss_text, loss = run_holdout_loss(
            repo,
            loss_script,
            last_trained,
            holdout_paths,
            args.holdout_max_records,
            args.dim,
            round_dir,
            f"phase{args.phase}_holdout_round{round_idx + 1:02d}.log",
        )
        with holdout_log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(
                f"round={round_idx + 1} n_data={loss['n_data']} mse={loss['mse']:.8g} "
                f"rmse={loss['rmse']:.8g} mae={loss['mae']:.8g} "
                f"fm_text={last_trained}\n"
            )
        print(loss_text, flush=True)
        in_file = last_trained

    shutil.copyfile(last_trained, final_trained_dir / f"{args.phase}_fm.txt")
    print("final_trained", final_trained_dir / f"{args.phase}_fm.txt", flush=True)


if __name__ == "__main__":
    main()
