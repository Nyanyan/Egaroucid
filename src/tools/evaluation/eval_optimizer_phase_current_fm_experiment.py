import argparse
import subprocess
from pathlib import Path


def parse_ids(text):
    if not text:
        return []
    return [int(elem.strip()) for elem in text.split(",") if elem.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=int)
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-patience", type=int, default=100)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-ratio", type=float, default=0.7)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--train-ids", required=True, help="comma-separated train data IDs")
    parser.add_argument("--initial-dir", default="")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--exe", default="src/tools/evaluation/eval_optimizer_fm_cuda_12_2_0.exe")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    work_dir = (repo / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "trained").mkdir(parents=True, exist_ok=True)

    train_root = Path(args.train_root)
    if not train_root.is_absolute():
        train_root = (repo / train_root).resolve()
    train_files = [
        train_root / str(args.phase) / f"{train_id}.dat"
        for train_id in parse_ids(args.train_ids)
    ]
    train_files = [path for path in train_files if path.exists() and path.stat().st_size > 0]
    if not train_files:
        raise RuntimeError("no non-empty training files found")

    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = (repo / exe).resolve()

    initial_dir = Path(args.initial_dir) if args.initial_dir else Path()
    if args.initial_dir and not initial_dir.is_absolute():
        initial_dir = (repo / initial_dir).resolve()
    in_file = initial_dir / f"{args.phase}_fm.txt" if args.initial_dir else work_dir / "missing_initial" / f"{args.phase}_fm.txt"

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
    ] + [str(path) for path in train_files]

    stderr_path = work_dir / f"phase{args.phase}_stderr.log"
    stdout_path = work_dir / f"phase{args.phase}_stdout.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.run(cmd, cwd=work_dir, stdout=stdout, stderr=stderr)

    summary = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
    summary_with_data = summary + " train_data_nums=" + str([int(path.stem) for path in train_files])
    with (work_dir / "opt_log.txt").open("a", encoding="utf-8") as f:
        f.write(summary_with_data + "\n")
    print(summary_with_data)
    print("returncode", proc.returncode)
    print("stderr", stderr_path)
    print("trained", work_dir / "trained" / f"{args.phase}_fm.txt")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
