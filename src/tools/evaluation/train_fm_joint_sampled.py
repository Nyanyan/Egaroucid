import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from shutil import which
from typing import Optional


MODEL_DIR_RE = re.compile(r"^\d{8}_\d+_.+")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sanitize_model_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_")
    if not sanitized:
        raise ValueError("model name is empty after sanitizing")
    return sanitized


def make_numbered_model_dir(root: Path, suffix: str) -> Path:
    model_dir = root / "model"
    date = datetime.now().strftime("%Y%m%d")
    max_index = 0
    if model_dir.is_dir():
        pattern = re.compile(rf"^{date}_(\d+)_")
        for child in model_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return model_dir / f"{date}_{max_index + 1}_{sanitize_model_name(suffix)}"


def validate_model_out_dir(root: Path, out_dir: Path) -> None:
    model_dir = root / "model"
    if path_is_relative_to(out_dir, model_dir) and not MODEL_DIR_RE.match(out_dir.name):
        raise ValueError(
            "model folder names must be formatted like YYYYMMDD_N_description, "
            f"but got: {out_dir}"
        )


def format_float(value: float) -> str:
    return f"{value:.12g}"


def build_optimizer(root: Path, compiler: str, source: Path, exe: Path) -> list[str]:
    exe.parent.mkdir(parents=True, exist_ok=True)
    return [
        compiler,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-o",
        str(exe),
        str(source),
    ]


def find_existing_path(candidates: list[str]) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def find_nvcc(value: str) -> str:
    if value:
        return value
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin\nvcc.exe",
    ]
    path = find_existing_path(candidates)
    if path is not None:
        return str(path)
    found = which("nvcc")
    return found or "nvcc"


def find_msvc_ccbin(value: str) -> Optional[str]:
    if value:
        return value
    base = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC")
    if not base.is_dir():
        return None
    versions = sorted(base.iterdir(), reverse=True)
    for version in versions:
        candidate = version / "bin" / "HostX64" / "x64"
        if (candidate / "cl.exe").exists():
            return str(candidate)
    return None


def build_cuda_optimizer(
    root: Path,
    nvcc: str,
    ccbin: Optional[str],
    source: Path,
    exe: Path,
    arch: str,
) -> list[str]:
    exe.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nvcc,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        f"-arch={arch}",
        "-allow-unsupported-compiler",
        "-o",
        str(exe),
        str(source),
    ]
    if ccbin:
        cmd[1:1] = ["-ccbin", ccbin]
    return cmd


def main() -> int:
    root = repo_root()
    default_data_root = (
        Path(os.environ.get("EGAROUCID_DATA", "E:/egaroucid_data"))
        / "train_data"
        / "bin_data"
        / "20241125_1"
    )

    parser = argparse.ArgumentParser(
        description="Train a joint linear + shared-FM evaluation from sampled records223+ indexed data."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--nvcc", default="")
    parser.add_argument("--cuda-ccbin", default="")
    parser.add_argument("--cuda-arch", default="sm_86")
    parser.add_argument("--optimizer-source", default="src/tools/evaluation/eval_optimizer_fm_joint.cpp")
    parser.add_argument("--optimizer-exe", default="src/tools/evaluation/eval_optimizer_fm_joint.exe")
    parser.add_argument("--base-eval", default="bin/resources/eval.egev2")
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--val-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--linear-lr", type=float, default=0.1)
    parser.add_argument("--fm-lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--linear-l2", type=float, default=0.0)
    parser.add_argument("--fm-l2", type=float, default=0.0)
    parser.add_argument("--grad-clip-raw", type=float, default=0.0)
    parser.add_argument("--linear-param-clip", type=float, default=4091.0)
    parser.add_argument("--fm-vector-clip", type=float, default=127.0)
    parser.add_argument("--init-std", type=float, default=0.01)
    parser.add_argument("--scale", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--record-start", type=int, default=223)
    parser.add_argument("--record-end", type=int, default=-1)
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-end", type=int, default=59)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument("--max-memory-gib", type=float, default=100.0)
    parser.add_argument("--train-metric-limit", type=int, default=1_000_000)
    parser.add_argument("--val-metric-limit", type=int, default=0)
    parser.add_argument("--read-mode", choices=("bulk", "scan", "seek"), default="bulk")
    parser.add_argument("--read-threads", type=int, default=4)
    parser.add_argument("--progress-interval-sec", type=int, default=30)
    args = parser.parse_args()

    if args.dim <= 0 or args.epochs < 0 or args.train_samples <= 0 or args.val_samples <= 0:
        raise ValueError("dim, epochs, train-samples and val-samples are invalid")
    if args.batch_size <= 0 or args.scale <= 0:
        raise ValueError("batch-size and scale must be positive")
    if args.record_end != -1 and args.record_end < args.record_start:
        raise ValueError("record-end must be -1 or greater than or equal to record-start")
    if args.phase_start < 0 or args.phase_end < args.phase_start or args.phase_end >= 60:
        raise ValueError("invalid phase range")
    if args.progress_interval_sec < 0:
        raise ValueError("progress-interval-sec must be non-negative")
    if args.read_threads <= 0:
        raise ValueError("read-threads must be positive")

    if args.backend == "cuda" and args.optimizer_source == "src/tools/evaluation/eval_optimizer_fm_joint.cpp":
        args.optimizer_source = "src/tools/evaluation/eval_optimizer_fm_joint_cuda.cu"
    if args.backend == "cuda" and args.optimizer_exe == "src/tools/evaluation/eval_optimizer_fm_joint.exe":
        args.optimizer_exe = "src/tools/evaluation/eval_optimizer_fm_joint_cuda.exe"

    optimizer_source = resolve_path(root, args.optimizer_source)
    optimizer_exe = resolve_path(root, args.optimizer_exe)
    base_eval = resolve_path(root, args.base_eval)
    data_root = resolve_path(root, args.data_root)

    if args.out_dir is None:
        suffix = args.model_name or (
            f"fm_joint_sampled_dim{args.dim}_records{args.record_start}plus"
            f"_train{args.train_samples}_val{args.val_samples}_e{args.epochs}"
        )
        out_dir = make_numbered_model_dir(root, suffix)
    else:
        out_dir = resolve_path(root, args.out_dir)
    validate_model_out_dir(root, out_dir)
    out_file = out_dir / (
        f"eval_dim{args.dim}_fmphase1_records{args.record_start}plus_joint_sampled.egevfm"
    )

    if args.backend == "cuda":
        build_cmd = build_cuda_optimizer(
            root,
            find_nvcc(args.nvcc),
            find_msvc_ccbin(args.cuda_ccbin),
            optimizer_source,
            optimizer_exe,
            args.cuda_arch,
        )
    else:
        build_cmd = build_optimizer(root, args.compiler, optimizer_source, optimizer_exe)
    train_cmd = [
        str(optimizer_exe),
        "--base-eval",
        str(base_eval),
        "--data-root",
        str(data_root),
        "--out-file",
        str(out_file),
        "--dim",
        str(args.dim),
        "--epochs",
        str(args.epochs),
        "--train-samples",
        str(args.train_samples),
        "--val-samples",
        str(args.val_samples),
        "--batch-size",
        str(args.batch_size),
        "--linear-lr",
        format_float(args.linear_lr),
        "--fm-lr",
        format_float(args.fm_lr),
        "--beta1",
        format_float(args.beta1),
        "--beta2",
        format_float(args.beta2),
        "--adam-eps",
        format_float(args.adam_eps),
        "--linear-l2",
        format_float(args.linear_l2),
        "--fm-l2",
        format_float(args.fm_l2),
        "--grad-clip-raw",
        format_float(args.grad_clip_raw),
        "--linear-param-clip",
        format_float(args.linear_param_clip),
        "--fm-vector-clip",
        format_float(args.fm_vector_clip),
        "--init-std",
        format_float(args.init_std),
        "--scale",
        str(args.scale),
        "--seed",
        str(args.seed),
        "--record-start",
        str(args.record_start),
        "--record-end",
        str(args.record_end),
        "--phase-start",
        str(args.phase_start),
        "--phase-end",
        str(args.phase_end),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--max-memory-gib",
        format_float(args.max_memory_gib),
        "--train-metric-limit",
        str(args.train_metric_limit),
        "--val-metric-limit",
        str(args.val_metric_limit),
        "--read-mode",
        args.read_mode,
        "--read-threads",
        str(args.read_threads),
        "--progress-interval-sec",
        str(args.progress_interval_sec),
        "--dry-run",
        "1" if args.dry_run else "0",
    ]

    manifest = {
        "method": "joint_linear_fm_adam_sampled_mse_cuda" if args.backend == "cuda" else "joint_linear_fm_adam_sampled_mse",
        "notes": [
            "linear term has 60 phases",
            "FM term has 1 shared phase",
            "linear and FM parameters are optimized together",
            "samples are drawn uniformly from all indexed records in the selected range",
            "all pattern features are used for the FM term",
        ],
        "out_dir": str(out_dir),
        "out_file": str(out_file),
        "optimizer_source": str(optimizer_source),
        "optimizer_exe": str(optimizer_exe),
        "backend": args.backend,
        "build_cmd": build_cmd,
        "train_cmd": train_cmd,
        "args": vars(args),
    }

    print("build command:")
    print(" ".join(build_cmd))
    print("train command:")
    print(" ".join(train_cmd))

    if args.execute or args.build:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.build:
        subprocess.run(build_cmd, cwd=root, check=True)

    if args.execute:
        if not optimizer_exe.exists():
            raise FileNotFoundError(f"optimizer executable does not exist: {optimizer_exe}")
        subprocess.run(train_cmd, cwd=root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
