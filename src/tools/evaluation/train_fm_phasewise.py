import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


INDEXED_RECORD_BYTES = 136
N_PHASES = 60


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def parse_record_ranges(info_path: Path) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    current: Optional[int] = None
    record_re = re.compile(r"^records(\d+)")
    range_re = re.compile(r"\[(\d+)\s*,\s*(\d+)\]")
    for line in info_path.read_text(encoding="utf-8").splitlines():
        record_match = record_re.match(line)
        if record_match:
            current = int(record_match.group(1))
            continue
        if current is None or current in ranges:
            continue
        range_match = range_re.search(line)
        if range_match:
            ranges[current] = (int(range_match.group(1)), int(range_match.group(2)))
    return ranges


def collect_phase_counts(
    data_root: Path,
    record_start: int,
    record_end: Optional[int],
    ranges: dict[int, tuple[int, int]],
) -> tuple[dict[int, int], int, int, list[str], int]:
    phase_counts = {phase: 0 for phase in range(N_PHASES)}
    total_records = 0
    total_bytes = 0
    max_record = record_start
    violations: list[str] = []

    for phase in range(N_PHASES):
        phase_dir = data_root / str(phase)
        if not phase_dir.is_dir():
            violations.append(f"missing phase directory: {phase_dir}")
            continue
        for file in phase_dir.glob("*.dat"):
            try:
                record = int(file.stem)
            except ValueError:
                continue
            if record < record_start:
                continue
            if record_end is not None and record > record_end:
                continue
            max_record = max(max_record, record)
            size = file.stat().st_size
            if size == 0:
                continue
            if size % INDEXED_RECORD_BYTES != 0:
                violations.append(f"bad record size: phase={phase} record={record} path={file}")
                continue
            if record not in ranges:
                violations.append(f"unknown declared phase range: phase={phase} record={record} path={file}")
                continue
            start_phase, end_phase = ranges[record]
            if not (start_phase <= phase <= end_phase):
                violations.append(
                    f"outside declared phase range: phase={phase} record={record} declared=[{start_phase},{end_phase}] path={file}"
                )
                continue
            records = size // INDEXED_RECORD_BYTES
            phase_counts[phase] += records
            total_records += records
            total_bytes += size

    return phase_counts, total_records, total_bytes, violations, max_record


def format_float(value: float) -> str:
    return f"{value:.12g}"


def run_or_print(cmd: list[str], stdout_path: Path, stderr_path: Path, execute: bool) -> int:
    print(" ".join(cmd))
    if not execute:
        return 0
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        completed = subprocess.run(cmd, stdout=stdout_file, stderr=stderr_file)
    return completed.returncode


def parse_training_logs(log_dir: Path, out_dir: Path, phase_start: int, phase_end: int) -> None:
    samples_re = re.compile(r"samples\s+(\d+)\s+dim\s+(\d+)")
    initial_re = re.compile(r"initial train_mae\s+([0-9.eE+-]+)\s+val_mae\s+([0-9.eE+-]+)")
    epoch_re = re.compile(
        r"epoch\s+(\d+)\s+elapsed_ms\s+(\d+)\s+train_mae\s+([0-9.eE+-]+)\s+val_mae\s+([0-9.eE+-]+)\s+best_epoch\s+(\d+)\s+best_val_mae\s+([0-9.eE+-]+)"
    )
    wrote_re = re.compile(r"wrote\s+.+\s+nonzero_quantized\s+(\d+)\s+max_abs_quantized\s+(\d+)")
    lines = [
        "phase\tsamples\tdim\tinitial_train_mae\tinitial_val_mae\tlast_epoch\tlast_elapsed_ms\tlast_train_mae\tlast_val_mae\tbest_epoch\tbest_val_mae\tnonzero_quantized\tmax_abs_quantized"
    ]
    for phase in range(phase_start, phase_end + 1):
        log_path = log_dir / f"phase_{phase:02d}.stderr.log"
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        samples = ""
        dim = ""
        initial_train = ""
        initial_val = ""
        last_epoch = ""
        last_elapsed = ""
        last_train = ""
        last_val = ""
        best_epoch = ""
        best_val = ""
        nonzero = ""
        max_abs = ""
        for line in text.splitlines():
            match = samples_re.search(line)
            if match:
                samples, dim = match.group(1), match.group(2)
                continue
            match = initial_re.search(line)
            if match:
                initial_train, initial_val = match.group(1), match.group(2)
                continue
            match = epoch_re.search(line)
            if match:
                last_epoch = match.group(1)
                last_elapsed = match.group(2)
                last_train = match.group(3)
                last_val = match.group(4)
                best_epoch = match.group(5)
                best_val = match.group(6)
                continue
            match = wrote_re.search(line)
            if match:
                nonzero, max_abs = match.group(1), match.group(2)
        lines.append(
            "\t".join(
                [
                    str(phase),
                    samples,
                    dim,
                    initial_train,
                    initial_val,
                    last_epoch,
                    last_elapsed,
                    last_train,
                    last_val,
                    best_epoch,
                    best_val,
                    nonzero,
                    max_abs,
                ]
            )
        )
    (out_dir / "training_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    default_data_root = Path(os.environ.get("EGAROUCID_DATA", "E:/egaroucid_data")) / "train_data" / "bin_data" / "20241125_1"

    parser = argparse.ArgumentParser(description="Train Dim2 FM evaluation phase by phase from records223+ indexed data.")
    parser.add_argument("--base-eval", default="bin/resources/eval.egev2")
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--out-dir", default="model/20260724_fm_records223_phasewise_dim2")
    parser.add_argument("--optimizer-exe", default="src/tools/evaluation/eval_optimizer_fm_records223.exe")
    parser.add_argument("--merge-exe", default="src/tools/evaluation/util/merge_egevfm_phases.exe")
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-end", type=int, default=59)
    parser.add_argument("--record-start", type=int, default=223)
    parser.add_argument("--record-end", type=int, default=None)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--fm-phases", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--active-pattern-mask", type=lambda x: int(x, 0), default=0x0520)
    parser.add_argument("--center-phase-target", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1.0e-5)
    parser.add_argument("--error-clip", type=float, default=4096.0)
    parser.add_argument("--vector-clip", type=float, default=7.5)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--target-clip", type=float, default=0.0)
    parser.add_argument("--score-clip", type=int, default=0)
    parser.add_argument("--max-estimated-memory-gib", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.phase_start < 0 or args.phase_end < args.phase_start or args.phase_end >= N_PHASES:
        raise ValueError("invalid phase range")
    if args.record_start < 0:
        raise ValueError("invalid record-start")
    if args.record_end is not None and args.record_end < args.record_start:
        raise ValueError("invalid record-end")

    base_eval = resolve_path(root, args.base_eval)
    data_root = resolve_path(root, args.data_root)
    out_dir = resolve_path(root, args.out_dir)
    optimizer_exe = resolve_path(root, args.optimizer_exe)
    merge_exe = resolve_path(root, args.merge_exe)
    log_dir = out_dir / "logs"
    phase_out_dir = out_dir / "phases"
    final_file = out_dir / "eval_dim2_fmphase60_records223plus_phasewise.egevfm"

    ranges = parse_record_ranges(root / "train_data" / "train_data_info.txt")
    phase_counts, total_records, total_bytes, violations, max_record = collect_phase_counts(
        data_root, args.record_start, args.record_end, ranges
    )
    if violations:
        for violation in violations[:50]:
            print(violation)
        raise RuntimeError(f"data validation failed: {len(violations)} violations")

    record_end_for_command = args.record_end if args.record_end is not None else max_record
    n_files = record_end_for_command - args.record_start + 1
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "execute": args.execute,
        "base_eval": str(base_eval),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "optimizer_exe": str(optimizer_exe),
        "merge_exe": str(merge_exe),
        "record_start": args.record_start,
        "record_end": record_end_for_command,
        "n_files": n_files,
        "total_records": total_records,
        "total_bytes": total_bytes,
        "phase_counts": phase_counts,
        "dim": args.dim,
        "fm_phases": args.fm_phases,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_records": args.max_records,
        "scale": args.scale,
        "seed": args.seed,
        "active_pattern_mask": f"0x{args.active_pattern_mask:04x}",
        "center_phase_target": args.center_phase_target,
        "l2": args.l2,
        "error_clip": args.error_clip,
        "vector_clip": args.vector_clip,
        "init_std": args.init_std,
        "target_clip": args.target_clip,
        "score_clip": args.score_clip,
        "commands": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    phase_out_dir.mkdir(parents=True, exist_ok=True)

    for phase in range(args.phase_start, args.phase_end + 1):
        count = phase_counts[phase]
        estimated_gib = count * (136 + 8) / (1024 ** 3)
        if args.max_estimated_memory_gib > 0.0 and estimated_gib > args.max_estimated_memory_gib:
            raise RuntimeError(
                f"estimated memory is too large: phase={phase} estimated_gib={estimated_gib:.3f}"
            )
        phase_file = phase_out_dir / f"phase_{phase:02d}.egevfm"
        cmd = [
            str(optimizer_exe),
            str(base_eval),
            str(data_root / str(phase)),
            str(args.record_start),
            str(n_files),
            str(phase_file),
            str(args.dim),
            str(args.fm_phases),
            str(args.epochs),
            format_float(args.lr),
            str(args.max_records),
            str(args.scale),
            str(args.seed + phase),
            f"0x{args.active_pattern_mask:04x}",
            str(args.center_phase_target),
            format_float(args.l2),
            format_float(args.error_clip),
            format_float(args.vector_clip),
            format_float(args.init_std),
            format_float(args.target_clip),
            str(args.score_clip),
            str(phase),
        ]
        manifest["commands"].append({"phase": phase, "records": count, "estimated_memory_gib": estimated_gib, "cmd": cmd})
        if args.skip_existing and phase_file.exists():
            print(f"skip existing phase {phase}: {phase_file}")
            continue
        code = run_or_print(cmd, log_dir / f"phase_{phase:02d}.stdout.log", log_dir / f"phase_{phase:02d}.stderr.log", args.execute)
        if code != 0:
            raise RuntimeError(f"optimizer failed: phase={phase} exit_code={code}")

    if not args.no_merge and args.phase_start == 0 and args.phase_end == N_PHASES - 1:
        merge_cmd = [str(merge_exe), str(final_file)]
        for phase in range(N_PHASES):
            merge_cmd.extend([str(phase), str(phase_out_dir / f"phase_{phase:02d}.egevfm")])
        manifest["merge_command"] = merge_cmd
        code = run_or_print(merge_cmd, log_dir / "merge.stdout.log", log_dir / "merge.stderr.log", args.execute)
        if code != 0:
            raise RuntimeError(f"merge failed: exit_code={code}")

    if args.execute:
        parse_training_logs(log_dir, out_dir, args.phase_start, args.phase_end)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    if not args.execute:
        print("dry run only; add --execute after experiment approval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
