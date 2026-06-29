import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_records(value):
    records = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("records"):
            part = part[len("records") :]
        records.append(int(part))
    return records


def parse_phase_counts(stderr_text):
    counts = {}
    for line in stderr_text.splitlines():
        match = re.fullmatch(r"phase\s+(\d+)\s+(\d+)", line.strip())
        if match:
            counts[int(match.group(1))] = int(match.group(2))
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-per-phase", type=int, default=5000)
    parser.add_argument("--min-per-phase", type=int, default=0)
    parser.add_argument("--use-n-moves-min", type=int, default=0)
    parser.add_argument("--use-n-moves-max", type=int, default=60)
    parser.add_argument(
        "--converter",
        default="src/tools/evaluation/data_board_to_idx_multi_phase_experiment_edax_linear.exe",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    records_root = Path(args.records_root)
    if not records_root.is_absolute():
        records_root = (repo / records_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo / output_root).resolve()
    converter = Path(args.converter)
    if not converter.is_absolute():
        converter = (repo / converter).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "conversion_summary.tsv"
    records = parse_records(args.records)

    with summary_path.open("w", encoding="utf-8", newline="\n") as summary:
        summary.write("record\treturn_code\tphase_counts\n")
        for record in records:
            input_dir = records_root / f"records{record}"
            if not input_dir.exists():
                print(f"missing records directory: {input_dir}", file=sys.stderr)
                return 1
            output_name = f"records{record}.dat"
            cmd = [
                str(converter),
                str(input_dir),
                "0",
                "1000000",
                str(output_root),
                output_name,
                str(args.max_per_phase),
                str(args.use_n_moves_min),
                str(args.use_n_moves_max),
                str(args.min_per_phase),
            ]
            proc = subprocess.run(
                cmd,
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with (output_root / f"records{record}.stdout.log").open(
                "w", encoding="utf-8", newline="\n"
            ) as f:
                f.write(proc.stdout)
            with (output_root / f"records{record}.stderr.log").open(
                "w", encoding="utf-8", newline="\n"
            ) as f:
                f.write(proc.stderr)
            counts = parse_phase_counts(proc.stderr)
            counts_text = ",".join(
                f"{phase}:{counts.get(phase, 0)}" for phase in range(60)
            )
            summary.write(f"records{record}\t{proc.returncode}\t{counts_text}\n")
            summary.flush()
            print(
                f"records{record} rc={proc.returncode} "
                f"min={min(counts.values()) if counts else 0} "
                f"max={max(counts.values()) if counts else 0}",
                flush=True,
            )
            if proc.returncode != 0:
                return proc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
