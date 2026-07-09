import argparse
from pathlib import Path


def parse_ids(text):
    if not text:
        return []
    return [int(elem.strip().removeprefix("records").removeprefix("batch")) for elem in text.split(",") if elem.strip()]


def find_input_file(input_phase_dir, train_id):
    candidates = [
        input_phase_dir / f"{train_id}.dat",
        input_phase_dir / f"records{train_id}.dat",
        input_phase_dir / f"batch{train_id}.dat",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--n-features", type=int, default=47)
    parser.add_argument("--block-records", type=int, default=8192)
    parser.add_argument("--max-train-file-bytes", type=int, default=0)
    args = parser.parse_args()

    if args.n_batches <= 0:
        raise ValueError("--n-batches must be positive")
    if args.n_features <= 0:
        raise ValueError("--n-features must be positive")
    if args.block_records <= 0:
        raise ValueError("--block-records must be positive")

    repo = Path(__file__).resolve().parents[3]
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = (repo / input_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo / output_root).resolve()

    input_phase_dir = input_root / str(args.phase)
    output_phase_dir = output_root / str(args.phase)
    output_phase_dir.mkdir(parents=True, exist_ok=True)

    record_size = 2 * (2 + args.n_features + 1)
    block_size = record_size * args.block_records
    train_ids = parse_ids(args.train_ids)
    if not train_ids:
        raise RuntimeError("no train IDs")

    writers = []
    batch_records = [0 for _ in range(args.n_batches)]
    batch_bytes = [0 for _ in range(args.n_batches)]
    used_files = []
    skipped_files = []
    try:
        for batch_idx in range(args.n_batches):
            writers.append((output_phase_dir / f"{batch_idx}.dat").open("wb"))

        next_batch = 0
        for train_id in train_ids:
            path = find_input_file(input_phase_dir, train_id)
            if not path.exists() or path.stat().st_size <= 0:
                skipped_files.append((train_id, "missing_or_empty"))
                continue
            size = path.stat().st_size
            if args.max_train_file_bytes > 0 and size > args.max_train_file_bytes:
                skipped_files.append((train_id, f"too_large:{size}"))
                continue
            if size % record_size != 0:
                raise ValueError(f"{path} size {size} is not divisible by record size {record_size}")

            file_records = 0
            with path.open("rb") as src:
                while True:
                    chunk = src.read(block_size)
                    if not chunk:
                        break
                    if len(chunk) % record_size != 0:
                        raise ValueError(f"{path} ended with a partial record block")
                    writers[next_batch].write(chunk)
                    n_records = len(chunk) // record_size
                    batch_records[next_batch] += n_records
                    batch_bytes[next_batch] += len(chunk)
                    file_records += n_records
                    next_batch = (next_batch + 1) % args.n_batches
            used_files.append((train_id, size, file_records, path.name))
    finally:
        for writer in writers:
            writer.close()

    summary_path = output_root / f"phase{args.phase}_minibatch_summary.txt"
    with summary_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"phase={args.phase}\n")
        f.write(f"n_features={args.n_features}\n")
        f.write(f"record_size={record_size}\n")
        f.write(f"block_records={args.block_records}\n")
        f.write(f"n_batches={args.n_batches}\n")
        f.write(f"input_root={input_root}\n")
        f.write(f"output_root={output_root}\n")
        f.write(f"train_ids={','.join(str(elem) for elem in train_ids)}\n")
        f.write(f"max_train_file_bytes={args.max_train_file_bytes}\n")
        f.write("used_files=" + ",".join(f"{tid}:{size}:{records}:{name}" for tid, size, records, name in used_files) + "\n")
        f.write("skipped_files=" + ",".join(f"{tid}:{reason}" for tid, reason in skipped_files) + "\n")
        for batch_idx in range(args.n_batches):
            f.write(f"batch{batch_idx}_records={batch_records[batch_idx]}\n")
            f.write(f"batch{batch_idx}_bytes={batch_bytes[batch_idx]}\n")

    print(f"wrote {output_phase_dir}")
    print(f"summary {summary_path}")
    for batch_idx in range(args.n_batches):
        print(f"batch {batch_idx} records {batch_records[batch_idx]} bytes {batch_bytes[batch_idx]}")


if __name__ == "__main__":
    main()
