#!/usr/bin/env python3
"""Inspect Egaroucid board_data files for NNUE training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RECORD_BYTES = 19
N_PHASES = 60
RECORD_DTYPE = np.dtype([
    ("player", "<u8"),
    ("opponent", "<u8"),
    ("player_color", "i1"),
    ("policy", "i1"),
    ("score", "i1"),
])
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def parse_record_num(path: Path) -> int | None:
    if not path.name.startswith("records"):
        return None
    suffix = path.name[len("records"):]
    return int(suffix) if suffix.isdigit() else None


def popcount_u64(values: np.ndarray) -> np.ndarray:
    return POPCOUNT8[values.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="E:/egaroucid_data/train_data/board_data")
    parser.add_argument("--record-start", type=int, default=223)
    parser.add_argument("--record-end", type=int, default=-1)
    parser.add_argument("--out", default="")
    parser.add_argument("--progress-interval-sec", type=float, default=30.0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    files: list[tuple[int, Path, int]] = []
    for record_dir in sorted(data_root.iterdir(), key=lambda p: parse_record_num(p) or -1):
        if not record_dir.is_dir():
            continue
        record_num = parse_record_num(record_dir)
        if record_num is None or record_num < args.record_start:
            continue
        if args.record_end >= 0 and record_num > args.record_end:
            continue
        for path in sorted(record_dir.glob("*.dat"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            records = path.stat().st_size // RECORD_BYTES
            if records > 0:
                files.append((record_num, path, records))

    phase_counts = np.zeros(N_PHASES, dtype=np.uint64)
    total_records = 0
    invalid_phase_records = 0
    start_ms = time.time()
    last_progress = start_ms
    for i, (record_num, path, records) in enumerate(files, start=1):
        raw = np.fromfile(path, dtype=RECORD_DTYPE)
        phases = popcount_u64(raw["player"] | raw["opponent"]).astype(np.int16) - 4
        valid = (phases >= 0) & (phases < N_PHASES)
        invalid_phase_records += int((~valid).sum())
        if np.any(valid):
            phase_counts += np.bincount(phases[valid].astype(np.uint8), minlength=N_PHASES).astype(np.uint64)
        total_records += int(records)
        now = time.time()
        if now - last_progress >= args.progress_interval_sec or i == len(files):
            elapsed = now - start_ms
            print(
                f"progress files {i}/{len(files)} records {total_records} "
                f"record {record_num} elapsed_sec {elapsed:.1f}",
                flush=True,
            )
            last_progress = now

    result = {
        "data_root": str(data_root.resolve()),
        "record_start": args.record_start,
        "record_end": args.record_end,
        "files": len(files),
        "total_records": int(total_records),
        "invalid_phase_records": int(invalid_phase_records),
        "phase_counts": [int(x) for x in phase_counts],
    }
    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
