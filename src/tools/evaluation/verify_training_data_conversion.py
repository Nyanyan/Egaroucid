#!/usr/bin/env python3
"""Verify board-data versus indexed-data counts and phase score histograms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


PHASES = (30, 35, 36, 40, 44)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_board_histograms(report_dir: Path) -> tuple[dict[int, Counter[int]], list[Path]]:
    paths = [
        report_dir / "board_label_histograms.json",
        report_dir / "phase36_sample" / "board_label_histograms.json",
    ]
    result: dict[int, Counter[int]] = defaultdict(Counter)
    used: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        used.append(path)
        for key, histogram in payload.items():
            phase_text = key.split("_", 1)[0]
            if not phase_text.startswith("phase"):
                continue
            phase = int(phase_text[5:])
            result[phase].update({int(score): int(count) for score, count in histogram.items()})
    return dict(result), used


def load_indexed_histograms(path: Path) -> dict[int, Counter[int]]:
    result: dict[int, Counter[int]] = defaultdict(Counter)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["population"] != "records_with_duplicates":
                continue
            result[int(row["phase"])][int(row["score"])] += int(row["count"])
    return dict(result)


def load_count_checks(report_dir: Path) -> tuple[dict[int, list[dict[str, str]]], list[Path]]:
    paths = [
        report_dir / "board_conversion_check.csv",
        report_dir / "phase36_sample" / "board_conversion_check.csv",
    ]
    result: dict[int, list[dict[str, str]]] = defaultdict(list)
    used: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        used.append(path)
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                result[int(row["phase"])].append(row)
    return dict(result), used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    indexed_path = report_dir / "training_dedup_by_id_score_histogram.csv"
    board_histograms, board_hist_paths = load_board_histograms(report_dir)
    indexed_histograms = load_indexed_histograms(indexed_path)
    count_checks, count_paths = load_count_checks(report_dir)

    rows = []
    for phase in PHASES:
        board_hist = board_histograms.get(phase, Counter())
        indexed_hist = indexed_histograms.get(phase, Counter())
        checks = count_checks.get(phase, [])
        differing_scores = sorted(
            score
            for score in set(board_hist) | set(indexed_hist)
            if board_hist[score] != indexed_hist[score]
        )
        rows.append(
            {
                "phase": phase,
                "empties": 60 - phase,
                "phase_id_rows": len(checks),
                "all_phase_id_record_counts_match": bool(checks)
                and all(row.get("record_count_matches") == "True" for row in checks),
                "board_records": sum(board_hist.values()),
                "indexed_records": sum(indexed_hist.values()),
                "phase_score_histogram_matches": bool(board_hist)
                and board_hist == indexed_hist,
                "differing_score_bin_count": len(differing_scores),
                "differing_score_bins": ";".join(map(str, differing_scores)),
                "per_data_id_score_histogram_status": (
                    "not_stored; per-ID record counts and aggregate distribution statistics were checked separately"
                ),
            }
        )

    output_path = report_dir / "board_conversion_verification.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in [indexed_path, *board_hist_paths, *count_paths]
        ],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "scope_note": (
            "Full score-bin equality is checked after summing all selected data IDs within each phase. "
            "Per-data-ID full histograms were not retained."
        ),
    }
    provenance_path = report_dir / "board_conversion_verification.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
