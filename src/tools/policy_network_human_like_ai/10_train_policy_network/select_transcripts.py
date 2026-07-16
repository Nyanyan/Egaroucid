#!/usr/bin/env python3
"""
Select random games from transcript_release/0002 for the human-like policy study.

The selected transcripts are written under:
  train_data/transcript/Egaroucid_Train_Data_v2_selected

Directory names in the source data are random-opening depths. The output keeps
the same depth directory names so conversion can skip the random opening moves.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import random
import shutil
from typing import Dict, Iterable, List, Tuple


GAMES_PER_SOURCE_FILE = 10000


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_source_root() -> Path:
    return repo_root() / "train_data" / "transcript_release" / "0002"


def default_output_root() -> Path:
    return repo_root() / "train_data" / "transcript" / "Egaroucid_Train_Data_v2_selected"


def discover_source_files(source_root: Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for path in source_root.rglob("*.txt"):
        if not path.parent.name.isdigit():
            continue
        files.append((int(path.parent.name), path))
    files.sort(key=lambda item: (item[0], str(item[1])))
    if not files:
        raise FileNotFoundError(f"no transcript txt files found under {source_root}")
    return files


def draw_unique_locations(n_games: int, n_files: int, rng: random.Random) -> List[Tuple[int, int]]:
    total_slots = n_files * GAMES_PER_SOURCE_FILE
    if n_games > total_slots:
        raise ValueError(f"requested {n_games} games, but only {total_slots} slots are available")
    locations = rng.sample(range(total_slots), n_games)
    locations.sort()
    return [(slot // GAMES_PER_SOURCE_FILE, slot % GAMES_PER_SOURCE_FILE) for slot in locations]


def read_selected_lines(path: Path, line_indices: Iterable[int]) -> Dict[int, str]:
    wanted = set(line_indices)
    result: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx in wanted:
                result[line_idx] = line.strip()
                if len(result) == len(wanted):
                    break
    missing = wanted - set(result)
    if missing:
        raise ValueError(f"{path} did not contain selected line indices: {sorted(missing)[:5]}")
    return result


def write_selected_games(
    selected: Dict[int, List[Tuple[int, str, Path]]],
    output_root: Path,
    games_per_output_file: int,
) -> List[dict]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    global_idx = 0
    for random_depth in sorted(selected):
        depth_dir = output_root / str(random_depth)
        depth_dir.mkdir(parents=True, exist_ok=True)
        file_idx = 0
        line_in_file = 0
        out = None
        try:
            for source_line_idx, transcript, source_path in selected[random_depth]:
                if out is None or line_in_file >= games_per_output_file:
                    if out is not None:
                        out.close()
                    out_path = depth_dir / f"{file_idx:07d}.txt"
                    out = out_path.open("w", encoding="utf-8", newline="\n")
                    file_idx += 1
                    line_in_file = 0
                out.write(transcript + "\n")
                manifest_rows.append(
                    {
                        "selected_index": global_idx,
                        "random_depth": random_depth,
                        "source_file": str(source_path.relative_to(repo_root())),
                        "source_line": source_line_idx,
                        "transcript_length": len(transcript) // 2,
                    }
                )
                global_idx += 1
                line_in_file += 1
        finally:
            if out is not None:
                out.close()
    return manifest_rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select random games from transcript_release/0002.")
    parser.add_argument("--source-root", type=Path, default=default_source_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--games-per-output-file", type=int, default=10000)
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    rng = random.Random(args.seed)
    source_files = discover_source_files(args.source_root)
    locations = draw_unique_locations(args.num_games, len(source_files), rng)

    by_file: Dict[int, List[int]] = defaultdict(list)
    for file_idx, line_idx in locations:
        by_file[file_idx].append(line_idx)

    selected: Dict[int, List[Tuple[int, str, Path]]] = defaultdict(list)
    for file_idx, line_indices in sorted(by_file.items()):
        random_depth, path = source_files[file_idx]
        lines = read_selected_lines(path, line_indices)
        for line_idx in sorted(lines):
            selected[random_depth].append((line_idx, lines[line_idx], path))

    manifest_rows = write_selected_games(selected, args.output_root, args.games_per_output_file)
    write_csv(args.output_root / "selection_manifest.csv", manifest_rows)
    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "num_source_files": len(source_files),
        "num_games": args.num_games,
        "seed": args.seed,
        "games_per_source_file_assumption": GAMES_PER_SOURCE_FILE,
        "random_depth_counts": {str(k): len(v) for k, v in sorted(selected.items())},
    }
    with (args.output_root / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
