#!/usr/bin/env python3
"""
Evaluate WTHOR human-move agreement for blended policy/Egaroucid output.

This script is intentionally slower than pure policy evaluation because it asks
Egaroucid for Console for hint scores on each position.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BLEND_DIR = SCRIPT_DIR.parents[0] / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import (  # noqa: E402
    BLACK,
    POLICY_SIZE,
    WHITE,
    BlendedPolicy,
    BoardState,
    default_egaroucid_exe,
    default_weights_file,
)
from evaluate_wthor_human_match import (  # noqa: E402
    BOARD_DTYPE,
    BOARD_SAMPLE_SIZE,
    equivalent_policy_mask,
    move_bucket,
)


def default_board_data_dir() -> Path:
    return Path(os.environ["EGAROUCID_DATA"]) / "train_data" / "board_data" / "records1"


def discover_dat_files(board_data_dir: Path) -> List[Path]:
    files = sorted(board_data_dir.glob("*.dat"), key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
    if not files:
        raise FileNotFoundError(f"no .dat files found in {board_data_dir}")
    return files


def count_position_samples(dat_files: Sequence[Path], max_positions: Optional[int]) -> int:
    total = 0
    for path in dat_files:
        n = path.stat().st_size // BOARD_SAMPLE_SIZE
        if max_positions is not None:
            n = min(n, max_positions - total)
        if n <= 0:
            break
        total += n
    return total


def choose_global_positions(total_positions: int, sample_positions: Optional[int], seed: int) -> Optional[np.ndarray]:
    if sample_positions is None:
        return None
    if total_positions <= 0:
        return np.empty(0, dtype=np.int64)
    n = min(sample_positions, total_positions)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_positions, size=n, replace=False).astype(np.int64, copy=False))


def iter_position_batches(path: Path, batch_size: int, max_positions: Optional[int]) -> Iterable[np.ndarray]:
    size = path.stat().st_size
    n_position_samples = size // BOARD_SAMPLE_SIZE
    if size % BOARD_SAMPLE_SIZE != 0:
        print(f"warning: {path} has trailing bytes and will be truncated")
    if max_positions is not None:
        n_position_samples = min(n_position_samples, max_positions)
    mmap = np.memmap(path, dtype=BOARD_DTYPE, mode="r", shape=(size // BOARD_SAMPLE_SIZE,))
    for start in range(0, n_position_samples, batch_size):
        end = min(n_position_samples, start + batch_size)
        yield np.asarray(mmap[start:end])


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def position_sample_to_state(position_sample) -> Tuple[BoardState, int, int, int, int, int]:
    player = int(position_sample["player"])
    opponent = int(position_sample["opponent"])
    side = int(position_sample["color"])
    policy = int(position_sample["policy"])
    if side == BLACK:
        black, white = player, opponent
    elif side == WHITE:
        black, white = opponent, player
    else:
        raise ValueError(f"invalid side color: {side}")
    move_number = popcount(player | opponent) - 3
    return BoardState(black, white, side), side, policy, player, opponent, move_number


def rank_for_distribution(distribution: np.ndarray, legal_policies: Sequence[int], target_policies: Sequence[int]) -> int:
    if not legal_policies:
        return POLICY_SIZE + 1
    legal = np.array(legal_policies, dtype=np.int64)
    targets = [policy for policy in target_policies if policy in legal_policies]
    if not targets:
        return POLICY_SIZE + 1
    target_prob = float(np.max(distribution[np.array(targets, dtype=np.int64)]))
    return 1 + int(np.count_nonzero(distribution[legal] > target_prob))


def equivalent_targets(player: int, opponent: int, policy: int) -> List[int]:
    mask = equivalent_policy_mask(
        np.array([player], dtype=np.uint64),
        np.array([opponent], dtype=np.uint64),
        np.array([policy], dtype=np.int64),
    )[0]
    return [int(i) for i in np.nonzero(mask)[0]]


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(text: str) -> List[float]:
    return [float(token) for token in text.split(",") if token.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def evaluate(args: argparse.Namespace) -> dict:
    blend_params = parse_float_list(args.blend_params)
    n_values = sorted(set(parse_int_list(args.top_n)))
    dat_files = discover_dat_files(args.board_data_dir)
    available_positions = count_position_samples(dat_files, args.max_positions)
    selected_global_positions = choose_global_positions(available_positions, args.sample_positions, args.sample_seed)
    bucket_names = [move_bucket(i) for i in range(1, 61, 10)]
    metrics = {
        blend: {
            "positions": 0,
            "exact_hits": {n: 0 for n in n_values},
            "symmetric_hits": {n: 0 for n in n_values},
            "bucket_positions": {bucket: 0 for bucket in bucket_names},
            "bucket_hits": {bucket: {n: 0 for n in n_values} for bucket in bucket_names},
        }
        for blend in blend_params
    }

    invalid_policy = 0
    illegal_label = 0
    raw_hint_samples = []
    blender = BlendedPolicy(
        weights=args.weights,
        egaroucid_exe=args.egaroucid_exe,
        egaroucid_level=args.egaroucid_level,
        egaroucid_threads=args.egaroucid_threads,
        egaroucid_timeout_sec=args.egaroucid_timeout_sec,
        score_temperature=args.score_temperature,
        legal_mask_policy=not args.no_legal_mask_policy,
    )

    remaining = args.max_positions
    total_seen = 0
    global_offset = 0
    for dat_file in dat_files:
        if remaining is not None and remaining <= 0:
            break
        for position_samples in iter_position_batches(dat_file, args.batch_size, remaining):
            if len(position_samples) == 0:
                continue
            if remaining is not None:
                remaining -= len(position_samples)
            batch_start = global_offset
            batch_end = batch_start + len(position_samples)
            global_offset = batch_end
            if selected_global_positions is not None:
                left = int(np.searchsorted(selected_global_positions, batch_start, side="left"))
                right = int(np.searchsorted(selected_global_positions, batch_end, side="left"))
                if left == right:
                    continue
                local_indices = selected_global_positions[left:right] - batch_start
                position_samples = position_samples[local_indices]
            for position_sample in position_samples:
                total_seen += 1
                state, side, policy, player, opponent, move_number = position_sample_to_state(position_sample)
                legal_policies = state.legal_policies(side)
                if policy < 0 or policy >= POLICY_SIZE:
                    invalid_policy += 1
                    continue
                if policy not in legal_policies:
                    illegal_label += 1
                    continue

                policy_dist = blender.policy_distribution(state, side, legal_policies)
                scores, raw_hint = blender.hint_runner.hint_scores(state, side)
                egaroucid_dist = blender.egaroucid_distribution(scores, legal_policies)
                if args.raw_hint_samples > 0 and len(raw_hint_samples) < args.raw_hint_samples:
                    raw_hint_samples.append({"index": total_seen - 1, "raw_hint": raw_hint})

                exact_targets = [policy]
                symmetric_targets = equivalent_targets(player, opponent, policy)
                bucket = move_bucket(move_number)

                for blend in blend_params:
                    distribution = policy_dist * blend + egaroucid_dist * (1.0 - blend)
                    total = float(np.sum(distribution[np.array(legal_policies, dtype=np.int64)]))
                    if total > 0.0:
                        distribution = distribution / total
                    exact_rank = rank_for_distribution(distribution, legal_policies, exact_targets)
                    symmetric_rank = rank_for_distribution(distribution, legal_policies, symmetric_targets)
                    m = metrics[blend]
                    m["positions"] += 1
                    m["bucket_positions"][bucket] += 1
                    for n in n_values:
                        if exact_rank <= n:
                            m["exact_hits"][n] += 1
                        if symmetric_rank <= n:
                            m["symmetric_hits"][n] += 1
                            m["bucket_hits"][bucket][n] += 1

            if args.verbose and total_seen and total_seen % args.progress_interval < len(position_samples):
                print(f"seen {total_seen} position_samples")

    topn_rows = []
    bucket_rows = []
    for blend in blend_params:
        m = metrics[blend]
        positions = m["positions"]
        for n in n_values:
            topn_rows.append(
                {
                    "blend_param": blend,
                    "top_n": n,
                    "exact_hits": m["exact_hits"][n],
                    "symmetric_hits": m["symmetric_hits"][n],
                    "positions": positions,
                    "exact_accuracy": m["exact_hits"][n] / positions if positions else 0.0,
                    "symmetric_accuracy": m["symmetric_hits"][n] / positions if positions else 0.0,
                }
            )
        for bucket in bucket_names:
            bucket_positions = m["bucket_positions"][bucket]
            for n in n_values:
                hits = m["bucket_hits"][bucket][n]
                bucket_rows.append(
                    {
                        "blend_param": blend,
                        "move_bucket": bucket,
                        "top_n": n,
                        "symmetric_hits": hits,
                        "positions": bucket_positions,
                        "symmetric_accuracy": hits / bucket_positions if bucket_positions else 0.0,
                    }
                )

    result = {
        "board_data_dir": str(args.board_data_dir),
        "weights": str(args.weights),
        "egaroucid_exe": str(args.egaroucid_exe),
        "egaroucid_level": args.egaroucid_level,
        "blend_params": blend_params,
        "available_positions": available_positions,
        "sample_positions": args.sample_positions,
        "sample_seed": args.sample_seed if args.sample_positions is not None else None,
        "invalid_policy_samples": invalid_policy,
        "illegal_label_samples": illegal_label,
        "topn": topn_rows,
        "move_bucket_topn": bucket_rows,
        "raw_hint_samples": raw_hint_samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wthor_blend_human_match.json").open("w") as f:
        json.dump(result, f, indent=2)
    write_csv(
        args.output_dir / "wthor_blend_human_match_topn.csv",
        topn_rows,
        ["blend_param", "top_n", "exact_hits", "symmetric_hits", "positions", "exact_accuracy", "symmetric_accuracy"],
    )
    write_csv(
        args.output_dir / "wthor_blend_human_match_by_move10.csv",
        bucket_rows,
        ["blend_param", "move_bucket", "top_n", "symmetric_hits", "positions", "symmetric_accuracy"],
    )
    return result


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WTHOR agreement for blended policy/Egaroucid output.")
    parser.add_argument("--board-data-dir", type=Path, default=default_board_data_dir())
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--egaroucid-exe", type=Path, default=default_egaroucid_exe())
    parser.add_argument("--egaroucid-level", type=int, default=21)
    parser.add_argument("--egaroucid-threads", type=int, default=1)
    parser.add_argument("--egaroucid-timeout-sec", type=float, default=30.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--no-legal-mask-policy", action="store_true")
    parser.add_argument("--blend-params", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--top-n", default="1,2,3,4,5,8,10,16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--sample-positions", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=613)
    parser.add_argument("--raw-hint-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output" / "wthor_blend_human_match")
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    result = evaluate(args)
    print("weights", result["weights"])
    print("board_data_dir", result["board_data_dir"])
    print("egaroucid_level", result["egaroucid_level"])
    print("available_positions", result["available_positions"])
    if result["sample_positions"] is not None:
        print("sample_positions", result["sample_positions"])
        print("sample_seed", result["sample_seed"])
    print("invalid_policy_samples", result["invalid_policy_samples"])
    print("illegal_label_samples", result["illegal_label_samples"])
    for row in result["topn"]:
        if row["top_n"] == 1:
            print(
                f"blend {row['blend_param']:g} top-1 exact {row['exact_accuracy'] * 100.0:.3f}% "
                f"symmetric {row['symmetric_accuracy'] * 100.0:.3f}% ({row['positions']})"
            )
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
