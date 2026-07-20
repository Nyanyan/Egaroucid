#!/usr/bin/env python3
"""
Evaluate policy-network agreement with WTHOR human moves.

This evaluator uses all WTHOR board-data position samples by default. It reports one
board-symmetry-aware top-N accuracy. Moves are equivalent when a board symmetry
leaves both player/opponent bitboards unchanged and maps the human move to the
predicted move.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


HUMAN_LIKE_AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUMAN_LIKE_AI_DIR))

from policy_accuracy import (  # noqa: E402
    equivalent_policy_mask,
    symmetry_aware_policy_ranks,
)


BOARD_SAMPLE_SIZE = 19
HW2 = 64
INPUT_SIZE = 128
POLICY_SIZE = 64

BOARD_DTYPE = np.dtype(
    [
        ("player", "<u8"),
        ("opponent", "<u8"),
        ("color", "i1"),
        ("policy", "i1"),
        ("score", "i1"),
    ],
    align=False,
)
assert BOARD_DTYPE.itemsize == BOARD_SAMPLE_SIZE

BIT_MASKS = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)).reshape(1, HW2)
POLICY_BIT_MASKS = (np.uint64(1) << np.arange(0, HW2, dtype=np.uint64)).reshape(1, HW2)
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
NOT_FILE_A = np.uint64(0x7F7F7F7F7F7F7F7F)
NOT_FILE_H = np.uint64(0xFEFEFEFEFEFEFEFE)


def shift_east(x: np.ndarray) -> np.ndarray:
    return (x & NOT_FILE_H) >> np.uint64(1)


def shift_west(x: np.ndarray) -> np.ndarray:
    return (x & NOT_FILE_A) << np.uint64(1)


def shift_north(x: np.ndarray) -> np.ndarray:
    return x >> np.uint64(8)


def shift_south(x: np.ndarray) -> np.ndarray:
    return (x << np.uint64(8)) & FULL


def shift_ne(x: np.ndarray) -> np.ndarray:
    return (x & NOT_FILE_H) >> np.uint64(9)


def shift_nw(x: np.ndarray) -> np.ndarray:
    return (x & NOT_FILE_A) >> np.uint64(7)


def shift_se(x: np.ndarray) -> np.ndarray:
    return ((x & NOT_FILE_H) << np.uint64(7)) & FULL


def shift_sw(x: np.ndarray) -> np.ndarray:
    return ((x & NOT_FILE_A) << np.uint64(9)) & FULL


SHIFT_FUNCS = (shift_east, shift_west, shift_north, shift_south, shift_ne, shift_nw, shift_se, shift_sw)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class BinaryPolicyNetwork:
    def __init__(self, path: Path):
        self.layers = []
        with path.open("rb") as f:
            magic = f.read(16)
            if magic != b"EGR_POLICY_V1\0\0\0":
                raise ValueError(f"invalid policy weights magic in {path}")
            version, n_layers, input_size, policy_size = struct.unpack("<IIII", f.read(16))
            if version != 1 or input_size != INPUT_SIZE or policy_size != POLICY_SIZE or n_layers <= 0:
                raise ValueError(f"unsupported policy weights header in {path}")
            expected_input = input_size
            for _ in range(n_layers):
                layer, expected_input = self._read_layer(f, expected_input)
                self.layers.append(layer)

    @staticmethod
    def _read_layer(f, expected_input: int):
        in_dim, out_dim, activation, alpha = struct.unpack("<IIIf", f.read(16))
        if in_dim != expected_input or out_dim <= 0:
            raise ValueError(f"invalid layer shape {in_dim}x{out_dim}")
        n_weights = in_dim * out_dim
        weights = np.frombuffer(f.read(n_weights * 4), dtype="<f4").copy().reshape(in_dim, out_dim)
        bias = np.frombuffer(f.read(out_dim * 4), dtype="<f4").copy()
        return (weights, bias, activation, alpha), out_dim

    @staticmethod
    def _forward_layer(x: np.ndarray, layer) -> np.ndarray:
        weights, bias, activation, alpha = layer
        y = x @ weights + bias
        if activation == 1:
            y = np.where(y >= 0.0, y, y * alpha)
        return y

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.layers:
            y = self._forward_layer(y, layer)
        return softmax(y)


def load_keras_model(path: Path):
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf.keras.models.load_model(path)


def default_board_data_dir() -> Path:
    data_root = os.environ.get("EGAROUCID_DATA")
    if not data_root:
        raise ValueError(
            "--board-data-dir is required when EGAROUCID_DATA is not set"
        )
    return Path(data_root) / "train_data" / "board_data" / "records1"


def discover_dat_files(board_data_dir: Path) -> List[Path]:
    files = sorted(board_data_dir.glob("*.dat"), key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
    if not files:
        raise FileNotFoundError(f"no .dat files found in {board_data_dir}")
    return files


def count_position_samples(
    dat_files: Sequence[Path],
    max_positions: Optional[int],
) -> int:
    total = 0
    for path in dat_files:
        positions = path.stat().st_size // BOARD_SAMPLE_SIZE
        if max_positions is not None:
            positions = min(positions, max_positions - total)
        if positions <= 0:
            break
        total += positions
    return total


def split_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("total split size must be positive")
    if ratio_sum <= 0.0:
        raise ValueError("train/validation/test ratios must have a positive sum")
    n_train = int(np.floor(total * (train_ratio / ratio_sum)))
    n_val = int(np.floor(total * (val_ratio / ratio_sum)))
    n_test = total - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"split produced an empty part: "
            f"train={n_train} val={n_val} test={n_test}"
        )
    return n_train, n_val, n_test


def choose_data_split_positions(
    total_positions: int,
    data_split: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> Tuple[Optional[np.ndarray], int]:
    if data_split == "all":
        return None, total_positions

    n_train, n_val, n_test = split_counts(
        total_positions,
        train_ratio,
        val_ratio,
        test_ratio,
    )
    order = np.random.default_rng(split_seed).permutation(total_positions)
    split_ranges = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }
    start, end = split_ranges[data_split]
    selected = np.sort(order[start:end].astype(np.int64, copy=False))
    return selected, len(selected)


def calc_legal_batch(player: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    empty = ~(player | opponent)
    legal = np.zeros_like(player, dtype=np.uint64)
    for shift in SHIFT_FUNCS:
        x = shift(player) & opponent
        for _ in range(5):
            x |= shift(x) & opponent
        legal |= shift(x) & empty
    return legal


def position_samples_to_features(position_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    player = position_samples["player"].astype(np.uint64, copy=False)
    opponent = position_samples["opponent"].astype(np.uint64, copy=False)
    policies = position_samples["policy"].astype(np.int64, copy=False)
    x = np.empty((len(position_samples), INPUT_SIZE), dtype=np.float32)
    x[:, :HW2] = ((player.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    x[:, HW2:] = ((opponent.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    legal = calc_legal_batch(player, opponent)
    move_numbers = np.count_nonzero(x, axis=1).astype(np.int64) - 3
    return x, policies, legal, player, opponent, move_numbers


def predict_batch(model, binary_network: Optional[BinaryPolicyNetwork], x: np.ndarray, batch_size: int) -> np.ndarray:
    if model is not None:
        return model.predict(x, batch_size=batch_size, verbose=0)
    if binary_network is None:
        raise RuntimeError("no model loaded")
    return binary_network.predict(x)


def valid_policy_mask(policies: np.ndarray, legal: np.ndarray) -> Tuple[np.ndarray, int, int]:
    valid_policy = (0 <= policies) & (policies < POLICY_SIZE)
    label_bits = np.zeros_like(legal, dtype=np.uint64)
    label_bits[valid_policy] = np.left_shift(np.uint64(1), policies[valid_policy].astype(np.uint64))
    legal_label = valid_policy & ((legal & label_bits) != 0)
    return legal_label, int(np.count_nonzero(~valid_policy)), int(np.count_nonzero(valid_policy & ~legal_label))


def legal_bitboard_to_mask(legal: np.ndarray) -> np.ndarray:
    return (legal.reshape(-1, 1) & POLICY_BIT_MASKS) != 0


def move_bucket(move_number: int) -> str:
    start = ((max(1, min(60, int(move_number))) - 1) // 10) * 10 + 1
    return f"{start:02d}_{start + 9:02d}"


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


def iter_position_batches_for_selected_globals(
    path: Path,
    file_global_start: int,
    available_positions: int,
    selected_global_positions: np.ndarray,
    batch_size: int,
) -> Iterable[np.ndarray]:
    size = path.stat().st_size
    file_positions = min(
        size // BOARD_SAMPLE_SIZE,
        max(0, available_positions - file_global_start),
    )
    file_global_end = file_global_start + file_positions
    left = int(
        np.searchsorted(
            selected_global_positions,
            file_global_start,
            side="left",
        )
    )
    right = int(
        np.searchsorted(
            selected_global_positions,
            file_global_end,
            side="left",
        )
    )
    if left == right:
        return
    local_indices = (
        selected_global_positions[left:right] - file_global_start
    )
    mmap = np.memmap(
        path,
        dtype=BOARD_DTYPE,
        mode="r",
        shape=(size // BOARD_SAMPLE_SIZE,),
    )
    for start in range(0, len(local_indices), batch_size):
        yield np.asarray(
            mmap[local_indices[start : start + batch_size]]
        )


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict:
    n_values = sorted(set(args.top_n))
    model = load_keras_model(args.model) if args.model is not None and args.model.exists() else None
    binary_network = None if model is not None else BinaryPolicyNetwork(args.weights)
    model_source = str(args.model) if model is not None else str(args.weights)
    dat_files = discover_dat_files(args.board_data_dir)
    available_positions = count_position_samples(
        dat_files,
        args.max_positions,
    )
    selected_global_positions, split_positions = choose_data_split_positions(
        available_positions,
        args.data_split,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.split_seed,
    )
    selected_position_set_sha256 = (
        hashlib.sha256(
            selected_global_positions.astype(
                "<i8",
                copy=False,
            ).tobytes()
        ).hexdigest()
        if selected_global_positions is not None
        else None
    )

    total_valid = 0
    invalid_policy = 0
    illegal_label = 0
    total_hits = {n: 0 for n in n_values}
    bucket_positions = {move_bucket(i): 0 for i in range(1, 61, 10)}
    bucket_hits = {bucket: {n: 0 for n in n_values} for bucket in bucket_positions}

    remaining = available_positions
    file_global_start = 0
    for dat_file in dat_files:
        file_positions = dat_file.stat().st_size // BOARD_SAMPLE_SIZE
        if file_global_start >= available_positions:
            break
        if selected_global_positions is None:
            batches = iter_position_batches(
                dat_file,
                args.batch_size,
                remaining,
            )
        else:
            batches = iter_position_batches_for_selected_globals(
                dat_file,
                file_global_start,
                available_positions,
                selected_global_positions,
                args.batch_size,
            )
        for position_samples in batches:
            if len(position_samples) == 0:
                continue
            if selected_global_positions is None:
                remaining -= len(position_samples)
            x, policies, legal, player, opponent, move_numbers = position_samples_to_features(position_samples)
            probabilities = predict_batch(model, binary_network, x, args.predict_batch_size)
            valid, n_invalid_policy, n_illegal_label = valid_policy_mask(policies, legal)
            invalid_policy += n_invalid_policy
            illegal_label += n_illegal_label
            if not np.any(valid):
                continue
            probabilities = probabilities[valid]
            policies = policies[valid]
            legal = legal[valid]
            player = player[valid]
            opponent = opponent[valid]
            move_numbers = move_numbers[valid]

            legal_mask = legal_bitboard_to_mask(legal)
            equiv_mask = equivalent_policy_mask(player, opponent, policies)
            rank = symmetry_aware_policy_ranks(
                probabilities,
                legal_mask,
                equiv_mask,
            )

            for n in n_values:
                total_hits[n] += int(np.count_nonzero(rank <= n))
            for bucket in bucket_positions:
                start, end = [int(v) for v in bucket.split("_")]
                mask = (start <= move_numbers) & (move_numbers <= end)
                bucket_positions[bucket] += int(np.count_nonzero(mask))
                if np.any(mask):
                    for n in n_values:
                        bucket_hits[bucket][n] += int(np.count_nonzero(rank[mask] <= n))

            total_valid += len(policies)
            if args.verbose and total_valid and total_valid % args.progress_interval < len(position_samples):
                print(f"evaluated {total_valid} valid positions")
        file_global_start += file_positions

    topn_rows = []
    for n in n_values:
        hits = total_hits[n]
        accuracy = hits / total_valid if total_valid else 0.0
        topn_rows.append(
            {
                "top_n": n,
                "hits": hits,
                "positions": total_valid,
                "accuracy": accuracy,
            }
        )
    bucket_rows = []
    for bucket in sorted(bucket_positions):
        positions = bucket_positions[bucket]
        for n in n_values:
            hits = bucket_hits[bucket][n]
            bucket_rows.append(
                {
                    "move_bucket": bucket,
                    "top_n": n,
                    "hits": hits,
                    "positions": positions,
                    "accuracy": hits / positions if positions else 0.0,
                }
            )
    result = {
        "board_data_dir": str(args.board_data_dir),
        "model_source": model_source,
        "data_split": args.data_split,
        "split_seed": (
            args.split_seed
            if args.data_split != "all"
            else None
        ),
        "split_ratios": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": args.test_ratio,
        },
        "available_positions": available_positions,
        "split_positions": split_positions,
        "selected_position_set_sha256": selected_position_set_sha256,
        "positions": total_valid,
        "agreement_definition": {
            "metric": "board_symmetry_aware",
            "description": (
                "手番側と相手側の石配置をそれぞれ不変に保つ盤面対称変換で、"
                "人間の実着手から移る合法手を同値手とする。"
                "同値手のいずれかが上位N手に入れば一致と数える。"
            ),
        },
        "ranking_definition": {
            "description": (
                "合法手を方策値の降順に一意に並べる。"
                "方策値が同じ場合はpolicy番号の昇順を使用する。"
            ),
            "tie_break": "ascending_policy_index",
        },
        "invalid_policy_samples": invalid_policy,
        "illegal_label_samples": illegal_label,
        "topn": topn_rows,
        "move_bucket_topn": bucket_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wthor_human_match.json").open("w") as f:
        json.dump(result, f, indent=2)
    write_csv(
        args.output_dir / "wthor_human_match_topn.csv",
        topn_rows,
        ["top_n", "hits", "positions", "accuracy"],
    )
    write_csv(
        args.output_dir / "wthor_human_match_by_move10.csv",
        bucket_rows,
        ["move_bucket", "top_n", "hits", "positions", "accuracy"],
    )
    return result


def parse_top_n(text: str) -> List[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WTHOR human move agreement with symmetry-aware matching.")
    parser.add_argument("--board-data-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--top-n", type=parse_top_n, default=parse_top_n("1,2,3,4,5,8,10,16"))
    parser.add_argument(
        "--data-split",
        choices=("all", "train", "val", "test"),
        default="all",
    )
    parser.add_argument("--split-seed", type=int, default=613)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "wthor_human_match")
    parser.add_argument("--progress-interval", type=int, default=1000000)
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = make_argparser()
    args = parser.parse_args(argv)
    if args.board_data_dir is None:
        try:
            args.board_data_dir = default_board_data_dir()
        except ValueError as error:
            parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print("model_source", result["model_source"])
    print("board_data_dir", result["board_data_dir"])
    print("data_split", result["data_split"])
    print("split_positions", result["split_positions"])
    if result["selected_position_set_sha256"] is not None:
        print(
            "selected_position_set_sha256",
            result["selected_position_set_sha256"],
        )
    print("positions", result["positions"])
    print("invalid_policy_samples", result["invalid_policy_samples"])
    print("illegal_label_samples", result["illegal_label_samples"])
    for row in result["topn"]:
        print(
            f"top-{row['top_n']:>2}: symmetry-aware "
            f"{row['accuracy'] * 100.0:.3f}%"
        )
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
