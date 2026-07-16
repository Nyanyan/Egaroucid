#!/usr/bin/env python3
"""
Evaluate policy-network agreement with WTHOR human moves.

This evaluator uses all WTHOR board-data records by default. It reports exact
top-N accuracy and symmetry-aware top-N accuracy. Symmetry-aware accuracy treats
moves as equivalent when a board symmetry leaves both player/opponent bitboards
unchanged and maps the human move to the predicted move.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BOARD_RECORD_SIZE = 19
HW = 8
HW2 = 64
HW2_M1 = 63
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
assert BOARD_DTYPE.itemsize == BOARD_RECORD_SIZE

BIT_MASKS = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)).reshape(1, HW2)
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


def policy_to_xy(policy: int) -> Tuple[int, int]:
    pos = HW2_M1 - policy
    return pos % HW, pos // HW


def xy_to_policy(x: int, y: int) -> int:
    return HW2_M1 - (y * HW + x)


def make_transform_maps() -> List[np.ndarray]:
    maps = []
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (HW - 1 - x, y),
        lambda x, y: (x, HW - 1 - y),
        lambda x, y: (HW - 1 - x, HW - 1 - y),
        lambda x, y: (y, x),
        lambda x, y: (HW - 1 - y, HW - 1 - x),
        lambda x, y: (HW - 1 - y, x),
        lambda x, y: (y, HW - 1 - x),
    )
    for transform in transforms:
        mapping = np.empty(HW2, dtype=np.int64)
        for policy in range(HW2):
            x, y = policy_to_xy(policy)
            tx, ty = transform(x, y)
            mapping[policy] = xy_to_policy(tx, ty)
        maps.append(mapping)
    return maps


TRANSFORM_MAPS = make_transform_maps()


def transform_bitboards(bits: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    result = np.zeros_like(bits, dtype=np.uint64)
    for old_policy, new_policy in enumerate(mapping):
        old_bit = np.uint64(1) << np.uint64(old_policy)
        new_bit = np.uint64(1) << np.uint64(new_policy)
        result |= np.where((bits & old_bit) != 0, new_bit, np.uint64(0))
    return result


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
    return Path(os.environ["EGAROUCID_DATA"]) / "train_data" / "board_data" / "records1"


def discover_dat_files(board_data_dir: Path) -> List[Path]:
    files = sorted(board_data_dir.glob("*.dat"), key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
    if not files:
        raise FileNotFoundError(f"no .dat files found in {board_data_dir}")
    return files


def calc_legal_batch(player: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    empty = ~(player | opponent)
    legal = np.zeros_like(player, dtype=np.uint64)
    for shift in SHIFT_FUNCS:
        x = shift(player) & opponent
        for _ in range(5):
            x |= shift(x) & opponent
        legal |= shift(x) & empty
    return legal


def records_to_features(records: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    player = records["player"].astype(np.uint64, copy=False)
    opponent = records["opponent"].astype(np.uint64, copy=False)
    policies = records["policy"].astype(np.int64, copy=False)
    x = np.empty((len(records), INPUT_SIZE), dtype=np.float32)
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
    return (legal.reshape(-1, 1) & BIT_MASKS) != 0


def equivalent_policy_mask(player: np.ndarray, opponent: np.ndarray, policies: np.ndarray) -> np.ndarray:
    equiv = np.zeros((len(policies), POLICY_SIZE), dtype=bool)
    rows = np.arange(len(policies))
    equiv[rows, policies] = True
    for mapping in TRANSFORM_MAPS[1:]:
        t_player = transform_bitboards(player, mapping)
        t_opponent = transform_bitboards(opponent, mapping)
        invariant = (t_player == player) & (t_opponent == opponent)
        if np.any(invariant):
            equiv[rows[invariant], mapping[policies[invariant]]] = True
    return equiv


def move_bucket(move_number: int) -> str:
    start = ((max(1, min(60, int(move_number))) - 1) // 10) * 10 + 1
    return f"{start:02d}_{start + 9:02d}"


def iter_record_batches(path: Path, batch_size: int, max_positions: Optional[int]) -> Iterable[np.ndarray]:
    size = path.stat().st_size
    n_records = size // BOARD_RECORD_SIZE
    if size % BOARD_RECORD_SIZE != 0:
        print(f"warning: {path} has trailing bytes and will be truncated")
    if max_positions is not None:
        n_records = min(n_records, max_positions)
    mmap = np.memmap(path, dtype=BOARD_DTYPE, mode="r", shape=(size // BOARD_RECORD_SIZE,))
    for start in range(0, n_records, batch_size):
        end = min(n_records, start + batch_size)
        yield np.asarray(mmap[start:end])


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

    total_valid = 0
    invalid_policy = 0
    illegal_label = 0
    exact_hits = {n: 0 for n in n_values}
    symmetric_hits = {n: 0 for n in n_values}
    bucket_positions = {move_bucket(i): 0 for i in range(1, 61, 10)}
    bucket_hits = {bucket: {n: 0 for n in n_values} for bucket in bucket_positions}

    remaining = args.max_positions
    for dat_file in discover_dat_files(args.board_data_dir):
        if remaining is not None and remaining <= 0:
            break
        for records in iter_record_batches(dat_file, args.batch_size, remaining):
            if len(records) == 0:
                continue
            if remaining is not None:
                remaining -= len(records)
            x, policies, legal, player, opponent, move_numbers = records_to_features(records)
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
            exact_label_prob = probabilities[np.arange(len(policies)), policies]
            exact_rank = 1 + np.count_nonzero((probabilities > exact_label_prob.reshape(-1, 1)) & legal_mask, axis=1)
            equiv_mask = equivalent_policy_mask(player, opponent, policies)
            equiv_prob = np.max(np.where(equiv_mask, probabilities, -1.0), axis=1)
            symmetric_rank = 1 + np.count_nonzero((probabilities > equiv_prob.reshape(-1, 1)) & legal_mask, axis=1)

            for n in n_values:
                exact_hits[n] += int(np.count_nonzero(exact_rank <= n))
                symmetric_hits[n] += int(np.count_nonzero(symmetric_rank <= n))
            for bucket in bucket_positions:
                start, end = [int(v) for v in bucket.split("_")]
                mask = (start <= move_numbers) & (move_numbers <= end)
                bucket_positions[bucket] += int(np.count_nonzero(mask))
                if np.any(mask):
                    for n in n_values:
                        bucket_hits[bucket][n] += int(np.count_nonzero(symmetric_rank[mask] <= n))

            total_valid += len(policies)
            if args.verbose and total_valid and total_valid % args.progress_interval < len(records):
                print(f"evaluated {total_valid} valid positions")

    topn_rows = []
    for n in n_values:
        topn_rows.append(
            {
                "top_n": n,
                "exact_hits": exact_hits[n],
                "symmetric_hits": symmetric_hits[n],
                "positions": total_valid,
                "exact_accuracy": exact_hits[n] / total_valid if total_valid else 0.0,
                "symmetric_accuracy": symmetric_hits[n] / total_valid if total_valid else 0.0,
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
                    "symmetric_hits": hits,
                    "positions": positions,
                    "symmetric_accuracy": hits / positions if positions else 0.0,
                }
            )
    result = {
        "board_data_dir": str(args.board_data_dir),
        "model_source": model_source,
        "positions": total_valid,
        "invalid_policy_records": invalid_policy,
        "illegal_label_records": illegal_label,
        "topn": topn_rows,
        "move_bucket_topn": bucket_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wthor_human_match.json").open("w") as f:
        json.dump(result, f, indent=2)
    write_csv(args.output_dir / "wthor_human_match_topn.csv", topn_rows, ["top_n", "exact_hits", "symmetric_hits", "positions", "exact_accuracy", "symmetric_accuracy"])
    write_csv(args.output_dir / "wthor_human_match_by_move10.csv", bucket_rows, ["move_bucket", "top_n", "symmetric_hits", "positions", "symmetric_accuracy"])
    return result


def parse_top_n(text: str) -> List[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WTHOR human move agreement with symmetry-aware matching.")
    parser.add_argument("--board-data-dir", type=Path, default=default_board_data_dir())
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--top-n", type=parse_top_n, default=parse_top_n("1,2,3,4,5,8,10,16"))
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "wthor_human_match")
    parser.add_argument("--progress-interval", type=int, default=1000000)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    result = evaluate(args)
    print("model_source", result["model_source"])
    print("board_data_dir", result["board_data_dir"])
    print("positions", result["positions"])
    print("invalid_policy_records", result["invalid_policy_records"])
    print("illegal_label_records", result["illegal_label_records"])
    for row in result["topn"]:
        print(
            f"top-{row['top_n']:>2}: exact {row['exact_accuracy'] * 100.0:.3f}% "
            f"symmetric {row['symmetric_accuracy'] * 100.0:.3f}%"
        )
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
