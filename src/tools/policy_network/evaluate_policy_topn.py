#!/usr/bin/env python3
"""
Evaluate masked top-N policy accuracy on WTHOR board-data records.

Related issue: #613

The default target is WTHOR human game data converted to Egaroucid board-data
records. In the training-data tree, this dataset is currently stored in the
records1 directory.
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
assert BOARD_DTYPE.itemsize == BOARD_RECORD_SIZE

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

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf.keras.models.load_model(path)


def default_board_data_dir() -> Path:
    return Path(os.environ["EGAROUCID_DATA"]) / "train_data" / "board_data" / "records1"


def default_model_file() -> Path:
    return Path(__file__).resolve().parent / "trained" / "playerop_final_issue613_128x3" / "best_model.h5"


def default_weights_file() -> Path:
    return Path(__file__).resolve().parent / "trained" / "playerop_final_issue613_128x3" / "best_policy_network_weights.bin"


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


def records_to_features(records: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    player = records["player"].astype(np.uint64, copy=False)
    opponent = records["opponent"].astype(np.uint64, copy=False)
    policies = records["policy"].astype(np.int64, copy=False)

    x = np.empty((len(records), INPUT_SIZE), dtype=np.float32)
    x[:, :HW2] = ((player.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    x[:, HW2:] = ((opponent.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    legal = calc_legal_batch(player, opponent)
    n_discs = np.count_nonzero(x, axis=1)
    return x, policies, legal, n_discs


def legal_bitboard_to_mask(legal: np.ndarray) -> np.ndarray:
    return (legal.reshape(-1, 1) & POLICY_BIT_MASKS) != 0


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


def update_policy_hits(
    probabilities: np.ndarray,
    policies: np.ndarray,
    legal: np.ndarray,
    n_values: Sequence[int],
    total_hits: Dict[int, int],
    total_by_phase: Dict[str, int],
    hits_by_phase: Dict[str, Dict[int, int]],
    n_discs: np.ndarray,
) -> None:
    legal_mask = legal_bitboard_to_mask(legal)
    label_prob = probabilities[np.arange(len(policies)), policies]
    rank = 1 + np.count_nonzero((probabilities > label_prob.reshape(-1, 1)) & legal_mask, axis=1)

    for n in n_values:
        total_hits[n] += int(np.count_nonzero(rank <= n))

    phase_names = phase_names_from_discs(n_discs)
    for phase in ("opening_4_20", "midgame_21_44", "endgame_45_64"):
        phase_mask = phase_names == phase
        phase_total = int(np.count_nonzero(phase_mask))
        total_by_phase[phase] += phase_total
        if phase_total:
            for n in n_values:
                hits_by_phase[phase][n] += int(np.count_nonzero(rank[phase_mask] <= n))


def phase_names_from_discs(n_discs: np.ndarray) -> np.ndarray:
    phase_names = np.empty(len(n_discs), dtype=object)
    phase_names[n_discs <= 20] = "opening_4_20"
    phase_names[(20 < n_discs) & (n_discs <= 44)] = "midgame_21_44"
    phase_names[44 < n_discs] = "endgame_45_64"
    return phase_names


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
        for row in rows:
            writer.writerow(row)


def evaluate(args: argparse.Namespace) -> dict:
    board_data_dir = args.board_data_dir
    dat_files = discover_dat_files(board_data_dir)
    n_values = sorted(set(args.top_n))
    max_n = max(n_values)
    if max_n > POLICY_SIZE:
        raise ValueError("top-n cannot exceed 64")

    model = None
    binary_network = None
    model_source = None
    if args.model is not None and args.model.exists():
        model = load_keras_model(args.model)
        model_source = str(args.model)
    else:
        weights = args.weights
        if weights is None:
            weights = default_weights_file()
        binary_network = BinaryPolicyNetwork(weights)
        model_source = str(weights)

    total_positions = 0
    invalid_policy = 0
    illegal_label = 0
    total_hits = {n: 0 for n in n_values}
    total_by_phase = {"opening_4_20": 0, "midgame_21_44": 0, "endgame_45_64": 0}
    hits_by_phase = {phase: {n: 0 for n in n_values} for phase in total_by_phase}

    remaining = args.max_positions
    for dat_file in dat_files:
        if remaining is not None and remaining <= 0:
            break
        for records in iter_record_batches(dat_file, args.batch_size, remaining):
            if len(records) == 0:
                continue
            if remaining is not None:
                remaining -= len(records)
            x, policies, legal, n_discs = records_to_features(records)
            probabilities = predict_batch(model, binary_network, x, args.predict_batch_size)
            valid, n_invalid_policy, n_illegal_label = valid_policy_mask(policies, legal)
            invalid_policy += n_invalid_policy
            illegal_label += n_illegal_label
            if not np.any(valid):
                continue

            probabilities = probabilities[valid]
            policies = policies[valid]
            legal = legal[valid]
            n_discs = n_discs[valid]
            update_policy_hits(probabilities, policies, legal, n_values, total_hits, total_by_phase, hits_by_phase, n_discs)

            total_positions += int(np.count_nonzero(valid))
            if args.verbose and total_positions and total_positions % args.progress_interval < len(records):
                print(f"evaluated {total_positions} valid positions")

    topn_rows = []
    for n in n_values:
        accuracy = total_hits[n] / total_positions if total_positions else 0.0
        topn_rows.append({"top_n": n, "hits": total_hits[n], "positions": total_positions, "accuracy": accuracy})

    phase_rows = []
    for phase, phase_total in total_by_phase.items():
        for n in n_values:
            accuracy = hits_by_phase[phase][n] / phase_total if phase_total else 0.0
            phase_rows.append({"phase": phase, "top_n": n, "hits": hits_by_phase[phase][n], "positions": phase_total, "accuracy": accuracy})

    result = {
        "board_data_dir": str(board_data_dir),
        "dat_files": [str(p) for p in dat_files],
        "model_source": model_source,
        "positions": total_positions,
        "invalid_policy_records": invalid_policy,
        "illegal_label_records": illegal_label,
        "topn": topn_rows,
        "phase_topn": phase_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wthor_topn_accuracy.json").open("w") as f:
        json.dump(result, f, indent=2)
    write_csv(args.output_dir / "wthor_topn_accuracy.csv", topn_rows, ["top_n", "hits", "positions", "accuracy"])
    write_csv(args.output_dir / "wthor_topn_accuracy_by_phase.csv", phase_rows, ["phase", "top_n", "hits", "positions", "accuracy"])
    return result


def parse_top_n(text: str) -> List[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate legal-masked top-N policy accuracy on WTHOR board data.")
    parser.add_argument("--board-data-dir", type=Path, default=default_board_data_dir())
    parser.add_argument("--model", type=Path, default=default_model_file(), help="Keras .h5 model. If missing, --weights is used.")
    parser.add_argument("--weights", type=Path, default=default_weights_file(), help="Binary weights fallback.")
    parser.add_argument("--top-n", type=parse_top_n, default=parse_top_n("1,2,3,4,5,8,10,16"))
    parser.add_argument("--batch-size", type=int, default=65536, help="Board-data read batch size.")
    parser.add_argument("--predict-batch-size", type=int, default=8192, help="Keras predict batch size.")
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "trained" / "playerop_wthor_eval")
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
        print(f"top-{row['top_n']:>2}: {row['accuracy'] * 100.0:.3f}% ({row['hits']}/{row['positions']})")
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
