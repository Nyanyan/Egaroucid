#!/usr/bin/env python3
"""
Train a compact policy network for Othello move prediction.

Related issue: #613

Input features are side-to-move bitboards:
  [player-to-move exists on 64 squares, opponent exists on 64 squares]

Current WTHOR direct training reads expanded position samples from:
  $EGAROUCID_DATA/train_data/board_data/records1

The legacy selected-data path is still available for comparison runs:
  train_data/board_data/Egaroucid_Train_Data_v2_selected/records0

Board-data binary sample layout, 19 bytes:
  uint64 player_to_move_bits
  uint64 opponent_bits
  int8   player_color
  int8   policy
  int8   score
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import shutil
import struct
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import __version__ as tf_version
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 as keras_l2


HUMAN_LIKE_AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUMAN_LIKE_AI_DIR))

from policy_accuracy import (  # noqa: E402
    equivalent_policy_mask,
    symmetry_aware_policy_ranks,
)


BOARD_SAMPLE_SIZE = 19
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

BIT_MASKS = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)).reshape(1, 64)
POLICY_BIT_MASKS = (np.uint64(1) << np.arange(0, 64, dtype=np.uint64)).reshape(1, 64)
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
NOT_FILE_A = np.uint64(0x7F7F7F7F7F7F7F7F)
NOT_FILE_H = np.uint64(0xFEFEFEFEFEFEFEFE)


@dataclass(frozen=True)
class BoardFile:
    path: Path
    n_position_samples: int


@dataclass(frozen=True)
class SplitArrays:
    x: np.ndarray
    policy: np.ndarray
    legal: np.ndarray
    player: np.ndarray
    opponent: np.ndarray
    n_discs: np.ndarray


@dataclass(frozen=True)
class ModelSpec:
    name: str
    width: int
    depth: int
    alpha: float
    dropout: float
    l2: float
    learning_rate: float


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def discover_board_files(data_root: Path, board_data_index_start: int, board_data_index_end: int) -> List[BoardFile]:
    files: List[BoardFile] = []
    for board_data_index in range(board_data_index_start, board_data_index_end + 1):
        board_data_dir = data_root / f"records{board_data_index}"
        if not board_data_dir.exists():
            raise FileNotFoundError(f"missing board data directory: {board_data_dir}")
        dat_files = sorted(board_data_dir.glob("*.dat"), key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
        if not dat_files:
            raise FileNotFoundError(f"no .dat files in {board_data_dir}")
        for path in dat_files:
            size = path.stat().st_size
            if size < BOARD_SAMPLE_SIZE:
                continue
            if size % BOARD_SAMPLE_SIZE != 0:
                print(f"warning: {path} size is not divisible by {BOARD_SAMPLE_SIZE}; trailing bytes are ignored")
            files.append(BoardFile(path=path, n_position_samples=size // BOARD_SAMPLE_SIZE))
    if not files:
        raise FileNotFoundError(f"no board data found under {data_root}")
    return files


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


def calc_legal_batch(player: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    empty = ~(player | opponent)
    legal = np.zeros_like(player, dtype=np.uint64)
    for shift in SHIFT_FUNCS:
        x = shift(player) & opponent
        for _ in range(5):
            x |= shift(x) & opponent
        legal |= shift(x) & empty
    return legal


def legal_bitboard_to_mask(legal: np.ndarray) -> np.ndarray:
    return (legal.reshape(-1, 1) & POLICY_BIT_MASKS) != 0


def valid_policy_mask(policies: np.ndarray, legal: np.ndarray) -> Tuple[np.ndarray, int, int]:
    valid_policy = (0 <= policies) & (policies < POLICY_SIZE)
    label_bits = np.zeros_like(legal, dtype=np.uint64)
    label_bits[valid_policy] = np.left_shift(np.uint64(1), policies[valid_policy].astype(np.uint64))
    legal_label = valid_policy & ((legal & label_bits) != 0)
    return legal_label, int(np.count_nonzero(~valid_policy)), int(np.count_nonzero(valid_policy & ~legal_label))


def allocate_counts(total: int, weights: Sequence[int]) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    raw = np.array(weights, dtype=np.float64) * (float(total) / float(sum(weights)))
    counts = np.floor(raw).astype(np.int64)
    missing = total - int(counts.sum())
    if missing > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:missing]] += 1
    return [int(v) for v in counts]


def position_samples_to_targets(position_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    player_bits = position_samples["player"].astype(np.uint64, copy=False)
    opponent_bits = position_samples["opponent"].astype(np.uint64, copy=False)
    n = len(position_samples)
    x = np.empty((n, INPUT_SIZE), dtype=np.float32)
    x[:, :64] = ((player_bits.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    x[:, 64:] = ((opponent_bits.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    y_policy = position_samples["policy"].astype(np.int64, copy=False)
    return x, y_policy


def position_samples_to_split_arrays(position_samples: np.ndarray) -> SplitArrays:
    player_bits = position_samples["player"].astype(np.uint64, copy=False)
    opponent_bits = position_samples["opponent"].astype(np.uint64, copy=False)
    x, y_policy = position_samples_to_targets(position_samples)
    legal = calc_legal_batch(player_bits, opponent_bits)
    n_discs = np.count_nonzero(x, axis=1).astype(np.int16, copy=False)
    return SplitArrays(
        x=x,
        policy=y_policy,
        legal=legal,
        player=player_bits,
        opponent=opponent_bits,
        n_discs=n_discs,
    )


def load_position_samples(files: Sequence[BoardFile], max_position_samples: Optional[int]) -> np.ndarray:
    chunks = []
    remaining = max_position_samples
    for board_file in files:
        n_position_samples = board_file.n_position_samples
        if remaining is not None:
            if remaining <= 0:
                break
            n_position_samples = min(n_position_samples, remaining)
            remaining -= n_position_samples
        if n_position_samples <= 0:
            continue
        mmap = np.memmap(board_file.path, dtype=BOARD_DTYPE, mode="r", shape=(board_file.n_position_samples,))
        chunks.append(np.asarray(mmap[:n_position_samples]).copy())
    if not chunks:
        raise ValueError("no position samples loaded")
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("total split size must be positive")
    if ratio_sum <= 0.0:
        raise ValueError("split ratios must have positive sum")
    n_train = int(np.floor(total * (train_ratio / ratio_sum)))
    n_val = int(np.floor(total * (val_ratio / ratio_sum)))
    n_test = total - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"split produced an empty part: train={n_train} val={n_val} test={n_test}")
    return n_train, n_val, n_test


def load_shuffled_three_way_split(
    files: Sequence[BoardFile],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_position_samples: Optional[int],
) -> Dict[str, SplitArrays]:
    rng = np.random.default_rng(seed)
    position_samples = load_position_samples(files, max_position_samples)
    policies = position_samples["policy"].astype(np.int64, copy=False)
    valid = (0 <= policies) & (policies < POLICY_SIZE)
    if not np.all(valid):
        position_samples = position_samples[valid]

    n_train, n_val, n_test = split_counts(len(position_samples), train_ratio, val_ratio, test_ratio)
    order = rng.permutation(len(position_samples))
    train_samples = position_samples[order[:n_train]]
    val_samples = position_samples[order[n_train : n_train + n_val]]
    test_samples = position_samples[order[n_train + n_val : n_train + n_val + n_test]]
    del position_samples, order

    return {
        "train": position_samples_to_split_arrays(train_samples),
        "val": position_samples_to_split_arrays(val_samples),
        "test": position_samples_to_split_arrays(test_samples),
    }


def append_sampled_positions(
    dest_x: np.ndarray,
    dest_policy: np.ndarray,
    offset: int,
    board_file: BoardFile,
    indices: np.ndarray,
) -> int:
    if len(indices) == 0:
        return offset
    mmap = np.memmap(board_file.path, dtype=BOARD_DTYPE, mode="r", shape=(board_file.n_position_samples,))
    position_samples = np.asarray(mmap[indices])
    x, y_policy = position_samples_to_targets(position_samples)
    valid = (0 <= y_policy) & (y_policy < POLICY_SIZE)
    if not np.all(valid):
        x = x[valid]
        y_policy = y_policy[valid]
    n = len(y_policy)
    dest_x[offset : offset + n] = x
    dest_policy[offset : offset + n] = y_policy
    return offset + n


def shuffle_split(
    x: np.ndarray,
    policy: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(policy))
    return x[order], policy[order]


def load_sampled_split(
    files: Sequence[BoardFile],
    max_train_samples: int,
    max_val_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = [f.n_position_samples for f in files]
    train_alloc = allocate_counts(max_train_samples, weights)
    val_alloc = allocate_counts(max_val_samples, weights)

    x_train = np.empty((sum(train_alloc), INPUT_SIZE), dtype=np.float32)
    p_train = np.empty(sum(train_alloc), dtype=np.int64)
    x_val = np.empty((sum(val_alloc), INPUT_SIZE), dtype=np.float32)
    p_val = np.empty(sum(val_alloc), dtype=np.int64)

    train_offset = 0
    val_offset = 0
    for i, board_file in enumerate(files):
        n_train = train_alloc[i]
        n_val = val_alloc[i]
        if n_train + n_val == 0:
            continue
        indices = rng.integers(0, board_file.n_position_samples, size=n_train + n_val, dtype=np.int64)
        train_offset = append_sampled_positions(x_train, p_train, train_offset, board_file, np.sort(indices[:n_train]))
        val_offset = append_sampled_positions(x_val, p_val, val_offset, board_file, np.sort(indices[n_train:]))

    x_train, p_train = x_train[:train_offset], p_train[:train_offset]
    x_val, p_val = x_val[:val_offset], p_val[:val_offset]
    x_train, p_train = shuffle_split(x_train, p_train, rng)
    x_val, p_val = shuffle_split(x_val, p_val, rng)
    return x_train, p_train, x_val, p_val


def parse_top_n(text: str) -> List[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def evaluate_policy_topn_arrays(
    model: Model,
    split_name: str,
    split: SplitArrays,
    n_values: Sequence[int],
    batch_size: int,
    predict_batch_size: int,
) -> dict:
    total_positions = 0
    invalid_policy = 0
    illegal_label = 0
    total_hits = {n: 0 for n in n_values}

    for start in range(0, len(split.policy), batch_size):
        end = min(len(split.policy), start + batch_size)
        probabilities = model.predict(split.x[start:end], batch_size=predict_batch_size, verbose=0)
        policies = split.policy[start:end]
        legal = split.legal[start:end]
        player = split.player[start:end]
        opponent = split.opponent[start:end]
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
        legal_mask = legal_bitboard_to_mask(legal)
        equivalent_mask = equivalent_policy_mask(
            player,
            opponent,
            policies,
        )
        rank = symmetry_aware_policy_ranks(
            probabilities,
            legal_mask,
            equivalent_mask,
        )
        for n in n_values:
            total_hits[n] += int(np.count_nonzero(rank <= n))
        total_positions += int(np.count_nonzero(valid))

    topn_rows = []
    for n in n_values:
        accuracy = total_hits[n] / total_positions if total_positions else 0.0
        topn_rows.append({"split": split_name, "top_n": n, "hits": total_hits[n], "positions": total_positions, "accuracy": accuracy})
    return {
        "split": split_name,
        "positions": total_positions,
        "invalid_policy_samples": invalid_policy,
        "illegal_label_samples": illegal_label,
        "agreement_definition": "board_symmetry_aware",
        "ranking_tie_break": "ascending_policy_index",
        "topn": topn_rows,
    }


def write_topn_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "top_n", "hits", "positions", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def parse_model_specs(text: str, args: argparse.Namespace) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) < 3:
                raise ValueError(f"invalid config '{token}', expected name:width:depth[:alpha[:dropout[:l2[:lr]]]]")
            name = parts[0]
            width = int(parts[1])
            depth = int(parts[2])
            alpha = float(parts[3]) if len(parts) > 3 else args.alpha
            dropout = float(parts[4]) if len(parts) > 4 else args.dropout
            l2_value = float(parts[5]) if len(parts) > 5 else args.l2
            lr = float(parts[6]) if len(parts) > 6 else args.learning_rate
        else:
            if "x" not in token:
                raise ValueError(f"invalid config '{token}', expected WIDTHxDEPTH")
            width_s, depth_s = token.lower().split("x", 1)
            width = int(width_s)
            depth = int(depth_s)
            alpha = args.alpha
            dropout = args.dropout
            l2_value = args.l2
            lr = args.learning_rate
            name = f"w{width}_d{depth}_a{alpha:g}"
        if width <= 0 or depth <= 0:
            raise ValueError(f"invalid config '{token}', width and depth must be positive")
        specs.append(ModelSpec(name, width, depth, alpha, dropout, l2_value, lr))
    if not specs:
        raise ValueError("no model configs were provided")
    return specs


def build_model(spec: ModelSpec) -> Model:
    kernel_regularizer = keras_l2(spec.l2) if spec.l2 > 0.0 else None
    inputs = Input(shape=(INPUT_SIZE,), name="board")
    x = inputs
    for layer_idx in range(spec.depth):
        x = Dense(
            spec.width,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"trunk_dense_{layer_idx}",
        )(x)
        x = LeakyReLU(alpha=spec.alpha, name=f"trunk_leaky_relu_{layer_idx}")(x)
        if spec.dropout > 0.0:
            x = Dropout(spec.dropout, name=f"trunk_dropout_{layer_idx}")(x)
    logits = Dense(POLICY_SIZE, name="policy_logits")(x)
    outputs = Softmax(name="policy")(logits)
    model = Model(inputs=inputs, outputs=outputs, name=spec.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=spec.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    return model


def _write_dense_layer(f, layer: Dense, activation: int, alpha: float) -> None:
    weights, bias = layer.get_weights()
    weights = np.asarray(weights, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    in_dim, out_dim = weights.shape
    f.write(struct.pack("<IIIf", in_dim, out_dim, activation, alpha))
    f.write(weights.astype("<f4", copy=False).tobytes(order="C"))
    f.write(bias.astype("<f4", copy=False).tobytes(order="C"))


def export_binary_weights(model: Model, spec: ModelSpec, out_file: Path) -> None:
    trunk_layers = [model.get_layer(f"trunk_dense_{i}") for i in range(spec.depth)]
    policy_layer = model.get_layer("policy_logits")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("wb") as f:
        f.write(b"EGR_POLICY_V1\0\0\0")
        f.write(struct.pack("<IIII", 1, len(trunk_layers) + 1, INPUT_SIZE, POLICY_SIZE))
        for layer in trunk_layers:
            _write_dense_layer(f, layer, 1, spec.alpha)
        _write_dense_layer(f, policy_layer, 0, 0.0)


def save_history_csv(history: tf.keras.callbacks.History, out_file: Path) -> None:
    keys = list(history.history.keys())
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for epoch_idx in range(len(history.history[keys[0]])):
            writer.writerow([epoch_idx + 1] + [history.history[key][epoch_idx] for key in keys])


def train_one_model(
    spec: ModelSpec,
    x_train: np.ndarray,
    p_train: np.ndarray,
    x_val: np.ndarray,
    p_val: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    split_eval_data: Optional[Dict[str, SplitArrays]] = None,
) -> dict:
    tf.keras.backend.clear_session()
    model = build_model(spec)
    run_dir = output_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== training {spec.name} ===")
    print(f"tensorflow {tf_version}")
    print(f"params {model.count_params()}")
    model.summary()

    callbacks = [CSVLogger(str(run_dir / "keras_log.csv"))]
    history = model.fit(
        x_train,
        p_train,
        validation_data=(x_val, p_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    save_history_csv(history, run_dir / "history.csv")
    adopted_epoch = len(history.epoch)
    adopted_idx = adopted_epoch - 1
    model.save(run_dir / "model.h5")
    export_binary_weights(model, spec, run_dir / "policy_network_weights.bin")

    result = {
        "name": spec.name,
        "spec": asdict(spec),
        "params": int(model.count_params()),
        "epochs_ran": len(history.epoch),
        "adopted_epoch": adopted_epoch,
        "adoption_rule": "use_model_after_requested_epoch_count",
        "validation_loss": float(history.history["val_loss"][adopted_idx]),
        "final_loss": float(history.history["loss"][-1]),
        "output_dir": str(run_dir),
    }
    if split_eval_data is not None and not args.no_split_topn:
        n_values = sorted(set(args.top_n))
        split_results = {}
        all_rows = []
        for split_name in ("train", "val", "test"):
            split_result = evaluate_policy_topn_arrays(
                model,
                split_name,
                split_eval_data[split_name],
                n_values,
                args.eval_batch_size,
                args.predict_batch_size,
            )
            split_results[split_name] = split_result
            all_rows.extend(split_result["topn"])
            top1 = next(row for row in split_result["topn"] if row["top_n"] == min(n_values))
            print(
                f"{split_name} legal-masked top-{top1['top_n']} "
                f"{top1['accuracy'] * 100.0:.3f}% ({top1['hits']}/{top1['positions']})"
            )
        result["split_topn"] = split_results
        val_topn = {int(row["top_n"]): float(row["accuracy"]) for row in split_results["val"]["topn"]}
        result["validation_top1"] = val_topn.get(1)
        result["validation_top3"] = val_topn.get(3)
        result["validation_top5"] = val_topn.get(5)
        result["validation_top10"] = val_topn.get(10)
        with (run_dir / "split_topn_accuracy.json").open("w", encoding="utf-8") as f:
            json.dump(split_results, f, indent=2)
        write_topn_rows(run_dir / "split_topn_accuracy.csv", all_rows)
    with (run_dir / "summary.json").open("w") as f:
        json.dump(result, f, indent=2)
    print("summary", json.dumps(result, indent=2))
    return result


def write_results(results: Sequence[dict], output_dir: Path) -> None:
    with (output_dir / "results.json").open("w") as f:
        json.dump(list(results), f, indent=2)
    fields = [
        "name",
        "params",
        "epochs_ran",
        "adopted_epoch",
        "adoption_rule",
        "validation_loss",
        "validation_top1",
        "validation_top3",
        "validation_top5",
        "validation_top10",
        "final_loss",
        "output_dir",
    ]
    with (output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def run_selection_key(result: dict) -> Tuple[float, float, int]:
    if result.get("validation_top1") is None or result.get("validation_top3") is None:
        raise ValueError(
            "model selection requires symmetry-aware validation top-1 and top-3; "
            "do not use --no-split-topn when comparing model configurations"
        )
    return (
        float(result["validation_top1"]),
        float(result["validation_top3"]),
        -int(result.get("params", 0)),
    )


def default_output_dir() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "trained" / stamp


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_data_root() -> Path:
    return repo_root() / "train_data" / "board_data" / "Egaroucid_Train_Data_v2_selected"


def default_wthor_data_root() -> Path:
    return Path(os.environ["EGAROUCID_DATA"]) / "train_data" / "board_data"


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train compact Othello policy networks with tensorflow.keras for the human-like AI study.")
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="default: train_data/board_data/Egaroucid_Train_Data_v2_selected")
    parser.add_argument("--board-data-index-start", type=int, default=0)
    parser.add_argument("--board-data-index-end", type=int, default=0)
    parser.add_argument("--wthor", action="store_true", help="Use $EGAROUCID_DATA/train_data/board_data/records1 as the source.")
    parser.add_argument("--split-mode", choices=["sampled", "shuffled"], default="sampled", help="sampled keeps the legacy random sampling path; shuffled makes disjoint train/val/test splits.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-position-samples", type=int, default=None, help="Optional cap before shuffled train/val/test split.")
    parser.add_argument("--configs", default="64x3,96x3,128x3,96x4")
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train-samples", type=int, default=1000000)
    parser.add_argument("--max-val-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--top-n", type=parse_top_n, default=parse_top_n("1,2,3,5,10"))
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--no-split-topn", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    set_reproducible_seed(args.seed)
    enable_memory_growth()

    if args.wthor:
        data_root = default_wthor_data_root()
        args.board_data_index_start = 1
        args.board_data_index_end = 1
        if args.split_mode == "sampled":
            args.split_mode = "shuffled"
    else:
        data_root = Path(args.data_root)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("tensorflow version", tf_version)
    print("data_root", data_root)
    print("output_dir", output_dir)
    print("issue", "#613")
    print("split_mode", args.split_mode)

    files = discover_board_files(data_root, args.board_data_index_start, args.board_data_index_end)
    print(f"board files {len(files)}")
    print(f"total_position_samples {sum(f.n_position_samples for f in files)}")

    split_eval_data: Optional[Dict[str, SplitArrays]] = None
    if args.split_mode == "sampled":
        print(f"sampling train={args.max_train_samples} val={args.max_val_samples}")
        x_train, p_train, x_val, p_val = load_sampled_split(files, args.max_train_samples, args.max_val_samples, args.seed)
        print("loaded", x_train.shape, p_train.shape, x_val.shape, p_val.shape)
    else:
        print(
            "shuffled split ratios "
            f"train={args.train_ratio} val={args.val_ratio} test={args.test_ratio} "
            f"max_position_samples={args.max_position_samples}"
        )
        split_eval_data = load_shuffled_three_way_split(
            files,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
            args.max_position_samples,
        )
        x_train = split_eval_data["train"].x
        p_train = split_eval_data["train"].policy
        x_val = split_eval_data["val"].x
        p_val = split_eval_data["val"].policy
        print(
            "loaded",
            x_train.shape,
            p_train.shape,
            x_val.shape,
            p_val.shape,
            split_eval_data["test"].x.shape,
            split_eval_data["test"].policy.shape,
        )

    results = []
    for spec in parse_model_specs(args.configs, args):
        results.append(train_one_model(spec, x_train, p_train, x_val, p_val, args, output_dir, split_eval_data))
        write_results(results, output_dir)

    selected = results[0] if len(results) == 1 else max(results, key=run_selection_key)
    selected_dir = Path(selected["output_dir"])
    shutil.copyfile(selected_dir / "policy_network_weights.bin", output_dir / "selected_policy_network_weights.bin")
    shutil.copyfile(selected_dir / "model.h5", output_dir / "selected_model.h5")
    with (output_dir / "selected_summary.json").open("w") as f:
        json.dump(selected, f, indent=2)
    print("\n=== selected final-epoch model ===")
    print(json.dumps(selected, indent=2))
    print("selected weights", output_dir / "selected_policy_network_weights.bin")


if __name__ == "__main__":
    main()
