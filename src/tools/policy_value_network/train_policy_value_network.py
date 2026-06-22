#!/usr/bin/env python3
"""
Train a compact policy-value network for Othello.

Related issue: #613

Input features are side-to-move bitboards:
  [player-to-move exists on 64 squares, opponent exists on 64 squares]

Outputs:
  policy: 64-way softmax over moves
  value:  tanh scalar, final disc difference from player-to-move perspective / 64
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
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import __version__ as tf_version
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 as keras_l2


BOARD_RECORD_SIZE = 19
INPUT_SIZE = 128
POLICY_SIZE = 64
VALUE_SCALE = 64.0

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

BIT_MASKS = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)).reshape(1, 64)


@dataclass(frozen=True)
class BoardFile:
    path: Path
    n_records: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    width: int
    depth: int
    alpha: float
    dropout: float
    l2: float
    learning_rate: float
    value_loss_weight: float


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


def discover_board_files(data_root: Path, record_start: int, record_end: int) -> List[BoardFile]:
    files: List[BoardFile] = []
    for record in range(record_start, record_end + 1):
        record_dir = data_root / f"records{record}"
        if not record_dir.exists():
            raise FileNotFoundError(f"missing board data directory: {record_dir}")
        dat_files = sorted(record_dir.glob("*.dat"), key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name))
        if not dat_files:
            raise FileNotFoundError(f"no .dat files in {record_dir}")
        for path in dat_files:
            size = path.stat().st_size
            if size < BOARD_RECORD_SIZE:
                continue
            if size % BOARD_RECORD_SIZE != 0:
                print(f"warning: {path} size is not divisible by {BOARD_RECORD_SIZE}; trailing bytes are ignored")
            files.append(BoardFile(path=path, n_records=size // BOARD_RECORD_SIZE))
    if not files:
        raise FileNotFoundError(f"no board data found under {data_root}")
    return files


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


def records_to_targets(records: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    player_bits = records["player"].astype(np.uint64, copy=False)
    opponent_bits = records["opponent"].astype(np.uint64, copy=False)
    n = len(records)
    x = np.empty((n, INPUT_SIZE), dtype=np.float32)
    x[:, :64] = ((player_bits.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    x[:, 64:] = ((opponent_bits.reshape(-1, 1) & BIT_MASKS) != 0).astype(np.float32)
    y_policy = records["policy"].astype(np.int64, copy=False)
    y_value = (records["score"].astype(np.float32, copy=False) / VALUE_SCALE).reshape(-1, 1)
    return x, y_policy, y_value


def append_sampled_records(
    dest_x: np.ndarray,
    dest_policy: np.ndarray,
    dest_value: np.ndarray,
    offset: int,
    board_file: BoardFile,
    indices: np.ndarray,
) -> int:
    if len(indices) == 0:
        return offset
    mmap = np.memmap(board_file.path, dtype=BOARD_DTYPE, mode="r", shape=(board_file.n_records,))
    records = np.asarray(mmap[indices])
    x, y_policy, y_value = records_to_targets(records)
    valid = (0 <= y_policy) & (y_policy < POLICY_SIZE)
    if not np.all(valid):
        x = x[valid]
        y_policy = y_policy[valid]
        y_value = y_value[valid]
    n = len(y_policy)
    dest_x[offset : offset + n] = x
    dest_policy[offset : offset + n] = y_policy
    dest_value[offset : offset + n] = y_value
    return offset + n


def shuffle_split(
    x: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = rng.permutation(len(policy))
    return x[order], policy[order], value[order]


def load_sampled_split(
    files: Sequence[BoardFile],
    max_train_samples: int,
    max_val_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = [f.n_records for f in files]
    train_alloc = allocate_counts(max_train_samples, weights)
    val_alloc = allocate_counts(max_val_samples, weights)

    x_train = np.empty((sum(train_alloc), INPUT_SIZE), dtype=np.float32)
    p_train = np.empty(sum(train_alloc), dtype=np.int64)
    v_train = np.empty((sum(train_alloc), 1), dtype=np.float32)
    x_val = np.empty((sum(val_alloc), INPUT_SIZE), dtype=np.float32)
    p_val = np.empty(sum(val_alloc), dtype=np.int64)
    v_val = np.empty((sum(val_alloc), 1), dtype=np.float32)

    train_offset = 0
    val_offset = 0
    for i, board_file in enumerate(files):
        n_train = train_alloc[i]
        n_val = val_alloc[i]
        if n_train + n_val == 0:
            continue
        indices = rng.integers(0, board_file.n_records, size=n_train + n_val, dtype=np.int64)
        train_offset = append_sampled_records(x_train, p_train, v_train, train_offset, board_file, np.sort(indices[:n_train]))
        val_offset = append_sampled_records(x_val, p_val, v_val, val_offset, board_file, np.sort(indices[n_train:]))

    x_train, p_train, v_train = x_train[:train_offset], p_train[:train_offset], v_train[:train_offset]
    x_val, p_val, v_val = x_val[:val_offset], p_val[:val_offset], v_val[:val_offset]
    x_train, p_train, v_train = shuffle_split(x_train, p_train, v_train, rng)
    x_val, p_val, v_val = shuffle_split(x_val, p_val, v_val, rng)
    return x_train, p_train, v_train, x_val, p_val, v_val


def parse_model_specs(text: str, args: argparse.Namespace) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) < 3:
                raise ValueError(f"invalid config '{token}', expected name:width:depth[:value_weight[:alpha[:dropout[:l2[:lr]]]]]")
            name = parts[0]
            width = int(parts[1])
            depth = int(parts[2])
            value_weight = float(parts[3]) if len(parts) > 3 else args.value_loss_weight
            alpha = float(parts[4]) if len(parts) > 4 else args.alpha
            dropout = float(parts[5]) if len(parts) > 5 else args.dropout
            l2_value = float(parts[6]) if len(parts) > 6 else args.l2
            lr = float(parts[7]) if len(parts) > 7 else args.learning_rate
        else:
            if "x" not in token:
                raise ValueError(f"invalid config '{token}', expected WIDTHxDEPTH")
            width_s, depth_s = token.lower().split("x", 1)
            width = int(width_s)
            depth = int(depth_s)
            value_weight = args.value_loss_weight
            alpha = args.alpha
            dropout = args.dropout
            l2_value = args.l2
            lr = args.learning_rate
            name = f"w{width}_d{depth}_vw{value_weight:g}_a{alpha:g}"
        if width <= 0 or depth <= 0:
            raise ValueError(f"invalid config '{token}', width and depth must be positive")
        specs.append(ModelSpec(name, width, depth, alpha, dropout, l2_value, lr, value_weight))
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
    policy_logits = Dense(POLICY_SIZE, name="policy_logits")(x)
    policy = Softmax(name="policy")(policy_logits)
    value = Dense(1, activation="tanh", name="value")(x)
    model = Model(inputs=inputs, outputs={"policy": policy, "value": value}, name=spec.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=spec.learning_rate),
        loss={
            "policy": tf.keras.losses.SparseCategoricalCrossentropy(),
            "value": tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={"policy": 1.0, "value": spec.value_loss_weight},
        metrics={
            "policy": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
            ],
            "value": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
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
    value_layer = model.get_layer("value")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("wb") as f:
        f.write(b"EGR_POLVAL_V1\0\0\0")
        f.write(struct.pack("<IIIII", 1, len(trunk_layers), INPUT_SIZE, POLICY_SIZE, 1))
        for layer in trunk_layers:
            _write_dense_layer(f, layer, 1, spec.alpha)
        _write_dense_layer(f, policy_layer, 0, 0.0)
        _write_dense_layer(f, value_layer, 2, 0.0)


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
    v_train: np.ndarray,
    x_val: np.ndarray,
    p_val: np.ndarray,
    v_val: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    tf.keras.backend.clear_session()
    model = build_model(spec)
    run_dir = output_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== training {spec.name} ===")
    print(f"tensorflow {tf_version}")
    print(f"params {model.count_params()}")
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, args.patience // 2), min_lr=args.min_learning_rate),
        CSVLogger(str(run_dir / "keras_log.csv")),
        ModelCheckpoint(str(run_dir / "best_model.h5"), monitor="val_loss", mode="min", save_best_only=True),
    ]
    history = model.fit(
        x_train,
        {"policy": p_train, "value": v_train},
        validation_data=(x_val, {"policy": p_val, "value": v_val}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    save_history_csv(history, run_dir / "history.csv")
    model.save(run_dir / "model.h5")
    export_binary_weights(model, spec, run_dir / "policy_value_network_weights.bin")

    val_loss = history.history["val_loss"]
    best_epoch = int(np.argmin(val_loss)) + 1
    best_idx = best_epoch - 1
    result = {
        "name": spec.name,
        "spec": asdict(spec),
        "params": int(model.count_params()),
        "epochs_ran": len(history.epoch),
        "best_epoch": best_epoch,
        "best_val_loss": float(history.history["val_loss"][best_idx]),
        "best_val_policy_accuracy": float(history.history["val_policy_accuracy"][best_idx]),
        "best_val_policy_top3": float(history.history["val_policy_top3"][best_idx]),
        "best_val_policy_top5": float(history.history["val_policy_top5"][best_idx]),
        "best_val_value_mae": float(history.history["val_value_mae"][best_idx]),
        "final_loss": float(history.history["loss"][-1]),
        "final_policy_accuracy": float(history.history["policy_accuracy"][-1]),
        "final_policy_top3": float(history.history["policy_top3"][-1]),
        "final_policy_top5": float(history.history["policy_top5"][-1]),
        "final_value_mae": float(history.history["value_mae"][-1]),
        "output_dir": str(run_dir),
    }
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
        "best_epoch",
        "best_val_loss",
        "best_val_policy_accuracy",
        "best_val_policy_top3",
        "best_val_policy_top5",
        "best_val_value_mae",
        "final_loss",
        "final_policy_accuracy",
        "final_policy_top3",
        "final_policy_top5",
        "final_value_mae",
        "output_dir",
    ]
    with (output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def default_output_dir() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "trained" / stamp


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train compact Othello policy-value networks with tensorflow.keras.")
    parser.add_argument("--data-root", default=None, help="default: $EGAROUCID_DATA/train_data/board_data")
    parser.add_argument("--record-start", type=int, default=259)
    parser.add_argument("--record-end", type=int, default=310)
    parser.add_argument("--configs", default="64x3,96x3,128x3,96x4")
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-learning-rate", type=float, default=0.00005)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train-samples", type=int, default=1000000)
    parser.add_argument("--max-val-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    set_reproducible_seed(args.seed)
    enable_memory_growth()

    if args.data_root is None:
        data_root = Path(os.environ["EGAROUCID_DATA"]) / "train_data" / "board_data"
    else:
        data_root = Path(args.data_root)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("tensorflow version", tf_version)
    print("data_root", data_root)
    print("output_dir", output_dir)
    print("issue", "#613")
    print("value target", "score / 64 from player-to-move perspective")

    files = discover_board_files(data_root, args.record_start, args.record_end)
    print(f"board files {len(files)}")
    print(f"total records {sum(f.n_records for f in files)}")
    print(f"sampling train={args.max_train_samples} val={args.max_val_samples}")

    x_train, p_train, v_train, x_val, p_val, v_val = load_sampled_split(files, args.max_train_samples, args.max_val_samples, args.seed)
    print("loaded", x_train.shape, p_train.shape, v_train.shape, x_val.shape, p_val.shape, v_val.shape)

    results = []
    for spec in parse_model_specs(args.configs, args):
        results.append(train_one_model(spec, x_train, p_train, v_train, x_val, p_val, v_val, args, output_dir))
        write_results(results, output_dir)

    best = min(results, key=lambda r: r["best_val_loss"])
    best_dir = Path(best["output_dir"])
    shutil.copyfile(best_dir / "policy_value_network_weights.bin", output_dir / "best_policy_value_network_weights.bin")
    shutil.copyfile(best_dir / "model.h5", output_dir / "best_model.h5")
    with (output_dir / "best_summary.json").open("w") as f:
        json.dump(best, f, indent=2)
    print("\n=== best ===")
    print(json.dumps(best, indent=2))
    print("best weights", output_dir / "best_policy_value_network_weights.bin")


if __name__ == "__main__":
    main()
