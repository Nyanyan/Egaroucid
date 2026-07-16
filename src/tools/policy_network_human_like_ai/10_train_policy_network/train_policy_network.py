#!/usr/bin/env python3
"""
Train a compact policy network for Othello move prediction.

Related issue: #613

Input features are side-to-move bitboards:
  [player-to-move exists on 64 squares, opponent exists on 64 squares]

Training data is read from selected transcript_release/0002 games converted to:
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


@dataclass(frozen=True)
class BoardFile:
    path: Path
    n_position_samples: int


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
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
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
        EarlyStopping(monitor="val_accuracy", mode="max", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, args.patience // 2), min_lr=args.min_learning_rate),
        CSVLogger(str(run_dir / "keras_log.csv")),
        ModelCheckpoint(str(run_dir / "best_model.h5"), monitor="val_accuracy", mode="max", save_best_only=True),
    ]
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
    model.save(run_dir / "model.h5")
    export_binary_weights(model, spec, run_dir / "policy_network_weights.bin")

    val_accuracy = history.history.get("val_accuracy", [])
    best_epoch = int(np.argmax(val_accuracy)) + 1 if val_accuracy else len(history.epoch)
    best_idx = best_epoch - 1
    result = {
        "name": spec.name,
        "spec": asdict(spec),
        "params": int(model.count_params()),
        "epochs_ran": len(history.epoch),
        "best_epoch": best_epoch,
        "best_val_loss": float(history.history["val_loss"][best_idx]),
        "best_val_accuracy": float(history.history["val_accuracy"][best_idx]),
        "best_val_top3": float(history.history["val_top3"][best_idx]),
        "best_val_top5": float(history.history["val_top5"][best_idx]),
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_top3": float(history.history["top3"][-1]),
        "final_top5": float(history.history["top5"][-1]),
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
        "best_val_accuracy",
        "best_val_top3",
        "best_val_top5",
        "final_loss",
        "final_accuracy",
        "final_top3",
        "final_top5",
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_data_root() -> Path:
    return repo_root() / "train_data" / "board_data" / "Egaroucid_Train_Data_v2_selected"


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train compact Othello policy networks with tensorflow.keras for the human-like AI study.")
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="default: train_data/board_data/Egaroucid_Train_Data_v2_selected")
    parser.add_argument("--board-data-index-start", type=int, default=0)
    parser.add_argument("--board-data-index-end", type=int, default=0)
    parser.add_argument("--configs", default="64x3,96x3,128x3,96x4")
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
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

    data_root = Path(args.data_root)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("tensorflow version", tf_version)
    print("data_root", data_root)
    print("output_dir", output_dir)
    print("issue", "#613")

    files = discover_board_files(data_root, args.board_data_index_start, args.board_data_index_end)
    print(f"board files {len(files)}")
    print(f"total_position_samples {sum(f.n_position_samples for f in files)}")
    print(f"sampling train={args.max_train_samples} val={args.max_val_samples}")

    x_train, p_train, x_val, p_val = load_sampled_split(files, args.max_train_samples, args.max_val_samples, args.seed)
    print("loaded", x_train.shape, p_train.shape, x_val.shape, p_val.shape)

    results = []
    for spec in parse_model_specs(args.configs, args):
        results.append(train_one_model(spec, x_train, p_train, x_val, p_val, args, output_dir))
        write_results(results, output_dir)

    best = max(results, key=lambda r: r["best_val_accuracy"])
    best_dir = Path(best["output_dir"])
    shutil.copyfile(best_dir / "policy_network_weights.bin", output_dir / "best_policy_network_weights.bin")
    shutil.copyfile(best_dir / "model.h5", output_dir / "best_model.h5")
    with (output_dir / "best_summary.json").open("w") as f:
        json.dump(best, f, indent=2)
    print("\n=== best ===")
    print(json.dumps(best, indent=2))
    print("best weights", output_dir / "best_policy_network_weights.bin")


if __name__ == "__main__":
    main()
