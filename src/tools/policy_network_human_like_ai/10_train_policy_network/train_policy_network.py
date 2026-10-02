#!/usr/bin/env python3
"""
Train a compact policy network for Othello move prediction.

Related issue: #613

Input features are side-to-move bitboards:
  [player-to-move exists on 64 squares, opponent exists on 64 squares]

Use --board-split-dir for numbered pipeline stage 03/04 output. Its game
partitions are loaded unchanged, including after eightfold augmentation.
The legacy WTHOR direct-training mode reads expanded position samples from:
  $EGAROUCID_DATA/train_data/board_data/records1

Data preprocessing is implemented in ../00_data_processing.
Use --split-manifest to load a previously prepared WTHOR game split.

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

sys.path.insert(0, str(HUMAN_LIKE_AI_DIR / "00_data_processing"))

from policy_data import (  # noqa: E402
    BOARD_DTYPE, BOARD_SAMPLE_SIZE, INPUT_SIZE, POLICY_SIZE,
    BIT_MASKS, POLICY_BIT_MASKS, FULL, NOT_FILE_A, NOT_FILE_H, SHIFT_FUNCS,
    BoardFile, SplitArrays, discover_board_files,
    shift_east, shift_west, shift_north, shift_south, shift_ne, shift_nw, shift_se, shift_sw,
    calc_legal_batch, legal_bitboard_to_mask, valid_policy_mask,
    position_samples_to_targets, position_samples_to_split_arrays,
    load_game_three_way_split, load_shuffled_three_way_split, load_prepared_board_splits, split_counts,
)

from policy_accuracy import (  # noqa: E402
    equivalent_policy_mask,
    symmetry_aware_policy_ranks,
)


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
        "dataset_split_manifest": str(output_dir / "dataset_split.json") if args.board_split_dir is None else None,
        "prepared_dataset_manifest": str(output_dir / "prepared_dataset.json") if args.board_split_dir is not None else None,
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
    return default_wthor_data_root()


def default_wthor_data_root() -> Path:
    data_root = os.environ.get("EGAROUCID_DATA")
    if not data_root:
        raise ValueError("set EGAROUCID_DATA or pass --data-root containing WTHOR records1")
    return Path(data_root) / "train_data" / "board_data"


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train compact Othello policy networks with tensorflow.keras for the human-like AI study.")
    parser.add_argument("--data-root", type=Path, default=None, help="Default: $EGAROUCID_DATA/train_data/board_data (WTHOR records1).")
    parser.add_argument("--board-split-dir", type=Path, default=None, help="Dataset directory produced by stage 03/04; load its train/val/test unchanged.")
    parser.add_argument("--board-data-index-start", type=int, default=1)
    parser.add_argument("--board-data-index-end", type=int, default=1)
    parser.add_argument("--wthor", action="store_true", help="Compatibility option: select records1, which is already the default.")
    parser.add_argument("--split-manifest", type=Path, default=None, help="Reuse a dataset_split.json created by 00_data_processing/prepare_wthor_data.py; its seed and ratios determine the split.")
    parser.add_argument("--split-mode", choices=["game", "sampled", "shuffled"], default="game", help="All modes split by game. shuffled is an alias for game; sampled caps positions within the disjoint game partitions.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--max-position-samples", type=int, default=None, help="Optional cap on source positions; keep only whole games before the game split.")
    parser.add_argument("--configs", default="64x3,96x3,128x3,96x4")
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train-samples", type=int, default=1000000, help="Position cap within the train game partition in sampled mode only.")
    parser.add_argument("--max-val-samples", type=int, default=100000, help="Position cap within the validation game partition in sampled mode only.")
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

    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.board_split_dir is not None:
        if (args.data_root is not None or args.wthor or args.split_manifest is not None
                or args.max_position_samples is not None or args.split_mode == "sampled"):
            raise ValueError("--board-split-dir already contains partitions; do not combine it with legacy source or resplitting options")
        print("prepared_board_splits", args.board_split_dir)
        split_eval_data = load_prepared_board_splits(args.board_split_dir, args.seed, output_dir)
    else:
        data_root = args.data_root if args.data_root is not None else default_data_root()
        if args.wthor:
            args.board_data_index_start = 1
            args.board_data_index_end = 1

        print("tensorflow version", tf_version)
        print("data_root", data_root)
        print("output_dir", output_dir)
        print("issue", "#613")
        print("split_mode", args.split_mode)

        files = discover_board_files(data_root, args.board_data_index_start, args.board_data_index_end)
        print(f"board files {len(files)}")
        print(f"total_position_samples {sum(f.n_position_samples for f in files)}")

        print(
            "game split ratios "
            f"train={args.train_ratio} val={args.val_ratio} test={args.test_ratio} "
            f"max_position_samples={args.max_position_samples}"
        )
        sample_caps = None
        if args.split_mode == "sampled":
            sample_caps = {"train": args.max_train_samples, "val": args.max_val_samples}
        split_eval_data = load_game_three_way_split(
            files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
            args.max_position_samples, sample_caps, output_dir,
            args.split_manifest,
        )
    x_train = split_eval_data["train"].x
    p_train = split_eval_data["train"].policy
    x_val = split_eval_data["val"].x
    p_val = split_eval_data["val"].policy
    print("loaded", {name: split.x.shape for name, split in split_eval_data.items()})

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
