#!/usr/bin/env python3
"""Train on prepared train/val partitions, then evaluate the selected model on test.

Resolve data from POLICY_DATA_ROOT/board_split_increased/POLICY_DATASET.
Preprocessing and partition membership belong to ../00_data_processing.
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
import sys
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import __version__ as tf_version
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 as keras_l2

from paper_policy_models import PAPER_FILTERS, build_paper_model
from prepared_policy_data import add_data_arguments, resolve_board_split_dir, load_prepared_splits
from training_runtime import TrainingRuntime, check_training_identity

HUMAN_LIKE_AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUMAN_LIKE_AI_DIR))

sys.path.insert(0, str(HUMAN_LIKE_AI_DIR / "00_data_processing"))

from policy_data import (  # noqa: E402
    INPUT_SIZE, POLICY_SIZE, SplitArrays,
    legal_bitboard_to_mask, valid_policy_mask, position_samples_to_split_arrays,
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
    architecture: str = "mlp"


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
    cross_entropy_sum = 0.0

    if hasattr(split, "batches"):
        chunks = (position_samples_to_split_arrays(samples) for samples in split.batches(batch_size))
    else:
        chunks = (SplitArrays(**{key: getattr(split, key)[start:start + batch_size]
                                for key in ("x", "policy", "legal", "player", "opponent", "n_discs")})
                  for start in range(0, len(split.policy), batch_size))
    for chunk in chunks:
        probabilities = model.predict(chunk.x, batch_size=predict_batch_size, verbose=0)
        policies, legal = chunk.policy, chunk.legal
        player, opponent = chunk.player, chunk.opponent
        valid, n_invalid_policy, n_illegal_label = valid_policy_mask(policies, legal)
        invalid_policy += n_invalid_policy
        illegal_label += n_illegal_label
        if not np.any(valid):
            continue

        probabilities = probabilities[valid]
        policies = policies[valid]
        label_probabilities = probabilities[np.arange(len(policies)), policies]
        cross_entropy_sum += float(-np.log(np.clip(label_probabilities, tf.keras.backend.epsilon(), 1.0)).sum(dtype=np.float64))
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
        "cross_entropy": cross_entropy_sum / total_positions if total_positions else None,
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
        preset = token.lower().replace("+", "_").replace("-", "_")
        if preset in PAPER_FILTERS:
            specs.append(ModelSpec(
                name=preset, width=128, depth=len(PAPER_FILTERS[preset]),
                alpha=0.0, dropout=0.0, l2=args.l2,
                learning_rate=args.learning_rate, architecture=preset,
            ))
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
                raise ValueError(f"invalid config '{token}', expected WIDTHxDEPTH or one of {', '.join(PAPER_FILTERS)}")
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
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("model configs must have distinct output names")
    return specs


def build_model(spec: ModelSpec, steps_per_execution: int = 1) -> Model:
    if spec.architecture != "mlp":
        model = build_paper_model(spec.architecture, spec.name, spec.l2)
        compile_model(model, spec.learning_rate, steps_per_execution)
        return model
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
    logits = Dense(POLICY_SIZE, name="policy_logits", dtype="float32")(x)
    outputs = Softmax(name="policy", dtype="float32")(logits)
    model = Model(inputs=inputs, outputs=outputs, name=spec.name)
    compile_model(model, spec.learning_rate, steps_per_execution)
    return model


def compile_model(model: Model, learning_rate: float, steps_per_execution: int = 1) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        steps_per_execution=steps_per_execution,
    )


def _write_dense_layer(f, layer: Dense, activation: int, alpha: float) -> None:
    weights, bias = layer.get_weights()
    weights = np.asarray(weights, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    in_dim, out_dim = weights.shape
    f.write(struct.pack("<IIIf", in_dim, out_dim, activation, alpha))
    f.write(weights.astype("<f4", copy=False).tobytes(order="C"))
    f.write(bias.astype("<f4", copy=False).tobytes(order="C"))


def export_binary_weights(model: Model, spec: ModelSpec, out_file: Path) -> None:
    if spec.architecture != "mlp":
        raise ValueError("EGR_POLICY_V1 supports only MLP models; use the CNN model.h5 with TensorFlow")
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


def train_one_model(spec: ModelSpec, splits: dict, args: argparse.Namespace, output_dir: Path) -> dict:
    tf.keras.backend.clear_session()
    set_reproducible_seed(args.seed)
    model = build_model(spec, args.steps_per_execution)
    run_dir = output_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== training {spec.name} ===", flush=True)
    print(f"tensorflow {tf_version}; params {model.count_params()}")
    model.summary()
    runtime = TrainingRuntime(model, run_dir, args.epochs,
                              (splits["train"].positions + args.batch_size - 1) // args.batch_size,
                              args.progress_interval_sec, args.checkpoint)
    if runtime.initial_epoch < args.epochs:
        model.fit(
            splits["train"].as_dataset(args.batch_size, shuffle=True, seed=args.seed, initial_epoch=runtime.initial_epoch),
            validation_data=splits["val"].as_dataset(args.batch_size),
            epochs=args.epochs,
            initial_epoch=runtime.initial_epoch,
            callbacks=[runtime, CSVLogger(str(run_dir / "keras_log.csv"), append=bool(runtime.initial_epoch))],
            verbose=2,
        )
    runtime.write_history()
    model.save(run_dir / "model.h5")
    weights_file = run_dir / "policy_network_weights.bin"
    if spec.architecture == "mlp":
        export_binary_weights(model, spec, weights_file)
    else:
        weights_file.unlink(missing_ok=True)
    result = {
        "name": spec.name, "spec": asdict(spec), "params": int(model.count_params()),
        "epochs_ran": len(runtime.records), "adopted_epoch": len(runtime.records),
        "adoption_rule": "use_model_after_requested_epoch_count",
        "validation_loss": runtime.records[-1]["val_loss"],
        "final_loss": runtime.records[-1]["loss"],
        "precision": args.precision,
        "steps_per_execution": args.steps_per_execution,
        "output_dir": str(run_dir),
        "prepared_dataset_manifest": str(output_dir / "prepared_dataset.json"),
        "binary_weights": str(weights_file) if spec.architecture == "mlp" else None,
    }
    validation = evaluate_policy_topn_arrays(model, "val", splits["val"], args.top_n,
                                            args.eval_batch_size, args.predict_batch_size)
    result["split_topn"] = {"val": validation}
    val_topn = {row["top_n"]: row["accuracy"] for row in validation["topn"]}
    for n in (1, 3, 5, 10):
        result[f"validation_top{n}"] = val_topn.get(n)
    (run_dir / "split_topn_accuracy.json").write_text(json.dumps({"val": validation}, indent=2), encoding="utf-8")
    write_topn_rows(run_dir / "split_topn_accuracy.csv", validation["topn"])
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"val top-1 {val_topn[1] * 100:.3f}%", flush=True)
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
            "include top-1 and top-3 when comparing model configurations"
        )
    return (
        float(result["validation_top1"]),
        float(result["validation_top3"]),
        -int(result.get("params", 0)),
    )


def save_selected_model(selected: dict, output_dir: Path) -> None:
    selected_dir = Path(selected["output_dir"])
    shutil.copyfile(selected_dir / "model.h5", output_dir / "selected_model.h5")
    weights_file = output_dir / "selected_policy_network_weights.bin"
    if selected["binary_weights"] is not None:
        shutil.copyfile(Path(selected["binary_weights"]), weights_file)
    else:
        weights_file.unlink(missing_ok=True)
    with (output_dir / "selected_summary.json").open("w") as f:
        json.dump(selected, f, indent=2)


def default_output_dir() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "trained" / stamp


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train compact Othello policy networks with tensorflow.keras for the human-like AI study.")
    add_data_arguments(parser)
    parser.add_argument("--configs", default="64x3,96x3,128x3,96x4", help="Comma-separated MLP specs or paper CNN presets: conv4, conv6, conv8, conv8_bn (Conv8+BN).")
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None, help="Default: 256 when configs include a CNN, otherwise 4096.")
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--top-n", type=parse_top_n, default=parse_top_n("1,2,3,5,10"))
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--predict-batch-size", type=int, default=None, help="Default: 256 when configs include a CNN, otherwise 8192.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--precision", choices=("float32", "mixed_float16"), default="float32")
    parser.add_argument("--steps-per-execution", type=int, default=1)
    parser.add_argument("--progress-interval-sec", type=float, default=30)
    parser.add_argument("--checkpoint", action="store_true", help="Save model and optimizer at each epoch boundary")
    parser.add_argument("--resume-training", action="store_true", help="Resume the same settings/data/code from an epoch checkpoint")
    return parser


def main() -> None:
    parser = make_argparser()
    args = parser.parse_args()
    if args.steps_per_execution < 1 or args.progress_interval_sec <= 0:
        parser.error("steps-per-execution and progress-interval-sec must be positive")
    if args.resume_training and not args.checkpoint:
        parser.error("--resume-training requires --checkpoint")
    tf.keras.mixed_precision.set_global_policy(args.precision)
    specs = parse_model_specs(args.configs, args)
    has_cnn = any(spec.architecture != "mlp" for spec in specs)
    if args.batch_size is None:
        args.batch_size = 256 if has_cnn else 4096
    if args.predict_batch_size is None:
        args.predict_batch_size = 256 if has_cnn else 8192
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.predict_batch_size) <= 0:
        parser.error("epochs and batch sizes must be positive")
    if not args.top_n or any(n < 1 or n > POLICY_SIZE for n in args.top_n):
        parser.error("--top-n must contain values from 1 to 64")
    args.top_n = sorted(set(args.top_n) | {1, 3})
    caps = {name: getattr(args, f"max_{name}_samples") for name in ("train", "val", "test")}
    if any(cap is not None and cap <= 0 for cap in caps.values()):
        parser.error("sample caps must be positive")
    try:
        directory = resolve_board_split_dir(args)
    except ValueError as exc:
        parser.error(str(exc))
    enable_memory_growth()
    set_reproducible_seed(args.seed)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    # Refuse to mix an earlier selected model or test report with a new run.
    check_training_identity(output_dir, directory, args)
    splits = load_prepared_splits(directory, caps, args.seed, output_dir)
    try:
        results = []
        for spec in specs:
            results.append(train_one_model(spec, splits, args, output_dir))
            write_results(results, output_dir)
        selected = max(results, key=run_selection_key)
        save_selected_model(selected, output_dir)
        # Test is evaluated exactly once, after validation-only model selection.
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(output_dir / "selected_model.h5")
        test_result = evaluate_policy_topn_arrays(model, "test", splits["test"], args.top_n,
                                                 args.eval_batch_size, args.predict_batch_size)
        test_result["model_name"] = selected["name"]
        (output_dir / "test_metrics.json").write_text(json.dumps(test_result, indent=2), encoding="utf-8")
        write_topn_rows(output_dir / "test_topn_accuracy.csv", test_result["topn"])
        selected["split_topn"]["test"] = test_result
        (output_dir / "selected_summary.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print("\n=== selected final-epoch model ===")
        print("selected model", selected["name"])
        for row in test_result["topn"]:
            print(f"test top-{row['top_n']}: {row['accuracy'] * 100:.3f}% ({row['hits']}/{row['positions']})")
        print("test cross entropy", test_result["cross_entropy"])
    finally:
        for split in splits.values():
            split.close()


if __name__ == "__main__":
    main()
