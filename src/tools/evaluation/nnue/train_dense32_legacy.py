#!/usr/bin/env python3
"""Reproduce the old Dense32 NNUE trainer on a sampled board_data cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input


N_PHASES = 60
LEGACY_BIT_SHIFTS = np.arange(63, -1, -1, dtype=np.uint64)


def next_model_dir(root: Path, date: str, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    used: list[int] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parts = child.name.split("_", 2)
        if len(parts) >= 2 and parts[0] == date and parts[1].isdigit():
            used.append(int(parts[1]))
    return root / f"{date}_{max(used, default=0) + 1}_{name}"


def select_indices(
    phases: np.ndarray,
    limit: int,
    phase_min: int,
    phase_max: int,
    phase_eq: int,
) -> np.ndarray:
    if phase_eq >= 0:
        mask = phases == phase_eq
    elif phase_min >= 0:
        mask = (phases >= phase_min) & (phases <= phase_max)
    else:
        mask = np.ones(phases.shape[0], dtype=np.bool_)
    idx = np.flatnonzero(mask)
    if limit > 0:
        idx = idx[:limit]
    return idx


def make_legacy_features(player: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    features = np.empty((player.shape[0], 128), dtype=np.float32)
    features[:, :64] = ((player[:, None] >> LEGACY_BIT_SHIFTS[None, :]) & 1).astype(np.float32, copy=False)
    features[:, 64:] = ((opponent[:, None] >> LEGACY_BIT_SHIFTS[None, :]) & 1).astype(np.float32, copy=False)
    return features


def build_model(hidden_size: int, hidden_layers: int) -> tf.keras.Model:
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(128,), name="in"))
    for i in range(hidden_layers):
        model.add(Dense(hidden_size, activation="relu", name=f"layer_{i}"))
    model.add(Dense(1, name="output_layer"))
    model.compile(loss="mse", metrics=["mae"], optimizer="adam")
    return model


def save_history(path: Path, history: tf.keras.callbacks.History) -> None:
    keys = list(history.history.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for i in range(len(history.history[keys[0]])):
            writer.writerow([i + 1, *[history.history[key][i] for key in keys]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-cache", default="model/nnue_records223plus_train10m_val1m_seed20260725_samples.npz")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--model-name", default="nnue_legacy_dense32_records223plus")
    parser.add_argument("--date", default="20260726")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=2_500_000)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--train-phase-min", type=int, default=12)
    parser.add_argument("--train-phase-max", type=int, default=59)
    parser.add_argument("--train-phase", type=int, default=-1)
    parser.add_argument("--val-phase", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    print("tensorflow version", tf.__version__, flush=True)

    out_dir = next_model_dir(Path(args.model_root), args.date, args.model_name)
    out_dir.mkdir(parents=True, exist_ok=False)

    loaded = np.load(Path(args.sample_cache))
    train_idx = select_indices(
        loaded["train_phase"],
        args.train_samples,
        args.train_phase_min,
        args.train_phase_max,
        args.train_phase,
    )
    val_idx = select_indices(
        loaded["val_phase"],
        args.val_samples,
        -1,
        N_PHASES - 1,
        args.val_phase,
    )

    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError(f"empty selection train {train_idx.size} val {val_idx.size}")

    print(
        f"cache {args.sample_cache} train_selected {train_idx.size} val_selected {val_idx.size} "
        f"train_phase_min {args.train_phase_min} train_phase_max {args.train_phase_max} "
        f"train_phase {args.train_phase} val_phase {args.val_phase}",
        flush=True,
    )
    print("building features", flush=True)
    train_x = make_legacy_features(loaded["train_player"][train_idx], loaded["train_opponent"][train_idx])
    val_x = make_legacy_features(loaded["val_player"][val_idx], loaded["val_opponent"][val_idx])
    train_y = loaded["train_score"][train_idx].astype(np.float32, copy=False)
    val_y = loaded["val_score"][val_idx].astype(np.float32, copy=False)

    meta = {
        "sample_cache": str(Path(args.sample_cache).resolve()),
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "train_phase_counts": np.bincount(loaded["train_phase"][train_idx], minlength=N_PHASES).astype(int).tolist(),
        "val_phase_counts": np.bincount(loaded["val_phase"][val_idx], minlength=N_PHASES).astype(int).tolist(),
        "train_score_mean": float(train_y.mean()),
        "train_score_mae_zero": float(np.abs(train_y).mean()),
        "val_score_mean": float(val_y.mean()),
        "val_score_mae_zero": float(np.abs(val_y).mean()),
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    model = build_model(args.hidden_size, args.hidden_layers)
    model.summary(print_fn=lambda line: print(line, flush=True))
    initial = model.evaluate(val_x, val_y, batch_size=args.batch_size, verbose=0, return_dict=True)
    print(f"initial val_loss {initial['loss']:.6f} val_mae {initial['mae']:.6f}", flush=True)

    callbacks = [EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)]
    history = model.fit(
        train_x,
        train_y,
        initial_epoch=0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        validation_data=(val_x, val_y),
        verbose=2,
        shuffle=True,
    )
    final_eval = model.evaluate(val_x, val_y, batch_size=args.batch_size, verbose=0, return_dict=True)
    print(f"final val_loss {final_eval['loss']:.6f} val_mae {final_eval['mae']:.6f}", flush=True)
    model.save(out_dir / "model.h5")
    save_history(out_dir / "history.csv", history)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
