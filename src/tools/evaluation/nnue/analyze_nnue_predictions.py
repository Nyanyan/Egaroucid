#!/usr/bin/env python3
"""Analyze NNUE predictions on an existing sample cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_nnue import (
    OthelloNNUE,
    ARCHES,
    evaluate_loss,
    load_sample_cache,
    quantize_model,
    quantized_forward,
)


def mae_mse(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    err = pred.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
    return float(np.mean(np.abs(err))), float(np.mean(err * err))


def phase_means(train_phase: np.ndarray, train_score: np.ndarray) -> np.ndarray:
    means = np.zeros(60, dtype=np.float32)
    for phase in range(60):
        idx = train_phase == phase
        if np.any(idx):
            means[phase] = float(np.mean(train_score[idx]))
    return means


@torch.no_grad()
def predict_float(
    model: OthelloNNUE,
    samples: dict[str, np.ndarray],
    prefix: str,
    batch_size: int,
    device: torch.device,
    limit: int,
) -> np.ndarray:
    model.eval()
    n = len(samples[f"{prefix}_score"])
    if limit > 0:
        n = min(n, limit)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = np.arange(start, end)
        player = samples[f"{prefix}_player"][idx]
        opponent = samples[f"{prefix}_opponent"][idx]
        features_np = np.concatenate([
            ((player[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & 1).astype(np.float32, copy=False),
            ((opponent[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & 1).astype(np.float32, copy=False),
        ], axis=1)
        phases = torch.from_numpy(samples[f"{prefix}_phase"][idx].astype(np.int64, copy=False)).to(device)
        features = torch.from_numpy(features_np).to(device)
        out[start:end] = model(features, phases).detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def summarize_by_phase(pred: np.ndarray, target: np.ndarray, phases: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for phase in range(60):
        idx = phases == phase
        if not np.any(idx):
            rows.append({"phase": phase, "count": 0})
            continue
        phase_pred = pred[idx]
        phase_target = target[idx]
        mae, mse = mae_mse(phase_pred, phase_target)
        rows.append({
            "phase": phase,
            "count": int(idx.sum()),
            "target_mean": float(np.mean(phase_target)),
            "target_mae_zero": float(np.mean(np.abs(phase_target))),
            "pred_mean": float(np.mean(phase_pred)),
            "mae": mae,
            "mse": mse,
        })
    return rows


def score_bucket_rows(pred: np.ndarray, target: np.ndarray) -> list[dict[str, float | int | str]]:
    buckets = [
        (-64, -33, "[-64,-33]"),
        (-32, -17, "[-32,-17]"),
        (-16, -9, "[-16,-9]"),
        (-8, -1, "[-8,-1]"),
        (0, 0, "[0,0]"),
        (1, 8, "[1,8]"),
        (9, 16, "[9,16]"),
        (17, 32, "[17,32]"),
        (33, 64, "[33,64]"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for lo, hi, label in buckets:
        idx = (target >= lo) & (target <= hi)
        if not np.any(idx):
            rows.append({"bucket": label, "count": 0})
            continue
        mae, mse = mae_mse(pred[idx], target[idx])
        rows.append({
            "bucket": label,
            "count": int(idx.sum()),
            "target_mean": float(np.mean(target[idx])),
            "pred_mean": float(np.mean(pred[idx])),
            "mae": mae,
            "mse": mse,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-cache", default="model/nnue_records223plus_train10m_val1m_seed20260725_samples.npz")
    parser.add_argument("--model-state", required=True)
    parser.add_argument("--arch", default="small", choices=sorted(ARCHES))
    parser.add_argument("--ft-dim", type=int, default=0)
    parser.add_argument("--hidden1", type=int, default=0)
    parser.add_argument("--hidden2", type=int, default=0)
    parser.add_argument("--layer-weight-scale", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arch_dims = ARCHES[args.arch]
    ft_dim = args.ft_dim or arch_dims[0]
    hidden1 = args.hidden1 or arch_dims[1]
    hidden2 = args.hidden2 or arch_dims[2]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    samples = load_sample_cache(Path(args.sample_cache))
    model = OthelloNNUE(ft_dim, hidden1, hidden2)
    checkpoint = torch.load(args.model_state, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    n = len(samples["val_score"])
    if args.limit > 0:
        n = min(n, args.limit)
    target = samples["val_score"][:n].astype(np.float32, copy=False)
    phases = samples["val_phase"][:n]

    float_mse, float_mae, float_n = evaluate_loss(model, samples, "val", args.batch_size, device, n)
    float_pred = predict_float(model, samples, "val", args.batch_size, device, n)
    q = quantize_model(model, args.layer_weight_scale)
    quant_pred = quantized_forward(q, samples["val_player"][:n], samples["val_opponent"][:n], phases)

    means = phase_means(samples["train_phase"], samples["train_score"])
    phase_mean_pred = means[phases]
    zero_pred = np.zeros(n, dtype=np.float32)

    zero_mae, zero_mse = mae_mse(zero_pred, target)
    phase_mean_mae, phase_mean_mse = mae_mse(phase_mean_pred, target)
    q_mae, q_mse = mae_mse(quant_pred, target)
    neg_float_mae, neg_float_mse = mae_mse(-float_pred, target)
    neg_q_mae, neg_q_mse = mae_mse(-quant_pred, target)

    result = {
        "sample_cache": str(Path(args.sample_cache).resolve()),
        "model_state": str(Path(args.model_state).resolve()),
        "arch": args.arch,
        "ft_dim": ft_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "layer_weight_scale": args.layer_weight_scale,
        "limit": int(n),
        "device": str(device),
        "overall": {
            "zero_mae": zero_mae,
            "zero_mse": zero_mse,
            "phase_mean_mae": phase_mean_mae,
            "phase_mean_mse": phase_mean_mse,
            "float_mae": float_mae,
            "float_mse": float_mse,
            "float_metric_samples": int(float_n),
            "quantized_mae": q_mae,
            "quantized_mse": q_mse,
            "negated_float_mae": neg_float_mae,
            "negated_float_mse": neg_float_mse,
            "negated_quantized_mae": neg_q_mae,
            "negated_quantized_mse": neg_q_mse,
            "target_mean": float(np.mean(target)),
            "float_pred_mean": float(np.mean(float_pred)),
            "quantized_pred_mean": float(np.mean(quant_pred)),
            "float_corr": float(np.corrcoef(float_pred, target)[0, 1]),
            "quantized_corr": float(np.corrcoef(quant_pred, target)[0, 1]),
        },
        "float_by_phase": summarize_by_phase(float_pred, target, phases),
        "quantized_by_phase": summarize_by_phase(quant_pred, target, phases),
        "quantized_by_score_bucket": score_bucket_rows(quant_pred, target),
    }
    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
