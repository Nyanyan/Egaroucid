#!/usr/bin/env python3
"""Calibrate pattern-input NNUE output heads with per-phase linear correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_pattern_nnue import (
    ARCHES,
    PatternNNUE,
    batch_iter,
    evaluate_loss,
    evaluate_quantized_loss,
    export_model,
    load_sample_cache,
    make_active_feature_columns,
    make_batch,
    next_model_dir,
)


def mae_mse(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    err = pred.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
    return float(np.mean(np.abs(err))), float(np.mean(err * err))


@torch.no_grad()
def predict_float(
    model: PatternNNUE,
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
    for idx in batch_iter(n, batch_size, False, 0):
        features, opponent_features, phases, _target = make_batch(samples, prefix, idx, device)
        out[idx] = model(features, phases, opponent_features).detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def summarize_by_phase(pred: np.ndarray, target: np.ndarray, phases: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for phase in range(60):
        idx = phases == phase
        count = int(idx.sum())
        if count == 0:
            rows.append({"phase": phase, "count": 0})
            continue
        phase_pred = pred[idx]
        phase_target = target[idx]
        mae, mse = mae_mse(phase_pred, phase_target)
        rows.append(
            {
                "phase": phase,
                "count": count,
                "target_mean": float(np.mean(phase_target)),
                "pred_mean": float(np.mean(phase_pred)),
                "mae": mae,
                "mse": mse,
            }
        )
    return rows


def fit_phase_calibration(
    pred: np.ndarray,
    target: np.ndarray,
    phases: np.ndarray,
    min_count: int,
    slope_min: float,
    slope_max: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    slopes = np.ones(60, dtype=np.float32)
    offsets = np.zeros(60, dtype=np.float32)
    rows: list[dict[str, float | int]] = []
    for phase in range(60):
        idx = phases == phase
        count = int(idx.sum())
        if count < min_count:
            rows.append({"phase": phase, "count": count, "slope": 1.0, "offset": 0.0, "used": 0})
            continue
        x = pred[idx].astype(np.float64, copy=False)
        y = target[idx].astype(np.float64, copy=False)
        var_x = float(np.var(x))
        slope = 1.0 if var_x <= 1.0e-12 else float(np.cov(x, y, bias=True)[0, 1] / var_x)
        slope = float(np.clip(slope, slope_min, slope_max))
        offset = float(np.mean(y) - slope * np.mean(x))
        slopes[phase] = slope
        offsets[phase] = offset
        before_mae, before_mse = mae_mse(x.astype(np.float32), y.astype(np.float32))
        after = (slope * x + offset).astype(np.float32)
        after_mae, after_mse = mae_mse(after, y.astype(np.float32))
        rows.append(
            {
                "phase": phase,
                "count": count,
                "slope": slope,
                "offset": offset,
                "used": 1,
                "before_mae": before_mae,
                "before_mse": before_mse,
                "after_mae": after_mae,
                "after_mse": after_mse,
            }
        )
    return slopes, offsets, rows


def apply_calibration(model: PatternNNUE, slopes: np.ndarray, offsets: np.ndarray) -> None:
    with torch.no_grad():
        slope_t = torch.from_numpy(slopes).to(model.output_weight.device, dtype=model.output_weight.dtype)
        offset_t = torch.from_numpy(offsets).to(model.output_bias.device, dtype=model.output_bias.dtype)
        model.output_weight.mul_(slope_t[:, None])
        model.output_bias.mul_(slope_t).add_(offset_t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-cache", required=True)
    parser.add_argument("--model-state", required=True)
    parser.add_argument("--arch", default="pft32", choices=sorted(ARCHES))
    parser.add_argument("--ft-dim", type=int, default=0)
    parser.add_argument("--hidden1", type=int, default=0)
    parser.add_argument("--hidden2", type=int, default=0)
    parser.add_argument("--perspectives", choices=["single", "dual"], default="dual")
    parser.add_argument("--active-eval-types", type=int, default=16)
    parser.add_argument("--train-limit", type=int, default=1_000_000)
    parser.add_argument("--val-limit", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--layer-weight-scale", type=int, default=32)
    parser.add_argument("--ft-weight-bits", type=int, choices=[8, 16], default=8)
    parser.add_argument("--slope-min", type=float, default=0.5)
    parser.add_argument("--slope-max", type=float, default=3.0)
    parser.add_argument("--min-count", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--model-name", default="nnue_pattern_output_calibrated")
    parser.add_argument("--date", default="")
    parser.add_argument("--out-file", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arch_dims = ARCHES[args.arch]
    ft_dim = args.ft_dim or arch_dims[0]
    hidden1 = args.hidden1 or arch_dims[1]
    hidden2 = args.hidden2 or arch_dims[2]
    active_columns = make_active_feature_columns(args.active_eval_types)
    perspectives = 2 if args.perspectives == "dual" else 1
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    samples = load_sample_cache(Path(args.sample_cache))
    samples["perspectives"] = perspectives
    model = PatternNNUE(ft_dim, hidden1, hidden2, perspectives, active_columns)
    try:
        checkpoint = torch.load(args.model_state, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_state, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    train_n = min(args.train_limit, len(samples["train_score"])) if args.train_limit > 0 else len(samples["train_score"])
    val_n = min(args.val_limit, len(samples["val_score"])) if args.val_limit > 0 else len(samples["val_score"])
    train_pred = predict_float(model, samples, "train", args.batch_size, device, train_n)
    train_target = samples["train_score"][:train_n].astype(np.float32, copy=False)
    train_phase = samples["train_phase"][:train_n]
    slopes, offsets, calibration_rows = fit_phase_calibration(
        train_pred,
        train_target,
        train_phase,
        args.min_count,
        args.slope_min,
        args.slope_max,
    )

    before_val_mse, before_val_mae, before_val_seen = evaluate_loss(model, samples, "val", args.batch_size, device, val_n)
    before_q_mse, before_q_mae, before_q_seen = evaluate_quantized_loss(
        model,
        samples,
        "val",
        args.batch_size,
        val_n,
        args.layer_weight_scale,
        args.ft_weight_bits,
    )
    apply_calibration(model, slopes, offsets)
    after_val_mse, after_val_mae, after_val_seen = evaluate_loss(model, samples, "val", args.batch_size, device, val_n)
    after_q_mse, after_q_mae, after_q_seen = evaluate_quantized_loss(
        model,
        samples,
        "val",
        args.batch_size,
        val_n,
        args.layer_weight_scale,
        args.ft_weight_bits,
    )

    val_pred = predict_float(model, samples, "val", args.batch_size, device, val_n)
    val_target = samples["val_score"][:val_n].astype(np.float32, copy=False)
    val_phase = samples["val_phase"][:val_n]

    date = args.date or __import__("datetime").datetime.now().strftime("%Y%m%d")
    out_dir = next_model_dir(Path(args.model_root), date, args.model_name)
    out_file = Path(args.out_file) if args.out_file else out_dir / f"eval_nnue_pattern_{args.arch}_calibrated.egevnnue"
    export_model(model, out_file, args.layer_weight_scale, args.ft_weight_bits)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "arch": args.arch,
            "ft_dim": ft_dim,
            "hidden1": hidden1,
            "hidden2": hidden2,
            "perspectives": perspectives,
            "active_eval_types": args.active_eval_types,
            "active_feature_columns": active_columns,
            "layer_weight_scale": args.layer_weight_scale,
            "ft_weight_bits": args.ft_weight_bits,
            "calibration_slopes": slopes,
            "calibration_offsets": offsets,
        },
        out_dir / "model_state.pt",
    )
    summary = {
        "method": "pattern_nnue_per_phase_output_linear_calibration",
        "sample_cache": str(Path(args.sample_cache).resolve()),
        "source_model_state": str(Path(args.model_state).resolve()),
        "arch": args.arch,
        "ft_dim": ft_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "perspectives": args.perspectives,
        "active_eval_types": args.active_eval_types,
        "train_limit": int(train_n),
        "val_limit": int(val_n),
        "layer_weight_scale": args.layer_weight_scale,
        "ft_weight_bits": args.ft_weight_bits,
        "slope_min": args.slope_min,
        "slope_max": args.slope_max,
        "before": {
            "float_val_mse": before_val_mse,
            "float_val_mae": before_val_mae,
            "float_val_samples": int(before_val_seen),
            "quantized_val_mse": before_q_mse,
            "quantized_val_mae": before_q_mae,
            "quantized_val_samples": int(before_q_seen),
        },
        "after": {
            "float_val_mse": after_val_mse,
            "float_val_mae": after_val_mae,
            "float_val_samples": int(after_val_seen),
            "quantized_val_mse": after_q_mse,
            "quantized_val_mae": after_q_mae,
            "quantized_val_samples": int(after_q_seen),
        },
        "calibration": calibration_rows,
        "after_float_by_phase": summarize_by_phase(val_pred, val_target, val_phase),
        "out_file": str(out_file.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["before"], indent=2), flush=True)
    print(json.dumps(summary["after"], indent=2), flush=True)
    print(f"wrote {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
