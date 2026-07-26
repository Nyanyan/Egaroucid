#!/usr/bin/env python3
"""Train pattern-score NNUE with validation-driven learning-rate decay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from train_pattern_score_nnue import (
    ARCHES,
    COLUMNWISE_FEATURE_STARTS,
    COLUMNWISE_TOTAL_INPUT_FEATURES,
    DEFAULT_LAYER_WEIGHT_SCALE,
    N_FEATURE_COLUMNS,
    N_PHASES,
    PATTERN_SCORE_INPUT_KIND,
    PATTERN_SCORE_PHASED_INPUT_KIND,
    PatternScoreNNUE,
    build_manifest,
    evaluate_loss,
    evaluate_quantized_loss,
    export_model,
    initialize_from_shared_state,
    iter_index_batches,
    load_pattern_score_samples,
    load_sample_cache,
    load_samples,
    make_batch,
    make_pattern_column_names,
    next_model_dir,
    prepare_samples_for_pattern_score,
    save_sample_cache,
    select_active_columns,
)


def now_ms() -> int:
    return int(time.time() * 1000)


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def install_tee(log_file: Path) -> tuple[TextIO, TextIO, TextIO]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fp = log_file.open("a", encoding="utf-8", buffering=1)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = Tee(sys.__stdout__, fp)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, fp)  # type: ignore[assignment]
    return fp, old_stdout, old_stderr


def restore_stdio(stdout: TextIO, stderr: TextIO) -> None:
    sys.stdout = stdout
    sys.stderr = stderr


def create_optimizers(
    model: PatternScoreNNUE,
    lr: float,
    dense_weight_decay: float,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    sparse_optimizer = torch.optim.SparseAdam([model.score_table.weight], lr=lr)
    dense_params = [
        model.input_bias,
        *model.hidden1.parameters(),
        *model.hidden2.parameters(),
        model.output_weight,
        model.output_bias,
    ]
    dense_optimizer = torch.optim.AdamW(dense_params, lr=lr, weight_decay=dense_weight_decay)
    return sparse_optimizer, dense_optimizer


def set_optimizer_lr(optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer], lr: float) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = lr


def copy_state_to_cpu(model: PatternScoreNNUE) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_history_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "epoch",
        "lr",
        "lr_after_epoch",
        "lr_drop_count",
        "elapsed_ms",
        "train_epoch_mse",
        "train_mse",
        "train_mae",
        "train_metric_samples",
        "val_mse",
        "val_mae",
        "val_metric_samples",
        "improved",
        "best_epoch",
        "best_val_mse",
        "best_val_mae",
        "epochs_without_val_mse_improvement",
        "lr_changed_after_epoch",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def write_line_chart_svg(
    path: Path,
    title: str,
    y_label: str,
    rows: list[dict[str, object]],
    series: list[tuple[str, str, str]],
    log_y: bool = False,
) -> None:
    if not rows:
        return
    points: dict[str, list[tuple[float, float]]] = {}
    y_values: list[float] = []
    for key, label, _color in series:
        pts: list[tuple[float, float]] = []
        for row in rows:
            epoch = as_float(row, "epoch")
            value = as_float(row, key)
            if epoch is None or value is None:
                continue
            if log_y:
                if value <= 0:
                    continue
                value = math.log10(value)
            pts.append((epoch, value))
            y_values.append(value)
        if pts:
            points[label] = pts
    if not y_values or not points:
        return

    width = 1040
    height = 560
    left = 82
    right = 250
    top = 56
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = 0.0
    x_max = max(float(row["epoch"]) for row in rows)
    y_min = min(y_values)
    y_max = max(y_values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def sx(x: float) -> float:
        if x_max <= x_min:
            return left
        return left + plot_w * ((x - x_min) / (x_max - x_min))

    def sy(y: float) -> float:
        return top + plot_h * ((y_max - y) / (y_max - y_min))

    def fmt_y(y: float) -> str:
        if log_y:
            return f"{10 ** y:.4g}"
        return f"{y:.3f}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="32" font-family="Arial, sans-serif" font-size="20" fill="#111827">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>',
    ]
    for i in range(6):
        t = i / 5.0
        y = y_min + (y_max - y_min) * t
        py = sy(y)
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left - 12}" y="{py + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#374151">{fmt_y(y)}</text>'
        )
    for x in range(0, int(x_max) + 1, max(1, int(math.ceil(x_max / 10.0)))):
        px = sx(float(x))
        parts.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        parts.append(
            f'<text x="{px:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#374151">{x}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="14" fill="#111827">epoch</text>'
    )
    parts.append(
        f'<text x="24" y="{top + plot_h / 2:.2f}" transform="rotate(-90 24 {top + plot_h / 2:.2f})" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">{y_label}</text>'
    )

    color_by_label = {label: color for _key, label, color in series}
    legend_x = left + plot_w + 28
    legend_y = top + 14
    for i, (label, pts) in enumerate(points.items()):
        color = color_by_label[label]
        d = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.4" fill="{color}"/>')
        ly = legend_y + i * 24
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 22}" y2="{ly}" stroke="{color}" stroke-width="2.8"/>')
        parts.append(
            f'<text x="{legend_x + 32}" y="{ly + 4}" font-family="Arial, sans-serif" '
            f'font-size="13" fill="#111827">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    best: dict[str, object],
    metadata: dict[str, object],
) -> None:
    save_history_csv(out_dir / "history.csv", rows)
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "best": best, "rows": rows}, f, ensure_ascii=False, indent=2)
    with (out_dir / "best.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    write_line_chart_svg(
        out_dir / "history_mse.svg",
        "Pattern-score NNUE MSE",
        "MSE",
        rows,
        [
            ("train_epoch_mse", "train epoch MSE", "#2563eb"),
            ("train_mse", "train MSE", "#16a34a"),
            ("val_mse", "validation MSE", "#dc2626"),
        ],
    )
    write_line_chart_svg(
        out_dir / "history_mae.svg",
        "Pattern-score NNUE MAE",
        "MAE",
        rows,
        [
            ("train_mae", "train MAE", "#16a34a"),
            ("val_mae", "validation MAE", "#dc2626"),
        ],
    )
    write_line_chart_svg(
        out_dir / "history_lr.svg",
        "Learning rate",
        "learning rate",
        rows,
        [("lr", "learning rate", "#7c3aed")],
        log_y=True,
    )
    write_readme(out_dir / "README.md", args, rows, best, metadata)


def write_readme(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    best: dict[str, object],
    metadata: dict[str, object],
) -> None:
    lines = [
        "# パターン配置点数NNUE 学習率半減スケジュール学習",
        "",
        "## 条件",
        "",
        f"- データルート: `{metadata['data_root']}`",
        f"- 使用範囲: records{metadata['record_start']}以降",
        f"- 利用可能局面数: {metadata['available_records']:,}",
        f"- train: {metadata['train_samples']:,}局面",
        f"- validation: {metadata['val_samples']:,}局面",
        f"- 最大epoch: {metadata['epochs']}",
        f"- 初期学習率: {args.initial_lr}",
        f"- 学習率を下げる条件: validation MSEが{args.lr_patience}epoch連続で過去最良を更新しない",
        f"- 学習率倍率: {args.lr_decay_factor}",
        f"- pattern set: `{args.pattern_set}`",
        f"- arch: `{args.arch}`",
        f"- 配置点数表: {'60フェーズ別' if args.phase_specific_score_table else '全フェーズ共通'}",
        f"- 初期化元: `{args.init_shared_state}`",
        f"- ログ: `{metadata['log_file']}`",
        "",
        "## 出力",
        "",
        f"- 評価関数: `{metadata['out_file']}`",
        "- 学習履歴: `history.csv`",
        "- MSEグラフ: `history_mse.svg`",
        "- MAEグラフ: `history_mae.svg`",
        "- 学習率グラフ: `history_lr.svg`",
        "",
        "## 最良値",
        "",
        f"- best epoch: {best.get('epoch')}",
        f"- best validation MSE: {best.get('val_mse')}",
        f"- best validation MAE: {best.get('val_mae')}",
        f"- best learning rate: {best.get('lr')}",
        "",
        "## epoch別",
        "",
        "| epoch | lr | train epoch MSE | train MSE | train MAE | validation MSE | validation MAE | best epoch | 改善なしepoch数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['epoch']} | {row['lr']} | {float(row['train_epoch_mse']):.6f} | "
            f"{float(row['train_mse']):.6f} | {float(row['train_mae']):.6f} | "
            f"{float(row['val_mse']):.6f} | {float(row['val_mae']):.6f} | "
            f"{row['best_epoch']} | {row['epochs_without_val_mse_improvement']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_best_model(
    out_dir: Path,
    out_file: Path,
    model: PatternScoreNNUE,
    best_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    active_columns: np.ndarray,
    active_column_names: list[str],
    input_dim: int,
    best: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_model(model, out_file, args.layer_weight_scale, args.ft_weight_bits, active_columns)
    torch.save(
        {
            "state_dict": best_state,
            "arch": args.arch,
            "input_dim": input_dim,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "active_feature_columns": active_columns,
            "active_feature_column_names": active_column_names,
            "pattern_set": args.pattern_set,
            "phase_specific_score_table": args.phase_specific_score_table,
            "init_shared_state": args.init_shared_state,
            "columnwise_feature_starts": COLUMNWISE_FEATURE_STARTS,
            "columnwise_total_input_features": COLUMNWISE_TOTAL_INPUT_FEATURES,
            "layer_weight_scale": args.layer_weight_scale,
            "ft_weight_bits": args.ft_weight_bits,
            "schedule": {
                "initial_lr": args.initial_lr,
                "lr_decay_factor": args.lr_decay_factor,
                "lr_patience": args.lr_patience,
                "min_delta": args.min_delta,
            },
            "best": best,
        },
        out_dir / "model_state.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("EGAROUCID_INDEXED_DATA", "E:/egaroucid_data/train_data/bin_data/20241125_1"))
    parser.add_argument("--record-start", type=int, default=223)
    parser.add_argument("--record-end", type=int, default=-1)
    parser.add_argument("--arch", choices=sorted(ARCHES), default="ps48_32")
    parser.add_argument("--pattern-set", choices=["mo_end4", "first12"], default="first12")
    parser.add_argument("--phase-specific-score-table", action="store_true")
    parser.add_argument("--hidden1", type=int, default=0)
    parser.add_argument("--hidden2", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=800_000_000)
    parser.add_argument("--val-samples", type=int, default=40_000_000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--initial-lr", type=float, default=1.0e-2)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--dense-weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--layer-weight-scale", type=int, default=DEFAULT_LAYER_WEIGHT_SCALE)
    parser.add_argument("--ft-weight-bits", type=int, choices=[8, 16], default=16)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--train-metric-limit", type=int, default=40_000_000)
    parser.add_argument("--val-metric-limit", type=int, default=0)
    parser.add_argument("--quantized-metric-limit", type=int, default=40_000_000)
    parser.add_argument("--progress-interval-sec", type=int, default=30)
    parser.add_argument("--full-read-ratio", type=float, default=0.10)
    parser.add_argument("--shuffle-block-size", type=int, default=5_000_000)
    parser.add_argument("--legacy-full-feature-storage", action="store_true")
    parser.add_argument("--sample-cache", default="")
    parser.add_argument("--load-state", default="")
    parser.add_argument("--init-shared-state", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    active_columns = select_active_columns(args.pattern_set)
    active_column_names = make_pattern_column_names(active_columns)
    input_dim = int(active_columns.size)
    arch_dims = ARCHES[args.arch]
    args.hidden1 = args.hidden1 or arch_dims[0]
    args.hidden2 = args.hidden2 or arch_dims[1]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)

    date = datetime.now().strftime("%Y%m%d")
    score_table_scope = "phase60" if args.phase_specific_score_table else "shared"
    model_name = args.model_name or (
        f"nnue_pattern_score_{args.pattern_set}_{score_table_scope}_{args.arch}_records{args.record_start}plus_"
        f"train{args.train_samples}_val{args.val_samples}_e{args.epochs}_lr{args.initial_lr:g}_halve_p{args.lr_patience}"
    )
    out_dir = next_model_dir(Path(args.model_root), date, model_name)
    out_file = Path(args.out_file) if args.out_file else out_dir / f"eval_nnue_pattern_score_{args.pattern_set}_{score_table_scope}_{args.arch}.egevnnue"
    log_file = Path(args.log_file) if args.log_file else out_dir / "train.log"
    tee_file, old_stdout, old_stderr = install_tee(log_file)
    try:
        print(f"log_file {log_file}", flush=True)
        print(f"out_dir {out_dir}", flush=True)
        print(f"out_file {out_file}", flush=True)
        print(
            f"schedule initial_lr {args.initial_lr} lr_patience {args.lr_patience} "
            f"lr_decay_factor {args.lr_decay_factor} min_delta {args.min_delta} "
            f"min_lr {args.min_lr} max_epochs {args.epochs}",
            flush=True,
        )

        data_root = Path(args.data_root)
        entries, manifest_phase_counts = build_manifest(data_root, args.record_start, args.record_end)
        total_records = entries[-1].begin + entries[-1].records if entries else 0
        print(
            f"manifest_files {len(entries)} total_records {total_records} data_root {data_root} "
            f"pattern_set {args.pattern_set} "
            f"active_feature_columns {','.join(str(int(x)) for x in active_columns)} "
            f"columnwise_total_input_features {COLUMNWISE_TOTAL_INPUT_FEATURES}",
            flush=True,
        )
        print(
            f"arch {args.arch} input_dim {input_dim} hidden1 {args.hidden1} "
            f"hidden2 {args.hidden2} ft_weight_bits {args.ft_weight_bits} "
            f"phase_specific_score_table {int(args.phase_specific_score_table)}",
            flush=True,
        )
        metadata = {
            "data_root": str(data_root.resolve()),
            "record_start": args.record_start,
            "record_end": args.record_end,
            "available_records": int(total_records),
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "initial_lr": args.initial_lr,
            "lr_patience": args.lr_patience,
            "lr_decay_factor": args.lr_decay_factor,
            "min_delta": args.min_delta,
            "min_lr": args.min_lr,
            "dense_weight_decay": args.dense_weight_decay,
            "arch": args.arch,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "input_kind": PATTERN_SCORE_PHASED_INPUT_KIND if args.phase_specific_score_table else PATTERN_SCORE_INPUT_KIND,
            "input_dim": input_dim,
            "input_feature_columns": N_FEATURE_COLUMNS,
            "active_feature_columns": active_columns.tolist(),
            "active_feature_column_names": active_column_names,
            "pattern_set": args.pattern_set,
            "phase_specific_score_table": args.phase_specific_score_table,
            "init_shared_state": args.init_shared_state,
            "load_state": args.load_state,
            "columnwise_total_input_features": COLUMNWISE_TOTAL_INPUT_FEATURES,
            "layer_weight_scale": args.layer_weight_scale,
            "ft_weight_bits": args.ft_weight_bits,
            "seed": args.seed,
            "legacy_full_feature_storage": args.legacy_full_feature_storage,
            "shuffle_block_size": args.shuffle_block_size,
            "manifest_phase_counts": [int(x) for x in manifest_phase_counts],
            "out_file": str(out_file),
            "log_file": str(log_file),
        }
        rows: list[dict[str, object]] = []
        best: dict[str, object] = {
            "epoch": 0,
            "lr": args.initial_lr,
            "val_mse": float("inf"),
            "val_mae": float("inf"),
        }
        if args.dry_run:
            write_outputs(out_dir, args, rows, best, metadata)
            return 0

        load_start_ms = now_ms()
        if args.sample_cache and Path(args.sample_cache).exists():
            print(f"loading_sample_cache {args.sample_cache}", flush=True)
            samples = load_sample_cache(Path(args.sample_cache))
            prepare_samples_for_pattern_score(samples, active_columns)
        else:
            if args.legacy_full_feature_storage:
                samples = load_samples(
                    entries,
                    args.train_samples,
                    args.val_samples,
                    args.seed,
                    args.progress_interval_sec,
                    args.full_read_ratio,
                )
                prepare_samples_for_pattern_score(samples, active_columns)
            else:
                samples = load_pattern_score_samples(
                    entries,
                    args.train_samples,
                    args.val_samples,
                    args.seed,
                    args.progress_interval_sec,
                    args.full_read_ratio,
                    active_columns,
                )
            if args.sample_cache:
                meta = {
                    "data_root": str(data_root.resolve()),
                    "record_start": args.record_start,
                    "record_end": args.record_end,
                    "available_records": int(total_records),
                    "train_samples": args.train_samples,
                    "val_samples": args.val_samples,
                    "seed": args.seed,
                    "input_feature_columns": N_FEATURE_COLUMNS,
                    "active_feature_columns": active_columns.tolist(),
                    "pattern_set": args.pattern_set,
                    "pattern_score_local_features": bool(samples.get("pattern_score_local_features", False)),
                    "feature_storage_dtype": str(samples["train_features"].dtype),
                    "score_storage_dtype": str(samples["train_score"].dtype),
                }
                print(f"writing_sample_cache {args.sample_cache}", flush=True)
                save_sample_cache(Path(args.sample_cache), samples, meta)
        metadata["sample_load_ms"] = now_ms() - load_start_ms
        metadata["pattern_score_local_features"] = bool(samples.get("pattern_score_local_features", False))
        metadata["feature_storage_dtype"] = str(samples["train_features"].dtype)
        metadata["score_storage_dtype"] = str(samples["train_score"].dtype)
        metadata["train_phase_counts"] = [int(x) for x in samples["train_phase_counts"]]
        metadata["val_phase_counts"] = [int(x) for x in samples["val_phase_counts"]]

        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        print(f"device {device}", flush=True)
        model = PatternScoreNNUE(input_dim, args.hidden1, args.hidden2, args.phase_specific_score_table).to(device)
        model.initialize_output_bias(samples["train_phase"], samples["train_score"])
        model.to(device)
        if args.init_shared_state:
            initialize_from_shared_state(model, Path(args.init_shared_state), device)
        if args.load_state:
            print(f"loading_state {args.load_state}", flush=True)
            try:
                checkpoint = torch.load(args.load_state, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(args.load_state, map_location=device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.to(device)

        current_lr = args.initial_lr
        optimizers = create_optimizers(model, current_lr, args.dense_weight_decay)
        train_mse, train_mae, train_n = evaluate_loss(model, samples, "train", args.batch_size, device, args.train_metric_limit)
        val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.val_metric_limit)
        best = {
            "epoch": 0,
            "lr": current_lr,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "train_mse": train_mse,
            "train_mae": train_mae,
            "train_metric_samples": train_n,
            "val_metric_samples": val_n,
        }
        best_state = copy_state_to_cpu(model)
        save_best_model(out_dir, out_file, model, best_state, args, active_columns, active_column_names, input_dim, best)
        write_outputs(out_dir, args, rows, best, metadata)
        print(
            f"initial train_mse {train_mse:.6f} train_mae {train_mae:.6f} train_metric_samples {train_n} "
            f"val_mse {val_mse:.6f} val_mae {val_mae:.6f} val_metric_samples {val_n}",
            flush=True,
        )

        epochs_without_improvement = 0
        lr_drop_count = 0
        for epoch in range(1, args.epochs + 1):
            epoch_lr = current_lr
            model.train()
            seen = 0
            running_loss = 0.0
            start_ms = now_ms()
            next_log_ms = start_ms + args.progress_interval_sec * 1000
            for idx in iter_index_batches(
                args.train_samples,
                args.batch_size,
                True,
                args.seed + epoch,
                args.shuffle_block_size,
            ):
                features, phases, target = make_batch(samples, "train", idx, device)
                pred = model(features, phases)
                loss = torch.mean((pred - target) ** 2)
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise RuntimeError(f"non_finite_loss epoch={epoch} seen={seen} lr={current_lr}")
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                batch_n = int(target.numel())
                seen += batch_n
                running_loss += float(loss.detach().cpu()) * batch_n
                current_ms = now_ms()
                if args.progress_interval_sec > 0 and current_ms >= next_log_ms:
                    print(
                        f"epoch_training epoch {epoch} lr {current_lr} processed {seen}/{args.train_samples} "
                        f"train_epoch_mse {running_loss / max(1, seen):.6f} "
                        f"elapsed_ms {current_ms - start_ms}",
                        flush=True,
                    )
                    next_log_ms = current_ms + args.progress_interval_sec * 1000

            train_epoch_mse = running_loss / max(1, seen)
            train_mse, train_mae, train_n = evaluate_loss(model, samples, "train", args.batch_size, device, args.train_metric_limit)
            val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.val_metric_limit)
            improved = val_mse < float(best["val_mse"]) - args.min_delta
            lr_changed = False
            if improved:
                epochs_without_improvement = 0
                best = {
                    "epoch": epoch,
                    "lr": current_lr,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "train_mse": train_mse,
                    "train_mae": train_mae,
                    "train_metric_samples": train_n,
                    "val_metric_samples": val_n,
                }
                best_state = copy_state_to_cpu(model)
                save_best_model(out_dir, out_file, model, best_state, args, active_columns, active_column_names, input_dim, best)
                print(
                    f"best_updated epoch {epoch} lr {current_lr} "
                    f"val_mse {val_mse:.6f} val_mae {val_mae:.6f}",
                    flush=True,
                )
            else:
                epochs_without_improvement += 1

            epochs_without_improvement_for_row = epochs_without_improvement
            if epochs_without_improvement >= args.lr_patience and epoch < args.epochs:
                old_lr = current_lr
                new_lr = max(args.min_lr, current_lr * args.lr_decay_factor) if args.min_lr > 0 else current_lr * args.lr_decay_factor
                if new_lr < old_lr:
                    current_lr = new_lr
                    set_optimizer_lr(optimizers, current_lr)
                    lr_drop_count += 1
                    lr_changed = True
                    print(
                        f"lr_reduced epoch {epoch} old_lr {old_lr} new_lr {current_lr} "
                        f"epochs_without_val_mse_improvement {epochs_without_improvement}",
                        flush=True,
                    )
                    epochs_without_improvement = 0
                else:
                    print(
                        f"lr_not_reduced epoch {epoch} lr {current_lr} min_lr {args.min_lr} "
                        f"epochs_without_val_mse_improvement {epochs_without_improvement}",
                        flush=True,
                    )

            row = {
                "epoch": epoch,
                "lr": epoch_lr,
                "lr_after_epoch": current_lr,
                "lr_drop_count": lr_drop_count,
                "elapsed_ms": now_ms() - start_ms,
                "train_epoch_mse": train_epoch_mse,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "train_metric_samples": train_n,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_metric_samples": val_n,
                "improved": int(improved),
                "best_epoch": best["epoch"],
                "best_val_mse": best["val_mse"],
                "best_val_mae": best["val_mae"],
                "epochs_without_val_mse_improvement": epochs_without_improvement_for_row,
                "lr_changed_after_epoch": int(lr_changed),
            }
            rows.append(row)
            write_outputs(out_dir, args, rows, best, metadata)
            print(
                f"epoch {epoch} elapsed_ms {row['elapsed_ms']} lr {row['lr']} "
                f"lr_after_epoch {row['lr_after_epoch']} "
                f"train_epoch_mse {train_epoch_mse:.6f} "
                f"train_mse {train_mse:.6f} train_mae {train_mae:.6f} "
                f"train_metric_samples {train_n} "
                f"val_mse {val_mse:.6f} val_mae {val_mae:.6f} "
                f"val_metric_samples {val_n} "
                f"best_epoch {best['epoch']} best_val_mse {float(best['val_mse']):.6f} "
                f"epochs_without_val_mse_improvement {epochs_without_improvement_for_row}",
                flush=True,
            )

        model.load_state_dict(best_state)
        model.to(device)
        export_model(model, out_file, args.layer_weight_scale, args.ft_weight_bits, active_columns)
        q_val_mse, q_val_mae, q_val_n = evaluate_quantized_loss(
            model,
            samples,
            "val",
            args.batch_size,
            args.quantized_metric_limit,
            args.layer_weight_scale,
            args.ft_weight_bits,
        )
        best["quantized_val_mse"] = q_val_mse
        best["quantized_val_mae"] = q_val_mae
        best["quantized_val_metric_samples"] = q_val_n
        save_best_model(out_dir, out_file, model, best_state, args, active_columns, active_column_names, input_dim, best)
        write_outputs(out_dir, args, rows, best, metadata)
        print(
            f"quantized_best val_mse {q_val_mse:.6f} val_mae {q_val_mae:.6f} "
            f"val_metric_samples {q_val_n}",
            flush=True,
        )
        print(f"wrote {out_file}", flush=True)
        print(f"wrote {out_dir / 'model_state.pt'}", flush=True)
        print(f"wrote {out_dir / 'history.csv'}", flush=True)
        print(f"wrote {out_dir / 'history_mse.svg'}", flush=True)
        print(f"wrote {out_dir / 'history_mae.svg'}", flush=True)
        return 0
    finally:
        tee_file.flush()
        restore_stdio(old_stdout, old_stderr)
        tee_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
