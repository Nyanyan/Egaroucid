#!/usr/bin/env python3
"""Sweep learning rates for pattern-score NNUE on one fixed sample set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from train_pattern_nnue import build_manifest
from train_pattern_score_nnue import (
    ARCHES,
    DEFAULT_LAYER_WEIGHT_SCALE,
    PatternScoreNNUE,
    evaluate_loss,
    initialize_from_shared_state,
    iter_index_batches,
    load_pattern_score_samples,
    make_batch,
    make_pattern_column_names,
    next_model_dir,
    select_active_columns,
)


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_lr_list(text: str) -> list[float]:
    values = [float(x) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("--lr-list must not be empty")
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lr",
                "epoch",
                "elapsed_ms",
                "train_epoch_mse",
                "train_mse",
                "train_mae",
                "train_metric_samples",
                "val_mse",
                "val_mae",
                "val_metric_samples",
                "best_epoch",
                "best_val_mse",
                "best_val_mae",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    initial_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# パターン配置点数NNUE 学習率探索\n\n")
        f.write("## 目的\n\n")
        f.write("800,000,000局面の本番同等データで、100epoch学習に使う学習率候補を短期学習で比較する。\n\n")
        f.write("## 条件\n\n")
        f.write(f"- データルート: `{Path(args.data_root).resolve()}`\n")
        f.write(f"- 使用範囲: records{args.record_start}以降\n")
        f.write(f"- train: {args.train_samples:,}局面\n")
        f.write(f"- validation: {args.val_samples:,}局面\n")
        f.write(f"- epoch: {args.epochs}\n")
        f.write(f"- 候補学習率: {', '.join(str(x) for x in parse_lr_list(args.lr_list))}\n")
        f.write(f"- 初期化元: `{args.init_shared_state}`\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- シャッフルブロック: {args.shuffle_block_size:,}局面\n")
        f.write(f"- サンプル読み込み時間: {metadata['sample_load_ms']:,} ms\n\n")
        f.write("## 初期値\n\n")
        f.write("| lr | train MSE | train MAE | validation MSE | validation MAE |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in initial_rows:
            f.write(
                f"| {row['lr']} | {row['train_mse']:.6f} | {row['train_mae']:.6f} | "
                f"{row['val_mse']:.6f} | {row['val_mae']:.6f} |\n"
            )
        f.write("\n## 最良値\n\n")
        f.write("| lr | best epoch | validation MSE | validation MAE |\n")
        f.write("|---:|---:|---:|---:|\n")
        for row in best_rows:
            f.write(
                f"| {row['lr']} | {row['best_epoch']} | "
                f"{row['best_val_mse']:.6f} | {row['best_val_mae']:.6f} |\n"
            )
        f.write("\n## epoch別\n\n")
        f.write("| lr | epoch | train epoch MSE | train MSE | train MAE | validation MSE | validation MAE | best epoch |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['lr']} | {row['epoch']} | {row['train_epoch_mse']:.6f} | "
                f"{row['train_mse']:.6f} | {row['train_mae']:.6f} | "
                f"{row['val_mse']:.6f} | {row['val_mae']:.6f} | {row['best_epoch']} |\n"
            )
        f.write("\n## 診断観点\n\n")
        f.write("- best epochが最終epochなら、未収束または学習率が低すぎる可能性を疑う。\n")
        f.write("- validation MSEが早期に悪化するなら、学習率過大または過学習を疑う。\n")
        f.write("- train epoch MSEだけでなく、validation MSE/MAEの落ち方を優先する。\n")


def make_model(args: argparse.Namespace, active_columns: np.ndarray, input_dim: int, device: torch.device) -> PatternScoreNNUE:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    hidden1 = args.hidden1
    hidden2 = args.hidden2
    if hidden1 <= 0 or hidden2 <= 0:
        arch_dims = ARCHES[args.arch]
        hidden1 = hidden1 or arch_dims[0]
        hidden2 = hidden2 or arch_dims[1]
    model = PatternScoreNNUE(input_dim, hidden1, hidden2, args.phase_specific_score_table).to(device)
    return model


def main() -> int:
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--lr-list", default="0.0003,0.001,0.003")
    parser.add_argument("--dense-weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--init-shared-state", required=True)
    parser.add_argument("--train-metric-limit", type=int, default=40_000_000)
    parser.add_argument("--val-metric-limit", type=int, default=40_000_000)
    parser.add_argument("--progress-interval-sec", type=int, default=30)
    parser.add_argument("--full-read-ratio", type=float, default=0.10)
    parser.add_argument("--shuffle-block-size", type=int, default=5_000_000)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    lr_values = parse_lr_list(args.lr_list)
    active_columns = select_active_columns(args.pattern_set)
    input_dim = int(active_columns.size)
    active_column_names = make_pattern_column_names(active_columns)
    data_root = Path(args.data_root)
    entries, manifest_phase_counts = build_manifest(data_root, args.record_start, args.record_end)
    total_records = entries[-1].begin + entries[-1].records if entries else 0
    print(
        f"manifest_files {len(entries)} total_records {total_records} data_root {data_root} "
        f"pattern_set {args.pattern_set} active_feature_columns {','.join(str(int(x)) for x in active_columns)}",
        flush=True,
    )
    print(
        f"lr_sweep train_samples {args.train_samples} val_samples {args.val_samples} "
        f"epochs {args.epochs} lr_list {','.join(str(x) for x in lr_values)}",
        flush=True,
    )

    load_start_ms = now_ms()
    samples = load_pattern_score_samples(
        entries,
        args.train_samples,
        args.val_samples,
        args.seed,
        args.progress_interval_sec,
        args.full_read_ratio,
        active_columns,
    )
    sample_load_ms = now_ms() - load_start_ms
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    date = datetime.now().strftime("%Y%m%d")
    model_name = args.model_name or (
        f"nnue_pattern_score_lr_sweep_{args.pattern_set}_train{args.train_samples}_val{args.val_samples}_e{args.epochs}"
    )
    out_dir = Path(args.out_dir) if args.out_dir else next_model_dir(Path(args.model_root), date, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    initial_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    metadata = {
        "data_root": str(data_root.resolve()),
        "record_start": args.record_start,
        "record_end": args.record_end,
        "available_records": int(total_records),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "epochs": args.epochs,
        "lr_list": lr_values,
        "pattern_set": args.pattern_set,
        "phase_specific_score_table": args.phase_specific_score_table,
        "active_feature_columns": active_columns.tolist(),
        "active_feature_column_names": active_column_names,
        "init_shared_state": args.init_shared_state,
        "sample_load_ms": sample_load_ms,
        "manifest_phase_counts": [int(x) for x in manifest_phase_counts],
        "train_phase_counts": [int(x) for x in samples["train_phase_counts"]],
        "val_phase_counts": [int(x) for x in samples["val_phase_counts"]],
        "feature_storage_dtype": str(samples["train_features"].dtype),
        "score_storage_dtype": str(samples["train_score"].dtype),
    }

    for lr in lr_values:
        print(f"lr_start lr {lr}", flush=True)
        model = make_model(args, active_columns, input_dim, device)
        model.initialize_output_bias(samples["train_phase"], samples["train_score"].astype(np.float32, copy=False))
        initialize_from_shared_state(model, Path(args.init_shared_state), device)
        model.to(device)
        sparse_optimizer = torch.optim.SparseAdam([model.score_table.weight], lr=lr)
        dense_params = [
            model.input_bias,
            *model.hidden1.parameters(),
            *model.hidden2.parameters(),
            model.output_weight,
            model.output_bias,
        ]
        dense_optimizer = torch.optim.AdamW(dense_params, lr=lr, weight_decay=args.dense_weight_decay)
        train_mse, train_mae, train_n = evaluate_loss(model, samples, "train", args.batch_size, device, args.train_metric_limit)
        val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.val_metric_limit)
        initial = {
            "lr": lr,
            "train_mse": train_mse,
            "train_mae": train_mae,
            "train_metric_samples": train_n,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_metric_samples": val_n,
        }
        initial_rows.append(initial)
        print(
            f"initial lr {lr} train_mse {train_mse:.6f} train_mae {train_mae:.6f} "
            f"train_metric_samples {train_n} val_mse {val_mse:.6f} val_mae {val_mae:.6f} "
            f"val_metric_samples {val_n}",
            flush=True,
        )
        best_epoch = 0
        best_val_mse = math.inf
        best_val_mae = math.inf
        for epoch in range(1, args.epochs + 1):
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
                sparse_optimizer.zero_grad(set_to_none=True)
                dense_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                sparse_optimizer.step()
                dense_optimizer.step()
                batch_n = int(target.numel())
                seen += batch_n
                running_loss += float(loss.detach().cpu()) * batch_n
                current_ms = now_ms()
                if args.progress_interval_sec > 0 and current_ms >= next_log_ms:
                    print(
                        f"lr_training lr {lr} epoch {epoch} processed {seen}/{args.train_samples} "
                        f"train_epoch_mse {running_loss / max(1, seen):.6f} elapsed_ms {current_ms - start_ms}",
                        flush=True,
                    )
                    next_log_ms = current_ms + args.progress_interval_sec * 1000
            train_mse, train_mae, train_n = evaluate_loss(model, samples, "train", args.batch_size, device, args.train_metric_limit)
            val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.val_metric_limit)
            if val_mse < best_val_mse:
                best_epoch = epoch
                best_val_mse = val_mse
                best_val_mae = val_mae
            row = {
                "lr": lr,
                "epoch": epoch,
                "elapsed_ms": now_ms() - start_ms,
                "train_epoch_mse": running_loss / max(1, seen),
                "train_mse": train_mse,
                "train_mae": train_mae,
                "train_metric_samples": train_n,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_metric_samples": val_n,
                "best_epoch": best_epoch,
                "best_val_mse": best_val_mse,
                "best_val_mae": best_val_mae,
            }
            all_rows.append(row)
            print(
                f"lr_epoch lr {lr} epoch {epoch} elapsed_ms {row['elapsed_ms']} "
                f"train_epoch_mse {row['train_epoch_mse']:.6f} train_mse {train_mse:.6f} "
                f"train_mae {train_mae:.6f} val_mse {val_mse:.6f} val_mae {val_mae:.6f} "
                f"best_epoch {best_epoch} best_val_mse {best_val_mse:.6f}",
                flush=True,
            )
        best_rows.append({"lr": lr, "best_epoch": best_epoch, "best_val_mse": best_val_mse, "best_val_mae": best_val_mae})
        del model, sparse_optimizer, dense_optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_dir / "lr_sweep.csv", all_rows)
    with (out_dir / "lr_sweep_initial.json").open("w", encoding="utf-8") as f:
        json.dump(initial_rows, f, ensure_ascii=False, indent=2)
    with (out_dir / "lr_sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "best": best_rows, "rows": all_rows}, f, ensure_ascii=False, indent=2)
    write_markdown(out_dir / "README.md", args, all_rows, initial_rows, best_rows, metadata)
    print(f"wrote {out_dir / 'lr_sweep.csv'}", flush=True)
    print(f"wrote {out_dir / 'README.md'}", flush=True)
    for row in best_rows:
        print(
            f"lr_summary lr {row['lr']} best_epoch {row['best_epoch']} "
            f"best_val_mse {row['best_val_mse']:.6f} best_val_mae {row['best_val_mae']:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
