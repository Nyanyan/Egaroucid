#!/usr/bin/env python3
"""Train and export an NNUE-only evaluation function for Egaroucid.

The training input is board_data/*.dat records:
    uint64 player, uint64 opponent, int8 player_color, int8 policy, int8 score

The exported file is consumed by src/tools/evaluation/nnue/evaluate_nnue.hpp.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


RECORD_BYTES = 19
N_PHASES = 60
INPUT_FEATURES = 128
STEP = 32
ACTIVATION_SCALE = 16
FT_SCALE = 256
LAYER_WEIGHT_SCALE = 16
LAYER_ACC_SCALE = ACTIVATION_SCALE * LAYER_WEIGHT_SCALE
FT_SHIFT = 4
HIDDEN_SHIFT = 4
OUTPUT_SHIFT = 3
ACTIVATION_CLIP_FLOAT = 127.0 / ACTIVATION_SCALE

ARCHES = {
    "ft32": (32, 32, 32),
    "ft64": (64, 32, 32),
    "small": (128, 32, 32),
    "medium": (256, 32, 32),
    "large": (384, 32, 32),
    "wide": (256, 64, 32),
}

RECORD_DTYPE = np.dtype([
    ("player", "<u8"),
    ("opponent", "<u8"),
    ("player_color", "i1"),
    ("policy", "i1"),
    ("score", "i1"),
])

POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
BIT_SHIFTS = np.arange(64, dtype=np.uint64)


@dataclass
class FileEntry:
    path: Path
    record_num: int
    records: int
    begin: int


class OthelloNNUE(nn.Module):
    def __init__(self, ft_dim: int, hidden1: int, hidden2: int):
        super().__init__()
        self.ft_dim = ft_dim
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.ft_bias = nn.Parameter(torch.zeros(ft_dim))
        self.ft_weight = nn.Parameter(torch.empty(INPUT_FEATURES, ft_dim))
        self.hidden1 = nn.Linear(ft_dim * 2, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.output_weight = nn.Parameter(torch.empty(N_PHASES, hidden2))
        self.output_bias = nn.Parameter(torch.zeros(N_PHASES))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ft_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.hidden1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.normal_(self.hidden2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.normal_(self.output_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_bias)

    @staticmethod
    def clipped_relu(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, ACTIVATION_CLIP_FLOAT)

    def forward(self, features: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        stm = self.ft_bias + features @ self.ft_weight
        swapped = torch.cat([features[:, 64:], features[:, :64]], dim=1)
        non_stm = self.ft_bias + swapped @ self.ft_weight
        x = torch.cat([self.clipped_relu(stm), self.clipped_relu(non_stm)], dim=1)
        x = self.clipped_relu(self.hidden1(x))
        x = self.clipped_relu(self.hidden2(x))
        return (x * self.output_weight[phases]).sum(dim=1) + self.output_bias[phases]


@dataclass
class QuantizedNNUE:
    ft_bias: np.ndarray
    ft_weight: np.ndarray
    hidden1_bias: np.ndarray
    hidden1_weight: np.ndarray
    hidden2_bias: np.ndarray
    hidden2_weight: np.ndarray
    output_bias: np.ndarray
    output_weight: np.ndarray
    post_padded: int
    hidden1_padded: int
    hidden2_padded: int


def next_model_dir(root: Path, date: str, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    used = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        prefix = child.name.split("_", 2)
        if len(prefix) >= 2 and prefix[0] == date and prefix[1].isdigit():
            used.append(int(prefix[1]))
    idx = max(used, default=0) + 1
    return root / f"{date}_{idx}_{name}"


def parse_record_num(path: Path) -> int | None:
    name = path.name
    if not name.startswith("records"):
        return None
    suffix = name[len("records"):]
    return int(suffix) if suffix.isdigit() else None


def build_manifest(data_root: Path, record_start: int, record_end: int) -> list[FileEntry]:
    entries: list[FileEntry] = []
    begin = 0
    for record_dir in sorted(data_root.iterdir(), key=lambda p: parse_record_num(p) or -1):
        if not record_dir.is_dir():
            continue
        record_num = parse_record_num(record_dir)
        if record_num is None or record_num < record_start:
            continue
        if record_end >= 0 and record_num > record_end:
            continue
        for path in sorted(record_dir.glob("*.dat"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            records = path.stat().st_size // RECORD_BYTES
            if records <= 0:
                continue
            entries.append(FileEntry(path=path, record_num=record_num, records=records, begin=begin))
            begin += records
    return entries


def phase_counts(entries: Iterable[FileEntry]) -> np.ndarray:
    counts = np.zeros(N_PHASES, dtype=np.uint64)
    for entry in entries:
        # board_data is not phase-partitioned; exact phase counts need reading.
        _ = entry
    return counts


def make_feature_matrix(player: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    p = ((player[:, None] >> BIT_SHIFTS[None, :]) & 1).astype(np.float32, copy=False)
    o = ((opponent[:, None] >> BIT_SHIFTS[None, :]) & 1).astype(np.float32, copy=False)
    return np.concatenate([p, o], axis=1)


def make_feature_matrix_int(player: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    p = ((player[:, None] >> BIT_SHIFTS[None, :]) & 1).astype(np.int16, copy=False)
    o = ((opponent[:, None] >> BIT_SHIFTS[None, :]) & 1).astype(np.int16, copy=False)
    return np.concatenate([p, o], axis=1)


def popcount_u64(values: np.ndarray) -> np.ndarray:
    return POPCOUNT8[values.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def sample_indices(total: int, need: int, train_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if need > total:
        raise ValueError(f"requested samples {need} exceeds available records {total}")
    rng = np.random.default_rng(seed)
    if need == total:
        chosen = np.arange(total, dtype=np.uint64)
    else:
        chunks: list[np.ndarray] = []
        have = 0
        chunk_size = min(max(need // 2, 1_000_000), 50_000_000)
        while have < need:
            remaining = need - have
            draw = min(max(int(remaining * 1.3), remaining + 1024), chunk_size)
            chunks.append(rng.integers(0, total, size=draw, dtype=np.uint64))
            chosen = np.unique(np.concatenate(chunks))
            if chosen.size > need:
                chosen = rng.choice(chosen, size=need, replace=False)
                chunks = [chosen]
            have = int(chosen.size)
        chosen = chunks[0] if len(chunks) == 1 else np.unique(np.concatenate(chunks))
        if chosen.size > need:
            chosen = rng.choice(chosen, size=need, replace=False)
    validation = np.zeros(need, dtype=np.bool_)
    if need > train_samples:
        validation_positions = rng.choice(need, size=need - train_samples, replace=False)
        validation[validation_positions] = True
    order = np.argsort(chosen, kind="stable")
    return chosen[order], validation[order]


def load_samples(
    entries: list[FileEntry],
    train_samples: int,
    val_samples: int,
    seed: int,
    progress_interval_files: int,
) -> dict[str, np.ndarray]:
    total = entries[-1].begin + entries[-1].records if entries else 0
    need = train_samples + val_samples
    chosen, validation = sample_indices(total, need, train_samples, seed)

    train_player = np.empty(train_samples, dtype=np.uint64)
    train_opponent = np.empty(train_samples, dtype=np.uint64)
    train_phase = np.empty(train_samples, dtype=np.uint8)
    train_score = np.empty(train_samples, dtype=np.float32)
    val_player = np.empty(val_samples, dtype=np.uint64)
    val_opponent = np.empty(val_samples, dtype=np.uint64)
    val_phase = np.empty(val_samples, dtype=np.uint8)
    val_score = np.empty(val_samples, dtype=np.float32)

    req_pos = 0
    train_pos = 0
    val_pos = 0
    train_phase_counts = np.zeros(N_PHASES, dtype=np.uint64)
    val_phase_counts = np.zeros(N_PHASES, dtype=np.uint64)
    files_read = 0

    for entry in entries:
        lo = entry.begin
        hi = entry.begin + entry.records
        start = req_pos
        while req_pos < len(chosen) and chosen[req_pos] < hi:
            req_pos += 1
        if req_pos == start:
            continue

        local_indices = chosen[start:req_pos] - lo
        raw = np.fromfile(entry.path, dtype=RECORD_DTYPE)
        selected = raw[local_indices]
        phases = (popcount_u64(selected["player"] | selected["opponent"]) - 4).astype(np.int16)
        valid = (phases >= 0) & (phases < N_PHASES)
        if not np.all(valid):
            selected = selected[valid]
            phases = phases[valid]
            local_validation = validation[start:req_pos][valid]
        else:
            local_validation = validation[start:req_pos]

        is_val = local_validation
        is_train = ~is_val
        n_train = int(is_train.sum())
        n_val = int(is_val.sum())
        if n_train:
            sl = slice(train_pos, train_pos + n_train)
            train_player[sl] = selected["player"][is_train]
            train_opponent[sl] = selected["opponent"][is_train]
            train_phase[sl] = phases[is_train].astype(np.uint8)
            train_score[sl] = selected["score"][is_train].astype(np.float32)
            train_phase_counts += np.bincount(train_phase[sl], minlength=N_PHASES).astype(np.uint64)
            train_pos += n_train
        if n_val:
            sl = slice(val_pos, val_pos + n_val)
            val_player[sl] = selected["player"][is_val]
            val_opponent[sl] = selected["opponent"][is_val]
            val_phase[sl] = phases[is_val].astype(np.uint8)
            val_score[sl] = selected["score"][is_val].astype(np.float32)
            val_phase_counts += np.bincount(val_phase[sl], minlength=N_PHASES).astype(np.uint64)
            val_pos += n_val
        files_read += 1
        if progress_interval_files > 0 and files_read % progress_interval_files == 0:
            print(
                f"sample_loading files_read {files_read} train {train_pos}/{train_samples} "
                f"val {val_pos}/{val_samples}",
                flush=True,
            )

    if train_pos != train_samples or val_pos != val_samples:
        raise RuntimeError(f"loaded train {train_pos}/{train_samples} val {val_pos}/{val_samples}")

    return {
        "train_player": train_player,
        "train_opponent": train_opponent,
        "train_phase": train_phase,
        "train_score": train_score,
        "val_player": val_player,
        "val_opponent": val_opponent,
        "val_phase": val_phase,
        "val_score": val_score,
        "train_phase_counts": train_phase_counts,
        "val_phase_counts": val_phase_counts,
    }


def sample_cache_meta(args: argparse.Namespace, entries: list[FileEntry]) -> dict[str, object]:
    return {
        "data_root": str(Path(args.data_root).resolve()),
        "record_start": args.record_start,
        "record_end": args.record_end,
        "available_records": int(entries[-1].begin + entries[-1].records) if entries else 0,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "seed": args.seed,
    }


def save_sample_cache(path: Path, samples: dict[str, np.ndarray], args: argparse.Namespace, entries: list[FileEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_player=samples["train_player"],
        train_opponent=samples["train_opponent"],
        train_phase=samples["train_phase"],
        train_score=samples["train_score"],
        val_player=samples["val_player"],
        val_opponent=samples["val_opponent"],
        val_phase=samples["val_phase"],
        val_score=samples["val_score"],
        train_phase_counts=samples["train_phase_counts"],
        val_phase_counts=samples["val_phase_counts"],
    )
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(sample_cache_meta(args, entries), indent=2),
        encoding="utf-8",
    )


def load_sample_cache(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {
        "train_player": loaded["train_player"],
        "train_opponent": loaded["train_opponent"],
        "train_phase": loaded["train_phase"],
        "train_score": loaded["train_score"],
        "val_player": loaded["val_player"],
        "val_opponent": loaded["val_opponent"],
        "val_phase": loaded["val_phase"],
        "val_score": loaded["val_score"],
        "train_phase_counts": loaded["train_phase_counts"],
        "val_phase_counts": loaded["val_phase_counts"],
    }


def batch_iter(n: int, batch_size: int, shuffle: bool, seed: int) -> Iterable[np.ndarray]:
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
    else:
        order = np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def make_batch(samples: dict[str, np.ndarray], prefix: str, idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    player = samples[f"{prefix}_player"][idx]
    opponent = samples[f"{prefix}_opponent"][idx]
    features_np = make_feature_matrix(player, opponent)
    phase_np = samples[f"{prefix}_phase"][idx].astype(np.int64, copy=False)
    score_np = samples[f"{prefix}_score"][idx]
    return (
        torch.from_numpy(features_np).to(device=device, non_blocking=True),
        torch.from_numpy(phase_np).to(device=device, non_blocking=True),
        torch.from_numpy(score_np).to(device=device, non_blocking=True),
    )


@torch.no_grad()
def evaluate_loss(model: OthelloNNUE, samples: dict[str, np.ndarray], prefix: str, batch_size: int, device: torch.device, limit: int) -> tuple[float, float, int]:
    model.eval()
    n = len(samples[f"{prefix}_score"])
    if limit > 0:
        n = min(n, limit)
    se = 0.0
    ae = 0.0
    seen = 0
    for idx in batch_iter(n, batch_size, False, 0):
        features, phases, target = make_batch(samples, prefix, idx, device)
        pred = model(features, phases)
        err = pred - target
        se += float((err * err).sum().detach().cpu())
        ae += float(err.abs().sum().detach().cpu())
        seen += int(target.numel())
    return se / max(1, seen), ae / max(1, seen), seen


def pad_int8_matrix(matrix: np.ndarray, padded_cols: int) -> np.ndarray:
    out = np.zeros((matrix.shape[0], padded_cols), dtype=np.int8)
    out[:, :matrix.shape[1]] = matrix
    return out


def quantize_to_int16(values: torch.Tensor, scale: float, clip: int = 32767) -> np.ndarray:
    arr = values.detach().cpu().numpy()
    return np.clip(np.rint(arr * scale), -clip, clip).astype("<i2")


def quantize_to_int8(values: torch.Tensor, scale: float, clip: int = 127) -> np.ndarray:
    arr = values.detach().cpu().numpy()
    return np.clip(np.rint(arr * scale), -clip, clip).astype("i1")


def quantize_model(model: OthelloNNUE) -> QuantizedNNUE:
    ft_dim = model.ft_dim
    hidden1 = model.hidden1_dim
    hidden2 = model.hidden2_dim
    post_padded = math.ceil((ft_dim * 2) / 32) * 32
    hidden1_padded = math.ceil(hidden1 / 32) * 32
    hidden2_padded = math.ceil(hidden2 / 32) * 32

    ft_bias = quantize_to_int16(model.ft_bias, FT_SCALE)
    ft_weight = quantize_to_int16(model.ft_weight, FT_SCALE)
    h1_bias = np.rint(model.hidden1.bias.detach().cpu().numpy() * LAYER_ACC_SCALE).astype("<i4")
    h1_weight = pad_int8_matrix(quantize_to_int8(model.hidden1.weight, LAYER_WEIGHT_SCALE), post_padded)
    h2_bias = np.rint(model.hidden2.bias.detach().cpu().numpy() * LAYER_ACC_SCALE).astype("<i4")
    h2_weight = pad_int8_matrix(quantize_to_int8(model.hidden2.weight, LAYER_WEIGHT_SCALE), hidden1_padded)
    out_bias = np.rint(model.output_bias.detach().cpu().numpy() * LAYER_ACC_SCALE).astype("<i4")
    out_weight = pad_int8_matrix(quantize_to_int8(model.output_weight, LAYER_WEIGHT_SCALE), hidden2_padded)
    return QuantizedNNUE(
        ft_bias=ft_bias,
        ft_weight=ft_weight,
        hidden1_bias=h1_bias,
        hidden1_weight=h1_weight,
        hidden2_bias=h2_bias,
        hidden2_weight=h2_weight,
        output_bias=out_bias,
        output_weight=out_weight,
        post_padded=post_padded,
        hidden1_padded=hidden1_padded,
        hidden2_padded=hidden2_padded,
    )


def export_model(model: OthelloNNUE, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    q = quantize_model(model)
    ft_dim = model.ft_dim
    hidden1 = model.hidden1_dim
    hidden2 = model.hidden2_dim
    with out_file.open("wb") as f:
        f.write(b"EGNNUE1\0")
        header = np.array([
            1,
            INPUT_FEATURES,
            ft_dim,
            hidden1,
            hidden2,
            N_PHASES,
            FT_SHIFT,
            HIDDEN_SHIFT,
            HIDDEN_SHIFT,
            OUTPUT_SHIFT,
        ], dtype="<u4")
        header.tofile(f)
        np.zeros(8, dtype="<u4").tofile(f)
        q.ft_bias.tofile(f)
        q.ft_weight.tofile(f)
        q.hidden1_bias.tofile(f)
        q.hidden1_weight.tofile(f)
        q.hidden2_bias.tofile(f)
        q.hidden2_weight.tofile(f)
        q.output_bias.tofile(f)
        q.output_weight.tofile(f)


def clamp_u8_shifted(values: np.ndarray, shift: int) -> np.ndarray:
    shifted = values >> shift if shift > 0 else values
    return np.clip(shifted, 0, 127).astype(np.uint8)


def rounded_shift_signed(values: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return values
    rounded = np.empty_like(values)
    non_negative = values >= 0
    rounded[non_negative] = (values[non_negative] + (1 << (shift - 1))) >> shift
    rounded[~non_negative] = -(((-values[~non_negative]) + (1 << (shift - 1))) >> shift)
    return rounded


def quantized_forward(q: QuantizedNNUE, player: np.ndarray, opponent: np.ndarray, phases: np.ndarray) -> np.ndarray:
    features = make_feature_matrix_int(player, opponent).astype(np.int32, copy=False)
    ft_weight = q.ft_weight.astype(np.int32, copy=False)
    stm = q.ft_bias.astype(np.int32, copy=False)[None, :] + features @ ft_weight
    swapped = np.concatenate([features[:, 64:], features[:, :64]], axis=1)
    non_stm = q.ft_bias.astype(np.int32, copy=False)[None, :] + swapped @ ft_weight

    post_input = np.zeros((features.shape[0], q.post_padded), dtype=np.uint8)
    post_input[:, :q.ft_bias.shape[0]] = clamp_u8_shifted(stm, FT_SHIFT)
    post_input[:, q.ft_bias.shape[0]:q.ft_bias.shape[0] * 2] = clamp_u8_shifted(non_stm, FT_SHIFT)

    h1_raw = q.hidden1_bias.astype(np.int32, copy=False)[None, :] + (
        post_input.astype(np.int32, copy=False) @ q.hidden1_weight.astype(np.int32, copy=False).T
    )
    hidden1 = np.zeros((features.shape[0], q.hidden1_padded), dtype=np.uint8)
    hidden1[:, :q.hidden1_bias.shape[0]] = clamp_u8_shifted(h1_raw, HIDDEN_SHIFT)

    h2_raw = q.hidden2_bias.astype(np.int32, copy=False)[None, :] + (
        hidden1.astype(np.int32, copy=False) @ q.hidden2_weight.astype(np.int32, copy=False).T
    )
    hidden2 = np.zeros((features.shape[0], q.hidden2_padded), dtype=np.uint8)
    hidden2[:, :q.hidden2_bias.shape[0]] = clamp_u8_shifted(h2_raw, HIDDEN_SHIFT)

    output_weights = q.output_weight[phases.astype(np.int64, copy=False)].astype(np.int32, copy=False)
    raw = q.output_bias[phases.astype(np.int64, copy=False)].astype(np.int32, copy=False)
    raw = raw + (hidden2.astype(np.int32, copy=False) * output_weights).sum(axis=1)
    raw = rounded_shift_signed(raw, OUTPUT_SHIFT)
    raw = raw + np.where(raw >= 0, STEP // 2, -(STEP // 2))
    values = np.trunc(raw.astype(np.float64) / float(STEP)).astype(np.int32)
    return np.clip(values, -64, 64).astype(np.float32)


@torch.no_grad()
def evaluate_quantized_loss(model: OthelloNNUE, samples: dict[str, np.ndarray], prefix: str, batch_size: int, limit: int) -> tuple[float, float, int]:
    q = quantize_model(model)
    n = len(samples[f"{prefix}_score"])
    if limit > 0:
        n = min(n, limit)
    se = 0.0
    ae = 0.0
    seen = 0
    for idx in batch_iter(n, batch_size, False, 0):
        pred = quantized_forward(
            q,
            samples[f"{prefix}_player"][idx],
            samples[f"{prefix}_opponent"][idx],
            samples[f"{prefix}_phase"][idx],
        )
        target = samples[f"{prefix}_score"][idx]
        err = pred - target
        se += float(np.square(err).sum())
        ae += float(np.abs(err).sum())
        seen += int(target.size)
    return se / max(1, seen), ae / max(1, seen), seen


def write_summary(out_dir: Path, args: argparse.Namespace, entries: list[FileEntry], samples: dict[str, np.ndarray] | None, best: dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "learning_method": "nnue_only_board_data_pytorch_mse",
        "data_root": str(Path(args.data_root).resolve()),
        "record_start": args.record_start,
        "record_end": args.record_end,
        "available_records": int(entries[-1].begin + entries[-1].records) if entries else 0,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "arch": args.arch,
        "ft_dim": args.ft_dim,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best": best,
    }
    if samples is not None:
        summary["train_phase_counts"] = [int(x) for x in samples["train_phase_counts"]]
        summary["val_phase_counts"] = [int(x) for x in samples["val_phase_counts"]]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("EGAROUCID_BOARD_DATA", "E:/egaroucid_data/train_data/board_data"))
    parser.add_argument("--record-start", type=int, default=223)
    parser.add_argument("--record-end", type=int, default=-1)
    parser.add_argument("--arch", choices=sorted(ARCHES), default="medium")
    parser.add_argument("--ft-dim", type=int, default=0)
    parser.add_argument("--hidden1", type=int, default=0)
    parser.add_argument("--hidden2", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=10_000_000)
    parser.add_argument("--val-samples", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--metric-limit", type=int, default=1_000_000)
    parser.add_argument("--quantized-metric-limit", type=int, default=1_000_000)
    parser.add_argument("--progress-interval-files", type=int, default=20)
    parser.add_argument("--sample-cache", default="")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arch_dims = ARCHES[args.arch]
    args.ft_dim = args.ft_dim or arch_dims[0]
    args.hidden1 = args.hidden1 or arch_dims[1]
    args.hidden2 = args.hidden2 or arch_dims[2]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)

    model_name = args.model_name or f"nnue_only_{args.arch}_records{args.record_start}plus_train{args.train_samples}_val{args.val_samples}_e{args.epochs}"
    out_dir = next_model_dir(Path(args.model_root), "20260725", model_name)
    out_file = Path(args.out_file) if args.out_file else out_dir / f"eval_nnue_{args.arch}.egevnnue"

    data_root = Path(args.data_root)
    entries = build_manifest(data_root, args.record_start, args.record_end)
    total_records = entries[-1].begin + entries[-1].records if entries else 0
    print(f"manifest_files {len(entries)} total_records {total_records} data_root {data_root}", flush=True)
    print(f"arch {args.arch} ft_dim {args.ft_dim} hidden1 {args.hidden1} hidden2 {args.hidden2}", flush=True)
    if args.dry_run:
        write_summary(out_dir, args, entries, None, {})
        return 0

    model = OthelloNNUE(args.ft_dim, args.hidden1, args.hidden2)
    best = {"epoch": 0, "val_mse": float("inf"), "val_mae": float("inf")}
    if args.init_only or args.epochs == 0:
        export_model(model, out_file)
        write_summary(out_dir, args, entries, None, best)
        print(f"wrote {out_file}", flush=True)
        return 0

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.sample_cache and Path(args.sample_cache).exists():
        print(f"loading_sample_cache {args.sample_cache}", flush=True)
        samples = load_sample_cache(Path(args.sample_cache))
    else:
        samples = load_samples(entries, args.train_samples, args.val_samples, args.seed, args.progress_interval_files)
        if args.sample_cache:
            print(f"writing_sample_cache {args.sample_cache}", flush=True)
            save_sample_cache(Path(args.sample_cache), samples, args, entries)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_mse, train_mae, train_n = evaluate_loss(model, samples, "train", args.batch_size, device, args.metric_limit)
    val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.metric_limit)
    print(f"initial train_mse {train_mse:.6f} train_mae {train_mae:.6f} train_metric_samples {train_n} "
          f"val_mse {val_mse:.6f} val_mae {val_mae:.6f} val_metric_samples {val_n}", flush=True)

    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = 0
        running_loss = 0.0
        for idx in batch_iter(args.train_samples, args.batch_size, True, args.seed + epoch):
            features, phases, target = make_batch(samples, "train", idx, device)
            pred = model(features, phases)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += int(target.numel())
            running_loss += float(loss.detach().cpu()) * int(target.numel())
        train_mse, train_mae, train_n = evaluate_loss(model, samples, "train", args.batch_size, device, args.metric_limit)
        val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.metric_limit)
        if val_mse < best["val_mse"]:
            best = {"epoch": epoch, "val_mse": val_mse, "val_mae": val_mae}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            export_model(model, out_file)
        print(
            f"epoch {epoch} train_epoch_mse {running_loss / max(1, seen):.6f} "
            f"train_mse {train_mse:.6f} train_mae {train_mae:.6f} "
            f"val_mse {val_mse:.6f} val_mae {val_mae:.6f} "
            f"best_epoch {best['epoch']} best_val_mse {best['val_mse']:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        export_model(model, out_file)
        q_val_mse, q_val_mae, q_val_n = evaluate_quantized_loss(
            model,
            samples,
            "val",
            args.batch_size,
            args.quantized_metric_limit,
        )
        best["quantized_val_mse"] = q_val_mse
        best["quantized_val_mae"] = q_val_mae
        best["quantized_val_metric_samples"] = q_val_n
        print(
            f"quantized_best val_mse {q_val_mse:.6f} val_mae {q_val_mae:.6f} "
            f"val_metric_samples {q_val_n}",
            flush=True,
        )

    write_summary(out_dir, args, entries, samples, best)
    print(f"wrote {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
