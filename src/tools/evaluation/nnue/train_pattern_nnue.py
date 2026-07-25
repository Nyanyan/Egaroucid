#!/usr/bin/env python3
"""Train an NNUE-style evaluator from indexed pattern features.

The input data is the phase-wise indexed teacher data used by the FM trainer:

    int16 n_discs
    int16 player
    uint16 features[ADJ_N_FEATURES]
    int16 score

The 65 indexed features are converted to one global categorical feature ID per
feature column.  This script is intentionally separate from train_nnue.py,
because it tests whether using the existing pattern categories solves the poor
loss of the 128 stone-bit input model before wiring the format into search.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


N_PHASES = 60
N_FEATURE_COLUMNS = 65
INDEXED_RECORD_BYTES = 2 + 2 + 2 * N_FEATURE_COLUMNS + 2
STEP = 32
ACTIVATION_SCALE = 16
FT_SCALE = 256
FT_SHIFT = 4
DEFAULT_LAYER_WEIGHT_SCALE = 32

P37 = 2187
P38 = 6561
P39 = 19683
P310 = 59049
MAX_STONE_NUM = 65

ADJ_EVAL_SIZES = np.array(
    [
        P38, P39, P38, P39,
        P38, P39, P37, P310,
        P310, P310, P310, P310,
        P310, P310, P310, P310,
        MAX_STONE_NUM,
    ],
    dtype=np.uint32,
)
ADJ_FEATURE_TO_EVAL_IDX = np.array(
    [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9, 9, 9,
        10, 10, 10, 10,
        11, 11, 11, 11,
        12, 12, 12, 12,
        13, 13, 13, 13,
        14, 14, 14, 14,
        15, 15, 15, 15,
        16,
    ],
    dtype=np.uint32,
)

FEATURE_STARTS = np.zeros(N_FEATURE_COLUMNS, dtype=np.uint32)
_start = 0
for _i in range(N_FEATURE_COLUMNS):
    if _i > 0 and ADJ_FEATURE_TO_EVAL_IDX[_i] > ADJ_FEATURE_TO_EVAL_IDX[_i - 1]:
        _start += int(ADJ_EVAL_SIZES[ADJ_FEATURE_TO_EVAL_IDX[_i - 1]])
    FEATURE_STARTS[_i] = _start
TOTAL_INPUT_FEATURES = int(ADJ_EVAL_SIZES.sum())

INDEXED_DTYPE = np.dtype(
    [
        ("n_discs", "<i2"),
        ("player", "<i2"),
        ("features", "<u2", (N_FEATURE_COLUMNS,)),
        ("score", "<i2"),
    ]
)


def _swap_ternary_value(value: int, n_digits: int) -> int:
    res = 0
    pow3 = 1
    x = value
    for _ in range(n_digits):
        digit = x % 3
        x //= 3
        if digit == 0:
            digit = 1
        elif digit == 1:
            digit = 0
        res += digit * pow3
        pow3 *= 3
    return res


SWAP_VALUE_TABLES: list[np.ndarray] = []
for _feature_idx in range(N_FEATURE_COLUMNS):
    _eval_idx = int(ADJ_FEATURE_TO_EVAL_IDX[_feature_idx])
    _size = int(ADJ_EVAL_SIZES[_eval_idx])
    if _feature_idx == N_FEATURE_COLUMNS - 1:
        SWAP_VALUE_TABLES.append(np.arange(_size, dtype=np.uint32))
    else:
        _digits = 0
        _pow = 1
        while _pow < _size:
            _pow *= 3
            _digits += 1
        SWAP_VALUE_TABLES.append(
            np.array([_swap_ternary_value(v, _digits) for v in range(_size)], dtype=np.uint32)
        )

ARCHES = {
    "pft16": (16, 16, 16),
    "pft32": (32, 32, 32),
    "pft64": (64, 32, 32),
    "pft128": (128, 32, 32),
    "pft128_wide": (128, 64, 32),
    "pft256": (256, 64, 32),
}


@dataclass
class FileEntry:
    path: Path
    phase: int
    record_num: int
    records: int
    begin: int


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_record_num(path: Path) -> int | None:
    if path.suffix.lower() != ".dat":
        return None
    try:
        return int(path.stem)
    except ValueError:
        return None


def build_manifest(data_root: Path, record_start: int, record_end: int) -> tuple[list[FileEntry], np.ndarray]:
    entries: list[FileEntry] = []
    phase_counts = np.zeros(N_PHASES, dtype=np.uint64)
    begin = 0
    for phase in range(N_PHASES):
        phase_dir = data_root / str(phase)
        if not phase_dir.exists():
            continue
        files: list[tuple[int, Path]] = []
        for path in phase_dir.iterdir():
            if not path.is_file():
                continue
            record_num = parse_record_num(path)
            if record_num is None:
                continue
            if record_num < record_start or (record_end >= 0 and record_num > record_end):
                continue
            files.append((record_num, path))
        for record_num, path in sorted(files):
            size = path.stat().st_size
            if size == 0:
                continue
            if size % INDEXED_RECORD_BYTES != 0:
                raise RuntimeError(
                    f"bad indexed data size: {path} bytes={size} record_bytes={INDEXED_RECORD_BYTES}"
                )
            records = size // INDEXED_RECORD_BYTES
            entries.append(FileEntry(path=path, phase=phase, record_num=record_num, records=records, begin=begin))
            phase_counts[phase] += records
            begin += records
    return entries, phase_counts


def next_model_dir(root: Path, date: str, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    used = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parts = child.name.split("_", 2)
        if len(parts) >= 2 and parts[0] == date and parts[1].isdigit():
            used.append(int(parts[1]))
    return root / f"{date}_{max(used, default=0) + 1}_{name}"


def sample_indices(total: int, need: int, train_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if need > total:
        raise ValueError(f"requested samples {need} exceeds available records {total}")
    rng = np.random.default_rng(seed)
    if need == total:
        chosen = np.arange(total, dtype=np.uint64)
    else:
        chosen = np.empty(0, dtype=np.uint64)
        while chosen.size < need:
            draw = min(max(int((need - chosen.size) * 1.35), 1_000_000), 50_000_000)
            block = rng.integers(0, total, size=draw, dtype=np.uint64)
            chosen = np.unique(np.concatenate([chosen, block]))
            if chosen.size > need * 2:
                chosen = rng.choice(chosen, size=need, replace=False)
        if chosen.size > need:
            chosen = rng.choice(chosen, size=need, replace=False)
    validation = np.zeros(need, dtype=np.bool_)
    if need > train_samples:
        validation_positions = rng.choice(need, size=need - train_samples, replace=False)
        validation[validation_positions] = True
    order = np.argsort(chosen, kind="stable")
    return chosen[order], validation[order]


def globalize_features(local_features: np.ndarray) -> np.ndarray:
    return local_features.astype(np.uint32, copy=False) + FEATURE_STARTS[None, :]


def load_selected_from_file(path: Path, local_indices: np.ndarray, full_read_ratio: float) -> np.ndarray:
    records = path.stat().st_size // INDEXED_RECORD_BYTES
    if local_indices.size / max(1, records) >= full_read_ratio:
        raw = np.fromfile(path, dtype=INDEXED_DTYPE)
        return raw[local_indices.astype(np.int64, copy=False)]
    mapped = np.memmap(path, dtype=INDEXED_DTYPE, mode="r")
    try:
        return np.asarray(mapped[local_indices.astype(np.int64, copy=False)])
    finally:
        del mapped


def load_samples(
    entries: list[FileEntry],
    train_samples: int,
    val_samples: int,
    seed: int,
    progress_interval_sec: int,
    full_read_ratio: float,
) -> dict[str, np.ndarray]:
    total = entries[-1].begin + entries[-1].records if entries else 0
    need = train_samples + val_samples
    chosen, validation = sample_indices(total, need, train_samples, seed)

    train_features = np.empty((train_samples, N_FEATURE_COLUMNS), dtype=np.uint32)
    val_features = np.empty((val_samples, N_FEATURE_COLUMNS), dtype=np.uint32)
    train_phase = np.empty(train_samples, dtype=np.uint8)
    val_phase = np.empty(val_samples, dtype=np.uint8)
    train_score = np.empty(train_samples, dtype=np.float32)
    val_score = np.empty(val_samples, dtype=np.float32)
    train_phase_counts = np.zeros(N_PHASES, dtype=np.uint64)
    val_phase_counts = np.zeros(N_PHASES, dtype=np.uint64)

    req_pos = 0
    train_pos = 0
    val_pos = 0
    files_read = 0
    bad_features = 0
    phase_mismatches = 0
    start_ms = now_ms()
    next_log_ms = start_ms + progress_interval_sec * 1000

    for entry in entries:
        lo = entry.begin
        hi = entry.begin + entry.records
        start = req_pos
        while req_pos < chosen.size and chosen[req_pos] < hi:
            req_pos += 1
        if req_pos == start:
            continue

        local_indices = chosen[start:req_pos] - lo
        selected = load_selected_from_file(entry.path, local_indices, full_read_ratio)
        local_validation = validation[start:req_pos]
        local_phase = np.full(selected.shape[0], entry.phase, dtype=np.uint8)
        expected_phase = selected["n_discs"].astype(np.int16, copy=False) - 4
        phase_mismatches += int(np.count_nonzero(expected_phase != entry.phase))
        local_features = selected["features"]
        max_allowed = ADJ_EVAL_SIZES[ADJ_FEATURE_TO_EVAL_IDX]
        bad_mask = np.any(local_features >= max_allowed[None, :], axis=1)
        bad_features += int(np.count_nonzero(bad_mask))
        if np.any(bad_mask):
            keep = ~bad_mask
            local_features = local_features[keep]
            local_phase = local_phase[keep]
            local_validation = local_validation[keep]
            selected = selected[keep]

        global_features = globalize_features(local_features)
        is_val = local_validation
        is_train = ~is_val
        n_train = int(is_train.sum())
        n_val = int(is_val.sum())
        if n_train:
            sl = slice(train_pos, train_pos + n_train)
            train_features[sl] = global_features[is_train]
            train_phase[sl] = local_phase[is_train]
            train_score[sl] = selected["score"][is_train].astype(np.float32)
            train_phase_counts += np.bincount(train_phase[sl], minlength=N_PHASES).astype(np.uint64)
            train_pos += n_train
        if n_val:
            sl = slice(val_pos, val_pos + n_val)
            val_features[sl] = global_features[is_val]
            val_phase[sl] = local_phase[is_val]
            val_score[sl] = selected["score"][is_val].astype(np.float32)
            val_phase_counts += np.bincount(val_phase[sl], minlength=N_PHASES).astype(np.uint64)
            val_pos += n_val

        files_read += 1
        current_ms = now_ms()
        if progress_interval_sec > 0 and current_ms >= next_log_ms:
            print(
                f"sample_loading files_read {files_read} train {train_pos}/{train_samples} "
                f"val {val_pos}/{val_samples} phase {entry.phase} record {entry.record_num} "
                f"elapsed_ms {current_ms - start_ms}",
                flush=True,
            )
            next_log_ms = current_ms + progress_interval_sec * 1000

    if train_pos != train_samples or val_pos != val_samples:
        raise RuntimeError(
            f"loaded train {train_pos}/{train_samples} val {val_pos}/{val_samples} "
            f"bad_features {bad_features}"
        )
    print(
        f"loaded train {train_pos} val {val_pos} bad_features {bad_features} "
        f"phase_mismatches {phase_mismatches} elapsed_ms {now_ms() - start_ms}",
        flush=True,
    )
    return {
        "train_features": train_features,
        "train_phase": train_phase,
        "train_score": train_score,
        "val_features": val_features,
        "val_phase": val_phase,
        "val_score": val_score,
        "train_phase_counts": train_phase_counts,
        "val_phase_counts": val_phase_counts,
    }


def save_sample_cache(path: Path, samples: dict[str, np.ndarray], meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_features=samples["train_features"],
        train_phase=samples["train_phase"],
        train_score=samples["train_score"],
        val_features=samples["val_features"],
        val_phase=samples["val_phase"],
        val_score=samples["val_score"],
        train_phase_counts=samples["train_phase_counts"],
        val_phase_counts=samples["val_phase_counts"],
    )
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_sample_cache(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {
        "train_features": loaded["train_features"],
        "train_phase": loaded["train_phase"],
        "train_score": loaded["train_score"],
        "val_features": loaded["val_features"],
        "val_phase": loaded["val_phase"],
        "val_score": loaded["val_score"],
        "train_phase_counts": loaded["train_phase_counts"],
        "val_phase_counts": loaded["val_phase_counts"],
    }


def make_active_feature_columns(active_eval_types: int) -> np.ndarray:
    if active_eval_types <= 0 or active_eval_types > int(ADJ_EVAL_SIZES.size - 1):
        raise ValueError(f"active_eval_types must be in [1, {int(ADJ_EVAL_SIZES.size - 1)}]")
    columns = [i for i in range(N_FEATURE_COLUMNS - 1) if int(ADJ_FEATURE_TO_EVAL_IDX[i]) < active_eval_types]
    columns.append(N_FEATURE_COLUMNS - 1)
    return np.array(columns, dtype=np.int64)


def feature_columns_to_mask(columns: np.ndarray) -> int:
    mask = 0
    for col in columns.tolist():
        mask |= 1 << int(col)
    return mask


class PatternNNUE(nn.Module):
    def __init__(self, ft_dim: int, hidden1: int, hidden2: int, perspectives: int, active_columns: np.ndarray):
        super().__init__()
        self.ft_dim = ft_dim
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.perspectives = perspectives
        self.active_columns = tuple(int(x) for x in active_columns.tolist())
        self.ft_weight = nn.Embedding(TOTAL_INPUT_FEATURES, ft_dim, sparse=True)
        self.ft_bias = nn.Parameter(torch.zeros(ft_dim))
        self.hidden1 = nn.Linear(ft_dim * perspectives, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.output_weight = nn.Parameter(torch.empty(N_PHASES, hidden2))
        self.output_bias = nn.Parameter(torch.zeros(N_PHASES))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ft_weight.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.hidden1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.normal_(self.hidden2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.normal_(self.output_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_bias)

    @staticmethod
    def clipped_relu(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 127.0 / 16.0)

    def forward(
        self,
        features: torch.Tensor,
        phases: torch.Tensor,
        opponent_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.ft_bias + self.ft_weight(features[:, self.active_columns]).sum(dim=1)
        if self.perspectives == 2:
            assert opponent_features is not None
            y = self.ft_bias + self.ft_weight(opponent_features[:, self.active_columns]).sum(dim=1)
            x = torch.cat([self.clipped_relu(x), self.clipped_relu(y)], dim=1)
        else:
            x = self.clipped_relu(x)
        x = self.clipped_relu(self.hidden1(x))
        x = self.clipped_relu(self.hidden2(x))
        return (x * self.output_weight[phases]).sum(dim=1) + self.output_bias[phases]


@dataclass
class QuantizedPatternNNUE:
    ft_bias: np.ndarray
    ft_weight: np.ndarray
    ft_weight_bits: int
    hidden1_bias: np.ndarray
    hidden1_weight: np.ndarray
    hidden2_bias: np.ndarray
    hidden2_weight: np.ndarray
    output_bias: np.ndarray
    output_weight: np.ndarray
    post_padded: int
    hidden1_padded: int
    hidden2_padded: int
    hidden_shift: int
    output_shift: int


def batch_iter(n: int, batch_size: int, shuffle: bool, seed: int) -> Iterable[np.ndarray]:
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
    else:
        order = np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def make_batch(
    samples: dict[str, np.ndarray],
    prefix: str,
    idx: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    features_np = samples[f"{prefix}_features"][idx].astype(np.int64, copy=False)
    phase_np = samples[f"{prefix}_phase"][idx].astype(np.int64, copy=False)
    score_np = samples[f"{prefix}_score"][idx]
    opponent_np = None
    if samples.get("perspectives", 1) == 2:
        opponent_np = make_opponent_features(features_np.astype(np.uint32, copy=False), phase_np.astype(np.uint8, copy=False)).astype(np.int64, copy=False)
    return (
        torch.from_numpy(features_np).to(device=device, non_blocking=True),
        None if opponent_np is None else torch.from_numpy(opponent_np).to(device=device, non_blocking=True),
        torch.from_numpy(phase_np).to(device=device, non_blocking=True),
        torch.from_numpy(score_np).to(device=device, non_blocking=True),
    )


def make_opponent_features(features: np.ndarray, phases: np.ndarray) -> np.ndarray:
    local = features - FEATURE_STARTS[None, :]
    out = np.empty_like(features, dtype=np.uint32)
    for col in range(N_FEATURE_COLUMNS - 1):
        out[:, col] = FEATURE_STARTS[col] + SWAP_VALUE_TABLES[col][local[:, col]]
    n_discs = phases.astype(np.uint32, copy=False) + 4
    out[:, N_FEATURE_COLUMNS - 1] = (
        FEATURE_STARTS[N_FEATURE_COLUMNS - 1] +
        n_discs -
        local[:, N_FEATURE_COLUMNS - 1]
    )
    return out


def pad_int8_matrix(matrix: np.ndarray, padded_cols: int) -> np.ndarray:
    out = np.zeros((matrix.shape[0], padded_cols), dtype=np.int8)
    out[:, : matrix.shape[1]] = matrix
    return out


def quantize_to_int16(values: torch.Tensor, scale: float, clip: int = 32767) -> np.ndarray:
    arr = values.detach().cpu().numpy()
    return np.clip(np.rint(arr * scale), -clip, clip).astype("<i2")


def quantize_to_int8(values: torch.Tensor, scale: float, clip: int = 127) -> np.ndarray:
    arr = values.detach().cpu().numpy()
    return np.clip(np.rint(arr * scale), -clip, clip).astype("i1")


def int_log2_power_of_two(value: int, name: str) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two: {value}")
    return value.bit_length() - 1


def make_quantization_shifts(layer_weight_scale: int) -> tuple[int, int]:
    hidden_shift = int_log2_power_of_two(layer_weight_scale, "layer_weight_scale")
    output_scale_divisor = ACTIVATION_SCALE * layer_weight_scale
    if output_scale_divisor % STEP != 0:
        raise ValueError(
            f"ACTIVATION_SCALE * layer_weight_scale must be divisible by STEP: "
            f"{ACTIVATION_SCALE} * {layer_weight_scale}"
        )
    output_shift = int_log2_power_of_two(output_scale_divisor // STEP, "output_scale_divisor / STEP")
    return hidden_shift, output_shift


def quantize_model(model: PatternNNUE, layer_weight_scale: int, ft_weight_bits: int = 16) -> QuantizedPatternNNUE:
    ft_dim = model.ft_dim
    hidden1 = model.hidden1_dim
    hidden2 = model.hidden2_dim
    post_padded = math.ceil((ft_dim * model.perspectives) / 32) * 32
    hidden1_padded = math.ceil(hidden1 / 32) * 32
    hidden2_padded = math.ceil(hidden2 / 32) * 32
    hidden_shift, output_shift = make_quantization_shifts(layer_weight_scale)
    layer_acc_scale = ACTIVATION_SCALE * layer_weight_scale

    ft_bias = quantize_to_int16(model.ft_bias, FT_SCALE)
    if ft_weight_bits == 8:
        ft_weight = quantize_to_int8(model.ft_weight.weight, FT_SCALE)
    elif ft_weight_bits == 16:
        ft_weight = quantize_to_int16(model.ft_weight.weight, FT_SCALE)
    else:
        raise ValueError(f"ft_weight_bits must be 8 or 16: {ft_weight_bits}")
    h1_bias = np.rint(model.hidden1.bias.detach().cpu().numpy() * layer_acc_scale).astype("<i4")
    h1_weight = pad_int8_matrix(quantize_to_int8(model.hidden1.weight, layer_weight_scale), post_padded)
    h2_bias = np.rint(model.hidden2.bias.detach().cpu().numpy() * layer_acc_scale).astype("<i4")
    h2_weight = pad_int8_matrix(quantize_to_int8(model.hidden2.weight, layer_weight_scale), hidden1_padded)
    out_bias = np.rint(model.output_bias.detach().cpu().numpy() * layer_acc_scale).astype("<i4")
    out_weight = pad_int8_matrix(quantize_to_int8(model.output_weight, layer_weight_scale), hidden2_padded)
    return QuantizedPatternNNUE(
        ft_bias=ft_bias,
        ft_weight=ft_weight,
        ft_weight_bits=ft_weight_bits,
        hidden1_bias=h1_bias,
        hidden1_weight=h1_weight,
        hidden2_bias=h2_bias,
        hidden2_weight=h2_weight,
        output_bias=out_bias,
        output_weight=out_weight,
        post_padded=post_padded,
        hidden1_padded=hidden1_padded,
        hidden2_padded=hidden2_padded,
        hidden_shift=hidden_shift,
        output_shift=output_shift,
    )


def export_model(model: PatternNNUE, out_file: Path, layer_weight_scale: int, ft_weight_bits: int) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    q = quantize_model(model, layer_weight_scale, ft_weight_bits)
    with out_file.open("wb") as f:
        f.write(b"EGNNUE1\0")
        version = 3 if model.perspectives == 2 else 2
        input_kind = 3 if model.perspectives == 2 else 2
        header = np.array(
            [
                version,
                TOTAL_INPUT_FEATURES,
                model.ft_dim,
                model.hidden1_dim,
                model.hidden2_dim,
                N_PHASES,
                FT_SHIFT,
                q.hidden_shift,
                q.hidden_shift,
                q.output_shift,
            ],
            dtype="<u4",
        )
        header.tofile(f)
        reserved = np.zeros(8, dtype="<u4")
        reserved[0] = input_kind
        reserved[1] = N_FEATURE_COLUMNS
        reserved[2] = model.perspectives
        active_mask = feature_columns_to_mask(np.array(model.active_columns, dtype=np.int64))
        reserved[3] = active_mask & 0xFFFFFFFF
        reserved[4] = (active_mask >> 32) & 0xFFFFFFFF
        reserved[5] = (active_mask >> 64) & 0xFFFFFFFF
        reserved[6] = q.ft_weight_bits
        reserved.tofile(f)
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


def quantized_forward(
    q: QuantizedPatternNNUE,
    features: np.ndarray,
    phases: np.ndarray,
    perspectives: int,
    active_columns: np.ndarray,
) -> np.ndarray:
    ft = q.ft_weight.astype(np.int32, copy=False)
    stm = q.ft_bias.astype(np.int32, copy=False)[None, :] + ft[features[:, active_columns]].sum(axis=1)
    post_input = np.zeros((features.shape[0], q.post_padded), dtype=np.uint8)
    post_input[:, : q.ft_bias.shape[0]] = clamp_u8_shifted(stm, FT_SHIFT)
    if perspectives == 2:
        opponent_features = make_opponent_features(features, phases)
        non_stm = q.ft_bias.astype(np.int32, copy=False)[None, :] + ft[opponent_features[:, active_columns]].sum(axis=1)
        post_input[:, q.ft_bias.shape[0] : q.ft_bias.shape[0] * 2] = clamp_u8_shifted(non_stm, FT_SHIFT)
    h1_raw = q.hidden1_bias.astype(np.int32, copy=False)[None, :] + (
        post_input.astype(np.int32, copy=False) @ q.hidden1_weight.astype(np.int32, copy=False).T
    )
    hidden1 = np.zeros((features.shape[0], q.hidden1_padded), dtype=np.uint8)
    hidden1[:, : q.hidden1_bias.shape[0]] = clamp_u8_shifted(h1_raw, q.hidden_shift)
    h2_raw = q.hidden2_bias.astype(np.int32, copy=False)[None, :] + (
        hidden1.astype(np.int32, copy=False) @ q.hidden2_weight.astype(np.int32, copy=False).T
    )
    hidden2 = np.zeros((features.shape[0], q.hidden2_padded), dtype=np.uint8)
    hidden2[:, : q.hidden2_bias.shape[0]] = clamp_u8_shifted(h2_raw, q.hidden_shift)
    phase_idx = phases.astype(np.int64, copy=False)
    output_weights = q.output_weight[phase_idx].astype(np.int32, copy=False)
    raw = q.output_bias[phase_idx].astype(np.int32, copy=False)
    raw = raw + (hidden2.astype(np.int32, copy=False) * output_weights).sum(axis=1)
    raw = rounded_shift_signed(raw, q.output_shift)
    raw = raw + np.where(raw >= 0, STEP // 2, -(STEP // 2))
    values = np.trunc(raw.astype(np.float64) / float(STEP)).astype(np.int32)
    return np.clip(values, -64, 64).astype(np.float32)


@torch.no_grad()
def evaluate_loss(
    model: PatternNNUE,
    samples: dict[str, np.ndarray],
    prefix: str,
    batch_size: int,
    device: torch.device,
    limit: int,
) -> tuple[float, float, int]:
    model.eval()
    n = len(samples[f"{prefix}_score"])
    if limit > 0:
        n = min(n, limit)
    se = 0.0
    ae = 0.0
    seen = 0
    for idx in batch_iter(n, batch_size, False, 0):
        features, opponent_features, phases, target = make_batch(samples, prefix, idx, device)
        pred = model(features, phases, opponent_features)
        err = pred - target
        se += float((err * err).sum().detach().cpu())
        ae += float(err.abs().sum().detach().cpu())
        seen += int(target.numel())
    return se / max(1, seen), ae / max(1, seen), seen


@torch.no_grad()
def evaluate_quantized_loss(
    model: PatternNNUE,
    samples: dict[str, np.ndarray],
    prefix: str,
    batch_size: int,
    limit: int,
    layer_weight_scale: int,
    ft_weight_bits: int,
) -> tuple[float, float, int]:
    q = quantize_model(model, layer_weight_scale, ft_weight_bits)
    active_columns = np.array(model.active_columns, dtype=np.int64)
    n = len(samples[f"{prefix}_score"])
    if limit > 0:
        n = min(n, limit)
    se = 0.0
    ae = 0.0
    seen = 0
    for idx in batch_iter(n, batch_size, False, 0):
        pred = quantized_forward(
            q,
            samples[f"{prefix}_features"][idx],
            samples[f"{prefix}_phase"][idx],
            model.perspectives,
            active_columns,
        )
        target = samples[f"{prefix}_score"][idx]
        err = pred - target
        se += float(np.square(err).sum())
        ae += float(np.abs(err).sum())
        seen += int(target.size)
    return se / max(1, seen), ae / max(1, seen), seen


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    entries: list[FileEntry],
    manifest_phase_counts: np.ndarray,
    samples: dict[str, np.ndarray] | None,
    best: dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_records = entries[-1].begin + entries[-1].records if entries else 0
    summary: dict[str, object] = {
        "learning_method": "nnue_pattern_input_pytorch_mse",
        "data_root": str(Path(args.data_root).resolve()),
        "record_start": args.record_start,
        "record_end": args.record_end,
        "available_records": int(total_records),
        "available_phase_counts": [int(x) for x in manifest_phase_counts],
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "used_ratio": (args.train_samples + args.val_samples) / max(1, total_records),
        "arch": args.arch,
        "ft_dim": args.ft_dim,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "perspectives": args.perspectives,
        "active_eval_types": args.active_eval_types,
        "active_feature_columns": [int(x) for x in args.active_columns],
        "active_feature_column_count": int(len(args.active_columns)),
        "input_feature_columns": N_FEATURE_COLUMNS,
        "total_input_features": TOTAL_INPUT_FEATURES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dense_weight_decay": args.dense_weight_decay,
        "layer_weight_scale": args.layer_weight_scale,
        "ft_weight_bits": args.ft_weight_bits,
        "seed": args.seed,
        "best": best,
    }
    if samples is not None:
        summary["train_phase_counts"] = [int(x) for x in samples["train_phase_counts"]]
        summary["val_phase_counts"] = [int(x) for x in samples["val_phase_counts"]]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("EGAROUCID_INDEXED_DATA", "E:/egaroucid_data/train_data/bin_data/20241125_1"))
    parser.add_argument("--record-start", type=int, default=223)
    parser.add_argument("--record-end", type=int, default=-1)
    parser.add_argument("--arch", choices=sorted(ARCHES), default="pft64")
    parser.add_argument("--ft-dim", type=int, default=0)
    parser.add_argument("--hidden1", type=int, default=0)
    parser.add_argument("--hidden2", type=int, default=0)
    parser.add_argument("--perspectives", choices=["single", "dual"], default="dual")
    parser.add_argument("--active-eval-types", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--val-samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--dense-weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--layer-weight-scale", type=int, default=DEFAULT_LAYER_WEIGHT_SCALE)
    parser.add_argument("--ft-weight-bits", type=int, choices=[8, 16], default=16)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--metric-limit", type=int, default=1_000_000)
    parser.add_argument("--quantized-metric-limit", type=int, default=1_000_000)
    parser.add_argument("--progress-interval-sec", type=int, default=30)
    parser.add_argument("--full-read-ratio", type=float, default=0.10)
    parser.add_argument("--sample-cache", default="")
    parser.add_argument("--load-state", default="")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arch_dims = ARCHES[args.arch]
    args.ft_dim = args.ft_dim or arch_dims[0]
    args.hidden1 = args.hidden1 or arch_dims[1]
    args.hidden2 = args.hidden2 or arch_dims[2]
    args.active_columns = make_active_feature_columns(args.active_eval_types)
    make_quantization_shifts(args.layer_weight_scale)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)

    date = datetime.now().strftime("%Y%m%d")
    model_name = args.model_name or (
        f"nnue_pattern_{args.perspectives}_{args.arch}_records{args.record_start}plus_"
        f"train{args.train_samples}_val{args.val_samples}_e{args.epochs}"
    )
    out_dir = next_model_dir(Path(args.model_root), date, model_name)
    out_file = Path(args.out_file) if args.out_file else out_dir / f"eval_nnue_pattern_{args.arch}.egevnnue"

    data_root = Path(args.data_root)
    entries, manifest_phase_counts = build_manifest(data_root, args.record_start, args.record_end)
    total_records = entries[-1].begin + entries[-1].records if entries else 0
    print(
        f"manifest_files {len(entries)} total_records {total_records} data_root {data_root} "
        f"input_feature_columns {N_FEATURE_COLUMNS} total_input_features {TOTAL_INPUT_FEATURES}",
        flush=True,
    )
    print(
        f"arch {args.arch} ft_dim {args.ft_dim} hidden1 {args.hidden1} hidden2 {args.hidden2} "
        f"ft_weight_bits {args.ft_weight_bits}",
        flush=True,
    )
    print(
        f"active_eval_types {args.active_eval_types} "
        f"active_feature_column_count {len(args.active_columns)} "
        f"active_feature_columns {','.join(str(int(x)) for x in args.active_columns)}",
        flush=True,
    )
    if args.dry_run:
        write_summary(out_dir, args, entries, manifest_phase_counts, None, {})
        return 0

    if args.sample_cache and Path(args.sample_cache).exists():
        print(f"loading_sample_cache {args.sample_cache}", flush=True)
        samples = load_sample_cache(Path(args.sample_cache))
    else:
        samples = load_samples(
            entries,
            args.train_samples,
            args.val_samples,
            args.seed,
            args.progress_interval_sec,
            args.full_read_ratio,
        )
        if args.sample_cache:
            total_records = entries[-1].begin + entries[-1].records if entries else 0
            meta = {
                "data_root": str(data_root.resolve()),
                "record_start": args.record_start,
                "record_end": args.record_end,
                "available_records": int(total_records),
                "train_samples": args.train_samples,
                "val_samples": args.val_samples,
                "seed": args.seed,
                "input_feature_columns": N_FEATURE_COLUMNS,
                "total_input_features": TOTAL_INPUT_FEATURES,
            }
            print(f"writing_sample_cache {args.sample_cache}", flush=True)
            save_sample_cache(Path(args.sample_cache), samples, meta)
    samples["perspectives"] = 2 if args.perspectives == "dual" else 1

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = PatternNNUE(args.ft_dim, args.hidden1, args.hidden2, samples["perspectives"], args.active_columns).to(device)
    if args.load_state:
        print(f"loading_state {args.load_state}", flush=True)
        try:
            checkpoint = torch.load(args.load_state, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.load_state, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        if args.export_only:
            out_dir.mkdir(parents=True, exist_ok=True)
            export_model(model, out_file, args.layer_weight_scale, args.ft_weight_bits)
            val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.metric_limit)
            q_val_mse, q_val_mae, q_val_n = evaluate_quantized_loss(
                model,
                samples,
                "val",
                args.batch_size,
                args.quantized_metric_limit,
                args.layer_weight_scale,
                args.ft_weight_bits,
            )
            best = {
                "epoch": int(checkpoint.get("best", {}).get("epoch", 0)) if isinstance(checkpoint, dict) else 0,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "quantized_val_mse": q_val_mse,
                "quantized_val_mae": q_val_mae,
                "quantized_val_metric_samples": q_val_n,
            }
            print(
                f"export_only val_mse {val_mse:.6f} val_mae {val_mae:.6f} val_metric_samples {val_n} "
                f"quantized_val_mse {q_val_mse:.6f} quantized_val_mae {q_val_mae:.6f} "
                f"quantized_val_metric_samples {q_val_n}",
                flush=True,
            )
            write_summary(out_dir, args, entries, manifest_phase_counts, samples, best)
            print(f"wrote {out_file}", flush=True)
            return 0
    sparse_optimizer = torch.optim.SparseAdam([model.ft_weight.weight], lr=args.lr)
    dense_params = [
        model.ft_bias,
        *model.hidden1.parameters(),
        *model.hidden2.parameters(),
        model.output_weight,
        model.output_bias,
    ]
    dense_optimizer = torch.optim.AdamW(dense_params, lr=args.lr, weight_decay=args.dense_weight_decay)

    train_mse, train_mae, train_n = evaluate_loss(model, samples, "train", args.batch_size, device, args.metric_limit)
    val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.metric_limit)
    print(
        f"initial train_mse {train_mse:.6f} train_mae {train_mae:.6f} train_metric_samples {train_n} "
        f"val_mse {val_mse:.6f} val_mae {val_mae:.6f} val_metric_samples {val_n}",
        flush=True,
    )

    best: dict[str, float] = {"epoch": 0, "val_mse": float("inf"), "val_mae": float("inf")}
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = 0
        running_loss = 0.0
        start_ms = now_ms()
        for idx in batch_iter(args.train_samples, args.batch_size, True, args.seed + epoch):
            features, opponent_features, phases, target = make_batch(samples, "train", idx, device)
            pred = model(features, phases, opponent_features)
            loss = torch.mean((pred - target) ** 2)
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()
            batch_n = int(target.numel())
            seen += batch_n
            running_loss += float(loss.detach().cpu()) * batch_n
        train_mse, train_mae, train_n = evaluate_loss(model, samples, "train", args.batch_size, device, args.metric_limit)
        val_mse, val_mae, val_n = evaluate_loss(model, samples, "val", args.batch_size, device, args.metric_limit)
        if val_mse < best["val_mse"]:
            best = {"epoch": epoch, "val_mse": val_mse, "val_mae": val_mae}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch} elapsed_ms {now_ms() - start_ms} "
            f"train_epoch_mse {running_loss / max(1, seen):.6f} "
            f"train_mse {train_mse:.6f} train_mae {train_mae:.6f} "
            f"val_mse {val_mse:.6f} val_mae {val_mae:.6f} "
            f"best_epoch {best['epoch']} best_val_mse {best['val_mse']:.6f}",
            flush=True,
        )

    if best_state is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(best_state)
        model.to(device)
        export_model(model, out_file, args.layer_weight_scale, args.ft_weight_bits)
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
        print(
            f"quantized_best val_mse {q_val_mse:.6f} val_mae {q_val_mae:.6f} "
            f"val_metric_samples {q_val_n}",
            flush=True,
        )
        torch.save(
            {
                "state_dict": best_state,
                "arch": args.arch,
                "ft_dim": args.ft_dim,
                "hidden1": args.hidden1,
                "hidden2": args.hidden2,
                "perspectives": samples["perspectives"],
                "active_eval_types": args.active_eval_types,
                "active_feature_columns": args.active_columns,
                "input_feature_columns": N_FEATURE_COLUMNS,
                "total_input_features": TOTAL_INPUT_FEATURES,
                "feature_starts": FEATURE_STARTS,
                "layer_weight_scale": args.layer_weight_scale,
                "ft_weight_bits": args.ft_weight_bits,
                "best": best,
            },
            out_dir / "model_state.pt",
        )
    write_summary(out_dir, args, entries, manifest_phase_counts, samples, best)
    print(f"wrote {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
