#!/usr/bin/env python3
"""Train a small NN from learned scalar pattern scores.

Each selected pattern feature column has its own table that maps a local
pattern arrangement to one learned scalar.  These scalars are then passed to a
very small neural network.
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

from train_pattern_nnue import (
    ACTIVATION_SCALE,
    ADJ_EVAL_SIZES,
    ADJ_FEATURE_TO_EVAL_IDX,
    FEATURE_STARTS,
    FT_SCALE,
    FT_SHIFT,
    N_FEATURE_COLUMNS,
    N_PHASES,
    STEP,
    build_manifest,
    int_log2_power_of_two,
    load_sample_cache,
    load_selected_from_file,
    load_samples,
    next_model_dir,
    save_sample_cache,
)


PATTERN_SCORE_INPUT_KIND = 4
PATTERN_SCORE_PHASED_INPUT_KIND = 5
DEFAULT_LAYER_WEIGHT_SCALE = 32
PATTERN_NAMES = [
    "hv2",
    "d6+2C+X",
    "hv3",
    "d7+2corner",
    "hv4",
    "corner9",
    "d5+2X",
    "d8+2C",
    "edge+2x",
    "triangle",
    "corner+block",
    "cross",
    "edge+y",
    "narrow_triangle",
    "fish",
    "anvil",
]


def now_ms() -> int:
    return int(time.time() * 1000)


def iter_index_batches(
    n: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    shuffle_block_size: int = 0,
) -> Iterable[np.ndarray]:
    if n <= 0:
        return
    if not shuffle:
        for start in range(0, n, batch_size):
            yield np.arange(start, min(n, start + batch_size), dtype=np.int64)
        return

    rng = np.random.default_rng(seed)
    if shuffle_block_size > 0 and n > shuffle_block_size:
        n_blocks = (n + shuffle_block_size - 1) // shuffle_block_size
        block_order = np.arange(n_blocks, dtype=np.int64)
        rng.shuffle(block_order)
        for block_idx in block_order:
            block_start = int(block_idx) * shuffle_block_size
            block_end = min(n, block_start + shuffle_block_size)
            order = np.arange(block_start, block_end, dtype=np.int64)
            rng.shuffle(order)
            for start in range(0, order.size, batch_size):
                yield order[start:start + batch_size]
        return

    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def make_columnwise_starts() -> np.ndarray:
    starts = np.zeros(N_FEATURE_COLUMNS, dtype=np.uint32)
    offset = 0
    for col in range(N_FEATURE_COLUMNS):
        starts[col] = offset
        offset += int(ADJ_EVAL_SIZES[int(ADJ_FEATURE_TO_EVAL_IDX[col])])
    return starts


COLUMNWISE_FEATURE_STARTS = make_columnwise_starts()
COLUMNWISE_TOTAL_INPUT_FEATURES = int(
    COLUMNWISE_FEATURE_STARTS[-1] + ADJ_EVAL_SIZES[int(ADJ_FEATURE_TO_EVAL_IDX[-1])]
)


ARCHES = {
    "ps16": (16, 16),
    "ps24": (24, 24),
    "ps32": (32, 32),
    "ps48_32": (32, 32),
    "ps48_48": (48, 48),
    "ps16_8": (16, 8),
}


def make_pattern_columns(pattern_start: int, pattern_count: int) -> np.ndarray:
    first_col = pattern_start * 4
    return np.arange(first_col, first_col + pattern_count * 4, dtype=np.int64)


def make_pattern_column_names(columns: np.ndarray) -> list[str]:
    names = []
    for col in columns.tolist():
        pattern_idx = int(col) // 4
        symmetry_idx = int(col) % 4
        pattern_name = PATTERN_NAMES[pattern_idx] if pattern_idx < len(PATTERN_NAMES) else f"pattern{pattern_idx}"
        names.append(f"{pattern_name}/{symmetry_idx}")
    return names


def select_active_columns(pattern_set: str) -> np.ndarray:
    if pattern_set == "mo_end4":
        return make_pattern_columns(8, 4)
    if pattern_set == "first12":
        return make_pattern_columns(0, 12)
    raise ValueError(f"unknown pattern_set: {pattern_set}")


def feature_columns_to_mask(columns: np.ndarray) -> tuple[int, int]:
    mask = 0
    for col in columns.tolist():
        mask |= 1 << int(col)
    return mask & ((1 << 64) - 1), mask >> 64


def pattern_score_ids_from_global(features: np.ndarray, active: np.ndarray) -> np.ndarray:
    local = features[:, active].astype(np.uint32, copy=False) - FEATURE_STARTS[active][None, :]
    return local + COLUMNWISE_FEATURE_STARTS[active][None, :]


def shrink_samples_to_pattern_score_ids(samples: dict[str, np.ndarray], active_columns: np.ndarray) -> None:
    samples["train_features"] = pattern_score_ids_from_global(samples["train_features"], active_columns)
    samples["val_features"] = pattern_score_ids_from_global(samples["val_features"], active_columns)


def prepare_samples_for_pattern_score(samples: dict[str, np.ndarray], active_columns: np.ndarray) -> None:
    train_features = samples["train_features"]
    if train_features.shape[1] == N_FEATURE_COLUMNS:
        shrink_samples_to_pattern_score_ids(samples, active_columns)
        samples["pattern_score_local_features"] = False
    elif train_features.shape[1] == active_columns.size:
        samples["pattern_score_local_features"] = train_features.dtype == np.uint16
    else:
        raise ValueError(
            f"unexpected sample feature shape {train_features.shape}; "
            f"expected {N_FEATURE_COLUMNS} or {active_columns.size} columns"
        )
    samples["active_columns"] = active_columns.astype(np.int64, copy=True)
    samples["active_column_starts"] = COLUMNWISE_FEATURE_STARTS[active_columns].astype(np.uint32, copy=True)


def batch_pattern_score_ids(samples: dict[str, np.ndarray], prefix: str, idx: np.ndarray) -> np.ndarray:
    features = samples[f"{prefix}_features"][idx]
    if samples.get("pattern_score_local_features", False):
        return features.astype(np.uint32, copy=False) + samples["active_column_starts"][None, :]
    return features


def load_pattern_score_samples(
    entries,
    train_samples: int,
    val_samples: int,
    seed: int,
    progress_interval_sec: int,
    full_read_ratio: float,
    active_columns: np.ndarray,
) -> dict[str, np.ndarray]:
    total = entries[-1].begin + entries[-1].records if entries else 0
    need = train_samples + val_samples
    chosen, validation = sample_indices_with_progress(total, need, train_samples, seed, progress_interval_sec)
    input_dim = int(active_columns.size)

    train_features = np.empty((train_samples, input_dim), dtype=np.uint16)
    val_features = np.empty((val_samples, input_dim), dtype=np.uint16)
    train_phase = np.empty(train_samples, dtype=np.uint8)
    val_phase = np.empty(val_samples, dtype=np.uint8)
    train_score = np.empty(train_samples, dtype=np.int8)
    val_score = np.empty(val_samples, dtype=np.int8)
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
    max_allowed = ADJ_EVAL_SIZES[ADJ_FEATURE_TO_EVAL_IDX[active_columns]]

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
        local_features = selected["features"][:, active_columns]
        bad_mask = np.any(local_features >= max_allowed[None, :], axis=1)
        bad_features += int(np.count_nonzero(bad_mask))
        if np.any(bad_mask):
            keep = ~bad_mask
            local_features = local_features[keep]
            local_phase = local_phase[keep]
            local_validation = local_validation[keep]
            selected = selected[keep]

        scores = selected["score"].astype(np.int16, copy=False)
        if scores.size and (int(scores.min()) < -128 or int(scores.max()) > 127):
            raise ValueError("score is outside int8 range")
        is_val = local_validation
        is_train = ~is_val
        n_train = int(is_train.sum())
        n_val = int(is_val.sum())
        if n_train:
            sl = slice(train_pos, train_pos + n_train)
            train_features[sl] = local_features[is_train]
            train_phase[sl] = local_phase[is_train]
            train_score[sl] = scores[is_train].astype(np.int8, copy=False)
            train_phase_counts += np.bincount(train_phase[sl], minlength=N_PHASES).astype(np.uint64)
            train_pos += n_train
        if n_val:
            sl = slice(val_pos, val_pos + n_val)
            val_features[sl] = local_features[is_val]
            val_phase[sl] = local_phase[is_val]
            val_score[sl] = scores[is_val].astype(np.int8, copy=False)
            val_phase_counts += np.bincount(val_phase[sl], minlength=N_PHASES).astype(np.uint64)
            val_pos += n_val

        files_read += 1
        current_ms = now_ms()
        if progress_interval_sec > 0 and current_ms >= next_log_ms:
            print(
                f"sample_loading_compact files_read {files_read} train {train_pos}/{train_samples} "
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
        f"loaded_compact train {train_pos} val {val_pos} bad_features {bad_features} "
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
        "pattern_score_local_features": True,
        "active_columns": active_columns.astype(np.int64, copy=True),
        "active_column_starts": COLUMNWISE_FEATURE_STARTS[active_columns].astype(np.uint32, copy=True),
    }


def sample_indices_with_progress(
    total: int,
    need: int,
    train_samples: int,
    seed: int,
    progress_interval_sec: int,
) -> tuple[np.ndarray, np.ndarray]:
    if need > total:
        raise ValueError(f"requested samples {need} exceeds available records {total}")
    rng = np.random.default_rng(seed)
    start_ms = now_ms()
    if need == total:
        chosen = np.arange(total, dtype=np.uint64)
    elif need == 0:
        chosen = np.empty(0, dtype=np.uint64)
    else:
        target_unique = min(total, max(need, int(math.ceil(need * 1.05))))
        draw = int(math.ceil(-float(total) * math.log1p(-float(target_unique) / float(total))))
        draw = min(total, max(draw, need))
        while True:
            print(
                f"sample_request_generation draw {draw} target_unique {target_unique} "
                f"need {need} total {total}",
                flush=True,
            )
            drawn = np.empty(draw, dtype=np.uint64)
            chunk = min(200_000_000, draw)
            for begin in range(0, draw, chunk):
                end = min(draw, begin + chunk)
                drawn[begin:end] = rng.integers(0, total, size=end - begin, dtype=np.uint64)
                current_ms = now_ms()
                if progress_interval_sec > 0:
                    print(
                        f"sample_request_generation generated {end}/{draw} elapsed_ms {current_ms - start_ms}",
                        flush=True,
                    )
            print(
                f"sample_request_generation unique_start drawn {draw} elapsed_ms {now_ms() - start_ms}",
                flush=True,
            )
            unique = np.unique(drawn)
            del drawn
            print(
                f"sample_request_generation unique_done unique {unique.size}/{need} "
                f"elapsed_ms {now_ms() - start_ms}",
                flush=True,
            )
            if unique.size >= need:
                rng.shuffle(unique)
                chosen = unique[:need].copy()
                del unique
                break
            del unique
            target_unique = min(total, max(need, int(math.ceil(target_unique * 1.10))))
            draw = int(math.ceil(-float(total) * math.log1p(-float(target_unique) / float(total))))
            draw = min(total, max(draw, need))
    print(f"sample_request_generation sort_start chosen {chosen.size} elapsed_ms {now_ms() - start_ms}", flush=True)
    chosen.sort(kind="stable")
    validation = np.zeros(need, dtype=np.bool_)
    val_samples = need - train_samples
    if val_samples > 0:
        validation_positions = rng.choice(need, size=val_samples, replace=False)
        validation[validation_positions] = True
    print(f"sample_request_generation done chosen {chosen.size} elapsed_ms {now_ms() - start_ms}", flush=True)
    return chosen, validation


class PatternScoreNNUE(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, phase_specific_score_table: bool):
        super().__init__()
        self.ft_dim = input_dim
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.phase_specific_score_table = phase_specific_score_table
        n_score_embeddings = COLUMNWISE_TOTAL_INPUT_FEATURES * N_PHASES if phase_specific_score_table else COLUMNWISE_TOTAL_INPUT_FEATURES
        self.score_table = nn.Embedding(n_score_embeddings, 1, sparse=True)
        self.input_bias = nn.Parameter(torch.full((input_dim,), 2.0))
        self.hidden1 = nn.Linear(input_dim, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.output_weight = nn.Parameter(torch.empty(N_PHASES, hidden2))
        self.output_bias = nn.Parameter(torch.zeros(N_PHASES))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.score_table.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.hidden1.weight)
        with torch.no_grad():
            for i in range(min(self.ft_dim, self.hidden1_dim)):
                self.hidden1.weight[i, i] = 1.0
        nn.init.zeros_(self.hidden1.bias)
        nn.init.zeros_(self.hidden2.weight)
        with torch.no_grad():
            for i in range(min(self.hidden1_dim, self.hidden2_dim)):
                self.hidden2.weight[i, i] = 1.0
        nn.init.zeros_(self.hidden2.bias)
        nn.init.normal_(self.output_weight, mean=0.0, std=0.10)
        nn.init.zeros_(self.output_bias)

    @staticmethod
    def clipped_relu(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 127.0 / 16.0)

    def initialize_output_bias(self, phases: np.ndarray, scores: np.ndarray) -> None:
        counts = np.bincount(phases.astype(np.int64, copy=False), minlength=N_PHASES)
        sums = np.bincount(phases.astype(np.int64, copy=False), weights=scores, minlength=N_PHASES)
        means = np.divide(sums, np.maximum(counts, 1), out=np.zeros_like(sums, dtype=np.float64), where=counts > 0)
        with torch.no_grad():
            self.output_bias.copy_(torch.from_numpy(means.astype(np.float32)))

    def forward(self, features: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        score_ids = features
        if self.phase_specific_score_table:
            score_ids = score_ids + phases[:, None] * COLUMNWISE_TOTAL_INPUT_FEATURES
        x = self.input_bias + self.score_table(score_ids).squeeze(-1)
        x = self.clipped_relu(x)
        x = self.clipped_relu(self.hidden1(x))
        x = self.clipped_relu(self.hidden2(x))
        return (x * self.output_weight[phases]).sum(dim=1) + self.output_bias[phases]


@dataclass
class QuantizedPatternScoreNNUE:
    ft_bias: np.ndarray
    score_table: np.ndarray
    hidden1_bias: np.ndarray
    hidden1_weight: np.ndarray
    hidden2_bias: np.ndarray
    hidden2_weight: np.ndarray
    output_bias: np.ndarray
    output_weight: np.ndarray
    hidden1_padded: int
    hidden2_padded: int
    hidden_shift: int
    output_shift: int
    ft_weight_bits: int
    phase_specific_score_table: bool


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


def make_quantization_shifts(layer_weight_scale: int) -> tuple[int, int]:
    hidden_shift = int_log2_power_of_two(layer_weight_scale, "layer_weight_scale")
    output_scale_divisor = ACTIVATION_SCALE * layer_weight_scale
    if output_scale_divisor % STEP != 0:
        raise ValueError(
            "ACTIVATION_SCALE * layer_weight_scale must be divisible by STEP: "
            f"{ACTIVATION_SCALE} * {layer_weight_scale}"
        )
    output_shift = int_log2_power_of_two(output_scale_divisor // STEP, "output_scale_divisor / STEP")
    return hidden_shift, output_shift


def quantize_model(
    model: PatternScoreNNUE,
    layer_weight_scale: int,
    ft_weight_bits: int,
) -> QuantizedPatternScoreNNUE:
    hidden1_padded = math.ceil(model.hidden1_dim / 32) * 32
    hidden2_padded = math.ceil(model.hidden2_dim / 32) * 32
    post_input_padded = math.ceil(model.ft_dim / 32) * 32
    hidden_shift, output_shift = make_quantization_shifts(layer_weight_scale)
    layer_acc_scale = ACTIVATION_SCALE * layer_weight_scale
    ft_bias = quantize_to_int16(model.input_bias, FT_SCALE)
    if ft_weight_bits == 8:
        score_table = quantize_to_int8(model.score_table.weight[:, 0], FT_SCALE)
    elif ft_weight_bits == 16:
        score_table = quantize_to_int16(model.score_table.weight[:, 0], FT_SCALE)
    else:
        raise ValueError(f"ft_weight_bits must be 8 or 16: {ft_weight_bits}")
    h1_bias = np.rint(model.hidden1.bias.detach().cpu().numpy() * layer_acc_scale).astype("<i4")
    h1_weight = pad_int8_matrix(quantize_to_int8(model.hidden1.weight, layer_weight_scale), post_input_padded)
    h2_bias = np.rint(model.hidden2.bias.detach().cpu().numpy() * layer_acc_scale).astype("<i4")
    h2_weight = pad_int8_matrix(quantize_to_int8(model.hidden2.weight, layer_weight_scale), hidden1_padded)
    out_bias = np.rint(model.output_bias.detach().cpu().numpy() * layer_acc_scale).astype("<i4")
    out_weight = pad_int8_matrix(quantize_to_int8(model.output_weight, layer_weight_scale), hidden2_padded)
    return QuantizedPatternScoreNNUE(
        ft_bias=ft_bias,
        score_table=score_table,
        hidden1_bias=h1_bias,
        hidden1_weight=h1_weight,
        hidden2_bias=h2_bias,
        hidden2_weight=h2_weight,
        output_bias=out_bias,
        output_weight=out_weight,
        hidden1_padded=hidden1_padded,
        hidden2_padded=hidden2_padded,
        hidden_shift=hidden_shift,
        output_shift=output_shift,
        ft_weight_bits=ft_weight_bits,
        phase_specific_score_table=model.phase_specific_score_table,
    )


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


def quantized_forward(q: QuantizedPatternScoreNNUE, features: np.ndarray, phases: np.ndarray) -> np.ndarray:
    score_ids = features
    if q.phase_specific_score_table:
        score_ids = score_ids + phases[:, None].astype(np.uint32, copy=False) * np.uint32(COLUMNWISE_TOTAL_INPUT_FEATURES)
    acc = q.ft_bias.astype(np.int32, copy=False)[None, :] + q.score_table[score_ids].astype(np.int32, copy=False)
    post_padded = math.ceil(acc.shape[1] / 32) * 32
    post_input = np.zeros((features.shape[0], post_padded), dtype=np.uint8)
    post_input[:, : acc.shape[1]] = clamp_u8_shifted(acc, FT_SHIFT)
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


def export_model(
    model: PatternScoreNNUE,
    out_file: Path,
    layer_weight_scale: int,
    ft_weight_bits: int,
    active_columns: np.ndarray,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    q = quantize_model(model, layer_weight_scale, ft_weight_bits)
    if model.phase_specific_score_table:
        ft_weight = q.score_table
    else:
        if ft_weight_bits == 8:
            ft_weight = np.zeros((COLUMNWISE_TOTAL_INPUT_FEATURES, model.ft_dim), dtype=np.int8)
        else:
            ft_weight = np.zeros((COLUMNWISE_TOTAL_INPUT_FEATURES, model.ft_dim), dtype="<i2")
        for slot, col in enumerate(active_columns.tolist()):
            start = int(COLUMNWISE_FEATURE_STARTS[col])
            size = int(ADJ_EVAL_SIZES[int(ADJ_FEATURE_TO_EVAL_IDX[col])])
            ft_weight[start:start + size, slot] = q.score_table[start:start + size]

    with out_file.open("wb") as f:
        f.write(b"EGNNUE1\0")
        version = 5 if model.phase_specific_score_table else 4
        input_kind = PATTERN_SCORE_PHASED_INPUT_KIND if model.phase_specific_score_table else PATTERN_SCORE_INPUT_KIND
        header = np.array(
            [
                version,
                COLUMNWISE_TOTAL_INPUT_FEATURES,
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
        mask_lo, mask_hi = feature_columns_to_mask(active_columns)
        reserved = np.zeros(8, dtype="<u4")
        reserved[0] = input_kind
        reserved[1] = N_FEATURE_COLUMNS
        reserved[2] = 1
        reserved[3] = mask_lo & 0xFFFFFFFF
        reserved[4] = (mask_lo >> 32) & 0xFFFFFFFF
        reserved[5] = mask_hi & 0xFFFFFFFF
        reserved[6] = ft_weight_bits
        reserved.tofile(f)
        q.ft_bias.tofile(f)
        ft_weight.tofile(f)
        q.hidden1_bias.tofile(f)
        q.hidden1_weight.tofile(f)
        q.hidden2_bias.tofile(f)
        q.hidden2_weight.tofile(f)
        q.output_bias.tofile(f)
        q.output_weight.tofile(f)


def make_batch(
    samples: dict[str, np.ndarray],
    prefix: str,
    idx: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features_np = batch_pattern_score_ids(samples, prefix, idx)
    return (
        torch.from_numpy(features_np.astype(np.int64, copy=False)).to(device=device, non_blocking=True),
        torch.from_numpy(samples[f"{prefix}_phase"][idx].astype(np.int64, copy=False)).to(device=device, non_blocking=True),
        torch.from_numpy(samples[f"{prefix}_score"][idx].astype(np.float32, copy=False)).to(device=device, non_blocking=True),
    )


def initialize_from_shared_state(model: PatternScoreNNUE, shared_state_path: Path, device: torch.device) -> None:
    if not model.phase_specific_score_table:
        raise ValueError("--init-shared-state is only valid with --phase-specific-score-table")
    print(f"initializing_from_shared_state {shared_state_path}", flush=True)
    try:
        checkpoint = torch.load(shared_state_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(shared_state_path, map_location=device)
    shared_state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    shared_score = shared_state["score_table.weight"].to(device=device)
    if shared_score.shape[0] != COLUMNWISE_TOTAL_INPUT_FEATURES or shared_score.shape[1] != 1:
        raise ValueError(f"incompatible shared score_table shape: {tuple(shared_score.shape)}")
    with torch.no_grad():
        model.score_table.weight.view(N_PHASES, COLUMNWISE_TOTAL_INPUT_FEATURES, 1).copy_(shared_score[None, :, :])
        for key in [
            "input_bias",
            "hidden1.weight",
            "hidden1.bias",
            "hidden2.weight",
            "hidden2.bias",
            "output_weight",
            "output_bias",
        ]:
            if key in shared_state and key in model.state_dict() and shared_state[key].shape == model.state_dict()[key].shape:
                model.state_dict()[key].copy_(shared_state[key].to(device=device))


@torch.no_grad()
def evaluate_loss(
    model: PatternScoreNNUE,
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
    for idx in iter_index_batches(n, batch_size, False, 0):
        features, phases, target = make_batch(samples, prefix, idx, device)
        pred = model(features, phases)
        err = pred - target
        se += float((err * err).sum().detach().cpu())
        ae += float(err.abs().sum().detach().cpu())
        seen += int(target.numel())
    return se / max(1, seen), ae / max(1, seen), seen


@torch.no_grad()
def evaluate_quantized_loss(
    model: PatternScoreNNUE,
    samples: dict[str, np.ndarray],
    prefix: str,
    batch_size: int,
    limit: int,
    layer_weight_scale: int,
    ft_weight_bits: int,
) -> tuple[float, float, int]:
    q = quantize_model(model, layer_weight_scale, ft_weight_bits)
    n = len(samples[f"{prefix}_score"])
    if limit > 0:
        n = min(n, limit)
    se = 0.0
    ae = 0.0
    seen = 0
    for idx in iter_index_batches(n, batch_size, False, 0):
        pred = quantized_forward(q, batch_pattern_score_ids(samples, prefix, idx), samples[f"{prefix}_phase"][idx])
        target = samples[f"{prefix}_score"][idx].astype(np.float32, copy=False)
        err = pred - target
        se += float(np.dot(err, err))
        ae += float(np.abs(err).sum())
        seen += int(target.size)
    return se / max(1, seen), ae / max(1, seen), seen


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    total_records: int,
    manifest_phase_counts: np.ndarray,
    samples: dict[str, np.ndarray] | None,
    best: dict[str, float],
    active_columns: np.ndarray,
    active_column_names: list[str],
    input_dim: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "kind": "pattern_score_nnue",
        "description": "learned scalar pattern scores followed by a small MLP",
        "data_root": str(Path(args.data_root).resolve()),
        "record_start": args.record_start,
        "record_end": args.record_end,
        "available_records": int(total_records),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
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
        "columnwise_total_input_features": COLUMNWISE_TOTAL_INPUT_FEATURES,
        "layer_weight_scale": args.layer_weight_scale,
        "ft_weight_bits": args.ft_weight_bits,
        "seed": args.seed,
        "legacy_full_feature_storage": args.legacy_full_feature_storage,
        "shuffle_block_size": args.shuffle_block_size,
        "manifest_phase_counts": [int(x) for x in manifest_phase_counts],
        "best": best,
    }
    if samples is not None:
        summary["pattern_score_local_features"] = bool(samples.get("pattern_score_local_features", False))
        summary["feature_storage_dtype"] = str(samples["train_features"].dtype)
        summary["score_storage_dtype"] = str(samples["train_score"].dtype)
        summary["train_phase_counts"] = [int(x) for x in samples["train_phase_counts"]]
        summary["val_phase_counts"] = [int(x) for x in samples["val_phase_counts"]]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("EGAROUCID_INDEXED_DATA", "E:/egaroucid_data/train_data/bin_data/20241125_1"))
    parser.add_argument("--record-start", type=int, default=223)
    parser.add_argument("--record-end", type=int, default=-1)
    parser.add_argument("--arch", choices=sorted(ARCHES), default="ps16")
    parser.add_argument("--pattern-set", choices=["mo_end4", "first12"], default="mo_end4")
    parser.add_argument("--phase-specific-score-table", action="store_true")
    parser.add_argument("--hidden1", type=int, default=0)
    parser.add_argument("--hidden2", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=10_000_000)
    parser.add_argument("--val-samples", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--dense-weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--layer-weight-scale", type=int, default=DEFAULT_LAYER_WEIGHT_SCALE)
    parser.add_argument("--ft-weight-bits", type=int, choices=[8, 16], default=8)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--metric-limit", type=int, default=1_000_000)
    parser.add_argument("--quantized-metric-limit", type=int, default=1_000_000)
    parser.add_argument("--progress-interval-sec", type=int, default=30)
    parser.add_argument("--full-read-ratio", type=float, default=0.10)
    parser.add_argument("--shuffle-block-size", type=int, default=0)
    parser.add_argument("--legacy-full-feature-storage", action="store_true")
    parser.add_argument("--sample-cache", default="")
    parser.add_argument("--load-state", default="")
    parser.add_argument("--init-shared-state", default="")
    parser.add_argument("--export-only", action="store_true")
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
        f"train{args.train_samples}_val{args.val_samples}_e{args.epochs}"
    )
    out_dir = next_model_dir(Path(args.model_root), date, model_name)
    out_file = Path(args.out_file) if args.out_file else out_dir / f"eval_nnue_pattern_score_{args.pattern_set}_{score_table_scope}_{args.arch}.egevnnue"

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
    if args.dry_run:
        write_summary(out_dir, args, total_records, manifest_phase_counts, None, {}, active_columns, active_column_names, input_dim)
        return 0

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

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
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
        if args.export_only:
            out_dir.mkdir(parents=True, exist_ok=True)
            export_model(model, out_file, args.layer_weight_scale, args.ft_weight_bits, active_columns)
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
            write_summary(out_dir, args, total_records, manifest_phase_counts, samples, best, active_columns, active_column_names, input_dim)
            print(f"wrote {out_file}", flush=True)
            return 0
    sparse_optimizer = torch.optim.SparseAdam([model.score_table.weight], lr=args.lr)
    dense_params = [
        model.input_bias,
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
                    f"epoch_training epoch {epoch} processed {seen}/{args.train_samples} "
                    f"train_epoch_mse {running_loss / max(1, seen):.6f} "
                    f"elapsed_ms {current_ms - start_ms}",
                    flush=True,
                )
                next_log_ms = current_ms + args.progress_interval_sec * 1000
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
        print(
            f"quantized_best val_mse {q_val_mse:.6f} val_mae {q_val_mae:.6f} "
            f"val_metric_samples {q_val_n}",
            flush=True,
        )
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
                "best": best,
            },
            out_dir / "model_state.pt",
        )
    write_summary(out_dir, args, total_records, manifest_phase_counts, samples, best, active_columns, active_column_names, input_dim)
    print(f"wrote {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
