#!/usr/bin/env python3
"""Compute phase-wise MSE/MAE on high-precision indexed test data.

The indexed data format is:

    int16 n_discs
    int16 player
    uint16 features[65]
    int16 score

This script evaluates the existing pattern evaluator, FM evaluator, and the
pattern-score NNUE candidates directly from their binary model files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from train_pattern_nnue import (
    ADJ_EVAL_SIZES,
    ADJ_FEATURE_TO_EVAL_IDX,
    INDEXED_DTYPE,
    INDEXED_RECORD_BYTES,
    N_FEATURE_COLUMNS,
    N_PHASES,
    STEP,
)
from train_pattern_score_nnue import COLUMNWISE_FEATURE_STARTS, COLUMNWISE_TOTAL_INPUT_FEATURES


N_PATTERN_FEATURES = 64
N_PATTERN_PARAMS_RAW = int(ADJ_EVAL_SIZES[:16].sum())
LINEAR_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + 65
SCORE_MAX = 64
N_ZEROS_PLUS = 1 << 12
FM_MAGIC = b"EGFM001\0"
NNUE_MAGIC_PREFIX = b"EGNNUE1"
FM_PHASE_RANGE_FLAG = 0x80000000
FM_PHASE_START_SHIFT = 16
FM_PHASE_END_SHIFT = 22
FM_PHASE_FLAG_MASK = 0x3F

PATTERN_SIZES = np.array(
    [8, 9, 8, 9, 8, 9, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    dtype=np.uint32,
)
POW3 = np.array([3 ** int(x) for x in PATTERN_SIZES], dtype=np.uint32)
PATTERN_FEATURE_LIMITS = np.repeat(POW3, 4).astype(np.uint32)

LINEAR_EVAL_STARTS = np.zeros(len(ADJ_EVAL_SIZES), dtype=np.uint32)
_linear_start = 0
for _idx, _size in enumerate(ADJ_EVAL_SIZES):
    LINEAR_EVAL_STARTS[_idx] = _linear_start
    _linear_start += int(_size)

FM_FEATURE_OFFSETS = np.zeros(N_PATTERN_FEATURES, dtype=np.uint32)
_fm_offset = 0
for _idx in range(N_PATTERN_FEATURES):
    FM_FEATURE_OFFSETS[_idx] = _fm_offset
    _fm_offset += int(POW3[_idx // 4])
FM_TOTAL_VECTORS = int(_fm_offset)


@dataclass
class Metrics:
    n: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        diff = pred.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
        self.n += int(diff.size)
        self.sum_sq += float(np.dot(diff, diff))
        self.sum_abs += float(np.abs(diff).sum())

    @property
    def mse(self) -> float:
        return self.sum_sq / self.n if self.n else float("nan")

    @property
    def mae(self) -> float:
        return self.sum_abs / self.n if self.n else float("nan")


def now_ms() -> int:
    return int(time.time() * 1000)


def rounded_shift_signed(values: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return values
    values = values.astype(np.int64, copy=False)
    half = 1 << (shift - 1)
    out = np.empty_like(values)
    non_negative = values >= 0
    out[non_negative] = (values[non_negative] + half) >> shift
    out[~non_negative] = -(((-values[~non_negative]) + half) >> shift)
    return out


def round_div_signed(values: np.ndarray, denom: int) -> np.ndarray:
    values = values.astype(np.int64, copy=False)
    out = np.empty_like(values)
    non_negative = values >= 0
    out[non_negative] = (values[non_negative] + denom // 2) // denom
    out[~non_negative] = -(((-values[~non_negative]) + denom // 2) // denom)
    return out


def quantized_score_from_raw(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.int64, copy=False)
    adjusted = raw + np.where(raw >= 0, STEP // 2, -(STEP // 2))
    score = np.trunc(adjusted.astype(np.float64) / float(STEP)).astype(np.int32)
    return np.clip(score, -SCORE_MAX, SCORE_MAX).astype(np.float32)


def read_u32(f) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise ValueError("unexpected end of file while reading uint32")
    return struct.unpack("<I", b)[0]


def read_i32(f) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise ValueError("unexpected end of file while reading int32")
    return struct.unpack("<i", b)[0]


def read_u64(f) -> int:
    b = f.read(8)
    if len(b) != 8:
        raise ValueError("unexpected end of file while reading uint64")
    return struct.unpack("<Q", b)[0]


def load_egev2(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        n_compressed = read_i32(f)
        payload = np.fromfile(f, dtype="<i2", count=n_compressed)
    if payload.size != n_compressed:
        raise ValueError(f"broken egev2 payload: {path}")
    chunks: list[np.ndarray] = []
    for value in payload.astype(np.int32, copy=False):
        if value >= N_ZEROS_PLUS:
            chunks.append(np.zeros(value - N_ZEROS_PLUS, dtype="<i2"))
        else:
            chunks.append(np.array([value], dtype="<i2"))
    params = np.concatenate(chunks) if chunks else np.empty(0, dtype="<i2")
    expected = N_PHASES * LINEAR_PARAMS_PER_PHASE
    if params.size != expected:
        raise ValueError(f"unexpected egev2 size: {path} has {params.size}, expected {expected}")
    return params.reshape(N_PHASES, LINEAR_PARAMS_PER_PHASE)


def load_linear_raw(path: Path) -> np.ndarray:
    params = np.fromfile(path, dtype="<i2")
    expected = N_PHASES * LINEAR_PARAMS_PER_PHASE
    if params.size != expected:
        raise ValueError(f"unexpected raw eval size: {path} has {params.size}, expected {expected}")
    return params.reshape(N_PHASES, LINEAR_PARAMS_PER_PHASE)


class LinearEvaluator:
    def __init__(self, name: str, path: Path, linear: np.ndarray):
        self.name = name
        self.path = path
        self.linear = linear.astype(np.int16, copy=False)

    @classmethod
    def from_file(cls, name: str, path: Path) -> "LinearEvaluator":
        with path.open("rb") as f:
            head = f.read(8)
        if head == FM_MAGIC:
            fm = FMEvaluator.from_file(name, path)
            return fm
        if path.suffix.lower() == ".egev2":
            return cls(name, path, load_egev2(path))
        return cls(name, path, load_linear_raw(path))

    def _raw_linear(self, features: np.ndarray, phase: int) -> np.ndarray:
        phase_params = self.linear[phase]
        raw = np.zeros(features.shape[0], dtype=np.int64)
        for col in range(N_FEATURE_COLUMNS):
            eval_idx = int(ADJ_FEATURE_TO_EVAL_IDX[col])
            start = int(LINEAR_EVAL_STARTS[eval_idx])
            raw += phase_params[start + features[:, col].astype(np.int64, copy=False)]
        return raw

    def predict(self, features: np.ndarray, phases: np.ndarray) -> np.ndarray:
        if np.unique(phases).size != 1:
            out = np.empty(features.shape[0], dtype=np.float32)
            for phase in np.unique(phases):
                mask = phases == phase
                out[mask] = self.predict(features[mask], phases[mask])
            return out
        phase = int(phases[0])
        raw = self._raw_linear(features, phase)
        return quantized_score_from_raw(raw)


class FMEvaluator(LinearEvaluator):
    def __init__(
        self,
        name: str,
        path: Path,
        linear: np.ndarray,
        n_fm_phases: int,
        fm_dim: int,
        fm_scale: int,
        flags: int,
        vectors: np.ndarray,
    ):
        super().__init__(name, path, linear)
        self.n_fm_phases = n_fm_phases
        self.fm_dim = fm_dim
        self.fm_scale = fm_scale
        self.flags = flags
        self.vectors = vectors.reshape(n_fm_phases, FM_TOTAL_VECTORS, fm_dim)
        self.active_features = self._active_features(flags)
        self.phase_enabled = self._phase_enabled(flags)
        self.phase_table = np.array(
            [min(n_fm_phases - 1, (phase * n_fm_phases) // N_PHASES) for phase in range(N_PHASES)],
            dtype=np.int64,
        )
        denom = 2 * fm_scale * fm_scale
        self.fm_denom = denom
        self.fm_denom_shift = int(math.log2(denom)) if denom > 0 and denom & (denom - 1) == 0 else -1

    @classmethod
    def from_file(cls, name: str, path: Path) -> "FMEvaluator":
        with path.open("rb") as f:
            magic = f.read(8)
            if magic != FM_MAGIC:
                raise ValueError(f"not an egevfm file: {path}")
            version = read_u32(f)
            n_phases = read_u32(f)
            linear_per_phase = read_u32(f)
            n_fm_phases = read_u32(f)
            n_features = read_u32(f)
            fm_dim = read_u32(f)
            fm_scale = read_i32(f)
            flags = read_u32(f)
            linear_count = read_u64(f)
            fm_count = read_u64(f)
            if (
                version != 1
                or n_phases != N_PHASES
                or linear_per_phase != LINEAR_PARAMS_PER_PHASE
                or n_features != N_PATTERN_FEATURES
                or linear_count != N_PHASES * LINEAR_PARAMS_PER_PHASE
                or fm_count != n_fm_phases * FM_TOTAL_VECTORS * fm_dim
            ):
                raise ValueError(f"unsupported egevfm header: {path}")
            linear = np.fromfile(f, dtype="<i2", count=int(linear_count))
            vectors = np.fromfile(f, dtype=np.int8, count=int(fm_count))
        if linear.size != linear_count or vectors.size != fm_count:
            raise ValueError(f"broken egevfm payload: {path}")
        return cls(
            name,
            path,
            linear.reshape(N_PHASES, LINEAR_PARAMS_PER_PHASE),
            int(n_fm_phases),
            int(fm_dim),
            int(fm_scale),
            int(flags),
            vectors,
        )

    @staticmethod
    def _active_features(flags: int) -> np.ndarray:
        mask = flags & 0xFFFF
        if mask == 0:
            return np.arange(N_PATTERN_FEATURES, dtype=np.int64)
        active = [idx for idx in range(N_PATTERN_FEATURES) if mask & (1 << (idx // 4))]
        return np.array(active, dtype=np.int64)

    @staticmethod
    def _phase_enabled(flags: int) -> np.ndarray:
        if flags & FM_PHASE_RANGE_FLAG == 0:
            return np.ones(N_PHASES, dtype=bool)
        start = (flags >> FM_PHASE_START_SHIFT) & FM_PHASE_FLAG_MASK
        end = (flags >> FM_PHASE_END_SHIFT) & FM_PHASE_FLAG_MASK
        enabled = np.zeros(N_PHASES, dtype=bool)
        enabled[start : end + 1] = True
        return enabled

    def _raw_fm(self, features: np.ndarray, phase: int) -> np.ndarray:
        if not self.phase_enabled[phase]:
            return np.zeros(features.shape[0], dtype=np.int64)
        fm_phase = int(self.phase_table[phase])
        active = self.active_features
        feature_values = features[:, active].astype(np.int64, copy=False)
        if np.any(feature_values >= PATTERN_FEATURE_LIMITS[active][None, :]):
            raise ValueError(f"feature value exceeds FM feature range at phase {phase}")
        row_ids = FM_FEATURE_OFFSETS[active][None, :] + feature_values
        selected = self.vectors[fm_phase, row_ids].astype(np.int32, copy=False)
        sums = selected.sum(axis=1, dtype=np.int64)
        square_sums = (selected.astype(np.int64, copy=False) * selected.astype(np.int64, copy=False)).sum(axis=1)
        diff = (sums * sums).sum(axis=1) - square_sums.sum(axis=1)
        if self.fm_denom_shift >= 0:
            return rounded_shift_signed(diff, self.fm_denom_shift)
        return round_div_signed(diff, self.fm_denom)

    def predict(self, features: np.ndarray, phases: np.ndarray) -> np.ndarray:
        if np.unique(phases).size != 1:
            out = np.empty(features.shape[0], dtype=np.float32)
            for phase in np.unique(phases):
                mask = phases == phase
                out[mask] = self.predict(features[mask], phases[mask])
            return out
        phase = int(phases[0])
        raw = self._raw_linear(features, phase) + self._raw_fm(features, phase)
        return quantized_score_from_raw(raw)


class PatternScoreNNUEEvaluator:
    def __init__(
        self,
        name: str,
        path: Path,
        header: dict[str, int],
        reserved: np.ndarray,
        ft_bias: np.ndarray,
        ft_weight: np.ndarray,
        hidden1_bias: np.ndarray,
        hidden1_weight: np.ndarray,
        hidden2_bias: np.ndarray,
        hidden2_weight: np.ndarray,
        output_bias: np.ndarray,
        output_weight: np.ndarray,
    ):
        self.name = name
        self.path = path
        self.header = header
        self.reserved = reserved
        self.ft_bias = ft_bias.astype(np.int32, copy=False)
        self.ft_weight = ft_weight
        self.hidden1_bias = hidden1_bias.astype(np.int32, copy=False)
        self.hidden1_weight = hidden1_weight.astype(np.int32, copy=False)
        self.hidden2_bias = hidden2_bias.astype(np.int32, copy=False)
        self.hidden2_weight = hidden2_weight.astype(np.int32, copy=False)
        self.output_bias = output_bias.astype(np.int32, copy=False)
        self.output_weight = output_weight.astype(np.int32, copy=False)
        self.input_kind = int(reserved[0])
        self.ft_weight_bits = int(reserved[6])
        self.active_columns = self._active_columns(reserved)
        self.post_input_padded = math.ceil(header["ft_dim"] / 32) * 32
        self.hidden1_padded = math.ceil(header["hidden1_dim"] / 32) * 32
        self.hidden2_padded = math.ceil(header["hidden2_dim"] / 32) * 32

    @classmethod
    def from_file(cls, name: str, path: Path) -> "PatternScoreNNUEEvaluator":
        with path.open("rb") as f:
            magic = f.read(8)
            if not magic.startswith(NNUE_MAGIC_PREFIX):
                raise ValueError(f"not an NNUE file: {path}")
            header_values = np.fromfile(f, dtype="<u4", count=10)
            reserved = np.fromfile(f, dtype="<u4", count=8)
            if header_values.size != 10 or reserved.size != 8:
                raise ValueError(f"broken NNUE header: {path}")
            keys = [
                "version",
                "input_features",
                "ft_dim",
                "hidden1_dim",
                "hidden2_dim",
                "n_phases",
                "ft_shift",
                "hidden1_shift",
                "hidden2_shift",
                "output_shift",
            ]
            header = {key: int(value) for key, value in zip(keys, header_values.tolist())}
            input_kind = int(reserved[0])
            ft_weight_bits = int(reserved[6])
            if header["n_phases"] != N_PHASES or input_kind not in (4, 5) or ft_weight_bits not in (8, 16):
                raise ValueError(f"unsupported pattern-score NNUE header: {path}")
            ft_dim = header["ft_dim"]
            hidden1_dim = header["hidden1_dim"]
            hidden2_dim = header["hidden2_dim"]
            post_input_padded = math.ceil(ft_dim / 32) * 32
            hidden1_padded = math.ceil(hidden1_dim / 32) * 32
            hidden2_padded = math.ceil(hidden2_dim / 32) * 32
            ft_bias = np.fromfile(f, dtype="<i2", count=ft_dim)
            if input_kind == 5:
                ft_weight_dtype = "<i2"
                ft_weight_shape = (N_PHASES, header["input_features"])
                ft_weight_count = N_PHASES * header["input_features"]
            else:
                ft_weight_dtype = np.int8 if ft_weight_bits == 8 else "<i2"
                ft_weight_shape = (header["input_features"], ft_dim)
                ft_weight_count = header["input_features"] * ft_dim
            ft_weight = np.fromfile(f, dtype=ft_weight_dtype, count=ft_weight_count).reshape(ft_weight_shape)
            hidden1_bias = np.fromfile(f, dtype="<i4", count=hidden1_dim)
            hidden1_weight = np.fromfile(f, dtype=np.int8, count=hidden1_dim * post_input_padded).reshape(hidden1_dim, post_input_padded)
            hidden2_bias = np.fromfile(f, dtype="<i4", count=hidden2_dim)
            hidden2_weight = np.fromfile(f, dtype=np.int8, count=hidden2_dim * hidden1_padded).reshape(hidden2_dim, hidden1_padded)
            output_bias = np.fromfile(f, dtype="<i4", count=N_PHASES)
            output_weight = np.fromfile(f, dtype=np.int8, count=N_PHASES * hidden2_padded).reshape(N_PHASES, hidden2_padded)
        if (
            ft_bias.size != ft_dim
            or hidden1_bias.size != hidden1_dim
            or hidden2_bias.size != hidden2_dim
            or output_bias.size != N_PHASES
        ):
            raise ValueError(f"broken pattern-score NNUE payload: {path}")
        return cls(
            name,
            path,
            header,
            reserved,
            ft_bias,
            ft_weight,
            hidden1_bias,
            hidden1_weight,
            hidden2_bias,
            hidden2_weight,
            output_bias,
            output_weight,
        )

    @staticmethod
    def _active_columns(reserved: np.ndarray) -> np.ndarray:
        mask_lo = int(reserved[3]) | (int(reserved[4]) << 32)
        mask_hi = int(reserved[5])
        active: list[int] = []
        for col in range(N_FEATURE_COLUMNS):
            if col < 64:
                if mask_lo & (1 << col):
                    active.append(col)
            else:
                if mask_hi & (1 << (col - 64)):
                    active.append(col)
        return np.array(active, dtype=np.int64)

    def _feature_ids(self, features: np.ndarray) -> np.ndarray:
        active = self.active_columns
        local = features[:, active].astype(np.uint32, copy=False)
        return local + COLUMNWISE_FEATURE_STARTS[active][None, :]

    def _ft_accumulate(self, features: np.ndarray, phases: np.ndarray) -> np.ndarray:
        ids = self._feature_ids(features)
        if self.input_kind == 5:
            phase_idx = phases.astype(np.int64, copy=False)
            acc = self.ft_bias[None, :] + self.ft_weight[phase_idx[:, None], ids].astype(np.int32, copy=False)
        else:
            acc = self.ft_bias[None, :] + self.ft_weight[ids].astype(np.int32, copy=False).sum(axis=1)
        return acc

    def predict(self, features: np.ndarray, phases: np.ndarray) -> np.ndarray:
        n = features.shape[0]
        acc = self._ft_accumulate(features, phases)
        post_input = np.zeros((n, self.post_input_padded), dtype=np.uint8)
        shifted = acc >> self.header["ft_shift"] if self.header["ft_shift"] > 0 else acc
        post_input[:, : self.header["ft_dim"]] = np.clip(shifted, 0, 127).astype(np.uint8)
        h1_raw = self.hidden1_bias[None, :] + post_input.astype(np.int32, copy=False) @ self.hidden1_weight.T
        hidden1 = np.zeros((n, self.hidden1_padded), dtype=np.uint8)
        h1_shifted = h1_raw >> self.header["hidden1_shift"] if self.header["hidden1_shift"] > 0 else h1_raw
        hidden1[:, : self.header["hidden1_dim"]] = np.clip(h1_shifted, 0, 127).astype(np.uint8)
        h2_raw = self.hidden2_bias[None, :] + hidden1.astype(np.int32, copy=False) @ self.hidden2_weight.T
        hidden2 = np.zeros((n, self.hidden2_padded), dtype=np.uint8)
        h2_shifted = h2_raw >> self.header["hidden2_shift"] if self.header["hidden2_shift"] > 0 else h2_raw
        hidden2[:, : self.header["hidden2_dim"]] = np.clip(h2_shifted, 0, 127).astype(np.uint8)
        phase_idx = phases.astype(np.int64, copy=False)
        raw = self.output_bias[phase_idx].copy()
        raw += (hidden2.astype(np.int32, copy=False) * self.output_weight[phase_idx]).sum(axis=1)
        raw = rounded_shift_signed(raw, self.header["output_shift"])
        return quantized_score_from_raw(raw)


def parse_dataset_specs(specs: Iterable[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"dataset spec must be name=record,record: {spec}")
        name, records_text = spec.split("=", 1)
        records = [int(x) for x in records_text.split(",") if x.strip()]
        if not name or not records:
            raise ValueError(f"invalid dataset spec: {spec}")
        result[name] = records
    return result


def parse_model_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"model spec must be name=path: {spec}")
        name, path_text = spec.split("=", 1)
        if not name or not path_text:
            raise ValueError(f"invalid model spec: {spec}")
        result.append((name, Path(path_text)))
    return result


def load_model(name: str, path: Path):
    with path.open("rb") as f:
        head = f.read(8)
    if head == FM_MAGIC:
        return FMEvaluator.from_file(name, path)
    if head.startswith(NNUE_MAGIC_PREFIX):
        return PatternScoreNNUEEvaluator.from_file(name, path)
    if path.suffix.lower() == ".egev2":
        return LinearEvaluator.from_file(name, path)
    return LinearEvaluator.from_file(name, path)


def data_paths_for_phase(data_root: Path, phase: int, records: list[int]) -> tuple[list[Path], list[Path]]:
    paths: list[Path] = []
    missing: list[Path] = []
    for record in records:
        path = data_root / str(phase) / f"{record}.dat"
        if path.exists():
            paths.append(path)
        else:
            missing.append(path)
    return paths, missing


def load_phase_data(paths: list[Path]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for path in paths:
        if path.stat().st_size % INDEXED_RECORD_BYTES != 0:
            raise ValueError(f"record file size is not divisible by {INDEXED_RECORD_BYTES}: {path}")
        arr = np.fromfile(path, dtype=INDEXED_DTYPE)
        arrays.append(arr)
    if not arrays:
        return np.empty(0, dtype=INDEXED_DTYPE)
    return np.concatenate(arrays)


def evaluate(
    data_root: Path,
    dataset_specs: dict[str, list[int]],
    models,
    batch_size: int,
    progress_interval_sec: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    overall: dict[tuple[str, str], Metrics] = {}
    data_counts: dict[str, list[int]] = {name: [] for name in dataset_specs}
    missing_files: dict[str, list[str]] = {name: [] for name in dataset_specs}
    phase_mismatch_count = 0
    start_ms = now_ms()
    last_progress = start_ms
    for dataset_name, records in dataset_specs.items():
        for phase in range(N_PHASES):
            paths, missing = data_paths_for_phase(data_root, phase, records)
            missing_files[dataset_name].extend(str(path) for path in missing)
            data = load_phase_data(paths)
            n = int(data.size)
            data_counts[dataset_name].append(n)
            if n == 0:
                continue
            expected_n_discs = phase + 4
            phase_mismatch_count += int(np.count_nonzero(data["n_discs"].astype(np.int32) != expected_n_discs))
            features = data["features"]
            target = data["score"].astype(np.float32, copy=False)
            phases = np.full(n, phase, dtype=np.int64)
            for model in models:
                metrics = Metrics()
                for begin in range(0, n, batch_size):
                    end = min(n, begin + batch_size)
                    pred = model.predict(features[begin:end], phases[begin:end])
                    metrics.update(pred, target[begin:end])
                rows.append(
                    {
                        "dataset": dataset_name,
                        "phase": phase,
                        "model": model.name,
                        "n": metrics.n,
                        "mse": metrics.mse,
                        "mae": metrics.mae,
                    }
                )
                overall.setdefault((dataset_name, model.name), Metrics()).sum_sq += metrics.sum_sq
                overall[(dataset_name, model.name)].sum_abs += metrics.sum_abs
                overall[(dataset_name, model.name)].n += metrics.n
            now = now_ms()
            if progress_interval_sec > 0 and now - last_progress >= progress_interval_sec * 1000:
                print(
                    f"progress dataset {dataset_name} phase {phase} elapsed_ms {now - start_ms}",
                    flush=True,
                )
                last_progress = now
    summary_rows = [
        {
            "dataset": dataset_name,
            "model": model_name,
            "n": metric.n,
            "mse": metric.mse,
            "mae": metric.mae,
        }
        for (dataset_name, model_name), metric in sorted(overall.items())
    ]
    metadata = {
        "data_root": str(data_root.resolve()),
        "datasets": dataset_specs,
        "data_counts": data_counts,
        "missing_files": missing_files,
        "phase_mismatch_count": phase_mismatch_count,
        "models": [
            {
                "name": model.name,
                "path": str(model.path.resolve()),
                "type": type(model).__name__,
            }
            for model in models
        ],
    }
    return rows, {"summary": summary_rows, "metadata": metadata}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "phase", "model", "n", "mse", "mae"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "n", "mse", "mae"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def model_color(model: str) -> str:
    colors = {
        "dim0": "#1f2937",
        "dim8_fm": "#2563eb",
        "nnue_first12_shared": "#dc2626",
        "nnue_first12_phase60": "#16a34a",
    }
    return colors.get(model, "#7c3aed")


def write_line_chart_svg(
    path: Path,
    title: str,
    series: dict[str, dict[int, float]],
    y_label: str,
    zero_baseline: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1100
    height = 640
    left = 82
    right = 250
    top = 58
    bottom = 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    phases = list(range(N_PHASES))
    values = [v for points in series.values() for v in points.values()]
    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    if zero_baseline:
        y_min = min(0.0, y_min)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    padding = (y_max - y_min) * 0.06
    y_min -= padding
    y_max += padding

    def x_of(phase: int) -> float:
        return left + plot_w * (phase / 59.0)

    def y_of(value: float) -> float:
        return top + plot_h * ((y_max - value) / (y_max - y_min))

    def fmt(value: float) -> str:
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>',
    ]
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = y_of(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" font-family="Arial, sans-serif" font-size="12" '
            f'text-anchor="end" fill="#374151">{fmt(value)}</text>'
        )
    for phase in range(0, N_PHASES, 10):
        x = x_of(phase)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24}" font-family="Arial, sans-serif" font-size="12" '
            f'text-anchor="middle" fill="#374151">{phase}</text>'
        )
    if y_min < 0 < y_max:
        y = y_of(0.0)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#6b7280" stroke-dasharray="5,4"/>')
    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" font-family="Arial, sans-serif" '
        f'font-size="14" text-anchor="middle" fill="#111827">phase</text>'
    )
    parts.append(
        f'<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22 {top + plot_h / 2:.2f})" '
        f'font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#111827">{html.escape(y_label)}</text>'
    )

    legend_x = left + plot_w + 28
    legend_y = top + 8
    for idx, (model, points) in enumerate(series.items()):
        color = model_color(model)
        ordered = [(phase, points[phase]) for phase in phases if phase in points]
        if len(ordered) >= 2:
            polyline = " ".join(f"{x_of(phase):.2f},{y_of(value):.2f}" for phase, value in ordered)
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            for phase, value in ordered[::5]:
                parts.append(f'<circle cx="{x_of(phase):.2f}" cy="{y_of(value):.2f}" r="2.2" fill="{color}"/>')
        y = legend_y + idx * 26
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{legend_x + 34}" y="{y + 4}" font-family="Arial, sans-serif" font-size="13" '
            f'fill="#111827">{html.escape(model)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_graphs(out_dir: Path, phase_rows: list[dict[str, object]]) -> list[str]:
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in phase_rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    written: list[str] = []
    for dataset, rows in sorted(by_dataset.items()):
        for metric in ("mae", "mse"):
            series: dict[str, dict[int, float]] = {}
            for row in rows:
                series.setdefault(str(row["model"]), {})[int(row["phase"])] = float(row[metric])
            filename = f"{dataset}_{metric}_by_phase.svg"
            write_line_chart_svg(
                out_dir / filename,
                f"{dataset}: {metric.upper()} by phase",
                series,
                metric.upper(),
                zero_baseline=True,
            )
            written.append(filename)
        if any(str(row["model"]) == "dim0" for row in rows):
            dim0 = {int(row["phase"]): float(row["mae"]) for row in rows if str(row["model"]) == "dim0"}
            delta_series: dict[str, dict[int, float]] = {}
            for row in rows:
                model = str(row["model"])
                if model == "dim0":
                    continue
                phase = int(row["phase"])
                if phase in dim0:
                    delta_series.setdefault(model, {})[phase] = float(row["mae"]) - dim0[phase]
            filename = f"{dataset}_mae_delta_vs_dim0.svg"
            write_line_chart_svg(
                out_dir / filename,
                f"{dataset}: MAE difference from dim0",
                delta_series,
                "MAE - dim0 MAE",
                zero_baseline=False,
            )
            written.append(filename)
    return written


def write_markdown_report(path: Path, phase_rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = summary["summary"]
    metadata = summary["metadata"]
    worst_rows = sorted(phase_rows, key=lambda row: float(row["mae"]), reverse=True)[:20]
    with path.open("w", encoding="utf-8") as f:
        f.write("# 高精度テスト集合に対するフェーズ別誤差\n\n")
        f.write("## 目的\n")
        f.write("ランダム局面集合と最善手筋に近い引き分け局面集合で、評価関数候補のMSEとMAEをフェーズごとに確認する。\n\n")
        f.write("## 使用データ\n")
        f.write(f"- データルート: `{metadata['data_root']}`\n")
        for name, records in metadata["datasets"].items():
            total = sum(metadata["data_counts"][name])
            missing = metadata["missing_files"].get(name, [])
            f.write(f"- `{name}`: 要求records {records}, 実際に読めた合計 {total} 局面\n")
            if missing:
                shown = ", ".join(f"`{x}`" for x in missing[:3])
                suffix = " ほか" if len(missing) > 3 else ""
                f.write(f"  - 見つからなかったファイル: {len(missing)} 件。例: {shown}{suffix}\n")
        f.write(f"- `n_discs` とフェーズディレクトリの不一致数: {metadata['phase_mismatch_count']}\n\n")
        f.write("## 評価対象\n")
        for model in metadata["models"]:
            f.write(f"- `{model['name']}`: `{model['path']}` ({model['type']})\n")
        graphs = summary.get("graphs", [])
        if graphs:
            f.write("\n## グラフ\n")
            for graph in graphs:
                f.write(f"![{graph}]({graph})\n\n")
        f.write("\n## 全体集計\n")
        f.write("| データ集合 | 評価関数 | 局面数 | MSE | MAE |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['dataset']} | {row['model']} | {row['n']} | "
                f"{float(row['mse']):.6f} | {float(row['mae']):.6f} |\n"
            )
        f.write("\n## MAEが大きいフェーズ上位20件\n")
        f.write("| データ集合 | フェーズ | 評価関数 | 局面数 | MSE | MAE |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in worst_rows:
            f.write(
                f"| {row['dataset']} | {row['phase']} | {row['model']} | {row['n']} | "
                f"{float(row['mse']):.6f} | {float(row['mae']):.6f} |\n"
            )
        f.write("\n## 全フェーズの結果\n")
        f.write("全フェーズの数値は同じディレクトリの `phase_metrics.csv` に保存した。\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="E:/egaroucid_data/train_data/bin_data/20241125_1")
    parser.add_argument("--dataset", action="append", default=["random=166,215", "drawline=167"])
    parser.add_argument(
        "--model",
        action="append",
        default=[
            "dim0=bin/resources/eval.egev2",
            "dim8_fm=bin/resources/eval_dim8_fm.egevfm",
            "nnue_first12_shared=model/20260726_48_nnue_pattern_score_first12_ps48_32_records223plus_train10m_val1m_e80_lr001_ft16/eval_nnue_pattern_score_first12_ps48_32.egevnnue",
            "nnue_first12_phase60=model/20260726_52_nnue_pattern_score_first12_phase60_from_shared_ps48_32_records223plus_train10m_val1m_e20_lr0001_ft16/eval_nnue_pattern_score_first12_phase60_ps48_32.egevnnue",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--out-dir", default="src/tools/evaluation/report/20260726_high_precision_phase_metrics")
    parser.add_argument("--progress-interval-sec", type=float, default=30.0)
    args = parser.parse_args()

    dataset_specs = parse_dataset_specs(args.dataset)
    model_specs = parse_model_specs(args.model)
    models = []
    for name, path in model_specs:
        print(f"loading_model {name} {path}", flush=True)
        models.append(load_model(name, path))
    rows, summary = evaluate(
        Path(args.data_root),
        dataset_specs,
        models,
        args.batch_size,
        args.progress_interval_sec,
    )
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "phase_metrics.csv", rows)
    write_summary_csv(out_dir / "summary_metrics.csv", summary["summary"])
    summary["graphs"] = write_graphs(out_dir, rows)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"phase_metrics": rows, **summary}, f, ensure_ascii=False, indent=2)
    write_markdown_report(out_dir / "README.md", rows, summary)
    print(f"wrote {out_dir / 'phase_metrics.csv'}", flush=True)
    print(f"wrote {out_dir / 'summary_metrics.csv'}", flush=True)
    print(f"wrote {out_dir / 'README.md'}", flush=True)
    for row in summary["summary"]:
        print(
            f"summary dataset {row['dataset']} model {row['model']} n {row['n']} "
            f"mse {float(row['mse']):.6f} mae {float(row['mae']):.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
