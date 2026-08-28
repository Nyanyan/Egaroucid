#!/usr/bin/env python3
"""Reproducible measurements for the phase evaluation training data.

This script is intentionally read-only with respect to the training corpus.  It
uses the optimizer log belonging to the deployed ``eval.egev2`` as the source
of truth for the historical training run, and independently executes the
current ``eval_optimizer_phase.py`` selection code with a mocked optimizer in
order to detect configuration drift.

The two binary layouts used here are defined by:

* ``data_board_to_idx.cpp``: 136-byte indexed records
* ``expand_transcript*.cpp``: 19-byte board records

Commands used for the 2026-08-28 investigation are recorded in the report
directory's ``commands.txt``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import re
import runpy
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, TypeVar

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = REPO_ROOT / "model" / "20260621_1_afterrand16_used_dev-eval"
DEFAULT_EVAL = REPO_ROOT / "bin" / "resources" / "eval.egev2"
DEFAULT_PHASES = tuple(range(28, 47))
TARGET_SAMPLE_PHASES = (30, 35, 40, 44)
INDEXED_RECORD_SIZE = 136
BOARD_RECORD_SIZE = 19
OPTIMIZER_MAX_RECORDS = 200_000_000
SEED = 20260828

INDEXED_DTYPE = np.dtype(
    [
        ("n_discs", "<i2"),
        ("player_color", "<i2"),
        ("features", "<u2", (65,)),
        ("score", "<i2"),
    ],
    align=False,
)
BOARD_DTYPE = np.dtype(
    [
        ("player", "<u8"),
        ("opponent", "<u8"),
        ("player_color", "i1"),
        ("policy", "i1"),
        ("score", "i1"),
    ],
    align=False,
)
assert INDEXED_DTYPE.itemsize == INDEXED_RECORD_SIZE
assert BOARD_DTYPE.itemsize == BOARD_RECORD_SIZE

SCORE_MIN = -32768
SCORE_MAX = 32767
SCORE_OFFSET = -SCORE_MIN
SCORE_HIST_SIZE = SCORE_MAX - SCORE_MIN + 1
PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

BANDS = (
    ("-64..-21", -64, -21),
    ("-20..-11", -20, -11),
    ("-10..-5", -10, -5),
    ("-4..-1", -4, -1),
    ("0", 0, 0),
    ("+1..+4", 1, 4),
    ("+5..+10", 5, 10),
    ("+11..+20", 11, 20),
    ("+21..+64", 21, 64),
)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def parse_phases(text: str) -> tuple[int, ...]:
    result: set[int] = set()
    for field in text.split(","):
        field = field.strip()
        if not field:
            continue
        if "-" in field:
            lo, hi = (int(value) for value in field.split("-", 1))
            result.update(range(lo, hi + 1))
        else:
            result.add(int(field))
    return tuple(sorted(result))


def load_data_range(path: Path) -> tuple[dict[str, list[int]], set[int]]:
    spec = importlib.util.spec_from_file_location("training_bias_data_range", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.board_n_moves, set(module.use_all_depth_data)


class _FakeStdout:
    def readline(self) -> bytes:
        return b"selection_probe\n"


class _FakePopen:
    def __init__(self, command: Sequence[str], **_: object) -> None:
        self.command = list(command)
        self.stdout = _FakeStdout()


def current_optimizer_ids(path: Path, phase: int) -> list[int]:
    """Execute the real selector while replacing only the optimizer process."""

    old_argv = sys.argv[:]
    old_path = sys.path[:]
    old_popen = subprocess.Popen
    captured = io.StringIO()
    try:
        sys.argv = [str(path), str(phase), "0", "0"]
        sys.path.insert(0, str(path.parent))
        subprocess.Popen = _FakePopen  # type: ignore[assignment]
        with contextlib.redirect_stdout(captured):
            runpy.run_path(str(path), run_name=f"__training_bias_phase_{phase}__")
    finally:
        subprocess.Popen = old_popen  # type: ignore[assignment]
        sys.argv = old_argv
        sys.path[:] = old_path
    match = re.search(r"train_data_nums=(\[[^\n]*\])", captured.getvalue())
    if not match:
        raise RuntimeError(f"selection output not found for phase {phase}")
    values = ast.literal_eval(match.group(1))
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise RuntimeError(f"invalid selection for phase {phase}: {values!r}")
    return values


def model_optimizer_log(path: Path) -> tuple[dict[int, list[int]], dict[int, dict[str, float]]]:
    ids: dict[int, list[int]] = {}
    metrics: dict[int, dict[str, float]] = {}
    pattern = re.compile(
        r"^phase (?P<phase>\d+).*?n_train_data (?P<train>\d+) "
        r"n_val_data (?P<val>\d+) score_avg (?P<score>[-+0-9.eE]+) "
        r"n_loop (?P<loops>\d+) MSE (?P<mse>[-+0-9.eE]+) "
        r"MAE (?P<mae>[-+0-9.eE]+) val_MSE (?P<vmse>[-+0-9.eE]+) "
        r"val_MAE (?P<vmae>[-+0-9.eE]+).*?train_data_nums=(?P<ids>\[[^\n]*\])$"
    )
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            match = pattern.match(raw_line.strip())
            if not match:
                continue
            phase = int(match.group("phase"))
            ids[phase] = list(ast.literal_eval(match.group("ids")))
            metrics[phase] = {
                "n_train_data": int(match.group("train")),
                "n_val_data": int(match.group("val")),
                "score_avg": float(match.group("score")),
                "n_loop": int(match.group("loops")),
                "train_mse": float(match.group("mse")),
                "train_mae": float(match.group("mae")),
                "val_mse": float(match.group("vmse")),
                "val_mae": float(match.group("vmae")),
            }
    if not ids:
        raise RuntimeError(f"no optimizer records parsed from {path}")
    return ids, metrics


def source_metadata(data_id: int, board_n_moves: dict[str, list[int]]) -> dict[str, object]:
    mn, mx = board_n_moves[str(data_id)]
    result: dict[str, object] = {
        "data_id": data_id,
        "available_phase_min": mn,
        "available_phase_max": mx,
        "random_moves": None,
        "starting_discs": None,
        "source_category": "other_or_unconfirmed",
        "generation_method": "not reconstructed from surviving files",
        "teacher_method": "terminal disc difference, signed for the side to move",
        "provenance_status": "unconfirmed",
    }
    if 259 <= data_id <= 310:
        result.update(
            random_moves=data_id - 251,
            starting_discs=4 + data_id - 251,
            source_category="egaroucid_vs_edax",
            generation_method=(
                "public training_data_v0002: random legal moves followed by "
                "Egaroucid 7.8.0 Level 11 vs Edax 4.5.5 Level 11"
            ),
            teacher_method="terminal disc difference of the completed game",
            provenance_status="release_documented_and_locally_counted",
        )
    elif 313 <= data_id <= 324:
        start_phase = 6 + (data_id - 313) // 2
        setup = "random_setup" if data_id % 2 else "random_setup_2"
        result.update(
            starting_discs=start_phase + 4,
            source_category=f"ggs_{setup}_starting_board",
            generation_method=(
                f"Egaroucid 7.8.1 Level 11 GGS game from a {setup} starting board; "
                "the surviving local manifest does not identify the opponent"
            ),
            teacher_method="terminal disc difference of the completed game",
            provenance_status="local_train_data_manifest_and_converter_confirmed",
        )
    elif data_id in (80, 311):
        result.update(
            source_category="first11_book",
            generation_method="first-11-ply book positions",
            teacher_method="book-derived stored score; not a terminal-game label",
            provenance_status="git_history_and_identical_file_size_checked",
        )
    elif 65 <= data_id <= 74:
        result.update(
            random_moves=data_id - 55,
            starting_discs=data_id - 51,
            source_category="egaroucid_selfplay",
            generation_method="Egaroucid 7.4.0 Level 11 self-play after random legal moves",
            provenance_status="local_train_data_manifest_confirmed",
        )
    elif data_id in {
        144: 13,
        145: 14,
        146: 15,
        147: 16,
        148: 17,
        149: 19,
        150: 25,
        151: 28,
        158: 20,
        159: 22,
        160: 23,
        161: 24,
        162: 26,
        163: 27,
        164: 29,
        165: 33,
    }:
        random_moves = {
            144: 13, 145: 14, 146: 15, 147: 16, 148: 17, 149: 19,
            150: 25, 151: 28, 158: 20, 159: 22, 160: 23, 161: 24,
            162: 26, 163: 27, 164: 29, 165: 33,
        }[data_id]
        result.update(
            random_moves=random_moves,
            starting_discs=random_moves + 4,
            source_category="egaroucid_selfplay",
            generation_method="Egaroucid 7.6.0 beta Level 11 self-play after random legal moves",
            provenance_status="local_train_data_manifest_confirmed",
        )
    elif 223 <= data_id <= 234:
        random_moves = data_id - 211
        result.update(
            random_moves=random_moves,
            starting_discs=random_moves + 4,
            source_category="egaroucid_selfplay",
            generation_method="Level 15 self-play after random legal moves",
            provenance_status="local_train_data_manifest_confirmed",
        )
    elif data_id in {20: 8, 21: 10, 24: 21, 25: 30, 29: 12, 30: 18, 31: 24, 82: 12}:
        random_moves = {20: 8, 21: 10, 24: 21, 25: 30, 29: 12, 30: 18, 31: 24, 82: 12}[data_id]
        result.update(
            random_moves=random_moves,
            starting_discs=random_moves + 4,
            source_category="legacy_random_start_games",
            generation_method="legacy Egaroucid self-play after a fixed number of random legal moves",
            provenance_status="local_train_data_manifest_confirmed",
        )
    elif data_id in (18, 19):
        result.update(
            source_category="legacy_mixed_random_start_games",
            generation_method="Egaroucid self-play after 10--19 random legal moves",
            provenance_status="local_train_data_manifest_confirms_range_not_per_record_count",
        )
    elif data_id == 38:
        result.update(
            source_category="legacy_mixed_random_start_games",
            generation_method="test games after 8, 9, 10, or 11 random legal moves",
            provenance_status="local_train_data_manifest_confirms_file_mapping_not_stored_in_indexed_data",
        )
    elif data_id == 214:
        result.update(
            source_category="book_start_selfplay",
            generation_method="Egaroucid 7.6.0 Level 11 games continued from first11_all positions",
            provenance_status="local_train_data_manifest_confirmed",
        )
    elif data_id == 37:
        result.update(
            source_category="book_derived",
            generation_method="book data",
            teacher_method="book-derived stored score; exact generation method not recovered",
            provenance_status="local_train_data_manifest_category_only",
        )
    elif data_id == 97:
        result.update(
            source_category="public_training_data_v0001",
            generation_method="public Egaroucid training_data release",
            provenance_status="release_documented_generation_mix_not_fully_recovered",
        )
    return result


def percentile_from_hist(hist: np.ndarray, probability: float) -> float:
    total = int(hist.sum())
    if total == 0:
        return math.nan
    target = probability * (total - 1)
    lo_rank = math.floor(target)
    hi_rank = math.ceil(target)
    cumulative = np.cumsum(hist, dtype=np.int64)
    lo_idx = int(np.searchsorted(cumulative, lo_rank + 1))
    hi_idx = int(np.searchsorted(cumulative, hi_rank + 1))
    lo = lo_idx + SCORE_MIN
    hi = hi_idx + SCORE_MIN
    return lo + (target - lo_rank) * (hi - lo)


def hist_stats(hist: np.ndarray) -> dict[str, object]:
    total = int(hist.sum())
    if total == 0:
        result: dict[str, object] = {
            "records": 0,
            "mean": math.nan,
            "stddev": math.nan,
            "positive_fraction": math.nan,
            "zero_fraction": math.nan,
            "negative_fraction": math.nan,
            "mean_abs_score": math.nan,
            "median": math.nan,
        }
        for probability in PERCENTILES:
            result[f"p{int(probability * 100):02d}"] = math.nan
        for name, _, _ in BANDS:
            result[f"band_{name}"] = 0
        return result
    values = np.arange(SCORE_MIN, SCORE_MAX + 1, dtype=np.float64)
    mean = float(np.dot(values, hist) / total)
    second = float(np.dot(values * values, hist) / total)
    result = {
        "records": total,
        "mean": mean,
        "stddev": math.sqrt(max(0.0, second - mean * mean)),
        "positive_fraction": float(hist[SCORE_OFFSET + 1 :].sum() / total),
        "zero_fraction": float(hist[SCORE_OFFSET] / total),
        "negative_fraction": float(hist[:SCORE_OFFSET].sum() / total),
        "mean_abs_score": float(np.dot(np.abs(values), hist) / total),
        "median": percentile_from_hist(hist, 0.5),
        "min_score": int(np.flatnonzero(hist)[0]) + SCORE_MIN,
        "max_score": int(np.flatnonzero(hist)[-1]) + SCORE_MIN,
    }
    for probability in PERCENTILES:
        result[f"p{int(probability * 100):02d}"] = percentile_from_hist(hist, probability)
    for name, lo, hi in BANDS:
        result[f"band_{name}"] = int(hist[lo + SCORE_OFFSET : hi + SCORE_OFFSET + 1].sum())
    return result


def iter_binary_chunks(path: Path, record_size: int, records_per_chunk: int = 1_000_000) -> Iterator[bytes]:
    chunk_bytes = record_size * records_per_chunk
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_bytes)
            if not chunk:
                break
            if len(chunk) % record_size:
                raise ValueError(f"partial record in {path}: {len(chunk)} bytes")
            yield chunk


def build_manifest_and_labels(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    bin_root = data_root / "train_data" / "bin_data" / "20241125_1"
    board_root = data_root / "train_data" / "board_data"
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = parse_phases(args.phases)

    board_n_moves, use_all_depth = load_data_range(EVAL_DIR / "data_range.py")
    historical_ids, optimizer_metrics = model_optimizer_log(model_dir / "opt_log.txt")
    current_ids = {
        phase: current_optimizer_ids(EVAL_DIR / "eval_optimizer_phase.py", phase)
        for phase in phases
    }
    missing_log_phases = sorted(set(phases) - set(historical_ids))
    if missing_log_phases:
        raise RuntimeError(f"model log lacks phases {missing_log_phases}")

    eval_path = Path(args.eval_file)
    model_eval = model_dir / "eval.egev2"
    eval_hash = sha256_file(eval_path)
    model_eval_hash = sha256_file(model_eval)
    if eval_hash != model_eval_hash:
        raise RuntimeError("deployed eval.egev2 does not match the selected model directory")

    distribution_rows: list[dict[str, object]] = []
    phase_records: list[dict[str, object]] = []
    input_files: list[dict[str, object]] = []
    overall_input_digest = hashlib.sha256()

    for phase in phases:
        ids = historical_ids[phase]
        total_count = 0
        file_entries: list[dict[str, object]] = []
        for data_id in ids:
            path = bin_root / str(phase) / f"{data_id}.dat"
            if not path.exists():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if size % INDEXED_RECORD_SIZE:
                raise ValueError(f"bad indexed file size: {path} = {size}")
            count = size // INDEXED_RECORD_SIZE
            file_entries.append({"data_id": data_id, "path": path, "bytes": size, "records": count})
            total_count += count
        actually_loaded = min(total_count, OPTIMIZER_MAX_RECORDS)
        remaining = actually_loaded
        phase_hist = np.zeros(SCORE_HIST_SIZE, dtype=np.int64)
        current_total = 0
        for entry in file_entries:
            path = entry["path"]
            assert isinstance(path, Path)
            allowed = min(int(entry["records"]), remaining)
            remaining -= allowed
            hist = np.zeros(SCORE_HIST_SIZE, dtype=np.int64)
            invalid_n_discs = 0
            invalid_player = 0
            digest = hashlib.sha256()
            seen = 0
            for chunk in iter_binary_chunks(path, INDEXED_RECORD_SIZE):
                digest.update(chunk)
                overall_input_digest.update(chunk)
                arr = np.frombuffer(chunk, dtype=INDEXED_DTYPE)
                if seen >= allowed:
                    seen += len(arr)
                    continue
                use = arr[: max(0, min(len(arr), allowed - seen))]
                seen += len(arr)
                if len(use):
                    invalid_n_discs += int(np.count_nonzero(use["n_discs"] != phase + 4))
                    invalid_player += int(np.count_nonzero((use["player_color"] != 0) & (use["player_color"] != 1)))
                    scores = use["score"].astype(np.int32, copy=False)
                    hist += np.bincount(scores + SCORE_OFFSET, minlength=SCORE_HIST_SIZE)
            if seen != int(entry["records"]):
                raise RuntimeError(f"short read for {path}: {seen} != {entry['records']}")
            phase_hist += hist
            loaded = int(hist.sum())
            current_total += loaded
            meta = source_metadata(int(entry["data_id"]), board_n_moves)
            after_random = None
            if meta["random_moves"] is not None:
                after_random = phase - int(meta["random_moves"])
            row = {
                "aggregation": "training_frequency",
                "phase": phase,
                "empties": 60 - phase,
                "data_id": int(entry["data_id"]),
                "source_category": meta["source_category"],
                "random_moves": meta["random_moves"],
                "moves_after_random": after_random,
                "share_of_phase": loaded / actually_loaded if actually_loaded else math.nan,
                **hist_stats(hist),
            }
            distribution_rows.append(row)
            input_entry = {
                **meta,
                "phase": phase,
                "empties": 60 - phase,
                "moves_after_random": after_random,
                "selected_by_model_log": True,
                "selected_by_current_script": int(entry["data_id"]) in current_ids[phase],
                "indexed_path": str(path),
                "indexed_bytes": int(entry["bytes"]),
                "file_records": int(entry["records"]),
                "loaded_records": loaded,
                "share_of_loaded_phase": loaded / actually_loaded if actually_loaded else math.nan,
                "sha256": digest.hexdigest(),
                "score_histogram_sha256": hashlib.sha256(
                    hist.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "invalid_n_discs_records": invalid_n_discs,
                "invalid_player_color_records": invalid_player,
                "board_data_dir": str(board_root / f"records{entry['data_id']}"),
                "board_data_exists": (board_root / f"records{entry['data_id']}").is_dir(),
            }
            input_files.append(input_entry)
        if current_total != actually_loaded:
            raise RuntimeError(f"phase {phase}: counted {current_total}, expected {actually_loaded}")
        log_total = int(optimizer_metrics[phase]["n_train_data"] + optimizer_metrics[phase]["n_val_data"])
        phase_row = {
            "aggregation": "training_frequency",
            "phase": phase,
            "empties": 60 - phase,
            "data_id": "ALL",
            "source_category": "ALL",
            "random_moves": "",
            "moves_after_random": "",
            "share_of_phase": 1.0,
            **hist_stats(phase_hist),
        }
        distribution_rows.append(phase_row)
        phase_records.append(
            {
                "phase": phase,
                "empties": 60 - phase,
                "model_data_ids": ids,
                "current_script_data_ids": current_ids[phase],
                "selection_matches": ids == current_ids[phase],
                "selected_file_records": total_count,
                "optimizer_cap": OPTIMIZER_MAX_RECORDS,
                "actually_loaded_records": actually_loaded,
                "logged_train_records": int(optimizer_metrics[phase]["n_train_data"]),
                "logged_validation_records": int(optimizer_metrics[phase]["n_val_data"]),
                "logged_total_records": log_total,
                "count_matches_log": log_total == actually_loaded,
                **optimizer_metrics[phase],
            }
        )

    distribution_path = output_dir / "label_distribution.csv"
    fieldnames: list[str] = []
    for row in distribution_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with distribution_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(distribution_rows)

    manifest = {
        "schema_version": 1,
        "generated_at_local": __import__("datetime").datetime.now().astimezone().isoformat(),
        "sampling_seed": SEED,
        "data_root": str(data_root),
        "bin_root": str(bin_root),
        "board_root": str(board_root),
        "phases": list(phases),
        "phase_semantics": "phase = discs_on_board - 4 = 60 - empty_squares",
        "record_layouts": {"indexed_bytes": INDEXED_RECORD_SIZE, "board_bytes": BOARD_RECORD_SIZE},
        "optimizer_max_records": OPTIMIZER_MAX_RECORDS,
        "deployed_eval": {
            "path": str(eval_path),
            "sha256": eval_hash,
            "bytes": eval_path.stat().st_size,
        },
        "matched_model": {
            "path": str(model_dir),
            "eval_path": str(model_eval),
            "eval_sha256": model_eval_hash,
            "optimizer_log": str(model_dir / "opt_log.txt"),
            "info": str(model_dir / "info.txt"),
        },
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "eval_blob": git_output("rev-parse", "HEAD:bin/resources/eval.egev2"),
            "eval_last_commit": git_output("log", "-1", "--format=%H %cI %s", "--", "bin/resources/eval.egev2"),
        },
        "use_all_depth_data": sorted(use_all_depth),
        "phase_records": phase_records,
        "input_files": input_files,
        "ordered_input_stream_sha256": overall_input_digest.hexdigest(),
        "notes": [
            "The ordered stream hash includes every selected indexed file in phase/data-ID order.",
            "The historical model log, not comments in data_range.py, determines deployed-weight inputs.",
            "Current selector execution is recorded separately and must match before claiming current reproducibility.",
        ],
    }
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "sampling_seed.txt").write_text(f"{SEED}\n", encoding="ascii")
    environment = {
        "generated_at_local": manifest["generated_at_local"],
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "processor": platform.processor(),
        "EGAROUCID_DATA": os.environ.get("EGAROUCID_DATA"),
        "git_head": manifest["git"]["head"],
        "git_status_before_analysis": git_output("status", "--short"),
        "files": {
            str(eval_path): eval_hash,
            str(model_eval): model_eval_hash,
            str(EVAL_DIR / "eval_optimizer_phase.py"): sha256_file(EVAL_DIR / "eval_optimizer_phase.py"),
            str(EVAL_DIR / "data_range.py"): sha256_file(EVAL_DIR / "data_range.py"),
            str(EVAL_DIR / "eval_optimizer_cuda.cu"): sha256_file(EVAL_DIR / "eval_optimizer_cuda.cu"),
        },
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {distribution_path}")
    print(f"wrote {output_dir / 'data_manifest.json'}")


def splitmix64(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=True)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def popcount_u64(values: np.ndarray, byte_lut: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.uint64)
    return byte_lut[contiguous.view(np.uint8).reshape(-1, 8)].sum(axis=1, dtype=np.uint8)


def score_band(score: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= score <= hi:
            return name
    return "outside_terminal_range"


@dataclass
class Candidate:
    priority: int
    phase: int
    data_id: int
    source_file: str
    source_record: int
    game_index: int
    player: int
    opponent: int
    player_color: int
    policy: int
    teacher: int


T = TypeVar("T")


def deterministic_seeded_order(
    values: Iterable[T], seed: int, namespace: str
) -> list[T]:
    """Return a stable seed-dependent order without relying on input order.

    Hash ordering is used instead of ``random.shuffle`` so that the result is
    reproducible across Python versions.  Callers pass unique values.
    """

    def order_key(value: T) -> tuple[bytes, str]:
        token = f"{seed}\0{namespace}\0{value!r}".encode("utf-8")
        return hashlib.sha256(token).digest(), repr(value)

    return sorted(values, key=order_key)


def _candidate_identity(candidate: Candidate) -> tuple[int, int, str, int]:
    return (
        candidate.phase,
        candidate.data_id,
        candidate.source_file,
        candidate.source_record,
    )


def select_balanced_stratified_candidates(
    strata: Mapping[tuple[int, int, str], Sequence[Candidate]],
    phase: int,
    target: int,
    seed: int = SEED,
) -> list[Candidate]:
    """Balance a sample over data IDs, then over score bands within each ID.

    Each outer round takes at most one record from every still-active data ID.
    Within a data ID, non-empty teacher-score bands are traversed in a seeded
    round-robin.  Seeded hash orders prevent a truncated final round from
    always favoring low-numbered IDs or the first score bands.
    """

    if target <= 0:
        return []

    pools: dict[int, dict[str, list[Candidate]]] = defaultdict(dict)
    for (candidate_phase, data_id, band), values in strata.items():
        if candidate_phase != phase or not values:
            continue
        ordered = sorted(
            values,
            key=lambda item: (
                item.priority,
                item.source_file,
                item.source_record,
            ),
        )
        for candidate in ordered:
            if candidate.phase != phase or candidate.data_id != data_id:
                raise ValueError(
                    "stratum key disagrees with candidate phase/data ID: "
                    f"key=({candidate_phase},{data_id},{band}) "
                    f"candidate=({candidate.phase},{candidate.data_id})"
                )
            if score_band(candidate.teacher) != band:
                raise ValueError(
                    "stratum key disagrees with candidate teacher band: "
                    f"key={band} candidate={score_band(candidate.teacher)}"
                )
        pools[data_id][band] = ordered

    if not pools:
        return []

    id_order = deterministic_seeded_order(
        pools, seed, f"phase={phase}:data-id-order"
    )
    band_orders = {
        data_id: deterministic_seeded_order(
            pools[data_id], seed, f"phase={phase}:data-id={data_id}:band-order"
        )
        for data_id in id_order
    }
    positions: dict[tuple[int, str], int] = {
        (data_id, band): 0
        for data_id, bands in pools.items()
        for band in bands
    }
    next_band: dict[int, int] = {data_id: 0 for data_id in id_order}

    def has_remaining(data_id: int) -> bool:
        return any(
            positions[(data_id, band)] < len(pools[data_id][band])
            for band in band_orders[data_id]
        )

    selected: list[Candidate] = []
    active_ids = list(id_order)
    while active_ids and len(selected) < target:
        added_in_round = False
        next_active_ids: list[int] = []
        for data_id in active_ids:
            bands = band_orders[data_id]
            start = next_band[data_id]
            chosen_band_index: int | None = None
            for offset in range(len(bands)):
                band_index = (start + offset) % len(bands)
                band = bands[band_index]
                position = positions[(data_id, band)]
                if position < len(pools[data_id][band]):
                    selected.append(pools[data_id][band][position])
                    positions[(data_id, band)] = position + 1
                    next_band[data_id] = (band_index + 1) % len(bands)
                    chosen_band_index = band_index
                    added_in_round = True
                    break
            if chosen_band_index is None:
                continue
            if has_remaining(data_id):
                next_active_ids.append(data_id)
            if len(selected) >= target:
                break
        if not added_in_round:
            break
        active_ids = next_active_ids
    return selected


def validate_stratified_coverage(
    strata: Mapping[tuple[int, int, str], Sequence[Candidate]],
    phase: int,
    target: int,
    selected: Sequence[Candidate],
) -> dict[str, object]:
    """Validate and summarize the two-level round-robin coverage contract."""

    available_by_id: dict[int, dict[str, int]] = defaultdict(dict)
    available_candidates = 0
    for (candidate_phase, data_id, band), values in strata.items():
        if candidate_phase != phase or not values:
            continue
        available_by_id[data_id][band] = len(values)
        available_candidates += len(values)

    eligible_ids = set(available_by_id)
    selected_by_id: Counter[int] = Counter()
    selected_by_id_band: Counter[tuple[int, str]] = Counter()
    identities: set[tuple[int, int, str, int]] = set()
    for candidate in selected:
        if candidate.phase != phase:
            raise RuntimeError(
                f"phase {phase}: selected candidate from phase {candidate.phase}"
            )
        band = score_band(candidate.teacher)
        if band not in available_by_id.get(candidate.data_id, {}):
            raise RuntimeError(
                f"phase {phase}: selected candidate outside an available stratum: "
                f"data_id={candidate.data_id} band={band}"
            )
        identity = _candidate_identity(candidate)
        if identity in identities:
            raise RuntimeError(
                f"phase {phase}: duplicate record selected: {identity}"
            )
        identities.add(identity)
        selected_by_id[candidate.data_id] += 1
        selected_by_id_band[(candidate.data_id, band)] += 1

    expected_rows = min(max(target, 0), available_candidates)
    if len(selected) != expected_rows:
        raise RuntimeError(
            f"phase {phase}: stratified rows={len(selected)}, expected={expected_rows}"
        )
    expected_id_coverage = min(expected_rows, len(eligible_ids))
    if len(selected_by_id) != expected_id_coverage:
        raise RuntimeError(
            f"phase {phase}: covered data IDs={len(selected_by_id)}, "
            f"expected={expected_id_coverage}; outer round-robin is not balanced"
        )
    if target >= len(eligible_ids) and set(selected_by_id) != eligible_ids:
        missing = sorted(eligible_ids - set(selected_by_id))
        raise RuntimeError(f"phase {phase}: eligible data IDs not covered: {missing}")

    for data_id, selected_count in selected_by_id.items():
        available_bands = set(available_by_id[data_id])
        selected_bands = {
            band
            for (selected_id, band), count in selected_by_id_band.items()
            if selected_id == data_id and count
        }
        expected_band_coverage = min(selected_count, len(available_bands))
        if len(selected_bands) != expected_band_coverage:
            raise RuntimeError(
                f"phase {phase} data ID {data_id}: covered bands="
                f"{len(selected_bands)}, expected={expected_band_coverage}; "
                "inner round-robin is not balanced"
            )

    selected_ids = set(selected_by_id)
    return {
        "phase": phase,
        "target_rows": target,
        "available_reservoir_candidates": available_candidates,
        "available_strata": sum(len(bands) for bands in available_by_id.values()),
        "eligible_data_id_count": len(eligible_ids),
        "eligible_data_ids": sorted(eligible_ids),
        "selected_rows": len(selected),
        "selected_data_id_count": len(selected_ids),
        "selected_data_ids": sorted(selected_ids),
        "unselected_eligible_data_ids": sorted(eligible_ids - selected_ids),
        "selected_rows_by_data_id": {
            str(data_id): selected_by_id[data_id]
            for data_id in sorted(selected_by_id)
        },
        "selected_rows_by_data_id_and_band": {
            f"{data_id}:{band}": selected_by_id_band[(data_id, band)]
            for data_id, band in sorted(selected_by_id_band)
        },
    }


def merge_lowest(existing: list[Candidate], additions: list[Candidate], limit: int) -> list[Candidate]:
    merged = existing + additions
    if len(merged) <= limit:
        return merged
    merged.sort(key=lambda item: item.priority)
    return merged[:limit]


def candidates_from_indices(
    arr: np.ndarray,
    indices: np.ndarray,
    priorities: np.ndarray,
    phase: int,
    data_id: int,
    source_file: Path,
    record_base: int,
    game_indices: np.ndarray,
) -> list[Candidate]:
    result: list[Candidate] = []
    for idx in indices.tolist():
        record = arr[idx]
        result.append(
            Candidate(
                priority=int(priorities[idx]),
                phase=phase,
                data_id=data_id,
                source_file=str(source_file),
                source_record=record_base + idx,
                game_index=int(game_indices[idx]),
                player=int(record["player"]),
                opponent=int(record["opponent"]),
                player_color=int(record["player_color"]),
                policy=int(record["policy"]),
                teacher=int(record["score"]),
            )
        )
    return result


def relative_board_string(player: int, opponent: int) -> str:
    cells = []
    for bit in range(63, -1, -1):
        mask = 1 << bit
        if player & mask:
            cells.append("X")
        elif opponent & mask:
            cells.append("O")
        else:
            cells.append("-")
    return "".join(cells) + " X"


def build_samples(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    board_root = data_root / "train_data" / "board_data"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    historical_ids, _ = model_optimizer_log(model_dir / "opt_log.txt")
    board_n_moves, _ = load_data_range(EVAL_DIR / "data_range.py")
    targets = {
        30: args.phase30,
        35: args.phase35,
        36: args.phase36,
        40: args.phase40,
        44: args.phase44,
    }
    phases = tuple(phase for phase, count in targets.items() if count > 0)
    ids_to_phases: dict[int, list[int]] = defaultdict(list)
    for phase in phases:
        for data_id in historical_ids[phase]:
            ids_to_phases[data_id].append(phase)

    frequency: dict[int, list[Candidate]] = {phase: [] for phase in phases}
    stratified: dict[tuple[int, int, str], list[Candidate]] = defaultdict(list)
    raw_counts: Counter[tuple[int, int]] = Counter()
    raw_hists: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(SCORE_HIST_SIZE, dtype=np.int64)
    )
    byte_lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
    scanned_bytes = 0

    for data_id in sorted(ids_to_phases):
        directory = board_root / f"records{data_id}"
        if not directory.is_dir():
            print(f"warning: missing board directory {directory}", file=sys.stderr)
            continue
        phase_set = set(ids_to_phases[data_id])
        for source_file in sorted(directory.glob("*.dat"), key=lambda path: path.name):
            size = source_file.stat().st_size
            if size % BOARD_RECORD_SIZE:
                raise ValueError(f"bad board record file size: {source_file} = {size}")
            record_base = 0
            game_base = 0
            previous_n_discs: int | None = None
            file_salt = np.uint64(
                int.from_bytes(
                    hashlib.sha256(str(source_file.relative_to(board_root)).encode("utf-8")).digest()[:8],
                    "little",
                )
            )
            with source_file.open("rb") as source:
                while True:
                    chunk = source.read(BOARD_RECORD_SIZE * args.chunk_records)
                    if not chunk:
                        break
                    scanned_bytes += len(chunk)
                    if len(chunk) % BOARD_RECORD_SIZE:
                        raise ValueError(f"partial board record in {source_file}")
                    arr = np.frombuffer(chunk, dtype=BOARD_DTYPE)
                    occupied = arr["player"] | arr["opponent"]
                    n_discs = popcount_u64(occupied, byte_lut).astype(np.int16)
                    starts = np.zeros(len(arr), dtype=np.int64)
                    if len(arr):
                        if previous_n_discs is not None and int(n_discs[0]) <= previous_n_discs:
                            starts[0] = 1
                        if len(arr) > 1:
                            starts[1:] = n_discs[1:] <= n_discs[:-1]
                    game_indices = game_base + np.cumsum(starts, dtype=np.int64)
                    if len(arr):
                        game_base = int(game_indices[-1])
                        previous_n_discs = int(n_discs[-1])
                    absolute = np.arange(record_base, record_base + len(arr), dtype=np.uint64)
                    salt = (
                        np.uint64(SEED)
                        ^ (np.uint64(data_id) << np.uint64(32))
                        ^ file_salt
                    )
                    priorities = splitmix64(absolute ^ salt)
                    for phase in phase_set:
                        selected = np.flatnonzero(n_discs == phase + 4)
                        if not len(selected):
                            continue
                        key = (phase, data_id)
                        raw_counts[key] += len(selected)
                        scores = arr["score"][selected].astype(np.int32, copy=False)
                        raw_hists[key] += np.bincount(scores + SCORE_OFFSET, minlength=SCORE_HIST_SIZE)

                        keep = min(targets[phase], len(selected))
                        if keep:
                            local_order = np.argpartition(priorities[selected], keep - 1)[:keep]
                            picked = selected[local_order]
                            additions = candidates_from_indices(
                                arr,
                                picked,
                                priorities,
                                phase,
                                data_id,
                                source_file,
                                record_base,
                                game_indices,
                            )
                            frequency[phase] = merge_lowest(frequency[phase], additions, targets[phase])

                        for band_name, lo, hi in BANDS:
                            in_band = selected[(scores >= lo) & (scores <= hi)]
                            if not len(in_band):
                                continue
                            keep_band = min(args.stratum_reservoir, len(in_band))
                            local_order = np.argpartition(priorities[in_band], keep_band - 1)[:keep_band]
                            picked = in_band[local_order]
                            additions = candidates_from_indices(
                                arr,
                                picked,
                                priorities,
                                phase,
                                data_id,
                                source_file,
                                record_base,
                                game_indices,
                            )
                            stratum_key = (phase, data_id, band_name)
                            stratified[stratum_key] = merge_lowest(
                                stratified[stratum_key], additions, args.stratum_reservoir
                            )
                    record_base += len(arr)
            if record_base != size // BOARD_RECORD_SIZE:
                raise RuntimeError(f"short read for {source_file}")
            print(
                f"sample scan data_id={data_id} file={source_file.name} "
                f"records={record_base} total_GiB={scanned_bytes / (1 << 30):.2f}",
                flush=True,
            )

    final_rows: list[dict[str, object]] = []
    stratified_coverage: list[dict[str, object]] = []
    for phase in phases:
        meta_cache = {data_id: source_metadata(data_id, board_n_moves) for data_id in historical_ids[phase]}
        for sample_kind, candidates in (("frequency", frequency[phase]),):
            for index, candidate in enumerate(sorted(candidates, key=lambda item: item.priority)):
                meta = meta_cache[candidate.data_id]
                final_rows.append(
                    {
                        "sample_id": f"p{phase}_{sample_kind}_{index:05d}",
                        "sample_kind": sample_kind,
                        "phase": phase,
                        "empties": 60 - phase,
                        "data_id": candidate.data_id,
                        "source_category": meta["source_category"],
                        "random_moves": meta["random_moves"],
                        "moves_after_random": (
                            phase - int(meta["random_moves"]) if meta["random_moves"] is not None else ""
                        ),
                        "teacher_band": score_band(candidate.teacher),
                        "teacher": candidate.teacher,
                        "player_color": candidate.player_color,
                        "policy": candidate.policy,
                        "source_file": candidate.source_file,
                        "source_record": candidate.source_record,
                        "game_index_in_file": candidate.game_index,
                        "priority": candidate.priority,
                        "board": relative_board_string(candidate.player, candidate.opponent),
                    }
                )

        balanced = select_balanced_stratified_candidates(
            stratified, phase, targets[phase], SEED
        )
        coverage = validate_stratified_coverage(
            stratified, phase, targets[phase], balanced
        )
        stratified_coverage.append(coverage)
        print(
            f"stratified coverage phase={phase} rows={coverage['selected_rows']} "
            f"data_ids={coverage['selected_data_id_count']}/"
            f"{coverage['eligible_data_id_count']} "
            f"strata={coverage['available_strata']}",
            flush=True,
        )
        for index, candidate in enumerate(balanced):
            meta = meta_cache[candidate.data_id]
            final_rows.append(
                {
                    "sample_id": f"p{phase}_stratified_{index:05d}",
                    "sample_kind": "stratified_data_id_teacher_band",
                    "phase": phase,
                    "empties": 60 - phase,
                    "data_id": candidate.data_id,
                    "source_category": meta["source_category"],
                    "random_moves": meta["random_moves"],
                    "moves_after_random": (
                        phase - int(meta["random_moves"]) if meta["random_moves"] is not None else ""
                    ),
                    "teacher_band": score_band(candidate.teacher),
                    "teacher": candidate.teacher,
                    "player_color": candidate.player_color,
                    "policy": candidate.policy,
                    "source_file": candidate.source_file,
                    "source_record": candidate.source_record,
                    "game_index_in_file": candidate.game_index,
                    "priority": candidate.priority,
                    "board": relative_board_string(candidate.player, candidate.opponent),
                }
            )

    sample_path = output_dir / "sample_positions.csv"
    with sample_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    sample_tsv_path = output_dir / "sample_positions.tsv"
    with sample_tsv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(final_rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(final_rows)

    check_rows: list[dict[str, object]] = []
    bin_root = data_root / "train_data" / "bin_data" / "20241125_1"
    for phase in phases:
        for data_id in historical_ids[phase]:
            indexed = bin_root / str(phase) / f"{data_id}.dat"
            indexed_count = indexed.stat().st_size // INDEXED_RECORD_SIZE
            raw_count = raw_counts[(phase, data_id)]
            check_rows.append(
                {
                    "phase": phase,
                    "empties": 60 - phase,
                    "data_id": data_id,
                    "indexed_records": indexed_count,
                    "board_records": raw_count,
                    "record_count_matches": indexed_count == raw_count,
                    "teacher_histogram_matches": "not_checked_here; compare label_distribution and raw histogram",
                    "board_data_exists": (board_root / f"records{data_id}").is_dir(),
                }
            )
    with (output_dir / "board_conversion_check.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(check_rows[0]))
        writer.writeheader()
        writer.writerows(check_rows)
    raw_hist_json = {
        f"phase{phase}_id{data_id}": {
            str(score + SCORE_MIN): int(count)
            for score, count in enumerate(hist.tolist())
            if count
        }
        for (phase, data_id), hist in raw_hists.items()
    }
    (output_dir / "board_label_histograms.json").write_text(
        json.dumps(raw_hist_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sampling_design = {
        "schema_version": 1,
        "seed": SEED,
        "frequency_sampling": (
            "uniform over loaded training records by the lowest deterministic "
            "SplitMix64 priorities; duplicates retain training frequency"
        ),
        "stratified_sampling": (
            "seeded hash order of data IDs; outer round-robin over data IDs; "
            "within each data ID, seeded hash order and round-robin over "
            "non-empty teacher score bands; lowest-priority reservoir records "
            "within each data-ID/band stratum"
        ),
        "stratified_coverage": stratified_coverage,
    }
    (output_dir / "sampling_design.json").write_text(
        json.dumps(sampling_design, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {sample_path} with {len(final_rows)} rows")
    print(f"wrote {sample_tsv_path} with {len(final_rows)} rows")
    print(f"wrote {output_dir / 'sampling_design.json'}")


def refresh_metadata(args: argparse.Namespace) -> None:
    """Refresh provenance fields without rescanning the large binary corpus."""

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "data_manifest.json"
    distribution_path = output_dir / "label_distribution.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    board_n_moves, _ = load_data_range(EVAL_DIR / "data_range.py")
    for entry in manifest["input_files"]:
        data_id = int(entry["data_id"])
        phase = int(entry["phase"])
        meta = source_metadata(data_id, board_n_moves)
        for key, value in meta.items():
            entry[key] = value
        entry["moves_after_random"] = (
            phase - int(meta["random_moves"]) if meta["random_moves"] is not None else None
        )
    manifest["metadata_refreshed_at_local"] = (
        __import__("datetime").datetime.now().astimezone().isoformat()
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with distribution_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0])
    for row in rows:
        if row["data_id"] == "ALL":
            continue
        data_id = int(row["data_id"])
        phase = int(row["phase"])
        meta = source_metadata(data_id, board_n_moves)
        row["source_category"] = str(meta["source_category"])
        row["random_moves"] = "" if meta["random_moves"] is None else str(meta["random_moves"])
        row["moves_after_random"] = (
            "" if meta["random_moves"] is None else str(phase - int(meta["random_moves"]))
        )
    with distribution_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"refreshed {manifest_path}")
    print(f"refreshed {distribution_path}")


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames if fieldnames is not None else list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=names, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_inputs(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    with (output_dir / "sample_positions.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as source:
        sample_rows = list(csv.DictReader(source, delimiter="\t"))
    sample_fields = list(sample_rows[0])
    per_kind_limits = {
        30: {"frequency": 10, "stratified_data_id_teacher_band": 10},
        35: {"frequency": 500, "stratified_data_id_teacher_band": 500},
        36: {"frequency": 500, "stratified_data_id_teacher_band": 500},
        40: {"frequency": 500, "stratified_data_id_teacher_band": 500},
        44: {"frequency": 500, "stratified_data_id_teacher_band": 500},
    }
    available_phases = {int(row["phase"]) for row in sample_rows}
    selected_by_phase: dict[int, list[dict[str, object]]] = {}
    for phase, kind_limits in per_kind_limits.items():
        if phase not in available_phases:
            continue
        selected: list[dict[str, object]] = []
        for kind, limit in kind_limits.items():
            selected.extend(
                row
                for row in sample_rows
                if int(row["phase"]) == phase and row["sample_kind"] == kind
            )
            if sum(1 for row in selected if row["sample_kind"] == kind) > limit:
                selected = [
                    row for row in selected
                    if row["sample_kind"] != kind
                ] + [
                    row for row in sample_rows
                    if int(row["phase"]) == phase and row["sample_kind"] == kind
                ][:limit]
        selected_by_phase[phase] = selected
        empties = 60 - phase
        write_tsv(
            output_dir / f"exact_input_{empties}_empties.tsv",
            selected,
            sample_fields,
        )
        for kind in kind_limits:
            kind_rows = [row for row in selected if row["sample_kind"] == kind]
            if kind_rows:
                safe_kind = (
                    "frequency"
                    if kind == "frequency"
                    else "stratified_data_id_teacher_band"
                )
                write_tsv(
                    output_dir
                    / f"exact_input_{empties}_empties_{safe_kind}.tsv",
                    kind_rows,
                    sample_fields,
                )
        print(
            f"wrote exact_input_{empties}_empties.tsv rows={len(selected)} "
            f"unique_boards={len({row['board'] for row in selected})}"
        )
    if 35 in selected_by_phase:
        write_tsv(
            output_dir / "phase35_search_input.tsv", selected_by_phase[35], sample_fields
        )
    if 40 in selected_by_phase:
        write_tsv(
            output_dir / "phase40_sibling_input.tsv", selected_by_phase[40], sample_fields
        )
    if 30 in selected_by_phase:
        write_tsv(
            output_dir / "phase30_probe_input.tsv", selected_by_phase[30][:1], sample_fields
        )

    ggs_path = Path(args.ggs_metadata)
    if ggs_path.is_file():
        ggs_rows: list[dict[str, object]] = []
        with ggs_path.open("r", encoding="utf-8") as source:
            for index, line in enumerate(source):
                item = json.loads(line)
                ggs_rows.append(
                    {
                        "sample_id": f"ggs_phase35_{index:05d}",
                        "sample_kind": "independent_ggs_log",
                        "phase": 35,
                        "empties": 25,
                        "source_category": "independent_ggs_log",
                        "source_file": item["game"],
                        "game_index_in_file": 0,
                        "ply": item["ply"],
                        "board": item["board"],
                    }
                )
        ggs_fields = list(ggs_rows[0])
        write_tsv(output_dir / "ggs_phase35_input.tsv", ggs_rows, ggs_fields)
        write_tsv(output_dir / "ggs_phase35_96_input.tsv", ggs_rows[:96], ggs_fields)
        latest_rows = sorted(
            ggs_rows, key=lambda row: str(row["source_file"]), reverse=True
        )[:96]
        write_tsv(
            output_dir / "ggs_phase35_latest96_input.tsv", latest_rows, ggs_fields
        )
        print(
            f"wrote GGS inputs rows={len(ggs_rows)}, seeded 96-row subset, "
            "and filename-date latest 96-row subset"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("EGAROUCID_DATA", ""),
        help="EGAROUCID_DATA root (defaults to the environment variable)",
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL))
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "benchmark" / "eval_training_bias_20260828")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest-labels")
    manifest.add_argument("--phases", default="28-46")
    manifest.set_defaults(func=build_manifest_and_labels)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--phase30", type=int, default=10)
    sample.add_argument("--phase35", type=int, default=500)
    sample.add_argument("--phase36", type=int, default=0)
    sample.add_argument("--phase40", type=int, default=1000)
    sample.add_argument("--phase44", type=int, default=1000)
    sample.add_argument("--stratum-reservoir", type=int, default=16)
    sample.add_argument("--chunk-records", type=int, default=2_000_000)
    sample.set_defaults(func=build_samples)

    refresh = subparsers.add_parser("refresh-metadata")
    refresh.set_defaults(func=refresh_metadata)

    prepare = subparsers.add_parser("prepare-inputs")
    prepare.add_argument(
        "--ggs-metadata",
        default=str(REPO_ROOT / "benchmark" / "eval_training_bias_20260828" / "ggs_25_empties.jsonl"),
    )
    prepare.set_defaults(func=prepare_inputs)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if not args.data_root:
        parser.error("--data-root or EGAROUCID_DATA is required")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
