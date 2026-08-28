#!/usr/bin/env python3
"""Aggregate exact-value, evaluator, sibling, and search-depth measurements.

The input files are the TSV artifacts produced by ``eval_training_bias_tool``
and ``run_edax_training_bias.py``.  Inputs are never modified.  Empty or
missing measurements are recorded explicitly and skipped unless
``--strict-missing`` is requested.

Confidence intervals use a deterministic cluster bootstrap.  The cluster is
``(source_file, game_index_in_file)``; when either component is unavailable,
the sample ID is used as an explicitly reported fallback cluster.  Each table
row has a context-derived random seed, so adding an unrelated group does not
change an existing interval.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


SCRIPT_VERSION = "1"
DEFAULT_SEED = 20260828
DEFAULT_BOOTSTRAP_REPS = 1000
DEPTHS = (0, 1, 2, 4, 5, 6, 8, 9, 10)
PHASES = (30, 35, 36, 40, 44)
EXPECTED_DATASET_SAMPLES = {
    "phase30": 20,
    "phase35": 1000,
    "phase36": 1000,
    "phase40": 1000,
    "phase44": 1000,
    "ggs_latest96": 96,
    "ggs_seeded96": 96,
}

SCORE_BANDS = (
    ("-64..-21", -64.0, -21.0),
    ("-20..-11", -20.0, -11.0),
    ("-10..-5", -10.0, -5.0),
    ("-4..-1", -4.0, -1.0),
    ("0", 0.0, 0.0),
    ("+1..+4", 1.0, 4.0),
    ("+5..+10", 5.0, 10.0),
    ("+11..+20", 11.0, 20.0),
    ("+21..+64", 21.0, 64.0),
)

COMPARISON_METRICS = (
    "bias",
    "mae",
    "mse",
    "slope",
    "intercept",
    "mean_abs_delta",
    "same_sign_smaller_abs_rate",
    "same_sign_smaller_abs_rate_nonzero_reference",
)


class AggregationError(RuntimeError):
    pass


class RunState:
    def __init__(self, strict_missing: bool) -> None:
        self.strict_missing = strict_missing
        self.inputs: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.counters: Counter[str] = Counter()

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print("warning: " + message, file=sys.stderr)

    def missing(self, path: Path, role: str, status: str) -> None:
        entry = {
            "role": role,
            "path": str(path.resolve()),
            "status": status,
            "rows": 0,
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": "",
        }
        self.inputs.append(entry)
        message = f"{role}: {status}: {path}"
        if self.strict_missing:
            raise AggregationError(message)
        self.warn(message)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_dir = repo_root / "benchmark" / "eval_training_bias_20260828"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be non-negative")
    if args.output_dir is None:
        args.output_dir = args.input_dir
    return args


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path, role: str, state: RunState, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            state.missing(path, role, "missing")
        return []
    if path.stat().st_size == 0:
        if required:
            state.missing(path, role, "empty")
        return []

    before = path.stat()
    rows: List[Dict[str, str]] = []
    malformed = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            state.missing(path, role, "missing_header")
            return []
        for row in reader:
            if (
                None in row
                or any(value is None for value in row.values())
                or not (row.get("sample_id") or "").strip()
            ):
                malformed += 1
                continue
            rows.append({key: (value or "") for key, value in row.items() if key is not None})
    after = path.stat()
    stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    digest = ""
    if stable:
        digest = sha256_file(path)
        final = path.stat()
        stable = after.st_size == final.st_size and after.st_mtime_ns == final.st_mtime_ns
    if not stable:
        digest = ""
        state.warn(
            f"input changed while being read/hashed; rerun after producer finishes: {path}"
        )
    if malformed:
        state.warn(f"{role}: skipped {malformed} malformed/incomplete rows: {path}")
    state.inputs.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "status": "used_unstable" if not stable else "used",
            "rows": len(rows),
            "malformed_rows": malformed,
            "bytes": after.st_size,
            "sha256": digest,
        }
    )
    return rows


def read_csv_rows(
    path: Path, role: str, state: RunState, required: bool = True
) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            state.missing(path, role, "missing")
        return []
    if path.stat().st_size == 0:
        if required:
            state.missing(path, role, "empty")
        return []
    before = path.stat()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [
            {key: (value or "") for key, value in row.items() if key is not None}
            for row in reader
            if None not in row and not any(value is None for value in row.values())
        ]
    after = path.stat()
    stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    digest = ""
    if stable:
        digest = sha256_file(path)
        final = path.stat()
        stable = after.st_size == final.st_size and after.st_mtime_ns == final.st_mtime_ns
    if not stable:
        state.warn(f"CSV changed while being read/hashed; rerun: {path}")
        digest = ""
    state.inputs.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "status": "used" if stable else "used_unstable",
            "rows": len(rows),
            "bytes": after.st_size,
            "sha256": digest,
        }
    )
    return rows


def read_json_object(
    path: Path, role: str, state: RunState, required: bool = True
) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            state.missing(path, role, "missing" if not path.exists() else "empty")
        return {}
    before = path.stat()
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    after = path.stat()
    stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    digest = sha256_file(path) if stable else ""
    state.inputs.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "status": "used" if stable else "used_unstable",
            "bytes": after.st_size,
            "sha256": digest,
        }
    )
    if not isinstance(payload, dict):
        raise AggregationError(f"expected JSON object: {path}")
    return payload


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> Optional[int]:
    number = as_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "complete"}


def first_nonempty(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def normalize_move(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"PASS", "PA"}:
        return "PS"
    return text


def score_sign(value: Optional[float]) -> str:
    if value is None:
        return ""
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


def score_band(value: Optional[float]) -> str:
    if value is None:
        return ""
    for name, lower, upper in SCORE_BANDS:
        if lower <= value <= upper:
            return name
    return "outside_-64_64"


def moves_after_random_bucket(value: Optional[int]) -> str:
    if value is None:
        return ""
    if value == 0:
        return "immediately_after_random"
    if 1 <= value <= 4:
        return "1..4"
    if 5 <= value <= 8:
        return "5..8"
    if 9 <= value <= 16:
        return "9..16"
    if value >= 17:
        return "17_plus"
    return "negative_or_invalid"


def cluster_id(row: Mapping[str, Any]) -> Tuple[str, bool]:
    source = first_nonempty(row, "source_file", "input_source_file")
    game = first_nonempty(row, "game_index_in_file", "input_game_index_in_file")
    if source and game:
        return source + "\x1f" + game, False
    sample_id = first_nonempty(row, "sample_id")
    return "sample_id\x1f" + sample_id, True


def dataset_from_name(name: str) -> Optional[str]:
    lower = name.lower()
    if "ggs_latest96" in lower:
        return "ggs_latest96"
    if "ggs96" in lower or "ggs_seeded96" in lower:
        return "ggs_seeded96"
    match = re.search(r"phase(\d+)", lower)
    if match:
        return "phase" + match.group(1)
    return None


def dataset_phase(dataset: str) -> Optional[int]:
    match = re.fullmatch(r"phase(\d+)", dataset)
    return int(match.group(1)) if match else (
        35 if dataset in {"ggs_latest96", "ggs_seeded96"} else None
    )


def boards_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    board_a = first_nonempty(a, "board", "root_normalized_board")
    board_b = first_nonempty(b, "board", "root_normalized_board")
    return not board_a or not board_b or board_a.strip().lower() == board_b.strip().lower()


def derived_seed(seed: int, context: str) -> int:
    digest = hashlib.sha256((str(seed) + "\0" + context).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def percentile(values: Iterable[float], probability: float) -> Optional[float]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    ss_y = sum((y - mean_y) ** 2 for y in ys)
    if ss_x <= 0.0 or ss_y <= 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return covariance / math.sqrt(ss_x * ss_y)


def comparison_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    pairs = [(as_float(row.get("x")), as_float(row.get("y"))) for row in rows]
    valid = [(x, y) for x, y in pairs if x is not None and y is not None]
    n = len(valid)
    if n == 0:
        return {name: None for name in COMPARISON_METRICS}
    xs = [pair[0] for pair in valid]
    ys = [pair[1] for pair in valid]
    residuals = [y - x for x, y in valid]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    slope = None if variance_x == 0.0 else sum(
        (x - mean_x) * (y - mean_y) for x, y in valid
    ) / variance_x
    intercept = None if slope is None else mean_y - slope * mean_x
    shrink_count = sum(
        1
        for x, y in valid
        if ((x > 0 and y > 0) or (x < 0 and y < 0)) and abs(y) < abs(x)
    )
    nonzero = sum(1 for x in xs if x != 0.0)
    return {
        "bias": statistics.fmean(residuals),
        "mae": statistics.fmean(abs(value) for value in residuals),
        "mse": statistics.fmean(value * value for value in residuals),
        "slope": slope,
        "intercept": intercept,
        "mean_abs_delta": statistics.fmean(abs(y) - abs(x) for x, y in valid),
        "same_sign_smaller_abs_rate": shrink_count / n,
        "same_sign_smaller_abs_rate_nonzero_reference": (
            shrink_count / nonzero if nonzero else None
        ),
    }


def bootstrap_metrics(
    rows: Sequence[Mapping[str, Any]],
    metric_function: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Optional[float]]],
    metric_names: Sequence[str],
    reps: int,
    seed: int,
    context: str,
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], int, int]:
    clusters: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    fallback_rows = 0
    for row in rows:
        key = str(row.get("cluster_id", "") or "")
        if not key:
            key, fallback = cluster_id(row)
            fallback_rows += int(fallback)
        elif str(row.get("cluster_fallback", "0")) == "1":
            fallback_rows += 1
        clusters[key].append(row)
    keys = sorted(clusters)
    empty = {name: (None, None) for name in metric_names}
    if reps <= 0 or len(keys) < 2:
        return empty, len(keys), fallback_rows
    rng = random.Random(derived_seed(seed, context))
    samples: Dict[str, List[float]] = {name: [] for name in metric_names}
    for _ in range(reps):
        selected: List[Mapping[str, Any]] = []
        for _cluster in range(len(keys)):
            selected.extend(clusters[keys[rng.randrange(len(keys))]])
        try:
            metrics = metric_function(selected)
        except Exception as exc:
            raise AggregationError(
                f"bootstrap metric failed for context={context!r}, "
                f"replicate_rows={len(selected)}, clusters={len(keys)}"
            ) from exc
        for name in metric_names:
            value = metrics.get(name)
            if value is not None and math.isfinite(float(value)):
                samples[name].append(float(value))
    # Do not report a conditional interval after silently discarding undefined
    # replicates (most often an OLS slope when a tiny resample has zero
    # reference variance).  Such an interval can look spuriously precise.
    return {
        name: (
            (percentile(samples[name], 0.025), percentile(samples[name], 0.975))
            if len(samples[name]) == reps
            else (None, None)
        )
        for name in metric_names
    }, len(keys), fallback_rows


def clean_output_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise AggregationError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]], overwrite: bool) -> None:
    prepare_output(path, overwrite)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean_output_value(row.get(name)) for name in fieldnames})
    temporary.replace(path)


def write_json(path: Path, payload: Any, overwrite: bool) -> None:
    prepare_output(path, overwrite)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def add_unique_root(
    destination: MutableMapping[str, Dict[str, str]],
    row: Dict[str, str],
    role: str,
    state: RunState,
) -> None:
    sample_id = row.get("sample_id", "")
    previous = destination.get(sample_id)
    if previous is None:
        destination[sample_id] = row
        return
    if not boards_match(previous, row):
        state.warn(f"{role}: duplicate sample_id has different boards: {sample_id}")
        state.counters["root_duplicate_board_conflicts"] += 1


def load_roots(
    input_dir: Path,
    state: RunState,
) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Dict[str, List[Tuple[str, Dict[str, str]]]]]]:
    roots: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    exact_companions: Dict[str, Dict[str, List[Tuple[str, Dict[str, str]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for phase in PHASES:
        dataset = f"phase{phase}"
        main_path = input_dir / f"egaroucid_phase{phase}_root.tsv"
        for row in read_tsv(main_path, f"egaroucid_{dataset}_root", state, required=True):
            if row.get("input_sample_kind") == "stratified_data_id_teacher_band":
                # Older runs used an ID-ordered truncated round-robin which
                # systematically omitted high data IDs.  Keep their valid
                # frequency-proportional half, and replace only the
                # stratified half with the corrected balanced run below.
                state.counters["excluded_legacy_biased_stratified_rows"] += 1
                continue
            add_unique_root(roots[dataset], row, f"egaroucid_{dataset}_root", state)
        stratified_path = input_dir / f"egaroucid_phase{phase}_balanced_strat_root.tsv"
        for row in read_tsv(
            stratified_path,
            f"egaroucid_{dataset}_balanced_strat_root",
            state,
            required=False,
        ):
            add_unique_root(
                roots[dataset], row, f"egaroucid_{dataset}_balanced_strat_root", state
            )
        companion = input_dir / f"egaroucid_phase{phase}_exact_root.tsv"
        for row in read_tsv(companion, f"egaroucid_{dataset}_exact_root", state, required=False):
            exact_companions[dataset][row["sample_id"]].append((companion.name, row))

    for dataset, stem in (
        ("ggs_latest96", "egaroucid_ggs_latest96"),
        ("ggs_seeded96", "egaroucid_ggs96"),
    ):
        ggs_path = input_dir / f"{stem}_root.tsv"
        for row in read_tsv(ggs_path, f"{stem}_root", state, required=True):
            add_unique_root(roots[dataset], row, f"{stem}_root", state)
        ggs_exact_path = input_dir / f"{stem}_exact_root.tsv"
        for row in read_tsv(
            # A complete exact result may already be embedded in the main
            # root TSV (the latest-96 run uses that form).  A separate
            # companion is therefore optional and is merged when present.
            ggs_exact_path, f"{stem}_exact_root", state, required=False
        ):
            exact_companions[dataset][row["sample_id"]].append(
                (ggs_exact_path.name, row)
            )

    expected_inputs = {
        "phase30": input_dir / "exact_input_30_empties.tsv",
        "phase35": input_dir / "phase35_search_input.tsv",
        "phase36": input_dir / "phase36_sample" / "sample_positions.tsv",
        "phase40": input_dir / "exact_input_20_empties.tsv",
        "phase44": input_dir / "exact_input_16_empties.tsv",
        "ggs_latest96": input_dir / "ggs_phase35_latest96_input.tsv",
        "ggs_seeded96": input_dir / "ggs_phase35_96_input.tsv",
    }
    for dataset, path in expected_inputs.items():
        expected_rows = read_tsv(
            path, f"expected_samples_{dataset}", state, required=False
        )
        if not expected_rows:
            continue
        expected_by_id = {row["sample_id"]: row for row in expected_rows}
        expected_ids = set(expected_by_id)
        observed_ids = set(roots.get(dataset, {}))
        mismatched_ids: List[str] = []
        for sample_id in sorted(expected_ids & observed_ids):
            expected_board = str(expected_by_id[sample_id].get("board", "")).strip()
            observed_board = str(
                first_nonempty(
                    # Expected input rows preserve their original X/O side-to-
                    # move marker.  The diagnostic's `board` field is that
                    # same input; `root_normalized_board` swaps an O-to-move
                    # position into X-to-move form and is only a fallback.
                    roots[dataset][sample_id], "board", "root_normalized_board"
                )
                or ""
            ).strip()
            if expected_board and observed_board and expected_board != observed_board:
                mismatched_ids.append(sample_id)
        if mismatched_ids:
            message = (
                f"{dataset} root board mismatch for {len(mismatched_ids)} sample IDs; "
                "stale measurements were excluded"
            )
            if state.strict_missing:
                raise AggregationError(message)
            state.warn(message)
            state.counters["root_board_mismatches"] += len(mismatched_ids)
            for sample_id in mismatched_ids:
                roots[dataset].pop(sample_id, None)
            observed_ids = set(roots.get(dataset, {}))
        extra_ids = observed_ids - expected_ids
        if extra_ids:
            state.warn(
                f"{dataset} has {len(extra_ids)} root sample IDs outside the current "
                "expected input; excluded"
            )
            state.counters["unexpected_root_sample_ids"] += len(extra_ids)
            for sample_id in extra_ids:
                roots[dataset].pop(sample_id, None)
            observed_ids = set(roots.get(dataset, {}))
        missing_ids = expected_ids - observed_ids
        if missing_ids:
            message = (
                f"{dataset} root measurement is incomplete: "
                f"{len(observed_ids & expected_ids)}/{len(expected_ids)} expected sample IDs present"
            )
            if state.strict_missing:
                raise AggregationError(message)
            state.warn(message)
            state.counters["incomplete_root_sample_ids"] += len(missing_ids)
    return dict(roots), exact_companions


def load_edax(
    input_dir: Path,
    state: RunState,
) -> Tuple[
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, List[Dict[str, str]]],
]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"depth_rows": defaultdict(list), "exact_rows": []})
    )
    child_results: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    paths = sorted(input_dir.glob("edax*/edax_results.tsv"))
    if not paths:
        state.warn(f"no Edax result TSVs found below {input_dir}")
    for path in paths:
        dataset = dataset_from_name(path.parent.name)
        if dataset is None:
            state.warn(f"cannot infer dataset from Edax directory; skipped: {path}")
            continue
        rows = read_tsv(path, f"edax_{path.parent.name}", state, required=True)
        if "balanced_strat" not in path.parent.name.lower():
            rows = [
                row
                for row in rows
                if first_nonempty(
                    row, "input_sample_kind", "input_input_sample_kind"
                )
                != "stratified_data_id_teacher_band"
            ]
        is_child_measurement = "sibling" in path.parent.name.lower() or any(
            row.get("input_edge_kind", "") for row in rows[:1]
        )
        if is_child_measurement:
            child_results[dataset].extend(rows)
            continue
        for row in rows:
            sample_id = row["sample_id"]
            if not is_true(row.get("complete")) or as_float(row.get("value")) is None:
                state.counters["edax_incomplete_rows"] += 1
                continue
            mode = row.get("mode", "")
            if mode == "exact":
                result[dataset][sample_id]["exact_rows"].append((path.parent.name, row))
                continue
            depth = as_int(row.get("requested_depth"))
            if mode == "fixed_depth" and depth in DEPTHS:
                result[dataset][sample_id]["depth_rows"][depth].append(
                    (path.parent.name, row)
                )
    return (
        {dataset: dict(samples) for dataset, samples in result.items()},
        {dataset: rows for dataset, rows in child_results.items()},
    )


def exact_candidate_from_egaroucid(
    row: Mapping[str, Any], source: str
) -> Optional[Dict[str, Any]]:
    value = as_float(row.get("exact_value"))
    if not is_true(row.get("exact_complete")) or value is None:
        return None
    return {
        "value": value,
        "best_move": normalize_move(row.get("exact_best_move")),
        "source": source,
        "row": row,
    }


def exact_candidate_from_edax(
    row: Mapping[str, Any], source: str
) -> Optional[Dict[str, Any]]:
    value = as_float(row.get("value"))
    if not is_true(row.get("complete")) or value is None:
        return None
    return {
        "value": value,
        "best_move": normalize_move(row.get("best_move")),
        "source": source,
        "row": row,
    }


def resolve_depth_row(
    candidates: Sequence[Tuple[str, Dict[str, str]]],
    dataset: str,
    sample_id: str,
    depth: int,
    state: RunState,
) -> Optional[Dict[str, str]]:
    if not candidates:
        return None
    values = {as_float(row.get("value")) for _source, row in candidates}
    values.discard(None)
    if len(values) > 1:
        state.counters["edax_depth_conflicts"] += 1
        if state.counters["edax_depth_conflicts"] <= 10:
            state.warn(
                f"Edax value conflict excluded: {dataset}/{sample_id}/depth{depth}: "
                + ",".join(str(value) for value in sorted(values))
            )
        return None
    return candidates[0][1]


def normalize_samples(
    roots: Mapping[str, Mapping[str, Dict[str, str]]],
    exact_companions: Mapping[str, Mapping[str, Sequence[Tuple[str, Dict[str, str]]]]],
    edax: Mapping[str, Mapping[str, Mapping[str, Any]]],
    state: RunState,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    normalized: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    # Every normalized position is anchored to a current Egaroucid root row.
    # This prevents an Edax-only row with a reused sample ID from reviving a
    # stale position after the root board/expected-input checks above.
    datasets = sorted(roots)
    for dataset in datasets:
        sample_ids: Set[str] = set(roots.get(dataset, {}))
        for sample_id in sorted(sample_ids):
            root = roots.get(dataset, {}).get(sample_id)
            edax_entry = edax.get(dataset, {}).get(sample_id, {})
            fallback_edax_rows: List[Dict[str, str]] = []
            for candidates in edax_entry.get("depth_rows", {}).values():
                fallback_edax_rows.extend(row for _source, row in candidates)
            fallback_edax_rows.extend(row for _source, row in edax_entry.get("exact_rows", []))
            base: Mapping[str, Any] = root or (fallback_edax_rows[0] if fallback_edax_rows else {})
            if not base:
                continue

            candidates: List[Dict[str, Any]] = []
            if root is not None:
                candidate = exact_candidate_from_egaroucid(root, f"egaroucid_{dataset}_root")
                if candidate is not None:
                    candidates.append(candidate)
            for source, row in exact_companions.get(dataset, {}).get(sample_id, []):
                if root is not None and not boards_match(root, row):
                    state.warn(f"exact companion board mismatch excluded: {dataset}/{sample_id}")
                    state.counters["exact_board_mismatches"] += 1
                    continue
                candidate = exact_candidate_from_egaroucid(row, source)
                if candidate is not None:
                    candidates.append(candidate)
            for source, row in edax_entry.get("exact_rows", []):
                if root is not None and not boards_match(root, row):
                    state.warn(f"Edax exact board mismatch excluded: {dataset}/{sample_id}")
                    state.counters["exact_board_mismatches"] += 1
                    continue
                candidate = exact_candidate_from_edax(row, "edax_" + source)
                if candidate is not None:
                    candidates.append(candidate)

            exact_values = {candidate["value"] for candidate in candidates}
            exact_value: Optional[float] = None
            exact_status = "missing"
            exact_sources = ""
            exact_best_moves: Set[str] = set()
            if len(exact_values) == 1:
                exact_value = next(iter(exact_values))
                exact_status = "complete_agreed" if len(candidates) > 1 else "complete_single_source"
                exact_sources = ";".join(sorted({candidate["source"] for candidate in candidates}))
                exact_best_moves = {
                    candidate["best_move"]
                    for candidate in candidates
                    if candidate["best_move"]
                }
            elif len(exact_values) > 1:
                exact_status = "conflict_excluded"
                state.counters["exact_value_conflicts"] += 1
                if state.counters["exact_value_conflicts"] <= 10:
                    state.warn(
                        f"exact value disagreement excluded: {dataset}/{sample_id}: "
                        + ",".join(str(value) for value in sorted(exact_values))
                    )

            phase = as_int(first_nonempty(base, "input_phase", "phase"))
            empties = as_int(first_nonempty(base, "input_empties", "empties", "root_n_empties"))
            data_id = as_int(first_nonempty(base, "input_data_id", "data_id"))
            random_moves = as_int(first_nonempty(base, "input_random_moves", "random_moves"))
            after_random = as_int(
                first_nonempty(base, "input_moves_after_random", "moves_after_random")
            )
            source_file = first_nonempty(base, "input_source_file", "source_file")
            game_index = first_nonempty(
                base, "input_game_index_in_file", "game_index_in_file"
            )
            cluster, fallback = cluster_id(base)
            depth_rows: Dict[int, Dict[str, str]] = {}
            for depth, depth_candidates in edax_entry.get("depth_rows", {}).items():
                matching = [
                    candidate
                    for candidate in depth_candidates
                    if root is None or boards_match(root, candidate[1])
                ]
                if len(matching) != len(depth_candidates):
                    state.counters["edax_depth_board_mismatches"] += (
                        len(depth_candidates) - len(matching)
                    )
                selected = resolve_depth_row(matching, dataset, sample_id, depth, state)
                if selected is not None:
                    depth_rows[depth] = selected

            normalized[dataset][sample_id] = {
                "dataset": dataset,
                "sample_id": sample_id,
                "root": root,
                "base": base,
                "phase": phase,
                "empties": empties,
                "sample_kind": first_nonempty(base, "input_sample_kind", "sample_kind"),
                "data_id": data_id,
                "source_category": first_nonempty(
                    base, "input_source_category", "source_category"
                ),
                "random_moves": random_moves,
                "moves_after_random": after_random,
                "moves_after_random_bucket": moves_after_random_bucket(after_random),
                "teacher": as_float(first_nonempty(base, "input_teacher", "teacher")),
                "teacher_band": first_nonempty(
                    base, "input_teacher_band", "teacher_band"
                ),
                "source_file": source_file,
                "game_index_in_file": game_index,
                "source_record": first_nonempty(
                    base, "input_source_record", "source_record"
                ),
                "player_color": first_nonempty(
                    base, "input_player_color", "player_color"
                ),
                "policy": first_nonempty(base, "input_policy", "policy"),
                "priority": first_nonempty(base, "input_priority", "priority"),
                "board": first_nonempty(base, "board", "root_normalized_board"),
                "root_legal_count": as_int(base.get("root_legal_count")),
                "egaroucid_static": as_float(root.get("static_value")) if root else None,
                "exact_value": exact_value,
                "exact_status": exact_status,
                "exact_sources": exact_sources,
                "exact_best_moves": exact_best_moves,
                "edax_depth_rows": depth_rows,
                "cluster_id": cluster,
                "cluster_fallback": int(fallback),
            }
    return {dataset: dict(samples) for dataset, samples in normalized.items()}


def sample_metadata(sample: Mapping[str, Any]) -> Dict[str, Any]:
    exact = as_float(sample.get("exact_value"))
    teacher = as_float(sample.get("teacher"))
    return {
        "dataset": sample.get("dataset"),
        "sample_id": sample.get("sample_id"),
        "phase": sample.get("phase"),
        "empties": sample.get("empties"),
        "sample_kind": sample.get("sample_kind"),
        "data_id": sample.get("data_id"),
        "source_category": sample.get("source_category"),
        "random_moves": sample.get("random_moves"),
        "moves_after_random": sample.get("moves_after_random"),
        "moves_after_random_bucket": sample.get("moves_after_random_bucket"),
        "teacher_band": sample.get("teacher_band") or score_band(teacher),
        "teacher_sign": score_sign(teacher),
        "exact_sign": score_sign(exact),
        "exact_score_band": score_band(exact),
        "source_file": sample.get("source_file"),
        "game_index_in_file": sample.get("game_index_in_file"),
        "source_record": sample.get("source_record"),
        "player_color": sample.get("player_color"),
        "policy": sample.get("policy"),
        "priority": sample.get("priority"),
        "board": sample.get("board"),
        "root_legal_count": sample.get("root_legal_count"),
        "cluster_id": sample.get("cluster_id"),
        "cluster_fallback": sample.get("cluster_fallback"),
        "exact_status": sample.get("exact_status"),
        "exact_sources": sample.get("exact_sources"),
        "exact_reference_moves": ";".join(sorted(sample.get("exact_best_moves", set()))),
    }


METADATA_FIELDS = (
    "dataset",
    "sample_id",
    "phase",
    "empties",
    "sample_kind",
    "data_id",
    "source_category",
    "random_moves",
    "moves_after_random",
    "moves_after_random_bucket",
    "teacher_band",
    "teacher_sign",
    "exact_sign",
    "exact_score_band",
    "source_file",
    "game_index_in_file",
    "source_record",
    "player_color",
    "policy",
    "priority",
    "board",
    "root_legal_count",
    "cluster_id",
    "cluster_fallback",
    "exact_status",
    "exact_sources",
    "exact_reference_moves",
)

LABEL_FIELDS = METADATA_FIELDS + (
    "z_teacher",
    "v_exact",
    "z_minus_v",
    "absolute_error",
    "squared_error",
    "abs_z_minus_abs_v",
    "same_sign_smaller_abs",
)

EVALUATOR_FIELDS = METADATA_FIELDS + (
    "z_teacher",
    "e_egaroucid_static",
    "v_exact",
    "d_edax_static",
    "z_minus_v",
    "e_minus_z",
    "e_minus_v",
    "d_minus_v",
    "abs_e_minus_abs_z",
    "abs_e_minus_abs_v",
    "abs_d_minus_abs_v",
    "forced_phase_minus1",
    "forced_value_minus1",
    "forced_phase_actual",
    "forced_value_actual",
    "forced_phase_plus1",
    "forced_value_plus1",
)

GROUP_DIMENSIONS = (
    "sample_kind",
    "exact_sign",
    "exact_score_band",
    "teacher_band",
    "teacher_sign",
    "data_id",
    "random_moves",
    "moves_after_random",
    "moves_after_random_bucket",
    "source_category",
    "root_legal_count",
    "player_color",
)

SUMMARY_BASE_FIELDS = (
    "dataset",
    "phase",
    "empties",
    "relation",
    "reference_name",
    "prediction_name",
    "group_dimension",
    "group_value",
    "status",
    "n",
    "n_clusters",
    "cluster_fallback_rows",
    "bootstrap_seed",
    "bootstrap_reps",
    "ci_method",
    "reference_negative_n",
    "reference_zero_n",
    "reference_positive_n",
    "same_sign_smaller_abs_n",
    "reference_nonzero_n",
)
SUMMARY_FIELDS = SUMMARY_BASE_FIELDS + tuple(
    field
    for metric in COMPARISON_METRICS
    for field in (metric, metric + "_ci_low", metric + "_ci_high")
)


def make_comparison_record(
    sample: Mapping[str, Any], x: float, y: float
) -> Dict[str, Any]:
    record = sample_metadata(sample)
    record["x"] = x
    record["y"] = y
    return record


def build_row_level_outputs(
    samples: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    label_rows: List[Dict[str, Any]] = []
    evaluator_rows: List[Dict[str, Any]] = []
    for dataset in sorted(samples):
        for sample_id in sorted(samples[dataset]):
            sample = samples[dataset][sample_id]
            metadata = sample_metadata(sample)
            z = as_float(sample.get("teacher"))
            v = as_float(sample.get("exact_value"))
            e = as_float(sample.get("egaroucid_static"))
            d_row = sample.get("edax_depth_rows", {}).get(0)
            d = as_float(d_row.get("value")) if d_row else None
            if z is not None and v is not None:
                residual = z - v
                same_sign_shrink = int(
                    ((z > 0 and v > 0) or (z < 0 and v < 0)) and abs(z) < abs(v)
                )
                label_rows.append(
                    {
                        **metadata,
                        "z_teacher": z,
                        "v_exact": v,
                        "z_minus_v": residual,
                        "absolute_error": abs(residual),
                        "squared_error": residual * residual,
                        "abs_z_minus_abs_v": abs(z) - abs(v),
                        "same_sign_smaller_abs": same_sign_shrink,
                    }
                )
            if v is not None and (z is not None or e is not None or d is not None):
                root = sample.get("root") or {}
                evaluator_rows.append(
                    {
                        **metadata,
                        "z_teacher": z,
                        "e_egaroucid_static": e,
                        "v_exact": v,
                        "d_edax_static": d,
                        "z_minus_v": z - v if z is not None else None,
                        "e_minus_z": e - z if e is not None and z is not None else None,
                        "e_minus_v": e - v if e is not None else None,
                        "d_minus_v": d - v if d is not None else None,
                        "abs_e_minus_abs_z": (
                            abs(e) - abs(z) if e is not None and z is not None else None
                        ),
                        "abs_e_minus_abs_v": abs(e) - abs(v) if e is not None else None,
                        "abs_d_minus_abs_v": abs(d) - abs(v) if d is not None else None,
                        "forced_phase_minus1": as_int(root.get("forced_phase_minus1")),
                        "forced_value_minus1": as_float(root.get("forced_value_minus1")),
                        "forced_phase_actual": as_int(root.get("forced_phase_actual")),
                        "forced_value_actual": as_float(root.get("forced_value_actual")),
                        "forced_phase_plus1": as_int(root.get("forced_phase_plus1")),
                        "forced_value_plus1": as_float(root.get("forced_value_plus1")),
                    }
                )
    return label_rows, evaluator_rows


def iter_grouped_records(
    records: Sequence[Mapping[str, Any]],
) -> Iterable[Tuple[str, str, List[Mapping[str, Any]]]]:
    yield "all", "all", list(records)
    for dimension in GROUP_DIMENSIONS:
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in records:
            value = row.get(dimension)
            if value is None or str(value) == "":
                continue
            groups[str(value)].append(row)
        for value in sorted(groups, key=lambda item: (len(item), item)):
            yield dimension, value, groups[value]


def comparison_summary_row(
    dataset: str,
    phase: Optional[int],
    relation: str,
    reference_name: str,
    prediction_name: str,
    dimension: str,
    value: str,
    rows: Sequence[Mapping[str, Any]],
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    point = comparison_metrics(rows)
    context = "|".join((dataset, relation, dimension, value))
    intervals, n_clusters, fallback_rows = bootstrap_metrics(
        rows, comparison_metrics, COMPARISON_METRICS, reps, seed, context
    )
    xs = [as_float(row.get("x")) for row in rows]
    xs = [value for value in xs if value is not None]
    shrink_n = sum(
        1
        for row in rows
        if (lambda x, y: x is not None and y is not None and
            ((x > 0 and y > 0) or (x < 0 and y < 0)) and abs(y) < abs(x))(
                as_float(row.get("x")), as_float(row.get("y"))
            )
    )
    result: Dict[str, Any] = {
        "dataset": dataset,
        "phase": phase,
        "empties": 60 - phase if phase is not None else None,
        "relation": relation,
        "reference_name": reference_name,
        "prediction_name": prediction_name,
        "group_dimension": dimension,
        "group_value": value,
        # This describes whether this relation/group has usable rows, not
        # whether the requested sampling target was fully measured.
        "status": "available_rows" if rows else "no_valid_rows",
        "n": len(rows),
        "n_clusters": n_clusters,
        "cluster_fallback_rows": fallback_rows,
        "bootstrap_seed": derived_seed(seed, context),
        "bootstrap_reps": reps if n_clusters >= 2 else 0,
        "ci_method": "source_file+game_index cluster bootstrap percentile 95%",
        "reference_negative_n": sum(x < 0 for x in xs),
        "reference_zero_n": sum(x == 0 for x in xs),
        "reference_positive_n": sum(x > 0 for x in xs),
        "same_sign_smaller_abs_n": shrink_n,
        "reference_nonzero_n": sum(x != 0 for x in xs),
    }
    for metric in COMPARISON_METRICS:
        result[metric] = point.get(metric)
        result[metric + "_ci_low"] = intervals[metric][0]
        result[metric + "_ci_high"] = intervals[metric][1]
    return result


def build_phase_summary(
    samples: Mapping[str, Mapping[str, Mapping[str, Any]]],
    reps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    relation_definitions = (
        ("label_vs_exact", "exact_value", "teacher", "v_exact", "z_teacher"),
        (
            "egaroucid_vs_label",
            "teacher",
            "egaroucid_static",
            "z_teacher",
            "e_egaroucid_static",
        ),
        (
            "egaroucid_vs_exact",
            "exact_value",
            "egaroucid_static",
            "v_exact",
            "e_egaroucid_static",
        ),
    )
    for dataset in sorted(samples):
        phase = dataset_phase(dataset)
        dataset_samples = list(samples[dataset].values())
        for relation, x_key, y_key, x_name, y_name in relation_definitions:
            records = [
                make_comparison_record(sample, float(sample[x_key]), float(sample[y_key]))
                for sample in dataset_samples
                if as_float(sample.get(x_key)) is not None
                and as_float(sample.get(y_key)) is not None
            ]
            for dimension, value, group in iter_grouped_records(records):
                output.append(
                    comparison_summary_row(
                        dataset,
                        phase,
                        relation,
                        x_name,
                        y_name,
                        dimension,
                        value,
                        group,
                        reps,
                        seed,
                    )
                )

        edax_records: List[Dict[str, Any]] = []
        edax_label_records: List[Dict[str, Any]] = []
        for sample in dataset_samples:
            exact = as_float(sample.get("exact_value"))
            teacher = as_float(sample.get("teacher"))
            depth_row = sample.get("edax_depth_rows", {}).get(0)
            edax_value = as_float(depth_row.get("value")) if depth_row else None
            if exact is not None and edax_value is not None:
                edax_records.append(make_comparison_record(sample, exact, edax_value))
            if teacher is not None and edax_value is not None:
                edax_label_records.append(
                    make_comparison_record(sample, teacher, edax_value)
                )
        for dimension, value, group in iter_grouped_records(edax_records):
            output.append(
                comparison_summary_row(
                    dataset,
                    phase,
                    "edax_vs_exact",
                    "v_exact",
                    "d_edax_static",
                    dimension,
                    value,
                    group,
                    reps,
                    seed,
                )
            )
        for dimension, value, group in iter_grouped_records(edax_label_records):
            output.append(
                comparison_summary_row(
                    dataset,
                    phase,
                    "edax_vs_label",
                    "z_teacher",
                    "d_edax_static",
                    dimension,
                    value,
                    group,
                    reps,
                    seed,
                )
            )

        for suffix, prediction_name in (("minus1", "forced_phase_minus1"),
                                         ("plus1", "forced_phase_plus1")):
            records = []
            for sample in dataset_samples:
                root = sample.get("root") or {}
                actual = as_float(root.get("forced_value_actual"))
                forced = as_float(root.get(f"forced_value_{suffix}"))
                if actual is not None and forced is not None:
                    records.append(make_comparison_record(sample, actual, forced))
            if records:
                output.append(
                    comparison_summary_row(
                        dataset,
                        phase,
                        f"forced_{suffix}_vs_actual_phase_weight",
                        "forced_actual_phase_value",
                        prediction_name,
                        "all",
                        "all",
                        records,
                        reps,
                        seed,
                    )
                )

    present = set(samples)
    for phase in PHASES:
        dataset = f"phase{phase}"
        if dataset not in present:
            output.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "empties": 60 - phase,
                    "relation": "all",
                    "group_dimension": "all",
                    "group_value": "all",
                    "status": "missing_or_empty_root_input",
                    "n": 0,
                    "bootstrap_seed": seed,
                    "bootstrap_reps": 0,
                    "ci_method": "source_file+game_index cluster bootstrap percentile 95%",
                }
            )
    return output


SIBLING_BOOTSTRAP_METRICS = (
    "strict_pair_ranking_accuracy",
    "strict_pair_ranking_half_tie_credit",
    "sibling_difference_mae",
    "sibling_residual_correlation",
    "predicted_best_set_overlap_rate",
    "predicted_best_set_min_loss_mean",
)

SELECTION_BOOTSTRAP_METRICS = (
    "best_move_match_rate",
    "selected_loss_mean",
)

SIBLING_FIELDS = (
    "dataset",
    "phase",
    "empties",
    "engine",
    "evaluation",
    "depth",
    "status",
    "n_parents",
    "n_children",
    "n_clusters",
    "cluster_fallback_parents",
    "bootstrap_seed",
    "bootstrap_reps",
    "ci_method",
    "all_sibling_pairs",
    "exact_tie_pairs",
    "strict_exact_order_pairs",
    "concordant_pairs",
    "discordant_pairs",
    "predicted_tie_pairs",
    "strict_pair_ranking_accuracy",
    "strict_pair_ranking_accuracy_ci_low",
    "strict_pair_ranking_accuracy_ci_high",
    "strict_pair_ranking_half_tie_credit",
    "strict_pair_ranking_half_tie_credit_ci_low",
    "strict_pair_ranking_half_tie_credit_ci_high",
    "sibling_difference_mae",
    "sibling_difference_mae_ci_low",
    "sibling_difference_mae_ci_high",
    "sibling_residual_correlation",
    "sibling_residual_correlation_ci_low",
    "sibling_residual_correlation_ci_high",
    "predicted_best_set_overlap_n",
    "predicted_best_set_overlap_rate",
    "predicted_best_set_overlap_rate_ci_low",
    "predicted_best_set_overlap_rate_ci_high",
    "predicted_best_set_min_loss_mean",
    "predicted_best_set_min_loss_mean_ci_low",
    "predicted_best_set_min_loss_mean_ci_high",
    "selection_available_parents",
    "selection_unresolved_parents",
    "selection_ambiguous_parents",
    "best_move_match_n",
    "best_move_match_rate",
    "best_move_match_rate_ci_low",
    "best_move_match_rate_ci_high",
    "selected_loss_mean",
    "selected_loss_mean_ci_low",
    "selected_loss_mean_ci_high",
    "selected_loss_max",
    "ranking_scope_note",
)


def load_child_measurements(
    input_dir: Path,
    state: RunState,
    edax_child_results: Mapping[str, Sequence[Mapping[str, str]]],
    samples: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, List[Dict[str, Any]]],
]:
    sibling_path = input_dir / "egaroucid_phase40_sibling_children.tsv"
    all_paths = sorted(input_dir.glob("egaroucid_*children.tsv"))
    if sibling_path not in all_paths:
        all_paths.append(sibling_path)
        all_paths.sort()
    ranking_paths = {
        "phase40": sibling_path,
        "ggs_latest96": input_dir / "egaroucid_ggs_latest96_children.tsv",
        "ggs_seeded96": input_dir / "egaroucid_ggs96_sibling_children.tsv",
    }
    phase40_balanced_sibling_path = (
        input_dir / "egaroucid_phase40_balanced_strat_children.tsv"
    )
    exact_maps: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    child_boards: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    child_board_conflicts: Set[Tuple[str, str, str]] = set()
    ranking_rows: Dict[str, List[Dict[str, str]]] = {}
    conflicts: Set[Tuple[str, str, str]] = set()
    for path in all_paths:
        required = path == sibling_path
        rows = read_tsv(path, f"children_{path.stem}", state, required=required)
        if "balanced_strat" not in path.name:
            rows = [
                row
                for row in rows
                if row.get("input_sample_kind")
                != "stratified_data_id_teacher_band"
            ]
        dataset = dataset_from_name(path.name)
        if dataset is None:
            continue
        if path == ranking_paths.get(dataset) or (
            dataset == "phase40" and path == phase40_balanced_sibling_path
        ):
            ranking_rows.setdefault(dataset, []).extend(rows)
        for row in rows:
            move = normalize_move(row.get("move"))
            sample_id = row.get("sample_id", "")
            board = first_nonempty(row, "child_normalized_board").strip().lower()
            if move and sample_id and board:
                previous_board = child_boards[dataset][sample_id].get(move)
                if previous_board is not None and previous_board != board:
                    child_board_conflicts.add((dataset, sample_id, move))
                else:
                    child_boards[dataset][sample_id][move] = board
            if not is_true(row.get("exact_complete")):
                continue
            value = as_float(row.get("exact_value_parent"))
            if value is None or not move or not sample_id:
                continue
            key = (dataset, sample_id, move)
            previous = exact_maps[dataset][sample_id].get(move)
            if previous is not None and previous != value:
                conflicts.add(key)
                continue
            exact_maps[dataset][sample_id][move] = value

    # Edax is invoked on each child with the child side to move.  Negate its
    # exact result to put it on the parent-side scale used by Egaroucid's
    # exact_value_parent.  This supplies exact child losses for independent
    # GGS roots even when the Egaroucid run embedded only root exact values.
    for dataset, rows in edax_child_results.items():
        for row in rows:
            if row.get("mode") != "exact" or not is_true(row.get("complete")):
                continue
            child_value = as_float(row.get("value"))
            move = normalize_move(row.get("input_move"))
            sample_id = row.get("sample_id", "")
            if child_value is None or not move or not sample_id:
                continue
            expected_board = child_boards.get(dataset, {}).get(sample_id, {}).get(move)
            measured_board = first_nonempty(row, "board").strip().lower()
            if expected_board and measured_board and expected_board != measured_board:
                state.counters["excluded_edax_exact_child_board_mismatch"] += 1
                continue
            value = -child_value
            key = (dataset, sample_id, move)
            previous = exact_maps[dataset][sample_id].get(move)
            if previous is not None and previous != value:
                conflicts.add(key)
                continue
            exact_maps[dataset][sample_id][move] = value
            state.counters["exact_child_values_from_edax"] += 1

    for dataset, sample_id, move in conflicts:
        exact_maps[dataset][sample_id].pop(move, None)
        state.counters["exact_child_value_conflicts"] += 1
    if conflicts:
        state.warn(f"excluded {len(conflicts)} conflicting exact child move values")
    for dataset, sample_id, move in child_board_conflicts:
        exact_maps[dataset][sample_id].pop(move, None)
        child_boards[dataset][sample_id].pop(move, None)
        state.counters["exact_child_board_conflicts"] += 1

    # A partially written parent must never define an exact best-move set or
    # selected-move loss.  Require every legal child (or the single pass
    # edge) and require the best child value to reproduce the root exact
    # value before exposing that parent's map to downstream metrics.
    for dataset in list(exact_maps):
        for sample_id in list(exact_maps[dataset]):
            sample = samples.get(dataset, {}).get(sample_id)
            root = (sample or {}).get("root") or {}
            legal_count = as_int(root.get("root_legal_count"))
            is_pass = is_true(root.get("root_is_pass"))
            is_terminal = is_true(root.get("root_is_terminal"))
            expected_children = (
                0 if is_terminal else 1 if is_pass else legal_count
            )
            root_exact = as_float((sample or {}).get("exact_value"))
            child_values = exact_maps[dataset][sample_id]
            complete = (
                expected_children is not None
                and len(child_values) == expected_children
                and expected_children > 0
                and root_exact is not None
                and max(child_values.values()) == root_exact
            )
            if not complete:
                exact_maps[dataset].pop(sample_id, None)
                state.counters["excluded_incomplete_exact_child_parents"] += 1

    edax_parent_values: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    edax_child_conflicts: Set[Tuple[str, str, str]] = set()
    for dataset, rows in edax_child_results.items():
        for row in rows:
            if (
                row.get("mode") != "fixed_depth"
                or not is_true(row.get("complete"))
                or as_int(row.get("requested_depth")) != 0
            ):
                continue
            child_value = as_float(row.get("value"))
            move = normalize_move(row.get("input_move"))
            sample_id = row.get("sample_id", "")
            if child_value is None or not move or not sample_id:
                continue
            expected_board = child_boards.get(dataset, {}).get(sample_id, {}).get(move)
            measured_board = first_nonempty(row, "board").strip().lower()
            if expected_board and measured_board and expected_board != measured_board:
                state.counters["excluded_edax_static_child_board_mismatch"] += 1
                continue
            parent_value = -child_value
            previous = edax_parent_values[dataset][sample_id].get(move)
            if previous is not None and previous != parent_value:
                edax_child_conflicts.add((dataset, sample_id, move))
                continue
            edax_parent_values[dataset][sample_id][move] = parent_value
    for dataset, sample_id, move in edax_child_conflicts:
        edax_parent_values[dataset][sample_id].pop(move, None)
    if edax_child_conflicts:
        state.warn(
            f"excluded {len(edax_child_conflicts)} conflicting Edax child static values"
        )

    parents_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for dataset, rows in ranking_rows.items():
        parents_by_id: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            move = normalize_move(row.get("move"))
            sample_id = row.get("sample_id", "")
            exact = exact_maps.get(dataset, {}).get(sample_id, {}).get(move)
            if exact is None or not move or not sample_id:
                continue
            parent = parents_by_id.get(sample_id)
            if parent is None:
                cluster, fallback = cluster_id(row)
                parent = {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "cluster_id": cluster,
                    "cluster_fallback": int(fallback),
                    "children": [],
                }
                parents_by_id[sample_id] = parent
            parent["children"].append(
                {
                    "move": move,
                    "exact": exact,
                    "static": as_float(row.get("static_value_parent"))
                    if is_true(row.get("static_available"))
                    else None,
                    # Edax's result is from the child side to move. Negation
                    # puts it on the parent-side scale.
                    "edax_static": edax_parent_values
                    .get(dataset, {})
                    .get(sample_id, {})
                    .get(move),
                    "row": row,
                }
            )
        parents_by_dataset[dataset] = [
            parents_by_id[key] for key in sorted(parents_by_id)
        ]
    return (
        {dataset: dict(sample_map) for dataset, sample_map in exact_maps.items()},
        parents_by_dataset,
    )


def static_sibling_metrics(
    parents: Sequence[Mapping[str, Any]], prediction_key: str
) -> Dict[str, Optional[float]]:
    all_pairs = 0
    exact_ties = 0
    strict_pairs = 0
    concordant = 0
    discordant = 0
    predicted_ties = 0
    difference_errors: List[float] = []
    residual_left: List[float] = []
    residual_right: List[float] = []
    best_overlap = 0
    best_overlap_denominator = 0
    best_set_losses: List[float] = []
    child_count = 0
    for parent in parents:
        children = [
            child
            for child in parent.get("children", [])
            if as_float(child.get("exact")) is not None
            and as_float(child.get(prediction_key)) is not None
        ]
        child_count += len(children)
        if not children:
            continue
        exact_best = max(float(child["exact"]) for child in children)
        predicted_best = max(float(child[prediction_key]) for child in children)
        predicted_best_children = [
            child for child in children if float(child[prediction_key]) == predicted_best
        ]
        best_overlap_denominator += 1
        if any(float(child["exact"]) == exact_best for child in predicted_best_children):
            best_overlap += 1
        best_set_losses.append(
            exact_best - max(float(child["exact"]) for child in predicted_best_children)
        )
        for left_index in range(len(children)):
            for right_index in range(left_index + 1, len(children)):
                left = children[left_index]
                right = children[right_index]
                exact_difference = float(left["exact"]) - float(right["exact"])
                predicted_difference = float(left[prediction_key]) - float(
                    right[prediction_key]
                )
                all_pairs += 1
                difference_errors.append(abs(predicted_difference - exact_difference))
                left_residual = float(left[prediction_key]) - float(left["exact"])
                right_residual = float(right[prediction_key]) - float(right["exact"])
                # Include both orientations so the residual correlation is not
                # affected by arbitrary move ordering.
                residual_left.extend((left_residual, right_residual))
                residual_right.extend((right_residual, left_residual))
                if exact_difference == 0.0:
                    exact_ties += 1
                else:
                    strict_pairs += 1
                    product = exact_difference * predicted_difference
                    if product > 0.0:
                        concordant += 1
                    elif product < 0.0:
                        discordant += 1
                    else:
                        predicted_ties += 1
    return {
        "n_parents": float(len(parents)),
        "n_children": float(child_count),
        "all_sibling_pairs": float(all_pairs),
        "exact_tie_pairs": float(exact_ties),
        "strict_exact_order_pairs": float(strict_pairs),
        "concordant_pairs": float(concordant),
        "discordant_pairs": float(discordant),
        "predicted_tie_pairs": float(predicted_ties),
        "strict_pair_ranking_accuracy": concordant / strict_pairs if strict_pairs else None,
        "strict_pair_ranking_half_tie_credit": (
            (concordant + 0.5 * predicted_ties) / strict_pairs if strict_pairs else None
        ),
        "sibling_difference_mae": (
            statistics.fmean(difference_errors) if difference_errors else None
        ),
        "sibling_residual_correlation": pearson(residual_left, residual_right),
        "predicted_best_set_overlap_n": float(best_overlap),
        "predicted_best_set_overlap_rate": (
            best_overlap / best_overlap_denominator if best_overlap_denominator else None
        ),
        "predicted_best_set_min_loss_mean": (
            statistics.fmean(best_set_losses) if best_set_losses else None
        ),
    }


def selected_moves_for_parent(
    parent: Mapping[str, Any],
    engine: str,
    depth: int,
    samples: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    if engine == "egaroucid":
        column = f"selected_by_depth_{depth}"
        return sorted(
            {
                child["move"]
                for child in parent.get("children", [])
                if is_true(child.get("row", {}).get(column))
            }
        )
    sample = samples.get(str(parent.get("sample_id", "")))
    if not sample:
        return []
    row = sample.get("edax_depth_rows", {}).get(depth)
    move = normalize_move(row.get("best_move")) if row else ""
    return [move] if move else []


def selection_metrics(
    parents: Sequence[Mapping[str, Any]],
    selector: Callable[[Mapping[str, Any]], Sequence[str]],
) -> Dict[str, Optional[float]]:
    available = 0
    unresolved = 0
    ambiguous = 0
    matches = 0
    losses: List[float] = []
    for parent in parents:
        exact_by_move = {
            str(child["move"]): float(child["exact"])
            for child in parent.get("children", [])
            if as_float(child.get("exact")) is not None
        }
        if not exact_by_move:
            continue
        selected = sorted(set(selector(parent)))
        recognized = [move for move in selected if move in exact_by_move]
        if not recognized:
            unresolved += 1
            continue
        available += 1
        ambiguous += int(len(recognized) > 1)
        exact_best = max(exact_by_move.values())
        # Multiple selected flags normally indicate a score tie.  The minimum
        # loss is reported and ambiguity is counted separately.
        selected_value = max(exact_by_move[move] for move in recognized)
        loss = exact_best - selected_value
        losses.append(loss)
        matches += int(loss == 0.0)
    return {
        "selection_available_parents": float(available),
        "selection_unresolved_parents": float(unresolved),
        "selection_ambiguous_parents": float(ambiguous),
        "best_move_match_n": float(matches),
        "best_move_match_rate": matches / available if available else None,
        "selected_loss_mean": statistics.fmean(losses) if losses else None,
        "selected_loss_max": max(losses) if losses else None,
    }


def build_sibling_summary(
    parents_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    samples: Mapping[str, Mapping[str, Mapping[str, Any]]],
    reps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for dataset in ("phase40", "ggs_latest96", "ggs_seeded96"):
        phase = dataset_phase(dataset)
        parents = list(parents_by_dataset.get(dataset, []))
        if not parents:
            output.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "empties": 60 - phase if phase is not None else None,
                    "engine": "egaroucid",
                    "evaluation": "all_child_static_values",
                    "status": "missing_empty_or_incomplete_exact_sibling_children",
                    "n_parents": 0,
                    "ranking_scope_note": (
                        "No parent with a complete exact child set was available."
                    ),
                }
            )
            continue

        for engine, prediction_key in (
            ("egaroucid", "static"),
            ("edax", "edax_static"),
        ):
            engine_parents = [
                parent
                for parent in parents
                if parent.get("children")
                and all(
                    as_float(child.get(prediction_key)) is not None
                    for child in parent.get("children", [])
                )
            ]
            function = lambda selected, prediction_key=prediction_key: static_sibling_metrics(
                selected, prediction_key
            )
            point = function(engine_parents)
            if not point.get("n_children"):
                continue
            context = f"{dataset}|{engine}|all_child_static_values"
            intervals, n_clusters, fallback = bootstrap_metrics(
                engine_parents,
                function,
                SIBLING_BOOTSTRAP_METRICS,
                reps,
                seed,
                context,
            )
            row: Dict[str, Any] = {
                "dataset": dataset,
                "phase": phase,
                "empties": 60 - phase if phase is not None else None,
                "engine": engine,
                "evaluation": "all_child_static_values",
                "depth": 0,
                "status": (
                    "complete"
                    if len(engine_parents) == len(parents)
                    else "partial_complete_parents_only"
                ),
                "n_clusters": n_clusters,
                "cluster_fallback_parents": fallback,
                "bootstrap_seed": derived_seed(seed, context),
                "bootstrap_reps": reps if n_clusters >= 2 else 0,
                "ci_method": "source_file+game_index parent-cluster bootstrap percentile 95%",
                "ranking_scope_note": (
                    "Ranking uses every unordered sibling pair with exact unequal pairs as the strict denominator; "
                    "residual correlation uses both orientations. Edax child-side values are negated to parent perspective."
                ),
                **point,
            }
            for metric in SIBLING_BOOTSTRAP_METRICS:
                row[metric + "_ci_low"] = intervals[metric][0]
                row[metric + "_ci_high"] = intervals[metric][1]
            output.append(row)

        # depth 0 is represented by the all-child static rows above.  A
        # zero-ply root evaluation itself has no selected move.
        for engine in ("egaroucid", "edax"):
            for depth in (value for value in DEPTHS if value > 0):
                selector = lambda parent, engine=engine, depth=depth, dataset=dataset: selected_moves_for_parent(
                    parent, engine, depth, samples.get(dataset, {})
                )
                point_selection = selection_metrics(parents, selector)
                if not point_selection.get("selection_available_parents"):
                    continue
                context = f"{dataset}|{engine}|selection|depth{depth}"
                function = lambda selected, selector=selector: selection_metrics(
                    selected, selector
                )
                intervals, n_clusters, fallback = bootstrap_metrics(
                    parents,
                    function,
                    SELECTION_BOOTSTRAP_METRICS,
                    reps,
                    seed,
                    context,
                )
                selection_row: Dict[str, Any] = {
                    "dataset": dataset,
                    "phase": phase,
                    "empties": 60 - phase if phase is not None else None,
                    "engine": engine,
                    "evaluation": "selected_move_only",
                    "depth": depth,
                    "status": "complete_selection_only_no_all_child_scores",
                    "n_parents": len(parents),
                    "n_children": sum(
                        len(parent.get("children", [])) for parent in parents
                    ),
                    "n_clusters": n_clusters,
                    "cluster_fallback_parents": fallback,
                    "bootstrap_seed": derived_seed(seed, context),
                    "bootstrap_reps": reps if n_clusters >= 2 else 0,
                    "ci_method": "source_file+game_index parent-cluster bootstrap percentile 95%",
                    "ranking_scope_note": (
                        "Only the engine-selected move is scored at this depth; all-sibling pair ranking metrics are unavailable."
                    ),
                    **point_selection,
                }
                for metric in SELECTION_BOOTSTRAP_METRICS:
                    selection_row[metric + "_ci_low"] = intervals[metric][0]
                    selection_row[metric + "_ci_high"] = intervals[metric][1]
                output.append(selection_row)
    return output


PAIRED_DEPTH_METRICS = (
    "egaroucid_minus_edax_bias",
    "egaroucid_minus_edax_absolute_error",
    "egaroucid_minus_edax_squared_error",
    "egaroucid_minus_edax_value",
)

SEARCH_BASE_FIELDS = (
    "dataset",
    "phase",
    "empties",
    "group_dimension",
    "group_value",
    "depth",
    "depth_parity",
    "expected_leaf_phase",
    "egaroucid_leaf_phase_mode",
    "egaroucid_leaf_phase_min",
    "egaroucid_leaf_phase_max",
    "reference_value_name",
    "reference_value_n",
    "status",
    "expected_sample_n",
    "dataset_sample_n",
    "exact_value_n",
    "previous_depth",
    "ci_method",
    "bootstrap_reps",
)
SEARCH_ENGINE_FIELDS = tuple(
    field
    for engine in ("egaroucid", "edax")
    for field in (
        engine + "_n",
        engine + "_n_clusters",
        engine + "_cluster_fallback_rows",
        engine + "_bootstrap_seed",
        *(
            metric_field
            for metric in COMPARISON_METRICS
            for metric_field in (
                engine + "_" + metric,
                engine + "_" + metric + "_ci_low",
                engine + "_" + metric + "_ci_high",
            )
        ),
        engine + "_previous_comparable_n",
        engine + "_worsened_n",
        engine + "_improved_n",
        engine + "_unchanged_n",
        engine + "_best_move_changed_n",
        engine + "_best_move_comparable_n",
        engine + "_pv_prefix_changed_n",
        engine + "_pv_comparable_n",
        engine + "_best_move_reference_n",
        engine + "_best_move_match_n",
        engine + "_best_move_match_rate",
        engine + "_best_move_match_rate_ci_low",
        engine + "_best_move_match_rate_ci_high",
        engine + "_selected_loss_n",
        engine + "_selected_loss_mean",
        engine + "_selected_loss_ci_low",
        engine + "_selected_loss_ci_high",
        engine + "_selected_loss_max",
    )
)
SEARCH_PAIRED_FIELDS = (
    "paired_n",
    "paired_n_clusters",
    "paired_cluster_fallback_rows",
    "paired_bootstrap_seed",
) + tuple(
    field
    for metric in PAIRED_DEPTH_METRICS
    for field in (metric, metric + "_ci_low", metric + "_ci_high")
) + (
    "paired_selected_loss_n",
    "egaroucid_minus_edax_selected_loss",
    "egaroucid_minus_edax_selected_loss_ci_low",
    "egaroucid_minus_edax_selected_loss_ci_high",
    "best_move_reference_note",
)
SEARCH_FIELDS = SEARCH_BASE_FIELDS + SEARCH_ENGINE_FIELDS + SEARCH_PAIRED_FIELDS

LABEL_DISTRIBUTION_LEGACY_FIELDS = (
    "aggregation",
    "phase",
    "empties",
    "data_id",
    "source_category",
    "random_moves",
    "moves_after_random",
    "share_of_phase",
    "records",
    "mean",
    "stddev",
    "positive_fraction",
    "zero_fraction",
    "negative_fraction",
    "mean_abs_score",
    "median",
    "min_score",
    "max_score",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "band_-64..-21",
    "band_-20..-11",
    "band_-10..-5",
    "band_-4..-1",
    "band_0",
    "band_+1..+4",
    "band_+5..+10",
    "band_+11..+20",
    "band_+21..+64",
)

LABEL_DISTRIBUTION_ADDED_FIELDS = (
    "distribution_origin",
    "population",
    "label_aggregation",
    "quantile_definition",
    "stddev_definition",
    "data_scope",
    "share_of_population",
    "share_denominator_records",
    "share_denominator_definition",
    "generation_method",
    "teacher_method",
    "provenance_status",
    "unclassified_half_boundary",
    "input_artifact",
)
LABEL_DISTRIBUTION_FIELDS = (
    LABEL_DISTRIBUTION_LEGACY_FIELDS + LABEL_DISTRIBUTION_ADDED_FIELDS
)


def copy_dedup_distribution_metrics(
    destination: MutableMapping[str, Any], source: Mapping[str, Any]
) -> None:
    mapping = {
        "records": "count",
        "mean": "mean",
        "stddev": "stddev",
        "positive_fraction": "positive_ratio",
        "zero_fraction": "zero_ratio",
        "negative_fraction": "negative_ratio",
        "mean_abs_score": "mean_abs",
        "median": "median",
        "p10": "p10",
        "p25": "p25",
        "p50": "p50",
        "p75": "p75",
        "p90": "p90",
        "p95": "p95",
        "p99": "p99",
        "band_-64..-21": "count_-64_-21",
        "band_-20..-11": "count_-20_-11",
        "band_-10..-5": "count_-10_-5",
        "band_-4..-1": "count_-4_-1",
        "band_0": "count_0",
        "band_+1..+4": "count_1_4",
        "band_+5..+10": "count_5_10",
        "band_+11..+20": "count_11_20",
        "band_+21..+64": "count_21_64",
        "unclassified_half_boundary": "unclassified_half_boundary",
    }
    for output_name, input_name in mapping.items():
        destination[output_name] = source.get(input_name, "")


def build_label_distribution(
    input_dir: Path, state: RunState
) -> List[Dict[str, Any]]:
    legacy_path = input_dir / "label_distribution.csv"
    by_id_path = input_dir / "training_dedup_by_id_phase_id_distribution_summary.csv"
    union_path = input_dir / "training_dedup_by_id_distribution_summary.csv"
    manifest_path = input_dir / "data_manifest.json"
    legacy_rows = read_csv_rows(legacy_path, "label_distribution_legacy", state, required=True)
    by_id_rows = read_csv_rows(by_id_path, "dedup_within_data_id_distribution", state, required=True)
    union_rows = read_csv_rows(union_path, "dedup_all_ids_distribution", state, required=True)
    manifest = read_json_object(manifest_path, "training_data_manifest", state, required=True)

    metadata: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    for entry in manifest.get("input_files", []):
        phase = as_int(entry.get("phase"))
        data_id = as_int(entry.get("data_id"))
        if phase is not None and data_id is not None:
            metadata[(phase, data_id)] = entry

    # Idempotence: retain every original phase-ID frequency row, but do not
    # append previously generated D4 rows again on --overwrite reruns.
    retained: List[Dict[str, Any]] = []
    generated_origins = {
        "within_data_id_d4_unique",
        "all_selected_ids_d4_unique",
    }
    for row in legacy_rows:
        if row.get("distribution_origin", "") in generated_origins:
            continue
        normalized = dict(row)
        phase = as_int(row.get("phase"))
        data_id = as_int(row.get("data_id"))
        info = metadata.get((phase, data_id)) if phase is not None and data_id is not None else None
        default_origin = (
            "phase_id_training_frequency"
            if data_id is not None
            else "phase_all_selected_ids_training_frequency"
        )
        normalized["distribution_origin"] = row.get(
            "distribution_origin", default_origin
        ) or default_origin
        normalized["population"] = row.get("population", "records_with_duplicates") or "records_with_duplicates"
        normalized["label_aggregation"] = row.get("label_aggregation", "none") or "none"
        normalized["quantile_definition"] = row.get(
            "quantile_definition", "numpy_linear_interpolation"
        ) or "numpy_linear_interpolation"
        normalized["stddev_definition"] = row.get(
            "stddev_definition", "population_ddof_0"
        ) or "population_ddof_0"
        default_scope = (
            "per_phase_data_id"
            if data_id is not None
            else "per_phase_all_selected_ids_aggregate"
        )
        normalized["data_scope"] = row.get("data_scope", default_scope) or default_scope
        normalized["share_of_population"] = row.get(
            "share_of_population", row.get("share_of_phase", "")
        )
        if info:
            normalized["generation_method"] = row.get("generation_method", "") or info.get(
                "generation_method", ""
            )
            normalized["teacher_method"] = row.get("teacher_method", "") or info.get(
                "teacher_method", ""
            )
            normalized["provenance_status"] = row.get("provenance_status", "") or info.get(
                "provenance_status", ""
            )
        normalized["unclassified_half_boundary"] = row.get(
            "unclassified_half_boundary", "0"
        ) or "0"
        normalized["input_artifact"] = row.get("input_artifact", legacy_path.name) or legacy_path.name
        retained.append(normalized)

    raw_denominators: Counter[int] = Counter()
    for row in retained:
        phase = as_int(row.get("phase"))
        data_id = as_int(row.get("data_id"))
        records = as_int(row.get("records"))
        if phase is not None and data_id is not None and records is not None:
            raw_denominators[phase] += records
    for row in retained:
        phase = as_int(row.get("phase"))
        if phase is not None:
            row["share_denominator_records"] = row.get(
                "share_denominator_records", raw_denominators[phase]
            )
            row["share_denominator_definition"] = row.get(
                "share_denominator_definition",
                "sum of records_with_duplicates over selected numeric data IDs in this phase; data_id=ALL is the aggregate row and is excluded from the sum",
            )

    unique_by_id = [
        row
        for row in by_id_rows
        if row.get("population") == "unique_canonical_positions_within_data_id"
        and row.get("aggregation") == "standard_median"
    ]
    unique_denominators: Counter[int] = Counter()
    for row in unique_by_id:
        phase = as_int(row.get("phase"))
        count = as_int(row.get("count"))
        if phase is not None and count is not None:
            unique_denominators[phase] += count

    appended: List[Dict[str, Any]] = []
    for source in unique_by_id:
        phase = as_int(source.get("phase"))
        data_id = as_int(source.get("data_id"))
        count = as_int(source.get("count"))
        if phase is None or data_id is None or count is None:
            continue
        info = metadata.get((phase, data_id), {})
        denominator = unique_denominators[phase]
        row: Dict[str, Any] = {
            "aggregation": source.get("aggregation"),
            "phase": phase,
            "empties": info.get("empties", 60 - phase),
            "data_id": data_id,
            "source_category": info.get("source_category", ""),
            "random_moves": info.get("random_moves", ""),
            "moves_after_random": info.get("moves_after_random", ""),
            "share_of_phase": count / denominator if denominator else None,
            "distribution_origin": "within_data_id_d4_unique",
            "population": source.get("population"),
            "label_aggregation": source.get("aggregation"),
            "quantile_definition": "nearest_rank",
            "stddev_definition": "population_ddof_0",
            "data_scope": "per_phase_data_id; D4 canonicalized only within each data ID",
            "share_of_population": count / denominator if denominator else None,
            "share_denominator_records": denominator,
            "share_denominator_definition": (
                "sum of unique_canonical_positions_within_data_id counts over selected data IDs in this phase; "
                "the same board in different data IDs contributes once to each ID"
            ),
            "generation_method": info.get("generation_method", ""),
            "teacher_method": info.get("teacher_method", ""),
            "provenance_status": info.get("provenance_status", ""),
            "input_artifact": by_id_path.name,
        }
        copy_dedup_distribution_metrics(row, source)
        appended.append(row)

    for source in union_rows:
        if source.get("population") != "unique_canonical_positions" or source.get(
            "aggregation"
        ) != "standard_median":
            continue
        phase = as_int(source.get("phase"))
        count = as_int(source.get("count"))
        if phase is None or count is None:
            continue
        row = {
            "aggregation": source.get("aggregation"),
            "phase": phase,
            "empties": 60 - phase,
            "data_id": "",
            "source_category": "mixed_all_selected_data_ids",
            "random_moves": "",
            "moves_after_random": "",
            "share_of_phase": 1.0,
            "distribution_origin": "all_selected_ids_d4_unique",
            "population": source.get("population"),
            "label_aggregation": source.get("aggregation"),
            "quantile_definition": "nearest_rank",
            "stddev_definition": "population_ddof_0",
            "data_scope": "per phase union of all selected data IDs; D4 canonicalized across data-ID boundaries",
            "share_of_population": 1.0,
            "share_denominator_records": count,
            "share_denominator_definition": (
                "all unique_canonical_positions after unioning selected data IDs in this phase"
            ),
            "generation_method": "mixed sources; see per-data-ID rows and data_manifest.json",
            "teacher_method": "mixed source rows; conflicting labels aggregated by standard median",
            "provenance_status": "derived_from_local_board_data_union",
            "input_artifact": union_path.name,
        }
        copy_dedup_distribution_metrics(row, source)
        appended.append(row)

    appended.sort(
        key=lambda row: (
            as_int(row.get("phase")) or -1,
            row.get("distribution_origin", ""),
            as_int(row.get("data_id")) if as_int(row.get("data_id")) is not None else -1,
        )
    )
    return retained + appended


def egaroucid_depth_measurement(
    sample: Mapping[str, Any], depth: int
) -> Optional[Dict[str, Any]]:
    root = sample.get("root")
    if not root:
        return None
    if not is_true(root.get(f"depth_{depth}_complete")):
        return None
    value = as_float(root.get(f"depth_{depth}_value"))
    if value is None:
        return None
    return {
        "value": value,
        "best_move": normalize_move(root.get(f"depth_{depth}_best_move")),
        "pv": str(root.get(f"depth_{depth}_pv_moves", "") or "").strip().upper(),
        "leaf_phase": as_int(root.get(f"depth_{depth}_pv_leaf_phase")),
    }


def edax_depth_measurement(
    sample: Mapping[str, Any], depth: int
) -> Optional[Dict[str, Any]]:
    row = sample.get("edax_depth_rows", {}).get(depth)
    if not row:
        return None
    value = as_float(row.get("value"))
    if value is None:
        return None
    return {
        "value": value,
        "best_move": normalize_move(row.get("best_move")),
        "pv": str(row.get("pv", "") or "").strip().upper(),
        "leaf_phase": None,
    }


def paired_depth_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    valid = [
        (
            as_float(row.get("v")),
            as_float(row.get("egaroucid")),
            as_float(row.get("edax")),
        )
        for row in rows
    ]
    valid = [(v, e, d) for v, e, d in valid if v is not None and e is not None and d is not None]
    if not valid:
        return {metric: None for metric in PAIRED_DEPTH_METRICS}
    return {
        "egaroucid_minus_edax_bias": statistics.fmean((e - v) - (d - v) for v, e, d in valid),
        "egaroucid_minus_edax_absolute_error": statistics.fmean(
            abs(e - v) - abs(d - v) for v, e, d in valid
        ),
        "egaroucid_minus_edax_squared_error": statistics.fmean(
            (e - v) ** 2 - (d - v) ** 2 for v, e, d in valid
        ),
        "egaroucid_minus_edax_value": statistics.fmean(e - d for _v, e, d in valid),
    }


def scalar_mean_metric(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    values = [as_float(row.get("scalar")) for row in rows]
    valid = [value for value in values if value is not None]
    return {"mean": statistics.fmean(valid) if valid else None}


def reference_moves_and_note(
    sample: Mapping[str, Any],
    exact_children: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Tuple[Set[str], str]:
    dataset = str(sample.get("dataset", ""))
    sample_id = str(sample.get("sample_id", ""))
    child_values = exact_children.get(dataset, {}).get(sample_id, {})
    if child_values:
        best = max(child_values.values())
        return (
            {move for move, value in child_values.items() if value == best},
            "all exact-complete child values; ties included",
        )
    moves = set(sample.get("exact_best_moves", set()))
    if moves:
        return moves, "single exact PV move(s); unobserved ties possible"
    return set(), "no exact best-move reference"


def selected_move_loss(
    sample: Mapping[str, Any],
    move: str,
    exact_children: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Optional[float]:
    values = exact_children.get(str(sample.get("dataset", "")), {}).get(
        str(sample.get("sample_id", "")), {}
    )
    normalized = normalize_move(move)
    if not values or normalized not in values:
        return None
    return max(values.values()) - values[normalized]


def previous_depth_counts(
    dataset_samples: Sequence[Mapping[str, Any]],
    engine: str,
    depth: int,
    previous_depth: Optional[int],
    reference_key: str,
) -> Dict[str, int]:
    result = {
        "previous_comparable_n": 0,
        "worsened_n": 0,
        "improved_n": 0,
        "unchanged_n": 0,
        "best_move_changed_n": 0,
        "best_move_comparable_n": 0,
        "pv_prefix_changed_n": 0,
        "pv_comparable_n": 0,
    }
    if previous_depth is None:
        return result
    getter = egaroucid_depth_measurement if engine == "egaroucid" else edax_depth_measurement
    for sample in dataset_samples:
        exact = as_float(sample.get(reference_key))
        current = getter(sample, depth)
        previous = getter(sample, previous_depth)
        if exact is None or current is None or previous is None:
            continue
        result["previous_comparable_n"] += 1
        current_error = abs(float(current["value"]) - exact)
        previous_error = abs(float(previous["value"]) - exact)
        if current_error > previous_error:
            result["worsened_n"] += 1
        elif current_error < previous_error:
            result["improved_n"] += 1
        else:
            result["unchanged_n"] += 1
        if current["best_move"] and previous["best_move"]:
            result["best_move_comparable_n"] += 1
            result["best_move_changed_n"] += int(
                current["best_move"] != previous["best_move"]
            )
        if current["pv"] and previous["pv"]:
            result["pv_comparable_n"] += 1
            current_pv = str(current["pv"]).split()
            previous_pv = str(previous["pv"]).split()
            result["pv_prefix_changed_n"] += int(
                current_pv[: len(previous_pv)] != previous_pv
            )
    return result


def mode_or_none(values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    counts = Counter(values)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def _build_search_depth_summary_core(
    samples: Mapping[str, Mapping[str, Mapping[str, Any]]],
    exact_children: Mapping[str, Mapping[str, Mapping[str, float]]],
    reps: int,
    seed: int,
    group_dimension: str = "all",
    group_value: str = "all",
    include_missing: bool = True,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    expected_datasets = [f"phase{phase}" for phase in PHASES] + [
        "ggs_latest96",
        "ggs_seeded96",
    ]
    for dataset in expected_datasets:
        dataset_samples = list(samples.get(dataset, {}).values())
        phase = dataset_phase(dataset)
        reference_key = "teacher" if dataset == "phase36" else "exact_value"
        reference_name = "z_teacher" if dataset == "phase36" else "v_exact"
        if not dataset_samples:
            if include_missing:
                output.append(
                    {
                        "dataset": dataset,
                        "phase": phase,
                        "empties": 60 - phase if phase is not None else None,
                        "group_dimension": group_dimension,
                        "group_value": group_value,
                        "status": "missing_or_empty_engine_input",
                        "expected_sample_n": EXPECTED_DATASET_SAMPLES.get(dataset),
                        "dataset_sample_n": 0,
                        "exact_value_n": 0,
                        "bootstrap_reps": 0,
                        "ci_method": "source_file+game_index cluster bootstrap percentile 95%",
                    }
                )
            continue
        exact_n = sum(
            as_float(sample.get("exact_value")) is not None
            for sample in dataset_samples
        )
        reference_n = sum(
            as_float(sample.get(reference_key)) is not None
            for sample in dataset_samples
        )
        target_n = (
            EXPECTED_DATASET_SAMPLES.get(dataset, len(dataset_samples))
            if group_dimension == "all"
            else len(dataset_samples)
        )
        for depth_index, depth in enumerate(DEPTHS):
            previous_depth = DEPTHS[depth_index - 1] if depth_index else None
            ega_records: List[Dict[str, Any]] = []
            edax_records: List[Dict[str, Any]] = []
            paired_records: List[Dict[str, Any]] = []
            ega_losses: List[Dict[str, Any]] = []
            edax_losses: List[Dict[str, Any]] = []
            ega_matches: List[Dict[str, Any]] = []
            edax_matches: List[Dict[str, Any]] = []
            paired_losses: List[Dict[str, Any]] = []
            leaf_phases: List[int] = []
            reference_notes: Counter[str] = Counter()
            move_counts = {
                "egaroucid_best_move_reference_n": 0,
                "egaroucid_best_move_match_n": 0,
                "edax_best_move_reference_n": 0,
                "edax_best_move_match_n": 0,
            }
            any_measurement = False
            for sample in dataset_samples:
                exact = as_float(sample.get(reference_key))
                ega = egaroucid_depth_measurement(sample, depth)
                edax_measurement = edax_depth_measurement(sample, depth)
                any_measurement = any_measurement or ega is not None or edax_measurement is not None
                if ega is not None and ega["leaf_phase"] is not None:
                    leaf_phases.append(int(ega["leaf_phase"]))
                if exact is None:
                    continue
                metadata = sample_metadata(sample)
                if ega is not None:
                    ega_records.append({**metadata, "x": exact, "y": ega["value"]})
                if edax_measurement is not None:
                    edax_records.append(
                        {**metadata, "x": exact, "y": edax_measurement["value"]}
                    )
                if ega is not None and edax_measurement is not None:
                    paired_records.append(
                        {
                            **metadata,
                            "v": exact,
                            "egaroucid": ega["value"],
                            "edax": edax_measurement["value"],
                        }
                    )
                if depth == 0:
                    # A zero-ply result is the root's static value and does
                    # not select a move.  Edax happens to print a move at
                    # level 0, but counting it only for Edax would make the
                    # two engines' selection metrics asymmetric.
                    reference_notes["depth0_static_has_no_move_selection"] += 1
                else:
                    references, note = reference_moves_and_note(sample, exact_children)
                    reference_notes[note] += 1
                    for engine, measurement in (
                        ("egaroucid", ega),
                        ("edax", edax_measurement),
                    ):
                        if measurement is None:
                            continue
                        move = measurement["best_move"]
                        if references and move:
                            move_counts[engine + "_best_move_reference_n"] += 1
                            matched = int(move in references)
                            move_counts[engine + "_best_move_match_n"] += matched
                            match_target = (
                                ega_matches if engine == "egaroucid" else edax_matches
                            )
                            match_target.append({**metadata, "scalar": matched})
                        loss = selected_move_loss(sample, move, exact_children)
                        if loss is not None:
                            target = (
                                ega_losses if engine == "egaroucid" else edax_losses
                            )
                            target.append({**metadata, "scalar": loss})
                    if ega is not None and edax_measurement is not None:
                        ega_loss = selected_move_loss(
                            sample, ega["best_move"], exact_children
                        )
                        edax_loss = selected_move_loss(
                            sample, edax_measurement["best_move"], exact_children
                        )
                        if ega_loss is not None and edax_loss is not None:
                            paired_losses.append(
                                {**metadata, "scalar": ega_loss - edax_loss}
                            )
            if not any_measurement:
                continue

            row: Dict[str, Any] = {
                "dataset": dataset,
                "phase": phase,
                "empties": 60 - phase if phase is not None else None,
                "group_dimension": group_dimension,
                "group_value": group_value,
                "depth": depth,
                "depth_parity": "even" if depth % 2 == 0 else "odd",
                "expected_leaf_phase": phase + depth if phase is not None else None,
                "egaroucid_leaf_phase_mode": mode_or_none(leaf_phases),
                "egaroucid_leaf_phase_min": min(leaf_phases) if leaf_phases else None,
                "egaroucid_leaf_phase_max": max(leaf_phases) if leaf_phases else None,
                "reference_value_name": reference_name,
                "reference_value_n": reference_n,
                "status": (
                    "missing_reference_values"
                    if reference_n == 0
                    else "complete_paired"
                    if len(paired_records) == target_n and reference_n == target_n
                    else "partial_reference_or_engine_results"
                ),
                "expected_sample_n": target_n,
                "dataset_sample_n": len(dataset_samples),
                "exact_value_n": exact_n,
                "previous_depth": previous_depth,
                "ci_method": "source_file+game_index cluster bootstrap percentile 95%",
                "bootstrap_reps": reps,
                **move_counts,
            }

            for engine, records in (("egaroucid", ega_records), ("edax", edax_records)):
                point = comparison_metrics(records)
                context = (
                    f"{dataset}|search|{group_dimension}={group_value}|{engine}|depth{depth}"
                )
                intervals, clusters, fallback = bootstrap_metrics(
                    records,
                    comparison_metrics,
                    COMPARISON_METRICS,
                    reps,
                    seed,
                    context,
                )
                row[engine + "_n"] = len(records)
                row[engine + "_n_clusters"] = clusters
                row[engine + "_cluster_fallback_rows"] = fallback
                row[engine + "_bootstrap_seed"] = derived_seed(seed, context)
                for metric in COMPARISON_METRICS:
                    row[engine + "_" + metric] = point[metric]
                    row[engine + "_" + metric + "_ci_low"] = intervals[metric][0]
                    row[engine + "_" + metric + "_ci_high"] = intervals[metric][1]
                previous = previous_depth_counts(
                    dataset_samples, engine, depth, previous_depth, reference_key
                )
                for name, count in previous.items():
                    row[engine + "_" + name] = count
                denominator = move_counts[engine + "_best_move_reference_n"]
                row[engine + "_best_move_match_rate"] = (
                    move_counts[engine + "_best_move_match_n"] / denominator
                    if denominator
                    else None
                )
                matches = ega_matches if engine == "egaroucid" else edax_matches
                match_context = (
                    f"{dataset}|search|{group_dimension}={group_value}|{engine}|"
                    f"best_move_match|depth{depth}"
                )
                match_intervals, _match_clusters, _match_fallback = bootstrap_metrics(
                    matches,
                    scalar_mean_metric,
                    ("mean",),
                    reps,
                    seed,
                    match_context,
                )
                row[engine + "_best_move_match_rate_ci_low"] = match_intervals["mean"][0]
                row[engine + "_best_move_match_rate_ci_high"] = match_intervals["mean"][1]
                losses = ega_losses if engine == "egaroucid" else edax_losses
                loss_point = scalar_mean_metric(losses)
                loss_context = (
                    f"{dataset}|search|{group_dimension}={group_value}|{engine}|"
                    f"selected_loss|depth{depth}"
                )
                loss_intervals, _loss_clusters, _loss_fallback = bootstrap_metrics(
                    losses,
                    scalar_mean_metric,
                    ("mean",),
                    reps,
                    seed,
                    loss_context,
                )
                loss_values = [float(item["scalar"]) for item in losses]
                row[engine + "_selected_loss_n"] = len(losses)
                row[engine + "_selected_loss_mean"] = loss_point["mean"]
                row[engine + "_selected_loss_ci_low"] = loss_intervals["mean"][0]
                row[engine + "_selected_loss_ci_high"] = loss_intervals["mean"][1]
                row[engine + "_selected_loss_max"] = max(loss_values) if loss_values else None

            paired_point = paired_depth_metrics(paired_records)
            paired_context = (
                f"{dataset}|search|{group_dimension}={group_value}|paired|depth{depth}"
            )
            paired_intervals, paired_clusters, paired_fallback = bootstrap_metrics(
                paired_records,
                paired_depth_metrics,
                PAIRED_DEPTH_METRICS,
                reps,
                seed,
                paired_context,
            )
            row["paired_n"] = len(paired_records)
            row["paired_n_clusters"] = paired_clusters
            row["paired_cluster_fallback_rows"] = paired_fallback
            row["paired_bootstrap_seed"] = derived_seed(seed, paired_context)
            for metric in PAIRED_DEPTH_METRICS:
                row[metric] = paired_point[metric]
                row[metric + "_ci_low"] = paired_intervals[metric][0]
                row[metric + "_ci_high"] = paired_intervals[metric][1]

            paired_loss_point = scalar_mean_metric(paired_losses)
            paired_loss_context = (
                f"{dataset}|search|{group_dimension}={group_value}|"
                f"paired_selected_loss|depth{depth}"
            )
            paired_loss_intervals, _clusters, _fallback = bootstrap_metrics(
                paired_losses,
                scalar_mean_metric,
                ("mean",),
                reps,
                seed,
                paired_loss_context,
            )
            row["paired_selected_loss_n"] = len(paired_losses)
            row["egaroucid_minus_edax_selected_loss"] = paired_loss_point["mean"]
            row["egaroucid_minus_edax_selected_loss_ci_low"] = paired_loss_intervals[
                "mean"
            ][0]
            row["egaroucid_minus_edax_selected_loss_ci_high"] = paired_loss_intervals[
                "mean"
            ][1]
            row["best_move_reference_note"] = "; ".join(
                f"{note}:n={count}" for note, count in sorted(reference_notes.items())
            )
            output.append(row)
    return output


def legal_count_bin(value: Optional[int]) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0_pass_or_terminal"
    if value <= 2:
        return "1..2"
    if value <= 4:
        return "3..4"
    if value <= 6:
        return "5..6"
    if value <= 8:
        return "7..8"
    return "9_plus"


def build_search_depth_summary(
    samples: Mapping[str, Mapping[str, Mapping[str, Any]]],
    exact_children: Mapping[str, Mapping[str, Mapping[str, float]]],
    reps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    output = _build_search_depth_summary_core(
        samples, exact_children, reps, seed, include_missing=True
    )
    for dataset in sorted(samples):
        dataset_samples = list(samples[dataset].values())
        reference_key = "teacher" if dataset == "phase36" else "exact_value"
        sign_dimension = "teacher_sign" if dataset == "phase36" else "exact_sign"
        band_dimension = (
            "teacher_score_band" if dataset == "phase36" else "exact_score_band"
        )
        dimensions: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {
            "source_category": defaultdict(list),
            sign_dimension: defaultdict(list),
            band_dimension: defaultdict(list),
            "root_legal_count_bin": defaultdict(list),
        }
        for sample in dataset_samples:
            source_category = str(sample.get("source_category", "") or "")
            reference = as_float(sample.get(reference_key))
            reference_sign = score_sign(reference)
            reference_band = score_band(reference)
            legal_bin = legal_count_bin(as_int(sample.get("root_legal_count")))
            if source_category:
                dimensions["source_category"][source_category].append(sample)
            if reference_sign:
                dimensions[sign_dimension][reference_sign].append(sample)
            if reference_band:
                dimensions[band_dimension][reference_band].append(sample)
            if legal_bin:
                dimensions["root_legal_count_bin"][legal_bin].append(sample)
        for dimension, groups in dimensions.items():
            for value in sorted(groups):
                subset = {dataset: {
                    str(sample["sample_id"]): sample for sample in groups[value]
                }}
                output.extend(
                    _build_search_depth_summary_core(
                        subset,
                        exact_children,
                        reps,
                        seed,
                        group_dimension=dimension,
                        group_value=value,
                        include_missing=False,
                    )
                )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise AggregationError(f"input directory does not exist: {input_dir}")

    output_paths = {
        "label_vs_exact": output_dir / "label_vs_exact.csv",
        "evaluator_vs_label_vs_exact": output_dir
        / "evaluator_vs_label_vs_exact.csv",
        "phase_summary": output_dir / "phase_summary.csv",
        "sibling_ranking": output_dir / "sibling_ranking.csv",
        "search_depth_comparison": output_dir / "search_depth_comparison.csv",
        "label_distribution": output_dir / "label_distribution.csv",
        "manifest": output_dir / "aggregation_manifest.json",
    }
    for path in output_paths.values():
        prepare_output(path, args.overwrite)

    state = RunState(args.strict_missing)
    print("loading root and Edax TSVs...", file=sys.stderr)
    roots, exact_companions = load_roots(input_dir, state)
    edax, edax_child_results = load_edax(input_dir, state)
    samples = normalize_samples(roots, exact_companions, edax, state)
    print("loading exact/sibling child TSVs...", file=sys.stderr)
    exact_children, sibling_parents = load_child_measurements(
        input_dir, state, edax_child_results, samples
    )
    print("merging frequency and D4-unique label distributions...", file=sys.stderr)
    label_distribution_rows = build_label_distribution(input_dir, state)

    label_rows, evaluator_rows = build_row_level_outputs(samples)
    print("bootstrapping phase/group comparisons...", file=sys.stderr)
    phase_rows = build_phase_summary(samples, args.bootstrap_reps, args.seed)
    print("bootstrapping sibling comparisons...", file=sys.stderr)
    sibling_rows = build_sibling_summary(
        sibling_parents, samples, args.bootstrap_reps, args.seed
    )
    print("bootstrapping search-depth comparisons...", file=sys.stderr)
    search_rows = build_search_depth_summary(
        samples, exact_children, args.bootstrap_reps, args.seed
    )

    write_csv(
        output_paths["label_vs_exact"], LABEL_FIELDS, label_rows, args.overwrite
    )
    write_csv(
        output_paths["evaluator_vs_label_vs_exact"],
        EVALUATOR_FIELDS,
        evaluator_rows,
        args.overwrite,
    )
    write_csv(
        output_paths["phase_summary"], SUMMARY_FIELDS, phase_rows, args.overwrite
    )
    write_csv(
        output_paths["sibling_ranking"],
        SIBLING_FIELDS,
        sibling_rows,
        args.overwrite,
    )
    write_csv(
        output_paths["search_depth_comparison"],
        SEARCH_FIELDS,
        search_rows,
        args.overwrite,
    )
    write_csv(
        output_paths["label_distribution"],
        LABEL_DISTRIBUTION_FIELDS,
        label_distribution_rows,
        args.overwrite,
    )

    generated_outputs = []
    for role, path in output_paths.items():
        if role == "manifest":
            continue
        generated_outputs.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sampling_seed": args.seed,
        "bootstrap_reps": args.bootstrap_reps,
        "confidence_interval": {
            "method": "percentile cluster bootstrap",
            "confidence": 0.95,
            "cluster": ["source_file", "game_index_in_file"],
            "fallback": "sample_id when source_file or game_index_in_file is missing",
            "percentile_interpolation": "linear at (n-1)*p",
            "undefined_replicates": "a confidence interval is left empty unless its metric is defined in every requested bootstrap replicate",
            "context_seed": "first 64 bits little-endian of SHA-256(base_seed + NUL + context)",
        },
        "definitions": {
            "regression": "prediction = slope * reference + intercept (ordinary least squares)",
            "bias": "mean(prediction - reference)",
            "mean_abs_delta": "mean(abs(prediction) - abs(reference))",
            "same_sign_smaller_abs_rate": "signs equal and nonzero, and abs(prediction) < abs(reference), divided by all rows",
            "search_paired_difference": "Egaroucid metric minus Edax metric on the same completed positions",
            "worsened": "absolute error at current listed depth exceeds absolute error at the previous listed depth",
            "pv_prefix_changed": "the complete previous-depth PV token sequence is not a prefix of the current-depth PV; unlike full-string comparison, mere PV extension is not counted as a change",
            "selected_loss": "maximum exact child value minus exact value of selected child, parent-side score",
            "sibling_rank": "strict exact unequal sibling pairs; predicted ties are reported separately",
            "sibling_residual_correlation": "Pearson correlation after adding both orientations of every unordered sibling residual pair",
            "phase36_search_reference": "phase36 has no exact root values; its search errors and teacher_sign/teacher_score_band groups use z_teacher as the reference, not v_exact",
        },
        "row_counts": {
            "normalized_datasets": {
                dataset: len(dataset_samples)
                for dataset, dataset_samples in sorted(samples.items())
            },
            "label_vs_exact": len(label_rows),
            "evaluator_vs_label_vs_exact": len(evaluator_rows),
            "phase_summary": len(phase_rows),
            "sibling_ranking": len(sibling_rows),
            "search_depth_comparison": len(search_rows),
            "label_distribution": len(label_distribution_rows),
        },
        "inputs": state.inputs,
        "warnings": state.warnings,
        "counters": dict(sorted(state.counters.items())),
        "outputs": generated_outputs,
    }
    write_json(output_paths["manifest"], manifest, args.overwrite)

    print(
        "aggregated: "
        f"label_rows={len(label_rows)} evaluator_rows={len(evaluator_rows)} "
        f"phase_summary_rows={len(phase_rows)} sibling_rows={len(sibling_rows)} "
        f"search_rows={len(search_rows)} label_distribution_rows={len(label_distribution_rows)}"
    )
    print(f"output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AggregationError as error:
        print("error: " + str(error), file=sys.stderr)
        raise SystemExit(1)
