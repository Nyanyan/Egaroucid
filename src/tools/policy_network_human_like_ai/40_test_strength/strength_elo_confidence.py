#!/usr/bin/env python3
"""XOT-cluster bootstrap confidence intervals for tournament Elo ratings."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BIN_DIR = REPO_ROOT / "bin"

sys.path.insert(0, str(BIN_DIR))

from elo_rating_backcal import fit_elo_from_winrates  # noqa: E402
from strength_tournament import (  # noqa: E402
    MatchSetResult,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_SEED = 613
DEFAULT_CONFIDENCE = 0.95
MINIMUM_CLUSTERS = 3


@dataclass(frozen=True)
class ClusteredEloData:
    experiment_id: str
    names: Tuple[str, ...]
    pairs: Tuple[Tuple[int, int], ...]
    set_indices: Tuple[int, ...]
    openings: Tuple[str, ...]
    paired_scores: np.ndarray
    raw_results_sha256: str
    manifest_sha256: str
    aggregate_sha256: str
    existing_point_elos: Mapping[str, float]


@dataclass(frozen=True)
class BcaInterval:
    low: float
    high: float
    bias_correction: float
    acceleration: float
    lower_quantile: float
    upper_quantile: float
    method: str
    fallback_reason: Optional[str]


@dataclass(frozen=True)
class ClusteredEloAnalysis:
    point_elos: np.ndarray
    intervals: Tuple[BcaInterval, ...]
    bootstrap_elos: np.ndarray
    bootstrap_cluster_rows: np.ndarray
    jackknife_elos: np.ndarray


def _canonical_task_identity(task: Mapping[str, object]) -> dict:
    return {
        "task_id": int(task["task_id"]),
        "p0_idx": int(task["p0_idx"]),
        "p1_idx": int(task["p1_idx"]),
        "set_index": int(task["set_index"]),
        "opening": str(task["opening"]),
    }


def load_clustered_elo_data(output_dir: Path) -> ClusteredEloData:
    """Load and strictly validate the balanced set-index/XOT result tensor."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "strength_manifest.json"
    aggregate_path = output_dir / "strength_results.json"
    raw_path = output_dir / "strength_games.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if int(aggregate.get("schema_version", -1)) != 3:
        raise ValueError("strength_results.json has an unsupported schema")
    experiment_id = str(manifest.get("experiment_id", ""))
    if not experiment_id:
        raise ValueError("strength_manifest.json has no experiment_id")
    if str(aggregate.get("experiment_id", "")) != experiment_id:
        raise ValueError("strength_results.json experiment_id mismatch")

    player_rows = aggregate.get("players")
    if not isinstance(player_rows, list) or len(player_rows) < 2:
        raise ValueError("strength_results.json has no valid player list")
    names = tuple(str(row["name"]) for row in player_rows)
    if len(set(names)) != len(names):
        raise ValueError("player names must be unique")
    player_count = len(names)
    pairs = tuple(
        (player_idx, opponent_idx)
        for player_idx in range(player_count)
        for opponent_idx in range(player_idx + 1, player_count)
    )
    pair_to_column = {pair: column for column, pair in enumerate(pairs)}
    completed_match_sets = int(aggregate.get("completed_match_sets", -1))
    total_match_sets = int(aggregate.get("total_match_sets", -1))
    if completed_match_sets != total_match_sets or total_match_sets < 1:
        raise ValueError("Elo intervals require a completed tournament")
    if int(aggregate.get("completed_actual_games", -1)) != 2 * total_match_sets:
        raise ValueError("completed actual-game count is inconsistent")
    if int(aggregate.get("total_actual_games", -1)) != 2 * total_match_sets:
        raise ValueError("planned actual-game count is inconsistent")

    existing_point_elos: Dict[str, float] = {}
    summary_rows = aggregate.get("summary")
    if not isinstance(summary_rows, list):
        raise ValueError("strength_results.json has no summary")
    for row in summary_rows:
        name = str(row.get("name", ""))
        value = row.get("paired_set_elo_descriptive")
        if name in names and value is not None:
            existing_point_elos[name] = float(value)
    if set(existing_point_elos) != set(names):
        raise ValueError("existing Elo point estimates are incomplete")

    by_set: Dict[int, Dict[Tuple[int, int], Tuple[float, str]]] = {}
    task_ids = set()
    with raw_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {raw_path}:{line_number}"
                ) from error
            if int(row.get("schema_version", -1)) != 2:
                raise ValueError(
                    f"unsupported result schema at line {line_number}"
                )
            task_data = row.get("task")
            result_data = row.get("result")
            if not isinstance(task_data, dict) or not isinstance(
                result_data, dict
            ):
                raise ValueError(
                    f"missing task or result at line {line_number}"
                )
            task = _canonical_task_identity(task_data)
            if task_data != task:
                raise ValueError(
                    f"non-canonical task identity at line {line_number}"
                )
            task_id = task["task_id"]
            if task_id in task_ids:
                raise ValueError(f"duplicate task_id {task_id}")
            task_ids.add(task_id)
            result = MatchSetResult.from_dict(result_data)
            if (
                result.task_id != task_id
                or result.p0_idx != task["p0_idx"]
                or result.p1_idx != task["p1_idx"]
                or result.set_index != task["set_index"]
                or result.opening != task["opening"]
            ):
                raise ValueError(
                    f"task/result identity mismatch at line {line_number}"
                )
            if not math.isfinite(result.p0_disc_diff):
                raise ValueError(
                    f"non-finite paired disc difference at line {line_number}"
                )
            color_games = {
                game.p0_is_black: game for game in result.color_games
            }
            if set(color_games) != {False, True} or any(
                (game.p0_idx, game.p1_idx)
                != (result.p0_idx, result.p1_idx)
                for game in result.color_games
            ):
                raise ValueError(
                    f"invalid color-swapped games at line {line_number}"
                )
            black_game = color_games[True]
            white_game = color_games[False]
            expected_average = (
                black_game.p0_disc_diff + white_game.p0_disc_diff
            ) / 2.0
            if (
                result.p0_black_disc_diff != black_game.p0_disc_diff
                or result.p0_white_disc_diff != white_game.p0_disc_diff
                or not math.isclose(
                    result.p0_disc_diff,
                    expected_average,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"inconsistent paired result at line {line_number}"
                )
            if not (
                0 <= result.p0_idx < player_count
                and 0 <= result.p1_idx < player_count
                and result.p0_idx != result.p1_idx
            ):
                raise ValueError(
                    f"invalid participant index at line {line_number}"
                )
            if result.p0_idx < result.p1_idx:
                pair = (result.p0_idx, result.p1_idx)
                score = result.p0_score
            else:
                pair = (result.p1_idx, result.p0_idx)
                score = 1.0 - result.p0_score
            cluster = by_set.setdefault(result.set_index, {})
            if pair in cluster:
                raise ValueError(
                    "duplicate player pair at set index "
                    f"{result.set_index}: {pair}"
                )
            cluster[pair] = (float(score), result.opening)

    set_indices = tuple(sorted(by_set))
    if len(set_indices) < MINIMUM_CLUSTERS:
        raise ValueError(
            f"at least {MINIMUM_CLUSTERS} complete XOT clusters are required"
        )
    if set_indices != tuple(range(len(set_indices))):
        raise ValueError("set indices must be contiguous and start at zero")
    paired_scores = np.empty((len(set_indices), len(pairs)), dtype=float)
    openings: List[str] = []
    expected_pairs = set(pairs)
    for cluster_row, set_index in enumerate(set_indices):
        cluster = by_set[set_index]
        if set(cluster) != expected_pairs:
            missing = sorted(expected_pairs - set(cluster))
            extra = sorted(set(cluster) - expected_pairs)
            raise ValueError(
                f"set index {set_index} is not a complete round robin: "
                f"missing={missing}, extra={extra}"
            )
        cluster_openings = {opening for _, opening in cluster.values()}
        if len(cluster_openings) != 1:
            raise ValueError(
                f"set index {set_index} uses multiple XOT openings"
            )
        openings.append(next(iter(cluster_openings)))
        for pair, (score, _) in cluster.items():
            paired_scores[cluster_row, pair_to_column[pair]] = score
    if len(set(openings)) != len(openings):
        raise ValueError("XOT openings must be unique across set indices")
    expected_total = len(set_indices) * len(pairs)
    if total_match_sets != expected_total:
        raise ValueError(
            "completed set count does not match the balanced XOT tensor"
        )
    planned_matrix = aggregate.get("planned_match_sets_by_pair")
    if not isinstance(planned_matrix, list) or len(planned_matrix) != player_count:
        raise ValueError("planned_match_sets_by_pair is not a player matrix")
    for player_idx, row in enumerate(planned_matrix):
        if not isinstance(row, list) or len(row) != player_count:
            raise ValueError("planned_match_sets_by_pair is not square")
        for opponent_idx, value in enumerate(row):
            expected = 0 if player_idx == opponent_idx else len(set_indices)
            if int(value) != expected:
                raise ValueError(
                    "planned_match_sets_by_pair is not a balanced completed "
                    "round robin"
                )
    configuration = manifest.get("configuration")
    schedule = (
        configuration.get("schedule")
        if isinstance(configuration, dict)
        else None
    )
    if not isinstance(schedule, dict):
        raise ValueError("strength_manifest.json has no schedule")
    if int(schedule.get("match_sets_per_pair", -1)) != len(set_indices):
        raise ValueError("manifest match_sets_per_pair does not match raw data")
    if int(schedule.get("total_match_sets", -1)) != expected_total:
        raise ValueError("manifest total_match_sets does not match raw data")
    if schedule.get("same_opening_sequence_for_every_pair") is not True:
        raise ValueError("manifest does not declare a shared opening schedule")

    return ClusteredEloData(
        experiment_id=experiment_id,
        names=names,
        pairs=pairs,
        set_indices=set_indices,
        openings=tuple(openings),
        paired_scores=paired_scores,
        raw_results_sha256=sha256_file(raw_path),
        manifest_sha256=sha256_file(manifest_path),
        aggregate_sha256=sha256_file(aggregate_path),
        existing_point_elos=existing_point_elos,
    )


def fit_elos_from_pair_scores(
    names: Sequence[str],
    pairs: Sequence[Tuple[int, int]],
    score_sums: Sequence[float],
    match_counts: Sequence[int],
) -> np.ndarray:
    """Fit the tournament's regularized, mean-1500 Elo estimator."""

    player_count = len(names)
    if len(pairs) != len(score_sums) or len(pairs) != len(match_counts):
        raise ValueError("pairs, score_sums, and match_counts must align")
    win_rates = np.full((player_count, player_count), 0.5, dtype=float)
    np.fill_diagonal(win_rates, np.nan)
    games = np.zeros((player_count, player_count), dtype=float)
    for (player_idx, opponent_idx), score_sum, count in zip(
        pairs,
        score_sums,
        match_counts,
    ):
        count = int(count)
        score_sum = float(score_sum)
        if count < 1:
            raise ValueError("every player pair must have at least one set")
        if not 0.0 <= score_sum <= float(count):
            raise ValueError("paired score sum is outside [0, count]")
        # This is exactly one fixed 0.5-score pseudo-observation per pair,
        # equivalent to adding 0.5 win and 0.5 loss. It is not resampled.
        smoothed_rate = (score_sum + 0.5) / (count + 1)
        win_rates[player_idx, opponent_idx] = smoothed_rate
        win_rates[opponent_idx, player_idx] = 1.0 - smoothed_rate
        games[player_idx, opponent_idx] = float(count + 1)
        games[opponent_idx, player_idx] = float(count + 1)
    ratings = fit_elo_from_winrates(
        win_rates,
        games=games,
        names=list(names),
    )
    return np.asarray([ratings[name] for name in names], dtype=float)


def make_bootstrap_cluster_rows(
    cluster_count: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if cluster_count < MINIMUM_CLUSTERS:
        raise ValueError(
            f"at least {MINIMUM_CLUSTERS} clusters are required"
        )
    if replicates < 1:
        raise ValueError("replicates must be positive")
    dtype = np.uint16 if cluster_count <= 65_536 else np.uint32
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return generator.integers(
        0,
        cluster_count,
        size=(int(replicates), cluster_count),
        dtype=dtype,
    )


def bootstrap_score_sums(
    paired_scores: np.ndarray,
    bootstrap_cluster_rows: np.ndarray,
) -> np.ndarray:
    """Apply one shared cluster resample to every player pair."""

    paired_scores = np.asarray(paired_scores, dtype=float)
    rows = np.asarray(bootstrap_cluster_rows)
    if paired_scores.ndim != 2 or rows.ndim != 2:
        raise ValueError("paired_scores and bootstrap rows must be matrices")
    if rows.shape[1] != paired_scores.shape[0]:
        raise ValueError("each bootstrap row must draw one row per cluster")
    if rows.size and (
        int(rows.min()) < 0 or int(rows.max()) >= paired_scores.shape[0]
    ):
        raise ValueError("bootstrap row contains an invalid cluster index")
    counts = np.zeros(
        (rows.shape[0], paired_scores.shape[0]),
        dtype=np.uint16 if paired_scores.shape[0] <= 65_535 else np.uint32,
    )
    for replicate_idx, sampled_rows in enumerate(rows):
        counts[replicate_idx] = np.bincount(
            sampled_rows,
            minlength=paired_scores.shape[0],
        )
    return counts @ paired_scores


def _fit_bootstrap_chunk(
    start: int,
    names: Tuple[str, ...],
    pairs: Tuple[Tuple[int, int], ...],
    score_sum_rows: np.ndarray,
    cluster_count: int,
) -> Tuple[int, np.ndarray]:
    counts = np.full(len(pairs), cluster_count, dtype=int)
    fitted = np.empty((len(score_sum_rows), len(names)), dtype=float)
    for row_idx, score_sums in enumerate(score_sum_rows):
        fitted[row_idx] = fit_elos_from_pair_scores(
            names,
            pairs,
            score_sums,
            counts,
        )
    return start, fitted


def fit_bootstrap_elos(
    names: Sequence[str],
    pairs: Sequence[Tuple[int, int]],
    score_sum_rows: np.ndarray,
    cluster_count: int,
    workers: int,
    progress: bool = False,
) -> np.ndarray:
    names = tuple(names)
    pairs = tuple(pairs)
    score_sum_rows = np.asarray(score_sum_rows, dtype=float)
    workers = max(1, int(workers))
    result = np.empty((len(score_sum_rows), len(names)), dtype=float)
    if not len(score_sum_rows):
        return result
    chunk_size = max(1, math.ceil(len(score_sum_rows) / (workers * 4)))
    chunks = [
        (start, score_sum_rows[start : start + chunk_size])
        for start in range(0, len(score_sum_rows), chunk_size)
    ]
    completed = 0
    if workers == 1:
        for start, chunk in chunks:
            _, fitted = _fit_bootstrap_chunk(
                start,
                names,
                pairs,
                chunk,
                cluster_count,
            )
            result[start : start + len(fitted)] = fitted
            completed += len(fitted)
            if progress:
                print(
                    f"bootstrap Elo fits: {completed}/{len(score_sum_rows)}",
                    flush=True,
                )
        return result

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _fit_bootstrap_chunk,
                start,
                names,
                pairs,
                chunk,
                cluster_count,
            )
            for start, chunk in chunks
        ]
        for future in as_completed(futures):
            start, fitted = future.result()
            result[start : start + len(fitted)] = fitted
            completed += len(fitted)
            if progress:
                print(
                    f"bootstrap Elo fits: {completed}/{len(score_sum_rows)}",
                    flush=True,
                )
    return result


def _percentile_interval(
    bootstrap_values: np.ndarray,
    confidence: float,
    reason: str,
) -> BcaInterval:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(
        bootstrap_values,
        [tail, 1.0 - tail],
        method="linear",
    )
    return BcaInterval(
        low=float(low),
        high=float(high),
        bias_correction=0.0,
        acceleration=0.0,
        lower_quantile=tail,
        upper_quantile=1.0 - tail,
        method="cluster_percentile_bootstrap_fallback",
        fallback_reason=reason,
    )


def bca_interval(
    point_estimate: float,
    bootstrap_values: Sequence[float],
    jackknife_values: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
) -> BcaInterval:
    """Return a pointwise BCa interval with an explicit safe fallback."""

    bootstrap = np.asarray(bootstrap_values, dtype=float)
    jackknife = np.asarray(jackknife_values, dtype=float)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if len(bootstrap) < 2 or len(jackknife) < MINIMUM_CLUSTERS:
        raise ValueError("too few bootstrap or jackknife estimates")
    if not (
        np.isfinite(point_estimate)
        and np.all(np.isfinite(bootstrap))
        and np.all(np.isfinite(jackknife))
    ):
        raise ValueError("Elo estimates must be finite")
    normal = NormalDist()
    less = float(np.count_nonzero(bootstrap < point_estimate))
    equal = float(np.count_nonzero(bootstrap == point_estimate))
    probability = (less + 0.5 * equal) / len(bootstrap)
    probability = min(
        1.0 - 0.5 / len(bootstrap),
        max(0.5 / len(bootstrap), probability),
    )
    bias_correction = normal.inv_cdf(probability)

    jackknife_mean = float(np.mean(jackknife))
    differences = jackknife_mean - jackknife
    sum_squared = float(np.sum(differences**2))
    denominator = 6.0 * sum_squared**1.5
    if denominator <= 0.0 or not math.isfinite(denominator):
        return _percentile_interval(
            bootstrap,
            confidence,
            "jackknife acceleration is undefined",
        )
    acceleration = float(np.sum(differences**3) / denominator)
    tail = (1.0 - confidence) / 2.0

    def adjusted_quantile(probability_: float) -> float:
        z = normal.inv_cdf(probability_)
        denominator_ = 1.0 - acceleration * (bias_correction + z)
        if abs(denominator_) < 1.0e-12:
            return math.nan
        return normal.cdf(
            bias_correction
            + (bias_correction + z) / denominator_
        )

    lower_quantile = adjusted_quantile(tail)
    upper_quantile = adjusted_quantile(1.0 - tail)
    if not (
        math.isfinite(lower_quantile)
        and math.isfinite(upper_quantile)
        and 0.0 <= lower_quantile < upper_quantile <= 1.0
    ):
        return _percentile_interval(
            bootstrap,
            confidence,
            "BCa adjusted quantiles are invalid",
        )
    low, high = np.quantile(
        bootstrap,
        [lower_quantile, upper_quantile],
        method="linear",
    )
    return BcaInterval(
        low=float(low),
        high=float(high),
        bias_correction=float(bias_correction),
        acceleration=acceleration,
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
        method="xot_cluster_bca_bootstrap",
        fallback_reason=None,
    )


def analyze_clustered_elos(
    data: ClusteredEloData,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    workers: int = 1,
    progress: bool = False,
) -> ClusteredEloAnalysis:
    cluster_count, pair_count = data.paired_scores.shape
    if pair_count != len(data.pairs):
        raise ValueError("paired score columns do not match player pairs")
    pair_counts = np.full(pair_count, cluster_count, dtype=int)
    point_elos = fit_elos_from_pair_scores(
        data.names,
        data.pairs,
        np.sum(data.paired_scores, axis=0),
        pair_counts,
    )
    for name, point_elo in zip(data.names, point_elos):
        if not math.isclose(
            point_elo,
            float(data.existing_point_elos[name]),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError(
                f"raw-log Elo does not match existing summary for {name}"
            )

    bootstrap_cluster_rows = make_bootstrap_cluster_rows(
        cluster_count,
        int(bootstrap_replicates),
        int(bootstrap_seed),
    )
    score_sum_rows = bootstrap_score_sums(
        data.paired_scores,
        bootstrap_cluster_rows,
    )
    bootstrap_elos = fit_bootstrap_elos(
        data.names,
        data.pairs,
        score_sum_rows,
        cluster_count,
        workers=workers,
        progress=progress,
    )

    jackknife_elos = np.empty((cluster_count, len(data.names)), dtype=float)
    total_scores = np.sum(data.paired_scores, axis=0)
    jackknife_counts = np.full(pair_count, cluster_count - 1, dtype=int)
    for cluster_row in range(cluster_count):
        jackknife_elos[cluster_row] = fit_elos_from_pair_scores(
            data.names,
            data.pairs,
            total_scores - data.paired_scores[cluster_row],
            jackknife_counts,
        )
    if not np.allclose(
        np.mean(bootstrap_elos, axis=1),
        1500.0,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("bootstrap Elo replicates are not centered at 1500")
    if not np.allclose(
        np.mean(jackknife_elos, axis=1),
        1500.0,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("jackknife Elo estimates are not centered at 1500")
    intervals = tuple(
        bca_interval(
            point_elos[player_idx],
            bootstrap_elos[:, player_idx],
            jackknife_elos[:, player_idx],
            confidence,
        )
        for player_idx in range(len(data.names))
    )
    return ClusteredEloAnalysis(
        point_elos=point_elos,
        intervals=intervals,
        bootstrap_elos=bootstrap_elos,
        bootstrap_cluster_rows=bootstrap_cluster_rows,
        jackknife_elos=jackknife_elos,
    )


def _csv_text(fieldnames: Sequence[str], rows: Sequence[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _bootstrap_index_hash(rows: np.ndarray) -> str:
    canonical = np.asarray(rows, dtype="<u4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def write_clustered_elo_outputs(
    output_dir: Path,
    data: ClusteredEloData,
    analysis: ClusteredEloAnalysis,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> None:
    if not math.isclose(
        confidence,
        DEFAULT_CONFIDENCE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("strength_elo_ci95 outputs require confidence=0.95")
    output_dir = Path(output_dir)
    summary_rows = []
    for player_idx, name in enumerate(data.names):
        point = float(analysis.point_elos[player_idx])
        interval = analysis.intervals[player_idx]
        summary_rows.append(
            {
                "name": name,
                "paired_set_elo": point,
                "paired_set_elo_ci95_low": interval.low,
                "paired_set_elo_ci95_high": interval.high,
                "paired_set_elo_ci95_lower_error": point - interval.low,
                "paired_set_elo_ci95_upper_error": interval.high - point,
                "interval_method": interval.method,
                "bca_bias_correction": interval.bias_correction,
                "bca_acceleration": interval.acceleration,
                "bca_lower_quantile": interval.lower_quantile,
                "bca_upper_quantile": interval.upper_quantile,
                "fallback_reason": interval.fallback_reason,
            }
        )
    summary_path = output_dir / "strength_elo_ci95.csv"
    atomic_write_text(
        summary_path,
        _csv_text(list(summary_rows[0]), summary_rows),
    )

    replicate_rows = []
    for replicate_idx, ratings in enumerate(analysis.bootstrap_elos):
        row = {"replicate": replicate_idx}
        row.update(
            {name: float(rating) for name, rating in zip(data.names, ratings)}
        )
        replicate_rows.append(row)
    replicates_path = output_dir / "strength_elo_bootstrap_replicates.csv"
    atomic_write_text(
        replicates_path,
        _csv_text(list(replicate_rows[0]), replicate_rows),
    )

    sampled_set_indices = np.asarray(data.set_indices)[
        analysis.bootstrap_cluster_rows
    ]
    index_rows = []
    index_columns = [
        f"draw_{draw_idx}" for draw_idx in range(sampled_set_indices.shape[1])
    ]
    for replicate_idx, sampled_indices in enumerate(sampled_set_indices):
        row = {"replicate": replicate_idx}
        row.update(
            {
                column: int(set_index)
                for column, set_index in zip(index_columns, sampled_indices)
            }
        )
        index_rows.append(row)
    indices_path = output_dir / "strength_elo_bootstrap_cluster_indices.csv"
    atomic_write_text(
        indices_path,
        _csv_text(["replicate"] + index_columns, index_rows),
    )

    generated_hashes = {
        path.name: sha256_file(path)
        for path in (summary_path, replicates_path, indices_path)
    }
    metadata = {
        "schema_version": 1,
        "experiment_id": data.experiment_id,
        "method": "xot_set_index_cluster_bca_bootstrap",
        "confidence": float(confidence),
        "interval_scope": (
            "pointwise per participant; not a simultaneous "
            f"{len(data.names)}-player band"
        ),
        "estimand": (
            "regularized relative Elo, centered so the "
            f"{len(data.names)}-player mean is 1500, under repeated sampling "
            "of XOT openings from the source "
            "opening population"
        ),
        "cluster_unit": (
            "set_index/XOT opening; each draw jointly includes all player pairs"
        ),
        "paired_observation": (
            "one score in {0, 0.5, 1} from the sign of the mean disc "
            "difference of two color-swapped games"
        ),
        "regularization": (
            "one fixed 0.5-score pseudo-observation (0.5 win and 0.5 loss) "
            "per player pair in every fit; the pseudo-observation is not "
            "resampled"
        ),
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "rng": "numpy.random.PCG64",
            "cluster_count": len(data.set_indices),
            "clusters_drawn_per_replicate": len(data.set_indices),
            "with_replacement": True,
            "bootstrap_cluster_row_sha256_u32_le": _bootstrap_index_hash(
                analysis.bootstrap_cluster_rows
            ),
        },
        "bca": {
            "jackknife": "leave one XOT/set_index cluster out",
            "jackknife_replicates": len(data.set_indices),
            "quantile_interpolation": "numpy linear",
            "fallback": (
                "ordinary cluster percentile interval only when BCa "
                "acceleration or adjusted quantiles are undefined"
            ),
        },
        "sample": {
            "participants": len(data.names),
            "player_pairs": len(data.pairs),
            "paired_match_sets": len(data.set_indices) * len(data.pairs),
            "actual_games": 2 * len(data.set_indices) * len(data.pairs),
            "set_indices": list(data.set_indices),
            "openings": list(data.openings),
        },
        "source_sha256": {
            "strength_games.jsonl": data.raw_results_sha256,
            "strength_manifest.json": data.manifest_sha256,
            "strength_results.json": data.aggregate_sha256,
            "strength_elo_confidence.py": sha256_file(
                Path(__file__).resolve()
            ),
            "reanalyze_strength_elo.py": sha256_file(
                SCRIPT_DIR / "reanalyze_strength_elo.py"
            ),
            "elo_rating_backcal.py": sha256_file(
                BIN_DIR / "elo_rating_backcal.py"
            ),
            "strength_tournament.py": sha256_file(
                SCRIPT_DIR / "strength_tournament.py"
            ),
        },
        "runtime_versions": {
            "python": sys.version,
            "numpy": np.__version__,
        },
        "generated_sha256": generated_hashes,
        "results": summary_rows,
        "limitations": [
            (
                "Intervals describe XOT-opening sampling variation only; "
                "they do not include model retraining, engine-setting, or "
                "opening-corpus uncertainty."
            ),
            (
                "Differences between two players must be assessed from "
                "within-replicate Elo differences, not by checking whether "
                "two marginal intervals overlap."
            ),
        ],
    }
    atomic_write_json(output_dir / "strength_elo_ci95.json", metadata)


def default_worker_count() -> int:
    return max(1, min(16, (os.cpu_count() or 1) // 2))
