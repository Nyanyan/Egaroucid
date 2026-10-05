#!/usr/bin/env python3
"""WTHOR人間着手一致率実験の標本化と科学計算を実装する。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BLEND_DIR = SCRIPT_DIR.parents[0] / "30_blend_with_egaroucid"
sys.path.insert(0, str(BLEND_DIR))

from blend_policy_with_egaroucid import (  # noqa: E402
    ALL_LEGAL_HINT_COUNT,
    POLICY_SIZE,
    BinaryPolicyNetwork,
    BoardState,
    board_to_features,
    require_complete_egaroucid_scores,
)
import evaluate_wthor_blend_human_match as wthor  # noqa: E402
from evaluate_wthor_human_match import calc_legal_batch, legal_bitboard_to_mask
from policy_accuracy import TRANSFORM_MAPS, transform_bitboards
from wthor_likelihood import legal_log_probabilities, likelihood_summary


BLEND_PARAMS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CONSOLE_LEVELS = (1, 5, 9, 13, 17, 21)
CONSOLE_REFERENCE_LEVEL = 21
TOP_N_VALUES = (1, 3)
CONSOLE_ONLY_HINT_COUNT = max(TOP_N_VALUES)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

StateKey = Tuple[int, int, int]


@dataclass(frozen=True)
class PositionGroup:
    black: int
    white: int
    side: int
    policy_counts: Tuple[Tuple[int, int], ...]

    @property
    def key(self) -> StateKey:
        return self.black, self.white, self.side

    @property
    def sample_count(self) -> int:
        return sum(count for _, count in self.policy_counts)


@dataclass(frozen=True)
class HintData:
    scores: Dict[int, float]
    move_order: Tuple[int, ...]


@dataclass(frozen=True)
class _PreparedPositionGroup:
    legal: Tuple[int, ...]
    valid_targets: Tuple[Tuple[int, Tuple[int, ...], int], ...]


@dataclass(frozen=True)
class PreparedAgreementGroups:
    """Read-only board data shared by both engines and final verification."""

    groups_by_key: Mapping[StateKey, Tuple[_PreparedPositionGroup, ...]]
    invalid_policy_samples: int
    illegal_label_samples: int


def wilson_interval(
    hits: int,
    total: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    rate = hits / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (rate * (1.0 - rate) + z2 / (4.0 * total)) / total
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def split_position_count(total: int, data_split: str) -> int:
    if data_split == "all":
        return total
    n_train, n_val, n_test = wthor.split_counts(
        total,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
    )
    return {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }[data_split]


def load_position_groups(
    dat_files: Sequence[Path],
    available_positions: int,
    positions: int | None,
    data_split: str,
    split_seed: int,
    sample_seed: int,
    split_manifest: Path = None,
) -> Tuple[List[PositionGroup], int, str, str]:
    if split_manifest is not None:
        sys.path.insert(0, str(SCRIPT_DIR.parent / "00_data_processing"))
        from wthor_dataset import load_manifest_positions
        members, _ = load_manifest_positions(split_manifest, dat_files, data_split)
        split_positions = len(members)
        if positions is None:
            positions = split_positions
        if positions > split_positions:
            raise ValueError(f"N={positions} exceeds saved {data_split} size {split_positions}")
        selected = (np.sort(members) if positions == split_positions else
                    np.sort(np.random.default_rng(sample_seed).choice(members, positions, replace=False)))
    else:
        selected, split_positions = wthor.choose_data_split_positions(
            available_positions, data_split, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
            split_seed, positions, sample_seed,
        )
    if positions is None:
        positions = split_positions
    if selected is None:
        selected = np.arange(split_positions, dtype=np.int64)
    if selected is None or len(selected) != positions:
        raise RuntimeError(
            f"expected {positions} selected positions, got "
            f"{0 if selected is None else len(selected)}"
        )

    policy_counts: Dict[StateKey, Counter] = {}
    sample_records_digest = hashlib.sha256()
    reader = wthor.PositionSampleReader(dat_files, available_positions)
    try:
        for global_position in selected:
            sample = reader.get(int(global_position))
            state, side, policy, _, _, _ = wthor.position_sample_to_state(
                sample
            )
            key = (int(state.black), int(state.white), int(side))
            policy_counts.setdefault(key, Counter())[int(policy)] += 1
            sample_records_digest.update(
                key[0].to_bytes(8, "little", signed=False)
            )
            sample_records_digest.update(
                key[1].to_bytes(8, "little", signed=False)
            )
            sample_records_digest.update(
                key[2].to_bytes(1, "little", signed=False)
            )
            sample_records_digest.update(
                int(policy).to_bytes(2, "little", signed=True)
            )
    finally:
        reader.close()

    groups = [
        PositionGroup(
            black=key[0],
            white=key[1],
            side=key[2],
            policy_counts=tuple(sorted(counts.items())),
        )
        for key, counts in policy_counts.items()
    ]
    selected_le = np.asarray(selected, dtype="<i8")
    sample_hash = hashlib.sha256(selected_le.tobytes()).hexdigest()
    return (
        groups,
        split_positions,
        sample_hash,
        sample_records_digest.hexdigest(),
    )


def predict_policy_logits(
    groups: Sequence[PositionGroup],
    weights: Path,
    batch_size: int,
    model: Path = None,
) -> Dict[StateKey, np.ndarray]:
    if model is not None:
        from policy_batch_server import PolicyBackend
        network = PolicyBackend("tensorflow", weights, model, output="logits")
        predict = network.predict
    else:
        network = BinaryPolicyNetwork(weights)
        predict = network.predict_logits
    logits: Dict[StateKey, np.ndarray] = {}
    for start in range(0, len(groups), batch_size):
        batch = groups[start : start + batch_size]
        features = np.concatenate(
            [
                board_to_features(
                    BoardState(group.black, group.white, group.side),
                    group.side,
                )
                for group in batch
            ],
            axis=0,
        )
        batch_logits = predict(features)
        for group, policy_logits in zip(batch, batch_logits):
            logits[group.key] = policy_logits.astype(
                np.float64,
                copy=True,
            )
    return logits


def egaroucid_log_scores(
    hint: HintData,
    legal: Sequence[int],
    score_temperature: float,
    *,
    require_all_legal: bool,
) -> np.ndarray:
    if require_all_legal:
        require_complete_egaroucid_scores(hint.scores, legal)
    scores = np.full(POLICY_SIZE, -np.inf, dtype=np.float64)
    legal_set = set(legal)
    for policy, score in hint.scores.items():
        if policy in legal_set:
            scores[policy] = score / score_temperature
    return scores


def empty_metric(with_likelihood=False, top_n_values=TOP_N_VALUES) -> dict:
    metric = {
        "positions": 0,
        "hits": {top_n: 0 for top_n in top_n_values},
    }
    if with_likelihood:
        metric.update(likelihood_sum=0.0, log_likelihood_sum=0.0)
    return metric


def add_rank(metric: dict, rank: int, count: int) -> None:
    metric["positions"] += count
    for top_n in metric["hits"]:
        if rank <= top_n:
            metric["hits"][top_n] += count


def _copy_metric(metric: dict) -> dict:
    return {
        **metric,
        "positions": int(metric["positions"]),
        "hits": {
            top_n: int(metric["hits"][top_n])
            for top_n in metric["hits"]
        },
    }


def _merge_metric(destination: dict, source: dict) -> None:
    destination["positions"] += source["positions"]
    for top_n in destination["hits"]:
        destination["hits"][top_n] += source["hits"][top_n]
    for field in ("likelihood_sum", "log_likelihood_sum"):
        if field in source:
            destination[field] += source[field]


def prepare_position_groups(
    groups: Sequence[PositionGroup],
    *,
    batch_size: int = 4096,
    progress_interval_sec: float = 10.0,
    on_progress: Callable[[int, int, float], None] | None = None,
) -> PreparedAgreementGroups:
    """Batch legal moves and board symmetries with bounded NumPy temporaries.

    Symmetries are checked once per board, including boards with multiple human
    labels. Callbacks receive (completed groups, total groups, elapsed seconds)
    at startup, at the requested interval, and on completion.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(progress_interval_sec) or progress_interval_sec < 0:
        raise ValueError("progress_interval_sec must be finite and nonnegative")
    groups_by_key: Dict[StateKey, Tuple[_PreparedPositionGroup, ...]] = {}
    invalid_policy_samples = 0
    illegal_label_samples = 0
    # Python integers avoid scalar NumPy indexing for each individual label.
    mappings = tuple(tuple(map(int, mapping)) for mapping in TRANSFORM_MAPS[1:])
    total = len(groups)
    started = last_report = time.perf_counter()
    if on_progress is not None:
        on_progress(0, total, 0.0)
    for start in range(0, total, batch_size):
        batch = groups[start : start + batch_size]
        player = np.fromiter(
            (g.black if g.side == wthor.BLACK else g.white for g in batch),
            dtype=np.uint64, count=len(batch),
        )
        opponent = np.fromiter(
            (g.white if g.side == wthor.BLACK else g.black for g in batch),
            dtype=np.uint64, count=len(batch),
        )
        legal_masks = legal_bitboard_to_mask(calc_legal_batch(player, opponent))
        invariant = np.empty((len(batch), len(mappings)), dtype=bool)
        for index, mapping in enumerate(TRANSFORM_MAPS[1:]):
            invariant[:, index] = (
                (transform_bitboards(player, mapping) == player)
                & (transform_bitboards(opponent, mapping) == opponent)
            )
        for group, legal_mask, symmetries in zip(batch, legal_masks, invariant):
            legal = tuple(np.flatnonzero(legal_mask).tolist())
            board_mappings = [mappings[i] for i in np.flatnonzero(symmetries)]
            valid_targets = []
            for policy, count in group.policy_counts:
                if policy < 0 or policy >= POLICY_SIZE:
                    invalid_policy_samples += count
                    continue
                if not legal_mask[policy]:
                    illegal_label_samples += count
                    continue
                targets = (policy,)
                if board_mappings:
                    targets = tuple(sorted({policy, *(m[policy] for m in board_mappings)}))
                valid_targets.append((policy, targets, count))
            prepared = _PreparedPositionGroup(legal, tuple(valid_targets))
            key = group.key
            groups_by_key[key] = groups_by_key.get(key, ()) + (prepared,)
        completed = start + len(batch)
        now = time.perf_counter()
        if on_progress is not None and (
            completed == total or now - last_report >= progress_interval_sec
        ):
            on_progress(completed, total, now - started)
            last_report = now
    return PreparedAgreementGroups(
        MappingProxyType(groups_by_key), invalid_policy_samples, illegal_label_samples,
    )


class IncrementalAgreementMetrics:
    """Accumulate agreement metrics as Egaroucid hints become available.

    A hint is counted at most once for each ``(level, state_key)`` pair.
    ``snapshot`` and ``result`` return independent copies that are safe for
    progress formatters to modify.
    """

    def __init__(
        self,
        groups: Sequence[PositionGroup],
        policy_logits: Mapping[StateKey, np.ndarray],
        score_temperature: float,
        top_n_values: Sequence[int] = TOP_N_VALUES,
        engine_likelihood: bool = False,
        *,
        prepared_groups: PreparedAgreementGroups | None = None,
        blend_params: Sequence[float] = BLEND_PARAMS,
    ) -> None:
        if not math.isfinite(score_temperature) or score_temperature <= 0.0:
            raise ValueError("score_temperature must be finite and positive")
        self._top_n_values = tuple(top_n_values)
        self._blend_params = tuple(blend_params)
        self._engine_likelihood = engine_likelihood
        self._policy_logits = policy_logits
        self._score_temperature = score_temperature
        if prepared_groups is None:
            prepared_groups = prepare_position_groups(groups)
        self._groups_by_key = prepared_groups.groups_by_key
        self.invalid_policy_samples = prepared_groups.invalid_policy_samples
        self.illegal_label_samples = prepared_groups.illegal_label_samples

        self._blend_metrics = {
            alpha: empty_metric(with_likelihood=True, top_n_values=self._top_n_values)
            for alpha in self._blend_params
        }
        self._console_metrics = {
            level: empty_metric(with_likelihood=engine_likelihood, top_n_values=self._top_n_values)
            for level in CONSOLE_LEVELS
            if level != CONSOLE_REFERENCE_LEVEL
        }
        self._seen_hints = set()

    @property
    def accepted_hint_count(self) -> int:
        return len(self._seen_hints)

    def add_hint(
        self,
        level: int,
        state_key: StateKey,
        hint: HintData,
    ) -> bool:
        """Add one hint, returning ``False`` when it was already counted."""
        if level not in CONSOLE_LEVELS:
            raise ValueError(f"unsupported console level: {level}")
        if state_key not in self._groups_by_key:
            raise KeyError(f"unknown state key: {state_key}")
        identity = (level, state_key)
        if identity in self._seen_hints:
            return False

        prepared_groups = self._groups_by_key[state_key]
        if level == CONSOLE_REFERENCE_LEVEL:
            additions = {
                alpha: empty_metric(with_likelihood=True, top_n_values=self._top_n_values)
                for alpha in self._blend_params
            }
            network_scores = self._policy_logits[state_key]
            for prepared in prepared_groups:
                legal = prepared.legal
                level_scores = egaroucid_log_scores(
                    hint,
                    legal,
                    self._score_temperature,
                    require_all_legal=True,
                )
                legal_indices = np.asarray(legal, dtype=np.int64)
                blend_scores = {}
                blend_log_probabilities = {}
                for alpha in self._blend_params:
                    scores = np.full(
                        POLICY_SIZE,
                        -np.inf,
                        dtype=np.float64,
                    )
                    if alpha == 0.0:
                        scores[legal_indices] = level_scores[legal_indices]
                    elif alpha == 1.0:
                        scores[legal_indices] = network_scores[legal_indices]
                    else:
                        scores[legal_indices] = (
                            alpha * network_scores[legal_indices]
                            + (1.0 - alpha) * level_scores[legal_indices]
                        )
                    blend_scores[alpha] = scores
                    blend_log_probabilities[alpha] = legal_log_probabilities(scores, legal)
                for policy, targets, count in prepared.valid_targets:
                    for alpha, scores in blend_scores.items():
                        rank = wthor.rank_for_distribution(
                            scores,
                            legal,
                            targets,
                            hint.move_order if alpha == 0.0 else None,
                        )
                        add_rank(additions[alpha], rank, count)
                        log_probability = float(blend_log_probabilities[alpha][policy])
                        additions[alpha]["likelihood_sum"] += count * math.exp(log_probability)
                        additions[alpha]["log_likelihood_sum"] += count * log_probability
            for alpha in self._blend_params:
                _merge_metric(
                    self._blend_metrics[alpha],
                    additions[alpha],
                )
        else:
            addition = empty_metric(with_likelihood=self._engine_likelihood, top_n_values=self._top_n_values)
            for prepared in prepared_groups:
                legal = prepared.legal
                scores = egaroucid_log_scores(
                    hint,
                    legal,
                    self._score_temperature,
                    require_all_legal=self._engine_likelihood,
                )
                log_probs = legal_log_probabilities(scores, legal) if self._engine_likelihood else None
                for policy, targets, count in prepared.valid_targets:
                    rank = wthor.rank_for_distribution(
                        scores,
                        legal,
                        targets,
                        hint.move_order,
                    )
                    add_rank(addition, rank, count)
                    if log_probs is not None:
                        addition["likelihood_sum"] += count * math.exp(float(log_probs[policy]))
                        addition["log_likelihood_sum"] += count * float(log_probs[policy])
            _merge_metric(self._console_metrics[level], addition)

        self._seen_hints.add(identity)
        return True

    def add_hints(
        self,
        level: int,
        hints: Mapping[StateKey, HintData],
    ) -> int:
        """Add all hints for one level and return the accepted row count."""
        return sum(
            self.add_hint(level, state_key, hint)
            for state_key, hint in hints.items()
        )

    def add_hint_rows(
        self,
        rows: Iterable[Tuple[int, StateKey, HintData]],
    ) -> int:
        """Add ``(level, state_key, hint)`` rows from a streaming worker."""
        return sum(
            self.add_hint(level, state_key, hint)
            for level, state_key, hint in rows
        )

    def snapshot(
        self,
    ) -> Tuple[dict, Dict[int, dict], int, int]:
        """Return a deep copy, including zero metrics for pending levels."""
        blend_metrics = {
            alpha: _copy_metric(self._blend_metrics[alpha])
            for alpha in self._blend_params
        }
        console_metrics = {
            level: _copy_metric(self._console_metrics[level])
            for level in CONSOLE_LEVELS
            if level != CONSOLE_REFERENCE_LEVEL
        }
        if 0.0 in blend_metrics:
            console_metrics[CONSOLE_REFERENCE_LEVEL] = _copy_metric(blend_metrics[0.0])
        return (
            blend_metrics,
            console_metrics,
            self.invalid_policy_samples,
            self.illegal_label_samples,
        )

    def result(
        self,
    ) -> Tuple[dict, Dict[int, dict], int, int]:
        """Return the current metrics as an independent snapshot."""
        return self.snapshot()


def evaluate_agreement(
    groups: Sequence[PositionGroup],
    hints: Dict[int, Dict[StateKey, HintData]],
    policy_logits: Dict[StateKey, np.ndarray],
    score_temperature: float,
    top_n_values: Sequence[int] = TOP_N_VALUES,
    engine_likelihood: bool = False,
    *,
    prepared_groups: PreparedAgreementGroups | None = None,
) -> Tuple[dict, Dict[int, dict], int, int]:
    metrics = IncrementalAgreementMetrics(
        groups,
        policy_logits,
        score_temperature,
        top_n_values,
        engine_likelihood,
        prepared_groups=prepared_groups,
    )
    for group in groups:
        metrics.add_hint(
            CONSOLE_REFERENCE_LEVEL,
            group.key,
            hints[CONSOLE_REFERENCE_LEVEL][group.key],
        )
        for level in CONSOLE_LEVELS:
            if level == CONSOLE_REFERENCE_LEVEL:
                continue
            metrics.add_hint(level, group.key, hints[level][group.key])
    return metrics.result()


class IncrementalEngineAgreementMetrics:
    """Accumulate native engine rankings once per level and board state."""

    def __init__(self, groups, levels, top_n_values=TOP_N_VALUES, engine_likelihood=False, score_temperature=1.0, *, prepared_groups=None):
        if not math.isfinite(score_temperature) or score_temperature <= 0:
            raise ValueError("score_temperature must be finite and positive")
        self._top_n_values = tuple(top_n_values)
        self._engine_likelihood = engine_likelihood
        self._score_temperature = score_temperature
        if prepared_groups is None:
            prepared_groups = prepare_position_groups(groups)
        self._groups = prepared_groups.groups_by_key
        self._metrics = {level: empty_metric(with_likelihood=engine_likelihood, top_n_values=self._top_n_values) for level in levels}
        self._seen = set()

    def add_hint(self, level, state_key, hint):
        if level not in self._metrics:
            raise ValueError(f"unsupported engine level: {level}")
        if state_key not in self._groups:
            raise KeyError(f"unknown state key: {state_key}")
        identity = (level, state_key)
        if identity in self._seen:
            return False
        addition = empty_metric(with_likelihood=self._engine_likelihood, top_n_values=self._top_n_values)
        for prepared in self._groups[state_key]:
            log_probs = None
            if self._engine_likelihood:
                scores = egaroucid_log_scores(hint, prepared.legal, self._score_temperature, require_all_legal=True)
                log_probs = legal_log_probabilities(scores, prepared.legal)
            for policy, targets, count in prepared.valid_targets:
                rank = min(
                    (index for index, move in enumerate(hint.move_order, 1) if move in targets),
                    default=POLICY_SIZE + 1,
                )
                add_rank(addition, rank, count)
                if log_probs is not None:
                    addition["likelihood_sum"] += count * math.exp(float(log_probs[policy]))
                    addition["log_likelihood_sum"] += count * float(log_probs[policy])
        _merge_metric(self._metrics[level], addition)
        self._seen.add(identity)
        return True

    def snapshot(self):
        return {level: _copy_metric(metric) for level, metric in self._metrics.items()}


def evaluate_engine_agreement(groups, hints, top_n_values=TOP_N_VALUES, engine_likelihood=False, score_temperature=1.0, *, prepared_groups=None):
    """Evaluate standalone top-N hints without treating level 21 as a blend."""
    metrics = IncrementalEngineAgreementMetrics(
        groups, hints, top_n_values, engine_likelihood, score_temperature,
        prepared_groups=prepared_groups,
    )
    for level, level_hints in hints.items():
        for group in groups:
            metrics.add_hint(level, group.key, level_hints[group.key])
    return metrics.snapshot()


def metrics_to_result(metrics: Dict[float, dict]) -> dict:
    topn = []
    for parameter, metric in metrics.items():
        for top_n in metric["hits"]:
            positions = int(metric["positions"])
            hits = int(metric["hits"][top_n])
            topn.append(
                {
                    "blend_param": float(parameter),
                    "top_n": top_n,
                    "hits": hits,
                    "positions": positions,
                    "accuracy": (
                        hits / positions if positions else 0.0
                    ),
                }
            )
    likelihood = [
        {"blend_param": float(parameter), **likelihood_summary(
            metric["likelihood_sum"], metric["log_likelihood_sum"], metric["positions"],
        )}
        for parameter, metric in metrics.items() if "likelihood_sum" in metric
    ]
    return {"topn": topn, "likelihood": likelihood}


def agreement_results_equal(first, second):
    """Counts are exact; floating sums may arrive in a different worker order."""
    if first[2:] != second[2:]:
        return False
    for left, right in zip(first[:2], second[:2]):
        if left.keys() != right.keys():
            return False
        for parameter, metric in left.items():
            other = right[parameter]
            if metric.keys() != other.keys():
                return False
            for field, value in metric.items():
                if field in ("likelihood_sum", "log_likelihood_sum"):
                    if not math.isclose(value, other[field], rel_tol=1e-12, abs_tol=1e-9):
                        return False
                elif value != other[field]:
                    return False
    return True


def _accuracy_summary(topn: Mapping[int, dict]) -> dict:
    positions = int(topn[1]["positions"])
    result = {"positions": positions}
    for top_n, metric in sorted(topn.items()):
        hits = int(metric["hits"])
        lower, upper = wilson_interval(hits, positions)
        result.update({
            f"top{top_n}_hits": hits,
            f"top{top_n}_accuracy": hits / positions if positions else 0.0,
            f"top{top_n}_ci95_lower": lower,
            f"top{top_n}_ci95_upper": upper,
        })
    return result


def make_blend_summary_rows(result: dict, blend_params=BLEND_PARAMS) -> List[dict]:
    likelihood = {
        round(float(row["blend_param"]), 10): {
            key: value for key, value in row.items() if key != "blend_param"
        }
        for row in result.get("likelihood", [])
    }
    return [
        {
            "alpha": alpha,
            **likelihood.get(round(alpha, 10), {}),
            **_accuracy_summary({
                int(row["top_n"]): row for row in result["topn"]
                if round(float(row["blend_param"]), 10) == round(alpha, 10)
            }),
        }
        for alpha in blend_params
    ]


def make_console_level_summary_row(
    level: int,
    result: dict,
    elapsed_sec: float,
    engine_seconds: float = 0.0,
) -> dict:
    return {
        "level": level,
        **{key: value for row in result.get("likelihood", [])
           if float(row["blend_param"]) == 0.0
           for key, value in row.items() if key != "blend_param"},
        **_accuracy_summary({
            int(row["top_n"]): row for row in result["topn"]
            if float(row["blend_param"]) == 0.0
        }),
        "elapsed_sec": float(elapsed_sec),
        "engine_seconds": float(engine_seconds),
    }


def make_level21_reuse_validation(
    blend_rows: Sequence[dict],
    console_rows: Sequence[dict],
) -> dict:
    alpha_zero = next(
        row for row in blend_rows if float(row["alpha"]) == 0.0
    )
    console21 = next(
        row
        for row in console_rows
        if int(row["level"]) == CONSOLE_REFERENCE_LEVEL
    )
    fields = (
        "positions",
        "top1_hits",
    ) + (("top3_hits",) if "top3_hits" in alpha_zero else ())
    return {
        "reference_console_level": CONSOLE_REFERENCE_LEVEL,
        "source": "same level-21 hint results as blend alpha=0.0",
        "duplicate_hint_evaluation_avoided": True,
        "aggregate_counts_equal": all(
            int(alpha_zero[field]) == int(console21[field])
            for field in fields
        ),
    }


def validate_aggregate_counts(
    requested_positions: int,
    blend_rows: Sequence[dict],
    console_rows: Sequence[dict],
    invalid_policy_samples: int,
    illegal_label_samples: int,
) -> None:
    valid_positions = (
        requested_positions
        - invalid_policy_samples
        - illegal_label_samples
    )
    if valid_positions < 0:
        raise RuntimeError("invalid sample accounting")
    for row in [*blend_rows, *console_rows]:
        positions = int(row["positions"])
        top1_hits = int(row["top1_hits"])
        top3_hits = int(row.get("top3_hits", top1_hits))
        if positions != valid_positions:
            raise RuntimeError(
                "agreement denominator mismatch: "
                f"expected={valid_positions}, row={row}"
            )
        if not 0 <= top1_hits <= top3_hits <= positions:
            raise RuntimeError(
                f"invalid top-1/top-3 aggregate counts: {row}"
            )
    if invalid_policy_samples or illegal_label_samples:
        raise RuntimeError(
            "WTHOR標本に不正または非合法な着手ラベルがあります: "
            f"invalid={invalid_policy_samples}, "
            f"illegal={illegal_label_samples}"
        )
