#!/usr/bin/env python3
"""WTHOR人間着手一致率実験の標本化と科学計算を実装する。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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


BLEND_PARAMS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CONSOLE_LEVELS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
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
    valid_targets: Tuple[Tuple[Tuple[int, ...], int], ...]


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
    positions: int,
    data_split: str,
    split_seed: int,
    sample_seed: int,
) -> Tuple[List[PositionGroup], int, str, str]:
    selected, split_positions = wthor.choose_data_split_positions(
        available_positions,
        data_split,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        split_seed,
        positions,
        sample_seed,
    )
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
) -> Dict[StateKey, np.ndarray]:
    network = BinaryPolicyNetwork(weights)
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
        batch_logits = network.predict_logits(features)
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


def empty_metric() -> dict:
    return {
        "positions": 0,
        "hits": {top_n: 0 for top_n in TOP_N_VALUES},
    }


def add_rank(metric: dict, rank: int, count: int) -> None:
    metric["positions"] += count
    for top_n in TOP_N_VALUES:
        if rank <= top_n:
            metric["hits"][top_n] += count


def _copy_metric(metric: dict) -> dict:
    return {
        "positions": int(metric["positions"]),
        "hits": {
            top_n: int(metric["hits"][top_n])
            for top_n in TOP_N_VALUES
        },
    }


def _merge_metric(destination: dict, source: dict) -> None:
    destination["positions"] += source["positions"]
    for top_n in TOP_N_VALUES:
        destination["hits"][top_n] += source["hits"][top_n]


def _prepare_position_group(
    group: PositionGroup,
) -> Tuple[_PreparedPositionGroup, int, int]:
    state = BoardState(group.black, group.white, group.side)
    legal = tuple(state.legal_policies(group.side))
    legal_set = set(legal)
    player, opponent = state.player_opponent_bits(group.side)
    valid_targets = []
    invalid_policy_samples = 0
    illegal_label_samples = 0
    for policy, count in group.policy_counts:
        if policy < 0 or policy >= POLICY_SIZE:
            invalid_policy_samples += count
            continue
        if policy not in legal_set:
            illegal_label_samples += count
            continue
        valid_targets.append(
            (
                tuple(
                    wthor.equivalent_targets(
                        player,
                        opponent,
                        policy,
                    )
                ),
                count,
            )
        )
    return (
        _PreparedPositionGroup(
            legal=legal,
            valid_targets=tuple(valid_targets),
        ),
        invalid_policy_samples,
        illegal_label_samples,
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
    ) -> None:
        self._policy_logits = policy_logits
        self._score_temperature = score_temperature
        self._groups_by_key: Dict[
            StateKey,
            List[_PreparedPositionGroup],
        ] = {}
        self.invalid_policy_samples = 0
        self.illegal_label_samples = 0
        for group in groups:
            prepared, invalid, illegal = _prepare_position_group(group)
            self._groups_by_key.setdefault(group.key, []).append(prepared)
            self.invalid_policy_samples += invalid
            self.illegal_label_samples += illegal

        self._blend_metrics = {
            alpha: empty_metric()
            for alpha in BLEND_PARAMS
        }
        self._console_metrics = {
            level: empty_metric()
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
                alpha: empty_metric()
                for alpha in BLEND_PARAMS
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
                for alpha in BLEND_PARAMS:
                    scores = np.full(
                        POLICY_SIZE,
                        -np.inf,
                        dtype=np.float64,
                    )
                    scores[legal_indices] = (
                        alpha * network_scores[legal_indices]
                        + (1.0 - alpha) * level_scores[legal_indices]
                    )
                    blend_scores[alpha] = scores
                for targets, count in prepared.valid_targets:
                    for alpha, scores in blend_scores.items():
                        rank = wthor.rank_for_distribution(
                            scores,
                            legal,
                            targets,
                            hint.move_order if alpha == 0.0 else None,
                        )
                        add_rank(additions[alpha], rank, count)
            for alpha in BLEND_PARAMS:
                _merge_metric(
                    self._blend_metrics[alpha],
                    additions[alpha],
                )
        else:
            addition = empty_metric()
            for prepared in prepared_groups:
                legal = prepared.legal
                scores = egaroucid_log_scores(
                    hint,
                    legal,
                    self._score_temperature,
                    require_all_legal=False,
                )
                for targets, count in prepared.valid_targets:
                    rank = wthor.rank_for_distribution(
                        scores,
                        legal,
                        targets,
                        hint.move_order,
                    )
                    add_rank(addition, rank, count)
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
            for alpha in BLEND_PARAMS
        }
        console_metrics = {
            level: _copy_metric(self._console_metrics[level])
            for level in CONSOLE_LEVELS
            if level != CONSOLE_REFERENCE_LEVEL
        }
        console_metrics[CONSOLE_REFERENCE_LEVEL] = _copy_metric(
            blend_metrics[0.0]
        )
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
) -> Tuple[dict, Dict[int, dict], int, int]:
    metrics = IncrementalAgreementMetrics(
        groups,
        policy_logits,
        score_temperature,
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


def metrics_to_result(metrics: Dict[float, dict]) -> dict:
    topn = []
    for parameter, metric in metrics.items():
        for top_n in TOP_N_VALUES:
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
    return {"topn": topn}


def make_blend_summary_rows(result: dict) -> List[dict]:
    topn = {
        (round(float(row["blend_param"]), 10), int(row["top_n"])): row
        for row in result["topn"]
    }
    rows = []
    for alpha in BLEND_PARAMS:
        top1 = topn[(round(alpha, 10), 1)]
        top3 = topn[(round(alpha, 10), 3)]
        positions = int(top1["positions"])
        top1_hits = int(top1["hits"])
        top3_hits = int(top3["hits"])
        top1_lower, top1_upper = wilson_interval(
            top1_hits,
            positions,
        )
        top3_lower, top3_upper = wilson_interval(
            top3_hits,
            positions,
        )
        rows.append(
            {
                "alpha": alpha,
                "positions": positions,
                "top1_hits": top1_hits,
                "top1_accuracy": (
                    top1_hits / positions if positions else 0.0
                ),
                "top1_ci95_lower": top1_lower,
                "top1_ci95_upper": top1_upper,
                "top3_hits": top3_hits,
                "top3_accuracy": (
                    top3_hits / positions if positions else 0.0
                ),
                "top3_ci95_lower": top3_lower,
                "top3_ci95_upper": top3_upper,
            }
        )
    return rows


def make_console_level_summary_row(
    level: int,
    result: dict,
    elapsed_sec: float,
    engine_seconds: float = 0.0,
) -> dict:
    topn = {
        int(row["top_n"]): row
        for row in result["topn"]
        if float(row["blend_param"]) == 0.0
    }
    top1 = topn[1]
    top3 = topn[3]
    positions = int(top1["positions"])
    top1_hits = int(top1["hits"])
    top3_hits = int(top3["hits"])
    top1_lower, top1_upper = wilson_interval(top1_hits, positions)
    top3_lower, top3_upper = wilson_interval(top3_hits, positions)
    return {
        "level": level,
        "positions": positions,
        "top1_hits": top1_hits,
        "top1_accuracy": (
            top1_hits / positions if positions else 0.0
        ),
        "top1_ci95_lower": top1_lower,
        "top1_ci95_upper": top1_upper,
        "top3_hits": top3_hits,
        "top3_accuracy": (
            top3_hits / positions if positions else 0.0
        ),
        "top3_ci95_lower": top3_lower,
        "top3_ci95_upper": top3_upper,
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
        "top3_hits",
    )
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
        top3_hits = int(row["top3_hits"])
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
