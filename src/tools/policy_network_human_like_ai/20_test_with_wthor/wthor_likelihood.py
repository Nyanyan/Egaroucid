"""Probability of the exact observed human move, normalized over legal moves."""

import math
import numpy as np


LIKELIHOOD_FIELDS = (
    "mean_likelihood", "mean_likelihood_percent",
    "mean_log_likelihood", "negative_log_likelihood",
)
LIKELIHOOD_DEFINITION = {
    "target": "exact_observed_human_move",
    "normalization": "all_legal_moves",
    "aggregation": "arithmetic_mean_over_position_samples_including_duplicates",
    "symmetry_equivalent_moves_summed": False,
    "mean_likelihood_percent": "100 * mean_likelihood",
    "mean_log_likelihood": "mean(log(p(human_move | position))), natural logarithm",
    "negative_log_likelihood": "-mean_log_likelihood (nats per position)",
    "null_log_likelihood": "no valid positions or zero observed probability",
}


def legal_log_probabilities(scores, legal):
    """Stable log-softmax of unnormalized log scores over all legal moves."""
    result = np.full(64, -np.inf, dtype=np.float64)
    indices = np.asarray(legal, dtype=np.int64)
    if not len(indices):
        return result
    values = np.asarray(scores, dtype=np.float64)[indices]
    if not np.all(np.isfinite(values)):
        raise ValueError("likelihood requires finite scores for every legal move")
    shifted = values - np.max(values)
    result[indices] = shifted - np.log(np.exp(shifted).sum())
    return result


def normalize_legal_probabilities(probabilities, legal_mask):
    """Condition model softmax output on the legal move set."""
    values = np.where(legal_mask, np.asarray(probabilities, dtype=np.float64), 0.0)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("invalid policy probabilities")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("policy assigns zero probability to all legal moves")
    return values / totals


def likelihood_summary(probability_sum, log_probability_sum, positions):
    mean = probability_sum / positions if positions else None
    log_mean = log_probability_sum / positions if positions else None
    if log_mean is not None and not math.isfinite(log_mean):
        log_mean = None
    return {
        "mean_likelihood": mean,
        "mean_likelihood_percent": 100.0 * mean if mean is not None else None,
        "mean_log_likelihood": log_mean,
        "negative_log_likelihood": -log_mean if log_mean is not None else None,
    }
