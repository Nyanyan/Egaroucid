#!/usr/bin/env python3
"""Shared ranking and board-symmetry rules for policy accuracy."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


BOARD_WIDTH = 8
POLICY_SIZE = BOARD_WIDTH * BOARD_WIDTH
POLICY_INDEX_MAX = POLICY_SIZE - 1


def policy_to_xy(policy: int) -> Tuple[int, int]:
    pos = POLICY_INDEX_MAX - policy
    return pos % BOARD_WIDTH, pos // BOARD_WIDTH


def xy_to_policy(x: int, y: int) -> int:
    return POLICY_INDEX_MAX - (y * BOARD_WIDTH + x)


def make_transform_maps() -> List[np.ndarray]:
    # The eight transformations of a square board: identity, rotations by
    # 90/180/270 degrees, and reflections across four axes.
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (BOARD_WIDTH - 1 - x, y),
        lambda x, y: (x, BOARD_WIDTH - 1 - y),
        lambda x, y: (BOARD_WIDTH - 1 - x, BOARD_WIDTH - 1 - y),
        lambda x, y: (y, x),
        lambda x, y: (BOARD_WIDTH - 1 - y, BOARD_WIDTH - 1 - x),
        lambda x, y: (BOARD_WIDTH - 1 - y, x),
        lambda x, y: (y, BOARD_WIDTH - 1 - x),
    )
    maps = []
    for transform in transforms:
        mapping = np.empty(POLICY_SIZE, dtype=np.int64)
        for policy in range(POLICY_SIZE):
            x, y = policy_to_xy(policy)
            tx, ty = transform(x, y)
            mapping[policy] = xy_to_policy(tx, ty)
        maps.append(mapping)
    return maps


TRANSFORM_MAPS = make_transform_maps()


def transform_bitboards(bits: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    result = np.zeros_like(bits, dtype=np.uint64)
    for old_policy, new_policy in enumerate(mapping):
        old_bit = np.uint64(1) << np.uint64(old_policy)
        new_bit = np.uint64(1) << np.uint64(new_policy)
        result |= np.where(
            (bits & old_bit) != 0,
            new_bit,
            np.uint64(0),
        )
    return result


def equivalent_policy_mask(
    player: np.ndarray,
    opponent: np.ndarray,
    policies: np.ndarray,
) -> np.ndarray:
    """Return moves equivalent to each label under board-invariant symmetries."""
    equiv = np.zeros((len(policies), POLICY_SIZE), dtype=bool)
    rows = np.arange(len(policies))
    equiv[rows, policies] = True
    for mapping in TRANSFORM_MAPS[1:]:
        invariant = (
            (transform_bitboards(player, mapping) == player)
            & (transform_bitboards(opponent, mapping) == opponent)
        )
        if np.any(invariant):
            equiv[rows[invariant], mapping[policies[invariant]]] = True
    return equiv


def symmetry_aware_policy_ranks(
    probabilities: np.ndarray,
    legal_mask: np.ndarray,
    equivalent_mask: np.ndarray,
) -> np.ndarray:
    """Return the best rank among symmetry-equivalent legal target moves."""
    masked_probabilities = np.where(legal_mask, probabilities, -np.inf)
    order = np.argsort(-masked_probabilities, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(probabilities))
    ranks[rows.reshape(-1, 1), order] = np.arange(POLICY_SIZE)
    equivalent_legal = equivalent_mask & legal_mask
    return (
        np.min(
            np.where(equivalent_legal, ranks, POLICY_SIZE),
            axis=1,
        )
        + 1
    )
