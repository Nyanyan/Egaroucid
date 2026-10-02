"""All eight square symmetries, applied identically to bitboards and labels."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_accuracy import TRANSFORM_MAPS


TRANSFORM_NAMES = ("identity", "mirror_vertical", "mirror_horizontal", "rotate_180",
                   "transpose", "anti_transpose", "rotate_90", "rotate_270")


def byte_tables(mapping):
    tables = np.zeros((8, 256), dtype=np.uint64)
    values = np.arange(256, dtype=np.uint64)
    for byte in range(8):
        for bit in range(8):
            tables[byte] |= np.where(values & np.uint64(1 << bit),
                                    np.uint64(1) << np.uint64(mapping[8 * byte + bit]), np.uint64(0))
    return tables


TABLES = tuple(byte_tables(mapping) for mapping in TRANSFORM_MAPS)


def transform_bits(bits, tables):
    result = np.zeros_like(bits, dtype=np.uint64)
    for byte in range(8):
        result |= tables[byte, ((bits >> np.uint64(8 * byte)) & np.uint64(255)).astype(np.uint8)]
    return result


def augment_positions(samples, game_ids):
    if len(samples) != len(game_ids):
        raise ValueError("board/game-ID count mismatch")
    if np.any((samples["policy"] < 0) | (samples["policy"] >= 64)):
        raise ValueError("invalid move label in board data")
    result = np.empty((len(samples), 8), dtype=samples.dtype)
    for transform, (mapping, tables) in enumerate(zip(TRANSFORM_MAPS, TABLES)):
        result["player"][:, transform] = transform_bits(samples["player"], tables)
        result["opponent"][:, transform] = transform_bits(samples["opponent"], tables)
        result["policy"][:, transform] = mapping[samples["policy"]]
        result["color"][:, transform] = samples["color"]
        result["score"][:, transform] = samples["score"]
    return result.reshape(-1), np.repeat(game_ids, 8)
