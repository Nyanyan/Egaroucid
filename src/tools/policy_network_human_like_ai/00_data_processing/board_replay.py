"""Replay batches of coordinate transcripts, including forced passes."""

import numpy as np

from policy_data import SHIFT_FUNCS, calc_legal_batch
from raw_formats import normalize_transcript
from wthor_dataset import BOARD_DTYPE


BYTE_COUNTS = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)


def flips_for_moves(player, opponent, moves):
    flips = np.zeros_like(player)
    for shift in SHIFT_FUNCS:
        captured = shift(moves) & opponent
        for _ in range(5):
            captured |= shift(captured) & opponent
        flips |= np.where((shift(captured) & player) != 0, captured, np.uint64(0))
    return flips


def bit_counts(values):
    return BYTE_COUNTS[values.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def replay_games(games):
    """Return board records, original game IDs, and per-game rejection reasons.

    `games` contains (game_id, normalized_transcript) pairs. Partial and illegal
    games contribute no records. Scores award remaining empties to the winner,
    matching Egaroucid's terminal score convention.
    """
    n = len(games)
    if not n:
        return np.empty(0, dtype=BOARD_DTYPE), np.empty(0, dtype="<u8"), []
    policies = np.zeros((n, 60), dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int16)
    valid = np.ones(n, dtype=bool)
    errors = {}
    for i, (game_id, transcript) in enumerate(games):
        try:
            transcript = normalize_transcript(transcript)
        except ValueError as error:
            valid[i] = False
            errors[i] = str(error)
            continue
        lengths[i] = len(transcript) // 2
        policies[i, :lengths[i]] = [63 - ((ord(transcript[j + 1]) - ord('1')) * 8
                                         + ord(transcript[j]) - ord('a'))
                                    for j in range(0, len(transcript), 2)]
    player = np.full(n, (1 << 28) | (1 << 35), dtype=np.uint64)
    opponent = np.full(n, (1 << 27) | (1 << 36), dtype=np.uint64)
    colors = np.zeros(n, dtype=np.int8)
    records = np.zeros((n, 60), dtype=BOARD_DTYPE)
    for ply in range(int(lengths.max())):
        active = valid & (lengths > ply)
        moves = np.left_shift(np.uint64(1), policies[:, ply].astype(np.uint64))
        flips = flips_for_moves(player, opponent, moves)
        may_pass = active & (flips == 0)
        if np.any(may_pass):
            forced_pass = may_pass & (calc_legal_batch(player, opponent) == 0)
            player, opponent = np.where(forced_pass, opponent, player), np.where(forced_pass, player, opponent)
            colors ^= forced_pass.astype(np.int8)
            flips = flips_for_moves(player, opponent, moves)
        illegal = active & (((moves & (player | opponent)) != 0) | (flips == 0))
        for i in np.flatnonzero(illegal):
            errors[int(i)] = f"illegal move at ply {ply + 1}"
        valid &= ~illegal
        active &= valid
        records["player"][:, ply] = player
        records["opponent"][:, ply] = opponent
        records["color"][:, ply] = colors
        records["policy"][:, ply] = policies[:, ply]
        next_player = opponent ^ flips
        next_opponent = player ^ (flips | moves)
        player = np.where(active, next_player, player)
        opponent = np.where(active, next_opponent, opponent)
        colors ^= active.astype(np.int8)
    incomplete = valid & ((calc_legal_batch(player, opponent) != 0) | (calc_legal_batch(opponent, player) != 0))
    for i in np.flatnonzero(incomplete):
        errors[int(i)] = "transcript ended before the game was over"
    valid &= ~incomplete
    player_count, opponent_count = bit_counts(player), bit_counts(opponent)
    score = player_count - opponent_count
    score += np.sign(score) * (64 - player_count - opponent_count)
    records["score"] = np.where(records["color"] == colors[:, None], score[:, None], -score[:, None])
    keep = valid[:, None] & (np.arange(60)[None, :] < lengths[:, None])
    ids = np.array([game[0] for game in games], dtype="<u8")
    rejected = [{"game_id": int(games[i][0]), "reason": error, "transcript": games[i][1]}
                for i, error in sorted(errors.items())]
    return records[keep], np.broadcast_to(ids[:, None], (n, 60))[keep], rejected
