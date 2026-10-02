"""Independent scalar Othello reference for dataset and replay tests."""

from typing import Optional, Tuple

import numpy as np

from wthor_dataset import BOARD_DTYPE


HW = 8
HW2 = 64
HW2_M1 = 63
BLACK = 0


def policy_from_coord(file_char: str, rank_char: str) -> int:
    file_char = file_char.lower()
    if not ("a" <= file_char <= "h") or not ("1" <= rank_char <= "8"):
        return -1
    x = ord(file_char) - ord("a")
    y = ord(rank_char) - ord("1")
    return HW2_M1 - (y * HW + x)


def bit_from_policy(policy: int) -> int:
    return 1 << policy


def initial_board() -> Tuple[int, int, int]:
    black = bit_from_policy(policy_from_coord("e", "4")) | bit_from_policy(policy_from_coord("d", "5"))
    white = bit_from_policy(policy_from_coord("d", "4")) | bit_from_policy(policy_from_coord("e", "5"))
    return black, white, BLACK


def calc_flips(player: int, opponent: int, policy: int) -> int:
    if policy < 0 or policy >= HW2:
        return 0
    move_bit = bit_from_policy(policy)
    if (player | opponent) & move_bit:
        return 0
    pos = HW2_M1 - policy
    x0 = pos % HW
    y0 = pos // HW
    flips = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        x = x0 + dx
        y = y0 + dy
        line = 0
        while 0 <= x < HW and 0 <= y < HW:
            p = HW2_M1 - (y * HW + x)
            bit = bit_from_policy(p)
            if opponent & bit:
                line |= bit
            else:
                if (player & bit) and line:
                    flips |= line
                break
            x += dx
            y += dy
    return flips


def legal_moves(player: int, opponent: int) -> int:
    occupied = player | opponent
    legal = 0
    for policy in range(HW2):
        bit = bit_from_policy(policy)
        if (occupied & bit) == 0 and calc_flips(player, opponent, policy):
            legal |= bit
    return legal


def apply_move(black: int, white: int, side: int, policy: int) -> Optional[Tuple[int, int, int]]:
    player = black if side == BLACK else white
    opponent = white if side == BLACK else black
    flips = calc_flips(player, opponent, policy)
    if not flips:
        return None
    player ^= flips | bit_from_policy(policy)
    opponent ^= flips
    if side == BLACK:
        black, white = player, opponent
    else:
        white, black = player, opponent
    return black, white, side ^ 1


def pass_if_needed(black: int, white: int, side: int) -> Tuple[int, bool]:
    player = black if side == BLACK else white
    opponent = white if side == BLACK else black
    if legal_moves(player, opponent):
        return side, True
    side ^= 1
    player = black if side == BLACK else white
    opponent = white if side == BLACK else black
    return side, bool(legal_moves(player, opponent))


def make_game(seed, max_moves=60):
    rng = np.random.default_rng(seed)
    black, white, side = initial_board()
    rows, moves = [], []
    for _ in range(max_moves):
        side, can_move = pass_if_needed(black, white, side)
        if not can_move:
            break
        player, opponent = (black, white) if side == 0 else (white, black)
        legal = legal_moves(player, opponent)
        policy = int(rng.choice([p for p in range(64) if legal & (1 << p)]))
        rows.append((player, opponent, side, policy, 0))
        moves.append(chr(ord("a") + (63 - policy) % 8) + str((63 - policy) // 8 + 1))
        black, white, side = apply_move(black, white, side, policy)
    return np.array(rows, dtype=BOARD_DTYPE), "".join(moves)
