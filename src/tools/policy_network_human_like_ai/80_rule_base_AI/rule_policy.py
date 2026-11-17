"""Small, interpretable, D4-invariant Othello rules; no search or neural net."""
from pathlib import Path
import json
import numpy as np
from numba import njit

SQUARE_NAMES = ('corner', 'C', 'edge_A', 'edge_B', 'X', 'inner_12',
                'inner_13', 'inner_22', 'inner_23', 'center')
FEATURE_NAMES = (tuple('square_' + s for s in SQUARE_NAMES)
    + tuple(f'{s}_corner_{owner}' for s in ('C', 'X') for owner in ('empty', 'own', 'opponent'))
    + tuple(f'flips_{i}' for i in range(1, 8)) + ('flips_8_plus',)
    + ('flips_scaled', 'flips_squared', 'most_flips', 'fewest_flips',
       'flip_directions', 'longest_flip_ray', 'opponent_mobility', 'own_mobility',
       'opponent_corner_moves', 'force_pass', 'move_adjacent_empty',
       'move_adjacent_own', 'move_adjacent_opponent', 'own_frontier_change',
       'opponent_frontier_change', 'edge_discs_gained', 'anchored_edge_gain',
       'own_potential_mobility_change', 'opponent_potential_mobility_change'))
PHASE_NAMES = ('early', 'middle', 'late')
FULL = np.uint64(0xffffffffffffffff)
CORNERS = np.uint64(0x8100000000000081)
EDGES = np.uint64(0xff818181818181ff)


@njit(cache=True)
def shift(b, direction):
    if direction == 0:
        return (b & np.uint64(0x7f7f7f7f7f7f7f7f)) << np.uint64(1)
    if direction == 1:
        return (b & np.uint64(0xfefefefefefefefe)) >> np.uint64(1)
    if direction == 2:
        return b << np.uint64(8)
    if direction == 3:
        return b >> np.uint64(8)
    if direction == 4:
        return (b & np.uint64(0x7f7f7f7f7f7f7f7f)) << np.uint64(9)
    if direction == 5:
        return (b & np.uint64(0xfefefefefefefefe)) << np.uint64(7)
    if direction == 6:
        return (b & np.uint64(0x7f7f7f7f7f7f7f7f)) >> np.uint64(7)
    return (b & np.uint64(0xfefefefefefefefe)) >> np.uint64(9)


@njit(cache=True)
def popcount(b):
    n = 0
    while b:
        b &= b - np.uint64(1)
        n += 1
    return n


@njit(cache=True)
def adjacent(b):
    result = np.uint64(0)
    for d in range(8):
        result |= shift(b, d)
    return result


@njit(cache=True)
def legal_moves(player, opponent):
    result = np.uint64(0)
    empty = ~(player | opponent)
    for d in range(8):
        ray = shift(player, d) & opponent
        for _ in range(5):
            ray |= shift(ray, d) & opponent
        result |= shift(ray, d) & empty
    return result


@njit(cache=True)
def flips_for_move(player, opponent, policy):
    move = np.uint64(1) << np.uint64(policy)
    flips = np.uint64(0)
    directions, longest = 0, 0
    for d in range(8):
        ray = shift(move, d) & opponent
        for _ in range(5):
            ray |= shift(ray, d) & opponent
        if shift(ray, d) & player:
            flips |= ray
            directions += 1
            longest = max(longest, popcount(ray))
    return flips, directions, longest


@njit(cache=True)
def anchored_edges(player):
    """Only contiguous edge discs connected to an owned corner are counted."""
    result = np.uint64(0)
    for corner, d1, d2 in ((0, 0, 2), (7, 1, 2), (56, 0, 3), (63, 1, 3)):
        bit = np.uint64(1) << np.uint64(corner)
        if player & bit:
            result |= bit
            for d in (d1, d2):
                ray = shift(bit, d) & player
                for _ in range(6):
                    ray |= shift(ray, d) & player
                result |= ray
    return popcount(result)


@njit(cache=True)
def phase_weights(player, opponent):
    t = (popcount(player | opponent) - 4) / 60.0
    return max(1.0 - 2*t, 0.0), 1.0 - abs(2*t - 1.0), max(2*t - 1.0, 0.0)


@njit(cache=True)
def extract(player, opponent):
    """Return ascending bit-index moves and one normalized rule vector per move."""
    legal = legal_moves(player, opponent)
    moves = np.empty(popcount(legal), dtype=np.int8)
    features = np.zeros((len(moves), len(FEATURE_NAMES)), dtype=np.float32)
    empty = ~(player | opponent)
    near_empty = adjacent(empty)
    frontier_p = popcount(player & near_empty)
    frontier_o = popcount(opponent & near_empty)
    potential_p = popcount(adjacent(opponent) & empty)
    potential_o = popcount(adjacent(player) & empty)
    anchored = anchored_edges(player)
    flip_counts = np.empty(len(moves), dtype=np.int16)
    j = 0
    for policy in range(64):
        move = np.uint64(1) << np.uint64(policy)
        if not (legal & move):
            continue
        moves[j] = policy
        f = features[j]
        flips, directions, longest = flips_for_move(player, opponent, policy)
        count = popcount(flips)
        flip_counts[j] = count
        x, y = policy % 8, policy // 8
        a, b = min(x, 7-x), min(y, 7-y)
        a, b = min(a, b), max(a, b)
        cls = 0
        if a == 0:
            cls = b
        elif a == 1:
            cls = 3 + b
        elif a == 2:
            cls = 5 + b
        else:
            cls = 9
        f[cls] = 1
        if cls == 1 or cls == 4:
            corner = (0 if x < 4 else 7) + (0 if y < 4 else 56)
            cb = np.uint64(1) << np.uint64(corner)
            owner = 1 if player & cb else (2 if opponent & cb else 0)
            f[10 + (0 if cls == 1 else 3) + owner] = 1
        f[16 + min(count, 8) - 1] = 1
        f[24], f[25] = count / 18.0, count * count / 324.0
        f[28], f[29] = directions / 8.0, longest / 6.0
        new_p, new_o = player | flips | move, opponent & ~flips
        new_empty = empty & ~move
        opp_legal = legal_moves(new_o, new_p)
        f[30] = popcount(opp_legal) / 32.0
        f[31] = popcount(legal_moves(new_p, new_o)) / 32.0
        f[32] = popcount(opp_legal & CORNERS) / 4.0
        f[33] = 1.0 if opp_legal == 0 else 0.0
        near_move = adjacent(move)
        f[34] = popcount(near_move & empty) / 8.0
        f[35] = popcount(near_move & player) / 8.0
        f[36] = popcount(near_move & opponent) / 8.0
        frontier = adjacent(new_empty)
        f[37] = (popcount(new_p & frontier) - frontier_p) / 18.0
        f[38] = (popcount(new_o & frontier) - frontier_o) / 18.0
        f[39] = popcount((flips | move) & EDGES) / 10.0
        f[40] = (anchored_edges(new_p) - anchored) / 10.0
        f[41] = (popcount(adjacent(new_o) & new_empty) - potential_p) / 32.0
        f[42] = (popcount(adjacent(new_p) & new_empty) - potential_o) / 32.0
        j += 1
    if len(moves):
        for j in range(len(moves)):
            features[j, 26] = 1.0 if flip_counts[j] == np.max(flip_counts) else 0.0
            features[j, 27] = 1.0 if flip_counts[j] == np.min(flip_counts) else 0.0
    return moves, features


def manual_weights():
    weights = np.zeros((len(FEATURE_NAMES), 3), dtype=np.float64)
    values = {'square_corner': 4, 'C_corner_empty': -1.5, 'X_corner_empty': -2,
              'square_edge_A': .5, 'square_edge_B': .5, 'flips_scaled': .5,
              'opponent_mobility': -1, 'opponent_corner_moves': -4,
              'anchored_edge_gain': 2, 'own_frontier_change': -.5}
    for name, value in values.items():
        weights[FEATURE_NAMES.index(name), :] = value
    return weights


class RulePolicy:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (len(FEATURE_NAMES), 3) or not np.isfinite(self.weights).all():
            raise ValueError('Wrong rule weight shape or nonfinite weights')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data['features'] != list(FEATURE_NAMES) or data['phases'] != list(PHASE_NAMES):
            raise ValueError('Feature schema differs')
        return cls(data['weights'])

    def predict(self, player, opponent):
        player, opponent = np.uint64(player), np.uint64(opponent)
        moves, features = extract(player, opponent)
        result = np.zeros(64, dtype=np.float64)
        if len(moves):
            phase = np.asarray(phase_weights(player, opponent))
            logits = features @ self.weights @ phase
            values = np.exp(logits - logits.max())
            result[moves] = values / values.sum()
        return result
