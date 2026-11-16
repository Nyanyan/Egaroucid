"""Lossless opening-policy graph import and compact spanning-tree serialization.

All board keys contain both complete bitboards. Geometric symmetries and
transpositions are merged before choosing a spanning tree. Each tree node stores
the union of policy moves and outgoing tree edges once, by legal-move rank.
"""
from collections import deque
from functools import lru_cache
import math
import re
import struct

MASK = (1 << 64)-1
popcount = getattr(int, 'bit_count', lambda value: bin(value).count('1'))
INITIAL = (0x0000000810000000, 0x0000001008000000)
RANGES = ((0x3220, 0x4DB4), (0x4DC0, 0x9FEE), (0xAC00, 0xD7A2))
DIRECTIONS = ((1, 0x7f7f7f7f7f7f7f7f), (-1, 0xfefefefefefefefe),
              (8, MASK), (-8, MASK), (9, 0x7f7f7f7f7f7f7f7f),
              (7, 0xfefefefefefefefe), (-7, 0x7f7f7f7f7f7f7f7f),
              (-9, 0xfefefefefefefefe))
COMB = [[math.comb(n, k) if k <= n else 0 for k in range(34)] for n in range(34)]
BOOK_MAGIC = b'BTBC'
BOOK_VERSION = 1
BOOK_HEADER = struct.Struct('<4sBB2xIII')


def shift(value, direction, mask):
    value &= mask
    return ((value << direction) & MASK) if direction > 0 else value >> -direction


@lru_cache(maxsize=262144)
def legal(p, o):
    moves = 0
    for direction, mask in DIRECTIONS:
        captured = shift(p, direction, mask) & o
        for _ in range(5):
            captured |= shift(captured, direction, mask) & o
        moves |= shift(captured, direction, mask)
    return moves & ~(p | o) & MASK


def play(p, o, bit):
    captured = 0
    for direction, mask in DIRECTIONS:
        cursor = shift(bit, direction, mask)
        line = 0
        while cursor & o:
            line |= cursor
            cursor = shift(cursor, direction, mask)
        if cursor & p:
            captured |= line
    if not captured or bit & (p | o):
        raise ValueError('illegal book move')
    return o ^ captured, p ^ captured ^ bit


def normalize_pass(board):
    p, o = board
    return (o, p) if not legal(p, o) and legal(o, p) else board


def transform(x, symmetry):
    if symmetry & 1:
        for distance, mask in ((9, 0x0055005500550055), (18, 0x0000333300003333),
                               (36, 0x000000000f0f0f0f)):
            a = (x ^ (x >> distance)) & mask
            x ^= a ^ (a << distance)
    if symmetry & 2:
        x = int.from_bytes(x.to_bytes(8, 'little'), 'big')
    if symmetry & 4:
        for distance, mask in ((1, 0x5555555555555555), (2, 0x3333333333333333),
                               (4, 0x0f0f0f0f0f0f0f0f)):
            x = ((x & mask) << distance) | ((x >> distance) & mask)
    return x


INVERSE = [next(t for t in range(8) if all(transform(transform(1 << i, s), t) == 1 << i
                                         for i in (0, 1, 8, 23))) for s in range(8)]


@lru_cache(maxsize=262144)
def canonical(p, o):
    candidates = [(transform(p, s), transform(o, s), s) for s in range(8)]
    best = min(candidates)
    return best[:2], tuple(s for a, b, s in candidates if (a, b) == best[:2])


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


def legacy_code(ch):
    offset = 0
    for lo, hi in RANGES:
        if lo <= ord(ch) < hi:
            return ord(ch)-lo+offset
        offset += hi-lo
    raise ValueError('invalid legacy book character')


def legacy_symbols(text):
    packed = ''.join(re.findall(r'u"([^"\n]*)"', text))
    if not packed or len(packed) % 2:
        raise ValueError('legacy book must contain pairs of packed characters')
    result = []
    for i in range(0, len(packed), 2):
        value = (legacy_code(packed[i]) << 15) | legacy_code(packed[i+1])
        result.extend((value >> (6*j)) & 63 for j in range(4, -1, -1))
    return result, len(packed)


class PolicyGraph:
    def __init__(self):
        self.policy = {}
        self.edges = {}
        self.imported_moves = 0

    def add(self, board, bit, register):
        child = normalize_pass(play(*board, bit))
        key, symmetries = canonical(*board)
        child_key, _ = canonical(*child)
        edges = self.edges.setdefault(key, {})
        for symmetry in symmetries:
            move = transform(bit, symmetry)
            if register:
                self.policy[key] = self.policy.get(key, 0) | move
            old = edges.setdefault(move, child_key)
            assert old == child_key
        self.imported_moves += 1
        return child

    def import_legacy(self, text, side):
        symbols, characters = legacy_symbols(text)
        memo, board, player, i = [], INITIAL, 0, 0
        while i < len(symbols):
            cell = symbols[i]
            i += 1
            if cell == 27:
                break
            if cell == 35:
                memo.append((board, player))
                continue
            if cell in (36, 28):
                index = len(memo)-1
                if cell == 28:
                    if i+2 > len(symbols):
                        raise ValueError('truncated legacy memo reference')
                    index = (symbols[i] << 6) | symbols[i+1]
                    i += 2
                if not 0 <= index < len(memo):
                    raise ValueError('invalid legacy memo reference')
                board, player = memo[index]
                continue
            if cell == 26:
                board, player = INITIAL, 0
            if not legal(*board):
                board, player = board[::-1], player ^ 1
            bit = 1 << cell
            if not bit & legal(*board):
                raise ValueError(f'illegal legacy move at symbol {i-1}')
            self.add(board, bit, player == side)
            board, player = play(*board, bit), player ^ 1
        return {'characters': characters, 'symbols': len(symbols), 'consumed_symbols': i,
                'memos': len(memo)}

    def import_transcript(self, transcript, side, start_ply=0, policy_plies=None):
        if not re.fullmatch(r'(?:[a-h][1-8])*', transcript):
            raise ValueError('transcript must contain lowercase coordinates')
        if side not in (0, 1) or not 0 <= start_ply <= len(transcript)//2:
            raise ValueError('invalid side or start_ply')
        board, player = INITIAL, 0
        selected = None if policy_plies is None else set(policy_plies)
        if selected is not None and any(type(ply) is not int or not start_ply <= ply < len(transcript)//2 for ply in selected):
            raise ValueError('policy_plies must reference moves after the opening')
        for i in range(0, len(transcript), 2):
            if not legal(*board):
                board, player = board[::-1], player ^ 1
            cell = (int(transcript[i+1])-1)*8+ord(transcript[i])-ord('a')
            bit = 1 << (63-cell)
            if selected is not None and i//2 in selected and player != side:
                raise ValueError('policy_plies includes an opponent move')
            register = player == side and i//2 >= start_ply and (selected is None or i//2 in selected)
            self.add(board, bit, register)
            board, player = play(*board, bit), player ^ 1

    def tree(self, max_discs=60, dfs=False):
        retained = {key: value for key, value in self.policy.items()
                    if popcount(key[0] | key[1]) <= max_discs}
        # Keep precisely the ancestors needed to reach retained policy nodes.
        reverse = {}
        for parent, edges in self.edges.items():
            for child in set(edges.values()):
                reverse.setdefault(child, []).append(parent)
        needed = set(retained)
        queue = list(needed)
        for child in queue:
            for parent in reverse.get(child, ()):
                if parent not in needed:
                    needed.add(parent)
                    queue.append(parent)
        root, _ = canonical(*INITIAL)
        if root not in needed:
            return [], retained
        pending, seen, nodes = deque([root]), {root}, []
        while pending:
            board = pending.pop() if dfs else pending.popleft()
            children = 0
            for bit, child in sorted(self.edges.get(board, {}).items()):
                if child in needed and child not in seen:
                    children |= bit
                    seen.add(child)
                    pending.append(child)
            policy = retained.get(board, 0)
            nodes.append((board, policy, children))
        assert set(retained) <= seen, 'unreachable imported policy'
        # A non-policy path can become redundant after another parent wins a
        # transposition. Remove its now-empty tail from the spanning tree.
        useful, pruned = set(), []
        for board, policy, children in reversed(nodes):
            children = sum(bit for bit in bits(children)
                           if self.edges[board][bit] in useful)
            if policy or children:
                useful.add(board)
                pruned.append((board, policy, children))
        return list(reversed(pruned)), retained


class BinaryEncoder:
    def __init__(self, arithmetic):
        self.arithmetic = arithmetic
        self.low, self.high, self.pending = 0, (1 << 32)-1, 0
        self.prob = [[1, 1] for _ in range(24)]
        self.output = []

    def emit(self, bit):
        self.output.append(bit)
        self.output.extend([1-bit]*self.pending)
        self.pending = 0

    def bit(self, value, context=-1):
        if not self.arithmetic:
            self.output.append(value)
            return
        p = self.prob[context] if context >= 0 else (1, 1)
        split = self.low+((self.high-self.low+1)*p[0]//sum(p))-1
        if value:
            self.low = split+1
        else:
            self.high = split
        while True:
            if self.high < 0x80000000:
                self.emit(0)
            elif self.low >= 0x80000000:
                self.emit(1)
                self.low -= 0x80000000
                self.high -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xc0000000:
                self.pending += 1
                self.low -= 0x40000000
                self.high -= 0x40000000
            else:
                break
            self.low = (self.low << 1) & 0xffffffff
            self.high = ((self.high << 1) | 1) & 0xffffffff
        if context >= 0:
            p[value] += 1
            if sum(p) >= 4096:
                p[:] = [(v+1)//2 for v in p]

    def uniform(self, value, total):
        assert 0 <= value < total
        n = total.bit_length()-1
        cutoff = (1 << (n+1))-total
        if value < cutoff:
            encoded, width = value, n
        else:
            encoded, width = value+cutoff, n+1
        for shift in range(width-1, -1, -1):
            self.bit((encoded >> shift) & 1)

    def finish(self):
        if self.arithmetic:
            self.pending += 1
            self.emit(0 if self.low < 0x40000000 else 1)
        return self.output


class BinaryDecoder:
    def __init__(self, data, arithmetic):
        self.data, self.arithmetic, self.position = data, arithmetic, 0
        self.low, self.high, self.value = 0, 0xffffffff, 0
        self.prob = [[1, 1] for _ in range(24)]
        if arithmetic:
            for _ in range(32):
                self.value = (self.value << 1) | self.read()

    def read(self):
        bit = self.data[self.position] if self.position < len(self.data) else 0
        self.position += 1
        return bit

    def bit(self, context=-1):
        if not self.arithmetic:
            return self.read()
        p = self.prob[context] if context >= 0 else (1, 1)
        split = self.low+((self.high-self.low+1)*p[0]//sum(p))-1
        bit = int(self.value > split)
        if bit:
            self.low = split+1
        else:
            self.high = split
        while True:
            if self.high < 0x80000000:
                pass
            elif self.low >= 0x80000000:
                self.low -= 0x80000000
                self.high -= 0x80000000
                self.value -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xc0000000:
                self.low -= 0x40000000
                self.high -= 0x40000000
                self.value -= 0x40000000
            else:
                break
            self.low = (self.low << 1) & 0xffffffff
            self.high = ((self.high << 1) | 1) & 0xffffffff
            self.value = ((self.value << 1) | self.read()) & 0xffffffff
        if context >= 0:
            p[bit] += 1
            if sum(p) >= 4096:
                p[:] = [(v+1)//2 for v in p]
        return bit

    def uniform(self, total):
        n = total.bit_length()-1
        cutoff = (1 << (n+1))-total
        value = 0
        for _ in range(n):
            value = (value << 1) | self.bit()
        return value if value < cutoff else ((value << 1) | self.bit())-cutoff


def encode_tree(nodes, arithmetic, parent_context=False):
    coder = BinaryEncoder(arithmetic)
    incoming = {}
    for board, policy, children in nodes:
        legal_bits = list(bits(legal(*board)))
        selected = [i for i, bit in enumerate(legal_bits) if (policy | children) & bit]
        n, count = len(legal_bits), len(selected)
        assert n <= 33 and count > 0
        stage = min(3, (popcount(board[0] | board[1])-4)//12)
        for k in range(1, n):
            more = int(count > k)
            coder.bit(more, stage*3+min(k-1, 2))
            if not more:
                break
        rank = sum(COMB[value][i+1] for i, value in enumerate(selected))
        coder.uniform(rank, COMB[n][count])
        for index in selected:
            bit = legal_bits[index]
            registered = int(bool(policy & bit))
            coder.bit(registered, 12+2*stage+incoming.get(board,0) if parent_context else 12+stage)
            if registered:
                coder.bit(int(bool(children & bit)), (20 if parent_context else 16)+stage)
            if parent_context and children & bit:
                incoming[canonical(*normalize_pass(play(*board,bit)))[0]] = registered
    return coder.finish()


def decode_tree(data, count, arithmetic, dfs=False, parent_context=False):
    if count == 0:
        return {}
    coder = BinaryDecoder(data, arithmetic)
    root, _ = canonical(*INITIAL)
    pending, result = deque([(root,0)]), {}
    for _ in range(count):
        board, incoming = pending.pop() if dfs else pending.popleft()
        legal_bits = list(bits(legal(*board)))
        n, selected_count = len(legal_bits), 1
        stage = min(3, (popcount(board[0] | board[1])-4)//12)
        for k in range(1, n):
            if not coder.bit(stage*3+min(k-1, 2)):
                break
            selected_count += 1
        rank = coder.uniform(COMB[n][selected_count])
        selected, limit = [], n-1
        for k in range(selected_count, 0, -1):
            while COMB[limit][k] > rank:
                limit -= 1
            selected.append(limit)
            rank -= COMB[limit][k]
            limit -= 1
        assert rank == 0
        policy = 0
        for index in reversed(selected):
            bit = legal_bits[index]
            registered = coder.bit(12+2*stage+incoming if parent_context else 12+stage)
            follow = coder.bit((20 if parent_context else 16)+stage) if registered else 1
            if registered:
                policy |= bit
            if follow:
                child = normalize_pass(play(*board, bit))
                pending.append((canonical(*child)[0],registered))
        if policy:
            assert board not in result
            result[board] = policy
    assert not pending
    return result


def pack_bits(data):
    padded = data+[0]*((-len(data)) % 8)
    return bytes(sum(padded[i+j] << (7-j) for j in range(8))
                 for i in range(0, len(padded), 8))


def unpack_bits(packed):
    return [(value >> shift) & 1 for value in packed for shift in range(7, -1, -1)]


def pack_book(data, nodes, positions, arithmetic, dfs=False, parent_context=False):
    flags = int(arithmetic) | (int(dfs) << 1) | (int(parent_context) << 2)
    return BOOK_HEADER.pack(BOOK_MAGIC, BOOK_VERSION, flags, nodes, positions, len(data)) + pack_bits(data)


def decode_book(packed):
    if len(packed) < BOOK_HEADER.size:
        raise ValueError('truncated book header')
    magic, version, flags, nodes, positions, bit_count = BOOK_HEADER.unpack_from(packed)
    if magic != BOOK_MAGIC or version != BOOK_VERSION or flags & ~7:
        raise ValueError('unsupported book format')
    payload = packed[BOOK_HEADER.size:]
    if len(payload) != (bit_count+7)//8:
        raise ValueError('book payload length does not match bit count')
    data = unpack_bits(payload)
    if any(data[bit_count:]):
        raise ValueError('book padding bits must be zero')
    result = decode_tree(data[:bit_count], nodes, bool(flags & 1), bool(flags & 2), bool(flags & 4))
    if len(result) != positions:
        raise ValueError('decoded book position count does not match header')
    return result
