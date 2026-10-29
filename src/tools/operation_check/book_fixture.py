"""GUIのブック操作に使う小規模で決定的なegbk3を作成する。"""
import random
import struct
from pathlib import Path

from reference_board import Board


def representative(player, opponent):
    candidates = []
    for symmetry in range(8):
        transformed = [0, 0]
        for side, bits in enumerate((player, opponent)):
            for cell in range(64):
                if not (bits >> cell) & 1:
                    continue
                y, x = divmod(cell, 8)
                if symmetry & 1:
                    x, y = y, x
                if symmetry & 2:
                    x = 7 - x
                if symmetry & 4:
                    y = 7 - y
                transformed[side] |= 1 << (y * 8 + x)
        candidates.append(tuple(transformed))
    return min(candidates)


def create_book(path, seed=652):
    random_source = random.Random(seed)
    positions = set()
    for game in range(12):
        board = Board()
        for ply in range(11):
            player = sum(1 << (63 - i) for i, disc in enumerate(board.cells) if disc == board.turn)
            opponent = sum(1 << (63 - i) for i, disc in enumerate(board.cells) if disc not in (board.turn, "-"))
            positions.add(representative(player, opponent))
            legal = sorted(board.legal())
            if not legal:
                break
            board.play(random_source.choice(legal))
    payload = b"DICUORAGE\x03" + struct.pack("<I", len(positions))
    for player, opponent in sorted(positions):
        payload += struct.pack("<QQbbIbbb", player, opponent, 0, 1, 1, 127, 64, -1)
    path.write_bytes(payload)
    return len(positions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print("Created", create_book(args.path), "positions:", args.path)
