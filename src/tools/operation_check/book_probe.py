"""GUIが保存したブックを、エンジンとは独立に構造検証する。"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from common import require


def inspect_book(path, count=None):
    data = path.read_bytes()
    positions = []
    if data.startswith(b"DICUORAGE\x03"):
        require(len(data) >= 14, "egbk3 header is truncated")
        declared = struct.unpack_from("<I", data, 10)[0]
        require(len(data) == 14 + 25 * declared, "egbk3 record count/size mismatch")
        for offset in range(14, len(data), 25):
            player, opponent, value, level, lines, leaf_value, leaf_move, leaf_level = struct.unpack_from("<QQbbIbbb", data, offset)
            require(-64 <= value <= 64, "invalid position value")
            require(leaf_move in range(65), "invalid leaf move")
            positions.append((player, opponent))
        kind = "egbk3"
    elif data.startswith(b"XADEKOOB\x04\x04"):
        require(len(data) >= 42, "Edax header is truncated")
        declared = struct.unpack_from("<I", data, 38)[0]
        offset = 42
        for _ in range(declared):
            require(offset + 40 <= len(data), "Edax record is truncated")
            player, opponent = struct.unpack_from("<QQ", data, offset)
            links = data[offset + 38]
            require(links <= 64, "invalid link count")
            offset += 40 + 2 * (links + 1)
            require(offset <= len(data), "Edax links are truncated")
            positions.append((player, opponent))
        require(offset == len(data), "Edax record count/size mismatch")
        kind = "edax"
    else:
        raise ValueError("Unknown book header")
    require(len(set(positions)) == declared, "duplicate positions")
    require(all(not (player & opponent) and bin(player | opponent).count("1") >= 4
                for player, opponent in positions), "invalid board")
    if count is not None:
        require(declared == count, f"Expected {count} positions, found {declared}")
    return {"passed": True, "format": kind, "positions": declared, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_book(args.book, args.count)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
