"""Board utilities and durable records shared by the eval-stick tools."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
JST = dt.timezone(dt.timedelta(hours=9))
DIRECTIONS = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x]


def stamp():
    return dt.datetime.now(JST).isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def json_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def coord(sq):
    return chr(97 + sq % 8) + str(1 + sq // 8)


def square(move):
    if len(move) != 2 or move[0] not in "abcdefgh" or move[1] not in "12345678":
        raise ValueError(move)
    return (int(move[1]) - 1) * 8 + ord(move[0]) - 97


def flips(board, turn, sq):
    if board[sq] != "-":
        return []
    y, x = divmod(sq, 8)
    other = "O" if turn == "X" else "X"
    out = []
    for dy, dx in DIRECTIONS:
        yy, xx = y + dy, x + dx
        line = []
        while 0 <= yy < 8 and 0 <= xx < 8 and board[yy * 8 + xx] == other:
            line.append(yy * 8 + xx)
            yy, xx = yy + dy, xx + dx
        if line and 0 <= yy < 8 and 0 <= xx < 8 and board[yy * 8 + xx] == turn:
            out.extend(line)
    return out


def legal(board, turn):
    return [coord(i) for i in range(64) if flips(board, turn, i)]


def play(board, turn, move):
    sq = square(move)
    ff = flips(board, turn, sq)
    if not ff:
        raise ValueError(("illegal move", board, turn, move))
    out = list(board)
    for i in ff + [sq]:
        out[i] = turn
    return "".join(out), "O" if turn == "X" else "X"


def transform(sq, symmetry):
    y, x = divmod(sq, 8)
    if symmetry >= 4:
        x = 7 - x
    for _ in range(symmetry % 4):
        y, x = x, 7 - y
    return y * 8 + x


def canonical(board, turn):
    variants = []
    for s in range(8):
        out = ["-"] * 64
        for i, ch in enumerate(board):
            out[transform(i, s)] = ch
        variants.append(("".join(out), s))
    b, s = min(variants)
    return b, turn, s


def transform_move(move, symmetry):
    return coord(transform(square(move), symmetry))


class EngineLock:
    """OS lock shared by builder and scorer; a crash releases the lock."""
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "reports/data_20261026_3/engine.lock")

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0)
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *args):
        self.file.close()


def kill_and_reap(process):
    if process.poll() is None:
        process.kill()  # terminate exactly the child engine PID we created
    process.wait()


def process_guard():
    if os.name != "nt":
        return
    command = ("Get-Process | Where-Object { $_.ProcessName -match "
               "'^(eg_rep_|Egaroucid|wEdax|clang|exact_helper|stick_)' } | "
               "Select-Object Id,ProcessName | ConvertTo-Json -Compress")
    p = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                       capture_output=True, text=True)
    if p.returncode or p.stdout.strip():
        raise RuntimeError("Another engine/compiler is active: " + p.stdout + p.stderr)
