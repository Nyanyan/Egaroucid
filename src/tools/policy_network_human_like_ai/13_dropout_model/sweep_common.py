"""Portable experiment metadata; importing this module does not initialize CUDA."""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
BANDS = tuple(f"{low}-{low + 200}" for low in range(800, 2200, 200))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_root():
    if os.environ.get("POLICY_DATA_ROOT"):
        return Path(os.environ["POLICY_DATA_ROOT"])
    if os.environ.get("EGAROUCID_DATA"):
        return Path(os.environ["EGAROUCID_DATA"]) / "20261002_policy_network_human_like_data"
    return None


def write_csv(path, rows, columns=None):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@contextmanager
def run_lock(output):
    """An OS lock is released on exit or interruption, including a closed console."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"This experiment is already running: {output}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
