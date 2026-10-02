"""Shared file contracts for raw -> transcripts -> split boards -> augmentation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from wthor_dataset import BOARD_DTYPE, SPLIT_NAMES, file_sha256


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=os.environ.get("POLICY_DATA_ROOT"),
                        help="Pipeline root containing raw_data, transcript, etc.; or set POLICY_DATA_ROOT.")
    parser.add_argument("--dataset", required=True, help="Dataset directory name (independent of its raw format).")
    parser.add_argument("--replace", action="store_true", help="Keep an existing nonempty output as a timestamped backup before replacing it.")


def dataset_directory(data_root: Path, stage: str, dataset: str) -> Path:
    if data_root is None:
        raise ValueError("pass --data-root or set POLICY_DATA_ROOT")
    if (not dataset or dataset in (".", "..") or dataset.endswith((".", " "))
            or any(c in dataset for c in '<>:"/\\|?*')
            or any(ord(c) < 32 for c in dataset)):
        raise ValueError("--dataset must be a single directory name")
    root = Path(data_root).resolve()
    result = root / stage / dataset
    if not result.resolve().is_relative_to(root) or result.resolve() == root:
        raise ValueError("dataset path escapes --data-root")
    return result


@contextmanager
def output_directory(destination: Path, replace: bool = False):
    """Publish only a complete stage; never erase a previous successful output."""
    destination = destination.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())) and not replace:
        raise FileExistsError(f"output exists: {destination}; use --replace to keep a backup and regenerate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)).resolve()
    backup = None
    try:
        yield temporary
        if destination.exists():
            if not destination.is_relative_to(destination.parent):
                raise ValueError("unsafe output path")
            if any(destination.iterdir()):
                backup = destination.with_name(destination.name + ".backup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
                if not backup.resolve().is_relative_to(destination.parent):
                    raise ValueError("unsafe backup path")
                destination.rename(backup)
                print("previous output saved:", backup, flush=True)
            else:
                destination.rmdir()
        temporary.rename(destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    finally:
        if temporary.exists():
            if not temporary.is_relative_to(destination.parent) or temporary == destination:
                raise ValueError("unsafe temporary cleanup path")
            shutil.rmtree(temporary)


def file_info(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": file_sha256(path)}


def write_manifest(directory: Path, stage: str, dataset: str, **metadata) -> dict:
    manifest = {"version": 1, "stage": stage, "dataset": dataset, **metadata}
    manifest["files"] = [file_info(p, directory) for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_manifest(directory: Path, expected_stage: str) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("stage") != expected_stage:
        raise ValueError(f"expected {expected_stage} pipeline output in {directory}")
    for entry in manifest["files"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("manifest path escapes dataset directory")
        if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"pipeline input checksum mismatch: {path}")
    return manifest


def source_info(directory: Path) -> dict:
    return {"directory": str(directory.resolve()), "manifest_sha256": file_sha256(directory / "manifest.json")}


class TranscriptWriter:
    def __init__(self, directory: Path, games_per_file: int):
        if games_per_file <= 0:
            raise ValueError("games_per_file must be positive")
        self.directory = directory
        self.limit = games_per_file
        self.games = 0
        self.text = self.ids = None
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, game_id: int, transcript: str) -> dict:
        if self.games % self.limit == 0:
            self.close()
            self.path = self.directory / f"{self.games // self.limit:07d}.txt"
            self.text = self.path.open("w", encoding="utf-8", newline="\n")
            self.ids = self.path.with_suffix(".game_ids").open("wb")
        self.text.write(transcript + "\n")
        self.ids.write(int(game_id).to_bytes(8, "little"))
        self.games += 1
        return {"file": self.path.name, "line": (self.games - 1) % self.limit + 1}

    def close(self):
        if self.text is not None:
            self.text.close()
            self.ids.close()
            self.text = self.ids = None


def iter_transcripts(directory: Path):
    for path in sorted(directory.glob("*.txt")):
        ids_path = path.with_suffix(".game_ids")
        if ids_path.stat().st_size % 8:
            raise ValueError(f"incomplete game ID in {ids_path}")
        ids = np.fromfile(ids_path, dtype="<u8")
        count = 0
        with path.open(encoding="utf-8") as source:
            for count, line in enumerate(source, 1):
                if count > len(ids) or not line.strip():
                    raise ValueError(f"transcript/game-ID mismatch in {path}:{count}")
                yield int(ids[count - 1]), line.strip(), path.name, count
        if count != len(ids):
            raise ValueError(f"transcript/game-ID count mismatch in {path}")


class BoardWriter:
    def __init__(self, directory: Path, positions_per_file: int):
        if positions_per_file <= 0:
            raise ValueError("positions_per_file must be positive")
        self.directory, self.limit, self.positions = directory, positions_per_file, 0
        self.board = self.ids = None
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, samples: np.ndarray, game_ids: np.ndarray):
        if samples.dtype != BOARD_DTYPE or len(samples) != len(game_ids):
            raise ValueError("board samples and game IDs must match")
        offset = 0
        while offset < len(samples):
            if self.positions % self.limit == 0:
                self.close()
                path = self.directory / f"{self.positions // self.limit:07d}.dat"
                self.board = path.open("wb")
                self.ids = path.with_suffix(".game_ids").open("wb")
            count = min(len(samples) - offset, self.limit - self.positions % self.limit)
            samples[offset:offset + count].tofile(self.board)
            game_ids[offset:offset + count].astype("<u8", copy=False).tofile(self.ids)
            self.positions += count
            offset += count

    def close(self):
        if self.board is not None:
            self.board.close()
            self.ids.close()
            self.board = self.ids = None
