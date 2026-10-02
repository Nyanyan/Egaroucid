"""Progress and epoch-boundary recovery for long prepared-data training runs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import time

import tensorflow as tf


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def check_training_identity(output, directory, args):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in {"resume_training", "output_dir", "progress_interval_sec"}}
    sources = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")
               if not p.name.startswith("test_")}
    identity = {"settings": settings, "tensorflow": tf.__version__,
                "manifest_sha256": digest(directory / "manifest.json"), "sources": sources}
    path = output / "training_identity.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume_training or not path.is_file():
            raise FileExistsError("Nonempty training output requires --resume-training and training_identity.json")
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Training settings/data/code changed; choose a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_text(path, json.dumps(identity, indent=2))


class TrainingRuntime(tf.keras.callbacks.Callback):
    def __init__(self, model, run_dir, epochs, steps, interval=30, checkpoint=False):
        super().__init__()
        self.run_dir = run_dir
        self.epochs, self.steps, self.interval = epochs, steps, interval
        self.records = []
        self.initial_epoch = 0
        self.manager = None
        if checkpoint:
            self.checkpoint = tf.train.Checkpoint(
                model=model, optimizer=model.optimizer,
                completed_epochs=tf.Variable(0, dtype=tf.int64, trainable=False),
                history_json=tf.Variable("[]", dtype=tf.string, trainable=False))
            self.manager = tf.train.CheckpointManager(self.checkpoint, str(run_dir / "checkpoint"), max_to_keep=2)
            if self.manager.latest_checkpoint:
                # Optimizer slot variables are restored lazily on the next update.
                self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
                self.initial_epoch = int(self.checkpoint.completed_epochs.numpy())
                self.records = json.loads(self.checkpoint.history_json.numpy().decode())
                print(f"Restored epoch {self.initial_epoch}/{epochs}: {self.manager.latest_checkpoint}", flush=True)
        self.started = self.last_report = time.monotonic()
        self.epoch = self.initial_epoch
        self.epoch_started = self.started

    def report(self, batch, phase, force=False):
        now = time.monotonic()
        if not force and now - self.last_report < self.interval:
            return
        self.last_report = now
        done = (self.epoch - self.initial_epoch) * self.steps + batch
        remaining = max(0, (self.epochs - self.epoch) * self.steps - batch)
        elapsed = now - self.started
        eta = elapsed / done * remaining if done else None
        progress = {"epoch": min(self.epoch + 1, self.epochs), "epochs": self.epochs,
                    "batch": batch, "batches": self.steps, "phase": phase,
                    "elapsed_sec": elapsed, "eta_sec": eta,
                    "completed_epochs": len(self.records), "updated_at_unix": time.time()}
        atomic_text(self.run_dir / "training_progress.json", json.dumps(progress, indent=2))
        eta_text = f"{eta / 3600:.2f} h" if eta is not None else "warming up"
        print(f"[training] epoch {progress['epoch']}/{self.epochs} | {phase} | "
              f"batch {batch:,}/{self.steps:,} | elapsed {elapsed / 3600:.2f} h | ETA {eta_text}", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_started = time.monotonic()
        self.report(0, "train", True)

    def on_train_batch_end(self, batch, logs=None):
        self.report(batch + 1, "train")

    def on_test_begin(self, logs=None):
        self.report(self.steps, "validation", True)

    def on_test_batch_end(self, batch, logs=None):
        self.report(self.steps, "validation")

    def on_epoch_end(self, epoch, logs=None):
        self.records.append({"epoch": epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}})
        if self.manager is not None:
            self.checkpoint.completed_epochs.assign(epoch + 1)
            self.checkpoint.history_json.assign(json.dumps(self.records))
            self.manager.save(checkpoint_number=epoch + 1)
        self.write_history()
        self.report(self.steps, "epoch saved", True)

    def write_history(self):
        if not self.records:
            return
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(self.records[0]))
        writer.writeheader()
        writer.writerows(self.records)
        atomic_text(self.run_dir / "history.csv", buffer.getvalue())

