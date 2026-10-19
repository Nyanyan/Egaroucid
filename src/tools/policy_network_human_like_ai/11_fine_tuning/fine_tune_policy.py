#!/usr/bin/env python3
"""Fine-tune a saved WTHOR policy on a game-disjoint Quest rating band."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
WORKSPACE = HERE.parents[4]
for directory in (PIPELINE, PIPELINE / "00_data_processing", PIPELINE / "10_train_policy_network"):
    sys.path.insert(0, str(directory))

from board_symmetry import augment_positions
from pipeline_common import load_manifest
from prepared_policy_data import PreparedSplit, load_prepared_splits
from wthor_dataset import file_sha256

DEFAULT_MODEL = WORKSPACE / "experiment_bat/20261002/results/train/selected_model.h5"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def run_lock(directory):
    """An OS lock is released even if the training process is terminated."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.lock").open("a+b") as stream:
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


class AugmentedTrain(PreparedSplit):
    """Expand all eight symmetries in bounded blocks, only for training."""
    def __init__(self, source):
        self.source = source
        self.positions = source.positions * 8
        self.weights = []

    def batches(self, batch_size, *, shuffle=False, seed=613, include_weights=False):
        # Each full expanded block is divisible by batch_size, including odd sizes.
        block = batch_size // math.gcd(batch_size, 8) * max(1, 8192 // batch_size)
        starts = np.arange(0, self.source.positions, block)
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(starts)
        for start in starts:
            samples = self.source.read(int(start), min(int(start) + block, self.source.positions))
            samples, _ = augment_positions(samples, np.zeros(len(samples), dtype=np.uint64))
            if shuffle:
                rng.shuffle(samples)
            for offset in range(0, len(samples), batch_size):
                batch = samples[offset:offset + batch_size]
                yield (batch, None) if include_weights else batch


def inspect_inputs(args):
    if args.epochs < 1 or args.batch_size < 1 or args.steps_per_execution < 1:
        raise ValueError("epochs, batch size and steps per execution must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    for name in ("max_train_samples", "max_val_samples"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    directory = args.board_split_dir
    if directory is None:
        if args.quest_root is None:
            raise ValueError("Set EGAROUCID_DATA, POLICY_DATA_ROOT, or --quest-root")
        directory = args.quest_root / "board_split" / args.rating_band
    directory = directory.resolve()
    manifest = load_manifest(directory, "board_split")
    if (manifest.get("split_unit") != "identical_transcript_group_across_all_rating_bands"
            or "rating_range" not in manifest or manifest.get("sample_weights")):
        raise ValueError("Expected unaugmented, uniformly weighted, game-disjoint Quest rating-band data")
    if manifest["dataset"] != args.rating_band:
        raise ValueError("--rating-band must match the Quest manifest's dataset")
    model = args.base_model.resolve()
    dataset = read_json(model.parent / "prepared_dataset.json")
    if dataset.get("dataset") != "WTHOR_1977_2024" or dataset.get("sample_weights"):
        raise ValueError("Base model must be ordinary, uniformly weighted WTHOR training")
    base_summary = read_json(model.parent / "selected_summary.json")
    return directory, manifest, {
        "path": str(model), "sha256": file_sha256(model),
        "architecture": base_summary.get("spec", {}).get("architecture"),
        "evidence": {name: file_sha256(model.parent / name) for name in
                     ("prepared_dataset.json", "selected_summary.json", "training_identity.json")},
    }


def training_identity(args, directory, manifest, base, tf_version):
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in ("resume", "dry_run", "output_dir", "progress_interval_sec")}
    sources = [Path(__file__), PIPELINE / "00_data_processing/board_symmetry.py",
               PIPELINE / "00_data_processing/policy_data.py",
               PIPELINE / "00_data_processing/pipeline_common.py", PIPELINE / "policy_accuracy.py",
               PIPELINE / "10_train_policy_network/prepared_policy_data.py",
               PIPELINE / "10_train_policy_network/training_runtime.py",
               PIPELINE / "10_train_policy_network/train_policy_network.py"]
    return dict(settings=settings, base_model=base, tensorflow=tf_version,
                source_directory=str(directory), source_manifest_sha256=file_sha256(directory / "manifest.json"),
                rating_range=manifest["rating_range"], selection="minimum validation cross entropy; earliest tie",
                augmentation=dict(train=args.augmentation, val=1, test=1),
                sources={str(p.relative_to(PIPELINE)): file_sha256(p) for p in sources})


def run(args):
    directory, manifest, base = inspect_inputs(args)
    print(f"Base WTHOR model: {base['path']}\nQuest band: {args.rating_band} {manifest['rating_range']}", flush=True)
    print(json.dumps(manifest["splits"], indent=2), flush=True)
    if args.dry_run:
        print("Input hashes and rating-band metadata checked; no training/output changes.", flush=True)
        return
    import tensorflow as tf
    from train_policy_network import enable_memory_growth, set_reproducible_seed, compile_model
    from training_runtime import TrainingRuntime

    enable_memory_growth()
    set_reproducible_seed(args.seed)
    if not tf.config.list_physical_devices("GPU") and not args.allow_cpu:
        raise RuntimeError("GPU not found; use --allow-cpu only for an intentional CPU run")
    output = args.output_dir.resolve()
    if output == directory or directory in output.parents or output == args.base_model.parent.resolve():
        raise ValueError("Use a separate output directory")
    identity = training_identity(args, directory, manifest, base, tf.__version__)
    with run_lock(output):
        identity_path = output / "training_identity.json"
        if identity_path.exists():
            if not args.resume or read_json(identity_path) != identity:
                raise ValueError("Existing run requires --resume with identical model/data/code/settings")
            if (output / "run_status.json").exists() and read_json(output / "run_status.json")["state"] == "complete":
                saved = read_json(output / "selected_summary.json")
                if file_sha256(output / "selected_model.h5") != saved["model_sha256"]:
                    raise ValueError("Completed model changed")
                print("Verified completed run; nothing to resume.", flush=True)
                return
        else:
            if any(p.name != "training.lock" for p in output.iterdir()):
                raise FileExistsError("Choose an empty output directory")
            atomic_json(identity_path, identity)
        with (output / "train.log").open("a", encoding="utf-8") as log:
            previous_stdout, previous_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
            splits = {}
            started = time.time()
            def status(state, **extra):
                atomic_json(output / "run_status.json", dict(state=state, pid=os.getpid(),
                            updated_at_unix=time.time(), started_at_unix=started, **extra))
            try:
                status("loading")
                caps = dict(train=args.max_train_samples, val=args.max_val_samples)
                splits = load_prepared_splits(directory, caps, args.seed, output)
                # The test partition is integrity-checked, never passed to fit or selection.
                train = AugmentedTrain(splits["train"]) if args.augmentation == 8 else splits["train"]
                metadata = read_json(output / "prepared_dataset.json")
                metadata.update(augmentation_factor=dict(train=args.augmentation, val=1, test=1),
                                rating_range=manifest["rating_range"], train_positions_per_epoch=train.positions)
                atomic_json(output / "prepared_dataset.json", metadata)
                model = tf.keras.models.load_model(args.base_model, compile=False)
                if tuple(model.input_shape[1:]) != (128,) or tuple(model.output_shape[1:]) != (64,):
                    raise ValueError("Expected 128 input features and 64 policy outputs")
                original_weights = hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()
                for layer in model.layers:
                    layer.trainable = True
                compile_model(model, args.learning_rate, args.steps_per_execution)
                if int(model.optimizer.iterations.numpy()) != 0:
                    raise ValueError("Fine-tuning must begin with a fresh optimizer")
                if hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest() != original_weights:
                    raise ValueError("Model initialization changed pretrained weights")
                runtime = TrainingRuntime(model, output, args.epochs,
                            math.ceil(train.positions / args.batch_size), args.progress_interval_sec, True)
                best_dir = output / "best_epochs"
                best_dir.mkdir(exist_ok=True)

                class SaveBest(tf.keras.callbacks.Callback):
                    def __init__(self):
                        super().__init__()
                        self.best = min((r["val_loss"] for r in runtime.records), default=float("inf"))

                    def on_epoch_end(self, epoch, logs=None):
                        if not all(math.isfinite(float(logs[k])) for k in ("loss", "val_loss")):
                            raise ValueError("Nonfinite training/validation loss")
                        value = float(logs["val_loss"])
                        if value < self.best:
                            self.best = value
                            destination = best_dir / f"epoch_{epoch + 1:03d}.h5"
                            temporary = destination.with_name(destination.stem + ".tmp.h5")
                            self.model.save(temporary, include_optimizer=False)
                            temporary.replace(destination)
                            print(f"[selection] epoch {epoch + 1}: val_loss={value:.6f}", flush=True)

                print(f"TensorFlow {tf.__version__}; GPU {tf.config.list_physical_devices('GPU')}", flush=True)
                print(f"All layers trainable; fresh Adam lr={args.learning_rate}; batch={args.batch_size}; "
                      f"{train.positions:,} positions/epoch; epochs={args.epochs}", flush=True)
                status("training", initial_epoch=runtime.initial_epoch, base_weights_sha256=original_weights)
                if runtime.initial_epoch < args.epochs:
                    model.fit(train.as_dataset(args.batch_size, shuffle=True, seed=args.seed,
                                               initial_epoch=runtime.initial_epoch),
                              validation_data=splits["val"].as_dataset(args.batch_size),
                              initial_epoch=runtime.initial_epoch, epochs=args.epochs,
                              callbacks=[SaveBest(), runtime], verbose=2)
                if len(runtime.records) != args.epochs:
                    raise ValueError("Training did not complete all requested epochs")
                best = min(runtime.records, key=lambda row: (row["val_loss"], row["epoch"]))
                chosen = best_dir / f"epoch_{best['epoch']:03d}.h5"
                temporary = output / "selected_model.tmp.h5"
                shutil.copyfile(chosen, temporary)
                temporary.replace(output / "selected_model.h5")
                saved = tf.keras.models.load_model(output / "selected_model.h5", compile=False)
                probe = next(iter(splits["val"].as_dataset(min(32, args.batch_size))))[0]
                probabilities = saved(probe, training=False).numpy()
                if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
                    raise ValueError("Saved policy failed probability checks")
                if file_sha256(args.base_model) != base["sha256"]:
                    raise ValueError("Base model changed during training")
                if file_sha256(directory / "manifest.json") != identity["source_manifest_sha256"]:
                    raise ValueError("Quest manifest changed during training")
                summary = dict(name="quest_finetuned_" + args.rating_band, base_model=base,
                    spec=dict(architecture=base["architecture"]), epochs_ran=len(runtime.records),
                    adopted_epoch=best["epoch"], validation_loss=best["val_loss"],
                    adoption_rule=identity["selection"], test_used_for_selection=False,
                    model_sha256=file_sha256(output / "selected_model.h5"),
                    train_positions_per_epoch=train.positions, rating_range=manifest["rating_range"],
                    elapsed_sec=time.time() - started)
                atomic_json(output / "selected_summary.json", summary)
                status("complete", returncode=0, selected_epoch=best["epoch"], model_sha256=summary["model_sha256"])
                print(f"COMPLETE: best epoch {best['epoch']}; model {output / 'selected_model.h5'}", flush=True)
            except BaseException as error:
                status("failed", error=repr(error))
                raise
            finally:
                for split in splits.values():
                    split.close()
                sys.stdout, sys.stderr = previous_stdout, previous_stderr


def make_argparser():
    data_root = os.environ.get("POLICY_DATA_ROOT")
    if not data_root and os.environ.get("EGAROUCID_DATA"):
        data_root = Path(os.environ["EGAROUCID_DATA"]) / "20261002_policy_network_human_like_data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quest-root", type=Path, default=Path(data_root) / "othello_quest_split" if data_root else None)
    parser.add_argument("--rating-band", default="under_800", help="Prepared band name, e.g. under_800 or 800-1000")
    parser.add_argument("--board-split-dir", type=Path, help="Override the prepared rating-band directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--steps-per-execution", type=int, default=32)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--progress-interval-sec", type=float, default=30)
    parser.add_argument("--max-train-samples", type=int, help="Smoke test only: original positions before augmentation")
    parser.add_argument("--max-val-samples", type=int, help="Smoke test only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(make_argparser().parse_args())
