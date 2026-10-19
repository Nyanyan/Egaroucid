"""Fine-tune ordinary WTHOR weights on prepared, uniformly weighted mistakes."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from fine_tune_policy import (DEFAULT_MODEL, PIPELINE, Tee, atomic_json, read_json, run_lock)
from pipeline_common import load_manifest
from prepared_policy_data import load_prepared_splits
from wthor_dataset import file_sha256


def inspect_inputs(args):
    if (args.epochs < 1 or args.batch_size < 1 or args.steps_per_execution < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Invalid training settings")
    directory = args.board_split_dir.resolve()
    manifest = load_manifest(directory, "board_split_increased")
    if (manifest.get("sample_weights") or manifest.get("split_unit") != "identical_transcript_group"
            or manifest.get("augmentation_factor") != dict(train=8, val=1, test=1)
            or manifest.get("selection", {}).get("minimum_loss") != args.min_loss
            or manifest["selection"].get("comparison") != ">="):
        raise ValueError("Expected uniform loss-filtered data with train x8 and val/test x1")
    for name, factor in (("train", 8), ("val", 1), ("test", 1)):
        split = manifest["splits"][name]
        if split["positions"] != factor * split["selected_source_positions"]:
            raise ValueError("Incorrect per-split augmentation count")
    model = args.base_model.resolve()
    dataset = read_json(model.parent / "prepared_dataset.json")
    if dataset.get("dataset") != manifest["source_dataset"] or dataset.get("sample_weights"):
        raise ValueError("Base model must use the same ordinary, uniformly weighted WTHOR dataset")
    summary = read_json(model.parent / "selected_summary.json")
    architecture = summary.get("spec", {}).get("architecture")
    if architecture != "conv8_bn":
        raise ValueError("Expected the ordinary Conv8+BN model")
    base = dict(path=str(model), sha256=file_sha256(model), architecture=architecture,
                evidence={name: file_sha256(model.parent / name) for name in
                          ("prepared_dataset.json", "selected_summary.json", "training_identity.json")})
    return directory, manifest, base


def compile_model(model, args):
    import tensorflow as tf
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    # A loaded float32 root can contain mixed_float16 layers. Explicit scaling is
    # necessary because Keras' automatic check only inspects the root policy.
    if any(layer.dtype_policy.name == "mixed_float16" for layer in model.layers):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  steps_per_execution=args.steps_per_execution)


def run(args):
    print("Validating model/data identity...", flush=True)
    directory, manifest, base = inspect_inputs(args)
    print(f"Base model: {base['path']}\nLoss >= {args.min_loss}; {manifest['splits']}", flush=True)
    if args.dry_run:
        return
    import tensorflow as tf
    from train_policy_network import (enable_memory_growth, set_reproducible_seed,
                                      evaluate_policy_topn_arrays)
    from training_runtime import TrainingRuntime

    enable_memory_growth()
    set_reproducible_seed(args.seed)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU required for this training run")
    output = args.output_dir.resolve()
    if output == directory or directory in output.parents or output == args.base_model.parent.resolve():
        raise ValueError("Choose a separate training output directory")
    sources = [Path(__file__), PIPELINE / "11_fine_tuning/fine_tune_policy.py",
               PIPELINE / "00_data_processing/prepare_loss_filtered_finetuning.py",
               PIPELINE / "00_data_processing/pipeline_common.py", PIPELINE / "00_data_processing/policy_data.py",
               PIPELINE / "policy_accuracy.py", PIPELINE / "10_train_policy_network/prepared_policy_data.py",
               PIPELINE / "10_train_policy_network/training_runtime.py",
               PIPELINE / "10_train_policy_network/train_policy_network.py"]
    identity = dict(settings={key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()
                              if key not in ("resume", "dry_run", "output_dir", "progress_interval_sec")},
                    base_model=base, tensorflow=tf.__version__, source_directory=str(directory),
                    source_manifest_sha256=file_sha256(directory / "manifest.json"),
                    selection="minimum validation cross entropy; earliest tie; epochs 1..N",
                    augmentation=dict(train=8, val=1, test=1), sample_weights="uniform",
                    optimizer="fresh Adam; dynamic loss scaling for mixed_float16 layers",
                    sources={str(path.relative_to(PIPELINE)): file_sha256(path) for path in sources})
    with run_lock(output):
        identity_path = output / "training_identity.json"
        if identity_path.exists():
            if not args.resume or read_json(identity_path) != identity:
                raise ValueError("Resume requires identical data/model/code/settings")
            if (output / "run_status.json").exists() and read_json(output / "run_status.json")["state"] == "complete":
                if file_sha256(output / "selected_model.h5") != read_json(output / "selected_summary.json")["model_sha256"]:
                    raise ValueError("Completed model checksum differs")
                print("Verified completed run.", flush=True)
                return
        else:
            if any(path.name != "training.lock" for path in output.iterdir()):
                raise FileExistsError("Choose an empty output directory")
            atomic_json(identity_path, identity)
        with (output / "train.log").open("a", encoding="utf-8") as log:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
            splits, started = {}, time.time()

            def status(state, **extra):
                atomic_json(output / "run_status.json", dict(state=state, pid=os.getpid(),
                            started_at_unix=started, updated_at_unix=time.time(), **extra))

            try:
                status("loading")
                splits = load_prepared_splits(directory, {}, args.seed, output)
                model = tf.keras.models.load_model(args.base_model, compile=False)
                if tuple(model.input_shape[1:]) != (128,) or tuple(model.output_shape[1:]) != (64,):
                    raise ValueError("Expected 128 board features and 64 policy outputs")
                initial_weights = hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()
                for layer in model.layers:
                    layer.trainable = True
                compile_model(model, args)
                if int(model.optimizer.iterations.numpy()) != 0:
                    raise ValueError("Fine-tuning requires a fresh optimizer")
                if hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest() != initial_weights:
                    raise ValueError("Pretrained weights changed during initialization")
                baseline_path = output / "baseline_validation.json"
                if not baseline_path.exists():
                    baseline_loss = float(model.evaluate(splits["val"].as_dataset(args.batch_size), verbose=0))
                    if not math.isfinite(baseline_loss):
                        raise ValueError("Nonfinite baseline validation loss")
                    atomic_json(baseline_path, dict(validation_loss=baseline_loss, model_sha256=base["sha256"]))
                runtime = TrainingRuntime(model, output, args.epochs,
                            math.ceil(splits["train"].positions / args.batch_size), args.progress_interval_sec, True)
                best_dir = output / "best_epochs"
                best_dir.mkdir(exist_ok=True)

                class SaveBest(tf.keras.callbacks.Callback):
                    def __init__(self):
                        super().__init__()
                        self.best = min((row["val_loss"] for row in runtime.records), default=float("inf"))

                    def on_epoch_end(self, epoch, logs=None):
                        if not all(math.isfinite(float(logs[key])) for key in ("loss", "val_loss")):
                            raise ValueError("Nonfinite loss during training")
                        if float(logs["val_loss"]) < self.best:
                            self.best = float(logs["val_loss"])
                            destination = best_dir / f"epoch_{epoch+1:03d}.h5"
                            temporary = destination.with_name(destination.stem + ".tmp.h5")
                            self.model.save(temporary, include_optimizer=False)
                            temporary.replace(destination)
                            print(f"[selection] epoch {epoch+1}: val_loss={self.best:.6f}", flush=True)

                print(f"TensorFlow {tf.__version__}; GPU {tf.config.list_physical_devices('GPU')}", flush=True)
                print(f"All layers trainable; Adam lr={args.learning_rate}; batch={args.batch_size}; "
                      f"{splits['train'].positions:,} positions/epoch x {args.epochs} epochs; "
                      f"optimizer={type(model.optimizer).__name__}", flush=True)
                status("training", initial_epoch=runtime.initial_epoch, base_weights_sha256=initial_weights)
                if runtime.initial_epoch < args.epochs:
                    model.fit(splits["train"].as_dataset(args.batch_size, shuffle=True, seed=args.seed,
                                                        initial_epoch=runtime.initial_epoch),
                              validation_data=splits["val"].as_dataset(args.batch_size),
                              initial_epoch=runtime.initial_epoch, epochs=args.epochs,
                              callbacks=[SaveBest(), runtime], verbose=2)
                if len(runtime.records) != args.epochs:
                    raise ValueError("Not all requested epochs completed")
                best = min(runtime.records, key=lambda row: (row["val_loss"], row["epoch"]))
                temporary = output / "selected_model.tmp.h5"
                shutil.copyfile(best_dir / f"epoch_{best['epoch']:03d}.h5", temporary)
                temporary.replace(output / "selected_model.h5")
                del model
                tf.keras.backend.clear_session()
                status("evaluating", selected_epoch=best["epoch"])
                evaluation = {}
                # Test is inspected only after the adopted epoch has been fixed.
                for name, path in (("base", args.base_model), ("finetuned", output / "selected_model.h5")):
                    evaluated = tf.keras.models.load_model(path, compile=False)
                    evaluation[name] = {}
                    for split_name in ("val", "test"):
                        print(f"[evaluation] {name} {split_name}: {splits[split_name].positions:,} positions", flush=True)
                        result = evaluate_policy_topn_arrays(evaluated, split_name, splits[split_name],
                                                              (1, 3), 16384, args.batch_size)
                        if (result["positions"] != splits[split_name].positions
                                or result["invalid_policy_samples"] or result["illegal_label_samples"]
                                or not math.isfinite(result["cross_entropy"])):
                            raise ValueError("Invalid evaluation samples or nonfinite cross entropy")
                        evaluation[name][split_name] = result
                        print(f"[evaluation] CE={result['cross_entropy']:.6f}; "
                              f"Top-1={result['topn'][0]['accuracy']:.4%}", flush=True)
                    del evaluated
                    tf.keras.backend.clear_session()
                if file_sha256(args.base_model) != base["sha256"] or file_sha256(directory / "manifest.json") != identity["source_manifest_sha256"]:
                    raise ValueError("Source model/data changed during training")
                summary = dict(name=f"wthor_miss_{args.min_loss}_finetuned", base_model=base,
                    spec=dict(architecture=base["architecture"]), minimum_disc_loss=args.min_loss,
                    epochs_ran=len(runtime.records), adopted_epoch=best["epoch"], validation_loss=best["val_loss"],
                    adoption_rule=identity["selection"], test_used_for_selection=False,
                    train_positions_per_epoch=splits["train"].positions,
                    augmentation=manifest["augmentation_factor"], sample_weighting="uniform",
                    model_sha256=file_sha256(output / "selected_model.h5"),
                    evaluation=evaluation, elapsed_sec=time.time()-started)
                atomic_json(output / "selected_summary.json", summary)
                status("complete", returncode=0, selected_epoch=best["epoch"], model_sha256=summary["model_sha256"])
                print(f"COMPLETE: adopted epoch {best['epoch']}; {output / 'selected_model.h5'}", flush=True)
            except BaseException as error:
                status("failed", error=repr(error))
                raise
            finally:
                for split in splits.values():
                    split.close()
                sys.stdout, sys.stderr = old_stdout, old_stderr


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--board-split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-loss", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps-per-execution", type=int, default=32)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--progress-interval-sec", type=float, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(make_argparser().parse_args())
