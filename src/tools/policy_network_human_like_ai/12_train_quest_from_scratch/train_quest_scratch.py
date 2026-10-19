"""Train Conv8+BN from random weights on a game-disjoint Quest rating band."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scratch_common import (HERE, BASE, DEFAULT_RUN, read, write, file_sha256,
                            run_lock, AugmentedTrain, Tee)
from pipeline_common import load_manifest
from prepared_policy_data import load_prepared_splits


def parser():
    data = os.environ.get('POLICY_DATA_ROOT')
    if not data and os.environ.get('EGAROUCID_DATA'):
        data = Path(os.environ['EGAROUCID_DATA']) / '20261002_policy_network_human_like_data'
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--quest-root', type=Path, default=Path(data) / 'othello_quest_split' if data else None)
    result.add_argument('--rating-band', default='under_800')
    result.add_argument('--output-dir', type=Path, default=DEFAULT_RUN)
    result.add_argument('--epochs', type=int, default=10)
    result.add_argument('--batch-size', type=int, default=512)
    result.add_argument('--learning-rate', type=float, default=.001)
    result.add_argument('--seed', type=int, default=613)
    result.add_argument('--precision', choices=['float32', 'mixed_float16'], default='mixed_float16')
    result.add_argument('--steps-per-execution', type=int, default=32)
    result.add_argument('--progress-interval-sec', type=float, default=30)
    result.add_argument('--max-train-samples', type=int, help='Smoke test only, before eightfold augmentation')
    result.add_argument('--max-val-samples', type=int, help='Smoke test only')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--allow-cpu', action='store_true')
    result.add_argument('--dry-run', action='store_true')
    return result


def inspect_inputs(args):
    for name in ('epochs', 'batch_size', 'steps_per_execution', 'progress_interval_sec', 'learning_rate'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('max_train_samples', 'max_val_samples'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.quest_root is None:
        raise ValueError('Set EGAROUCID_DATA, POLICY_DATA_ROOT or --quest-root')
    directory = (args.quest_root / 'board_split' / args.rating_band).resolve()
    manifest = load_manifest(directory, 'board_split')
    if (manifest.get('split_unit') != 'identical_transcript_group_across_all_rating_bands'
            or manifest.get('sample_weights') or manifest['dataset'] != args.rating_band
            or 'rating_range' not in manifest):
        raise ValueError('Expected unaugmented, uniformly weighted, game-disjoint Quest data')
    output = args.output_dir.resolve()
    if output == directory or directory in output.parents:
        raise ValueError('Use a separate output directory')
    return directory, manifest


def fresh_model(args):
    # Construct every layer from its initializer. No load_model/load_weights path here.
    import tensorflow as tf
    from paper_policy_models import build_paper_model
    from train_policy_network import compile_model, set_reproducible_seed
    tf.keras.mixed_precision.set_global_policy(args.precision)
    set_reproducible_seed(args.seed)
    model = build_paper_model('conv8_bn', 'quest_scratch_' + args.rating_band, 0.0)
    compile_model(model, args.learning_rate, args.steps_per_execution)
    if tuple(model.input_shape[1:]) != (128,) or tuple(model.output_shape[1:]) != (64,):
        raise ValueError('Unexpected policy architecture')
    if int(model.optimizer.iterations.numpy()) != 0:
        raise ValueError('A new optimizer must have zero updates')
    return model


def run(args):
    directory, manifest = inspect_inputs(args)
    print(f'Random Conv8+BN initialization; Quest only: {directory}', flush=True)
    if args.dry_run:
        print(f"Verified manifest: {manifest['rating_range']}; {manifest['splits']}", flush=True)
        return
    import tensorflow as tf
    from train_policy_network import enable_memory_growth
    from training_runtime import TrainingRuntime
    enable_memory_growth()
    if not tf.config.list_physical_devices('GPU') and not args.allow_cpu:
        raise RuntimeError('GPU not found; use --allow-cpu only for an intentional CPU run')
    sources = [Path(__file__), HERE / 'scratch_common.py', BASE / '11_fine_tuning/fine_tune_policy.py',
               BASE / '00_data_processing/board_symmetry.py', BASE / '00_data_processing/policy_data.py',
               BASE / '00_data_processing/pipeline_common.py', BASE / 'policy_accuracy.py',
               BASE / '10_train_policy_network/prepared_policy_data.py',
               BASE / '10_train_policy_network/training_runtime.py',
               BASE / '10_train_policy_network/train_policy_network.py',
               BASE / '10_train_policy_network/paper_policy_models.py']
    identity = dict(initialization='random_from_architecture', pretrained_weights_loaded=False,
        architecture='conv8_bn', l2=0.0, tensorflow=tf.__version__,
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if k not in ('resume', 'dry_run', 'output_dir', 'progress_interval_sec')},
        source_directory=str(directory), source_manifest_sha256=file_sha256(directory / 'manifest.json'),
        rating_range=manifest['rating_range'], augmentation=dict(train=8, val=1, test=1),
        selection='minimum validation cross entropy; earliest tie',
        sources={str(p.relative_to(BASE)): file_sha256(p) for p in sources})
    output = args.output_dir.resolve()
    with run_lock(output):
        identity_path = output / 'training_identity.json'
        if identity_path.exists():
            if not args.resume or read(identity_path) != identity:
                raise ValueError('Existing run requires --resume with identical data/code/settings')
            if (output / 'run_status.json').exists() and read(output / 'run_status.json')['state'] == 'complete':
                if file_sha256(output / 'selected_model.h5') != read(output / 'selected_summary.json')['model_sha256']:
                    raise ValueError('Completed model changed')
                print('Verified completed run; nothing to resume.', flush=True)
                return
        else:
            if any(p.name != 'training.lock' for p in output.iterdir()):
                raise FileExistsError('Choose an empty output directory')
            write(identity_path, identity)
        with (output / 'train.log').open('a', encoding='utf-8') as log:
            previous = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
            splits = {}
            started = time.time()
            def status(state, **extra):
                write(output / 'run_status.json', dict(state=state, pid=os.getpid(), started_at_unix=started,
                    updated_at_unix=time.time(), **extra))
            try:
                status('loading')
                splits = load_prepared_splits(directory, dict(train=args.max_train_samples,
                    val=args.max_val_samples), args.seed, output)
                train = AugmentedTrain(splits['train'])
                metadata = read(output / 'prepared_dataset.json')
                metadata.update(augmentation_factor=dict(train=8, val=1, test=1),
                    rating_range=manifest['rating_range'], train_positions_per_epoch=train.positions)
                write(output / 'prepared_dataset.json', metadata)
                model = fresh_model(args)
                initial = dict(pretrained_weights_loaded=False, method='paper_policy_models.build_paper_model',
                    weights_sha256=hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest(),
                    optimizer_iterations=int(model.optimizer.iterations.numpy()), parameters=model.count_params(),
                    seed=args.seed, architecture='conv8_bn', precision=args.precision)
                if (output / 'initialization.json').exists() and read(output / 'initialization.json') != initial:
                    raise ValueError('Random initialization is not reproducible for this resume')
                write(output / 'initialization.json', initial)
                runtime = TrainingRuntime(model, output, args.epochs,
                    math.ceil(train.positions / args.batch_size), args.progress_interval_sec, True)
                best_dir = output / 'best_epochs'
                best_dir.mkdir(exist_ok=True)
                class SaveBest(tf.keras.callbacks.Callback):
                    def __init__(self):
                        super().__init__()
                        self.best = min((r['val_loss'] for r in runtime.records), default=float('inf'))
                    def on_epoch_end(self, epoch, logs=None):
                        if not all(math.isfinite(float(logs[k])) for k in ('loss', 'val_loss')):
                            raise ValueError('Nonfinite loss')
                        if float(logs['val_loss']) < self.best:
                            self.best = float(logs['val_loss'])
                            destination = best_dir / f'epoch_{epoch + 1:03d}.h5'
                            temporary = destination.with_name(destination.stem + '.tmp.h5')
                            self.model.save(temporary, include_optimizer=False)
                            temporary.replace(destination)
                            print(f'[selection] epoch {epoch + 1}: val_loss={self.best:.6f}', flush=True)
                status('training', initial_epoch=runtime.initial_epoch)
                print(f'Fresh Conv8+BN, Adam lr={args.learning_rate}, {train.positions:,} positions/epoch', flush=True)
                if runtime.initial_epoch < args.epochs:
                    model.fit(train.as_dataset(args.batch_size, shuffle=True, seed=args.seed,
                        initial_epoch=runtime.initial_epoch), validation_data=splits['val'].as_dataset(args.batch_size),
                        epochs=args.epochs, initial_epoch=runtime.initial_epoch, callbacks=[SaveBest(), runtime], verbose=2)
                if len(runtime.records) != args.epochs:
                    raise ValueError('Not all requested epochs completed')
                best = min(runtime.records, key=lambda r: (r['val_loss'], r['epoch']))
                temporary = output / 'selected_model.tmp.h5'
                shutil.copyfile(best_dir / f"epoch_{best['epoch']:03d}.h5", temporary)
                temporary.replace(output / 'selected_model.h5')
                saved = tf.keras.models.load_model(output / 'selected_model.h5', compile=False)
                probe = next(iter(splits['val'].as_dataset(min(32, args.batch_size))))[0]
                prediction = saved(probe, training=False).numpy()
                if not np.isfinite(prediction).all() or not np.allclose(prediction.sum(axis=1), 1, atol=1e-5):
                    raise ValueError('Saved policy failed probability checks')
                if file_sha256(directory / 'manifest.json') != identity['source_manifest_sha256']:
                    raise ValueError('Quest manifest changed')
                summary = dict(name='quest_scratch_' + args.rating_band, initialization=initial,
                    spec=dict(architecture='conv8_bn', learning_rate=args.learning_rate, l2=0.0),
                    epochs_ran=len(runtime.records), adopted_epoch=best['epoch'], validation_loss=best['val_loss'],
                    adoption_rule=identity['selection'], test_used_for_selection=False,
                    model_sha256=file_sha256(output / 'selected_model.h5'),
                    train_positions_per_epoch=train.positions, rating_range=manifest['rating_range'],
                    elapsed_sec=time.time() - started)
                write(output / 'selected_summary.json', summary)
                status('complete', returncode=0, selected_epoch=best['epoch'], model_sha256=summary['model_sha256'])
                print(f"COMPLETE: best epoch {best['epoch']}; {output / 'selected_model.h5'}", flush=True)
            except BaseException as error:
                status('failed', error=repr(error))
                raise
            finally:
                for split in splits.values():
                    split.close()
                sys.stdout, sys.stderr = previous


if __name__ == '__main__':
    run(parser().parse_args())
