"""Paths and validation for policies trained only on Quest from random weights."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
for directory in (BASE, BASE / '00_data_processing', BASE / '10_train_policy_network',
                  BASE / '11_fine_tuning', BASE / '20_test_with_wthor'):
    sys.path.insert(0, str(directory))

from fine_tune_policy import atomic_json as write, read_json as read, run_lock, AugmentedTrain, Tee
from wthor_dataset import file_sha256

DEFAULT_RUN = HERE / 'output/quest_under800_20261010'
DEFAULT_RESULTS = HERE / 'output/quest_under800_20261010_analysis'
MODEL_ID = 'quest_scratch_under800'


def validate_model(model):
    model = Path(model).resolve()
    root = model.parent
    identity = read(root / 'training_identity.json')
    selected = read(root / 'selected_summary.json')
    status = read(root / 'run_status.json')
    initial = read(root / 'initialization.json')
    if (identity['initialization'] != 'random_from_architecture'
            or identity['pretrained_weights_loaded'] is not False
            or initial['pretrained_weights_loaded'] is not False
            or initial['optimizer_iterations'] != 0
            or selected['test_used_for_selection'] is not False
            or selected['epochs_ran'] != identity['settings']['epochs']
            or status['state'] != 'complete' or status['returncode'] != 0):
        raise ValueError('Expected a completed, randomly initialized Quest-only training run')
    if identity['rating_range'] != dict(min_inclusive=None, max_exclusive=800):
        raise ValueError('This comparison requires Quest ratings below 800')
    if identity['settings']['max_train_samples'] or identity['settings']['max_val_samples']:
        raise ValueError('Smoke-test models cannot enter the full evaluation')
    if file_sha256(model) != selected['model_sha256'] or status['model_sha256'] != selected['model_sha256']:
        raise ValueError('Selected model differs from the completed training run')
    source = Path(identity['source_directory'])
    if file_sha256(source / 'manifest.json') != identity['source_manifest_sha256']:
        raise ValueError('Training data manifest changed')
    return {name: dict(path=str(root / name), sha256=file_sha256(root / name)) for name in
            ('selected_model.h5', 'selected_summary.json', 'training_identity.json',
             'prepared_dataset.json', 'initialization.json', 'run_status.json')}
