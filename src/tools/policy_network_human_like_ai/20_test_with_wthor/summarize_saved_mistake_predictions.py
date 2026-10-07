"""Add loss thresholds using saved predictions; never run an engine or network."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_mistake_test_models import MODEL_NAMES, THRESHOLDS, atomic_json, verify_subsets
from evaluate_wthor_human_match import calc_legal_batch, equivalent_policy_mask
from wthor_human_match_evaluation import wilson_interval
from wthor_human_match_experiment import file_sha256, write_csv
from prepare_mistake_test_data import read_test_records
from extend_mistake_test_subsets import export_from_union


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(run_dir, thresholds=(1, 2, 4, 6, 10, 20)):
    thresholds = tuple(sorted(set(thresholds)))
    if not thresholds or min(thresholds) < 1 or max(thresholds) > 128:
        raise ValueError('Thresholds must be integers between 1 and 128')
    base = run_dir / 'evaluation'
    output = base / 'extended_thresholds'
    summary_path = base / 'mistake_test_summary.json'
    predictions_path = base / 'predictions.npz'
    original, verified = read(summary_path), read(base / 'verification.json')
    assert read(run_dir / 'run_status.json')['state'] == 'complete'
    assert verified['status'] == 'passed'
    assert verified['summary_sha256'] == file_sha256(summary_path)
    assert verified['predictions_sha256'] == file_sha256(predictions_path)
    counts, records, losses, indices, union_dir, manifests = verify_subsets(run_dir / 'datasets')
    assert original['identity']['dataset_manifests'] == manifests
    assert original['identity']['dataset_counts_sha256'] == file_sha256(run_dir / 'datasets/dataset_counts.json')
    _, game_ids = read_test_records(union_dir)
    source_losses = np.fromfile(run_dir / 'datasets/source_test.losses', dtype='u1')
    with np.load(predictions_path, allow_pickle=False) as saved:
        assert tuple(saved['model_names']) == MODEL_NAMES
        assert np.array_equal(saved['source_indices'], indices)
        assert np.array_equal(saved['losses'], losses)
        assert np.array_equal(saved['human_policies'], records['policy'])
        predicted = saved['predicted_policies'].copy()
        targets = saved['symmetry_target_masks'].copy()
    assert predicted.shape == (len(records), 27) and np.all((predicted >= 0) & (predicted < 64))
    # Recompute board symmetry and legality independently of the saved target masks.
    target_bool = equivalent_policy_mask(records['player'], records['opponent'], records['policy'].astype(np.int64))
    bits = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
    assert np.array_equal((target_bool.astype(np.uint64) * bits).sum(axis=1, dtype=np.uint64), targets)
    legal_masks = calc_legal_batch(records['player'], records['opponent'])
    move_masks = np.left_shift(np.uint64(1), predicted.astype(np.uint64))
    assert np.all((move_masks & legal_masks[:, None]) != 0)
    legal_count = ((legal_masks[:, None] & bits) != 0).sum(axis=1)
    expected_random = target_bool.sum(axis=1) / legal_count
    matched = (move_masks & targets[:, None]) != 0
    exact = predicted == records['policy'][:, None]
    old_rows = {(r['model'], r['minimum_loss']): r for r in original['results']}
    rows = []
    for col, model in enumerate(MODEL_NAMES):
        for t in thresholds:
            selected = losses >= t
            assert np.array_equal(indices[selected], np.flatnonzero(source_losses >= t))
            n = int(selected.sum())
            if not n:
                raise ValueError(f'No positions for threshold {t}')
            hits, strict = int(matched[selected, col].sum()), int(exact[selected, col].sum())
            low, high = wilson_interval(hits, n)
            row = dict(model=model, minimum_loss=t, requested_positions=n, positions=n,
                       top1_hits=hits, top1_accuracy=hits/n, top1_accuracy_percent=hits/n*100,
                       top1_ci95_lower=low, top1_ci95_upper=high, exact_top1_hits=strict,
                       exact_top1_accuracy=strict/n,
                       expected_random_accuracy=float(expected_random[selected].mean()) if model == 'random' else None,
                       status='complete')
            if (model, t) in old_rows:
                old = old_rows[model, t]
                for key, value in row.items():
                    if isinstance(value, float):
                        assert abs(old[key] - value) < 1e-12, (model, t, key)
                    else:
                        assert old[key] == value, (model, t, key)
                row = old.copy()  # Preserve all previously published numeric values exactly.
            else:
                # Independent scalar recount for every new model/threshold result.
                selected_rows = np.flatnonzero(selected)
                manual_hits = sum(bool(int(targets[i]) & (1 << int(predicted[i, col]))) for i in selected_rows)
                manual_exact = sum(int(predicted[i, col]) == int(records['policy'][i]) for i in selected_rows)
                assert (manual_hits, manual_exact) == (hits, strict)
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    extra = [t for t in thresholds if t not in THRESHOLDS]
    new_counts = export_from_union(output / 'datasets', records, game_ids, losses, indices, counts, extra)
    count_index = {r['minimum_loss']: r for r in counts['datasets'] + new_counts}
    dataset_counts = [count_index[t] for t in thresholds]
    identity = {**original['identity'], 'thresholds': list(thresholds),
                'aggregation': 'All matching records from verified predictions.npz; no new engine/network calls',
                'original_summary_sha256': verified['summary_sha256'],
                'additional_dataset_manifests': {r['dataset']: file_sha256(output / 'datasets' / r['dataset'] / 'manifest.json') for r in new_counts}}
    extended = {**{k: original[k] for k in ['version','agreement_definition','exact_agreement_definition','random_definition','confidence_interval','unique_states','evaluated_union_positions','predictions_sha256']},
                'identity': identity, 'dataset_counts': dataset_counts, 'results': rows}
    atomic_json(output / 'mistake_test_summary.json', extended)
    write_csv(output / 'mistake_test_summary.csv', rows, list(rows[0]))
    write_csv(output / 'dataset_counts.csv', dataset_counts, list(dataset_counts[0]))
    verification = dict(status='passed', model_count=27, threshold_count=len(thresholds), result_rows=len(rows),
                        source_positions=counts['source_positions'], union_positions=len(records),
                        thresholds=list(thresholds), original_results_unchanged=sum((r['model'], r['minimum_loss']) in old_rows for r in rows),
                        checks=['verified original summary and predictions hashes',
                                'exact nested membership against all original source losses',
                                'recomputed symmetry masks and legal predictions',
                                'original 108 results preserved exactly',
                                'additional hit counts independently recounted using scalar bit tests',
                                'new dataset records, game IDs, losses and source indices read back and compared'],
                        summary_sha256=file_sha256(output / 'mistake_test_summary.json'),
                        predictions_sha256=verified['predictions_sha256'])
    atomic_json(output / 'verification.json', verification)
    print(json.dumps({'status':'passed', 'rows':len(rows), 'datasets':dataset_counts}, ensure_ascii=True), flush=True)
    return extended


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).parent / 'output/mistake_test_20261005')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[1,2,4,6,10,20])
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.thresholds)
