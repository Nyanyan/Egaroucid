"""Export extra nested subsets from the already verified >=1-loss union."""
import numpy as np
from pipeline_common import write_manifest
from prepare_mistake_test_data import read_test_records


def export_from_union(output, records, game_ids, losses, source_indices, counts, thresholds):
    result = []
    for threshold in thresholds:
        selected = losses >= threshold
        name = f"{counts['source_dataset']}_MISS_{threshold}"
        directory = output / name
        test = directory / 'test'
        test.mkdir(parents=True, exist_ok=True)
        stem = test / '0000000'
        records[selected].tofile(stem.with_suffix('.dat'))
        game_ids[selected].astype('<u8').tofile(stem.with_suffix('.game_ids'))
        losses[selected].astype('u1').tofile(stem.with_suffix('.losses'))
        source_indices[selected].astype('<u8').tofile(stem.with_suffix('.source_indices'))
        n, games = int(selected.sum()), len(np.unique(game_ids[selected]))
        row = dict(dataset=name, minimum_loss=threshold, positions=n, games=games,
                   source_positions=counts['source_positions'], fraction=n/counts['source_positions'])
        write_manifest(directory, 'board_split', name, source=counts['source'], analysis=counts['analysis'],
                       source_split='test', augmentation=False,
                       record_format='<QQbbb: player, opponent, color, policy, score', record_bytes=19,
                       game_ids_format='uint64 little-endian; original game ID per position',
                       losses_format='uint8; cached disc loss per position',
                       source_indices_format='uint64 little-endian; zero-based index in original test',
                       selection={'minimum_loss': threshold, 'comparison': '>=',
                                  'loss': 'max(played_score, alternative_score) - played_score'},
                       splits={'test': {'positions': n, 'games': games}})
        saved_records, saved_ids = read_test_records(test)
        assert np.array_equal(saved_records, records[selected])
        assert np.array_equal(saved_ids, game_ids[selected])
        assert np.array_equal(np.fromfile(stem.with_suffix('.source_indices'), dtype='<u8'), source_indices[selected])
        assert np.array_equal(np.fromfile(stem.with_suffix('.losses'), dtype='u1'), losses[selected])
        result.append(row)
    return result
