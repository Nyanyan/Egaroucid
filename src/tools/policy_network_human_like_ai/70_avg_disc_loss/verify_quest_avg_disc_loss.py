"""Independently recount saved console rows and verify the published averages."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib


def verify(output):
    report = json.loads((output / 'quest_avg_disc_loss.json').read_text(encoding='utf-8'))
    assert report['status'] == 'complete' and 1 <= report['engine']['level'] <= 60
    source = Path(report['source'])
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    histogram = defaultdict(Counter)
    configuration = hashlib.sha256(json.dumps(report['engine'], sort_keys=True).encode()).hexdigest()
    db = sqlite3.connect((output / 'cache' / 'analyze.sqlite3').as_uri() + '?mode=ro', uri=True)
    games = 0
    try:
        with (source / 'transcript_split' / 'all' / 'games.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                game = json.loads(line)
                if game['split'] != 'test':
                    continue
                transcript = game['transcript']
                key = hashlib.sha256(transcript.encode('ascii')).hexdigest()
                saved = db.execute('SELECT raw_zlib FROM analyses WHERE configuration=? AND key=?',
                                   (configuration, key)).fetchone()
                assert saved is not None
                rows = 0
                for line in zlib.decompress(saved[0]).decode('utf-8').splitlines():
                    cells = [cell.strip() for cell in line.split('|')]
                    if len(cells) != 11 or not cells[1].isdigit():
                        continue
                    side = cells[2].lower()
                    rating = game[side + '_rating']
                    band = 'under_800' if rating < 800 else f'{rating // 200 * 200}-{rating // 200 * 200 + 200}'
                    played = int(cells[5])
                    alternative = played if cells[6].lower() == 'none' else int(cells[8])
                    histogram[band][max(0, alternative - played)] += 1
                    rows += 1
                assert rows == len(transcript) // 2
                games += 1
    finally:
        db.close()
    assert games == report['games'] == manifest['splits']['test']['games']
    assert sum(sum(h.values()) for h in histogram.values()) == report['positions']
    for row in report['detailed']:
        band = row['rating_band']
        hist = histogram[band]
        count = sum(hist.values())
        loss = sum(value * n for value, n in hist.items())
        assert count == manifest['bands'][band]['test']['positions'] == row['positions']
        assert loss == row['total_disc_loss'] and hist[0] == row['zero_loss_positions']
        assert loss / count == row['mean_disc_loss']
        assert {str(k): v for k, v in hist.items()} == report['histograms'][band]
    merged = defaultdict(Counter)
    for band, hist in histogram.items():
        merged['2200+' if band != 'under_800' and int(band.split('-')[0]) >= 2200 else band].update(hist)
    for row in report['nine_bands']:
        hist = merged[row['rating_band']]
        count = sum(hist.values())
        loss = sum(value * n for value, n in hist.items())
        assert count == row['positions'] and loss == row['total_disc_loss']
        assert hist[0] == row['zero_loss_positions'] and loss / count == row['mean_disc_loss']
    for name, expected in [('detailed', report['detailed']), ('9bands', report['nine_bands'])]:
        with (output / f'quest_avg_disc_loss_{name}.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(expected)
        for row, saved in zip(rows, expected):
            assert row['rating_band'] == saved['rating_band']
            for field in ('positions', 'total_disc_loss', 'zero_loss_positions'):
                assert int(row[field]) == saved[field]
            assert float(row['mean_disc_loss']) == saved['mean_disc_loss']
    result = dict(status='passed', games=games, positions=report['positions'],
                  verification='Independent raw console score recount, per-band manifest counts, histograms and CSV averages')
    (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'output' / 'quest_test_l10')
    verify(parser.parse_args().output.resolve())
