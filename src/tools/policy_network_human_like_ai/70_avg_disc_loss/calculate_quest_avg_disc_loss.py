"""Average disc loss of every unaugmented Quest test move at a selected level."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import datetime, timezone
import json
import os
import queue
import subprocess
import threading
from functools import partial
from pathlib import Path
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '00_data_processing'))
from analysis_pipeline import add_analysis_arguments, analyze_missing, Progress, validate_positive
from board_replay import replay_games
from miss_analysis import AnalysisCache, engine_identity, parse_analysis
from miss_analysis import AnalyzeEngine
import analysis_pipeline
from pipeline_common import iter_transcripts, load_manifest
from prepare_othello_quest_data import rating_band
from wthor_dataset import BOARD_DTYPE, file_sha256
sys.path.insert(0, str(HERE.parent / '40_test_strength'))
from strength_engine import _create_windows_kill_job, _close_windows_job


class QuestAnalyzeEngine(AnalyzeEngine):
    """Selected level; warm table, one search thread, kill on parent exit."""

    def __init__(self, exe, hash_level, timeout, cancel, *, level):
        self.timeout, self.cancel = timeout, cancel
        self.reader, self._job = None, None
        self.proc = subprocess.Popen(
            [str(exe), '-l', str(level), '-t', '1', '-hash', str(hash_level),
             '-quiet', '-nobook', '-noboard', '-m', '3'],
            cwd=exe.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            self._job = _create_windows_kill_job(self.proc)
            if os.name == 'nt' and self._job is None:
                raise RuntimeError('Could not attach engine to a kill-on-close job')
            self.output = queue.Queue()
            self.reader = threading.Thread(target=self._pump, daemon=True)
            self.reader.start()
            self._read_prompt()
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            if self.reader is not None:
                self.reader.join(timeout=2)
            self.proc.stdin.close()
            self.proc.stdout.close()
        finally:
            _close_windows_job(self._job)
            self._job = None


def atomic_json(path, data):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def summary_rows(histograms):
    def order(band):
        return -1 if band == 'under_800' else int(band.rstrip('+').split('-')[0])
    rows = []
    for band in sorted(histograms, key=order):
        histogram = histograms[band]
        count = sum(histogram.values())
        loss = sum(value * n for value, n in histogram.items())
        rows.append(dict(rating_band=band, positions=count, total_disc_loss=loss,
                         mean_disc_loss=loss / count if count else None,
                         zero_loss_positions=histogram[0]))
    return rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    validate_positive(args, 'workers', 'games_per_batch', 'timeout_sec', 'progress_seconds')
    if not 0 <= args.hash_level <= 29:
        raise ValueError('hash level must be 0..29')
    source = args.quest_root.resolve()
    transcripts = source / 'transcript_split' / 'all'
    manifest = load_manifest(transcripts, 'transcript_split')
    root_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    games = list(iter_transcripts(transcripts / 'test'))
    metadata = {}
    with (transcripts / 'games.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] == 'test':
                if row['game_id'] in metadata:
                    raise ValueError('duplicate test game ID')
                metadata[row['game_id']] = row
    if len(games) != manifest['splits']['test']['games'] or set(metadata) != {g[0] for g in games}:
        raise ValueError('test game membership differs from manifest')
    unique, expected_boards, expected_ids = {}, defaultdict(list), defaultdict(list)
    progress = Progress('Verify test boards and actual player ratings', len(games), args.progress_seconds)
    for start in range(0, len(games), args.games_per_batch):
        batch = games[start:start + args.games_per_batch]
        boards, ids, errors = replay_games([(g[0], g[1]) for g in batch], allow_unfinished=True)
        if errors:
            raise ValueError(f'Invalid test games: {errors[:3]}')
        offset = 0
        for game_id, transcript, _, _ in batch:
            row = metadata[game_id]
            if transcript != row['transcript']:
                raise ValueError('test transcript differs from rating metadata')
            count = len(transcript) // 2
            colors = boards['color'][offset:offset + count].tobytes()
            unique.setdefault(transcript, colors)
            offset += count
        labels = np.asarray([rating_band(metadata[int(gid)]['black_rating' if color == 0 else 'white_rating'])
                             for gid, color in zip(ids, boards['color'])])
        for band in np.unique(labels):
            mask = labels == band
            expected_boards[str(band)].append(boards[mask])
            expected_ids[str(band)].append(ids[mask])
        progress.show(min(start + len(batch), len(games)))
    for band, arrays in expected_boards.items():
        directory = source / 'board_split' / band
        board_manifest = load_manifest(directory, 'board_split')
        paths = sorted((directory / 'test').glob('*.dat'))
        actual = np.concatenate([np.fromfile(p, dtype=BOARD_DTYPE) for p in paths])
        actual_ids = np.concatenate([np.fromfile(p.with_suffix('.game_ids'), dtype='<u8') for p in paths])
        expected = np.concatenate(arrays)
        if (not np.array_equal(actual, expected) or not np.array_equal(actual_ids, np.concatenate(expected_ids[band]))
                or len(actual) != board_manifest['splits']['test']['positions']):
            raise ValueError(f'Test boards/labels/IDs differ in {band}')
    if sum(len(a) for arrays in expected_boards.values() for a in arrays) != root_manifest['splits']['test']['positions']:
        raise ValueError('test position count mismatch')
    del expected_boards, expected_ids
    progress.show(len(games), 'All test records and game IDs match existing unaugmented boards', force=True)
    args.egaroucid_exe = args.egaroucid_exe.resolve()
    identity = {**engine_identity(args.egaroucid_exe, args.hash_level), 'level': args.level}
    analysis_pipeline.AnalyzeEngine = partial(QuestAnalyzeEngine, level=args.level)
    print(f'Egaroucid level {args.level}: {args.workers} processes x 1 thread; timeout {args.timeout_sec}s/game', flush=True)
    with closing(AnalysisCache(args.output / 'cache', identity)) as cache:
        stats = analyze_missing(unique, cache, args, game_counts=Counter(g[1] for g in games))
        histograms = defaultdict(Counter)
        game_rows = []
        progress = Progress('Aggregate disc loss', len(games), args.progress_seconds)
        for i, (game_id, transcript, _, _) in enumerate(games, 1):
            rows = parse_analysis(cache.get(transcript), transcript, unique[transcript])
            per_game = defaultdict(Counter)
            for move in rows:
                color = 'black' if move['color'] == 0 else 'white'
                band = rating_band(metadata[game_id][color + '_rating'])
                histograms[band][move['loss']] += 1
                per_game[band][move['loss']] += 1
            game_rows.extend(dict(game_id=game_id, **row) for row in summary_rows(per_game))
            progress.show(i)
        detailed = summary_rows(histograms)
        merged = defaultdict(Counter)
        for band, histogram in histograms.items():
            merged['2200+' if band != 'under_800' and int(band.split('-')[0]) >= 2200 else band].update(histogram)
        combined = summary_rows(merged)
        all_count = sum(r['positions'] for r in detailed)
        if all_count != root_manifest['splits']['test']['positions']:
            raise ValueError('aggregated denominator differs from test boards')
        write_csv(args.output / 'quest_avg_disc_loss_detailed.csv', detailed)
        write_csv(args.output / 'quest_avg_disc_loss_9bands.csv', combined)
        write_csv(args.output / 'quest_avg_disc_loss_by_game.csv', game_rows)
        report = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                      source=str(source), source_manifest_sha256=file_sha256(source / 'manifest.json'),
                      transcript_manifest_sha256=file_sha256(transcripts / 'manifest.json'),
                      engine=identity, analysis=stats, games=len(games), positions=all_count,
                      includes_unfinished_games=True, augmentation=False,
                      loss_formula='max(played_score, alternative_score) - played_score; no alternative => 0',
                      counting_unit='every test move, including zero loss; forced passes excluded',
                      verification='every replayed board, move label, color and game ID equals existing test board files',
                      detailed=detailed, nine_bands=combined,
                      histograms={band: dict(sorted(hist.items())) for band, hist in histograms.items()})
        atomic_json(args.output / 'quest_avg_disc_loss.json', report)
    for row in combined:
        print(f"{row['rating_band']:>12}: n={row['positions']:,}; mean disc loss={row['mean_disc_loss']:.6f}", flush=True)
    print('COMPLETE:', args.output, flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--quest-root', type=Path, required=True)
    result.add_argument('--level', type=int, choices=range(1, 61), default=10)
    result.add_argument('--output', type=Path)
    add_analysis_arguments(result)
    result.set_defaults(workers=min(24, os.cpu_count() or 1))
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    args.output = (args.output or HERE / 'output' / f'quest_test_l{args.level}').resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'status.json', dict(status='running', level=args.level, pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        run(args)
    except BaseException as error:
        atomic_json(args.output / 'status.json', dict(status='failed', error=str(error)))
        traceback.print_exc()
        sys.exit(1)
    atomic_json(args.output / 'status.json', dict(status='complete'))
