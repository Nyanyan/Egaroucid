import argparse
import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import ggs_analyze as ggs


class StoneLossTests(unittest.TestCase):
    def test_repeated_analysis_retains_engine_and_does_not_clear_table(self):
        engine = ggs.Engine.__new__(ggs.Engine)
        sent = []
        engine.send = sent.append
        output = iter([
            '| 60| Black| h8| 60@100%| -4| -| -| -| |',
            'Mistake Rate', '| White |',
            '| 60| Black| h8| 60@100%| -6| a1| 60@100%| -4| |',
            'Mistake Rate', '| White |',
        ])
        engine.readline = lambda: next(output)
        first = engine.analyze('first X', 'h8', lambda row: None)
        second = engine.analyze('second X', 'h8', lambda row: None)
        self.assertEqual(sent, ['setboard first X', 'play h8', 'analyzefast',
                                'setboard second X', 'play h8', 'analyzefast'])
        self.assertIsNone(first[0].alt_move)
        self.assertEqual(second[0].alt_score - second[0].played_score, 2)

    def game(self, suffix, color, score):
        record = ggs.GameRecord(Path('fixture.txt'), '.1.' + suffix,
                               'egrcd' if color == 'X' else 'opponent',
                               'opponent' if color == 'X' else 'egrcd',
                               ggs.START_BOARD[:64], 'X', ['f5'],
                               score if color == 'X' else -score, datetime(2026, 10, 3))
        return ggs.SelectedGame(record, color, 'opponent', score, '負け', 1, '.1', -2, 2)

    def move(self, game, played_score, alt_score, depth='35@74%', alt_depth=None):
        position = ggs.Position(0, ggs.START_BOARD[:64], game.my_color, ggs.coord_to_index('f5'))
        analysis = dict(played_score=played_score, played_depth=depth,
                        alt_move='c4' if alt_score is not None else None,
                        alt_score=alt_score, alt_depth=(alt_depth or depth) if alt_score is not None else '')
        return ggs.move_result(position, game, 'f5', analysis, 1, None)

    def test_stone_loss_does_not_require_result_change(self):
        game = self.game('0', 'X', 8)
        moves = [self.move(game, 6, 8), self.move(game, -12, -10),
                 self.move(game, 5, 6), self.move(game, 6, 6), self.move(game, 8, 6)]
        analysis = ggs.GameAnalysis(1, game, moves, game.record.initial_cells, True, 5)
        self.assertEqual([m['loss'] for m in analysis.loss_moves()], [2, 2, 1])
        self.assertEqual(len(analysis.loss_moves(False)), 0)

    def test_no_alternative_does_not_create_loss(self):
        self.assertEqual(self.move(self.game('0', 'X', 0), -8, None)['loss'], 0)

    def test_partial_pair_is_not_aggregated(self):
        game = self.game('0', 'X', 8)
        move = self.move(game, 6, 8)
        analysis = ggs.GameAnalysis(1, game, [move], game.record.initial_cells, True, 1)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            ggs.write_csvs(out, [analysis], 4)
            with (out / 'matches.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['analysis_complete'], '0')
            self.assertEqual(rows[0]['my_loss_moves'], '')
            self.assertEqual(rows[0]['my_loss_total'], '')

    def test_both_values_must_be_exact(self):
        move = self.move(self.game('0', 'X', 0), -2, 0, '60@100%')
        self.assertEqual(ggs.loss_certainty(move), '完全読みで確認')
        move['alt_depth'] = '60@99%'
        self.assertEqual(ggs.loss_certainty(move), '近似解析による推定')
        move['alt_depth'] = '59@100%'
        self.assertEqual(ggs.loss_certainty(move), '近似解析による推定')

    def test_report_and_csv_group_both_games_and_keep_small_losses(self):
        analyses = []
        for no, game in enumerate([self.game('0', 'X', 8), self.game('1', 'O', -10)], 1):
            move = self.move(game, 6 if no == 1 else -12, 8 if no == 1 else -10)
            analyses.append(ggs.GameAnalysis(no, game, [move], game.record.initial_cells, True, 1))
        args = argparse.Namespace(me=['egrcd'], unit='match')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            ggs.write_report(out / 'report.md', analyses, [], [], args,
                             ggs.Schedule(35, '74', 43, 30, ['88', '93', '98', '99']), [], ggs.SelectionStats(), 0)
            ggs.write_csvs(out, analyses, 4)
            text = (out / 'report.md').read_text(encoding='utf-8')
            self.assertNotIn('敗着候補', text)
            self.assertNotIn('区分', text)
            self.assertIn('石損があったのは2手', text)
            table_width = None
            for line in text.splitlines():
                if line.startswith('|'):
                    width = len(line.split('|'))
                    if table_width is not None:
                        self.assertEqual(width, table_width, line)
                    table_width = width
                else:
                    table_width = None
            with (out / 'matches.csv').open(encoding='utf-8-sig') as f:
                matches = list(csv.DictReader(f))
            with (out / 'moves.csv').open(encoding='utf-8-sig') as f:
                moves = list(csv.DictReader(f))
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]['my_loss_moves'], '2')
            self.assertEqual(matches[0]['my_loss_total'], '4')
            self.assertEqual([m['loss'] for m in moves], ['2', '2'])
            self.assertNotIn('result_worsened', moves[0])
            self.assertFalse((out / 'games.csv').exists())


if __name__ == '__main__':
    unittest.main()
