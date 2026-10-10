import argparse
import csv
import queue
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import ggs_analyze as ggs


class StoneLossTests(unittest.TestCase):
    def test_progress_callback_during_silent_search(self):
        engine = ggs.Engine.__new__(ggs.Engine)
        engine.output = queue.Queue()
        calls = []

        def on_wait():
            calls.append(True)
            engine.output.put('> result\n')

        with patch.object(ggs, 'ENGINE_WAIT_SECONDS', 0.001):
            self.assertEqual(engine.readline(on_wait), 'result')
        self.assertEqual(calls, [True])

    def test_repeated_analysis_retains_engine_and_does_not_clear_table(self):
        engine = ggs.Engine.__new__(ggs.Engine)
        sent = []
        engine.send = sent.append
        engine.progress = lambda: '解析完了 1/2局（0/1組）'
        output = iter([
            '| 60| Black| h8| 60@100%| -4| -| -| -| |',
            'Mistake Rate', '| White |',
            '| 60| Black| h8| 60@100%| -6| a1| 60@100%| -4| |',
            'Mistake Rate', '| White |',
        ])
        engine.readline = lambda on_wait: next(output)
        first = engine.analyze('first X', 'h8', lambda row: None)
        second = engine.analyze('second X', 'h8', lambda row: None)
        self.assertEqual(sent, ['setboard first X', 'play h8', 'analyzefast',
                                'setboard second X', 'play h8', 'analyzefast'])
        self.assertIsNone(first[0].alt_move)
        self.assertEqual(second[0].alt_score - second[0].played_score, 2)

    def test_wait_message_includes_completed_games(self):
        engine = ggs.Engine.__new__(ggs.Engine)
        engine.send = lambda line: None
        engine.progress = lambda: '解析完了 5/18局（2/9組） | ETA: 残り約1時間20分（完了見込み 10/04 00:30）'
        output = iter(['Mistake Rate', '| White |'])

        def readline(on_wait):
            on_wait()
            return next(output)

        engine.readline = readline
        with patch.object(ggs, 'log') as logged:
            engine.analyze(ggs.START_BOARD, 'f5', lambda row: None, '[6/18 .44.1]')
        messages = [call.args[0] for call in logged.call_args_list]
        self.assertEqual(len(messages), 1)
        self.assertIn('解析完了 5/18局（2/9組）', messages[0])
        self.assertIn('ETA: 残り約1時間20分', messages[0])

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

    def test_explicit_client_log_restricts_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chosen, other = root / 'chosen_ggs.log', root / 'other_ggs.log'
            chosen.touch()
            other.touch()
            self.assertEqual(ggs.find_client_logs([root], [chosen], recursive=True), [chosen.resolve()])

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


class ScheduleTests(unittest.TestCase):
    def parsed(self, *options):
        with patch.object(sys, 'argv', ['ggs_analyze.py', 'fixture.txt', *options]):
            return ggs.parse_args()

    def test_all_three_profiles_cover_every_empty_count_as_requested(self):
        expected = {
            'original': [(1, 30, 'exact', 60, '100'), (31, 34, 'end', 60, '99'),
                         (35, 37, 'end', 60, '98'), (38, 40, 'end', 60, '93'),
                         (41, 43, 'end', 60, '88'), (44, 60, 'mid', 35, '74')],
            'exact32': [(1, 32, 'exact', 60, '100'), (33, 34, 'end', 60, '99'),
                        (35, 36, 'end', 60, '98'), (37, 38, 'end', 60, '93'),
                        (39, 40, 'end', 60, '88'), (41, 46, 'end', 60, '74'),
                        (47, 60, 'mid', 35, '88')],
            'exact36': [(1, 36, 'exact', 60, '100'), (37, 39, 'end', 60, '99'),
                        (40, 43, 'end', 60, '98'), (44, 47, 'end', 60, '93'),
                        (48, 51, 'end', 60, '88'), (52, 55, 'end', 60, '74'),
                        (56, 60, 'mid', 39, '88')],
        }
        for name, ranges in expected.items():
            with self.subTest(profile=name):
                args = self.parsed('--profile', name)
                self.assertFalse(args.profile_modified)
                schedule = ggs.schedule_from_args(args)
                for empties in range(1, 61):
                    phases = [phase for phase in schedule.phases if phase.empties_lo <= empties <= phase.empties_hi]
                    self.assertEqual(len(phases), 1, (name, empties))
                    phase = phases[0]
                    setting = next((kind, depth, prob) for lo, hi, kind, depth, prob in ranges if lo <= empties <= hi)
                    self.assertEqual((phase.kind, phase.depth, phase.prob), setting, (name, empties))

    def test_default_profile_is_exact36_and_console_ranges_are_correct(self):
        args = self.parsed()
        self.assertEqual(args.profile, 'exact36')
        schedule = ggs.schedule_from_args(args)
        self.assertEqual(schedule.dpr_args(), [
            '-dpr', '1', '5', '39', '88', '-dpr', '6', '9', '60', '74',
            '-dpr', '10', '13', '60', '88', '-dpr', '14', '17', '60', '93',
            '-dpr', '18', '21', '60', '98', '-dpr', '22', '24', '60', '99',
            '-dpr', '25', '60', '60', '100'])

    def test_profiles_preserve_existing_cache_keys_and_are_isolated(self):
        legacy = {'original': ggs.Schedule(35, '74', 43, 30, ['88', '93', '98', '99']),
                  'exact32': ggs.Schedule(35, '88', 46, 32, ['74', '74', '74', '88', '93', '98', '99'])}
        game = StoneLossTests().game('0', 'X', 0)
        for name, previous in legacy.items():
            named = ggs.schedule_from_args(self.parsed('--profile', name))
            self.assertEqual(ggs.schedule_cache_key(previous, False), ggs.schedule_cache_key(named, False))
            self.assertEqual(ggs.game_cache_key(game, previous, False), ggs.game_cache_key(game, named, False))
        keys = {ggs.schedule_cache_key(ggs.schedule_from_args(self.parsed('--profile', name)), False)
                for name in ggs.SEARCH_PROFILES}
        self.assertEqual(len(keys), 3)

    def test_midgame_override_keeps_profile_endgame_ranges(self):
        args = self.parsed('--profile', 'exact36', '--mid-depth', '21')
        self.assertTrue(args.profile_modified)
        actual = ggs.schedule_from_args(args)
        self.assertEqual(actual.phases[0].depth, 21)
        self.assertEqual(actual.phases[1:], ggs.SEARCH_PROFILES['exact36'].schedule().phases[1:])

    def test_old_manual_options_still_reproduce_exact32(self):
        args = self.parsed('--mid-depth', '35', '--mid-prob', '88', '--end-empties', '46',
                           '--exact-empties', '32', '--end-probs', '74,74,74,88,93,98,99')
        self.assertEqual(ggs.schedule_from_args(args).dpr_args(), ggs.SEARCH_PROFILES['exact32'].schedule().dpr_args())

    def test_invalid_range_widths_are_rejected(self):
        with patch.object(sys, 'stderr'), self.assertRaises(SystemExit) as error:
            self.parsed('--profile', 'exact36', '--end-sizes', '4,4,4,4,4')
        self.assertEqual(error.exception.code, 2)


class ETATests(unittest.TestCase):
    def setUp(self):
        self.schedule = ggs.Schedule(35, '88', 46, 32, ['74', '74', '74', '88', '93', '98', '99'])
        self.game = StoneLossTests().game('0', 'X', 0)
        self.known = ggs.PositionTable()

    def position(self, empties, move_offset=0):
        return ggs.Position(0, 'X' * (64 - empties) + '-' * empties, 'O', 64 - empties + move_offset)

    def remember(self, position, seconds):
        analysis = dict(played_score=0, played_depth='60@100%', alt_move=None, alt_score=None, alt_depth='')
        self.known.add(position.board_str, ggs.index_to_coord(position.move), analysis, seconds, 'previous run')

    def estimator(self, positions, jobs=1):
        eta = ggs.AnalysisETA(self.schedule, jobs, self.known.entries())
        eta.add_game(self.game, positions, self.known)
        return eta

    def test_resume_excludes_cached_and_shared_positions_and_counts_running_time(self):
        cached = self.position(20)
        pending = self.position(20, 1)
        self.remember(cached, 300)
        eta = self.estimator([cached, pending])
        eta.add_game(replace(self.game, my_color='O'), [cached, pending], self.known)
        self.assertEqual(eta.remaining_seconds(), 300)
        with patch.object(ggs.time, 'perf_counter', return_value=10):
            eta.start(pending)
        with patch.object(ggs.time, 'perf_counter', return_value=40):
            self.assertEqual(eta.remaining_seconds(), 270)
        eta.finish(pending, 60)
        self.assertEqual(eta.remaining_seconds(), 0)
        self.assertIn('残り0分（解析完了）', eta.text())

    def test_current_timings_replace_history_and_update_completion_date(self):
        self.remember(self.position(20), 300)
        first, second = self.position(20, 1), self.position(20, 2)
        eta = self.estimator([first, second])
        self.assertEqual(eta.remaining_seconds(), 600)
        eta.finish(first, 120)
        self.assertEqual(eta.remaining_seconds(), 120)
        with patch.object(ggs, 'datetime') as clock:
            clock.now.return_value = datetime(2026, 10, 3, 23, 59)
            text = eta.text()
        self.assertIn('残り約2分', text)
        self.assertIn('完了見込み 10/04 00:01', text)

    def test_no_estimate_uses_times_from_a_different_search_phase(self):
        self.remember(self.position(20), 1)
        first, second = self.position(50), self.position(49)
        eta = self.estimator([first, second])
        self.assertIsNone(eta.remaining_seconds())
        self.assertIn('算出待ち', eta.text())
        eta.finish(first, 60)
        self.assertEqual(eta.remaining_seconds(), 60)
        eta.failed = True
        self.assertIn('算出不可（解析失敗あり）', eta.text())

    def test_zero_rounded_timings_do_not_claim_unfinished_work_is_complete(self):
        self.remember(self.position(20), 0)
        eta = self.estimator([self.position(20, 1)])
        self.assertEqual(eta.remaining_seconds(), 0)
        self.assertIn('残り1分未満', eta.text())
        self.assertNotIn('解析完了', eta.text())

    def test_parallelism_is_applied_to_whole_matches(self):
        self.remember(self.position(20), 300)
        eta = self.estimator([self.position(20, 1), self.position(20, 2)], jobs=2)
        self.assertEqual(eta.remaining_seconds(), 600)  # One match still uses one process.
        eta.add_game(replace(self.game, match_no=2), [self.position(20, 3)], self.known)
        self.assertEqual(eta.remaining_seconds(), 600)
        eta.add_game(replace(self.game, match_no=3), [self.position(20, 4)], self.known)
        self.assertEqual(eta.remaining_seconds(), 600)


if __name__ == '__main__':
    unittest.main()
