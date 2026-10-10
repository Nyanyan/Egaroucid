import unittest

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



if __name__ == '__main__':
    unittest.main()
