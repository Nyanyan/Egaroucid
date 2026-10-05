import unittest
from unittest.mock import Mock, patch

import numpy as np

from policy_opponent_gtp import PolicyOpponent, geometric_model_distribution, make_argparser
from blend_policy_with_egaroucid import BoardState, coord_to_policy
from strength_resources import GameThreadBudget
from strength_tournament import MatchSetTask, PlayerSpec


class PolicyOpponentsTest(unittest.TestCase):
    def test_geometric_formula_endpoints_and_legal_mask(self):
        normal, weighted = np.zeros(64), np.zeros(64)
        legal = [19, 26, 37]
        normal[legal] = [.7, .2, .1]
        weighted[legal] = [.1, .3, .6]
        normal[0] = weighted[0] = 1e100  # illegal values cannot win
        for beta in (0, .2, .4, .6, .8, 1):
            expected = normal[legal] ** (1 - beta) * weighted[legal] ** beta
            expected /= expected.sum()
            actual = geometric_model_distribution(normal, weighted, legal, beta, logits=False)
            np.testing.assert_allclose(actual[legal], expected)
            self.assertEqual(0, actual[0])
            self.assertAlmostEqual(1, actual.sum())
        np.testing.assert_allclose(geometric_model_distribution(normal, None, legal, 0, logits=False)[legal], normal[legal])
        np.testing.assert_allclose(geometric_model_distribution(None, weighted, legal, 1, logits=False)[legal], weighted[legal])

    def test_logits_match_geometric_probabilities_and_remain_stable(self):
        legal = [0, 1, 2]
        normal, weighted = np.zeros(64), np.zeros(64)
        normal[legal], weighted[legal] = [-2, 1, 3], [2, -4, 1]
        for beta in (.2, .4, .6, .8):
            probability_result = geometric_model_distribution(np.exp(normal), np.exp(weighted), legal, beta, logits=False)
            logits_result = geometric_model_distribution(normal + 10000, weighted - 10000, legal, beta)
            np.testing.assert_allclose(logits_result, probability_result, atol=1e-12)
        self.assertTrue(np.isfinite(logits_result).all())

    def test_empty_boards_zeros_and_invalid_values(self):
        self.assertEqual(0, geometric_model_distribution(None, None, [], .5).sum())
        with self.assertRaisesRegex(ValueError, 'shared'):
            geometric_model_distribution(np.zeros(64), np.zeros(64), [0, 1], .5, logits=False)
        with self.assertRaises(ValueError):
            geometric_model_distribution(np.ones(64), np.ones(64), [0], float('nan'))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            geometric_model_distribution(np.full(64, np.nan), np.ones(64), [0], .5)

    def test_mix_selects_argmax_and_queries_correct_endpoints(self):
        legal = BoardState.initial().legal_policies(0)
        normal, weighted = np.zeros((1, 64)), np.zeros((1, 64))
        normal[0, legal[0]], weighted[0, legal[-1]] = 4, 5
        for beta, expected_move in ((0, legal[0]), (.2, legal[0]), (.8, legal[-1]), (1, legal[-1])):
            clients = []
            def client(host, port, timeout):
                instance = Mock()
                instance.predict.return_value = normal if port == 101 else weighted
                clients.append((port, instance))
                return instance
            args = make_argparser().parse_args(['--mode', 'geometric', '--beta', str(beta),
                                                '--normal-port', '101', '--weighted-port', '102'])
            with patch('policy_opponent_gtp.PolicyBatchClient', side_effect=client):
                engine = PolicyOpponent(args)
                self.assertEqual(expected_move, coord_to_policy(engine.genmove('black')))
                self.assertEqual(({101} if beta == 0 else {102} if beta == 1 else {101, 102}),
                                 {port for port, _ in clients})
                engine.blender.close()
                for _, instance in clients:
                    instance.close.assert_called_once()

    def test_random_uses_uniform_choice_and_deterministic_game_seed(self):
        args = make_argparser().parse_args(['--mode', 'random'])
        with patch('policy_opponent_gtp.PolicyBatchClient') as client:
            engine = PolicyOpponent(args)
            legal = engine.state.legal_policies(0)
            engine.rng = Mock()
            engine.rng.choice.return_value = legal[-1]
            self.assertEqual(legal[-1], coord_to_policy(engine.genmove('black')))
            engine.rng.choice.assert_called_once_with(legal)
            client.assert_not_called()
        sequences = []
        engine = PolicyOpponent(args)
        for seed in (12345, 98765, 12345):
            engine.dispatch('set_game_seed', [str(seed)])
            engine.clear_board()
            moves = []
            for _ in range(20):
                side = engine.state.side
                moves.append(engine.genmove('black' if side == 0 else 'white'))
            sequences.append(moves)
        self.assertEqual(sequences[0], sequences[2])
        self.assertNotEqual(sequences[0], sequences[1])

    def test_random_pass_and_policy_players_reserve_slots_without_search_threads(self):
        args = make_argparser().parse_args(['--mode', 'random'])
        engine = PolicyOpponent(args)
        engine.setboard(['X' * 64, 'black'])
        self.assertEqual('pass', engine.genmove('black'))
        specs = [PlayerSpec('search', ('engine',), 8, False),
                 PlayerSpec('model_blend_beta_0.2', ('wrapper',), 8, True, 1.0),
                 PlayerSpec('random', ('wrapper',), 8, False, 1.0, random_seed=57)]
        budget = GameThreadBudget(specs)
        tasks = [MatchSetTask(0, 0, 1, 0, 'x'), MatchSetTask(1, 1, 2, 0, 'x')]
        for task in tasks:
            self.assertTrue(budget.reserve_set(task))
        budget.distribute(tasks)
        self.assertEqual(70, budget.allocated)
        self.assertEqual([32, 32], budget.player_threads(0))
        self.assertEqual(64, budget.allocated_search_threads)
        self.assertEqual(6, budget.allocated_policy_processes)
        self.assertEqual([], budget.player_threads(1))
        self.assertEqual([], budget.player_threads(2))


if __name__ == '__main__':
    unittest.main()
