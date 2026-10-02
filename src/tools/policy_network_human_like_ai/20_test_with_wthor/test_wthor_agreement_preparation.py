#!/usr/bin/env python3

import io
import random
import unittest
from unittest.mock import patch

from evaluate_wthor_blend_human_match import equivalent_targets
from policy_accuracy import TRANSFORM_MAPS
from test_wthor_incremental_agreement import make_fixture
from wthor_human_match_evaluation import (
    BoardState,
    CONSOLE_LEVELS,
    IncrementalAgreementMetrics,
    IncrementalEngineAgreementMetrics,
    PositionGroup,
    evaluate_agreement,
    evaluate_engine_agreement,
    prepare_position_groups,
)
from wthor_human_match_experiment import report_preparation_progress


def scalar_reference(group):
    """Original per-position rules, independent of the batched preparation."""
    state = BoardState(group.black, group.white, group.side)
    legal = tuple(state.legal_policies(group.side))
    player, opponent = state.player_opponent_bits(group.side)
    targets = []
    invalid = illegal = 0
    for policy, count in group.policy_counts:
        if not 0 <= policy < 64:
            invalid += count
        elif policy not in legal:
            illegal += count
        else:
            targets.append((policy, tuple(equivalent_targets(player, opponent, policy)), count))
    return legal, tuple(targets), invalid, illegal


def preparation_fixtures():
    rng = random.Random(731)
    states = [(0, 0), ((1 << 64) - 1, 0), (0, (1 << 64) - 1)]
    initial = BoardState.initial()
    states.append((initial.black, initial.white))
    # Generate positions invariant under each of the eight transformations.
    for mapping in TRANSFORM_MAPS:
        black = white = visited = 0
        for policy in range(64):
            if visited & (1 << policy):
                continue
            orbit = 0
            while not orbit & (1 << policy):
                orbit |= 1 << policy
                policy = int(mapping[policy])
            visited |= orbit
            color = rng.randrange(3)
            if color == 0:
                black |= orbit
            elif color == 1:
                white |= orbit
        states.append((black, white))
    # Arbitrary disjoint boards stress corners, edges, and no-move positions.
    for _ in range(16):
        black = rng.getrandbits(64)
        white = rng.getrandbits(64) & ~black
        states.append((black, white))
    state = initial.copy()
    for turn in range(60):
        legal = state.legal_policies(state.side)
        if not legal:
            state.side ^= 1
            legal = state.legal_policies(state.side)
            if not legal:
                break
        if turn % 4 == 0:
            states.append((state.black, state.white))
        state.apply_move(state.side, rng.choice(legal))
    return [
        PositionGroup(black, white, side, tuple((p, p % 5 + 1) for p in range(-1, 65)))
        for black, white in dict.fromkeys(states) for side in (0, 1)
    ]


class AgreementPreparationTest(unittest.TestCase):
    def test_batches_match_original_legal_and_symmetry_rules(self):
        groups = preparation_fixtures()
        prepared = prepare_position_groups(groups, batch_size=13)
        invalid = illegal = 0
        for group in groups:
            legal, targets, bad_policy, bad_label = scalar_reference(group)
            actual, = prepared.groups_by_key[group.key]
            self.assertEqual(legal, actual.legal, group.key)
            self.assertEqual(targets, actual.valid_targets, group.key)
            invalid += bad_policy
            illegal += bad_label
        self.assertEqual(invalid, prepared.invalid_policy_samples)
        self.assertEqual(illegal, prepared.illegal_label_samples)

    def test_shared_data_is_reused_for_both_engines_and_final_checks(self):
        group, hints, logits = make_fixture()
        groups = [group, PositionGroup(group.black, group.white, group.side, group.policy_counts[-1:])]
        expected_blend = evaluate_agreement(groups, hints, logits, 1.0)
        expected_engine = evaluate_engine_agreement(groups, hints)
        prepared = prepare_position_groups(groups, batch_size=1)
        self.assertEqual(2, len(prepared.groups_by_key[group.key]))
        with self.assertRaises(TypeError):
            prepared.groups_by_key[group.key] = ()
        with patch("wthor_human_match_evaluation.prepare_position_groups", side_effect=AssertionError("repeated preparation")):
            blend = IncrementalAgreementMetrics(groups, logits, 1.0, prepared_groups=prepared)
            engine = IncrementalEngineAgreementMetrics(groups, CONSOLE_LEVELS, prepared_groups=prepared)
            self.assertIs(prepared.groups_by_key, blend._groups_by_key)
            self.assertIs(prepared.groups_by_key, engine._groups)
            for level in CONSOLE_LEVELS:
                hint = hints[level][group.key]
                blend.add_hint(level, group.key, hint)
                engine.add_hint(level, group.key, hint)
                self.assertFalse(blend.add_hint(level, group.key, hint))
                self.assertFalse(engine.add_hint(level, group.key, hint))
            self.assertEqual(expected_blend, blend.result())
            self.assertEqual(expected_engine, engine.snapshot())
            self.assertEqual(expected_blend, evaluate_agreement(groups, hints, logits, 1.0, prepared_groups=prepared))
            self.assertEqual(expected_engine, evaluate_engine_agreement(groups, hints, prepared_groups=prepared))

    def test_progress_reports_start_interval_and_completion(self):
        group, _, _ = make_fixture()
        reports = []
        with patch("wthor_human_match_evaluation.time.perf_counter", side_effect=[0.0, 2.0, 4.0, 6.0]):
            prepare_position_groups(
                [group] * 3, batch_size=1, progress_interval_sec=3.0,
                on_progress=lambda *args: reports.append(args),
            )
        self.assertEqual([(0, 3, 0.0), (2, 3, 4.0), (3, 3, 6.0)], reports)

    def test_empty_input_reports_completion(self):
        reports = []
        prepared = prepare_position_groups([], on_progress=lambda *args: reports.append(args))
        self.assertEqual({}, prepared.groups_by_key)
        self.assertEqual(0, prepared.invalid_policy_samples)
        self.assertEqual(0, prepared.illegal_label_samples)
        self.assertEqual([(0, 0, 0.0)], reports)

    def test_invalid_batch_or_interval_is_rejected(self):
        for kwargs in ({"batch_size": 0}, {"progress_interval_sec": -1}, {"progress_interval_sec": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_position_groups([], **kwargs)

    def test_progress_log_includes_count_rate_and_eta(self):
        stream = io.StringIO()
        with patch("sys.stderr", stream):
            report_preparation_progress(0, 2000, 0.0)
            report_preparation_progress(1000, 2000, 2.0)
            report_preparation_progress(2000, 2000, 4.0)
        output = stream.getvalue()
        self.assertIn("0/2,000状態 (0.0%)", output)
        self.assertIn("1,000/2,000状態 (50.0%)", output)
        self.assertIn("速度=500状態/秒", output)
        self.assertIn("残り=", output)
        self.assertIn("2,000/2,000状態 (100.0%)", output)


if __name__ == "__main__":
    unittest.main()
