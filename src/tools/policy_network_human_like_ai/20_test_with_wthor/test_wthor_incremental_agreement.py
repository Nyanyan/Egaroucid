#!/usr/bin/env python3

import unittest

import numpy as np

from wthor_human_match_evaluation import (
    BLEND_PARAMS,
    CONSOLE_LEVELS,
    CONSOLE_REFERENCE_LEVEL,
    POLICY_SIZE,
    BoardState,
    HintData,
    IncrementalAgreementMetrics,
    PositionGroup,
    evaluate_agreement,
)


def make_asymmetric_state() -> BoardState:
    state = BoardState.initial()
    for use_last_move in (False, True, False):
        legal = state.legal_policies(state.side)
        state.apply_move(
            state.side,
            legal[-1] if use_last_move else legal[0],
        )
    return state


def make_fixture():
    state = make_asymmetric_state()
    legal = state.legal_policies(state.side)
    illegal = next(
        policy
        for policy in range(POLICY_SIZE)
        if policy not in legal
    )
    group = PositionGroup(
        black=state.black,
        white=state.white,
        side=state.side,
        policy_counts=(
            (-1, 2),
            (illegal, 3),
            (legal[0], 4),
            (legal[-1], 2),
        ),
    )

    move_order = tuple(reversed(legal))
    scores = {
        policy: float(len(move_order) - rank)
        for rank, policy in enumerate(move_order)
    }
    hint = HintData(scores=scores, move_order=move_order)
    hints = {
        level: {group.key: hint}
        for level in CONSOLE_LEVELS
    }

    logits = np.full(POLICY_SIZE, -100.0, dtype=np.float64)
    for rank, policy in enumerate(legal):
        logits[policy] = float(rank)
    return group, hints, {group.key: logits}


class IncrementalAgreementMetricsTest(unittest.TestCase):
    def test_incremental_result_matches_complete_evaluation(self) -> None:
        group, hints, policy_logits = make_fixture()
        expected = evaluate_agreement(
            [group],
            hints,
            policy_logits,
            score_temperature=1.0,
        )

        metrics = IncrementalAgreementMetrics(
            [group],
            policy_logits,
            score_temperature=1.0,
        )
        rows = [
            (level, group.key, hints[level][group.key])
            for level in reversed(CONSOLE_LEVELS)
        ]
        self.assertEqual(len(CONSOLE_LEVELS), metrics.add_hint_rows(rows))
        self.assertEqual(0, metrics.add_hint_rows(rows))

        self.assertEqual(expected, metrics.result())
        blend_metrics, console_metrics, invalid, illegal = metrics.result()
        self.assertEqual(2, invalid)
        self.assertEqual(3, illegal)
        for alpha in BLEND_PARAMS:
            self.assertEqual(6, blend_metrics[alpha]["positions"])
        for level in CONSOLE_LEVELS:
            self.assertEqual(6, console_metrics[level]["positions"])

    def test_partial_snapshot_has_zeroes_and_is_independent(self) -> None:
        group, hints, policy_logits = make_fixture()
        metrics = IncrementalAgreementMetrics(
            [group],
            policy_logits,
            score_temperature=1.0,
        )

        blend_metrics, console_metrics, invalid, illegal = metrics.snapshot()
        self.assertEqual(set(BLEND_PARAMS), set(blend_metrics))
        self.assertEqual(set(CONSOLE_LEVELS), set(console_metrics))
        self.assertTrue(
            all(metric["positions"] == 0 for metric in blend_metrics.values())
        )
        self.assertTrue(
            all(
                metric["positions"] == 0
                for metric in console_metrics.values()
            )
        )
        self.assertEqual(2, invalid)
        self.assertEqual(3, illegal)

        console_level = next(
            level
            for level in CONSOLE_LEVELS
            if level != CONSOLE_REFERENCE_LEVEL
        )
        self.assertTrue(
            metrics.add_hint(
                console_level,
                group.key,
                hints[console_level][group.key],
            )
        )
        blend_metrics, console_metrics, _, _ = metrics.snapshot()
        self.assertEqual(6, console_metrics[console_level]["positions"])
        self.assertTrue(
            all(metric["positions"] == 0 for metric in blend_metrics.values())
        )
        self.assertEqual(
            0,
            console_metrics[CONSOLE_REFERENCE_LEVEL]["positions"],
        )

        console_metrics[console_level]["hits"][1] = -1
        _, fresh_console_metrics, _, _ = metrics.snapshot()
        self.assertGreaterEqual(
            fresh_console_metrics[console_level]["hits"][1],
            0,
        )

    def test_level21_is_counted_once_and_matches_alpha_zero(self) -> None:
        group, hints, policy_logits = make_fixture()
        metrics = IncrementalAgreementMetrics(
            [group],
            policy_logits,
            score_temperature=1.0,
        )
        level21_hint = hints[CONSOLE_REFERENCE_LEVEL][group.key]

        self.assertTrue(
            metrics.add_hint(
                CONSOLE_REFERENCE_LEVEL,
                group.key,
                level21_hint,
            )
        )
        self.assertFalse(
            metrics.add_hint(
                CONSOLE_REFERENCE_LEVEL,
                group.key,
                level21_hint,
            )
        )

        blend_metrics, console_metrics, _, _ = metrics.result()
        self.assertEqual(
            blend_metrics[0.0],
            console_metrics[CONSOLE_REFERENCE_LEVEL],
        )
        self.assertEqual(6, blend_metrics[0.0]["positions"])
        self.assertEqual(1, metrics.accepted_hint_count)


if __name__ == "__main__":
    unittest.main()
